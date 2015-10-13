"""
Dispersed Phases
================

Create several objects and functions to manipulate dispersed phase particles

The `single_bubble_model`, `stratified_plume_model`, and `bent_plume_model` 
all must handle dispersed phase particles in several different ways.  This 
module defines several particle classes that provide seamless interfaces to
the `dbm` module.  It also defines several functions that aid in manipulating
common input data to create the inputs needed to initialize these particle
classes.  These classes and functions originated in the older versions of
the `single_bubble_model` and `stratified_plume_model`.  This module is a 
re-factorization of these modules during creation of the `bent_plume_model`, 
which allows all particle manipulations to reside in one place.

Notes 
----- 
These class objects and helper functions are used throughout the TAMOC 
modeling suite. 

See Also
--------
`stratified_plume_model` : Predicts the plume solution for quiescent ambient
    conditions or weak crossflows, where the intrusion (outer plume) 
    interacts with the upward rising plume in a double-plume integral model
    approach.  Such a situation is handeled properly in the 
    `stratified_plume_model` and would violate the assumption of non-
    iteracting Lagrangian plume elements as required in this module.

`single_bubble_model` : Tracks the trajectory of a single bubble, drop or 
    particle through the water column.  The numerical solution, including
    the various object types and their functionality, used here follows the
    pattern in the `single_bubble_model`.  The main difference is the more
    complex state space and governing equations.

`bent_plume_model` : Simulates a multiphase plume as a Lagrangian plume 
    model, which makes the model much more amenable to a crossflow.  This 
    model is similar to the `stratified_plume_model`, except that it does
    not have an outer plume that interacts with the inner plume (Lagrangian
    elements are independent).  

"""
# S. Socolofsky, October 2014, Texas A&M University <socolofs@tamu.edu>.
from tamoc import seawater
from tamoc import dbm

import numpy as np
from scipy.optimize import fsolve
import unicodedata
from copy import copy


# ----------------------------------------------------------------------------
# Define the Particle objects for the multiphase behavior in the TAMOC models
# ----------------------------------------------------------------------------

class SingleParticle(object):
    """
    Interface to the `dbm` module and container for model parameters
    
    This class provides a uniform interface to the `dbm` module objects and
    methods and stores the particle-specific model parameters.  Because the
    methods for `dbm.FluidParticle` and `dbm.InsolubleParticle` sometimes have 
    different inputs and different outputs, there needs to be a method to 
    support these interface differences in a single location.  This object
    solves that problem by providing a single interface and uniform outputs
    for the particle properties needed by the single bubble model.  This also
    affords a convenient place to store the particle-specific model 
    parameters and behavior, such as mass transfer reduction factor, etc., 
    turning off heat transfer once the particle matches the ambient 
    temperature and turning off the particle buoyancy once the particle is
    dissolved.
    
    Parameters
    ----------
    dbm_particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Object describing the particle properties and behavior
    m0 : ndarray
        Initial masses of the components of the `dbm` particle object (kg)
    T0 : float
        Initial temperature of the of `dbm` particle object (K)
    K : float, default = 1.
        Mass transfer reduction factor (--).
    K_T : float, default = 1.
        Heat transfer reduction factor (--).
    fdis : float, default = 1e-6
        Fraction of the initial total mass (--) remaining when the particle 
        should be considered dissolved.
    t_hyd : float, default = 0.
        Hydrate film formation time (s).  Mass transfer is computed by clean
        bubble methods for t less than t_hyd and by dirty bubble methods
        thereafter.  The default behavior is to assume the particle is dirty
        or hydrate covered from the release.
    
    Attributes
    ----------
    particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Stores the `dbm_particle` object passed to `__init__()`.
    composition : str list
        Copy of the `composition` attribute of the `dbm_particle` object.
    m0 : ndarray
        Initial masses (kg) of the particle components
    T0 : float
        Initial temperature (K) of the particle
    cp : float
        Heat capacity at constant pressure (J/(kg K)) of the particle.
    K : float
        Mass transfer reduction factor (--)
    K_T : float
        Heat transfer reduction factor (--)
    fdis : float
        Fraction of initial mass remaining as total dissolution (--)
    diss_indices : ndarray bool
        Indices of m0 that are non-zero.
    
    Notes
    -----
    This object only provides an interface to the `return_all` and 
    `diameter` methods of the `dbm` module objects.  The intent is to be as 
    fast as possible while providing a single location for the necessary 
    `if`-statements needed to select between soluble and insoluble particle 
    methods and facilitate turning heat transfer and dissolution on and off
    as necessary at the simulation progresses.  
    
    Dissolution is turned off component by component as each components mass
    becomes fdis times smaller than the initial mass.  Once all of the initial
    components have been turned off, the particle is assumed to have a 
    density equation to the ambient water and a slip velocity of zero.
    
    Heat transfer is turned off once the particle comes within 0.1 K of the
    ambient temperature.  Thereafter, the temperature is forced to track 
    the ambient temperature.
    
    """
    def __init__(self, dbm_particle, m0, T0, K=1., K_T=1., fdis=1.e-6, 
                 t_hyd=0.):
        super(SingleParticle, self).__init__()
        
        # Make sure the masses are in a numpy array
        if not isinstance(m0, np.ndarray):
            if not isinstance(m0, list):
                m0 = np.array([m0])
            else:
                m0 = np.array(m0)
        
        # Store the input parameters
        self.particle = dbm_particle
        self.composition = dbm_particle.composition
        self.m0 = m0
        self.T0 = T0
        self.cp = seawater.cp() * 0.5
        
        # Store the particle-specific model parameters
        self.K = K
        self.K_T = K_T
        self.fdis = fdis
        self.t_hyd = t_hyd
        
        # Store parameters to track the dissolution of the initial masses
        self.diss_indices = self.m0 > 0
    
    def properties(self, m, T, P, Sa, Ta, t):
        """
        Return the particle properties from the discrete bubble model
        
        Provides a single interface to the `return_all` methods of the fluid 
        and insoluble particle objects defined in the `dbm` module.  
        This method also applies the particle-specific model parameters to 
        adjust the mass and heat transfer and determine the dissolution state.
        
        Parameters
        ----------
        m : float
             mass of the particle (kg)
        T : float
             particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        t : float
            age of the particle--time since it was released into the water 
            column (s)
        
        Returns
        -------
        A tuple containing:
            
            us : float
                slip velocity (m/s)
            rho_p : float
                particle density (kg/m^3)
            A : float 
                surface area (m^2)
            Cs : ndarray, size (nc)
                solubility (kg/m^3)
            K * beta : ndarray, size (nc)
                effective mass transfer coefficient(s) (m/s)
            K_T * beta_T : float
                effective heat transfer coefficient (m/s)
            T : float
                temperature of the particle (K)
        
        Notes
        -----
        For insoluble particles, `Cs` and `beta` are undefined.  This method
        returns values for these variables that will result in no 
        dissolution and will also protect model simulations from undefined
        mathematical operations (e.g., divide by zero).
        
        """
        # Turn off heat transfer when at equilibrium.  This will be a 
        # persistent change, so it only has to happen once.
        if self.K_T > 0. and np.abs(Ta - T) < 0.5:
            self.K_T = 0.
        
        # Use the right temperature
        if self.K_T == 0.:
            T = Ta
        
        # Decide which slip velocity and mass and heat transfer to use
        if t < self.t_hyd:
            # Treat the particle as clean for slip velocity and mass
            # transfer
            status = 1
        else:
            # Use the dirty bubble slip velocity and mass transfer
            status = -1
        
        # Distinguish between soluble and insoluble particles
        if self.particle.issoluble:
                        
            # Get the DBM results
            m[m<0] = 0.   # stop oscillations at small mass
            shape, de, rho_p, us, A, Cs, beta, beta_T = \
                self.particle.return_all(m, T, P, Sa, Ta, status)
            
            # Turn off dissolution for "dissolved" components
            frac_diss = np.ones(np.size(m))
            frac_diss[self.diss_indices] = \
                m[self.diss_indices] / self.m0[self.diss_indices]
            beta[frac_diss < self.fdis] = 0.
            
            # Shut down bubble forces when particles fully dissolve
            if np.sum(beta[self.diss_indices]) == 0.:
                # Injected chemicals have dissolved
                if np.sum(m[self.diss_indices]) > \
                    np.sum(m[~self.diss_indices]):
                    # The whole particle has dissolved
                    us = 0.0
                    rho_p = seawater.density(Ta, Sa, P)
        
        else:
            # Get the particle properties
            shape, de, rho_p, us, A, beta_T = \
                self.particle.return_all(m[0], T, P, Sa, Ta, status)
            beta = np.array([])
            Cs = np.array([])
        
        # Return the particle properties
        return (us, rho_p, A, Cs, self.K * beta, self.K_T * beta_T, T)
    
    def diameter(self, m, T, P, Sa, Ta):
        """
        Compute the diameter of a particle from mass and density
        
        Computes the diameter of a particle using the methods in the `dbm`
        module.  This method is used in the post-processor of the `Model`
        object, but not in the actual simulation.  
        
        Parameters
        ----------
        m : float
             mass of the particle (kg)
        T : float
             particle temperature (K)
        P : float
            particle pressure (Pa)
        Sa : float
            salinity of ambient seawater (psu)
        Ta : float
            temperature of ambient seawater (K)
        
        Returns
        -------
        de : float
            diameter of the particle (m)
        
        """
        # Distinguish between soluble and insoluble particles
        if self.particle.issoluble:
            de = self.particle.diameter(m, T, P)
        else:
            de = self.particle.diameter(m, T, P, Sa, Ta)
        
        # Return the diameter
        return de


class PlumeParticle(SingleParticle):
    """
    Interface to the `dbm` module and container for the model parameters
    
    As in the `single_bubble_model.Particle` class, this object provides a
    uniform interface to the `dbm` module objects and captures the 
    particle-specific model parameters.
    
    Parameters
    ----------
    dbm_particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Object describing the particle properties and behavior
    m0 : ndarray
        Initial masses of one particle for the components of the 
        `dbm_particle` object (kg)
    T0 : float
        Initial temperature of the of `dbm` particle object (K)
    nb0 : float
        Initial number flux of particles at the release (--)
    lambda_1 : float 
        spreading rate of the dispersed phase in a plume (--)
    P : float
        Local pressure (Pa)
    Sa : float
        Local salinity surrounding the particle (psu)
    Ta : float
        Local temperature surrounding the particle (K)
    K : float, default = 1.
        Mass transfer reduction factor (--).
    K_T : float, default = 1.
        Heat transfer reduction factor (--).
    fdis : float, default = 0.01
        Fraction of the initial total mass (--) remaining when the particle 
        should be considered dissolved.
    t_hyd : float, default = 0.
        Hydrate film formation time (s).  Mass transfer is computed by clean
        bubble methods for t less than t_hyd and by dirty bubble methods
        thereafter.  The default behavior is to assume the particle is dirty
        or hydrate covered from the release.
    
    Attributes
    ----------
    particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Stores the `dbm_particle` object passed to `__init__()`.
    composition : str list
        Copy of the `composition` attribute of the `dbm_particle` object.
    m0 : ndarray
        Initial masses (kg) of the particle components
    T0 : float
        Initial temperature (K) of the particle
    cp : float
        Heat capacity at constant pressure (J/(kg K)) of the particle.
    K : float
        Mass transfer reduction factor (--)
    K_T : float
        Heat transfer reduction factor (--)
    fdis : float
        Fraction of initial mass remaining as total dissolution (--)
    diss_indices : ndarray bool
        Indices of m0 that are non-zero.
    nb0 : float
        Initial number flux of particles at the release (--)
    lambda_1 : float 
        Spreading rate of the dispersed phase in a plume (--)
    m : ndarray
        Current masses of the particle components (kg)
    T : float
        Current temperature of the particle (K)
    us : float
        Slip velocity (m/s)
    rho_p : float
        Particle density (kg/m^3)
    A : float
        Particle surface area (m^2)
    Cs : ndarray
        Solubility of each dissolving component in the particle (kg/m^3)
    beta : ndarray
        Mass transfer coefficients (m/s)
    beta_T : float
        Heat transfer coefficient (m/s)
    
    See Also
    --------
    single_bubble_model.Particle
    
    Notes
    -----
    This object inherits the `single_bubble_model.Particle` object, which
    defines the attributes: `particle`, `composition`, `m0`, `T0`, `cp`, 
    `K`, `K_T`, `fdis`, and `diss_indices` and the methods
    `single_bubble_model.Particle.properties`, and 
    `single_bubble_model.Particle.diameter`.
    
    """
    def __init__(self, dbm_particle, m0, T0, nb0, lambda_1, P, Sa, Ta, 
                 K=1., K_T=1., fdis=1.e-6, t_hyd=0.):
        super(PlumeParticle, self).__init__(dbm_particle, m0, T0, K, K_T, 
                                            fdis, t_hyd)
        
        # Store the input variables related to the particle description
        self.nb0 = nb0
        
        # Store the model parameters
        self.lambda_1 = lambda_1
        
        # Set the local masses and temperature to their initial values.  The
        # particle age is zero at instantiation
        self.update(m0, T0, P, Sa, Ta, 0.)
    
    def update(self, m, T, P, Sa, Ta, t):
        """
        Store the instantaneous values of the particle properties
        
        During the simulation, it is often helpful to keep the state space
        variables for each particle stored within the particle, especially
        since each particle type (soluble or insoluble) can have different
        sizes of arrays for m.
        
        Parameters
        ----------
        m : ndarray
            Current masses (kg) of the particle components
        T : float
            Current temperature (K) of the particle
        P : float
            Local pressure (Pa)
        Sa : float
            Local salinity surrounding the particle (psu)
        Ta : float
            Local temperature surrounding the particle (K)       
        t : float
            age of the particle--time since it was released into the water 
            column (s)
        
        """
        # Make sure the masses are in a numpy array
        if not isinstance(m, np.ndarray):
            if not isinstance(m, list):
                m = np.array([m])
            else:
                m = np.array(m)
        
        # Update the variables with their currrent values
        self.m = m
        if np.sum(self.m) > 0.:
            self.us,  self.rho_p,  self.A, self.Cs, self.beta, \
                self.beta_T, self.T = self.properties(m, T, P, Sa, Ta, t)
        else:
            self.us = 0.
            self.rho_p = seawater.density(Ta, Sa, P)
            self.A = 0.
            self.Cs = np.zeros(len(self.composition))
            self.beta = np.zeros(len(self.composition))
            self.beta_T = 0.
            self.T = Ta


# ----------------------------------------------------------------------------
# Functions that help to create SingleParticle and PlumeParticle objects
# ----------------------------------------------------------------------------

def initial_conditions(profile, z0, dbm_particle, yk, q, q_type, de, 
                       T0=None):
    """
    Define standard initial conditions for a PlumeParticle from flow rate
    
    Returns the standard variables describing a particle as needed to 
    initializae a PlumeParticle object from specification of the dispersed phase
    flow rate.  
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    z0 : float
        Depth of the release point (m)
    dbm_particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Object describing the particle properties and behavior
    yk : ndarray
        Vector of mol fractions of each component of the dispersed phase 
        particle.  If the particle is a `dbm.InsolubleParticle`, then yk 
        should be equal to one.
    q : float
        Flux of the dispersed phase, either as the volume flux (m^3/s) at 
        standard conditions, defined as 0 deg C and 1 bar, or as mass flux 
        (kg/s).
    q_type : int
        Determines the type of flux units.  0: we want the mass of a single 
        particle (hence q = None since it is currently unknown), 1: q is 
        volume flux, 2: q is mass flux
    de : float
        Initial diameter (m) of the particle
    T0 : float, default = None
        Initial temperature of the of `dbm` particle object (K).  If None, 
        then T0 is set equal to the ambient temperature.
    
    Returns
    -------
    m0 : ndarray
        Initial masses of the components of one particle in the `dbm` 
        particle object (kg)
    T0 : float
        Initial temperature of the of `dbm` particle object (K)
    nb0 : float
        Initial number flux of particles at the release (--)
    P : float
        Local pressure (Pa)
    Sa : float
        Local salinity surrounding the particle (psu)
    Ta : float
        Local temperature surrounding the particle (K)
    
    """
    # Make sure yk is an array
    if not isinstance(yk, np.ndarray):
        if not isinstance(yk, list):
            yk = np.array([yk])
        else:
            yk = np.array(yk)
    
    # Get the ambient conditions at the release
    Ta, Sa, P = profile.get_values(z0, ['temperature', 'salinity', 
                                        'pressure'])
    
    # Get the particle temperature
    if T0 is None:
        T0 = copy(Ta)
    
    # Compute the density at standard and in situ conditions
    if dbm_particle.issoluble:
        mf = dbm_particle.mass_frac(yk)
        rho_N = dbm_particle.density(mf, 273.15, 1.e5)
        rho_p = dbm_particle.density(mf, T0, P)
    else:
        mf = 1.
        rho_N = dbm_particle.density(273.15, 1.e5, 0., 273.15)
        rho_p = dbm_particle.density(T0, P, Sa, Ta)
    
    # Get the mass and number flux of particles
    if q_type == 0:
        # Compute the mass flux of a single particle from the given diameter
        if dbm_particle.issoluble:
            m0 = dbm_particle.masses_by_diameter(de, T0, P, yk)
        else:
            m0 = dbm_particle.mass_by_diameter(de, T0, P, Sa, Ta)
        nb0 = 1.
    else:
        if q_type == 1:
            # Compute the total mass flux from the given volume flux at STP
            m_dot = q * rho_N
        else:
            # The input flux is the total mass flux
            m_dot = q
        
        # Get the source volume flux and particle number flux
        Q = m_dot / rho_p
        nb0 = Q / (np.pi * de**3 / 6.)
        
        # Get the initial particle mass(es)
        m0 = m_dot / nb0 * mf
    
    # Return the standard variables
    return (m0, T0, nb0, P, Sa, Ta)


# ----------------------------------------------------------------------------
# Functions to save and load a particle to an open netCDF4 dataset
# ----------------------------------------------------------------------------

def save_particle_to_nc_file(nc, chem_names, particles, K_T0):
    """
    Write the particle attributes to a netCDF output file
    
    Writes all of the object attributes for a `SingleParticle` or 
    `PlumeParticle` object to a netCDF output file.
    
    Parameters
    ----------
    nc : `netCDF4.Dataset` object
        A `netCDF4.Dataset` object that is open and where the particle 
        attributes should be written
    chem_names : str list
        A list of chemical names in the composition of the `dbm` objects
        in these particles
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or 
        `bent_plume_model.Particle` objects describing each dispersed phase 
        in the simulation
    K_T0 : ndarray
        Array of the initial values of the heat transfer reduction factor.
    
    """
    # Make sure the particles variable is iterable
    if not isinstance(particles, list):
        particles = [particles]
    
    # Make sure K_T0 is an array
    if not isinstance(K_T0, np.ndarray):
        if not isinstance(K_T0, list):
            K_T0 = np.array([K_T0])
        else:
            K_T0 = np.array(K_T0)
    
    # Count the number of particles
    nparticles = nc.createDimension('nparticles', len(particles))
    ngroups = nc.createDimension('ngroups', 15)
    if len(chem_names) > 0:
        nchems = nc.createDimension('nchems', len(chem_names))
    else:
        nchems = nc.createDimension('nchems', 1)
    num = nc.createDimension('num', 1)
    
    # Save the particle composition
    nc.composition = ' '.join(chem_names)
    
    # Create the dataset descriptions for all the particle variables
    particle_type = nc.createVariable('particle_type', 'i4', ('num',))
    particle_type.long_name = 'dispersed_phases Particle type'
    particle_type.standard_name = 'particle_type'
    particle_type.units = '0: Single, 1:Plume, 2:Bent plume particle'
    
    issoluble = nc.createVariable('issoluble', 'i4', ('nparticles',))
    issoluble.long_name = 'solubility (0: false, 1: true)'
    issoluble.standard_name = 'issoluble'
    issoluble.units = 'boolean'
    
    isair = nc.createVariable('isair', 'i4', ('nparticles',))
    isair.long_name = 'fluid is air (0: false, 1: true)'
    isair.standard_name = 'isair'
    isair.units = 'boolean'
    
    isfluid = nc.createVariable('isfluid', 'i4', ('nparticles',))
    isfluid.long_name = 'Fluid status (0: false, 1: true)'
    isfluid.standard_name = 'isfluid'
    isfluid.units = 'boolean'
    
    iscompressible = nc.createVariable('iscompressible', 'i4', 
                                       ('nparticles',))
    iscompressible.long_name = 'Compressibility (0: false, 1: true)'
    iscompressible.standard_name = 'iscompressible'
    iscompressible.units = 'boolean'
    
    calc_delta = nc.createVariable('calc_delta', 'i4', ('nparticles',))
    calc_delta.long_name = 'Calculate delta (-1: false, 1: true)'
    calc_delta.standard_name = 'calc_delta'
    calc_delta.units = 'boolean'
    
    extern_data = nc.createVariable('extern_data', 'i4', ('nparticles',))
    extern_data.long_name = 'External chem database (0: false, 1: true)'
    extern_data.standard_name = 'extern_data'
    extern_data.units = 'boolean'
    
    fp_type = nc.createVariable('fp_type', 'i4', ('nparticles',))
    fp_type.long_name = 'fluid phase (0: gas, 1: liquid, 2: solid)'
    fp_type.standard_name = 'fp_type'
    fp_type.units = 'nondimensional'
    
    rho_p = nc.createVariable('rho_p', 'f8', ('nparticles',))
    rho_p.long_name = 'particle density'
    rho_p.standard_name = 'rho_p'
    rho_p.units = 'kg/m^3'
    
    gamma = nc.createVariable('gamma', 'f8', ('nparticles',))
    gamma.long_name = 'API Gravity'
    gamma.standard_name = 'gamma'
    gamma.units = 'deg API'
    
    beta = nc.createVariable('beta', 'f8', ('nparticles',))
    beta.long_name = 'thermal expansion coefficient'
    beta.standard_name = 'beta'
    beta.units = 'Pa^(-1)'
    
    co = nc.createVariable('co', 'f8', ('nparticles',))
    co.long_name = 'isothermal compressibility coefficient'
    co.standard_name = 'co'
    co.units = 'K^(-1)'
        
    sigma_correction = nc.createVariable('sigma_correction', 'f8', 
        ('nparticles',))
    sigma_correction.long_name = 'interfacial tension reduction factor (--)'
    sigma_correction.standard_name = 'sigma_correction'
    sigma_correction.units = 'nondimensional'
    
    delta_groups = nc.createVariable('delta_groups', 'f8', ('nparticles', 
        'nchems', 'ngroups'))
    delta_groups.long_name = 'group contribution method delta groups'
    delta_groups.standard_name = 'delta_groups'
    delta_groups.units = 'nondimensional'
    
    m0 = nc.createVariable('m0', 'f8', ('nparticles', 'nchems'))
    m0.long_name = 'initial mass flux'
    m0.standard_name = 'm0'
    m0.units = 'kg/s'
    
    T0 = nc.createVariable('T0', 'f8', ('nparticles'))
    T0.long_name = 'initial temperature'
    T0.standard_name = 'T0'
    T0.units = 'K'
        
    K = nc.createVariable('K', 'f8', ('nparticles',))
    K.long_name = 'mass transfer reduction factor'
    K.standard_name = 'K'
    K.units = 'nondimensional'
    
    K_T = nc.createVariable('K_T', 'f8', ('nparticles',))
    K_T.long_name = 'heat transfer reduction factor'
    K_T.standard_name = 'K_T'
    K_T.units = 'nondimensional'
    
    fdis = nc.createVariable('fdis', 'f8', ('nparticles',))
    fdis.long_name = 'dissolution criteria'
    fdis.standard_name = 'fdis'
    fdis.units = 'nondimensional'
    
    t_hyd = nc.createVariable('t_hyd', 'f8', ('nparticles',))
    t_hyd.long_name = 'hydrate formation time'
    t_hyd.standard_name = 't_hyd'
    t_hyd.units = 's'
    
    # Check if these are plume particle objects
    try:
        particles[0].integrate
        # Must be bent_plume_model.Particle object
        particle_type[0] = 2
        nb0 = nc.createVariable('nb0', 'f8', ('nparticles'))
        nb0.long_name = 'initial bubble number flux'
        nb0.standard_name = 'nb0'
        nb0.units = 's^(-1)'
        
        nbe = nc.createVariable('nbe', 'f8', ('nparticles'))
        nbe.long_name = 'number of bubbles following plume element'
        nbe.standard_name = 'nbe'
        nbe.units = 'count'
        
        lambda_1 = nc.createVariable('lambda_1', 'f8', ('nparticles'))
        lambda_1.long_name = 'bubble spreading ratio'
        lambda_1.standard_name = 'lambda_1'
        lambda_1.units = 'nondimensional'
        
        integrate = nc.createVariable('integrate', 'i4', ('nparticles',))
        integrate.long_name = 'Particle status (0: false, 1: true)'
        integrate.standard_name = 'integrate'
        integrate.units = 'boolean'
        
        sim_stored = nc.createVariable('sim_stored', 'i4', ('nparticles',))
        sim_stored.long_name = 'Tracking state (0: false, 1: true)'
        sim_stored.standard_name = 'sim_stored'
        sim_stored.units = 'boolean'
        
        farfield = nc.createVariable('farfield', 'i4', ('nparticles',))
        farfield.long_name = 'Farfield simualtion (0: false, 1: true)'
        farfield.standard_name = 'farfield'
        farfield.units = 'boolean'
        
        tp = nc.createVariable('tp', 'f8', ('nparticles'))
        tp.long_name = 'time'
        tp.standard_name = 't'
        tp.units = 's'
        
        xp = nc.createVariable('xp', 'f8', ('nparticles'))
        xp.long_name = 'x-coordinate'
        xp.standard_name = 'x'
        xp.units = 'm'
        
        yp = nc.createVariable('yp', 'f8', ('nparticles'))
        yp.long_name = 'y-coordinate'
        yp.standard_name = 'y'
        yp.units = 'm'
        
        zp = nc.createVariable('zp', 'f8', ('nparticles'))
        zp.long_name = 'z-coordinate'
        zp.standard_name = 'z'
        zp.units = 'm'
        zp.axis = 'Z'
        zp.positive = 'down'
         
        te = nc.createVariable('te', 'f8', ('nparticles'))
        te.long_name = 'particle exit time'
        te.standard_name = 'te'
        te.units = 's'
        
        xe = nc.createVariable('xe', 'f8', ('nparticles'))
        xe.long_name = 'particle exit x-coordinate'
        xe.standard_name = 'xe'
        xe.units = 'm'
        
        ye = nc.createVariable('ye', 'f8', ('nparticles'))
        ye.long_name = 'particle exit y-coordinate'
        ye.standard_name = 'ye'
        ye.units = 'm'
        
        ze = nc.createVariable('ze', 'f8', ('nparticles'))
        ze.long_name = 'particle exit z-coordinate'
        ze.standard_name = 'ze'
        ze.units = 'm'
        ze.axis = 'Z'
        ze.positive = 'down'
    
    except AttributeError:
        try:
            particles[0].nb0
            # Must be PlumeParticle object
            particle_type[0] = 1
            nb0 = nc.createVariable('nb0', 'f8', ('nparticles'))
            nb0.long_name = 'initial bubble number flux'
            nb0.standard_name = 'nb0'
            nb0.units = 's^(-1)'
            
            lambda_1 = nc.createVariable('lambda_1', 'f8', ('nparticles'))
            lambda_1.long_name = 'bubble spreading ratio'
            lambda_1.standard_name = 'lambda_1'
            lambda_1.units = 'nondimensional'
        
        except AttributeError:
            particle_type[0] = 0
   
    # Check if we need to reserve space to store an external chemical data
    # base of user_data
    next_chems = 0
    for i in range(len(particles)):
        if particles[i].particle.issoluble:
            if len(particles[i].particle.user_data) > next_chems:
                next_chems = len(particles[i].particle.user_data)
                user_composition = particles[i].particle.user_data.keys()
    
    if next_chems > 0:
        next_chems = nc.createDimension('next_chems', next_chems)
        nc.user_composition = ' '.join(user_composition)
        M = nc.createVariable('M', 'f8', ('nparticles', 'next_chems'))
        M.long_name = 'molecular weight'
        M.standard_name = 'M'
        M.units = 'kg/mol'
        
        Pc = nc.createVariable('Pc', 'f8', ('nparticles', 'next_chems'))
        Pc.long_name = 'pressure at the critical point'
        Pc.standard_name = 'Pc'
        Pc.units = 'Pa'
        
        Tc = nc.createVariable('Tc', 'f8', ('nparticles', 'next_chems'))
        Tc.long_name = 'temperature at the critical point'
        Tc.standard_name = 'Tc'
        Tc.units = 'K'
        
        Vc = nc.createVariable('Vc', 'f8', ('nparticles', 'next_chems'))
        Vc.long_name = 'molar volume at the critical point'
        Vc.standard_name = 'Vc'
        Vc.units = 'm^3/mol'
        
        Tb = nc.createVariable('Tb', 'f8', ('nparticles', 'next_chems'))
        Tb.long_name = 'boiling point'
        Tb.standard_name = 'Tb'
        Tb.units = 'K'
        
        Vb = nc.createVariable('Vb', 'f8', ('nparticles', 'next_chems'))
        Vb.long_name = 'molar volume at the boiling point'
        Vb.standard_name = 'Vb'
        Vb.units = 'm^3/mol'
        
        omega = nc.createVariable('omega', 'f8', ('nparticles', 
            'next_chems'))
        omega.long_name = 'acentric factor'
        omega.standard_name = 'omega'
        omega.units = 'nondimensional'
        
        kh_0 = nc.createVariable('kh_0', 'f8', ('nparticles', 'next_chems'))
        kh_0.long_name = 'Henrys law constant at 298.15 K'
        kh_0.standard_name = 'kh_0'
        kh_0.units = 'kg/(m^3 Pa)'
        
        neg_dH_solR = nc.createVariable('neg_dH_solR', 'f8', ('nparticles', 
            'next_chems'))
        neg_dH_solR.long_name = 'negative of the enthalpy of solution / R'
        neg_dH_solR.standard_name = 'neg_dH_solR'
        neg_dH_solR.units = 'K'
        
        nu_bar = nc.createVariable('nu_bar', 'f8', ('nparticles', 
            'next_chems'))
        nu_bar.long_name = 'specific volume at infinite dilution'
        nu_bar.standard_name = 'nu_bar'
        nu_bar.units = 'm^3/mol'
        
        B = nc.createVariable('B', 'f8', ('nparticles', 'next_chems'))
        B.long_name = 'diffusivity model coefficient B'
        B.standard_name = 'B'
        B.units = 'm^2/s'
        
        dE = nc.createVariable('dE', 'f8', ('nparticles', 'next_chems'))
        dE.long_name = 'diffusivity model coefficient dE'
        dE.standard_name = 'dE'
        dE.units = 'J/mol'
        
        K_salt = nc.createVariable('K_salt', 'f8', ('nparticles', 
            'next_chems'))
        K_salt.long_name = 'Setschenow salting out correction for solubility'
        K_salt.standard_name = 'K_salt'
        K_salt.units = 'm^3/mol'
    
    # Store the values for each particle in the list
    for i in range(len(particles)):
        
        # Store the variables needed to create dbm particle objects
        if particles[i].particle.issoluble:
            issoluble[i] = 1
            isfluid[i] = 1
            isair[i] = particles[i].particle.isair
            iscompressible[i] = 1
            fp_type[i] = particles[i].particle.fp_type
            calc_delta[i] = particles[i].particle.calc_delta
            if len(particles[i].particle.user_data) == 0:
                extern_data[i] = 0
            else:
                extern_data[i] = 1
                user_data = particles[i].particle.user_data
                for j in range(len(user_composition)):
                    M[i,j] = user_data[user_composition[j]]['M']
                    Pc[i,j] = user_data[user_composition[j]]['Pc']
                    Tc[i,j] = user_data[user_composition[j]]['Tc']
                    Vc[i,j] = user_data[user_composition[j]]['Vc']
                    Tb[i,j] = user_data[user_composition[j]]['Tb']
                    Vb[i,j] = user_data[user_composition[j]]['Vb']
                    omega[i,j] = user_data[user_composition[j]]['omega']
                    kh_0[i,j] = user_data[user_composition[j]]['kh_0']
                    neg_dH_solR[i,j] = \
                        user_data[user_composition[j]]['-dH_solR']
                    nu_bar[i,j] = user_data[user_composition[j]]['nu_bar']
                    B[i,j] = user_data[user_composition[j]]['B']
                    dE[i,j] = user_data[user_composition[j]]['dE']
                    K_salt[i,j] = user_data[user_composition[j]]['K_salt']
            sigma_correction[i] = particles[i].particle.sigma_correction
            if particles[i].particle.calc_delta:
                delta_groups[i,:,:] = particles[i].particle.delta_groups
            else:
                delta_groups[i,:,:] = np.zeros((len(chem_names),15))
            m0[i,:] = particles[i].m0
            rho_p[i] = -1.
            gamma[i] = -1.
            beta[i] = -1.
            co[i] = -1.
        else:
            issoluble[i] = 0
            isair[i] = 0
            if particles[i].particle.isfluid:
                isfluid[i] = 1
            else:
                isfluid[i] = 0
            if particles[i].particle.iscompressible:
                iscompressible[i] = 1
            else:
                iscompressible[i] = 0
            fp_type[i] = 3
            calc_delta[i] = -1
            sigma_correction[i] = 1.
            m0[i,0] = particles[i].m0
            rho_p[i] = particles[i].particle.rho_p
            gamma[i] = particles[i].particle.gamma
            beta[i] = particles[i].particle.beta
            co[i] = particles[i].particle.co
        
        # Store the variables needed to create dispersed_phases SingleParticle
        # or PlumeParticle objects
        T0[i] = particles[i].T0
        K[i] = particles[i].K
        K_T[i] = K_T0[i]
        fdis[i] = particles[i].fdis
        t_hyd[i] = particles[i].t_hyd
        
        if particle_type[0] == 1 or particle_type[0] == 2:
            nb0[i] = particles[i].nb0
            lambda_1[i] = particles[i].lambda_1
        
        if particle_type[0] == 2:
            nb0[i] = particles[i].nb0
            nbe[i] = particles[i].nbe
            lambda_1[i] = particles[i].lambda_1
            integrate[i] = particles[i].integrate
            sim_stored[i] = particles[i].sim_stored
            farfield[i] = particles[i].farfield
            tp[i] = particles[i].t
            xp[i] = particles[i].x
            yp[i] = particles[i].y
            zp[i] = particles[i].z
            try:
                te[i] = particles[i].te
                xe[i] = particles[i].xe
                ye[i] = particles[i].ye
                ze[i] = particles[i].ze
            except:
                pass

def load_particle_from_nc_file(nc):
    """
    Read the complete `particles` list from a netCDF output file
    
    Creates the `particles` list of `SingleParticle`, `PlumeParticle`, or
    `bent_plume_model.Particle` objects from the attributes stored in a 
    netCDF output file.
    
    Parameters
    ----------
    nc : `netCDF4.Dataset` object
        A `netCDF4.Dataset` object that is open and where the particle 
        attributes should be written
    particle_type : int
        The particle type is either 0: `SingleParticle`, 1: `PlumeParticle`
        or 2: `bent_plume_model.Particle`
    X0 : ndarray
        Vector of initial positions for the `bent_plume_model.Particle` 
        objects.
    
    """
    # All particles have the same composition
    chem_names = str(nc.composition).split()
    
    # Load each particle object separately
    particles = []
    for i in range(len(nc.dimensions['nparticles'])):
        
        # Create the correct dbm object
        if nc.variables['issoluble'][i]:
            if nc.variables['extern_data'][i]:
                user_data = {}
                user_composition = str(nc.user_composition).split()
                for j in range(len(user_composition)):
                    user_data[user_composition[j]] = {}
                    user_data[user_composition[j]]['M'] = \
                        nc.variables['M'][i,j]
                    user_data[user_composition[j]]['Pc'] = \
                        nc.variables['Pc'][i,j]
                    user_data[user_composition[j]]['Tc'] = \
                        nc.variables['Tc'][i,j]
                    user_data[user_composition[j]]['Vc'] = \
                        nc.variables['Vc'][i,j]
                    user_data[user_composition[j]]['Tb'] = \
                        nc.variables['Tb'][i,j]
                    user_data[user_composition[j]]['Vb'] = \
                        nc.variables['Vb'][i,j]
                    user_data[user_composition[j]]['omega'] = \
                        nc.variables['omega'][i,j]
                    user_data[user_composition[j]]['kh_0'] = \
                        nc.variables['kh_0'][i,j]
                    user_data[user_composition[j]]['-dH_solR'] = \
                        nc.variables['neg_dH_solR'][i,j]
                    user_data[user_composition[j]]['nu_bar'] = \
                        nc.variables['nu_bar'][i,j]
                    user_data[user_composition[j]]['B'] = \
                        nc.variables['B'][i,j]
                    user_data[user_composition[j]]['dE'] = \
                        nc.variables['dE'][i,j]
                    user_data[user_composition[j]]['K_salt'] = \
                        nc.variables['K_salt'][i,j]
            else:
                user_data = {}
            if nc.variables['calc_delta'][i]:
                delta_groups = nc.variables['delta_groups'][i,:,:]
            else:
                delta_groups = None
            particle = dbm.FluidParticle(chem_names, 
                fp_type=nc.variables['fp_type'][i], 
                user_data=user_data, 
                delta_groups=delta_groups, 
                isair=nc.variables['isair'][i], 
                sigma_correction=nc.variables['sigma_correction'][i])
            m0 = nc.variables['m0'][i,:]
        else:
            if nc.variables['isfluid'][i]:
                isfluid = True
            else:
                isfluid = False
            if nc.variables['iscompressible'][i]:
                iscompressible = True
            else:
                iscompressible = False
            particle = dbm.InsolubleParticle(isfluid, iscompressible, 
                rho_p=nc.variables['rho_p'][i], 
                gamma=nc.variables['gamma'][i], 
                beta=nc.variables['beta'][i], 
                co=nc.variables['co'][i])
            m0 = nc.variables['m0'][i,0]
        
        # Create the right dispersed_phases object
        if nc.variables['particle_type'][0] == 2:
            from tamoc import bent_plume_model as bpm
            particle = bpm.Particle(nc.variables['xp'][i], 
                nc.variables['yp'][i], nc.variables['zp'][i], particle, m0, 
                nc.variables['T0'][i], nc.variables['nb0'][i], 
                nc.variables['lambda_1'][i], nc.variables['P'][0], 
                nc.variables['Sa'][0], nc.variables['Ta'][0], 
                nc.variables['K'][i], nc.variables['K_T'][i], 
                nc.variables['fdis'][i], nc.variables['t_hyd'][i])
            particle.nbe = nc.variables['nbe'][i]
            particle.t = nc.variables['tp'][i]
            particle.integrate = nc.variables['integrate'][i]
            particle.sim_stored = nc.variables['sim_stored'][i]
            particle.farfield = nc.variables['farfield'][i]
            if nc.variables['te'][i] > 0.:
                particle.te = nc.variables['te'][i]
                particle.xe = nc.variables['xe'][i]
                particle.ye = nc.variables['ye'][i]
                particle.ze = nc.variables['ze'][i]
                
        elif nc.variables['particle_type'][0] == 1:
            particle  = PlumeParticle(particle, m0, 
                nc.variables['T0'][i], nc.variables['nb0'][i], 
                nc.variables['lambda_1'][i], nc.variables['P'][0], 
                nc.variables['Sa'][0], nc.variables['Ta'][0], 
                nc.variables['K'][i], nc.variables['K_T'][i], 
                nc.variables['fdis'][i], nc.variables['t_hyd'][i])
        else:
            particle  = SingleParticle(particle, m0, 
                nc.variables['T0'][i], nc.variables['K'][i],
                nc.variables['K_T'][i], nc.variables['fdis'][i],
                nc.variables['t_hyd'][i])
        
        # Add this particle to the particles list
        particles.append(particle)
    
    # Return the list of particles and their composition
    return (particles, chem_names)


# ----------------------------------------------------------------------------
# Functions for shear entrainment
# ----------------------------------------------------------------------------

def shear_entrainment(U, Us, rho, rho_a, b, sin_p, p):
    """
    Compute the entrainment coefficient for shear entrainment 
    
    Computes the entrainment coefficient for the shear entrainment for a top
    hat model.  This code can be used by both the bent plume model and the 
    stratified plume model.  It is based on the concepts for shear entrainment
    in Lee and Cheung (1990) and adapted by the model in Jirka (2004).  The
    model works for pure jets, pure plumes, and buoyant jets.
    
    Parameters
    ----------
    U : float
        Top hat velocity of entrained plume water (m/s)
    Us : float
        Component of the ambient current projected along the plume 
        centerline (m/s)
    rho : float
        Density of the entrained plume fluid (kg/m^3)
    rho_a : float
        Density of the ambient water at the current height (kg/m^3)
    sin_p : float
        Sine of the angle phi from the horizontal with down being positive (up 
        is - pi/2)
        Cosine of the angle theta from the crossflow direction
    p : `bent_plume_model.ModelParams` or `stratified_plume_model.ModelParams`
        Object containing the present model parameters
    
    Returns
    -------
    alpha_s : float
        The shear entrainment coefficient (--)
    
    """
    # Gaussian model jet entrainment coefficient
    alpha_j = p.alpha_j
    
    # Gaussian model plume entrainment coefficient
    if rho_a == rho:
        # This is a pure jet
        alpha_p = 0.
    else:
        # This is a plume; compute the densimetric Gaussian Froude number
        F1 = 2. * np.abs(U - Us) / np.sqrt(p.g * np.abs(rho_a - rho) * (1. + 
             1.2**2) / 1.2**2 / rho_a * b / np.sqrt(2.))
        
        # Follow Figure 13 in Jirka (2004)
        if np.abs(F1**2 / sin_p) > p.alpha_Fr / 0.028:
            alpha_p = - np.sign(rho_a - rho) * p.alpha_Fr * sin_p / F1**2
        else:
            alpha_p = - (0.083 - p.alpha_j) / (p.alpha_Fr / 0.028) * F1**2 / \
                      sin_p * np.sign(rho_a - rho)
    
    # Compute the total shear entrainment coefficient for the top-hat model
    if (np.abs(U - Us) + U) == 0:
        alpha_s = np.sqrt(2.) * alpha_j
    else:
        alpha_s = np.sqrt(2.) * (alpha_j + alpha_p) * 2. * U / \
                  (np.abs(U - Us) + U)
    
    # Return the total shear entrainment coefficient
    return alpha_s


# ----------------------------------------------------------------------------
# Functions for hydrate skin model
# ----------------------------------------------------------------------------

def hydrate_formation_time(dbm_obj, z, m, T, profile):
    """
    Compute the hydrate formation time
    
    Computes the time to form a hydrate shell using the empirical model from
    Jun et al. (2015).  If the particle is above the hydrate stability zone,
    the formation time is np.inf.  If it is below the hydrate statbility
    line, the maximum formation time t_star is computed based on the particle
    diameter.  For high hydrate subcooling, the formation time can be 
    accelerated by a factor phi = f(extent of subcooling).  The final 
    hydrate formation time is t_hyd = phi * t_star.
    
    The idea behind this model is that bubbles or droplets in the ocen may 
    form a hydrate shell that results in dirty-bubble mass and heat transfer
    and rise velocity.  This algorithm sets the time to form the shell based
    on measured field data by Rehder et al. (2002).  The model has been 
    validated to field data in Romer et al. (2012), McGinnis et al. (2006), 
    Warkinski et al. (2014), and the GISR field experiments.
    
    Parameters
    ----------
    dbm_obj : `dbm.FluidParticle` object
        Discrete bubble model `dbm.FluidParticle` object.  Since this method
        must calculate the hydrate stability temperature, it cannot be used
        on `dbm.InsolubleParticle` objects.  A hydrate formation time can 
        still be set for those particles, but not estimated from this 
        function.
    z : float
        Release depth (m)
    m : ndarray
        Initial masses of the components of the `dbm_obj` (kg)
    T : float
        Initial temperature of the of `dbm~_obj` particle (K)
    profile : `ambient.Profile` object
        An object containing the ambient CTD data and associated methods.  
    
    Returns
    -------
    t_hyd : float
        Hydrate formation time (s)
    
    """
    # Get the ambient properties at the depth
    Ta, Sa, P = profile.get_values(z, ['temperature', 'salinity', 
                                   'pressure'])
    
    # Compute the diameter of the particle
    de = dbm_obj.diameter(m, T, P)
    
    # Estimate the hydrate stability temperature
    T_hyd = dbm_obj.hydrate_stability(m, P)
    
    if T_hyd < Ta:
        # The particle is above the hydrate stability zone...assume hydrates
        # never form.
        t_hyd = np.inf
    
    else:
        # The particle is below the hydrate stability zone, compute the 
        # skin formation time.  
        t_star = 85206. * de - 243.276
        
        if t_star < 0.:
            # Hydate skin formation time cannot be zero.
            t_star = 0.
        
        # Get the subcooling acceleration factor.
        phi = -0.1158 * (T_hyd - Ta) + 2.2692
        
        if phi > 1.:
            # Acceleration cannot be more than one.
            phi = 1.
            
        elif phi < 0.:
            # Acceleration cannot be less than 0.
            phi = 0.
        
        # Compute the in situ hydrate formation time
        t_hyd = phi * t_star
    
    # Return the formation time
    return t_hyd


# ----------------------------------------------------------------------------
# Functions to generate initial conditions for models using these objects
# ----------------------------------------------------------------------------

def zfe_volume_flux(profile, particles, p, X0, R):
    """
    Initial volume for a multiphase plume
    
    Uses the Wueest et al. (1992) plume Froude number method to estimate
    the amount of entrainment at the source of a dispersed phase plume with
    zero continuous phase flux (e.g., a pure bubble, droplet, or particle 
    plume)
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the single bubble model simulation.
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or 
        `bent_plume_model.Particle` objects describing each dispersed phase 
        in the simulation
    p : `stratified_plume_model.ModelParams` or `bent_plume_model.ModelParams`
        Object containing the fixed model parameters for one of the integral 
        plume models
    X0 : float
        (x, y, depth) coordinates of the release point (m)
    R : float
        Radius of the equivalent circular area of the release (m)
    
    """
    # The initial condition is valid at the diffuser (e.g., no virtual point
    # source for the Wuest et al. 1992 initial conditions).  Send back 
    # exactly what the user supplied
    X = X0
    
    # Get X0 as a three-dimensional vector for generality
    if not isinstance(X0, np.ndarray):
        if not isinstance(X0, list):
            X0 = np.array([0., 0., X0])
        else:
            X0 = np.array(X0)
    
    # Get the ambient conditions at the discharge
    Ta, Sa, P = profile.get_values(X0[2], ['temperature', 'salinity', 
                                   'pressure'])
    rho = seawater.density(Ta, Sa, P)
    
    # Update the particle objects and pull out the multiphase properties.
    # Since this is the release, the particle age is zero.
    lambda_1 = np.zeros(len(particles))
    us = np.zeros(len(particles))
    rho_p = np.zeros(len(particles))
    Q = np.zeros(len(particles))
    for i in range(len(particles)):
        particles[i].update(particles[i].m, particles[i].T, P, Sa, Ta, 0.)
        lambda_1[i] = particles[i].lambda_1
        us[i] = particles[i].us
        rho_p[i] = particles[i].rho_p
        Q[i] = np.sum(particles[i].m) * particles[i].nb0 / rho_p[i]
    
    # Compute the buoyancy flux weighted average of lambda_1
    lambda_ave = bf_average(particles, rho, p.g, p.rho_r, lambda_1)
    
    # Calculate the initial velocity of entrained ambient fluid
    u_0 = np.sum(Q) / (np.pi * (lambda_ave * R)**2)
    u = wuest_ic(u_0, particles, lambda_1, lambda_ave, us, rho_p, rho, Q, R, 
                 p.g, p.Fr_0)
    
    # The initial plume width is the discharge port width
    A = np.pi * R**2
    
    # Calcualte the volume flux
    Q = A * u
    
    return (Q, A, X, Ta, Sa, P, rho)


def wuest_ic(u_0, particles, lambda_1, lambda_ave, us, rho_p, rho, Q, R, 
             g, Fr_0):
    """
    Compute the initial velocity of entrained ambient fluid
    
    Computes the initial velocity of the entrained ambient fluid following 
    the method in Wueest et al. (1992).  This method is implicit; thus, an 
    initial guess for the velocity and a root-finding approach is required.
    
    Parameters
    ----------
    u_0 : float
        Initial guess for the entrained fluid velocity (m/s)
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or 
        `bent_plume_model.Particle` objects describing each dispersed phase 
        in the simulation
    lambda_1 : ndarray
        Spreading rate of the each dispersed phase particle in a plume (--)
    lambda_ave : float
        Buoyancy flux averaged value of lambda_1 (--)
    us : ndarray
        Slip velocity of each of the dispersed phase particles (m/s)
    rho_p : ndarray
        Density of each of the dispersed phase particles (kg/m^3)
    rho : float
        Density of the local ambient continuous phase fluid (kg/m^3)
    Q : ndarray
        Total volume flux of particles for each dispersed phase (m^3/s)
    R : float
        Radius of the release port (m)
    g : float
        Acceleration of gravity (m/s^2)
    Fr_0 : float
        Desired initial plume Froude number (--)
    
    Returns
    -------
    u : float
        The converged value of the entrained fluid velocity in m/s at the 
        release location in order to achieve the specified value of Fr_0.
    
    """
    # The Wuest et al. (1992) initial condition is implicit; define the 
    # residual for use in a root-finding algorithm
    def residual(u):
        """
        Compute the residual of the Wueest et al. (1992) initial condition
        using the current guess for the initial velocity u.
        
        Parameters
        ----------
        u : float
            the current guess for the initial velocity (m/s)
        
        Notes
        -----
        All parameters of `wuest_ic` are global to this function since it is
        a subfunction of `wuest_ic`.
        
        """
        # Get the void fraction for the current estimate of the mixture of 
        # dispersed phases and entrained ambient water
        xi = np.zeros(len(particles))
        for i in range(len(particles)):
            xi[i] = Q[i] / (np.pi * lambda_1[i]**2 * R**2 * (us[i] + 
                    2. * u / (1. + lambda_1[i]**2)))
        
        # Get the mixed-fluid plume density
        rho_m = np.sum(xi * rho_p) + (1. - np.sum(xi)) * rho
        
        # Calculate the deviation from the desired Froude number
        return Fr_0 - u / np.sqrt(2. * lambda_ave * R * g * 
                                  (rho - rho_m) / rho_m)
    
    return fsolve(residual, u_0)[0]


def bf_average(particles, rho, g, rho_r, parm):
    """
    Compute a buoyancy-flux-weighted average of `parm`
    
    Computes a weighted average of the values in `parm` using the kinematic
    buoyancy flux of each particle containing parm as the weight in the 
    average calculation.  
    
    Parameters
    ----------
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or 
        `bent_plume_model.Particle` objects describing each dispersed phase 
        in the simulation
    rho : float
        Local density of ambient fluid outside plume (kg/m^3).
    g : float
        Acceleration of gravity (m/s^2).
    rho_r : float
        Model reference density (kg/m^3).
    parm : ndarray
        Numpy array of parameters to average, one value for each 
        dispersed phase entry (same as elements in parm).
    
    Returns
    -------
    parm_ave : float
        The weighted average of `parm`.
    
    """
    # Compute the total buoyancy flux of each dispersed phase particle in the 
    # simulation
    F = np.zeros(len(particles))
    for i in range(len(particles)):
        # Get the total particle volume flux
        Q = np.sum(particles[i].m) * particles[i].nb0 / particles[i].rho_p
        # Compute the particle kinematic buoyancy flux
        F[i] = g * (rho - particles[i].rho_p) / rho_r * Q
    
    # Return the buoyancy-flux-weighted value of parm
    if np.sum(F) == 0.:
        parm = 0.
    else:
        parm = np.sum(F * parm) / np.sum(F)
    
    return parm


def get_chem_names(particles):
    """
    Create a list of chemical names for the dispersed phase particles
    
    Reads the composition attribute of each particle in a `particles` list
    and compiles a unique list of particle names.
    
    Parameters
    ----------
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or 
        `bent_plume_model.Particle` objects describing each dispersed phase 
        in the simulation
    
    Returns
    -------
    chem_names : str list
        List of the chemical composition of particles undergoing dissolution
        in the `particles` list
    
    """
    # Initialize a list to store the names
    chem_names = []
    
    # Add the chemicals that are part of the particle composition
    for i in range(len(particles)):
        if particles[i].particle.issoluble:
            chem_names += [chem for chem in particles[i].composition if
                           chem not in chem_names]
    
    # Return the list of chemical names
    return chem_names


def particles_state_space(particles, nb):
    """
    Create the state space describing the dispersed phase properties
    
    Constructs a complete state space of masses and heat content for all of
    the particles in the `particles` list.
    
    Parameters
    ----------
    particles : list of `Particle` objects
        List of `SingleParticle`, `PlumeParticle`, or 
        `bent_plume_model.Particle` objects describing each dispersed phase 
        in the simulation
    nb : ndarray
        Array of particle numbers for forming the state space.  nb can be in 
        number/T, which will give state space variables in mass flux (M/T) or
        in number, which will give state space variables in mass.
    
    Returns
    -------
    y : ndarray
        Array of state space variables for the `particles` objects.
    
    """
    # Get the state variables of each particle, one particle as a time
    y = []
    for i in range(len(particles)):
        
        # Masses of each element in the particle
        y.extend(particles[i].m * nb[i])
        
        # Add in the heat flux of the particle
        y.append(np.sum(particles[i].m) * nb[i] * 
                 particles[i].cp * particles[i].T)
        
        # Initialize the particle age to zero
        y.append(0.)
        
        # Initialize the particle positions to the center of the plume
        y.extend([0., 0., 0.])
    
    # Return the state space as a list
    return y

