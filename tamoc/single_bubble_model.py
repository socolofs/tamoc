"""
Single Bubble Model
===================

Simulate the trajectory of a particle rising through the water column

This module defines the classes, methods, and functions necessary to simulate
the rise of a single particle (bubble, droplet or solid particle) through the
water column. The ambient water properties are provided through the
`ambient.Profile` class object, which contains a netCDF4-classic dataset of
CTD data and the needed interpolation methods. The `dbm` class objects
`dbm.FluidParticle` and `dbm.InsolubleParticle` report the properties and
behavior of the particle during the simulation.

Notes
-----
This model solves for the trajectory `vec(x)` by the simple transport 
equation::

    d vec(x) / dt = vec(u)

where `vec(u)` is the vector velocity of the particle, which may include the
rise velocity and an ambient current. The rise velocity depends on the
particle size, which changes with pressure (if compressible) and as a result
of mass transfer (when soluble). Hence, this equation is usually coupled to a
system of equations for the change in mass of each chemical component in the
particle `m_i`, given by::

    d (m_i) / dt = - beta * A * (Cs - C)

where `Cs` is the local solubility of component `i` and `C` is the local 
concentration of component `i` in the surrounding water; `beta` is the mass
transfer coefficient and `A` is the surface area.  Methods to compute
`beta`, `Cs`, and `A` are provided in the `dbm` module.  Since source fluids
may have different temperature than the ambient, heat transfer is also 
modeled::

    d H / dt = - rho_p * cp * A * beta_T * (T - Ta)

where `H` is the heat content, given by `m_p * cp * T`; `beta_T` is the heat 
transfer coefficient and `m_p` is the total mass of the particle.  Since some 
mass is lost due to dissolution, the particle temperature must be adjusted 
by::

    d H / dt = cp * d (m_p) / dt * T        # Note d (m_p) / dt < 0

and for the heat of solution, using::

    d H / dt = sum (d (m_i) /dt * dH_solR * Ru / M)

where `dH_solR` is the enthalpy of solution divided by the universal gas 
constant (`Ru`) and `M` is the molecular weight of constituent `i`.

When the particle becomes very small, the heat transfer and dissolution 
become unstable, leading to rapid oscillations in the predicted particle 
temperature.  To avoid this problem, this module accounts for heat transfer
until the particle temperature reaches equilibrium with the seawater (which
happens very quickly).  Thereafter, the particle is assumed to be equal to 
the temperature of the ambient water.  

The equations for heat and mass transfer and for slip velocity are
discontinuous at the boundaries between particle shapes (e.g., ellipsoid and
spherical cap, etc.), and this can sometimes lead to the solution getting
stuck at the shape transition. The convergence criteria for the ODE solver are
set at an optimal compromise for accuracy and for allowing a diverse range of
particles to be simulated. Nonetheless, there are situations where these
discontinuities may still break the solution. 

Finally, if the diameter of a fluid particle is observed to rapidly increase, 
this is usually associated with a phase change from liquid to gas.  The
diagnostic plots help to identify these effects by plotting the state space
together with several descriptive variables, including diameter, density, 
and shape.  

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from tamoc import dbm
from tamoc import ambient
from tamoc import seawater

from netCDF4 import Dataset
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import copy
from scipy import integrate
from string import join, capwords
from warnings import warn
import os

class Model(object):
    """
    Master class object for controlling and post-processing the simulation
    
    This is the main program interface, and the only object or function in 
    this module that the user should call.  At instantiation, the model 
    parameters and the ambient water column data are organized.  For a given
    simulation, the user passes a `dbm` particle object and its initial 
    conditions (e.g., mass, temperature, location) to the `simulate` method,
    and the object computes the trajectory and plots the resulting path and
    particle properties.  The simulation results can be stored to and loaded
    from a netCDF file using the `save_sim` and `load_sim` methods.  An 
    ascii table of data for the state space for reading into other programs 
    (e.g., Matlab) can be output using the `save_txt` method.  The object
    can only store simulation results in its attribute variables for one 
    simulation at a time.  Each time a new simulation is run or a past 
    simulation results file is loaded, the current simulation (if present) is
    overwritten.  
    
    Parameters
    ----------
    profile : `ambient.Profile` object, default = None
        An object containing the ambient CTD data and associated methods.  
        The netCDF dataset stored in the `ambient.Profile` object may be open 
        or closed at instantiation.  If open, the initializer will close the 
        file since this model does not support changing the ambient data once 
        initialized.
    simfile: str, default = None
        File name of a netCDF file containing the results of a previous 
        simulation run.  
    
    Attributes
    ----------
    profile : `ambient.Profile` object
        Ambient CTD data for the model simulation
    p : `ModelParams` object
        Set of model parameters not adjustable by the user
    sim_stored : bool
        Flag indicating whether or not simulation results exist in the object
        namespace
    particle : `Particle` object
        Interface to the `dbm` module and container for particle-specific 
        parameters
    t : ndarray
        Times (s) associated with the state space
    y : ndarray
        State space along the trajectory of the particle
    z0 : float
        The release depth (m)
    de : float
        Initial diameter of the particle (m)
    yk : ndarray
        Initial mole fractions of each chemical component (--)
    T0 : float, optional
        Initial temperature (K) of the particle at release
    K : float, default = 1.
        Mass transfer reduction factor (--)
    K_T : float, default = 1.
        Heat transfer reduction factor (--)
    fdis : float, default = 1.e-6
        Remainder fractin that turns off dissolution for each component (--)
    delta_t : float, default = 0.1 s
        Maximum time step to use (s) in the simulation output   
    
    See Also
    --------
    simulate, save_sim, load_sim
    
    Notes
    -----
    The `Model` object will be initialized either with the `profile` data 
    making it read to start a new simulation or with the results of a previous
    simulation stored in `simfile`.
    
    """
    def __init__(self, profile=None, simfile=None):
        super(Model, self).__init__()
        
        if profile is None:
            # Create a Model object from a saved file
            self.load_sim(simfile)
        else:
            # Create a new Model object
            self.profile = profile
            profile.close_nc()
            
            # Enter the model parameters that the user cannot adjust
            self.p = ModelParams(self.profile)
            self.sim_stored = False
    
    def simulate(self, particle, z0, de, yk, T0=None, K=1., K_T=1., 
                 fdis=1.e-6, delta_t=0.1):
        """
        Simulate the trajectory of a particle from given initial conditions
        
        Simulate the trajectory of a particle (bubble, droplet or solid 
        particle) until the particle dissolves or until it reaches the free
        surface.  
        
        Parameters
        ----------
        particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
            Object describing the properties and behavior of the particle.
        z0 : float
            The release depth (m) of the particle in the simulation
        de : float
            Initial diameter of the particle (m)
        yk : ndarray
            Initial mole fractions of each component in the particle (--)
        T0 : float, optional
            Initial temperature (K) of the particle at release if not equal 
            to the temperature of the surrounding fluid.  If omitted, the 
            model will set T0 to the ambient temperature.
        K : float, default = 1.
            Mass transfer reduction factor (--). Pre-multiplies the mass 
            transfer coefficients providing amplification (>1) or retardation 
            (<1) of the dissolution.  
        K_T : float, default = 1.
            Heat transfer reduction factor (--). Pre-multiplies the heat 
            transfer coefficient providing amplification (>1) or retardation 
            (<1) of the heat flux.
        fdis : float, default = 1.e-6
            Fraction of the initial total mass remaining (--) for each 
            component in the particle when the particle should be considered 
            dissolved.
        delta_t : float, default = 0.1 s
            Maximum time step to use (s) in the simulation.  The ODE solver
            in `calculate_path` is set up with adaptive step size integration, 
            so in theory this value determines the largest step size in the 
            output data, but not the numerical stability of the calculation.
        
        See Also
        --------
        post_process, calculate_path, plot_state_space
        
        Notes
        -----
        This method fills the object attributes `particle`, `t` and `y` 
        following successful simulation of the particle trajectory.  It also
        stores all the input variables as object attributes that do not
        change during simulation.
        
        """
        # Make sure yk is an ndarray
        if not isinstance(yk, np.ndarray):
            if not isinstance(yk, list):
                yk = np.array([yk])
            else:
                yk = np.array(yk)
        
        # Check if the right number of elements are in yk
        if len(yk) != len(particle.composition):
            print 'Wrong number of mole fractions:'
            print '   yk : %d entries' % len(yk)
            print '   composition : %d components\n' % \
                                    len(particle.composition)
            return
        
        # Save the input variables since they may be modified during the
        # simulation
        self.z0 = z0
        self.de = de
        self.yk = yk
        self.T0 = T0
        self.K = K
        self.K_T = K_T
        self.fdis = fdis
        self.delta_t = delta_t
        
        # Get the initial conditions for the simulation run
        (self.particle, y0) = sbm_ic(self.profile, particle, z0, de, yk, T0, 
                                     K, K_T, fdis)
        
        # Open the simulation module
        print '\n-- TEXAS A&M OIL-SPILL CALCULATOR (TAMOC) --\n'
        
        # Calculate the trajectory
        print 'Calculate the trajectory...'
        self.t, self.y = calculate_path(self.profile, self.particle, self.p, 
                                        y0, delta_t)
        print 'Simulation complete.\n '
        self.sim_stored = True
        
    def save_sim(self, fname, profile_path, profile_info):
        """
        Save the current simulation results
        
        Save the current simulation results and the model parameters so that
        all information needed to rebuild the class object is stored in a 
        file.  The output data are stored in netCDF4-classic format.
        
        Parameters
        ----------
        fname : str
            File name of the file to write
        profile_path : str
            String stating the file path relative to the directory where
            the output will be saved to the ambient profile data.  
        profile_info : str
            Text describing the ambient profile data.
        
        Notes
        -----
        It does not make sense to store the ambient data together with every
        simulation output file.  On the other hand, the simulation results
        may be meaningless without the context of the ambient data.  The 
        parameter `profile_path` provides a means to automatically load the
        ambient data assuming the profile data are kept in the same place 
        relative to the output file.  Since this cannot be guaranteed, the 
        `profile_info` variable provides additional descriptive information 
        so that the ambient data can be identified if they have been moved.
        
        """
        if self.sim_stored is False:
            print 'No simulation results to store...'
            print 'Saved nothing to netCDF file.\n'
            return
        
        # Create the netCDF dataset object
        nc = Dataset(fname, 'w', format='NETCDF4_CLASSIC')
        
        # Set the global attributes
        nc.Conventions = 'TAMOC Single Bubble Model'
        nc.Metadata_Conventions = 'TAMOC Python Module'
        nc.featureType = 'profile'
        nc.cdm_data_type = 'Profile'
        nc.nodc_template_version = 'NODC_NetCDF_Profile_Orthogonal_Template_v1.0'
        nc.title = 'Simulation results for the TAMOC Single Bubble Model'
        nc.summary = profile_path
        nc.source = profile_info
        nc.date_created = datetime.today().isoformat(' ')
        nc.date_modified = datetime.today().isoformat(' ')
        nc.history = 'Creation'
        nc.composition = ' '.join(self.particle.composition)
        
        # Create variables for the dimensions
        z = nc.createDimension('z', None)
        p = nc.createDimension('profile', 1)
        nchems = nc.createDimension('nchems', len(self.particle.composition))
        
        # Store the dbm particle object instantiation variables
        if self.particle.particle.issoluble:
            nc.issoluble = 'True'
            fp_type = nc.createVariable('fp_type', 'i4', ('profile',))
            fp_type.long_name = 'fluid phase (0: gas, 1: liquid)'
            fp_type.standard_name = 'fp_type'
            fp_type.units = 'nondimensional'
            fp_type[0] = self.particle.particle.fp_type
        else:
            nc.issoluble = 'False'
            if self.particle.particle.isfluid:
                nc.isfluid = 'True'
            else:
                nc.isfluid = 'False'
            if self.particle.particle.iscompressible:
                nc.iscompressible = 'True'
            else:
                nc.iscompressible = 'False'
            rho_p = nc.createVariable('rho_p', 'f8', ('profile',))
            rho_p.long_name = 'particle density'
            rho_p.standard_name = 'rho_p'
            rho_p.units = 'kg/m^3'
            rho_p[0] = self.particle.particle.rho_p
            gamma = nc.createVariable('gamma', 'f8', ('profile',))
            gamma.long_name = 'API Gravity'
            gamma.standard_name = 'gamma'
            gamma.units = 'deg API'
            gamma[0] = self.particle.particle.gamma
            beta = nc.createVariable('beta', 'f8', ('profile',))
            beta.long_name = 'thermal expansion coefficient'
            beta.standard_name = 'beta'
            beta.units = 'Pa^(-1)'
            beta[0] = self.particle.particle.beta
            co = nc.createVariable('co', 'f8', ('profile',))
            co.long_name = 'isothermal compressibility coefficient'
            co.standard_name = 'co'
            co.units = 'K^(-1)'
            co[0] = self.particle.particle.co
        
        # Create the time variable
        time = nc.createVariable('t', 'f8', ('z',))
        time.long_name = 'time coordinate'
        time.standard_name = 'time'
        time.units = 'seconds since release'
        time.axis = 'T'
        time[:] = self.t
        
        # Create the depth variable
        z = nc.createVariable('z', 'f8', ('z',))
        z.long_name = 'depth below the water surface'
        z.standard_name = 'depth'
        z.units = 'm'
        z.axis = 'Z'
        z.positive = 'down'
        z.valid_min = np.min(self.y)
        z.valid_max = np.max(self.y)
        z[:] = self.y[:,0]
        
        # Create variables for the chemical components
        for i in range(len(self.particle.composition)):
            chem = nc.createVariable(self.particle.composition[i], 'f8', 
                                     ('z',))
            chem.long_name = join(self.particle.composition[i].split('_'))
            chem.standard_name = capwords(chem.long_name)
            chem.units = 'kg'
            chem.coordinate = 'time'
            chem[:] = self.y[:,i+1]
        
        # Create a variable for the heat
        h = nc.createVariable('h', 'f8', ('z',))
        h.long_name = 'heat content'
        h.standard_name = 'heat'
        h.units = 'J'
        h.coordinate = 'time'
        h[:] = self.y[:,-1]
        
        # Store the model attributes
        z0 = nc.createVariable('z0', 'f8', ('profile',))
        z0.long_name = 'release depth'
        z0.standard_name = 'z0'
        z0.units = 'm'
        z0[0] = self.z0
        
        de = nc.createVariable('de', 'f8', ('profile',))
        de.long_name = 'initial depth'
        de.standard_name = 'de'
        de.units = 'm'
        de[0] = self.de
        
        yk = nc.createVariable('yk', 'f8', ('nchems',))
        yk.long_name = 'initial mole fractions'
        yk.standard_name = 'yk'
        yk.units = 'mole fraction'
        yk[:] = self.yk
        
        T0 = nc.createVariable('T0', 'f8', ('profile',))
        T0.long_name = 'initial temperature'
        T0.standard_name = 'T0'
        T0.units = 'K'
        if self.T0 is None:
            T0[0] = self.profile.get_values(self.z0, 'temperature')
        else:
            T0[0] = self.T0
        
        K = nc.createVariable('K', 'f8', ('profile',))
        K.long_name = 'mass transfer reduction factor'
        K.standard_name = 'K'
        K.units = 'nondimensional'
        K[0] = self.K
        
        K_T = nc.createVariable('K_T', 'f8', ('profile',))
        K_T.long_name = 'heat transfer reduction factor'
        K_T.standard_name = 'K_T'
        K_T.units = 'nondimensional'
        K_T[0] = self.K_T
        
        fdis = nc.createVariable('fdis', 'f8', ('profile',))
        fdis.long_name = 'dissolution criteria'
        fdis.standard_name = 'fdis'
        fdis.units = 'nondimensional'
        fdis[0] = self.fdis
        
        delta_t = nc.createVariable('delta_t', 'f8', ('profile',))
        delta_t.long_name = 'maximum simulation output time step'
        delta_t.standard_name = 'delta_t'
        delta_t.units = 'seconds'
        delta_t[0] = self.delta_t
        
        # Close the netCDF dataset
        nc.close()
    
    def save_txt(self, fname, profile_path, profile_info):
        """
        Save the state space in ascii text format for exporting
        
        Save the state space (dependent and independent variables) in an 
        ascii text file for exporting to other programs (e.g., Matlab).
        
        Parameters
        ----------
        fname : str
            Name of the output file
        profile_path : str
            String stating the file path relative to the directory where
            the output will be saved to the ambient profile data.  
        profile_info : str
            Text describing the ambient profile data (less than 60 
            characters).
            
        Notes
        -----
        The output data will be organized in columns, with each column
        as follows:
        
            0 : Time (s)
            1 : Depth (m)
            2 : (n-1) : Masses of the particle components (kg)
            n : Heat (m_p * cp * T) (J)
        
        A header will be written at the top of the file with the specific 
        details for that file.
        
        The file is written using the `numpy.savetxt` method.
        
        """
        if self.sim_stored is False:
            print 'No simulation results to store...'
            print 'Saved nothing to txt file.\n'
            return
        
        # Create the header string that contains the column descriptions
        p_list = ['Single Bubble Model ASCII Output File \n']
        p_list.append('Created: ' + datetime.today().isoformat(' ') + '\n')
        p_list.append('\n')
        p_list.append('Column Descriptions:\n')
        p_list.append('    0:  Time in s\n')
        p_list.append('    1:  Depth in m\n')
        for i in range(len(self.particle.composition)):
            p_list.append('    %d:  Mass of %s in particle in kg\n' % \
                          (i+2, self.particle.composition[i]))
        p_list.append('    %d: Heat content (m_p * cp * T) in J' % (i+3))
        p_list.append('\n')
        p_list.append('Ambient Profile: %s' % (profile_path))
        p_list.append('Ambient Info:    %s' % (profile_info))
        p_list.append('\n')
        header = ''.join(p_list)
        
        # Assemble the output data
        data = np.hstack((np.atleast_2d(self.t).transpose(), self.y))
        
        # Write the text file
        # For numpy version 1.7.0 or later:
        # np.savetxt(fname, data, header=header, comments='%')
        # Otherwise:
        np.savetxt(fname, data)
    
    def load_sim(self, fname):
        """
        Load in a saved simulation result file for post-processing
        
        Load in a saved simulation result file and rebuild the `Model`
        object attributes.  The input files are in netCDF4-classic data 
        format.
        
        Parameters
        ----------
        fname : str
            File name of the file to read
        
        Notes
        -----
        This method will attempt to load the ambient profile data from the
        `profile_path` attribute of the `fname` netCDF file.  If the load 
        fails, a warning will be reported to the terminal, but the other 
        steps of loading the `Model` object attributes will be performed.
        
        """
        # Open the netCDF dataset object containing the simulation results
        nc = Dataset(fname)
        
        # Try to open the profile data
        try:
            nc_path = os.path.normpath(os.path.join(os.getcwd(), \
                                       os.path.dirname(fname)))
            prf_path = os.path.normpath(os.path.join(nc_path, nc.summary))
            amb_data = Dataset(prf_path)
            self.profile = ambient.Profile(amb_data, chem_names='all')
            self.profile.close_nc()
            self.p = ModelParams(self.profile)
        except RuntimeError:
            message = ['File not found: %s' % prf_path]
            message.append(' ... Continuing without profile data')
            warn(''.join(message))
        
        # Create the DBM model particle object
        composition = nc.composition.split()
        if nc.issoluble == 'True':
            particle = dbm.FluidParticle(composition, 
                nc.variables['fp_type'][0])
        else:
            if nc.isfluid == 'True':
                isfluid = True
            else:
                isfluid = False
            if nc.iscompressible == 'True':
                iscompressible = True
            else:
                iscompressible = False
            particle = dbm.InsolubleParticle(isfluid, iscompressible, 
                rho_p=nc.variables['rho_p'][0], 
                gamma=nc.variables['gamma'][0], 
                beta=nc.variables['beta'][0], 
                co=nc.variables['co'][0])
        
        # Extract the state space data
        self.t = nc.variables['t'][:]
        z = nc.variables['z'][:]
        m = np.zeros((len(self.t), len(particle.composition)))
        for i in range(len(particle.composition)):
            m[:,i] = nc.variables[particle.composition[i]][:]
        h = nc.variables['h'][:]
        self.y = np.hstack((np.atleast_2d(z).transpose(), m, 
                            np.atleast_2d(h).transpose()))
        
        # Extract the initial conditions
        self.z0 = nc.variables['z0'][0]
        self.de = nc.variables['de'][0]
        self.yk = nc.variables['yk'][:]
        self.T0 = nc.variables['T0'][0]
        self.K = nc.variables['K'][0]
        self.K_T = nc.variables['K_T'][0]
        self.fdis = nc.variables['fdis'][0]
        self.delta_t = nc.variables['delta_t'][0]
        
        # Reload the Particle object
        self.particle = Particle(particle, m[0,1:-1], self.T0, self.K, 
                                 self.K_T, self.fdis)
        
        # Close the netCDF dataset
        nc.close()
        self.sim_stored = True
    
    def post_process(self, fig=1):
        """
        Plot the simulation state space and key interrogation parameters
        
        Plot the standard set of post-processing figures, including the 
        state space and the key derived variables.
        
        Parameters
        ----------
        fig : int
            Figure number to pass to the plotting methods
        
        See Also
        --------
        plot_state_space
        
        """
        if self.sim_stored is False:
            print 'No simulation results to plot...'
            print 'Plotting nothing.\n'
            return
        
        # Plot the results        
        print 'Plotting the results...'
        plot_state_space(self.profile, self.particle, self.p, self.t, 
                         self.y, fig)
        print 'Done.\n'
    

class ModelParams(object):
    """
    Fixed model parameters for the single bubble model
    
    This class stores the set of model parameters that should not be adjusted
    by the user and that are needed by the single bubble model.
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the single bubble model simulation.
        
    Attributes
    ----------
    rho_r : float
        Reference density (kg/m^3) evaluated at mid-depth of the water body.
    
    """
    def __init__(self, profile):
        super(ModelParams, self).__init__()
        
        # Store a reference density for the water column
        z_ave = profile.z_max - (profile.z_max - profile.z_min) / 2.
        T, S, P = profile.get_values(z_ave, ['temperature', 'salinity', 
                                     'pressure'])
        self.rho_r = seawater.density(T, S, P)
    

class Particle(object):
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
    fdis : float, default = 0.01
        Fraction of the initial total mass (--) remaining when the particle 
        should be considered dissolved.
    
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
    def __init__(self, dbm_particle, m0, T0, K=1., K_T=1., fdis=1.e-6):
        super(Particle, self).__init__()
        
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
        
        # Store parameters to track the dissolution of the initial masses
        self.diss_indices = self.m0 > 0
    
    def properties(self, m, T, P, Sa, Ta):
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
        if self.K_T > 0. and np.abs(Ta - T) < 0.1:
            self.K_T = 0.
        
        # Use the right temperature
        if self.K_T == 0.:
            T = Ta
        
        # Distinguish between soluble and insoluble particles
        if self.particle.issoluble:
            
            # Get the DBM results
            m[m<0] = 0.   # stop oscillations at small mass
            shape, de, rho_p, us, A, Cs, beta, beta_T = \
                self.particle.return_all(m, T, P, Sa, Ta)
            
            # Turn off dissolution for "dissolved" components
            frac_diss = m[self.diss_indices] / self.m0[self.diss_indices]
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
                self.particle.return_all(m[0], T, P, Sa, Ta)
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


# ----------------------------------------------------------------------------
# Functions to compute the trajectory
# ----------------------------------------------------------------------------

def calculate_path(profile, particle, p, y0, delta_t):
    """
    Calculate the trajectory of a particle
    
    Calculate the trajectory of a particle by integrating its path using
    the `scipy.integrate.ode` object and associated methods.
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        Ambient CTD data for the model simulation
    particle : `Particle` object
        Object describing the properties and behavior of the particle.
    p : `ModelParams` object
        Collection of model parameters passed to `derivs`.
    y0 : ndarray
        Initial values of the state space (depth in m, masses in kg, and heat 
        content in J of the particle) at the release point
    delta_t : float
        Maximum step size (s) to take in the integration
    
    Notes
    -----
    The differential equation in `derivs` is written with respect to time, so
    the independent variable in this simulation is time.  The vertical 
    coordinate; therefore, becomes a dependent variable, along with the masses
    of each component in the particle and the particle temperature.  Thus, 
    the state space is::
    
        y = np.hstack((z0, m0, H0))
    
    where `H0` is the initial heat content, `m_p * cp * T0`.  The variables 
    in the state space can be returned by::
    
        >>> import seawater
        >>> z = y[0]
        >>> m = y[1:-1]
        >>> T = y[-1] / (np.sum(y[1:-1]) * particle.cp)
    
    """
    # Create the integrator object:  use "vode" with "backward 
    # differentiation formula" for stiff ODEs
    r = integrate.ode(derivs).set_integrator('vode', method='bdf', atol=1.e-6, 
        rtol=1e-3, order=5, max_step=delta_t)
    
    # Initialize the state space
    t0 = 0.
    r.set_initial_value(y0, t0)
    
    # Set passing variables for derivs method
    r.set_f_params(profile, particle, p)
    
    # Create vectors (using the list data type) to store the solution
    t = [t0]
    y = [y0]
    
    # Integrate to the free surface (profile.z_min)
    k = 0
    psteps = 250.
    stop = False
    while r.successful() and not stop:
        
        # Print progress to the screen
        if np.remainder(np.float(k), psteps) == 0.:
            print '    Depth:  %g (m), t:  %g (s), k: %d' % \
                (r.y[0], t[-1], k)
        
        # Perform one step of the integration
        r.integrate(t[-1] + delta_t, step=True)
        
        # Store the results
        if particle.K_T == 0:
            # Make the state-space heat correct
            Ta = profile.get_values(r.y[0], 'temperature')
            r.y[-1] = np.sum(r.y[1:-1]) * particle.cp * Ta
        t.append(r.t)
        y.append(r.y)
        k += 1
        
        # Evaluate stop criteria
        if r.successful():
            # Check if bubble dissolved (us = 0) or reached the free surface
            us = - (y[-2][0] - y[-1][0]) / (t[-2] - t[-1])
            if r.y[0] <= profile.z_min or us <= 0.:
                stop = True
    
    # Remove any negative depths due to overshooting the free surface
    t = np.array(t)
    y = np.array(y)
    rows = y[:,0] >= 0
    t = t[rows,:]
    y = y[rows,:] 
    
    # Return the solution
    print '    Depth:  %g (m), t:  %g (s), k: %d' % \
        (y[-1,0], t[-1], k)
    return (t, y)


def derivs(t, y, profile, particle, p):
    """
    Compute the RHS of the ODE for the trajectory of a single particle
    
    Compute the right-hand-side of the governing system of ordinary 
    differential equations for the trajectory of a single particle rising
    through the water column.  
    
    Parameters
    ----------
    t : float
        Current simulation time (s)
    y : ndarray
        Model state space.  Includes the current depth (m), the masses (kg) 
        of each component of the particle, and the particle heat content
        (J)
    profile : `ambient.Profile` object
        Ambient CTD data for the model simulation
    particle : `Particle` object
        Object describing the properties and behavior of the particle.
    p : `ModelParams` object
        Object containing the model parameters
    
    Notes
    -----
    This function is called by the ODE solver `scipy.integrate.ode`.  This 
    function should not generally be called by the user.
    
    """
    # Set up the output variable
    yp = np.zeros(y.shape)
    
    # Extract the state space variables for speed and ease of reading code
    z = y[0]
    m = y[1:-1]
    T = y[-1] / (np.sum(m) * particle.cp)
    
    # Get the ambient profile data
    Ta, Sa, P = profile.get_values(z, ['temperature', 'salinity', 'pressure'])
    C = profile.get_values(z, particle.composition)
    
    # Get the particle properties
    (us, rho_p, A, Cs, beta, beta_T, T) = particle.properties(m, T, P, Sa, Ta)
    
    # Advection
    yp[0] = -us
    
    # Dissolution
    if len(Cs) > 0:
        yp[1:-1] = - A * beta[:] * (Cs[:] - C[:])
    
    # Account for heat transfer (ignore heat of solution since it is 
    # negligible in the beginning as the particle approaches equilibrium)
    yp[-1] =  - rho_p * particle.cp * A * beta_T * (T - Ta)
    
    # Account for heat lost due to decrease in mass
    yp[-1] += particle.cp * np.sum(yp[1:-1]) * T
    
    # Return the derivatives
    return yp


def sbm_ic(profile, particle, z0, de, yk, T0, K, K_T, fdis):
    """
    Set the initial conditions for a single bubble model simulation
    
    Set up the state space at the release point for the single bubble model
    simulation
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        Ambient CTD data for the model simulation
    particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Object describing the properties and behavior of the particle.
    z0 : float
        The release depth (m) of the particle in the simulation
    de : float
        Initial diameter of the particle (m)
    yk : ndarray
        Initial mole fractions of each component in the particle (--)
    T0 : float, optional
        Initial temperature (K) of the particle at release if not equal 
        to the temperature of the surrounding fluid.  If omitted, the 
        model will set T0 to the ambient temperature.
    K : float, default = 1.
        Mass transfer reduction factor (--). Pre-multiplies the mass 
        transfer coefficients providing amplification (>1) or retardation 
        (<1) of the dissolution.  
    K_T : float, default = 1.
        Heat transfer reduction factor (--). Pre-multiplies the heat 
        transfer coefficient providing amplification (>1) or retardation 
        (<1) of the heat flux.
    fdis : float, default = 0.01
        Fraction of the initial total mass (--) remaining when the 
        particle should be considered dissolved.
    
    Return
    ------
    particle : `Particle` object
        A `Particle` object containing a unified interface to the `dbm`
        module and the particle-specific model parameters (e.g., mass 
        transfer reduction factor, etc.)
    y0 : ndarray
        Model state space at the release point.  Includes the current depth 
        (m), the masses (kg) of each component of the particle, and the 
        particle heat content (J)
    
    Notes
    -----
    This function converts an initial diameter and a list of mole fractions
    to the actual mass of each component in a particle.  This seems like 
    the most common method a single particle would be initialized.  Note, 
    however, that the user does not specify the mass: it is calculated in 
    this function.  If the same diameter particle is released as a deeper 
    depth, it will contain more mass (due to compressibility).  Likewise, 
    if the composition is changed while the depth and diameter are 
    maintained constant, the mass will change, altering the trajectory
    and simulation results.  If the mass is to be kept constant, this must
    be done outside this routine and the correct diameter calculated and
    passed to this function.
    
    """
    # Make sure the mole fractions are in an array
    if not isinstance(yk, np.ndarray):
        if not isinstance(yk, list):
            yk = np.array([yk])
        else:
            yk = np.array(yk)
    
    # Get the ambient conditions
    Ta, Sa, P = profile.get_values(z0, ['temperature', 'salinity', 
                                   'pressure'])
                                   
    # Determine the temperature to pass to the integration
    if T0 is None or K_T == 0.:
        T0 = copy(Ta)
    
    # Get the initial masses and density of the particle
    if particle.issoluble:
        m0 = particle.masses_by_diameter(de, T0, P, yk)
    else:
        m0 = particle.mass_by_diameter(de, T0, P, Sa, Ta)
    
    # Initialize a Particle object
    particle = Particle(particle, m0, T0, K, K_T, fdis)
    
    # Assemble the state space
    y0 = np.hstack((z0, m0, T0 * np.sum(m0) * particle.cp))
    
    # Return the particle object and the state space
    return (particle, y0)


# ----------------------------------------------------------------------------
# Functions to post process the simulation solution
# ----------------------------------------------------------------------------

def plot_state_space(profile, particle, p, t, y, fig):
    """
    Create the basic plots to interrogate the solution for the particle path
    
    Plots the basic state space variables for a solution of the particle
    trajectory.  
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        Ambient CTD data for the model simulation
    particle : `Particle` object
        Object describing the properties and behavior of the particle.
    p : `ModelParams` object
        Collection of model parameters passed to `derivs`.
    t : ndarray
        Times (s) associated with the state space for the trajectory of the 
        particle
    y : ndarray
        State space along the trajectory of the particle.  The state space
        includes the depth (m), masses (kg) of the particle components, and 
        the particle heat content (J).  Each variable is contained in a 
        separate column of `y`.
    fig : int
        Figure number to place the first of the plots.
    
    Notes
    -----
    Creates three figure windows:
    
    1. State space variables versus time
    2. Particle diameter, shape, density, and temperature
    3. Solubility, mass transfer, and surface area
    
    """
    # Extract the state space variables
    z = y[:,0]
    m = y[:,1:-1]
    T = np.array([y[i,-1] / (np.sum(m[i,:]) * particle.cp) 
                 for i in range(len(z))])
    
    # Compute the diameter and save the ambient temperature
    rho_p = np.zeros(t.shape)
    A = np.zeros(t.shape)
    Cs = np.zeros((t.shape[0], len(particle.composition)))
    beta = np.zeros((t.shape[0], len(particle.composition)))
    Ta = np.zeros(t.shape)
    shape = np.zeros(t.shape)
    de = np.zeros(t.shape)
    us = np.zeros(t.shape)
    P = np.zeros(t.shape)
    Sa = np.zeros(t.shape)
    N = np.zeros(t.shape)
    T_fun = np.zeros(t.shape)
    for i in range(len(t)):
        Ta[i], Sa[i], P[i] = profile.get_values(z[i], ['temperature', 
                             'salinity', 'pressure'])
        N[i] = profile.buoyancy_frequency(z[i], h=0.005)
        (us[i], rho_p[i], A[i], Cs_local, beta_local, beta_T, T_fun[i]) = \
            particle.properties(m[i,:], T[i], P[i], Sa[i], Ta[i])
        if len(Cs_local) > 0:
            Cs[i,:] = Cs_local
            beta[i,:] = beta_local
        shape[i] = particle.particle.particle_shape(m[i,:], T[i], P[i], 
                   Sa[i], Ta[i])[0]
        de[i] = particle.diameter(m[i,:], T[i], P[i], Sa[i], Ta[i])
    
    # Start by plotting the raw state space versus t
    plt.figure(fig)
    plt.clf()
    
    # Depth
    ax1 = plt.subplot(221)
    plt.tight_layout()
    ax1.plot(z, t)
    ax1.set_xlabel('Depth (m)')
    ax1.set_ylabel('Time (s)')
    ax1.locator_params(tight=True, nbins=6)
    ax1.grid(True)
    
    # Slip Velocity
    ax2 = plt.subplot(222)
    plt.tight_layout()
    ax2.plot(us, t)
    ax2.set_xlabel('Slip velocity (m/s)')
    ax2.locator_params(tight=True, nbins=6)
    ax2.grid(True)
    
    # Masses
    ax3 = plt.subplot(223)
    plt.tight_layout()
    ax3.semilogx(m, t)
    ax3.set_xlabel('Component masses (kg)')
    ax3.locator_params(axis='y', tight=True, nbins=6)
    #ax3.xaxis.set_major_locator(mpl.ticker.LogLocator(base=1e2))
    ax3.grid(True)
    
    # Heat
    ax4 = plt.subplot(224)
    plt.tight_layout()
    ax4.semilogx(y[:,-1], t)
    ax4.set_xlabel('Heat (J)')
    ax4.locator_params(axis='y', tight=True, nbins=6)
    #ax4.xaxis.set_major_locator(mpl.ticker.LogLocator(base=1e2))
    ax4.grid(True)
    
    plt.show()
    
    # Plot derived variables related to diameter
    plt.figure(fig+1)
    plt.clf()
    
    # Diameter
    ax1 = plt.subplot(221)
    plt.tight_layout()
    ax1.semilogx(de * 1000, z)
    ax1.set_xlabel('Diameter (mm)')
    ax1.set_ylabel('Depth (m)')
    ax1.locator_params(axis='y', tight=True, nbins=6)
    #ax1.xaxis.set_major_locator(mpl.ticker.LogLocator(base=1e2))
    ax1.invert_yaxis()
    ax1.grid(True)
    
    # Shape
    ax2 = plt.subplot(222)
    plt.tight_layout()
    ax2.plot(shape, z)
    ax2.set_xlabel('Shape (--)')
    ax2.set_xlim((0, 4))
    ax2.invert_yaxis()
    ax2.grid(which='major', axis='x')
    ax2.locator_params(tight=True, nbins=4)
    ax2.grid(True)
    
    # Density
    ax3 = plt.subplot(223)
    plt.tight_layout()
    ax3.plot(rho_p, z)
    ax3.set_xlabel('Density (kg)')
    ax3.set_ylabel('Depth (m)')
    ax3.invert_yaxis()
    ax3.locator_params(tight=True, nbins=6)
    ax3.grid(True)
    
    # Temperature
    ax4 = plt.subplot(224)
    plt.tight_layout()
    ax4.plot(T, z)
    ax4.plot(T_fun, z)
    ax4.plot(Ta, z)
    ax4.set_xlabel('Temperature (K)')
    ax4.invert_yaxis()
    ax4.locator_params(tight=True, nbins=6)
    ax4.grid(True)
    
    plt.show()
    
    # Plot dissolution data
    plt.figure(fig+2)
    plt.clf()
    
    # Masses
    ax1 = plt.subplot(221)
    plt.tight_layout()
    ax1.semilogx(m, z)
    ax1.set_xlabel('Component masses (kg)')
    ax1.set_ylabel('Depth (m)')
    ax1.locator_params(axis='y', tight=True, nbins=6)
    #ax1.xaxis.set_major_locator(mpl.ticker.LogLocator(base=1e2))
    ax1.invert_yaxis()
    ax1.grid(True)
    
    # Solubility
    ax2 = plt.subplot(222)
    plt.tight_layout()
    ax2.plot(Cs, z)
    ax2.set_xlabel('Solubility (kg/m^3)')
    ax2.locator_params(tight=True, nbins=6)
    ax2.invert_yaxis()
    ax2.grid(True)
    
    # Mass transfer coefficient
    ax3 = plt.subplot(223)
    plt.tight_layout()
    ax3.plot(beta, z)
    ax3.set_xlabel('Mass transfer (m/s)')
    ax3.invert_yaxis()
    ax3.locator_params(tight=True, nbins=6)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax3.grid(True)
    
    # Area
    ax4 = plt.subplot(224)
    plt.tight_layout()
    ax4.semilogx(A, z)
    ax4.set_xlabel('Surface area (m^2)')
    ax4.locator_params(axis='y', tight=True, nbins=6)
    #ax4.xaxis.set_major_locator(mpl.ticker.LogLocator(base=1e2))
    ax4.invert_yaxis()
    ax4.grid(True)
    
    plt.show()
    
    # Plot dissolution data
    plt.figure(fig+3)
    plt.clf()
    
    # CTD Temperature
    ax1 = plt.subplot(221)
    plt.tight_layout()
    ax1.plot(Ta - 273.15, z)
    ax1.set_xlabel('Temperature (deg C)')
    ax1.set_ylabel('Depth (m)')
    ax1.locator_params(tight=True, nbins=6)
    ax1.invert_yaxis()
    ax1.grid(True)
    
    ax2 = plt.subplot(222)
    plt.tight_layout()
    ax2.plot(Sa, z)
    ax2.set_xlabel('Salinity (psu)')
    ax2.locator_params(tight=True, nbins=6)
    ax2.invert_yaxis()
    ax2.grid(True)
    
    ax3 = plt.subplot(223)
    plt.tight_layout()
    ax3.plot(P, z)
    ax3.set_xlabel('Pressure (Pa)')
    ax3.set_ylabel('Depth (m)')
    ax3.locator_params(tight=True, nbins=6)
    ax3.invert_yaxis()
    ax3.grid(True)
    
    ax4= plt.subplot(224)
    plt.tight_layout()
    ax4.plot(N, z)
    ax4.set_xlabel('Buoyancy Frequency (1/s)')
    ax4.locator_params(tight=True, nbins=6)
    ax4.invert_yaxis()
    ax4.grid(True)
    
    plt.show()