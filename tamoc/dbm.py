"""
DBM Module
==========

Define objects that interface with the DBM functions in ``dbm_f``.

This module defines high-level Python class objects that wrap the individual 
functions that comprise the Discrete Bubble Model (DBM) in ``dbm_f``.  

These are particularly useful as an interface to the chemical property data
contained in ``./data/ChemData.csv`` and for pre- and post-processing of data
needed by the ``TAMOC`` simulation modules. These classes provide efficient 
data management of the chemical properties needed by the ``dbm_f`` functions 
and ensure proper behavior for the equations of state.

Notes
-----
The functions defining most equations of state and fluid particle physics are
contained in the ``dbm_f`` library.  ``dbm_f`` contains object code compiled
from the Fortran sources::

    ./src/dbm_eos.f95
    ./src/dbm_phys.f95
    ./src/math_funcs.f95

There are two additional functions defined in this module to complete the 
equations of state calculations.  These are::

    dbm.equilibrium
    dbm.gas_liq_eq

which compute the partitioning between gas and liquid of each component in 
a mixture.  For this calculation, iteration is required until the fugacities 
of each component in the mixture are equal in both gas and liquid.  Because 
for ``TAMOC`` this would generally only be done once at the start of a 
simulation to establish initial conditions and because `scipy.optimize` 
provides a nice Python interface to a fast zero-solver, these two elements of 
the discrete bubble model have not been ported to Fortran and reside in the 
`dbm` module instead of the ``dbm_f`` library.  

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

# Use these imports for deployment
from tamoc import chemical_properties as chem
from tamoc import dbm_f
from tamoc import seawater

import numpy as np
from scipy.optimize import fsolve

class FluidMixture(object):
    """
    Class object for a fluid mixture
    
    This object defines the behavior of a fluid mixture defined as a standard
    thermodynamic system.  The mixture may contain just liquid phase, a 
    gas and liquid phase together, or a pure gas phase.  The Peng-Robinson
    equation of state returns the properties of each phase in the mixture.
    If the mixture is pure liquid or pure gas, the properties of each phase
    will be the same; otherwise, the gas properties will be in the first 
    row of all two-dimensional return variables and the liquid properties in 
    the second row.
    
    Parameters
    ----------
    composition : string list, length nc
        Contains the names of the chemical components in the mixture
        using the same key names as in ./data/ChemData.csv
    delta : ndarray, size (nc, nc)
        Binary interaction coefficients for the Peng-Robinson equation of 
        state.  If not passed at instantiation, Python will assume a 
        full matrix of zeros.
    
    Attributes
    ----------
    nc : integer
        Number of chemical components in the mixture
    issoluble : logical, True
        Indicates the object contents are soluble
    M : ndarray, size (nc)
        Molecular weights (kg/mol)
    Pc : ndarray, size (nc)
        Critical pressures (Pa)
    Tc : ndarray, size (nc)
        Critical temperatures (K)
    omega : ndarray, size (nc)
        Acentric factors (--)
    kh_0 : ndarray, size (nc)
        Henry's law constants at 298.15 K and 101325 Pa (kg/(m^3 Pa))
    dH_solR : ndarray, size (nc)
        Enthalpies of solution / Ru (K)
    nu_bar : ndarray, size (nc)
        Partial molar volumes at infinite dilution (m^3/mol)
    B : ndarray, size (nc)
        White and Houghton (1966) pre-exponential factor (m^2/s)
    dE : ndarray, size (nc)
        Activation energy (J/mol)
    
    See Also
    --------
    chemical_properties, FluidParticle, InsolubleParticle
    
    Examples
    --------
    >>> air = FluidMixture(['nitrogen', 'oxygen'])
    >>> yk = np.array([0.79, 0.21])
    >>> m = air.masses(yk)
    >>> air.density(m, 273.15+10., 101325.)
    array([[ 1.24260911],
           [ 1.24260911]])
    
    """
    def __init__(self, composition, delta=None):
        super(FluidMixture, self).__init__()
        
        # Check the data type of the inputs and fix if necessary
        if not isinstance(composition, list):
            composition = [composition]
        
        if isinstance(delta, float) or isinstance(delta, list):
            delta = np.atleast_2d(delta)
        
        # Store the input variables and some of their derived properties
        self.composition = composition
        self.nc = len(composition)
        if delta is None:
            self.delta = np.zeros((self.nc, self.nc))
        else:
            self.delta = delta
        
        # Initialize the chemical composition variables
        self.M = np.zeros(self.nc)
        self.Pc = np.zeros(self.nc)
        self.Tc = np.zeros(self.nc)
        self.omega = np.zeros(self.nc)
        self.kh_0 = np.zeros(self.nc)
        self.dH_solR = np.zeros(self.nc)
        self.nu_bar = np.zeros(self.nc)
        self.B = np.zeros(self.nc)
        self.dE = np.zeros(self.nc)
        
        # Fill the chemical composition variables from the chem database
        for i in range(self.nc):
            self.M[i] = chem.data[composition[i]]['M']
            self.Pc[i] = chem.data[composition[i]]['Pc']
            self.Tc[i] = chem.data[composition[i]]['Tc']
            self.omega[i] = chem.data[composition[i]]['omega']
            self.kh_0[i] = chem.data[composition[i]]['kh_0']
            self.dH_solR[i] = chem.data[composition[i]]['dH_solR']
            if chem.data[composition[i]]['nu_bar'] < 0.:
                self.nu_bar[i] = 33.e-6
            else:
                self.nu_bar[i] = chem.data[composition[i]]['nu_bar']
            
            if chem.data[composition[i]]['B'] < 0.:
                self.B[i] = 5.0 * 1.e-2 / 100.**2.
            else:
                self.B[i] = chem.data[composition[i]]['B']
            
            if chem.data[composition[i]]['dE'] < 0.:
                self.dE[i] = 4000. / 0.238846
            else:
                self.dE[i] = chem.data[composition[i]]['dE']
        
        # Specify that adequate information is contained in the object to 
        # run the solubility methods
        self.issoluble = True
        
        # Ideal gas constant
        self.Ru = 8.314510  # (J/(kg K))
    
    def masses(self, n):
        """
        Convert the moles of each component in a mixture to their masses (kg).
        
        Parameters
        ----------
        n : ndarray, size (nc)
            moles of each component in a mixture (--)
        
        Returns
        -------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        
        """
        return n * self.M
    
    def mass_frac(self, n):
        """
        Calculate the mass fraction (--) from the number of moles of each 
        component in a mixture.
        
        Parameters
        ----------
        n : ndarray, size (nc)
            moles of each component in a mixture (--)
        
        Returns
        -------
        mf : ndarray, size (nc)
            mass fractions of each component in a mixture (--)
        
        """
        m = self.masses(n)
        return m / np.sum(m)
    
    def moles(self, m):
        """
        Convert the masses of each component in a mixture to their moles 
        (mol).
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        
        Returns
        -------
        n : ndarray, size (nc)
            moles of each component in a mixture (--)
        
        """
        return m / self.M
    
    def mol_frac(self, m):
        """
        Calcualte the mole fraction (--) from the masses of each component in 
        a mixture.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)        
        
        Returns
        -------
        yk : ndarray, size (nc)
            mole fraction of each component in a mixture (--)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        return dbm_f.mole_fraction(m, self.M)
    
    def partial_pressures(self, m, P):
        """
        Compute the partial pressure (Pa) of each component in a mixture.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        P : float
            mixture pressure (Pa)
        
        Returns
        -------
        Pk : ndarray, size (nc)
            partial pressures of each component in a mixture (Pa)
        
        """
        yk = self.mol_frac(m)
        return P * yk
    
    def density(self, m, T, P):
        """
        Compute the gas and liquid density (kg/m^3) of a fluid mixture at the 
        given state.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        
        Returns
        -------
        rho_p : ndarray, size (2,1)
            density of the gas phase (row 1) and liquid phase (row 2) of a 
            fluid mixture (kg/m^3)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        return dbm_f.density(T, P, m, self.M, self.Pc, self.Tc, self.omega, 
                             self.delta)
    
    def fugacity(self, m, T, P):
        """
        Compute the gas and liquid fugacity (Pa) of a fluid mixture at the 
        given state.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        
        Returns
        -------
        fk : ndarray, size (2, nc)
            fugacity coefficients of the gas phase (row 1) and liquid phase
            (row 2) for each component in a mixture (Pa)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        return dbm_f.fugacity(T, P, m, self.M, self.Pc, self.Tc, self.omega, 
                              self.delta)
    
    def equilibrium(self, m, T, P):
        """
        Computes the equilibrium composition of a gas/liquid mixture.
        
        Computes the equilibrium composition of a gas/liquid mixture in the 
        two-phase region of the thermodynamic state space using methods 
        described by McCain (1990) using K-factor.  
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        
        Returns
        -------
        A tuple containing:
            
            m : ndarray, size(2, nc)
                masses of each component in a mixture at equilibrium between 
                the gas and liquid phases (kg)
            xi : ndarray, size(2, nc)
                mole fractions of each component in the mixture at equilibrium 
                between the gas and liquid phases (--)
            K : ndarray, size(nc)
                partition coefficients expressed as K-factor (--)
        
        Notes
        -----
        Uses the function `dbm.equilibrium`, which performs an optimization
        using wrote iteration.
        
        """
        # Get the mole fractions and K-factors at equilibrium
        (xi, K) = equilibrium(m, T, P, self.M, self.Pc, self.Tc, 
                              self.omega, self.delta)
        
        # Get the total moles of each molecule (both phases together)
        n_tot = self.moles(m)
        
        # Get the total number of moles in gas phase using the first 
        # component in the mixture (note that this is independent of 
        # which component you pick):
        ng = (n_tot[0] - (xi[1,0] * np.sum(n_tot)))/(xi[0,0]-xi[1,0])
        
        # Get the moles of each component in gas (line 1) and liquid (line 2) 
        # phase
        n = np.zeros((2, self.nc))
        n[0,:] = xi[0,:] * ng
        n[1,:] = xi[1,:] * (np.sum(n_tot) - ng)
        
        # Finally converts to mass
        m = np.zeros((2, self.nc))
        for i in range(2):
            m[i,:] = self.masses(n[i,:])
        
        return (m, xi, K)
    
    def solubility(self, m, T, P, Sa):
        """
        Compute the solubility (kg/m^3) of each component of a mixture in both
        gas and liquid dissolving into seawater.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        T : float
            mixture temperature (K)
        P : float
            mixture pressure (Pa)
        Sa : float
            salinity of the ambient seawter (psu)
        
        Returns
        -------
        Cs : ndarray, size (2, nc)
            solubility of the gas phase (row 1) and liquid phase (row 2) for 
            each component in a mixture (kg/m^3)
        
        Notes
        -----
        It is assumed that the mixture is at the same pressure as the ambient 
        seawater and that the temperature at the interface is that of the 
        mixture.
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        # Compute the Henry's law coefficients using the temperature of the
        # seawater
        kh = dbm_f.kh_insitu(T, P, Sa, self.kh_0, self.dH_solR, self.nu_bar)
        
        # Compute the mixture fugacity using the temperature of the mixture
        f = FluidMixture.fugacity(self, m, T, P)
        
        # Get the solubility of each phase separately
        Cs = np.zeros((2,self.nc))
        Cs[0,:] = dbm_f.sw_solubility(f[0,:], kh)
        Cs[1,:] = dbm_f.sw_solubility(f[1,:], kh)
        return Cs
    
    def diffusivity(self, Ta):
        """
        Compute the diffusivity (m^2/s) of each component of a mixture into 
        seawater at the given temperature.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        Ta : float
            temperature of ambient seawater (K)
        
        Returns
        -------
        D : ndarray, size (nc)
            diffusion coefficients for each component of a mixture into 
            seawater (m^2/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        return dbm_f.diffusivity(Ta, self.B, self.dE)
    
    def hydrate_stability(self, m, P):
        """
        Compute the hydrate formation temperature at the given pressure
        
        Use the K_vsi method from Sloan and Koh (2008) to compute the hydrate
        formation/dissociation temperature at the given pressure.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a mixture (kg)
        P : float
            ambient pressure (Pa)
        
        Returns
        -------
        T_hyd : float
            critical hydrate stability temperature (K)
        
        Notes
        -----
        This method relys on the data fitted in Equation 4-1 in Sloan and Koh
        (2008), which is over a restricted range of temperature and pressure.
        In particular, when the pressure is outside (usually lower than) the 
        range of data, the model can predict spurious results.
        
        TODO (S. Socolofsky, October 1, 2013):  Get the original papers and 
        understand the limits of the range of applicability of this model.  
        Use this understanding to put bounds on the computation and ensure
        that accurate results are always returned.
        
        """
        # Get the mole fraction of the gas components in the K_vsi model in 
        # the order assumed in the model.
        gases = ['methane', 'ethane', 'propane', 'i-butane', 'n-butane', 
                 'nitrogen', 'carbon_dioxide', 'hydrogen_sulfide']
        m_gases = np.zeros(len(gases))
        for i in range(len(gases)):
            if gases[i] in self.composition:
                m_gases[i] = m[self.composition.index(gases[i])]
        
        def residual(T_hyd):
            """
            Compute the residual of sum (yi / K_vsi) - 1.
            
            Computes the residual of Equation 4-3 in Sloan and Koh for use
            in obtaining the critical formation temperature using a root-
            finding algorithm.
            
            Parameters
            ----------
            T_hyd : float
                current guess for the hydrate formation temperature (K)
            
            Returns
            -------
            res : float
                difference between sum (yi / K_vsi) - 1.
            
            """
            # Get the current value of the partition coefficients
            (K_vsi, yk) = dbm_f.kvsi_hydrate(T_hyd, P, m_gases)
            
            return np.sum(yk / K_vsi) - 1
        
        # Use root-finding to get the critical formation pressure
        T_hyd = fsolve(residual, 283.15)
        
        # Return the formation temperature
        return T_hyd


class FluidParticle(FluidMixture):
    """
    Class object for a soluble fluid particle
    
    This object defines the behavior of a soluble fluid particle.  The object
    inherits the internal variables and methods from the `FluidMixture` 
    object, but limits the output to a single phase, defined by the internal 
    variable `fp_type`.  It further extends the `FluidMixture` class to 
    include the properties inherent to particles (e.g., shape, diameter, slip 
    velocity, etc.).  
    
    Parameters
    ----------
    composition : string list, length nc
        Contains the names of the chemical components in the mixture
        using the same key names as in ./data/ChemData.csv
    fp_type : integer
        Defines the fluid type (0 = gas, 1 = liquid) that is expected to be 
        contained in the bubble.  This is needed because the Peng-Robinson
        equation of state returns values for both phases of a mixture.  This
        variable allows the class to automatically return the values for the
        desired phase.
    delta : ndarray, size (nc, nc)
        Binary interaction coefficients for the Peng-Robinson equation of 
        state.  If not passed at instantiation, Python will assume a 
        full matrix of zeros.
    
    Notes
    -----
    The attributes are identical to those defined for a `FluidMixture`
    
    See Also
    --------
    FluidMixture, chemical_properties, InsolubleParticle
    
    Examples
    --------
    >>> bub = FluidParticle(['nitrogen', 'oxygen'], fp_type=0)
    >>> yk = np.array([0.79, 0.21])
    >>> T = 273.15 + 30.
    >>> P = 10.e5
    >>> Sa = 35.
    >>> Ta = 273.15 + 20.
    >>> m = bub.masses_by_diameter(0.01, T, P, yk)
    array([  4.61873994e-06,   1.40243772e-06])
    >>> bub.density(m, T, P)
    11.499602249012074
    >>> bub.slip_velocity(m, T, P, Sa, Ta)
    0.22197023589052
    
    """
    def __init__(self, composition, fp_type=0., delta=None):
        super(FluidParticle, self).__init__(composition, delta)
        
        # Store the input variables
        self.fp_type = fp_type
    
    def density(self, m, T, P):
        """
        Compute the particle density (kg/m^3) of the fluid in the phase given 
        by `fp_type`.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in the particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        
        Returns
        -------
        rho_p : float
            density of the fluid particle (kg/m^3)
        
        Notes
        -----
        Uses the density method in the `FluidMixture` object, but only returns
        the value for the phase given by `fp_type`.
        
        """
        return FluidMixture.density(self, m, T, P)[self.fp_type, 0]
    
    def fugacity(self, m, T, P):
        """
        Compute the particle fugacities (Pa) of the fluid in the phase given 
        by `fp_type`.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        
        Returns
        -------
        fk : ndarray, size (nc)
            fugacities of each component of the fluid particle (Pa)
        
        Notes
        -----
        Uses the fugacity method in the `FluidMixture` object, but only 
        returns the values for the phase given by `fp_type`.
        
        """
        return FluidMixture.fugacity(self, m, T, P)[self.fp_type, :]
    
    def solubility(self, m, T, P, Sa):
        """
        Compute the solubility (kg/m^3) of each component of a particle into 
        seawater for the phase given by `fp_type`.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        S : float
            salinity of the ambient seawter (psu)
        
        Returns
        -------
        Cs : ndarray, size (nc)
            solubilities of each component of the fluid particle into 
            seawater (kg/m^3)
        
        Notes
        -----
        It is assumed that the mixture is at the same pressure as the ambient 
        seawater and that the temperature at the interface is that of the 
        particle.
        
        Uses the solubility method in the `FluidMixture` object, but only 
        returns the values for the phase given by `fp_type`.
        
        """
        return FluidMixture.solubility(self, m, T, P, Sa)[self.fp_type, :]
    
    def masses_by_diameter(self, de, T, P, yk):
        """
        Find the masses (kg) of each component in a particle with equivalent 
        spherical diameter `de` and mole fractions `yk`.
        
        Parameters
        ----------
        de : float
            equivalent spherical diameter (m)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        yk : ndarray, size (nc)
            mole fractions of each component in the particle (--)
        
        Returns
        -------
        m : ndarray, size (nc)
            masses of each component in a fluid particle (kg)
        
        """
        if isinstance(yk, list):
            yk = np.array(yk)
        
        # Get the masses for one mole of fluid
        m = self.masses(yk)
        
        # Compute the actual total mass of the fluid particle using the 
        # density computed from one mole of fluid
        m_tot = 1.0 / 6.0 * np.pi * de**3 * self.density(m, T, P)
        
        # Determine the number of moles in the fluid particle
        n = yk * m_tot / np.sum(m)
        
        return self.masses(n)
    
    def diameter(self, m, T, P):
        """
        Compute the equivalent spherical diameter (m) of a single fluid 
        particle.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
        T : float
            particle temperature (K)
        P : float
            particle pressure (Pa)
        
        Returns
        -------
        de : float
            equivalent spherical diameter of a fluid particle (m)
        
        """
        return (6.0 * np.sum(m) / (np.pi * self.density(m, T, P)))**(1.0/3.0)
    
    def particle_shape(self, m, T, P, Sa, Ta):
        """
        Determine the shape of a fluid particle from the properties of the 
        particle and surrounding fluid.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
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
            
            shape : integer
                1 - sphere, 2 - ellipsoid, 3 - spherical cap
            de : float
                equivalent spherical diameter (m)
            rho_p : float
                particle density (kg/m^3)
            rho : float
                ambient seawater density (kg/m^3)
            mu : float
                ambient seawater dynamic viscosity (Pa s)
            sigma : float
                interfacial tension (N/m)
        
        Notes
        -----
        As for the solubility calculation, we use the particle temperature to 
        calculate properties at the interface (e.g., to calculate the 
        interfacial tension) and the ambient temperature for properties of 
        the bulk continuous phase (e.g., density and viscosity).
        
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Compute the fluid particle and ambient properties
        de = self.diameter(m, T, P)
        rho_p = self.density(m, T, P)
        rho = seawater.density(Ta, Sa, P)
        mu = seawater.mu(Ta)
        sigma = seawater.sigma(T)
        
        shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
        
        return (shape, de, rho_p, rho, mu, sigma)
    
    def slip_velocity(self, m, T, P, Sa, Ta):
        """
        Compute the slip velocity (m/s) of a fluid particle.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
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
        us : float
            slip velocity of the fluid particle (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        if shape == 1:
            us = dbm_f.us_sphere(de, rho_p, rho, mu)
        elif shape == 2:
            us = dbm_f.us_ellipsoid(de, rho_p, rho, mu, sigma)
        else:
            us = dbm_f.us_spherical_cap(de, rho_p, rho)
        
        return us
    
    def surface_area(self, m, T, P, Sa, Ta):
        """
        Compute the surface area (m^2) of a fluid particle.
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
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
        A : float
            surface area of the fluid particle (m^2)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        if shape == 3:
            # Compute the surface area of a spherical cap bubble
            us = self.slip_velocity(m, T, P, Sa, Ta)
            theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
            A = dbm_f.surface_area_sc(de, theta_w)
        else:
            # Compute the area of the equivalent sphere:
            A = np.pi * de**2
        
        return A
    
    def mass_transfer(self, m, T, P, Sa, Ta):
        """
        Compute the mass transfer coefficients (m/s) for each component in a 
        fluid particle
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
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
        beta : ndarray, size (nc)
            mass transfer coefficient for each component in a fluid particle
            (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.  This method
        checks for hydrate stability and returns a reduced mass transfer 
        coefficient when hydrate shells are predicted to be present.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        # Compute the slip velocity
        us = self.slip_velocity(m, T, P, Sa, Ta)
        
        # Compute the appropriate mass transfer coefficients
        if shape == 1:
            beta = dbm_f.xfer_sphere(de, us, rho, mu, self.diffusivity(Ta))
        elif shape == 2:
            beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, self.diffusivity(Ta))
        else:
            beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, 
                                            self.diffusivity(Ta))
        return beta
    
    def heat_transfer(self, m, T, P, Sa, Ta):
        """
        Compute the heat transfer coefficient (m/s) for a fluid particle
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
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
        beta_T : float
            heat transfer coefficient for a fluid particle (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        # Get the thermal conductivity of seawater
        k = seawater.k()
        
        # Compute the slip velocity
        us = self.slip_velocity(m, T, P, Sa, Ta)
        
        # Compute the appropriate heat transfer coefficients.  Assume the 
        # heat transfer has the same form as the mass transfer with the 
        # diffusivity replaced by the thermal conductivity
        if shape == 1:
            beta = dbm_f.xfer_sphere(de, us, rho, mu, k)
        elif shape == 2:
            beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, k)
        else:
            beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, k)
        
        return beta
    
    def return_all(self, m, T, P, Sa, Ta):
        """
        Compute all of the dynamic properties of the bubble in an efficient
        manner (e.g., minimizing replicate calls to functions).
        
        This method repeats the calculations in the individual property 
        methods, and does not call the methods already defined.  This is done
        so that multiple calls to functions (e.g., slip velocity) do not 
        occur.  
        
        Parameters
        ----------
        m : ndarray, size (nc)
            masses of each component in a particle (kg)
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
            
            shape : integer
                1 - sphere, 2 - ellipsoid, 3 - spherical cap
            de : float
                equivalent spherical diameter (m)
            rho_p : float
                particle density (kg/m^3)
            us : float
                slip velocity (m/s)
            A : float 
                surface area (m^2)
            Cs : ndarray, size (nc)
                solubility (kg/m^3)
            beta : ndarray, size (nc)
                mass transfer coefficient (m/s)
            beta_T : float
                heat transfer coefficient (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_eos.f95``.  This method
        checks for hydrate stability and returns a reduced mass transfer 
        coefficient when hydrate shells are predicted to be present.
        
        """
        # Ambient properties of seawater
        rho = seawater.density(Ta, Sa, P)
        mu = seawater.mu(Ta)
        sigma = seawater.sigma(T)
        D = dbm_f.diffusivity(Ta, self.B, self.dE)
        k = seawater.k()
        
        # Particle density, equivalent diameter and shape
        rho_p = dbm_f.density(T, P, m, self.M, self.Pc, self.Tc, self.omega, 
                             self.delta)[self.fp_type, 0]
        de = (6.0 * np.sum(m) / (np.pi * rho_p))**(1.0/3.0)
        shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
        
        # Solubility
        f = dbm_f.fugacity(T, P, m, self.M, self.Pc, self.Tc, self.omega, 
                           self.delta)
        kh = dbm_f.kh_insitu(T, P, Sa, self.kh_0, self.dH_solR, self.nu_bar)
        Cs = dbm_f.sw_solubility(f[self.fp_type,:], kh)
        
        # Check hydrate stability
        K_hyd = 1.
        T_hyd = self.hydrate_stability(m, P)
        if T < T_hyd:
            K_hyd = 1.0
        
        # Shape-specific properties
        if shape == 1:
            us = dbm_f.us_sphere(de, rho_p, rho, mu)
            A = np.pi * de**2
            beta = dbm_f.xfer_sphere(de, us, rho, mu, D)
            beta_T = dbm_f.xfer_sphere(de, us, rho, mu, k)[0]
        elif shape == 2:
            us = dbm_f.us_ellipsoid(de, rho_p, rho, mu, sigma)
            A = np.pi * de**2
            beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, D)
            beta_T = dbm_f.xfer_ellipsoid(de, us, rho, mu, k)[0]
        else:
            us = dbm_f.us_spherical_cap(de, rho_p, rho)
            theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
            A = dbm_f.surface_area_sc(de, theta_w)
            beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, D)
            beta_T = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, k)[0]
        
        return (shape, de, rho_p, us, A, Cs, K_hyd * beta, beta_T)
    

class InsolubleParticle(object):
    """
    Class object for an insoluble (inert) fluid particle
    
    This object defines the behavior of an inert fluid particle.  The purpose
    of this class is to simulate particle that cannot be described by the 
    Peng-Robinson equation of state, such as sand, or that do not require
    such computational expense, such as dead oil.
    
    Parameters
    ----------
    isfluid : logical
        `True` or `False`; states whether or not the inert particle could have 
        a mobile interface.  For example, choose `True` for oil and `False` 
        for sand.
    iscompressible : logical
        `True` or `False`; selects the equation of state for density.  `True` 
        uses the API gravity, isothermal compression and isobaric thermal 
        expansion; whereas, `False` returns the constant density specified at 
        instantiation.
    rho_p : float
        particle density (default value is 930 kg/m^3)
    gamma : float
        API gravity (default value is 30 deg API)
    beta : float
        thermal expansion coefficient (default value is 0.0007 K^(-1))
    co : float
        isothermal compressibility coefficient (default value is 
        2.90075e-9 Pa^(-1))
    
    Attributes
    ----------
    composition : string list
        Set equal to ['inert']
    nc : integer
        Number of components, set equal to 1
    issoluble : logical, False
        Indicates the particle is not soluble
    
    See Also
    --------
    FluidMixture, chemical_properties, FluidParticle
    
    Examples
    --------
    >>> oil = InsolubleParticle(True, True)
    >>> T = 273.15 + 30.
    >>> P = 10.e5
    >>> Sa = 35.
    >>> Ta = 273.15 + 20.
    >>> m = oil.mass_by_diameter(0.01, T, P, Sa, Ta)
    0.00045487710681354078
    >>> oil.density(T, P, Sa, Ta)
    868.75128058458085
    >>> oil.slip_velocity(m, T, P, Sa, Ta)
    0.14332887200025926
    
    """
    def __init__(self, isfluid, iscompressible, rho_p=930., gamma=30., 
                 beta=0.0007, co=2.90075e-9):
        super(InsolubleParticle, self).__init__()
        
        # Store the input variables
        self.isfluid = isfluid
        self.iscompressible = iscompressible
        self.rho_p = rho_p
        self.gamma = gamma
        self.beta = beta
        self.co = co
        
        # Specify that the particle is not soluble and is therefore treated
        # like a single substance
        self.issoluble = False
        self.nc = 1
        self.composition = ['inert']
    
    def density(self, T, P, Sa, Ta):
        """
        Compute the density (kg/m^3) of an inert fluid particle.
        
        If the particle is compressible, this method computes the particle 
        density following McCain (1990), using the API gravity (ee pages 224 
        and following in McCain).  This would be typical of an oil.
        
        Otherwise, the method returns the constant density stored in the 
        internal variable `rho_p`.
        
        Parameters
        ----------
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
        rho_p : float
            density of the inert particle (kg/m^3)
        
        """
        if self.iscompressible:
            
            # Set the standard conditions for the API gravity
            P_stp = 101325.0
            T_stp = 273.15 + (60.0 - 32.0) * 5.0 / 9.0
            
            # Get the density of water at standard conditions
            rho_stp = seawater.density(T_stp, 0., P_stp)
            
            # Use the API gravity equation (8-2) in McCain (1990) to get the
            # fluid particle density at standard conditions
            gamma_0 = 141.5 / (self.gamma + 131.5)
            rho_p = gamma_0 * rho_stp
            
            # Isothermal compression to in-situ pressure using equation (8-19) 
            # in McCain (1990).
            rho_p = rho_p * np.exp(self.co * (P - P_stp))
            
            # Isobaric compression to oil temperature using equation (8-28) in
            # McCain (1990).
            rho_p = rho_p * (1 - self.beta * (T - T_stp))
        
        else:
            rho_p = self.rho_p
        
        return rho_p
    
    def mass_by_diameter(self, de, T, P, Sa, Ta):
        """
        Compute the mass (kg) of an inert fluid particle with equivalent 
        spherical diameter `de`.
        
        Parameters
        ----------
        de : float
            equivalent spherical diameter (m)
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
        m : float
            mass of the fluid particle (kg)
        
        """
        return 1.0 / 6.0 * np.pi * de**3 * self.density(T, P, Sa, Ta)
    
    def diameter(self, m, T, P, Sa, Ta):
        """
        Compute the diameter (m) of an inert fluid particle of mass `m`.
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (m)
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
        de : diameter of the fluid particle (m)
        
        """
        return (6.0 * m / (np.pi * self.density(T, P, Sa, Ta)))**(1.0/3.0)
    
    def particle_shape(self, m, T, P, Sa, Ta):
        """
        Determine the shape of an inert fluid particle from the properties of 
        the particle and surrounding fluid.
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (m)
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
        
            shape : integer
                1 - sphere, 2 - ellipsoid, 3 - spherical cap, 4 - rigid
            de : float
                equivalent spherical diameter (m)
            rho_p : float
                particle density (kg/m^3)
            rho : float
                ambient seawater density (kg/m^3)
            mu : float
                ambient seawater dynamic viscosity (Pa s)
            sigma : float
                interfacial tension (N/m)
        
        Notes
        -----
        We use the particle temperature to calculate properties at the 
        interface (e.g., to calculate the interfacial tension) and the 
        ambient temperature for properties of the bulk continuous phase 
        (e.g., density and viscosity).
        
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Compute the fluid particle properties
        de = self.diameter(m, T, P, Sa, Ta)
        rho_p = self.density(T, P, Sa, Ta)
        rho = seawater.density(Ta, Sa, P)
        mu = seawater.mu(Ta)
        sigma = seawater.sigma(T)
        
        # Compute the particle shape
        if self.isfluid:
            shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
        else:
            shape = 4
        
        return (shape, de, rho_p, rho, mu, sigma)
    
    def slip_velocity(self, m, T, P, Sa, Ta):
        """
        Compute the slip velocity (m/s) of an inert fluid particle.
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (kg)
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
        us : float
            slip velocity of the inert particle (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        if shape == 1 or shape == 4:
            us = dbm_f.us_sphere(de, rho_p, rho, mu)
        elif shape == 2:
            us = dbm_f.us_ellipsoid(de, rho_p, rho, mu, sigma)
        else:
            us = dbm_f.us_spherical_cap(de, rho_p, rho)
        
        return us
    
    def surface_area(self, m, T, P, Sa, Ta):
        """
        Compute the surface area (m^2) of an inert fluid particle.
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (kg)
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
        A : float
            surface area of the inert particle (m^2)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Get the particle properties
        shape, de, rho_p, rho, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        if shape == 3:
            # Compute the surface area of a spherical cap bubble
            us = self.slip_velocity(m, T, P, Sa, Ta)
            theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
            A = dbm_f.surface_area_sc(de, theta_w)
        else:
            # Compute the area of the equivalent sphere:
            A = np.pi * de**2
        
        return A
    
    def heat_transfer(self, m, T, P, Sa, Ta):
        """
        Compute the heat transfer coefficients (m/s) for an inert fluid 
        particle.
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (kg)
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
        beta_T : float
            heat transfer coefficient for the inert particle (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
       """
        # Get the particle properties
        shape, de, rho_p, rho, mu, sigma = \
             self.particle_shape(m, T, P, Sa, Ta)
        
        # Get the thermal conductivity of seawater
        k = seawater.k()
        
        # Compute the slip velocity
        us = self.slip_velocity(m, T, P, Sa, Ta)
        
        # Compute the appropriate heat transfer coefficients.  Assume the 
        # heat transfer has the same form as the mass transfer with the 
        # diffusivity replaced by the thermal conductivity
        if shape == 1 or shape == 4:
            beta = dbm_f.xfer_sphere(de, us, rho, mu, k)
        elif shape == 2:
            beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, k)
        else:
            beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, k)
        
        return beta
    
    def return_all(self, m, T, P, Sa, Ta):
        """
        Compute all of the dynamic properties of an inert fluid particle in 
        an efficient manner (e.g., minimizing replicate calls to functions).
        
        This method repeats the calculations in the individual property 
        methods, and does not call the methods already defined.  This is done
        so that multiple calls to functions (e.g., slip velocity) do not 
        occur.  
        
        Parameters
        ----------
        m : float
            mass of the inert fluid particle (kg)
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
            shape : integer
                1 - sphere, 2 - ellipsoid, 3 - spherical cap, 4 - rigid
            de : float
                equivalent spherical diameter (m)
            rho_p : float
                particle density (kg/m^3)
            us : float
                slip velocity (m/s)
            A : float
                surface area (m^2)
            beta_T : float
                heat transfer coefficient (m/s)
        
        Notes
        -----
        Uses the Fortran subroutines in ``./src/dbm_phys.f95``.
        
        """
        # Ambient properties of seawater
        rho = seawater.density(Ta, Sa, P)
        mu = seawater.mu(Ta)
        sigma = seawater.sigma(T)
        k = seawater.k()
        
        # Particle density, equivalent diameter and shape
        rho_p = self.density(T, P, Sa, Ta)
        de = (6.0 * m / (np.pi * rho_p))**(1.0/3.0)
        if self.isfluid:
            shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
        else:
            shape = 4
        
        # Shape-specific properties
        if shape == 1 or shape == 4:
            us = dbm_f.us_sphere(de, rho_p, rho, mu)
            A = np.pi * de**2
            beta_T = dbm_f.xfer_sphere(de, us, rho, mu, k)[0]
        elif shape == 2:
            us = dbm_f.us_ellipsoid(de, rho_p, rho, mu, sigma)
            A = np.pi * de**2
            beta_T = dbm_f.xfer_ellipsoid(de, us, rho, mu, k)[0]
        else:
            us = dbm_f.us_spherical_cap(de, rho_p, rho)
            theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
            A = dbm_f.surface_area_sc(de, theta_w)
            beta_T = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, k)[0]
        
        return (shape, de, rho_p, us, A, beta_T)
    

# ----------------------------------------------------------------------------
# Functions used by classes to compute gas/liquid equilibrium of a mixture
# ----------------------------------------------------------------------------

def equilibrium(m_0, T, P, M, Pc, Tc, omega, delta):
    """
    Compute the equilibrium composition of a mixture using the P-R EOS
    
    Computes the mole fraction composition for the gas and liquid phases of a
    mixture using the Peng-Robinson equation of state and the methodology
    described in McCain (1990), Properties of Petroleum Fluids, 2nd Edition,
    PennWell Publishing Company, Tulsa, Oklahoma.
    
    Parameters
    ----------
    T : float
        temperature (K)
    P : float
        pressure (Pa)
    m_0 : ndarray, size (nc)
        masses of each component present in the whole mixture (gas plus 
        liquid, kg)
    M : ndarray, size (nc)
        Molecular weights (kg/mol)
    Pc : ndarray, size (nc)
        Critical pressures (Pa)
    Tc : ndarray, size (nc)
        Critical temperatures (K)
    omega : ndarray, size (nc)
        Acentric factors (--)
    delta : ndarray, size (nc, nc)
        Binary interaction coefficients for the Peng-Robinson equation of 
        state.  If not passed at instantiation, Python will assume a 
        full matrix of zeros.
    
    Returns
    -------
    xi : ndarray, size(2, nc)
        Mole fraction of each component in the mixture.  Row 1 gives the
        values for the gas phase and Row 2 gives the values for the liquid 
        phase (--)
    
    Notes
    -----
    The method estimates the K-factors giving the mole fractions in gas and
    liquid and optimizes the K-factor estimates until they converge on the
    ratio of the actual fugacity coefficients (phi_liq / phi_gas).
    Convergence uses a squared relative error as in McCain (1990).
    
    """
    # Compute the residual of the K-factor optimization
    def find_K(K):
        """
        Evaluate the update function for finding K-factor
        
        Evaluates the new guess for K-factor following McCain (1990) p. 426, 
        equation (15-23) as explained on p. 430 in connection with equation
        (15-31).
        
        Parameters
        ----------
        T, P, m_0, M, Pc, Tc, omega, delta = constant and inherited
            from above
        K : ndarray
            The current guess for the K-factor (--)
        
        Returns
        -------
        K_new : ndarray
            New guess for K-factor
        
        """
        # Get the mixture composition for the current K-factor
        xi = gas_liq_eq(m_0, M, K)
        
        # Get tha gas and liquid fugacities for the current composition
        f_gas = dbm_f.fugacity(T, P, xi[0,:]*M, M, Pc, Tc, omega, delta)[0,:]
        f_liq = dbm_f.fugacity(T, P, xi[1,:]*M, M, Pc, Tc, omega, delta)[1,:]
        
        # Update K using K = (phi_liq / phi_gas)
        
        K_new = (f_liq / (xi[1,:] * P)) / (f_gas / (xi[0,:] * P))
        
        # Calculate the cost function
        return K_new
            
    # Get an initial guess for the K-factors using equation B-61 on p. 525
    K_0 = np.exp(5.37 * (1. + omega) * (1 - Tc / T)) / (P / Pc)
    
    # Find the optimal values of K-factor following algorithm on p. 430
    tol = 1.49012e-8
    eps = 1.0
    while eps > tol:
        K = K_0[:]
        K_0 = find_K(K)
        eps = np.sum((K - K_0)**2 / (K * K_0))
    
    # Return the optimized mixture composition
    return (gas_liq_eq(m_0, M, K), K)

def gas_liq_eq(m, M, K):
    """
    Compute the gas and liquid partitioning from K-factor
    
    Compute the gas and liquid mole fractions of a mixture based on the 
    K-factor.
    
    Parameters
    ----------
    m : ndarray, size (nc)
        masses of each component present in the whole mixture (gas plus 
        liquid, kg)
    M : ndarray, size (nc)
        Molecular weights (kg/mol)
    K : ndarray, size (nc)
        K-factor for partitioning between liquid and gas (--)
    
    Returns
    -------
    xi : ndarray, size(2, nc)
        Mole fraction of each component in the mixture.  Row 1 gives the
        values for the gas phase and Row 2 gives the values for the liquid 
        phase (--)
    
    Notes
    -----
    The solution is based on equations (12-17) and (12-18) in McCain (1990) 
    p. 355.
    
    These equations require iteration until the correct ratio of moles of gas 
    to moles of liquid is obtained.  This iteration is performed by 
    `scipy.optimize.fzero`, and computes the excess liquid content
    for a given guess of the gas fraction until converged to zero.
    
    """
    # Compute the mole fraction of the total mixture
    moles = m / M
    zj = moles / np.sum(moles)
    
    # Compute the residual of the liquid composition
    def residual(ng):
        """
        Evaluate the liquid composition for the current guess of gas fraction
        
        The equation to find the composition of a mixture using K-factor
        requires iteration until the correct ratio of moles of gas to moles of
        liquid is obtained.  This function computes the excess liquid content
        for a given guess of the gas fraction.
        
        Input variables are:
            m, M, K, moles  = constant and inherited from above
            ng = current guess of moles gas fraction in the mixture (--)
        
        """
        # Compute the composition and return the residual error
        return (1. - np.sum(zj / (1. + ng * (K - 1.))))
    
    # Find the gas and liquid fraction for the mixture
    ng = fsolve(residual, 0.5)
    nl = 1. - ng
    
    # Return the mixture composition
    return np.array([zj / (1. + nl * (1./ K - 1.)), 
                     zj / (1. + ng * (K - 1.))])

