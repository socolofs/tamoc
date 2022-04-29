from tamoc.dbm import InsolubleParticle
from tamoc import seawater

import numpy as np

# Select the best available equations of state module
try:
    from tamoc import dbm_f
except ImportError:
    from tamoc import dbm_p as dbm_f

class EmpiricalParticle(InsolubleParticle):
    """
    Class object for a soluble fluid particle based on empirical equations
    
    This object defines the behavior of a soluble fluid particle using
    simplified equations of state for density (compressibility or constant
    density) and solubility (K_ow and activity coefficients).
    
    This class inherits the ``InsolubleParticle`` class and adds methods for
    solubility and mass transfer. As with the ``InsolubleParticle`` class,
    all physical particle properties (e.g., shape, rise velocity, etc.) use
    the same equations as the ``FluidParticle`` class. Hence, this class is
    as close to a ``FluidParticle`` as we can get, but without using the
    Peng-Robinson equation of stat.
    
    This class should be used when the computations should be more efficient,
    as it does not use any of the Peng-Robinson equation of state
    calculations while retaining all other capabilities.
    
    Parameters
    ----------
    composition : string list, length nc
        Contains the names of the chemical components in the mixture
        using the same key names as in ./data/ChemData.csv
    fp_type : integer
        Defines the fluid type (0 = gas, 1 = liquid, 2 = solid) that is
        expected to be contained in the bubble. This is needed because the
        Peng-Robinson equation of state returns values for both phases of a
        mixture. This variable allows the class to automatically return the
        values for the desired phase.
    delta : ndarray, size (nc, nc)
        Binary interaction coefficients for the Peng-Robinson equation of 
        state.  If not passed at instantiation, Python will assume a 
        full matrix of zeros.
    user_data : dict
        A dictionary of chemical property data.  If not specified, the data
        loaded from `/tamoc/data/ChemData.csv` by ``chemical_properties`` will
        be used.  To load a different properties database, use the 
        ``chemical_properties.load_data`` function to load in a custom 
        database, and pass that data to this object as `user_data`.
    delta_groups : ndarray (nc, 15)
        Provides the group contribution numbers (normalized) for each 
        component in the mixture for the 15 groups used by the Privat and
        Jaubert (2012) group contribution method for binary interaction 
        coefficients.  Default is None, in which case the values in `delta`
        will be used.
    isair : bool
        Flag indicating whether or not fluid is air.  The methods for 
        viscosity and interfacial tension below use correlations developed
        for hydocarbons.  If `isair` is False (default value), these built
        in methods are used.  If `isair` is True, then these methods are 
        replaced with correlations between air and seawater.
    sigma_correction : float
        Correction factor to adjust the interfacial tension value supplied by
        the default model to a value measured for the mixture of interest.
        The correction factor should be computed as sigma_measured / 
        sigma_model at a single P and T value.  For the FluidParticle class, 
        sigma_correction is a scalar applied to the phase defined by 
        fp_type.
    
    Notes
    -----
    The attributes are identical to those defined for a `FluidMixture`
    
    See Also
    --------
    FluidMixture, chemical_properties, InsolubleParticle
    
    """
    def __init__(self, isfluid, oil_db=None,
        sigma_correction=1, fp_type=1, **kwargs):
        
        # Store the input parameters
        self.isfluid = isfluid
        self.oil_db = oil_db
        self.sigma_correction = sigma_correction
        self.fp_type = fp_type
        self.kwargs = kwargs
        
        # Look for bulk properties in the oil_db if it exists
        if not isinstance(oil_db, type(None)):
            
            # Name of the EmpiricalParticle substance
            if 'name' in oil_db:
                self.fluid_name = oil_db['name']
            else:
                self.fluid_name = 'EmpiricalParticle'
            
            # Density parameters
            if isfluid:
                if 'rho_p' in oil_db:
                    self.rho_p = oil_db['rho_p']
                    self.gamma = None
                    self.iscompressible = False
                elif 'API_gravity' in oil_db:
                    self.rho_p = None
                    self.gamma = oil_db['API_gravity']
                    self.iscompressible = True
                elif 'gamma' in oil_db:
                    self.rho_p = None
                    self.gamma = oil_db['gamma']
                    self.iscompressible = True
                else:
                    self.rho_p = None
                    self.gamma = 30.
                    self.iscompressible = True
            else:
                if 'rho_p' in oil_db:
                    self.rho_p = oil_db['rho_p']
                    self.gamma = None
                    self.iscompressible = False
                else:
                    self.rho_p = 1600.
                    self.gamma = None
                    self.iscompressible = False
            
            # Thermal expansion coefficient
            if 'beta' in oil_db:
                self.beta = oil_db['beta']
            elif 'Ktemp' in oil_db:
                self.beta = oil_db['Ktemp']
            else:
                self.beta = 0.0007
            
            # Compression coefficient
            if 'co' in oil_db:
                self.co = oil_db['co']
            else:
                self.co = 2.90075e-9
            
            # Dynamic Viscosity and temperature parameter
            if 'dynamic_viscosity' in oil_db:
                self.mu_0 = oil_db['dynamic_viscosity']
            else:
                self.mu_0 = 0.020
            
            if 'Ctemp' in oil_db:
                self.C_t = oil_db['Ctemp']
            else:
                self.C_t = 0.
            
            # Interface Tension
            if 'sigma' in oil_db:
                self.sigma = sigma
            elif 'interface_tension' in oil_db:
                self.sigma = oil_db['Interface_tension']
            else:
                self.sigma = 0.020
        
        # Otherwise, look in keyword arguments or set default values
        else:
            
            # Pull the properties provided by the user or use default values
            if isfluid:
                if 'rho_p' in kwargs:
                    self.rho_p = kwargs['rho_p']
                    self.gamma = None
                    self.iscompressible = False
                elif 'gamma' in kwargs:
                    self.rho_p = None
                    self.gamma = kwargs['gamma']
                    self.iscompressible = True
                else:
                    self.rho_p = None
                    self.gamma = 30.
                    self.iscompressible = True
            else:
                if 'rho_p' in kwargs:
                    self.rho_p = kwargs['rho_p']
                    self.gamma = None
                    self.iscompressible = False
                else:
                    self.rho_p = 1600.
                    self.gamma = None
                    self.iscompressible = False
            if 'beta' in kwargs:
                self.beta = kwargs['beta']
            else:
                self.beta = 0.0007
            if 'co' in kwargs:
                self.co = kwargs['co']
            else:
                self.co = 2.90075e-9
            if 'mu_0' in kwargs:
                self.mu_0 = kwargs['mu_0']
            else:
                self.mu_0 = 0.020
            if 'Ctemp' in kwargs:
                self.C_t = kwargs['Ctemp']
            else:
                self.C_t = 0.
            if 'sigma' in kwargs:
                self.sigma = sigma
            else:
                self.sigma = 0.020
        
        # Pull the composition data if it exists
        if not isinstance(oil_db, type(None)):
            
            # Check if the user specified a composition
            if 'composition' in oil_db or 'simap_names' in oil_db:
            
                # Parse the component properties from the oil_db dictionary
                if 'composition' in oil_db:
                    self.comp_names = oil_db['composition']
                else:
                    self.comp_names = oil_db['simap_names']
                self.M = oil_db['molecular_weight']
                self.D = oil_db['diffusion_coefficient']
                self.Cs_0 = oil_db['solubility']
                self.log_Kow = oil_db['log_Kow']
                self.E_sol = oil_db['E_solubility']
                if 'k_bio' in oil_db:
                    self.k_bio = oil_db['k_bio']
                elif 'dgwl' in oil_db:
                    self.k_bio = oil_db['dgwl']
                elif 'dgwu' in oil_db:
                    self.k_bio = oil_db['dgwu']
                else:
                    self.k_bio = np.zeros(len(self.comp_names))
                if 't_bio' in oil_db:
                    self.t_bio = oil_db['t_bio']
                else:
                    self.t_bio = np.zeros(len(self.comp_names))
            
            # Otherwise, look in the keyword arguments
            else:
                
                # Biodegradation rate
                if 'k_bio' in oil_db:
                    self.k_bio = oil_db['k_bio']
                elif 'dgwl' in oil_db:
                    self.k_bio = oil_db['dgwl']
                elif 'dgwu' in oil_db:
                    self.k_bio = oil_db['dgwu']
                else:
                    self.k_bio = 0.
                
                # Biodegradation lag time
                if 't_bio' in oil_db:
                    self.t_bio = oil_db['t_bio']
                else:
                    self.t_bio = 0.
        
        # Look for properties in the keywork arguments
        else:
            
            # Biodegradation rate
            if 'k_bio' in kwargs:
                self.k_bio = kwargs['k_bio']
            else:
                self.k_bio = 0.
            
            # Biodegradation lag time
            if 't_bio' in kwargs:
                self.t_bio = kwargs['t_bio']
            else:
                self.t_bio = 0.
            
            # Set the other parameters to empty or zero
            self.comp_names = None
            self.M = None
            self.D = None
            self.Cs_0 = None
            self.log_Kow = None
            self.E_sol = None
        
        # Send the extracted parameters to the InsolubleParticle class
        super(EmpiricalParticle, self).__init__(self.isfluid, 
            self.iscompressible, self.rho_p, self.gamma, self.beta, self.co,
            self.k_bio, self.t_bio, self.fp_type)
        
        # Reset some of the flags that get set by the InsolubleParticle
        # __init__() method
        if not isinstance(self.comp_names, type(None)):
            self.composition = self.comp_names
        self.nc = len(self.composition)
        if not isinstance(self.Cs_0, type(None)):
            self.issoluble = True
        else:
            self.issoluble = False
        
    def density(self, m, T, P):
        """
        Density using empirical equation of state
        
        Compute the particle density either as a constant or using the
        compressibility and thermal expansion equations typical for computing
        oil density.
        
        Parameters
        ----------
        m : ndarray
            Masses of each pseudo-component in the mixture. This parameter is
            not presently used, but is included to be consistent with the
            FluidParticle API for density() and to allow future versions to
            consider oil density changes due to weathering.
        T : float
            Oil temperature (K)
        P : float
            In situ pressure (Pa)
        
        Returns
        -------
        rho : float
            Density (kg/m^3) of the present particle at the given 
            conditions
        
        Notes
        -----
        Compressibility equations are used if the ``iscompressible`` particle
        flag is ``True``. Compressibility is computed in two stages: a
        compression step due to pressure and an expansion step due to
        temperature. Each of these are governed by the particle attributes
        ``co`` for pressure and ``beta`` for temperature.
        
        """
        # Use the function for an InsolubleParticle
        return InsolubleParticle.density(self, T, P, 0., T)
        
    def viscosity(self, m, T, P):
        """
        Viscosity using empirical relations fit to data
        
        Parameters
        ----------
        m : ndarray
            Masses of each pseudo-component in the mixture. This parameter is
            not presently used, but is included to be consistent with the
            FluidParticle API for density() and to allow future versions to
            consider oil density changes due to weathering.
        T : float
            Oil temperature (K)
        P : float
            In situ pressure (Pa).  This parameter is likewise also not 
            presently used
        
        Returns
        -------
        mu : float
            Dynamic viscosity (Pa s) of the particle in seawater
        
        """
        if self.isfluid:
            # Set the standard conditions
            T_stp = 273.15 + (60.0 - 32.0) * 5.0 / 9.0
            
            # Use an exponential fit to the viscosity data
            mu = self.mu_0 * np.exp(self.C_t * (1. / T - 1. / T_stp))
        
        else:
            # Particle is solid; thus, viscosity is virtually infinite
            mu = np.inf
        
        # Return the results
        return mu
    
    def interface_tension(self, m, T, S, P):
        """
        Interfacial tension using measured values
        
        Parameters
        ----------
        m : ndarray
            Masses of each pseudo-component in the mixture
        T : float
            Oil temperature (K)
        S : float
            In situ salinity (psu) of seawater
        P : float
            In situ pressure (Pa)
        
        Returns
        -------
        sigma : float
            Interfacial tension (N m) of the present particle in seawater
            adjusted by the constant ``sigma_correction`` for dispersant 
            addition
        
        Notes
        -----
        None of the input parameters are presently used in this function.
        They are included in order to have the same API as a
        ``FluidParticle``. The present method uses the measured value
        provided at object instantiation and adjusts for dispersant by the
        constant factor ``sigma_correction``.
        
        """
        if self.isfluid:
            # Use the measured value provided by the user
            sigma = self.sigma_correction * self.sigma
        
        else:
            # Particle is solid; thus, interface tension is virtually infinite
            sigma = np.inf
        
        # Return the measured value corrected for dispersant effects
        return sigma
    
    def biodegradation_rate(self, t, lag_time=True):
        """
        Determine the present biodegradation rate constant
        
        Returns the first-order biodegradation rate constant after the 
        simulation time exceeds the bacterial community response lag time.  
        
        Parameters
        ----------
        t : float
            current simulation time (s)
        lag_time : bool, default = True
            flag indicating whether the biodegradation rates should include
            a lag time (True) or not (False).
        
        Returns
        -------
        k_bio : ndarray, size (nc)
            first-order biodegradation rate constants (1/s)
        
        Notes
        -----
        The user provides both the first-order rate constants and the 
        constant lag times in the `user_data` provided to the model or
        using the results in the `TAMOC` dataset ``./data/BioData.csv``.
        
        """
        # Make a copy of the biodegradation rate data
        k_bio = np.copy(self.k_bio)
        
        # Turn all values to zero for pseudo-components that have not yet
        # passed the lag time
        if lag_time:
            k_bio[self.t_bio > t] = 0.
        
        # Return an array of results
        return k_bio
    
    def diffusivity(self):
        """
        docstring for diffusivity
        
        """
        if self.issoluble:
            D = self.D
        else:
            D = 0.
        
        # Return a value for the diffusivity
        return D
    
    def solubility(self):
        """
        
        """
        if self.issoluble:
            Cs = self.Cs
        else:
            Cs = 0.
        
        # Return a value for the solubility
        return Cs
    
    def masses_by_diameter(self, de, T, P, mass_frac):
        """
        Find the masses of each component in a particle of given size
        
        Find the masses (kg) of each component in a particle with equivalent
        spherical diameter ``de`` and mass fractions in the mixture
        ``mass_frac``.
        
        Parameters
        ----------
        de : float
            Equivalent spherical diameter (m)
        T : float
            Particle temperature (K)
        P : float
            Particle pressure (Pa)
        mass_frac : ndarray
            Mass fractions of each component in the particle (--)
        
        Returns
        -------
        m : ndarray
            masses of each component in a fluid particle (kg)
        
        """
        # Make sure the input data are in an array
        if isinstance(mass_frac, list):
            mass_frac = np.array(mass_frac)
        
        # Compute the actual total mass of the fluid particle using the 
        # density computed from one kilograph of fluid
        m_tot = 1.0 / 6.0 * np.pi * de**3 * self.density(mass_frac, T, P)
        
        # Determine the actual mass of each component in the mixture
        m = mass_frac * m_tot
        
        # Return the masses
        return m
    
    def diameter(self, m, T, P):
        """
        Compute the diameter (m) of the present particle with masses ``m``
        
        Parameters
        ----------
        m : ndarray
            Masses of each pseudo-component in the mixture. 
        T : float
            Oil temperature (K)
        P : float
            In situ pressure (Pa).  This parameter is likewise also not 
            presently used
        
        Returns
        -------
        de : float
            Diameter (m) of the present particle
        
        """
        # Use the method for an InsolubleParticle
        return (6.0 * np.sum(m) / (np.pi * self.density(m, T, P)))**(1.0/3.0)
    
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
            mu_p : float
                dispersed phase dynamic viscosity (Pa s)
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
        de = self.diameter(m, T, P)
        rho_p = self.density(m, T, P)
        rho = seawater.density(Ta, Sa, P)
        mu = seawater.mu(Ta, Sa, P)
        mu_p = self.viscosity(m, T, P)
        sigma = self.interface_tension(m, T, Sa, P)
        
        # Compute the particle shape
        if self.isfluid:
            shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
        else:
            shape = 4
        
        return (shape, de, rho_p, rho, mu_p, mu, sigma)
    
    def mass_transfer(m, T, P, Sa, Ta, status=-1):
        """
        
        """
        # Get the diffusivities
        D = self.diffusivity(Ta, Sa, P)
        
        if not self.issoluble:
            # No composition is defined...we have no mass transfer
            beta = np.zeros(1)
        
        else:
            # Get the particle properties
            shape, de, rho_p, rho, mu_p, mu, sigma = \
                 self.particle_shape(m, T, P, Sa, Ta)
            
            # Compute the slip velocity
            us = self.slip_velocity(m, T, P, Sa, Ta, status)
            
            # Compute the appropriate mass transfer coefficients
            if shape == 1:
                beta = dbm_f.xfer_sphere(de, us, rho, mu, D, sigma, mu_p, 
                                         self.fp_type, status)
            elif shape == 2:
                beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, D, sigma, mu_p, 
                                            self.fp_type, status)
            else:
                beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, D,
                     status)
        
        # Return 
        return beta
    
    def return_all(self, m, T, P, Sa, Ta, status=-1):
        """
        Compute all of the dynamic properties of an EmpiricalParticle
        
        Computes all of the dynamic properties (e.g., slip velocity, mass
        transfer coefficients, surface area) in an efficient manner (e.g.,
        minimizing replicate calls to functions).
        
        This method repeats the calculations in the individual property
        methods and does not call the methods already defined. This is done
        so that multiple calls to functions (e.g., slip velocity) do not
        occur. As a result, for changes made in the methods above to be
        active in a simulation, they must also be made in this method.
        
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
        status : int
            flag indicating whether the particle is clean (status = 1) or
            dirty (status = -1).  Default value is -1.
        
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
        mu = seawater.mu(Ta, Sa, P)
        sigma = self.interface_tension(m, T, Sa, P)
        D = self.diffusivity()
        k = seawater.k(Ta, Sa, P) / (rho * seawater.cp())
        
        # Particle density, equivalent diameter and shape
        rho_p = self.density(m, T, P)
        de = (6.0 * np.sum(m) / (np.pi * rho_p))**(1.0/3.0)
        if self.isfluid:
            shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
        else:
            shape = 4
        
        # Other particle properties
        mu_p = self.viscosity(m, T, P)
                
        # Solubility
        Cs = self.solubility()
        K_hyd = 1.0
        
        # Shape-specific properties
        if shape == 1 or shape == 4:
            us = dbm_f.us_sphere(de, rho_p, rho, mu)
            A = np.pi * de**2
            if not self.issoluble:
                beta = np.zeros(1)
            else:
                beta = dbm_f.xfer_sphere(de, us, rho, mu, D, sigma, mu_p, 
                                         self.fp_type, status)
            beta_T = dbm_f.xfer_sphere(de, us, rho, mu, k, sigma, mu_p, 
                                       self.fp_type, status)[0]
        elif shape == 2:
            us = dbm_f.us_ellipsoid(de, rho_p, rho, mu_p, mu, sigma, status)
            A = np.pi * de**2
            if not self.issoluble:
                beta = np.zeros(1)
            else:
                beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, D, sigma, mu_p, 
                                            self.fp_type, status)
            beta_T = dbm_f.xfer_ellipsoid(de, us, rho, mu, k, sigma, mu_p, 
                                          self.fp_type, status)[0]
        else:
            us = dbm_f.us_spherical_cap(de, rho_p, rho)
            theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
            A = dbm_f.surface_area_sc(de, theta_w)
            if self.issoluble:
                beta = np.zeros(1)
            else:
                beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, D,
                                                status)
            beta_T = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, k, 
                                              status)[0]
        
        return (shape, de, rho_p, us, A, Cs, K_hyd * beta, beta_T)
        
        