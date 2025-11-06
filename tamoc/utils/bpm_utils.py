"""
bpm_utils.py
------------

Tools to aid in setting up and running bent plume model simulations

"""
# S. Socolofsky, Texas A&M University, August 2025, <socolofs@tamu.edu>

# Compatibility definitions
unicode = type(u' ')

from tamoc.utils import eos_utils
from tamoc import ambient, dbm, dispersed_phases, bent_plume_model
from tamoc import particle_size_models, dbm_utilities

import numpy as np

class BPM_Sim(object):
    """
    Class for handling bent plume model simulations
    
    Parameters 
    ----------
    **kwargs
        Keyword arguments:
    
        substance : various
            Data describing the spilled substance.  See Notes below for 
            details.
        water : various
            Data describing the ambient water temperature and salinity profile.
            See Notes below for details.
        current : various
            Data describing the ambient current velocity profile.  See Notes
            below for details.
        K : float
            The dissolution mass transfer reduction factor, --
        K_T : float
            The head transfer reduction factor, --
        f_dis : float
            A parameter specifying the mass fraction remaining of the 
            original mass of a given chemical component when that component
            should be considered to be dissolved
        t_hyd : float
            The surfactant or hydrate transition time (s) from clean bubble
            to dirty bubble mass transfer rates
        lag_time : bool
            Flag indicating whether the lag time should be included before
            starting first-order biodegradation of each chemical component
            of the fluid mixture.
        single_phase_particles : bool
            Flag indicating whether the `FluidParticle` objects should be 
            computed as single-phase or whether they may become mixtures of
            gas and liquid within a single fluid particle.  The model is 
            faster for single-phase particles than mixed-phase.  Gas-phase
            petroleum at a deep release may form some liquid-phase attached
            matter before surfacing in some simulations.  Set this parameter
            to `False` to allow mixed-phase particle formation.
        X0 : ndarray
            Array of coordinates for the release point in east, north, and
            depth, (m)
        D : float
            Diameter of the release orifice, m
        phi_0 : float
            Vertical angle of the release (rad) from horizontal with z
            positive down.  A vertical, upward release should have `phi_0`
            set to `-np.pi/2.`
        theta_0 : float
            Horizontal angle of the release (rad) from east.
        Qp : float
            Volume flow rate (m^3/s) of produced water in the release
        Sp : flaot
            Salinity of produced water in the release, psu
        Tj : float
            Temperature of the released fluids, K
        cj : float
            Concentration of passive tracers in the release fluid, kg/m^3
        tracers : str list
            Name for the tracer at the release.
        track : bool
            Flag indicating whether to track the fluid particles above the
            intrusion point
        dt_max  : float
            Maximum step size to allow in the plume simulation, s
        release_rate_type : str
            String stating whether the mass flow rate is specified 
            (`mass_flowrate`) or the volume flow rate (`volume_flowrate`)
        m0 : float
            Value for the total mass flow rate (kg/s) of the release.  Set to
            `None` if volume flow rate is specified instead
        q0 : float
            Value for the total volume flow rate (std bbl/d) or the release.
            Set to `None` if mass flow rate is specified instead
        gor : float
            Value for the gas-to-oil ratio (std ft^3/bbl) if 
        gas_model : str
            String indicating which model to use to compute the gas bubble 
            sizes.  Set of `None` if the gas bubble size is provided as 
            input.  Set to `wang_etal` otherwise.
        pdf_gas : str
            String stating what size distribution to use if gas bubble sizes
            are computed by the model.  Options are 'lognormal' or 
            'rosin-rammler'
        n_gas : int
            Number of gas bubble sizes for the distribution
        de_gas : ndarray
            Equivalent spherical diameter for each bubble size in the gas-
            phase bubble size distribution, m
        vf_gas : ndarray
            Volume fraction for each bubble size in the gas-phase bubble
            size distribution, --
        liq_model : str
            String indicating which model to use to compute the liquid 
            droplet sizes.  Set of `None` if the liquid drolet size is 
            provided as input.  Set to `li_etal` or 'sintef' otherwise.
        pdf_liq : str
            String stating what size distribution to use if liquid droplet 
            sizes are computed by the model.  Options are 'lognormal' or 
            'rosin-rammler'
        n_liq : int
            Number of liquid droplet sizes for the distribution
        de_liq : ndarray
            Equivalent spherical diameter for each liquid droplet in the 
            liquid-phase droplet size distribution, m
        vf_liq : ndarray
            Volume fraction for each droplet size in the liquid-phase droplet
            size distribution, --

    Notes
    -----
    The spilled substance can either be taken from the NOAA OilLibrary or
    can be created from individual pseudo-components in TAMOC.  The user may
    define the `substance` in one of two ways:

    substance : str
        Provide a unique OilLibrary ID number from the NOAA Python
        OilLibrary package
    substance : dict
        Use the chemical properties database provided with TAMOC.  In this
        case, use the dictionary keyword `composition` to pass a list
        of chemical property names and the keyword `masses` to pass a
        list of mass fractions for each component in the composition
        list.  If the masses variable does not sum to unity, this function
        will compute an equivalent mass fraction that does.

    Likewise, the ambient water column data can be provided through several
    different options.  The `water` variable contains temperature and salinity
    data.  The user may define the `water` in the following ways:

    water : None
        Indicates that we have no information about the ambient temperature
        or salinity.  In this case, the model will import data for the
        world-ocean average.
    water : dict
        If we only know the water temperature and salinity at the surface,
        this may be passed through a dictionary with keywords `temperature`
        and `salinity`.  In this case, the model will import data for the
        world-ocean average and adjust the data to have the given temperature
        and salinity at the surface.
    water : netCDF4.Dataset
        If a 'netCDF4.Dataset' object already contains the ambient CTD
        data in a format appropriate for the `ambient.Profile` object, then
        this can be passed.  In this case, it is assumed that the dataset
        includes the currents; hence, the `currents` variable will be
        ignored.
    water : ambient.Profile
        If we already created our own ambient Profile object, then this
        object can be used directly.
    water : str
        If we stored the water column profile in a file, we may provide the
        file path to this file via the string stored in water. If this string
        ends in '.nc', it is assumed that this file contains a netCDF4
        dataset. Otherwise, this file should contain columns in the following
        order: depth (m), temperature (deg C), salinity (psu), velocity in
        the x-direction (m/s), velocity in the y-direction (m/s). Since this
        option includes the currents, the current variable will be ignored in
        this case. A comment string of `#` may be used in the text file.
    water : ndarray
        The profile data can be passed as an `ndarray` of data organized
        as columns containing depth, temperature, salinity, pressure, and
        any remaining chemical properties, optionally including ambient
        currents
    
    Finally, current profile data can be provided through several different
    options.  The user may define the `current` in the following ways:

    current : float
        This is assumed to be the current velocity along the x-axis and will
        be uniform over the depth
    current : ndarray
        This is assumed to contain the current velocity in the x- and y- (and
        optionally also z-) directions. If this is a one-dimensional array,
        then these currents will be assumed to be uniform over the depth. If
        this is a multi-dimensional array, then these values as assumed to
        contain a profile of data, with the depth (m) as the first column of
        data.
    current : None
        Set `currents` to `None` is the `water` data already include the
        current information.
    

    
    """
    def __init__(self, **kwargs):
        
        # Create object
        super(BPM_Sim, self).__init__()
        
        # Set some default values
        defaults = {
            'z0' : 100,
            'd0' : 0.1,
            'substance' : {
                'composition' : ['methane', 'ethane', 'propane',
                               'toluene', 'benzene'],
                'masses' : np.array([0.2, 0.03, 0.02, 0.25, 0.5])
            },
            'q_oil' : 20000.,
            'gor' : 0.,
            'x0' : 0.,
            'y0' : 0.,
            'u0' : None,
            'phi_0' : -np.pi / 2.,
            'theta_0' : 0.,
            'num_gas_elements' : 10,
            'num_oil_elements' : 25,
            'water' : None,
            'current' : np.array([0.1, 0., 0.]),
            'ca' : 'all',
            'size_distribution' : None
        }
        
        # Extract all default values but overwrite with values in kwargs
        # if they were passed
        for key, default_value in defaults.items():
            setattr(self, key, kwargs.get(key, default_value))
        
        # Extract remaining kwargs not in default values
        for key in kwargs:
            if key not in defaults:
                setattr(self, key, kwargs[key])
        
        # Prepare to update the model parameters
        self.update_profile = True
        if hasattr(self, 'dbm_mixture'):
            # The user provided a FluidMixture...check for mass_frac
            if not hasattr(self, 'mass_frac'):
                mssg = 'tamoc.utils.BPM_Sim:  an ambient.FluidMixture '+ \
                'object was supplied without a corresponding array of ' + \
                'mass fractions.  Please provide mass fractions.'
                raise RuntimeError(mssg)
                
            else:
                # We have FluidMixture and mass_frac
                self.update_substance = False
        
        else:
            # The user did not provide the dbm_mixture object; we need to 
            # create it
            self.update_substance = True
        
        # Update all model parameters in preparation of a simulation
        self._update()
        
    def _update(self):
        """
        Update the model parameters with the current model settings.  This
        method sets the following class attributes:
        
        Attributes 
        ----------
        release_rate_type : string
            Flag indicating whether mass flow rate is given ('mass_flowrate')
            or volume flow rate ('volume_flowrate')
        m0 : ndarray
            Array of mass flow rates (kg/s) for each chemical component 
            in the fluid mixture.  The user may specify the total mass flow
            rate (kg/s) of the mixture; or, if the volume flow rate and 
            gas-to-oil ratio is specified, this parameter is computed by 
            this method
        q0 : ndarray
            If volume flow rate is specified, this parameter holds the liquid
            volume flow rate in std bbl/d.  Otherwise, this parameter is set
            to `None`
        gor : ndarray
            If the volume flow rate is specified, this parameter reports 
            the gas to oil ratio (std ft^3/bbl) of natural gas to be added 
            to the mixture.  
        Vj : float
            Velocity of the produced water flowing out of the release, m/s
        Pj : float
            Pressure of the ambient fluid at the release point, Pa
        psm : tamoc.particle_size_model
            The particle size model object for this release
        m0_gas : ndarray
            Array of mass flow rates (kg/s) of each chemical component of the
            gas-phase fluid at the release
        m0_liq : ndarray
            Array of mass flow rates (kg/s) of each chemical component of the
            liquid-phase fluid at the release
        yk0_gas : ndarray
            Array of mole fractions (--) of each chemical component of the 
            gas-phase fluid at the release
        yk0_liq : ndarray
            Array of mole fractions (--) of each chemical component of the
            liquid-phase fluid at the release
        de_gas : ndarray
            Array of equivalent spherical diameters for each bubble size 
            class in the gas-phase bubble size distribution
        de_liq : ndarray
            Array of equivalent spherical diameters for each droplet size
            class in the liquid-phase droplet size distribution
        vf_gas : ndarray
            Array of volume fractions (--) for each bubble size class in the
            gas-phase bubble size distribution
        vf_liq : ndarray
            Array of volume fractions (--) for each droplet size class in the
            liquid-phase droplet size distribution
        mf_gas : ndarray
            Array of mass flow rates (kg/s) for each bubble size class in
            the gas-phase bubble size distribution
        mf_liq : ndarray
            Array of mass flow rates (kg/s) for each droplet size class in 
            the liquid-phase droplet size distribution
        gas : dbm.FluidParticle
            A discrete bubble model `FluidParticle` object for the gas-phase
            fluid at the release.
        liq : dbm.FluidParticle
            A discrete bubble model `FluidParticle` object for the liquid-
            phase fluid at the release.
        particles : list
            A list of bent plume model `Particle` objects that will be 
            released into the bent plume model for the simulation
        bpm : bent_plume_model.Model
            The bent plume model `Model` object that will hold the TAMOC 
            simulation
        update : bool
            Flag indicating whether the model parameters are up-to-date and
            ready for the `simulate` method to run.
        
        """
        # Echo progress to the user
        print('\nSetting up BPM Simulation Object...')
        
        # Get the ambient profile if needed
        if self.update_profile:
            print('    Building a profile object...')
            self._get_profile()
        
        # Get the fluid mixture object if needed
        if self.update_substance:
            print('    Creating the FluidMixture object...')
            self._get_substance()
        
        # Echo progress to the screen
        print('\nSetting up the spill parameters...')
        
        if self.mix_gas_for_gor:
            # Add natural gas to the mixture to match a given GOR
            self.dbm_mixture, self.mass_frac = \
                eos_utils.adjust_mass_frac_for_gor(self.dbm_mixture, 
                    self.mass_frac, self.gor)
        
        if self.add_air_to_dbm_object:
            # Add air components to the dbm_mixture to allow for gas 
            # stripping from the water column
            self.dbm_mixture, self.mass_frac = \
                eos_utils.add_air_comps_to_oil(self.dbm_mixture, 
                    self.mass_frac)
        
        if self.release_rate_type == 'mass_flowrate':
            # The user specified the total mixture mass flow rate...
            # Convert to flow rate for each component of the mixture.
            # Note that in this case the user must have already added gas
            # to achieve the desired GOR, so we just need to mass fraction
            # converted to mass flow rate
            self.m0 = self.m0 * self.mass_frac
        
        elif self.release_rate_type == 'volume_flowrate':
            # Get the component mass flow rates to match the given volume
            # flow rate
            self.m0 = eos_utils.mass_flowrate_from_volume_flowrate(
                self.dbm_mixture, self.mass_frac, self.q0)
            
        # Determine the produced water discharge velocity
        self.Vj = self.Qp / (np.pi * self.d0**2 / 4.)
        
        # Create the particle size model
        self.Pj = self.profile.get_values(self.X0[2], 'pressure')[0] 
        self.psm = particle_size_models.Model(self.profile, self.dbm_mixture,
            self.m0, self.X0[2], self.Tj, self.Pj)
        
        # Perform flash equilibrium at the release
        Ta, Pa = self.profile.get_values(self.X0[2], ['temperature', 
            'pressure'])
        print('\nSetting initial conditions with:  ')
        print(f'    Ta = {Ta:g}, Pa = {Pa:g}')
        m, xi, K = self.dbm_mixture.equilibrium(self.m0, Ta, Pa)
        self.m0_gas = m[0,:]
        self.m0_liq = m[1,:]
        self.yk0_gas = self.dbm_mixture.mol_frac(self.m0_gas)
        self.yk0_liq = self.dbm_mixture.mol_frac(self.m0_liq)        
        
        # Simulate the bubble/droplet breakup is needed
        if not isinstance(self.gas_model, type(None)):
            # We need to use the particle size model to get the bubble and 
            # droplet size distributions
            self.psm.simulate(self.d0, self.gas_model, self.pdf_gas, 
                self.liq_model, self.pdf_liq)
            self.de_gas, self.vf_gas, self.de_liq, self.vf_liq = \
                self.psm.get_distributions(self.n_gas, self.n_liq)
                    
        # Get the mass flow rate for each bubble and droplet size
        self.mf_gas = self.vf_gas * np.sum(self.m0_gas)
        self.mf_liq = self.vf_liq * np.sum(self.m0_liq)

        # Create gas and liquid dbm.FluidParticle objects
        user_data = self.dbm_mixture.user_data
        delta = self.dbm_mixture.delta
        delta_groups = self.dbm_mixture.delta_groups
        if self.single_phase_particles:
            fp_gas = 0
            fp_liq = 1
        else:
            fp_gas = 2
            fp_liq = 2
        sigma_correction = self.dbm_mixture.sigma_correction[0]
        isair = self.dbm_mixture.isair
        self.gas = dbm.FluidParticle(self.dbm_mixture.composition, 
            fp_type=fp_gas, delta=delta, delta_groups=delta_groups, 
            user_data=user_data, isair=isair, 
            sigma_correction=sigma_correction)
        self.liq = dbm.FluidParticle(self.dbm_mixture.composition,
            fp_type=fp_liq, delta=delta, delta_groups=delta_groups, 
            user_data=user_data, isair=isair, 
            sigma_correction=sigma_correction)
        
        # Create a single list of gas and liquid particles
        self.particles = []
        get_plume_particles(self.particles, self.profile, self.X0,
            self.gas, self.yk0_gas, self.mf_gas, self.de_gas, self.Tj, 
            0.9, self.K, self.K_T, self.fdis, self.t_hyd, self.lag_time)
        get_plume_particles(self.particles, self.profile, self.X0,
            self.liq, self.yk0_liq, self.mf_liq, self.de_liq, self.Tj,
            0.98, self.K, self.K_T, self.fdis, self.t_hyd, self.lag_time)
        
        # Create the bent plume model 
        self.bpm = bent_plume_model.Model(self.profile)
        
        # Store a flag indicating that the model is ready to run
        self.update = True
    
    def _get_profile(self):
        """
        Create an `ambient.Profile` object for a bent plume model simulation
        
        Use the data stored in `self.water` and `self.current` to create an
        `ambient.Profile` object and store in `self.profile`.  Possible 
        options for the `self.water` and `self.current` variables are listed
        in the notes below.        
        
        Notes
        -----
        The `self.water` variable contains information about the ambient
        temperature and salinity profile. Possible choices for `self.water`
        include the following:

        water : None
            Indicates that we have no information about the ambient
            temperature or salinity. In this case, the model will import data
            for the world-ocean average.
        water : dict
            If we only know the water temperature and salinity at the surface,
            this may be passed through a dictionary with keywords
            `temperature` and `salinity`. In this case, the model will import
            data for the world-ocean average and adjust the data to have the
            given temperature and salinity at the surface.
        water : 'netCDF4.Dataset' object
            If a 'netCDF4.Dataset' object already contains the ambient CTD
            data in a format appropriate for the `ambient.Profile` object,
            then this can be passed. In this case, it is assumed that the
            dataset includes the currents; hence, the `currents` variable will
            be ignored.
        water : `ambient.Profile` object
            If we already created our own ambient Profile object, then this
            object can be used directly.
        water : str
            If we stored the water column profile in a file, we may provide
            the file path to this file via the string stored in water. If this
            string ends in '.nc', it is assumed that this file contains a
            netCDF4 dataset. Otherwise, this file should contain columns in
            the following order: depth (m), temperature (deg C), salinity
            (psu), velocity in the x-direction (m/s), velocity in the
            y-direction (m/s). Since this option includes the currents, the
            current variable will be ignored in this case. A comment string of
            `#` may be used in the text file.
        water : ndarray
            If the data are provided as an `ndarray`, then these can be 
            passed to the profile constructor as is; hence, this option is 
            also enabled.

        The `self.current` variable contains information about the ambient
        current profile. Possible choices for `current` include the following:

        current : float
            This is assumed to be the current velocity along the x-axis and
            will be uniform over the depth
        current : ndarray
            This is assumed to contain the current velocity in the x- and y-
            (and optionally also z-) directions. If this is a one-dimensional
            array, then these currents will be assumed to be uniform over the
            depth. If this is a multi-dimensional array, then these values are
            assumed to contain a profile of data, with the depth (m) as the
            first column of data.

        """
        from netCDF4 import date2num, Dataset
        NoneType = type(None)
        done = False
    
        # Extract the temperature and salinity data
        if isinstance(self.water, NoneType):
            # Use the world-ocean average T(z) and S(z)
            data = None

        if isinstance(self.water, dict):
            # Get the water temperature and salinity at the surface
            Ts = self.water['temperature']
            Ss = self.water['salinity']

            # Create a data array of depth, temperature, and salinity
            data = np.array([0., Ts, Ss])

        if isinstance(self.water, Dataset):
            # A netCDF4 Dataset containing all of the profile data is stored
            # in water.  Use that to create the Profile object
            self.profile = ambient.Profile(self.water, 
                chem_names=self.chem_names)
            done = True

        if isinstance(self.water, ambient.Profile):
            # An ambient.Profile object has been provided; use that profile
            self.profile = self.water
            done = True

        elif isinstance(self.water, str) or isinstance(self.water, unicode):

            if self.water[-3:] == '.nc':
                # Water contains a path to a netCDF4 dataset.  Use this to
                # create the Profile object
                self.profile = ambient.Profile(self.water,
                    chem_names=self.chem_names)
                done = True

            else:
                # This must be a relative path to a text file
                self.profile = ambient_utils.profile_from_txt(
                    self.water, self.chem_names, None, None, 
                    self.stabilize_profile, self.err, self.add_air_to_profile)
                done = True

        # Create the `ambient.Profile` object
        if not done:
            # Any of the above methods that create an ndarray of data will
            # be converted to a profile on this line.
            self.profile = ambient.Profile(self.water, current=self.current, 
                current_units='m/s')
        
        # Set the update flag to False since the profile is now up-to-date
        self.update_profile = False
    
    def _get_substance(self):
        """
        Get the mixture object and mass fractions for the spilled substance
        
        Create a `dbm.FluidMixture` object and initialize the `mass_frac`
        array defining the substance spilled at the release.  See the Notes
        section below for ways to specify the fluid substance released, which
        follows the approach used in the older `Blowout` object of `tamoc`.
        
        Notes
        -----
        We pass the information needed to create the `dbm_mixture` through
        the `substance` parameter of the `kwargs` list.  There are two  
        options available:
        
        substance : dict
            composition : list
                The dictionary entry for `composition` should contain a 
                list of string names that correspond to chemical components
                in the `tamoc` database of chemical properties.
            masses : ndarray
                The dictionary entry for masses should contain the mass 
                fractions of each chemical component at the release
        
        substance : str
            A file name with optional relative or absolute file path a a 
            `.json` file in the format of an Adios DB data record.  For this
            substance, an equivalent `tamoc` fluid mixture is created using
            the methods described in Gros et al. (MPB, vol. 37, 2018).
        
        """
        if isinstance(self.substance, dict):
            self.dbm_mixture = dbm.FluidMixture(self.substance['composition'])
            self.mass_frac = self.substance['masses']
    
        elif isinstance(self.substance, str):
            composition, mass_frac, user_data, delta, delta_groups, units = \
                dbm_utilities.load_adios_oil(self.substance)
            self.dbm_mixture = dbm.FluidMixture(composition, 
                user_data=user_data, delta=delta, delta_groups=delta_groups)
            self.mass_frac = mass_frac
        
        else:
            print(f'ERROR:  A substance of type {type(substance)} is not')
            print('        defined for a TAMOC Bent Plume Model Simulation.')
        
        # Set the update flag to False since the mixture is now up-to-date
        self.update_substance = False
        

    def simulate(self):
        """
        Run a bent plume model simulation
        
        Run the bent plume model using the present simulation settings
        
        Notes
        -----
        This method updates the `bpm` attribute of this class with an 
        object that contains the simulation result.
        
        """
        # Make sure the class attributes are up-to-date
        if not self.update:
            self._update()
        
        # Determine how far to simulate along the plume trajectory
        sd_max = 5. * self.X0[2] / self.d0
        
        # Run the bent plume model        
        self.bpm.simulate(self.X0, self.d0, self.Vj, self.phi_0, self.theta_0,
            self.Sp, self.Tj, self.cj, self.tracers, 
            particles=self.particles, track=self.track, dt_max=self.dt_max, 
            sd_max=sd_max)
        
        # Set the flag to indicate that the model has run and needs to be
        # updated before it is run again
        self.update = False
    
    def update_water_data(self, water):
        """
        Update the temperature/salinity data stored for the profile object
        
        When this object is used in a PyGnome simulation, the profile data
        may change with time.  This method accepts a new set of data for
        the `water` parameter, sets the `update` flag indicating whether this
        data has been inserted into the profile to `False`, and sets the
        `sim_stored` flag for the `bpm` object to `False`, indicating the
        present model data do not reflect any existing model simulation.  
        These two flags will cause other functions to move this data into the 
        profile object and re-run the simulation.
        
        Parameters
        ----------
        water : various
            Data describing the ambient water temperature and salinity
            profile.  See Notes below for details.

        Notes
        -----
        The ambient water column data can be provided through several
        different options. The `water` variable contains temperature and
        salinity data. The user may define the `water` in the following ways:

        water : None
            Indicates that we have no information about the ambient
            temperature or salinity. In this case, the model will import data
            for the world-ocean average.
        water : dict
            If we only know the water temperature and salinity at the
            surface, this may be passed through a dictionary with keywords
            `temperature` and `salinity`. In this case, the model will import
            data for the world-ocean average and adjust the data to have the
            given temperature and salinity at the surface.
        water : 'netCDF4.Dataset'
            If a 'netCDF4.Dataset' object already contains the ambient CTD
            data in a format appropriate for the `ambient.Profile` object,
            then this can be passed. In this case, it is assumed that the
            dataset includes the currents; hence, the `currents` variable
            will be ignored.
        water : `ambient.Profile` object
            If we already created our own ambient Profile object, then this
            object can be used directly.
        water = str
            If we stored the water column profile in a file, we may provide
            the file path to this file via the string stored in water. If
            this string ends in '.nc', it is assumed that this file contains
            a netCDF4 dataset. Otherwise, this file should contain columns in
            the following order: depth (m), temperature (deg C), salinity
            (psu), velocity in the x-direction (m/s), velocity in the
            y-direction (m/s). Since this option includes the currents, the
            current variable will be ignored in this case. A comment string
            of `#` may be used in the text file.
        
        """
        self.water = water
        self.update_profile = True
        self.bpm.sim_stored = False
        self.update = False
    
    def update_current_data(self, current):
        """        
        Update the current data stored for the profile object
        
        When this object is used in a PyGnome simulation, the profile data
        may change with time.  This method accepts a new set of data for
        the `water` parameter, sets the `update` flag indicating whether this
        data has been inserted into the profile to `False` and sets the
        `sim_stored` flag to `False`.  These two flags will cause other 
        functions to move this data into the profile object and re-run the
        simulation.

        Notes
        -----
        Current profile data can be provided through several different
        options. The user may define the `current` in the following ways:

        current : float
            This is assumed to be the current velocity along the x-axis and
            will be uniform over the depth
        current : ndarray
            This is assumed to contain the current velocity in the x- and y-
            (and optionally also z-) directions. If this is a one-dimensional
            array, then these currents will be assumed to be uniform over the
            depth. If this is a multi-dimensional array, then these values are
            assumed to contain a profile of data, with the depth (m) as the
            first column of data.

        """
        self.current = current
        self.update_profile = True
        self.bpm.sim_stored = False
        self.update = False
        
    def _report_run_bpm(self):
        """
        Print an error message requesting user to run the `simulate` method
        
        """
        print('\nERROR:  Bent plume model simulation not available.')
        print('        Run the `simulate` method before calling this')
        print('        method.\n')
        return (0,)
    
    def plot_ambient_profile(self, fig=1, clear_fig=True):
        """
        Plot the ambient profile data
        
        Parameters 
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces one figure
            with multiple subplots.
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        fig : plt.figure
            Returns a handle to the figure that could be used for saving 
            the figure
        
        """
        print('Plotting the ambient profile data...')
        # Plot the variables affecting the physics of the simulation
        self.profile.plot_physical_profiles(fig, clear_fig)
        
        # Plot all other variables
        self.profile.plot_chem_profiles(fig+1, clear_fig)
        print('Done.')
    
    def plot_initial_psds(self, fig=3, clear_fig=True):
        """
        Plot the initial bubble and droplet size distributions
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces one figure
            with multiple subplots.
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        fig : plt.figure
            Returns a handle to the figure that could be used for saving 
            the figure
        
        """
        if self.bpm.sim_stored:
            return self.bpm.plot_psds(fig, 0, 0, clear_fig)
        else:
            return _report_run_bpm()
    
    def plot_state_space(self, fig=4, clear_fig=True):
        """
        Plot the bent plume model state space
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces one figure
            with multiple subplots.
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        fig : plt.figure
            Returns a handle to the figure that could be used for saving 
            the figure
        
        """
        if self.bpm.sim_stored:
            return self.bpm.plot_state_space(fig, clear_fig)
        else:
            return _report_run_bpm()
        
    def plot_all_variables(self, fig=5, clear_fig=True):
        """
        Plot all variables from the bent plume model solution
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces multiple
            figures.
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        fig : plt.figure
            Returns a handle to the figure that could be used for saving 
            the figure
        
        """
        if self.bpm.sim_stored:
            return self.bpm.plot_all_variables(fig, clear_fig)
        else:
            return _report_run_bpm()
    
    def plot_fractions_dissolved(self, fig=100, clear_fig=True):
        """
        Plot the fraction dissolved for each tracked chemical component
        
        This method creates three figures.  The first figure displays the 
        fate of released chemicals in the near-field plume.  The second
        figure displays the fate of released chemicals in the far-field
        portion of the simulation only.  The third figure displays the fate
        of the released chemicals through the whole near-field and far-field
        simulation domains.        
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces multiple
            figures.
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        figs : list
            Returns a list of handles to the figures that could be used for
            saving the figures
        
        """
        if not self.bpm.sim_stored:
            return _report_run_bpm()
            
        else:
            # Plot only those compounds with a non-zero mass flow rate at the
            # release
            chems = []
            for i in range(len(self.dbm_mixture.composition)):
                if self.mass_frac[i] > 0:
                    chems.append(self.dbm_mixture.composition[i])
        
            # Create a list to hold the figures
            figs = []
        
            # Start with the near-field simulation
            f = self.bpm.plot_fractions_dissolved(fig, chems=chems, stage=0, 
                clear=clear_fig, title='End of Plume Stage')
            figs.append(f)
        
            # Then the far-field simulation
            f = self.bpm.plot_fractions_dissolved(fig+1, chems=chems, stage=1,
                clear=clear_fig, title='Within the Far Field')
            figs.append(f)
        
            # And finally the whole simulation
            f = self.bpm.plot_fractions_dissolved(fig+2, chems=chems, 
                stage=-1, clear=clear_fig, title='Over the Whole Simulation')
            figs.append(f)
        
            return figs
    
    def plot_mass_balance(self, fig=200, t_max=-1, clear_fig=True):
        """
        Plot the time-history of the mass balance
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces multiple
            figures.
        t_max : float, default=-1
            The maximum time to include in the history plot (days).  If -1,
            then the maximum surfacing time in the simulation is used.        
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        fig : list
            Returns a handles to the figure that could be used for saving the
            figures
        
        """
        if not self.bpm.sim_stored:
            return _report_run_bpm()
            
        else:            
            # Plot only those compounds with a non-zero mass flow rate at the
            # release
            chems = []
            for i in range(len(self.dbm_mixture.composition)):
                if self.mass_frac[i] > 0:
                    chems.append(self.dbm_mixture.composition[i])
            
            # Create the plot
            return self.bpm.plot_mass_balance(fig, chems=chems, fp_type=-1, 
                t_max=t_max, clear=clear_fig)
        

def get_plume_particles(particles, profile, X0, dbm_fluid, yk, mf, de, 
    Tj, lambda_1, K, K_T, fdis, t_hyd, lag_time):
    """
    Create plume particles for use in the bent and stratified plume models
    
    Create `bent_plume_model.Particle` objects from a given size distribution
    and mass flowrates.  These objects, though for the `bent_plume_model`, are
    compatible with simulations of the bent plume or stratified plume models
    
    Parameters
    ----------
    particles : list
        List to which to append new particles
    profile : ambient.Profile
        An ambient profile object for getting ambient properties
    X0 : ndarray
        Array containing the release location of the plume (x, y, z), m
    dbm_fluid : dbm.FluidParticle
        A `dbm.FluidParticle` object for the present set of particles.  This
        object differs from the `dbm.FluidMixture` in that it is expected to
        be single-phse or nearly single-phase and can report bubble and
        droplet properties.
    yk : ndarray
        Array of mole fractions for each component in the fluid mixture.  
        Theses should be the mole fractions for the single-phase fluid 
        described by the `dbm_fluid` object.
    mf : ndarray
        Array of mass flow rates (kg/s) for each bubble or droplet in the
        set of particles created by this function
    de : ndarray
        Array of corresponding equivalent spherical diameters (m) for each
        bubble or droplet in the set of particles created by this function
    Tj : float
        Temperature of the bubbles or droplets.
    lamba_1 : float, default=0.9
        Spreading ratio of dispersed phases to the entrained fluid phase.
        This value is typically between 0.9 and 1 for small particles with
        low inertia and low rise velocity.  Large bubbles or heavy sediment 
        particles may have values between 0.6 and 0.8.  The model is not very 
        sensitive to this value.
    K : float
        Mass transfer reduction factor
    K_T : float
        Heat transfer reduction factor
    fdis : float
        Fraction of initial mass remaining to consider one component of the
        fluid mixture fully dissolved
    t_hyd : float
        Mass transfer surfactant / hydrate transition time, s.  Particle
        mass transfer rates transition from clean to dirty mass transfer
        coefficients after a time `t_hyd`
    lag_time : bool
        Flag that indicates whether (True) or not (False) to use the
        biodegradation lag times data.
    
    Notes
    -----
    Because lists are mutable in Python, the input `particles` list will be
    modified in place by this function.  Because the input list will
    become the desired output list, we do not provide a return value so that
    the user never believes the input list would be left alone.
    
    This function creates the particles that will be released into the plume
    models. Hence, these particles must be computed at the in situ pressure of
    the ambient water at the release location. It is not possible to specify
    some other pipeline pressure when creating these bubbles or droplets
    because they will immediately be subjected to the ambient pressure in the
    plume.
    
    """
    # Make sure the input mass flow rate and diameters are lists or arrays
    if isinstance(mf, float):
        mf = np.array([mf])
    if isinstance(de, float):
        de = np.array([de])
    
    # Create each particle separately
    for i in range(len(de)):
        
        # Get the initial conditions for this particle size class
        m0, T0, nb0, Pa, Sa, Ta = dispersed_phases.initial_conditions(
            profile, X0[2], dbm_fluid, yk, mf[i], 2, de[i], Tj
        )
        
        # Use these initial conditions to create the particle object and 
        # add it to the list
        particles.append(bent_plume_model.Particle(
            X0[0], X0[1], X0[2], dbm_fluid, m0, Tj, nb0, lambda_1, Pa, Sa, 
            Ta, K=K, K_T=K_T, fdis=fdis, t_hyd=t_hyd, lag_time=lag_time
        )) 
       