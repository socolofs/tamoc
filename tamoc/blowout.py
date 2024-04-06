"""
Blowout Module
==============

This module provides a `Blowout` object to make creation of oil well blowout
scenarios using the `bent_plume_model` simple. In particular, this object
helps coordinate creation of the `ambient.Profile` data and the blowout
initial conditions, including generation of the bubble and droplet size
distributions. This module relies on the `dbm_utilities` module to do things
like mixing natural gas into a dead oil to create a live oil, etc. and uses
the `particle_size_models` to generate bubbles and droplets.

See Also
--------
bent_plume_model, dbm, dbm_utilities, particle_size_models

Notes
-----
This module provides the option to obtain oil properties from distillation
cut data available from the National Oceanic and Atmospheric Administration
(NOAA) Oil Library. To make use of this capability, you will need to install
the OilLibrary package into your Python environment. You may access this
package here::

    https://github.com/NOAA-ORR-ERD/OilLibrary

Once this package is installed, you may load an oil into a `dbm` fluid object
using the Adios ID Number (e.g., AD01554) index of the OilLibrary.

"""
# S. Socolofsky, February 2020, Texas A&M University, <socolofs@tamu.edu>

from __future__ import (absolute_import, division, print_function)
unicode = type(u' ')

from tamoc import seawater, ambient, dbm, dispersed_phases, bent_plume_model
from tamoc import particle_size_models as psm
from tamoc import dbm_utilities

from datetime import datetime
from netCDF4 import date2num, Dataset

import numpy as np


class Blowout(object):
    """
    Class to facilitiate creating simulations using the  `bent_plume_model`

    Class to help set up all of the elements necessary to run a blowout
    simulation using the ``TAMOC`` `bent_plume_model`. This includes creating
    the `ambient.Profile` object, defining an oil and gas composition using
    the `dbm.FluidMixture` and `dbm.FluidParticle` objects, and generating
    `particle` lists with initial conditions for the gas bubble and oil
    droplet size distributions at the orifice.

    This class is designed for use in subsea oil well blowouts, hence, flow
    rates of oil and gas can be specified using dead oil rates in bbl/d and
    gas-to-oil ratio (GOR) in ft^3/bbl.

    Parameters
    ----------
    z0 : float, default=100
        Depth of the release point (m)
    d0 : float, default=0.1
        Equivalent circular diameter of the release (m)
    substance : str or list of str, default=[methane]
        The chemical composition of the released petroleum fluid.  If using
        the chemical property data distributed with TAMOC, this should be a
        list of TAMOC chemical property names.  If using an oil from the
        NOAA OilLibrary, this should be a string containing the Adios oil
        ID number (e.g., "AD01554" for Louisiana Light Sweet).
    q_oil : float, default=20000.
        Release rate of the dead oil composition at the release point in
        stock barrels of oil per day.
    gor : float, default=0.
        Gas to oil ratio at standard surface conditions in standard cubic
        feet per stock barrel of oil
    x0 : float, default=0
        x-coordinate of the release (m)
    y0 : float, default=0
        y-coordinate of the release (m)
    u0 : float, default=None
        Exit velocity of continuous-phase fluid at the release.  This is
        only used when produced water exits.  For a pure oil and gas release,
        this should be zero or None.
    phi_0 : float, default=-np.pi / 2. (vertical release)
        Vertical angle of the release relative to the horizontal plane; z is
        positive down so that -pi/2 represents a vertically upward flowing
        release (rad)
    theta_0 : float, default=0.
        Horizontal angle of the release relative to the x-direction (rad)
    num_gas_elements : int, default=10
        Number of gas bubble sizes to include in the gas bubble size
        distribution
    num_oil_elements : int, default=25
        Number of oil droplet sizes to include in the oil droplet size
        distribution
    water : various
        Data describing the ambient water temperature and salinity profile.
        See Notes below for details.
    current : various
        Data describing the ambient current velocity profile.  See Notes
        below for details.
    ca : str or list
        Atmospheric gases to add to the oil composition data. The default is
        'all', which will add nitrogen, oxygen, argon, and carbon dioxide to
        the oil composition data. For other choices, include a list of
        compound names. If all desired compounds are already in the oil
        composition data, this list should be empty.

    Attributes
    ----------
    z0 : float, default=100
        Depth of the release point (m)
    d0 : float, default=0.1
        Equivalent circular diameter of the release (m)
    substance : str or list of str, default=[methane]
        The chemical composition of the released petroleum fluid.  If using
        the chemical property data distributed with TAMOC, this should be a
        list of TAMOC chemical property names.  If using an oil from the
        NOAA OilLibrary, this should be a string containing the Adios oil
        ID number (e.g., 'AD01554' for Louisiana Light Sweet).
    q_oil : float, default=20000.
        Release rate of the dead oil composition at the release point in
        stock barrels of oil per day.
    gor : float, default=0.
        Gas to oil ratio at standard surface conditions in standard cubic
        feet per stock barrel of oil
    x0 : float, default=0
        x-coordinate of the release (m)
    y0 : float, default=0
        y-coordinate of the release (m)
    u0 : float, default=None
        Exit velocity of continuous-phase fluid at the release.  This is
        only used when produced water exits.  For a pure oil and gas release,
        this should be zero or None.
    phi_0 : float, default=-np.pi / 2. (vertical release)
        Vertical angle of the release relative to the horizontal plane; z is
        positive down so that -pi/2 represents a vertically upward flowing
        release (rad)
    theta_0 : float, default=0.
        Horizontal angle of the release relative to the x-direction (rad)
    num_gas_elements : int, default=10
        Number of gas bubble sizes to include in the gas bubble size
        distribution
    num_oil_elements : int, default=25
        Number of oil droplet sizes to include in the oil droplet size
        distribution
    water : various
        Data describing the ambient water temperature and salinity profile.
        See Notes below for details.
    current : various
        Data describing the ambient current velocity profile.  See Notes
        below for details.
    profile : `ambient.Profile` object
        An `ambient.Profile` object containing the ambient CTD and current
        information
    T0 : float
        Ambient water temperature (K) at the release
    S0 : float
        Ambient water salinity (psu) at the release
    P0 : float
        Ambient water pressure (Pa) at the release
    gas : float
        A `dbm.FluidParticle` object defining the gas-phase fluid at the
        release
    liq : float
        A `dbm.FluidParticle` object defining the liquid-phase fluid at the
        release
    d_gas : ndarray
        Equivalent spherical diameters (m) of the gas bubbles at the release
    vf_gas : ndarray
        Volume fraction of gas in each of the diameters stored in `d_gas`
    d_liq : ndarray
        Equivalent spherical diameters (m) of the liquid droplets at the
        release
    vf_liq : ndarray
        Volume fraction of liquid in each of the diameters stored in `d_liq`
    disp_phases : list
        List of `bent_plume_model.Particle` objects that define each gas
        bubble and liquid droplet released from the orifice
    bpm : `bent_plume_model.Model` object
        A `bent_plume_model.Model` object that contains the simulation run
        defined by the present class object.

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
    water = str
        If we stored the water column profile in a file, we may provide the
        file path to this file via the string stored in water. If this string
        ends in '.nc', it is assumed that this file contains a netCDF4
        dataset. Otherwise, this file should contain columns in the following
        order: depth (m), temperature (deg C), salinity (psu), velocity in
        the x-direction (m/s), velocity in the y-direction (m/s). Since this
        option includes the currents, the current variable will be ignored in
        this case. A comment string of `#` may be used in the text file.

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

    """
    def __init__(self,
                 z0=100,
                 d0=0.1,
                 substance={
                     'composition' : ['methane', 'ethane', 'propane',
                                    'toluene', 'benzene'],
                     'masses' : np.array([0.2, 0.03, 0.02, 0.25, 0.5])
                 },
                 q_oil=20000.,
                 gor=0.,
                 x0=0.,
                 y0=0.,
                 u0=None,
                 phi_0=-np.pi / 2.,
                 theta_0=0.,
                 num_gas_elements=10,
                 num_oil_elements=25,
                 water=None,
                 current=np.array([0.1, 0., 0.]),
                 ca='all',
                 size_distribution=None
                 ):

        super(Blowout, self).__init__()

        # Store the model parameters
        self.z0 = z0
        self.d0 = d0
        self.substance = substance
        self.q_oil = q_oil
        self.gor = gor
        self.x0 = x0
        self.y0 = y0
        self.u0 = u0
        self.phi_0 = phi_0
        self.theta_0 = theta_0
        self.num_gas_elements = num_gas_elements
        self.num_oil_elements = num_oil_elements
        self.size_distribution = size_distribution
        self.water = water
        self.current = current
        
        # Set some additional default parameters
        self.track = True
        
        # Create a list of atmospheric gases
        if ca == 'all':
            self.ca = ['nitrogen', 'oxygen', 'argon', 'carbon_dioxide']
            self.new_oil = True
        else:
            self.ca = ca
            self.new_oil = True
        
        # Decide which phase flow rate is reported through q_oil
        if self.num_oil_elements > 0:
            # User is simulating oil; hence, oil flow rate should be given
            self.q_type = 1
        else:
            # User is simulating gas only; hence, gas flow rate should be
            # given
            self.q_type = 0

        # Create the remaining object attributes needed to set up a `tamoc`
        # `bent_plume_model` simulation
        self._update()

    def _update(self):
        """
        Initialize bent_plume_model for simulation run

        Set up the ambient profile, initial conditions, and model parameters
        for a new simulation run of the `bent_plume_model`.

        """
        # Get an ambient Profile object
        self.profile = get_ambient_profile(self.water, self.current,
                       ca=self.ca)


        # Import the oil with the desired gas-to-oil ratio
        if self.new_oil:
            self.oil, self.mass_flux = dbm_utilities.get_oil(self.substance,
                                                             self.q_oil,
                                                             self.gor,
                                                             self.ca,
                                                             self.q_type)
            self.new_oil = False

        # Find the ocean conditions at the release
        self.T0, self.S0, self.P0 = self.profile.get_values(self.z0,
                                       ['temperature',
                                        'salinity',
                                        'pressure'])

        # Define some of the constant initial conditions
        self.Sj = 0.
        self.Tj = self.T0
        self.cj = 1.
        self.tracers = ['tracer']

        # Compute the equilibrium mixture properties at the release
        m, xi, K = self.oil.equilibrium(self.mass_flux, self.Tj, self.P0)

        # Create the discrete bubble model objects for gas and liquid
        self.gas = dbm.FluidParticle(self.oil.composition,
                                     fp_type=0,
                                     delta=self.oil.delta,
                                     user_data=self.oil.user_data)
        self.liq = dbm.FluidParticle(self.oil.composition,
                                     fp_type=1,
                                     delta=self.oil.delta,
                                     user_data=self.oil.user_data)

        # Compute the bubble and droplet volume size distributions
        if self.size_distribution == None:
            self.breakup_model = psm.Model(self.profile, self.oil, 
                self.mass_flux, self.z0, self.Tj)
            self.breakup_model.simulate(self.d0, model_gas='wang_etal',
                model_oil='sintef')
            self.d_gas, self.vf_gas, self.d_liq, self.vf_liq = \
                self.breakup_model.get_distributions(self.num_gas_elements,
                self.num_oil_elements)
        else:
            self.breakup_model = None
            self.d_gas = self.size_distribution['d_gas']
            self.vf_gas = self.size_distribution['vf_gas']
            self.d_liq = self.size_distribution['d_liq']
            self.vf_liq = self.size_distribution['vf_liq']

        # Create the `bent_plume_model` particle list
        self.disp_phases = []
        self.disp_phases += particles(np.sum(m[0,:]), self.d_gas,
                                      self.vf_gas, self.profile, self.gas,
                                      xi[0,:], 0., 0., self.z0, self.Tj,
                                      0.9, False)
        self.disp_phases += particles(np.sum(m[1,:]), self.d_liq,
                                      self.vf_liq, self.profile, self.liq,
                                      xi[1,:], 0., 0., self.z0, self.Tj,
                                      0.98, False)

        # Set some of the hidden model parameters
        self.dt_max = 5. * 3600.
        self.sd_max = 300. * self.z0 / self.d0

        # Create the initialized `bent_plume_model` object
        self.bpm = bent_plume_model.Model(self.profile)

        # Set the flag to indicate the model is ready to run
        self.update = True

    def simulate(self):
        """
        Run a bent_plume_model simulation for the present conditions

        Calls the `bent_plume_model.Model.simulate()` method with the initial
        conditions presently stored in the class object. This method does not
        have any input parameters and does not return a value. After the
        simulation is run, the `bent_plume_model.Model` object will store the
        solution, and this object is stored as the `bpm` attribute of the
        present class.

        """
        # Check whether we need to update the model initial conditions
        if not self.update:
            self._update()

        # Run the new simulation
        self.bpm.simulate(np.array([self.x0, self.y0, self.z0]),
                          self.d0,
                          self.u0,
                          self.phi_0,
                          self.theta_0,
                          self.Sj,
                          self.Tj,
                          self.cj,
                          self.tracers,
                          self.disp_phases,
                          self.track,
                          self.dt_max,
                          self.sd_max)

        # Set the flag to indicate that the model has run and needs to be
        # updated before it is run again
        self.update = False

    def save_sim(self, fname, profile_path, profile_info):
        """
        Save the `bent_plume_model` complete solution in netCDF format

        Parameters
        ----------
        fname : str
            File name of the netCDF file to write
        profile_path : str
            String stating the file path to the ambient profile data relative
            to the directory where `fname` will be saved.
        profile_info : str
            Single line of text describing the ambient profile data.

        """
        if self.bpm.sim_stored is False:
            print('No simulation results available to store...')
            print('Run Blowout.simulate() first.\n')
            return

        self.bpm.save_sim(fname, profile_path, profile_info)

    def save_txt(self, base_name, profile_path, profile_info):
        """
        Save the `bent_plume_model` state space in ascii text format

        Parameters
        ----------
        base_name : str
            Base file name for the output file.  This method will append the
            .txt file extension to the data output and write a second file
            with the header information called base_name_header.txt.  If the
            particles that left the plume were tracked in the farfield, it
            will also save the trajectory of those particles as
            base_name_nnn.txt (output data) and base_name_nnn_header.txt
            (header data for far field data).
        profile_path : str
            String stating the file path to the ambient profile data relative
            to the directory where `fname` will be saved.
        profile_info : str
            Single line of text describing the ambient profile data.

        """
        if self.bpm.sim_stored is False:
            print('No simulation results available to store...')
            print('Run Blowout.simulate() first.\n')
            return

        self.bpm.save_txt(base_name, profile_path, profile_info)

    def plot_state_space(self, fig=1):
        """
        Plot the `bent_plume_model` state space solution

        Parameters
        ----------
        fig : int or MPL Figure object
            MPL Figure() on which to plot
            or
            Number of the figure window in which to draw the plot
            (Figure will be created for you with the provided fig number)

        Returns
        -------
        fig : MPL Figure
            The MPL figure of the created plot

        """
        if self.bpm.sim_stored is False:
            print('No simulation results available to analyze...')
            print('Run Blowout.simulate() first.\n')
            return

        return self.bpm.plot_state_space(fig)

    def plot_all_variables(self, fig=2):
        """
        Plot all variables for the `bent_plume_model` solution

        Parameters
        ----------
        fig : int or MPL Figure object
            MPL Figure() on which to plot
            or
            Number of the figure window in which to draw the plot
            (Figure will be created for you with the provided fig number)

        Returns
        -------
        fig : MPL Figure
            The MPL figure of the created plot


        """
        if self.bpm.sim_stored is False:
            print('No simulation results available to analyze...')
            print('Run Blowout.simulate() first.\n')
            return

        return self.bpm.plot_all_variables(fig)

    def update_release_depth(self, z0):
        """
        Change the release depth (m) to use in a model simulation

        Parameters
        ----------
        z0 : float, default=100
            Depth of the release point (m)

        """
        self.z0 = z0
        self.update = False
        self.bpm.sim_stored = False

    def update_orifice_diameter(self, d0):
        """
        Change the orifice diametr (m) to use in a model simulation

        Parameters
        ----------
        d0 : float, default=0.1
            Equivalent circular diameter of the release (m)

        """
        self.d0 = d0
        self.update = False
        self.bpm.sim_stored = False

    def update_substance(self, substance):
        """
        Change the OilLibrary ID number to use in a model simulation
        
        Parameters
        ----------
        substance : str or list of str, default=[methane]
            The chemical composition of the released petroleum fluid. If
            using the chemical property data distributed with TAMOC, this
            should be a list of TAMOC chemical property names. If using an
            oil from the NOAA OilLibrary, this should be a string containing
            the Adios oil ID number (e.g., 'AD01554' for Louisiana Light
            Sweet).

        Notes
        -----
        The spilled substance can either be taken from the NOAA OilLibrary or
        can be created from individual pseudo-components in TAMOC. The user
        may define the `substance` in one of two ways:

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

        """
        self.substance = substance
        self.update = False
        self.new_oil = True
        self.bpm.sim_stored = False

    def update_q_oil(self, q_oil):
        """
        Change the oil flow rate (bbl/d) to use in a model simulation

        Parameters
        ----------
        q_oil : float, default=20000.
            Release rate of the dead oil composition at the release point in
            stock barrels of oil per day.

        """
        self.q_oil = q_oil
        self.update = False
        self.new_oil = True
        self.bpm.sim_stored = False

    def update_gor(self, gor):
        """
        Change the gas-to-oil ratio (std ft^3/bbl) to use in a model
        simulation

        Parameters
        ----------
        gor : float, default=0.
            Gas to oil ratio at standard surface conditions in standard cubic
            feet per stock barrel of oil

        """
        self.gor = gor
        self.update = False
        self.new_oil = True
        self.bpm.sim_stored = False

    def update_produced_water(self, u0):
        """
        Change the amount of produced water (m/s) exiting with the oil and
        gas through the orifice

        Parameters
        ----------
        u0 : float, default=None
            Exit velocity of continuous-phase fluid at the release. This is
            only used when produced water exits. For a pure oil and gas
            release, this should be zero or None.

        """
        self.u0 = u0
        self.update = False
        self.bpm.sim_stored = False

    def update_vertical_orientation(self, phi_0):
        """
        Change the vertical orientation (rad) of the release

        Parameters
        ----------
        phi_0 : float, default=-np.pi / 2. (vertical release)
            Vertical angle of the release relative to the horizontal plane; z
            is positive down so that -pi/2 represents a vertically upward
            flowing release (rad)

        """
        self.phi_0 = phi_0
        self.update = False
        self.bpm.sim_stored = False

    def update_horizontal_orientation(self, theta_0):
        """
        Change the horizontal orientation (rad) of the release

        Parameters
        ----------
        theta_0 : float, default=0.
            Horizontal angle of the release relative to the x-direction (rad)

        """
        self.theta_0 = theta_0
        self.update = False
        self.bpm.sim_stored = False

    def update_num_gas_elements(self, num_gas_elements):
        """
        Change the number of gas bubbles to include in the simulation

        Parameters
        ----------
        num_gas_elements : int, default=10
            Number of gas bubble sizes to include in the gas bubble size
            distribution

        """
        self.num_gas_elements = num_gas_elements
        self.update = False
        self.bpm.sim_stored = False

    def update_num_oil_elements(self, num_oil_elements):
        """
        Change the number of oil droplets to include in the simulation

        Parameters
        ----------
        num_oil_elements : int, default=25
            Number of oil droplet sizes to include in the oil droplet size
            distribution

        """
        self.num_oil_elements = num_oil_elements
        self.update = False
        self.bpm.sim_stored = False

    def update_water_data(self, water):
        """
        Change the ambient temperature and salinity profile data

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
        self.update = False
        self.bpm.sim_stored = False

    def update_current_data(self, current):
        """
        Change the ambient current profile data

        Parameters
        ----------
        current : various
            Data describing the ambient current velocity profile.  See Notes
            below for details.

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
            depth. If this is a multi-dimensional array, then these values as
            assumed to contain a profile of data, with the depth (m) as the
            first column of data.

        """
        self.current = current
        self.update = False
        self.bpm.sim_stored = False


    def update_track_particles(self, track):
        """
        Set whether the fluid particles are tracked in the far-field

        Parameters
        ----------
        track : bool
            Flag indicating whether or not to track the fluid particles at
            the end of the near-field into the far-field. Default behavior of
            the blowout object is to set track = True.

        """
        self.track = track
        self.update = False
        self.bpm.sim_stored = False


# --- Helper functions used by the Blowout object ---

def particles(m_tot, d, vf, profile, oil, yk, x0, y0, z0, Tj, lambda_1,
              lag_time, t_hyd=0):
    """
    Create particles to add to a bent plume model simulation

    Creates bent_plume_model.Particle objects for the given particle
    properties so that they can be added to the total list of particles
    in the simulation.

    Parameters
    ----------
    m_tot : float
        Total mass flux of this fluid phase in the simulation (kg/s)
    d : np.array
        Array of particle sizes for this fluid phase (m)
    vf : np.array
        Array of volume fractions for each particle size for this fluid
        phase (--).  This array should sum to 1.0.
    profile : ambient.Profile
        An ambient.Profile object with the ambient ocean water column data
    oil : dbm.FluidParticle
        A dbm.FluidParticle object that contains the desired oil database
        composition
    yk : np.array
        Mole fractions of each compound in the chemical database of the oil
        dbm.FluidParticle object (--).
    x0, y0, z0 : floats
        Initial position of the particles in the simulation domain (m).  Note
        that x0 and y0 should be zero for particles starting on the plume
        centerline.
    Tj : float
        Initial temperature of the particles in the jet (K)
    lambda_1 : float
        Value of the dispersed phase spreading parameter of the jet integral
        model (--).
    lag_time : bool
        Flag that indicates whether (True) or not (False) to use the
        biodegradation lag times data.
    t_hyd : float, default=0
        Hydrate formation time (s).  Default value is zero, which indicates 
        that particles are dirty or hydrated immediately upon formation.

    Returns
    -------
    disp_phases : list of bent_plume_model.Particle objects
        List of `bent_plume_model.Particle` objects to be added to the
        present bent plume model simulation based on the given input data.

    Notes
    -----
    See the documentation for the `bent_plume_model` for more
    information on the `Particle` object.

    """
    # Create an empty list of particles
    disp_phases = []

    # Add each particle in the distribution separately
    for i in range(len(d)):

        # Get the total mass flux of this fluid phase for the present
        # particle size
        mb0 = vf[i] * m_tot

        # Get the properties of these particles at the source
        (m0, T0, nb0, P, Sa, Ta) = dispersed_phases.initial_conditions(
            profile, z0, oil, yk, mb0, 2, d[i], Tj)

        # Append these particles to the list of particles in the simulation
        disp_phases.append(bent_plume_model.Particle(x0, y0, z0, oil, m0, T0,
            nb0, lambda_1, P, Sa, Ta, K=1., K_T=1., fdis=1.e-6, t_hyd=t_hyd,
            lag_time=lag_time))

    # Return the list of particles
    return disp_phases


def get_ambient_profile(water, current, **kwargs):
    """
    Create an `ambient.Profile` object from the given ambient data

    Based on the water column information provided, make an appropriate
    choice and create the `ambient.Profile` object required for a `tamoc`
    simulation.

    Parameters
    ----------
    water : various
        Data describing the ambient water temperature and salinity profile.
        See Notes below for details.
    current : various
        Data describing the ambient current velocity profile.  See Notes
        below for details.
    **kwargs : dict
        Dictionary of optional keyword arguments that can be used when
        creating an ambient.Profile object from a text file.  Optional
        arguments include:

        summary : str
            String describing the simulation for which this data will be used.
        source : str
            String documenting the source of the ambient ocean data provided.
        sea_name : str
            NC-compliant name for the ocean water body as a string.
        p_lat : float
            Latitude (deg)
        p_lon : float
            Longitude, negative is west of 0 (deg)
        p_time : netCDF4 time format
            Date and time of the CTD data using netCDF4.date2num().
        ca : list, default=[]
            List of dissolved atmospheric gases to include in the ambient
            ocean data as a derived concentration; choices are 'nitrogen',
            'oxygen', 'argon', and 'carbon_dioxide'.

        If any of these arguments are not passed, default values will be
        assigned by this function.

    Notes
    -----
    The `water` variable contains information about the ambient temperature
    and salinity profile.  Possible choices for `water` include the following:

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
    water : 'netCDF4.Dataset' object
        If a 'netCDF4.Dataset' object already contains the ambient CTD
        data in a format appropriate for the `ambient.Profile` object, then
        this can be passed.  In this case, it is assumed that the dataset
        includes the currents; hence, the `currents` variable will be
        ignored.
    water : `ambient.Profile` object
        If we already created our own ambient Profile object, then this
        object can be used directly.
    water = str
        If we stored the water column profile in a file, we may provide the
        file path to this file via the string stored in water. If this string
        ends in '.nc', it is assumed that this file contains a netCDF4
        dataset. Otherwise, this file should contain columns in the following
        order: depth (m), temperature (deg C), salinity (psu), velocity in
        the x-direction (m/s), velocity in the y-direction (m/s). Since this
        option includes the currents, the current variable will be ignored in
        this case. A comment string of `#` may be used in the text file.

    The `current` variable contains information about the ambient current
    profile.  Possible choices for `current` include the following:

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

    """
    NoneType = type(None)
    done = False
    
    # Extract the temperature and salinity data
    if isinstance(water, NoneType):

        # Use the world-ocean average T(z) and S(z)
        data = None

    if isinstance(water, dict):

        # Get the water temperature and salinity at the surface
        Ts = water['temperature']
        Ss = water['salinity']

        # Create a data array of depth, temperature, and salinity
        data = np.array([0., Ts, Ss])

    if isinstance(water, Dataset):

        # A netCDF4 Dataset containing all of the profile data is stored
        # in water.  Use that to create the Profile object
        profile = ambient.Profile(water, chem_names='all')
        done = True

    if isinstance(water, ambient.Profile):
        
        # An ambient.Profile object has been provided; use that profile
        profile = water
        done = True

    elif isinstance(water, str) or isinstance(water, unicode):

        if water[-3:] == '.nc':

            # Water contains a path to a netCDF4 dataset.  Use this to
            # create the Profile object
            profile = ambient.Profile(water, chem_names='all')
            done = True

        else:

            # This must be a relative path to a text file
            fname = water
            x0 = np.array([0., 0.])
            try:
                ca = kwargs['ca']
            except:
                ca = []
            try:
                summary = kwargs['summary']
            except:
                summary = 'CTD text file stored in:  ' + water
            try:
                source = kwargs['source']
            except:
                source = 'tamoc.blowout.get_ctd_from_txt()'
            try:
                sea_name = kwargs['sea_name']
            except:
                sea_name = 'Text File'
            try:
                p_lon = kwargs['p_lon']
            except:
                p_lon = x0[0]
            try:
                p_lat = kwargs['p_lat']
            except:
                p_lat = x0[1]
            try:
                p_time = kwargs['p_time']
            except:
                p_time = date2num(datetime.now(),
                         units = 'seconds since 1970-01-01 00:00:00 0:00',
                         calendar = 'julian')

            profile = get_ctd_from_txt(fname, summary, source,
                                       sea_name, p_lat, p_lon, p_time,
                                       ca)
            done = True

    # Create the `ambient.Profile` object
    if not done:
        profile = ambient.Profile(data, current=current, current_units='m/s')

    # Returen the profile
    return profile


def get_ctd_from_txt(fname, summary, source, sea_name, p_lat, p_lon,
    p_time, ca=[]):
    """
    Create an ambient.Profile object from a text file of ocean property data

    Read the CTD and current data in the given filename (fname) and use that
    data to create an ambient.Profile object for use in TAMOC. This function
    is built to work with an ascii file organized with data stored in columns
    that report depth (m), temperature (deg C), salinity (psu), u-component
    of velocity (m/s) and v-component of velocity (m/s).

    Parameters
    ----------
    fname : str
        String containing the relative path to the water column data file.
    summary : str
        String describing the simulation for which this data will be used.
    source : str
        String documenting the source of the ambient ocean data provided.
    sea_name : str
        NC-compliant name for the ocean water body as a string.
    p_lat : float
        Latitude (deg)
    p_lon : float
        Longitude, negative is west of 0 (deg)
    p_time : netCDF4 time format
        Date and time of the CTD data using netCDF4.date2num().
    ca : list, default=[]
        List of dissolved atmospheric gases to include in the ambient ocean
        data as a derived concentration; choices are 'nitrogen', 'oxygen',
        'argon', and 'carbon_dioxide'.

    Returns
    -------
    profile : ambient.Profile
        Returns an ambient.Profile object for manipulating ambient water
        column data in TAMOC.

    """
    # Read in the data
    data = np.loadtxt(fname, comments='#')

    # Describe what should be stored in this dataset
    units = ['m', 'deg C', 'psu', 'm/s', 'm/s']
    labels = ['z', 'temperature', 'salinity', 'ua', 'va']
    comments = ['modeled', 'modeled', 'modeled', 'modeled', 'modeled']

    # Extract a file name for the netCDF4 dataset that will hold this data
    # based on the name of the text file.
    nc_name = '.'.join(fname.split('.')[:-1])  # remove text file .-extension
    nc_name = nc_name + '.nc'

    # Create the ambient.Profile object
    profile = create_ambient_profile(data, labels, units, comments, nc_name,
        summary, source, sea_name, p_lat, p_lon, p_time, ca)

    return profile


def create_ambient_profile(data, labels, units, comments, nc_name, summary,
    source, sea_name, p_lat, p_lon, p_time, ca=[]):
    """
    Create an ambient Profile object from given data

    Create an ambient.Profile object using the given CTD and current data.
    This function performs some standard operations to this data (unit
    conversion, computation of pressure, insertion of concentrations for
    dissolved gases, etc.) and returns the working ambient.Profile object.
    The idea behind this function is to separate data manipulation and
    creation of the ambient.Profile object from fetching of the data itself.

    Parameters
    ----------
    data : np.array
        Array of the ambient ocean data to write to the CTD file.  The
        contents and dimensions of this data are specified in the labels
        and units lists, below.
    labels : list
        List of string names of each variable in the data array.
    units : list
        List of units as strings for each variable in the data array.
    comments : list
        List of comments as strings that explain the types of data in the
        data array.  Typical comments include 'measured', 'modeled', or
        'computed'.
    nc_name : str
        String containing the file path and file name to use when creating
        the netCDF4 dataset that will contain this data.
    summary : str
        String describing the simulation for which this data will be used.
    source : str
        String documenting the source of the ambient ocean data provided.
    sea_name : str
        NC-compliant name for the ocean water body as a string.
    p_lat : float
        Latitude (deg)
    p_lon : float
        Longitude, negative is west of 0 (deg)
    p_time : netCDF4 time format
        Date and time of the CTD data using netCDF4.date2num().
    ca : list, default=[]
        List of gases for which to compute a standard dissolved gas profile;
        choices are 'nitrogen', 'oxygen', 'argon', and 'carbon_dioxide'.

    Returns
    -------
    profile : ambient.Profile
        Returns an ambient.Profile object for manipulating ambient water
        column data in TAMOC.

    """
    # Convert the data to standard units
    data, units = ambient.convert_units(data, units)

    # Create an empty netCDF4-classic datast to store this CTD data
    nc = ambient.create_nc_db(nc_name, summary, source, sea_name, p_lat,
                              p_lon, p_time)

    # Put the CTD and current profile data into the ambient netCDF file
    nc = ambient.fill_nc_db(nc, data, labels, units, comments, 0)

    # Compute and insert the pressure data
    z = nc.variables['z'][:]
    T = nc.variables['temperature'][:]
    S = nc.variables['salinity'][:]
    P = ambient.compute_pressure(z, T, S, 0)
    P_data = np.vstack((z, P)).transpose()
    nc = ambient.fill_nc_db(nc, P_data, ['z', 'pressure'], ['m', 'Pa'],
                            ['measured', 'computed'], 0)

    # Use this netCDF file to create an ambient object
    profile = ambient.Profile(nc, ztsp=['z', 'temperature', 'salinity',
                 'pressure', 'ua', 'va'])

    # Compute dissolved gas profiles to add to this dataset
    if len(ca) > 0:
        
        profile.add_computed_gas_concentrations()
    
        # Create a gas mixture object for air
        #gases = ['nitrogen', 'oxygen', 'argon', 'carbon_dioxide']
        #air = dbm.FluidMixture(gases)
        #yk = np.array([0.78084, 0.20946, 0.009340, 0.00036])
        #m = air.masses(yk)

        # Set atmospheric conditions
        #Pa = 101325.

        # Compute the desired concentrations
        #for i in range(len(ca)):

            # Initialize a dataset of concentration data
            #conc = np.zeros(len(profile.z))

            # Compute the concentrations at each depth
            #for j in range(len(conc)):

                # Get the local water column properties
                #T, S, P = profile.get_values(profile.z[j], ['temperature',
                #   'salinity', 'pressure'])

                # Compute the gas solubility at this temperature and salinity
                # at the sea surface
                #Cs = air.solubility(m, T, Pa, S)[0,:]

                # Adjust the solubility to the present depth
                #Cs = Cs * seawater.density(T, S, P) / \
                #    seawater.density(T, S, 101325.)

                # Extract the right chemical
                #conc[j] = Cs[gases.index(ca[i])]

            # Add this computed dissolved gas to the Profile dataset
            #data = np.vstack((profile.z, conc)).transpose()
            #symbols = ['z', ca[i]]
            #units = ['m', 'kg/m^3']
            #comments = ['measured', 'computed from CTD data']
            #profile.append(data, symbols, units, comments, 0)

    # Close the netCDF dataset
    profile.close_nc()

    # Return the profile object
    return profile


