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
behavior of the particle during the simulation. An interface to the `dbm`
objects is provided by the Particle class objects defined in
`dispersed_phases`.

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

    d H / dt = sum (d (m_i) /dt * dH_solR_i * Ru / M_i)

where `dH_solR` is the enthalpy of solution divided by the universal gas
constant (`Ru`) and `M_i` is the molecular weight of constituent `i`.

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
and shape.  However, there is no mechanism in this module to allow a droplet
to break up into multiple bubbles.

"""
# S. Socolofsky, November 2014, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import model_share
from tamoc import seawater
from tamoc import ambient
from tamoc import dbm
from tamoc import dispersed_phases

from netCDF4 import Dataset
from datetime import datetime

import numpy as np
# mpl imports moved to plotting functions
# import matplotlib.pyplot as plt
# import matplotlib as mpl
from copy import copy
from scipy import integrate
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
    particle : `dispersed_phases.SingleParticle` object
        Interface to the `dbm` module and container for particle-specific
        parameters
    t : ndarray
        Times (s) associated with the state space
    y : ndarray
        State space along the trajectory of the particle
    z0 : float
        The release depth (m)
    x0 : float, default = 0.
        The release x-coordinate (m)
    y0 : float, default = 0.
        The release y-coordinate (m)
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
        Remainder fraction that turns off dissolution for each component (--)
    delta_t : float, default = 0.1 s
        Maximum time step to use (s) in the simulation output

    See Also
    --------
    simulate, save_sim, load_sim

    Notes
    -----
    The `Model` object will be initialized either with the `profile` data
    making it ready to start a new simulation or with the results of a
    previous simulation stored in `simfile`.

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

    def simulate(self, particle, X0, de, yk, T0=None, K=1., K_T=1.,
                 fdis=1.e-6, t_hyd=0., lag_time=True, delta_t=0.1):
        """
        Simulate the trajectory of a particle from given initial conditions

        Simulate the trajectory of a particle (bubble, droplet or solid
        particle) until the particle dissolves or until it reaches the free
        surface.

        Parameters
        ----------
        particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
            Object describing the properties and behavior of the particle.
        X0 : float or ndarray
            The release localtion (x0, y0, z0) depth (m) of the particle in
            the simulation.  If float, x0 = y0 = 0 is assumed.
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
        t_hyd : float, default = 0.
            Hydrate film formation time (s).  Mass transfer is computed by clean
            bubble methods for t less than t_hyd and by dirty bubble methods
            thereafter.  The default behavior is to assume the particle is dirty
            or hydrate covered from the release.
        lag_time : bool, default = True
            flag indicating whether the biodegradation rates should include
            a lag time (True) or not (False).  Default value is True.
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
        # Check the initial position and make it an array.
        if not isinstance(X0, np.ndarray):
            if not isinstance(X0, list):
                X0 = np.array([0., 0., X0])
            else:
                X0 = np.array(X0)

        # Make sure yk is an ndarray
        if not isinstance(yk, np.ndarray):
            if not isinstance(yk, list):
                yk = np.array([yk])
            else:
                yk = np.array(yk)

        # Check if the right number of elements are in yk
        if len(yk) != len(particle.composition):
            print('Wrong number of mole fractions:')
            print('   yk : %d entries' % len(yk))
            print('   composition : %d components\n' %
                                    len(particle.composition))
            return

        # Save the input variables that are not part of the self.particle
        # object
        self.K_T0 = K_T
        self.delta_t = delta_t

        # Get the initial conditions for the simulation run
        (self.particle, y0) = sbm_ic(self.profile, particle, X0, de, yk, T0,
                                     K, K_T, fdis, t_hyd, lag_time)

        # Open the simulation module
        print('\n-- TEXAS A&M OIL-SPILL CALCULATOR (TAMOC) --')
        print('-- Single Bubble Model                    --\n')

        # Calculate the trajectory
        print('Calculate the trajectory...')
        self.t, self.y = calculate_path(self.profile, self.particle, self.p,
                                        y0, delta_t)
        print('Simulation complete.\n ')
        self.sim_stored = True

        # Restart heat transfer
        self.particle.K_T = self.K_T0

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
            print('No simulation results to store...')
            print('Saved nothing to netCDF file.\n')
            return

        # Create the netCDF dataset object
        title = 'Simulation results for the TAMOC Single Bubble Model'
        nc = model_share.tamoc_nc_file(fname, title, profile_path, profile_info)

        # Create variables for the dimensions
        z = nc.createDimension('z', None)
        p = nc.createDimension('profile', 1)
        ns = nc.createDimension('ns', len(self.y[0,:]))

        # Create variables for the model initial conditions
        K_T0 = nc.createVariable('K_T0', 'f8', ('profile',))
        K_T0.long_name = 'Initial heat transfer reduction factor'
        K_T0.standard_name = 'K_T0'
        K_T0.units = 'nondimensional'

        delta_t = nc.createVariable('delta_t', 'f8', ('profile',))
        delta_t.long_name = 'maximum simulation output time step'
        delta_t.standard_name = 'delta_t'
        delta_t.units = 'seconds'

        # Create variables for the independent variable
        t = nc.createVariable('t', 'f8', ('z',))
        t.long_name = 'time coordinate'
        t.standard_name = 'time'
        t.units = 'seconds since release'
        t.axis = 'T'

        # Create variables for the state space
        y = nc.createVariable('y', 'f8', ('z', 'ns',))
        y.long_name = 'solution state space'
        y.standard_name = 'y'
        y.units = 'variable'
        y.coordinate = 't'

        # Store the initial conditions and model setup
        K_T0[0] = self.K_T0
        delta_t[0] = self.delta_t

        # Store the dbm particle object
        dispersed_phases.save_particle_to_nc_file(nc,
            self.particle.composition, self.particle, self.K_T0)

        # Save the model simulation result
        t[:] = self.t
        for i in range(len(nc.dimensions['ns'])):
            y[0:len(self.t),i] = self.y[:,i]

        # Close the netCDF dataset
        nc.close()

    def save_txt(self, base_name, profile_path, profile_info):
        """
        Save the state space in ascii text format for exporting

        Save the state space (dependent and independent variables) in an
        ascii text file for exporting to other programs (e.g., Matlab).

        Parameters
        ----------
        base_name : str
            The main name of the output file.  This method writes two files:
            the data are stored in base_name.txt, and the header information
            describing each row of data are saved in base_name_header.txt.
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
            print('No simulation results to store...')
            print('Saved nothing to txt file.\n')
            return

        # Create the header string that contains the column descriptions
        p_list = ['Single Bubble Model ASCII Output File \n']
        p_list.append('Created: ' + datetime.today().isoformat(' ') + '\n\n')
        p_list.append('Simulation based on CTD data in:\n')
        p_list.append(profile_path)
        p_list.append('\n\n')
        p_list.append(profile_info)
        p_list.append('\n\n')
        p_list.append('Column Descriptions:\n')
        p_list.append('    0:  Time in s\n')
        p_list.append('    1:  x-coordinate in m\n')
        p_list.append('    2:  y-coordinate in m\n')
        p_list.append('    3:  Depth in m\n')
        for i in range(len(self.particle.composition)):
            p_list.append('    %d:  Mass of %s in particle in kg\n' % \
                          (i+4, self.particle.composition[i]))
        p_list.append('    %d:  Heat content (m_p * cp * T) in J\n' % (i+5))
        header = ''.join(p_list)

        # Assemble and write the output data
        data = np.hstack((np.atleast_2d(self.t).transpose(), self.y))
        np.savetxt(base_name + '.txt', data)
        with open(base_name + '_header.txt', 'w') as txt_file:
            txt_file.write(header)

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
        self.profile = model_share.profile_from_model_savefile(nc, fname)
        if self.profile is not None:
            self.p = ModelParams(self.profile)
        else:
            self.p = None

        # Load in the dispersed_phases.Particle object
        self.particle = \
            dispersed_phases.load_particle_from_nc_file(nc)[0][0]

        # Extract the state space data
        self.t = nc.variables['t'][:]
        ns = len(nc.dimensions['ns'])
        self.y = np.zeros((len(self.t), ns))
        for i in range(ns):
            self.y[:,i] = nc.variables['y'][0:len(self.t), i]

        # Extract the initial conditions
        self.K_T0 = nc.variables['K_T0'][0]
        self.delta_t = nc.variables['delta_t'][0]

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
            print('No simulation results to plot...')
            print('Plotting nothing.\n')
            return

        # Plot the results
        print('Plotting the results...')
        plot_state_space(self.profile, self.particle, self.p, self.t,
                         self.y, fig)
        print('Done.\n')


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
    g : float
        Acceleration of gravity (m/s^2)
    Ru : float
        Ideal gas constant (J/mol/K)

    """
    def __init__(self, profile):
        super(ModelParams, self).__init__()

        # Store a reference density for the water column
        z_ave = profile.z_max - (profile.z_max - profile.z_min) / 2.
        T, S, P = profile.get_values(z_ave, ['temperature', 'salinity',
                                     'pressure'])
        self.rho_r = seawater.density(T, S, P)

        # Store some physical constants
        self.g = 9.81
        self.Ru = 8.314510


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
    particle : `LagrangianParticle` object
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
        >>> z = y[2]
        >>> m = y[3:-1]
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
    psteps = 10.
    stop = False
    while r.successful() and not stop:

        # Print progress to the screen
        m0 = np.sum(y[0][3:-1])
        mt = np.sum(y[-1][3:-1])
        f = mt / m0
        if np.remainder(np.float(k), psteps) == 0.:
            print('    Depth:  %g (m), t:  %g (s), k: %d, f: %g (--)' %
                (r.y[2], t[-1], k, f))

        # Perform one step of the integration
        r.integrate(t[-1] + delta_t, step=True)

        # Store the results
        if particle.K_T == 0:
            # Make the state-space heat correct
            Ta = profile.get_values(r.y[2], 'temperature')
            r.y[-1] = np.sum(r.y[3:-1]) * particle.cp * Ta
        for i in range(len(r.y[3:-1])):
            if r.y[i+3] < 0.:
                # Concentration should not overshoot zero
                r.y[i+3] = 0.
        t.append(r.t)
        y.append(r.y)
        k += 1

        # Evaluate stop criteria
        if r.successful():
            # Check if bubble dissolved (us = 0 or based on fdis) or reached 
            # the free surface
            us = - (y[-2][2] - y[-1][2]) / (t[-2] - t[-1])
            if r.y[2] <= profile.z_min or us <= 0. or f < particle.fdis:
                stop = True
            if k > 300000:
                stop = True
            if t[-1] > 1209600:
                # Particle has reached 14 days of simulation
                stop = True

    # Remove any negative depths due to overshooting the free surface
    t = np.array(t)
    y = np.array(y)
    rows = y[:,2] >= 0
    t = t[rows]
    y = y[rows,:]

    # Return the solution
    print('    Depth:  %g (m), t:  %g (s), k: %d' %
        (y[-1,2], t[-1], k))
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
    particle : `LagrangianParticle` object
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
    z = y[2]
    m = y[3:-1]
    T = y[-1] / (np.sum(m) * particle.cp)

    # Get the ambient profile data
    Ta, Sa, P = profile.get_values(z, ['temperature', 'salinity', 'pressure'])
    ua, va, wa = profile.get_values(z, ['ua', 'va', 'wa'])
    C = profile.get_values(z, particle.composition)

    # Get the physical particle properties
    (us, rho_p, A, Cs, beta, beta_T, T) = particle.properties(m, T, P, Sa,
                                                              Ta, t)
    
    # Get the biodegradation rate constants
    k_bio = particle.biodegradation_rate(t)
    
    # Advection
    yp[0] = ua
    yp[1] = va
    yp[2] = -us - wa

    # Dissolution
    if len(Cs) > 0:
        md_diss = - A * beta[:] * (Cs[:] - C[:])
    else:
        md_diss = np.array([0.])

    # Biodegradation
    md_biodeg = -k_bio * m
    yp[3:-1] = md_diss + md_biodeg

    # Account for heat transfer (ignore heat of solution since it is
    # negligible in the beginning as the particle approaches equilibrium)
    yp[-1] =  - rho_p * particle.cp * A * beta_T * (T - Ta)

    # Account for heat lost due to decrease in mass
    yp[-1] += particle.cp * np.sum(md_diss + md_biodeg) * T

    # Return the derivatives
    return yp


def sbm_ic(profile, particle, X0, de, yk, T0, K, K_T, fdis, t_hyd, lag_time):
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
    X0 : ndarray
        The release location (x, y, y) in m of the particle in the simulation
    de : float
        Initial diameter of the particle (m)
    yk : ndarray
        Initial mole fractions of each component in the particle (--)
    T0 : float, optional
        Initial temperature (K) of the particle at release if not equal
        to the temperature of the surrounding fluid.  If omitted, the
        model will set T0 to the ambient temperature.
    K : float
        Mass transfer reduction factor (--). Pre-multiplies the mass
        transfer coefficients providing amplification (>1) or retardation
        (<1) of the dissolution.
    K_T : float
        Heat transfer reduction factor (--). Pre-multiplies the heat
        transfer coefficient providing amplification (>1) or retardation
        (<1) of the heat flux.
    fdis : float
        Fraction of the initial total mass (--) remaining when the
        particle should be considered dissolved.
    t_hyd : float
        Hydrate film formation time (s).  Mass transfer is computed by clean
        bubble methods for t less than t_hyd and by dirty bubble methods
        thereafter.  The default behavior is to assume the particle is dirty
        or hydrate covered from the release.

    Returns
    -------
    particle : `LagrangianParticle` object
        A `LagrangianParticle` object containing a unified interface to the
        `dbm` module and the particle-specific model parameters (e.g., mass
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
    # Get the particle initial conditions from the dispersed_phases module
    m0, T0, nb0, P, Sa, Ta = dispersed_phases.initial_conditions(profile,
        X0[2], particle, yk, None, 0, de, T0)

    # Initialize a LagrangianParticle object
    particle = dispersed_phases.SingleParticle(particle, m0, T0, K, K_T,
               fdis, t_hyd, lag_time)

    # Assemble the state space
    y0 = np.hstack((X0, m0, T0 * np.sum(m0) * particle.cp))

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
    particle : `LagrangianParticle` object
        Object describing the properties and behavior of the particle.
    p : `ModelParams` object
        Collection of model parameters passed to `derivs`.
    t : ndarray
        Times (s) associated with the state space for the trajectory of the
        particle
    y : ndarray
        State space along the trajectory of the particle.  The state space
        includes the location (m), masses (kg) of the particle components, and
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
    # imported here so MPL will only be imported if you need it.
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Extract the state space variables

    xi = y[:,0]
    yi = y[:,1]
    zi = y[:,2]
    m = y[:,3:-1]
    T = np.array([y[i,-1] / (np.sum(m[i,:]) * particle.cp)
                 for i in range(len(zi))])

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
        Ta[i], Sa[i], P[i] = profile.get_values(zi[i], ['temperature',
                             'salinity', 'pressure'])
        N[i] = profile.buoyancy_frequency(zi[i], h=0.005)
        (us[i], rho_p[i], A[i], Cs_local, beta_local, beta_T, T_fun[i]) = \
            particle.properties(m[i,:], T[i], P[i], Sa[i], Ta[i], t[i])
        if len(Cs_local) > 0:
            Cs[i,:] = Cs_local
            beta[i,:] = beta_local
        shape[i] = particle.particle.particle_shape(m[i,:], T[i], P[i],
                   Sa[i], Ta[i])[0]
        de[i] = particle.diameter(m[i,:], T[i], P[i], Sa[i], Ta[i])

    # Start by plotting the raw state space versus t
    plt.figure(fig)
    plt.clf()
    plt.show()

    # Depth
    ax1 = plt.subplot(221)
    ax1.plot(zi, t)
    ax1.set_xlabel('Depth (m)')
    ax1.set_ylabel('Time (s)')
    ax1.locator_params(tight=True, nbins=6)
    ax1.grid(True)

    # Slip Velocity
    ax2 = plt.subplot(222)
    ax2.plot(us, t)
    ax2.set_xlabel('Slip velocity (m/s)')
    ax2.locator_params(tight=True, nbins=6)
    ax2.grid(True)

    # Masses
    ax3 = plt.subplot(223)
    ax3.semilogx(m, t)
    ax3.set_xlabel('Component masses (kg)')
    ax3.locator_params(axis='y', tight=True, nbins=6)
    #ax3.xaxis.set_major_locator(mpl.ticker.LogLocator(base=1e2))
    ax3.grid(True)

    # Heat
    ax4 = plt.subplot(224)
    ax4.semilogx(y[:,-1], t)
    ax4.set_xlabel('Heat (J)')
    ax4.locator_params(axis='y', tight=True, nbins=6)
    #ax4.xaxis.set_major_locator(mpl.ticker.LogLocator(base=1e2))
    ax4.grid(True)

    plt.draw()

    # Plot derived variables related to diameter
    plt.figure(fig+1)
    plt.clf()
    plt.show()

    # Diameter
    ax1 = plt.subplot(221)
    ax1.semilogx(de * 1000, zi)
    ax1.set_xlabel('Diameter (mm)')
    ax1.set_ylabel('Depth (m)')
    ax1.locator_params(axis='y', tight=True, nbins=6)
    #ax1.xaxis.set_major_locator(mpl.ticker.LogLocator(base=1e2))
    ax1.invert_yaxis()
    ax1.grid(True)

    # Shape
    ax2 = plt.subplot(222)
    ax2.plot(shape, zi)
    ax2.set_xlabel('Shape (--)')
    ax2.set_xlim((0, 4))
    ax2.invert_yaxis()
    ax2.grid(which='major', axis='x')
    ax2.locator_params(tight=True, nbins=4)
    ax2.grid(True)

    # Density
    ax3 = plt.subplot(223)
    ax3.plot(rho_p, zi)
    ax3.set_xlabel('Density (kg)')
    ax3.set_ylabel('Depth (m)')
    ax3.invert_yaxis()
    ax3.locator_params(tight=True, nbins=6)
    ax3.grid(True)

    # Temperature
    ax4 = plt.subplot(224)
    ax4.plot(T, zi)
    ax4.plot(T_fun, zi)
    ax4.plot(Ta, zi)
    ax4.set_xlabel('Temperature (K)')
    ax4.invert_yaxis()
    ax4.locator_params(tight=True, nbins=6)
    ax4.grid(True)
    plt.draw()

    # Plot dissolution data
    plt.figure(fig+2)
    plt.clf()
    plt.show()

    # Masses
    ax1 = plt.subplot(221)
    ax1.semilogx(m, zi)
    ax1.set_xlabel('Component masses (kg)')
    ax1.set_ylabel('Depth (m)')
    ax1.locator_params(axis='y', tight=True, nbins=6)
    #ax1.xaxis.set_major_locator(mpl.ticker.LogLocator(base=1e2))
    ax1.invert_yaxis()
    ax1.grid(True)

    # Solubility
    ax2 = plt.subplot(222)
    ax2.plot(Cs, zi)
    ax2.set_xlabel('Solubility (kg/m^3)')
    ax2.locator_params(tight=True, nbins=6)
    ax2.invert_yaxis()
    ax2.grid(True)

    # Mass transfer coefficient
    ax3 = plt.subplot(223)
    ax3.plot(beta, zi)
    ax3.set_xlabel('Mass transfer (m/s)')
    ax3.invert_yaxis()
    ax3.locator_params(tight=True, nbins=6)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax3.grid(True)

    # Area
    ax4 = plt.subplot(224)
    ax4.semilogx(A, zi)
    ax4.set_xlabel('Surface area (m^2)')
    ax4.locator_params(axis='y', tight=True, nbins=6)
    #ax4.xaxis.set_major_locator(mpl.ticker.LogLocator(base=1e2))
    ax4.invert_yaxis()
    ax4.grid(True)

    plt.draw()

    # Plot dissolution data
    plt.figure(fig+3)
    plt.clf()
    plt.show()

    # CTD Temperature
    ax1 = plt.subplot(221)
    ax1.plot(Ta - 273.15, zi)
    ax1.set_xlabel('Temperature (deg C)')
    ax1.set_ylabel('Depth (m)')
    ax1.locator_params(tight=True, nbins=6)
    ax1.invert_yaxis()
    ax1.grid(True)

    ax2 = plt.subplot(222)
    ax2.plot(Sa, zi)
    ax2.set_xlabel('Salinity (psu)')
    ax2.locator_params(tight=True, nbins=6)
    ax2.invert_yaxis()
    ax2.grid(True)

    ax3 = plt.subplot(223)
    ax3.plot(P, zi)
    ax3.set_xlabel('Pressure (Pa)')
    ax3.set_ylabel('Depth (m)')
    ax3.locator_params(tight=True, nbins=6)
    ax3.invert_yaxis()
    ax3.grid(True)

    ax4= plt.subplot(224)
    ax4.plot(N, zi)
    ax4.set_xlabel('Buoyancy Frequency (1/s)')
    ax4.locator_params(tight=True, nbins=6)
    ax4.invert_yaxis()
    ax4.grid(True)

    plt.draw()

