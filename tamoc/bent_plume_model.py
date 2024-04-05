"""
Bent Plume Model
================

Simulate a buoyant plume in crossflowing ambient conditions

This module defines the classes, methods, and functions necessary to simulate
the buoyant plume behavior in crossflowing ambient conditions, where the
intrusion layer is not expected to interact with the rising stage of the
plume. The ambient water properties are provided through an `ambient.Profile`
class object, which contains a netCDF4-classic dataset of CTD data and the
needed interpolation methods. The `dbm` class objects `dbm.FluidParticle` and
`dbm.InsolubleParticle` report the properties of the dispersed phase during
the simulation, and these methods are provided to the model through the
objects defined in `dispersed_phases`.

This module manages the setup, simulation, and post-processing for the model.
The numerical solution is contained in the `lpm` module.

Notes
-----
This model is a Lagrangian plume integral model following the approach in
Lee and Cheung (1990) for single-phase plumes, updated using the shear
entrainment formulation in Jirka (2004), and adapted to multiphase plumes
following the methods of Johansen (2000, 2003) and Zheng and Yapa (1997).
Several modifications are made to make the model consistent with the approach
in Socolofsky et al. (2008) and to match the available validation data.

The model can run as a single-phase or multi-phase plume.  A single-phase
plume simply has an empty `particles` list.

See Also
--------
stratified_plume_model, single_bubble_model

"""
# S. Socolofsky, November 2014, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import model_share
from tamoc import ambient
from tamoc import seawater
from tamoc import dbm
from tamoc import single_bubble_model
from tamoc import dispersed_phases
from tamoc import lmp

from netCDF4 import Dataset
from datetime import datetime

import numpy as np
from numpy.linalg import inv
from scipy.optimize import fsolve

# ----------------------------------------------------------------------------
# Main Model object
# ----------------------------------------------------------------------------

class Model(object):
    """
    Master class object for controlling and post-processing the simulation

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
    profile : `ambient.Profile`
        Ambient CTD data
    got_profile : bool
        Flag indicating whether or not the profile object was successfully
        loaded into the `Model` object memory
    p : `ModelParams`
        Container for the fixed model parameters
    sim_stored : bool
        Flag indicating whether or not a simulation result is stored in the
        object memory
    X : ndarray
        Release location (x, y, z) in (m)
    D : float
        Diameter for the equivalent circular cross-section of the release (m)
    Vj : float
        Scalar value of the magnitude of the discharge velocity for continuous
        phase fluid in the discharge.  This variable should be 0. or None for
        a pure multiphase discharge.
    phi_0 : float
        Vertical angle from the horizontal for the discharge orientation
        (rad in range +/- pi/2)
    theta_0 : float
        Horizontal angle from the x-axis for the discharge orientation.  The
        x-axis is taken in the direction of the ambient current.  (rad in
        range 0 to 2 pi)
    Sj : float
        Salinity of the continuous phase fluid in the discharge (psu)
    Tj : float
        Temperature of the continuous phase fluid in the discharge (K)
    cj : ndarray
        Concentration of passive tracers in the discharge (user-defined)
    tracers : string list
        List of passive tracers in the discharge.  These can be chemicals
        present in the ambient `profile` data, and if so, entrainment of these
        chemicals will change the concentrations computed for these tracers.
        However, none of these concentrations are used in the dissolution of
        the dispersed phase.  Hence, `tracers` should not contain any
        chemicals present in the dispersed phase particles.
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the
        simulation
    track : bool
        Flag indicating whether or not to track the particle through
        the water column using the `single_bubble_model`.
    dt_max : float
        Maximum step size to take in the storage of the simulation solution
        (s)
    sd_max : float
        Maximum number of orifice diameters to compute the solution along
        the plume centerline (m/m)
    K_T0 : ndarray
        Array of heat transfer reduction factors for the particle objects
        which is used to restart heat transfer after the simulation ends
        since during simulation, heat transfer is turned off.
    chem_names : string list
        List of chemical parameters to track for the dissolution.  Only the
        parameters in this list will be used to set background concentration
        for the dissolution, and the concentrations of these parameters are
        computed separately from those listed in `tracers` or inputed from
        the discharge through `cj`.
    q_local : `LagElement` object
        Object that translates the `Model` state space `t` and `q` into the
        comprehensive list of derived variables.
    t : ndarray
        Array of times computed in the solution (s)
    q : ndarray
        Array of state space values computed in the solution

    See Also
    --------
    simulate, save_sim, save_txt, load_sim, plot_state_space,
    plot_all_variables

    """
    def __init__(self, profile=None, simfile=None):
        super(Model, self).__init__()

        if profile is None:
            # Create a Model object from a save file
            self.load_sim(simfile)

        else:
            # Create a new Model object
            self.profile = profile
            self.got_profile = True
            profile.close_nc()

            # Set the model parameters that the user cannot adjust
            self.p = ModelParams(self.profile)

            # Indicate that the simulation has not yet been conducted
            self.sim_stored = False

    def simulate(self, X, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers,
                 particles=[], track=False, dt_max=60., sd_max=350.):
        """
        Simulate the plume dynamics from given initial conditions

        Simulate the buoyant plume using a Lagrangian plume integral model
        approach until the plume reaches the surface, the integration
        exceeds the given s/D (`sd_max`), or the intrusion reaches a point of
        neutral buoyancy.

        Parameters
        ----------
        X : ndarray
            Release location (x, y, z) in (m)
        D : float
            Diameter for the equivalent circular cross-section of the release
            (m)
        Vj : float
            Scalar value of the magnitude of the discharge velocity for
            continuous phase fluid in the discharge.  This variable should be
            0. or None for a pure multiphase discharge.
        phi_0 : float
            Vertical angle from the horizontal for the discharge orientation
            (rad in range +/- pi/2)
        theta_0 : float
            Horizontal angle from the x-axis for the discharge orientation.
            The x-axis is taken in the direction of the ambient current.
            (rad in range 0 to 2 pi)
        Sj : float
            Salinity of the continuous phase fluid in the discharge (psu)
        Tj : float
            Temperature of the continuous phase fluid in the discharge (T)
        cj : ndarray
            Concentration of passive tracers in the discharge (user-defined)
        tracers : string list
            List of passive tracers in the discharge.  These can be chemicals
            present in the ambient `profile` data, and if so, entrainment of
            these chemicals will change the concentrations computed for these
            tracers.  However, none of these concentrations are used in the
            dissolution of the dispersed phase.  Hence, `tracers` should not
            contain any chemicals present in the dispersed phase particles.
        particles : list of `Particle` objects
            List of `Particle` objects describing each dispersed phase in the
            simulation
        track : bool
            Flag indicating whether or not to track the particle through
            the water column using the `single_bubble_model`.
        dt_max : float
            Maximum step size to take in the storage of the simulation
            solution (s)
        sd_max : float
            Maximum number of orifice diameters to compute the solution along
            the plume centerline (m/m)

        """
        # Make sure the position is an array
        if not isinstance(X, np.ndarray):
            if not isinstance(X, list):
                # Assume user specified the depth only
                X = np.array([0., 0., X])
            else:
                X = np.array(X)

        # Make sure the tracer data are in an array
        if not isinstance(cj, np.ndarray):
            if not isinstance(cj, list):
                cj = np.array([cj])
            else:
                cj = np.array(cj)
        if not isinstance(tracers, list):
            tracers = [tracers]

        # Store the input parameters
        self.X = X
        self.D = D
        self.Vj = Vj
        self.phi_0 = phi_0
        self.theta_0 = theta_0
        self.Sj = Sj
        self.Tj = Tj
        self.cj = cj
        self.tracers = tracers
        self.particles = particles
        self.track = track
        self.dt_max = dt_max
        self.sd_max = sd_max
        self.K_T0 = np.array([self.particles[i].K_T for i in
                              range(len(self.particles))])
        if len(self.particles) > 0:
            self.composition = self.particles[0].particle.composition
        else:
            self.composition = []

        # Create the initial state space from the given input variables
        t0, q0, self.chem_names = lmp.main_ic(self.profile,
            self.particles, self.X, self.D, self.Vj, self.phi_0,
            self.theta_0, self.Sj, self.Tj, self.cj, self.tracers, self.p)

        # Store the initial conditions in a Lagrangian element object
        self.q_local = LagElement(t0, q0, D, self.profile, self.p,
                       self.particles, self.tracers, self.chem_names)
            
        # Compute the buoyant jet trajectory
        print('\n-- TEXAS A&M OIL-SPILL CALCULATOR (TAMOC) --')
        print('-- Bent Plume Model                       --\n')
        self.t, self.q, = lmp.calculate(t0, q0, self.q_local, self.profile,
            self.p, self.particles, lmp.derivs, self.dt_max, self.sd_max)

        # Track the particles
        if self.track:
            for i in range(len(self.particles)):
                # Initialize flag indicating whether the particle was tracked
                self.particles[i].tracked = False
                
                if particles[i].integrate is False and particles[i].z > 0.:
                    print('\nTracking Particle %d of %d:' %
                        (i+1, len(self.particles)))
                    particles[i].run_sbm(self.profile)
                    particles[i].tracked = True
                
                # Code below forces tracking of all particles, including
                # those trapped in the intrusion layer
                elif particles[i].z > 0:
                    # Update information at the end of the near-field
                    particles[i].te = particles[i].t
                    particles[i].xe = particles[i].x
                    particles[i].ye = particles[i].y
                    particles[i].ze = particles[i].z
                    # Track the particles
                    print('\nTracking Plume Particle %d of %d:' %
                        (i+1, len(self.particles)))
                    particles[i].run_sbm(self.profile)
                    self.particles[i].tracked = True

        # Update the status of the solution
        self.sim_stored = True

        # Update the status of the particles
        for i in range(len(self.particles)):
            self.particles[i].sim_stored = True
            self.particles[i].K_T = self.K_T0[i]
    
    def get_intrusion_initial_condition(self):
        """
        Extract the concentrations of dissolved compounds entering the intrusion
        
        Extract the concentration of dissolved compounds in seawater at the end
        of the near-field simulation.  This represents the concentrations of 
        dissolved material entering the intrusion layer.
        
        Returns
        -------
        Cp : ndarray
            Array of dissolved concentrations (kg/m^3) for each compound in the
            model composition at the end of the near-field plume simulation
        z0 : float
            Depth of the intrusion centerline (m)
        h : float
            Total thickness of the intrusion layer (m)
        L : float
            Total width of the intrusion layer (m)
        
        Notes
        -----
        To predict the geometry of the intrusion layer (depth and width), we
        use thickness equation from Socolofsky et al. (2011, GRL) and 
        conservation of mass.
        
        """
        # Update the LagElement at the end of the near-field plume
        self.q_local.update(self.t[-1], self.q[-1], self.profile, self.p,
            self.particles)
        
        # Extract the concentrations
        Cp = self.q_local.c_chems
        
        # Depth of the plume centerline
        z0 = self.q_local.z
        if z0 <= 0.:
            # The plume reached the surface...get data just below
            z0 = 0.1
        
        # Use the intrusion formation equations from Socolofsky et al.
        # (GRL, 2011).
        N = self.profile.buoyancy_frequency(z0)
        U = self.q_local.hvel
        Q = np.pi * self.q_local.b**2 * self.q_local.V
        
        # The intrusion is assumed to travel with the mean currents
        ua, va = self.profile.get_values(z0, ['ua', 'va'])
        Ua = np.sqrt(ua**2 + va**2)
        
        # Estimate the total intrusion thickness after buoyant collapse using
        # the equation from Akar and Jirka (1995) as defined in Socolofsky 
        # et al. (2011)
        h = 2.4 * Ua / N    # total thickness
        
        # Compute the intrusion layer width to preserve the given mass flux
        # and velocity
        if h > 2. * self.q_local.b:
            # Thickness is predicted to be greater than plume width -- ensure
            # width is preserved
            L = 2. * self.q_local.b
            h = Q / (Ua * L)
        else:
            # Thickness is less than plume width -- this is expected, so 
            # compute width to preserve mass
            L = Q / (Ua * h) 
        
        # Return the results
        return (Cp, z0, h, L)
    
    def get_intrusion_concentration(self, x, max_C=True):
        """
        Compute the concentrations of dissolved compounds in the intrusion
        
        Computes the concentrations of dissolved compounds in the intrusion
        layer using the wastewater outfall solution in Chin (2013), pp. 325ff.
        
        Parameters
        ----------
        x : ndarray
            Array of three-dimensional positions where the concentrations should
            be computed. Each row of `x` corresponds to a different point, with
            the columns of `x` giving the x-, y-, and z-coordinates of each 
            point.  The x-axis is aligned along the plume centerline, with the
            y-axis transverse in the horizontal plane, and the z-axis the 
            vertical axis, taken as positive downward.
        max_C : bool, default=True
            A flag indicating how to interpret the horizontal coordinates x and
            y. The maximum concentration will occur along a line parallel to
            the currents, which may change direction with height. If `max_C` is
            `True`, then the x-coordinate is taken as the distance along the
            current direction and the y-coordinate is taken as the distance
            perpendicular to x. Thus, if y=0, then this will always return the
            maximum concentration a distance x from the particle center of
            mass. If `max_C` is `False`, then the x- and y-coordinates are to
            be taken as absolute coordinates in the reference frame of the
            release. If they are located upstream of the particle center, the
            concentration will be zero; otherwise, the analytical solution is
            computed and the corresponding concentrations returned.
        
        Returns
        -------
        Cp : ndarray
            A two-dimensional array of concentrations at the given points. Each
            row of `Cp` corresponds to a row of `x`; the columns of `Cp`
            each correspond to a compound in the mixture composition.
        
        Notes
        -----
        This method uses the far-field model for wastewater outfalls suggested
        by Chin (John Wiley and Sons, Inc., 2013, 2nd edition, pp. 326). This
        model assumes no vertical turbulent diffusion and takes lateral
        apparent diffusion coefficients from the Okubo-diagram.
        
        There is no analytical solution for the whole concentration field for
        the scenario of a finite-size source. Here, the analytical solution is
        only valid for the plume centerline. Off the centerline, we can use a
        Gaussian distribution to estimate concentrations, but these may not
        completely conserve mass since off the plume centerline, this is an
        approximation, and not an analytical solution.
        
        """
        # Get the initial concentration and width of the intrusion
        Cp_0, z0, h, L = self.get_intrusion_initial_condition()
        
        # Get the horizontal ambient velocity
        ua, va = self.profile.get_values(z0, ['ua', 'va'])
        hvel = np.sqrt(ua**2 + va**2)
        
        # Convert to a coordinate system linked to the currents
        npoints, ndimensions = x.shape
        Lx = np.zeros(npoints)
        Ly = np.zeros(npoints)
        if max_C:
            Lx = x[:,0]
            Ly = x[:,1]
        else:
            # Create the coordinate transformation at the present location
            Ua = np.sqrt(ua**2 + va**2)
            theta = np.arctan2(va, ua)
            if theta < 0.:
                theta = 2. * np.pi + theta
            R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),
                np.cos(theta)]])
            
            # Convert the given coordinates
            for i in range(npoints):
                alpha = np.arctan2(x[i,1], x[i,0])
                if alpha < 0.:
                    alpha = 2. * np.pi + alpha
                if np.abs(theta - alpha) < np.pi / 2.:
                    # The chosen point is in the downstream direction: a non-zero
                    # solution for concentration exists
                    xs = np.matmul(R.transpose(), x[i,0:2])
                    Lx[i] = xs[0]
                    Ly[i] = xs[1]
                else:
                    # The chosen point is upstream of the concentration field: 
                    # the concentrations will be zero
                    Lx[i] = -9999.
                    Ly[i] = -9999.
        
        # Get the initial value of the diffusivity using equation 9-59 in 
        # Chin (2013)
        Ka = 0.0103 * (L * 100.)**(1.15)
        epsilon_0 = Ka / 100.**2
        beta = 12. * epsilon_0 / (hvel * L)

        # Compute the values for each point
        Cp = np.zeros((npoints, len(self.composition)))
        for i in range(npoints):
                    
            if np.abs(x[i,2] - z0) > h / 2.:
                # Above or below intrusion
                Cp[i,:] = np.zeros(len(self.composition))

            elif Lx[i] == -9999.:
                # This point is upstream of the intrusion
                Cp[i,:] = np.zeros(len(self.composition))
            
            else:
                # Get the centerline concentration
                from scipy.special import erf
                Cp_m = Cp_0 * erf(np.sqrt(3. / (2. * (1. + 2./3. * beta * 
                    Lx[i] / L)**3 - 1.)))
            
                # Estimate the transverse width of the plume
                l_y = L * (1. + 2./3. * beta * Lx[i] / L)**(3./2.)
                sigma_y = l_y / (2. * np.sqrt(3.))
        
                # Use a Gaussian to estimate the concentration drop outside
                # the main intrusion
                Cp[i,:] = Cp_m * np.exp(-(Ly[i]**2) / (2. * 
                    sigma_y**2))
        
        return Cp
    
    def get_grid_concentrations(self, x, max_C=True):
        """
        Compute the concentrations in the far-field from all particles
        
        Uses the method `Particle.field_concentrations` to compute the 
        contributions from each `Particle` and then superposes these values to 
        yield the total dissolved concentrations at the requested points.
        
        Parameters
        ----------
        x : ndarray
            Array of three-dimensional positions where the concentrations should
            be computed. Each row of `x` corresponds to a different point, with
            the columns of `x` giving the x-, y-, and z-coordinates of each 
            point.
        max_C : bool, default=True
            A flag indicating how to interpret the horizontal coordinates x and
            y. The maximum concentration will occur along a line parallel to
            the currents, which may change direction with height. If `max_C` is
            `True`, then the x-coordinate is taken as the distance along the
            current direction and the y-coordinate is taken as the distance
            perpendicular to x. Thus, if y=0, then this will always return the
            maximum concentration a distance x from the particle center of
            mass. If `max_C` is `False`, then the x- and y-coordinates are to
            be taken as absolute coordinates in the reference frame of the
            release. If they are located upstream of the particle center, the
            concentration will be zero; otherwise, the analytical solution is
            computed and the corresponding concentrations returned.
        
        Returns
        -------
        C_vals : ndarray
            A two-dimensional array of concentrations at the given points. Each
            row of `C_vals` corresponds to a row of `x`; the columns of `C_vals`
            each correspond to a compound in the `Particle` composition.
        
        Notes
        -----
        When a simulation includes more than one `Particle`, the `max_C=True`
        flag should be interpreted as follows. Each particle may be located
        along a different line in the horizontal r-theta plane. If all
        particles are not along the same line as the currents, then this method
        will treat them as if they are so aligned. Hence, it will always return
        the absolute maximum possible concentration. If the real concentration
        of the distributed particles is desired, then `max_C` should be set as
        `False`, and the concentrations will be computed at the given `x`
        points using absolute coordinates. The user will have to ensure that
        this set of points includes all relevant points in relation to the
        currents at each depth.
        
        """
        # Set up a matrix to hold the concentration data
        Cvals = np.zeros((x.shape[0], len(self.composition)))
        
        # Get the concentrations for each particle and use superposition to get
        # the concentration of the whole field
        for i in range(len(self.particles)):
            Cvals += self.particles[i].grid_concentrations(x, max_C)
        
        # Add in the concentrations from the intrusion layer
        Cvals += self.get_intrusion_concentration(x, max_C)
        
        # Return the result
        return Cvals
    
    def get_planar_concentrations(self, x, y, z, max_C=True):
        """
        Compute far-field concentrations on a designated plane
        
        This method is similar to `get_grid_concentrations`, but uses the
        `np.meshgrid` method to compute concentrations on a grid of points in
        one of the xy-, xz-, or yz-planes. These data can easily be visualized
        using the `plt.pcolor` plotting method. 
        
        Parameters
        ----------
        x : float or ndarray
            Coordinate(s) along the x-axis for which concentration values are 
            desired
        y : float or ndarray
            Coordinate(s) along the y-axis for which concentration values are
            desired
        z : float or ndarray
            Coordinate(s) along the z-axis for which concentration values are
            desired
        max_C : bool, default=True
            A flag indicating how to interpret the horizontal coordinates x and
            y. The maximum concentration will occur along a line parallel to
            the currents, which may change direction with height. If `max_C` is
            `True`, then the x-coordinate is taken as the distance along the
            current direction and the y-coordinate is taken as the distance
            perpendicular to x. Thus, if y=0, then this will always return the
            maximum concentration a distance x from the particle center of
            mass. If `max_C` is `False`, then the x- and y-coordinates are to
            be taken as absolute coordinates in the reference frame of the
            release. If they are located upstream of the particle center, the
            concentration will be zero; otherwise, the analytical solution is
            computed and the corresponding concentrations returned.
        
        Returns
        -------
        plane_0 : ndarray
            `np.meshgrid` coordinates for one of the coordinate axes along the 
            defined concentration plane
        plane_1 : ndarray
            `np.meshgrid` coordinates for the other coordinate axis along the
            defined concentration plane
        Cp : ndarray
            A three-dimensional array of concentration data on the defined
            plane. The first two coordinates of the array correspond to the
            coordinates of the defined plane; the third coordinate corresponds
            to each compound in the model composition.
            
        Notes
        -----
        One of the parameters `x`, `y`, or `z` should be a float or 1x1 array;
        the desired plane will be defined by the other two variables and going
        through this 1x1 array point.
        
        See also the notes above for `get_grid_concentrations` to understand
        the implications of the `max_C` parameter when more than one `Particle`
        object is present within a simulation.
        
        Example
        -------
        >>> # Create a bent_plume_model.Model object called bpm
        >>> # Execute the bpm.simulate() method
        >>> # Then, compute and plot far-field concentrations as follows
        >>> x = 2500
        >>> y = np.linspace(-15., 15., num=25)
        >>> z = np.linspace(0., 500., num=100)
        >>> yp, zp, Cp = bpm.get_planar_concentrations(x, y, z)
        >>> plt.pcolor(yp, zp, Cp[:,:,0])
        >>> plt.colorbar()
        >>> plt.gca().invert_yaxis()
        >>> plt.show()
        
        """
        # Make sure all the position data are arrays
        if isinstance(x, float):
            x = np.array([x])
        if isinstance(y, float):
            y = np.array([y])
        if isinstance(z, float):
            z = np.array([z])
        
        # Determine which plane to compute
        if len(x) == 1:
            # yz-plane
            plane = np.meshgrid(y, z)
            fixed_pt = x
            indices = [0, 1, 2]
        elif len(y) == 1:
            # xz-plane
            plane = np.meshgrid(x, z)
            fixed_pt = y
            indices = [1, 0, 2]
        else:
            # xy-plane
            plane = np.meshgrid(x, y)
            fixed_pt = z
            indices = [2, 0, 1]
        
        # Set up a three-dimensional matrix to hold the concentration data
        nx, ny = plane[0].shape
        Cp = np.zeros((nx, ny, len(self.composition)))
        
        # Compute all of the data
        print('\nComputing the far-field planar concentration data...')
        x = np.zeros((ny, 3))
        for i in range(nx):
            print('   --> Level %3.3d of %3.3d' % (i+1, nx))
            x[:,indices[0]] = fixed_pt
            x[:,indices[1]] = plane[0][i,:]
            x[:,indices[2]] = plane[1][i,:]
            Cvals = self.get_grid_concentrations(x, max_C)
            for j in range(len(self.composition)):
                Cp[i,:,j] = Cvals[:,j]
        print('Done.')
        
        # Return the computed data
        return (plane[0], plane[1], Cp)

    def get_derived_variables(self, track_chems=None):
        """
        Extract an array of derived variables for the present model solution
        
        The bent plume model state space does not contain many of the derived
        variables that one may want to analyze (e.g., the plume velocity,
        width, temperature, concentrations, etc.). This method uses the
        built-in conversion tools in the bent plume model `LagElement` class to
        compute many common derived results and stores these in an array. This
        method also builds a list of strings describing the data in the array.
        The class method `save_derived_variables`, which obtains its data from
        this method, should be used to save these data to a file.
        
        Parameters
        ----------
        track_chems : list, default=None
            A list of string names for the chemicals to include in the output
            file.  The default is `None`, which will cause this method to save
            all tracked chemicals.
        
        Returns
        -------
        data : ndarray
            The array of output data written to disk
        var_names : list
            A list of string names describing the data returned.  Each
            element of this list describes a column of the data in `data`
        num_p : int
            Number of plume particles in the solution output
        num_c : int
            Number of tracked chemicals included in the solution output.  This
            variable should equal either the length of the given list of 
            chemicals (`track_chems`) or the total number of chemicals tracked
            in the simulation (e.g., if `track_chems` is `None`).
        
        Notes
        -----
        This function only reports results for the Lagrangian plume.  If some of 
        the plume particles leave the plume and rise through the water column
        based on a `single_bubble_model` simulation, these results are not 
        included.  Use the `report_mass_fluxes` or `report_surfacing_fluxes`  
        with the appropriate `stage` flags to get the single bubble model results.
        
        """
        # Check if the simulation has been computed
        if not self.sim_stored:
            print('\nERROR:  You must run a simulation before computing the')
            print('        derived output.  Use the method simulate() to ')
            print('        conduct the required simulation. \n')
            return (0)
    
        # Get the names of each chemical and those we want to track
        chem_names = self.q_local.chem_names.copy()
        for particle in self.particles:
            for chem in particle.composition:
                if chem not in chem_names:
                    chem_names.append(chem)
        if isinstance(track_chems, type(None)):
            track_chems = chem_names.copy()
    
        # Figure out how many particles and how many chemicals are tracked
        num_p = self.q_local.np
        num_c = len(track_chems)
        
        # Create a blank list to store annotated variable names
        var_names = []
        
        # Compute the number of needed output rows (adapt this line as 
        # additional outputs are added in the lines below)
        num_cols = 13 + num_c + num_p * (3 + num_c)
        
        # Create a data array to hold this data
        data = np.zeros((len(self.t), num_cols))    
    
        # Loop through each time step, compute the derived variables, and save 
        # them to the output array in the appropriate locations
        for i in range(len(self.t)):
    
            # Compute the derived variables at the present output time
            self.q_local.update(self.t[i], self.q[i,:], self.profile, self.p, 
                self.particles)
            col = 0            
        
            # Compute the coordinates of the plume edge
            x = self.q_local.x
            y = self.q_local.y
            z = self.q_local.z
            b = self.q_local.b
            Sz = self.q_local.sin_p
            Sx = self.q_local.cos_p * self.q_local.cos_t
            Sy = self.q_local.cos_p * self.q_local.sin_t
            x1, z1, x2, z2 = width_projection(Sx, Sz, b)
            x_xz_l = x + x1
            x_xz_r = x + x2
            z_xz_l = z + z1
            z_xz_r = z + z2
            y1, z1, y2, z2 = width_projection(Sy, Sz, b)
            y_yz_l = y + y1
            y_yz_r = y + y2
            z_yz_l = z + z1
            z_yz_r = z + z2

            # Store the plume properties geometric properties            
            if i == 0:
                var_names.append('Plume centerline x-coordinate (m)')
            data[i,col] = x
            col += 1
            if i == 0:
                var_names.append('Plume centerline y-coorindate (m)')
            data[i,col] = y
            col += 1
            if i == 0:
                var_names.append('Plume centerline z-coordinate (m)')
            data[i,col] = z
            col += 1
            if i == 0:
                var_names.append('Plume left boundary; x in xz-plane (m)')
            data[i,col] = x_xz_l
            col += 1
            if i == 0:
                var_names.append('Plume left boundary; z in xz-plane (m)')
            data[i,col] = z_xz_l
            col += 1
            if i == 0:
                var_names.append('Plume right boundary; x in xz-plane (m)')
            data[i,col] = x_xz_r
            col += 1
            if i == 0:
                var_names.append('Plume right boundary; z in xz-plane (m)')
            data[i,col] = z_xz_r
            col += 1
            if i == 0:
                var_names.append('Plume left boundary; y in yz-plane (m)')
            data[i,col] = y_yz_l
            col += 1
            if i == 0:
                var_names.append('Plume left boundary; z in yz-plane (m)')
            data[i,col] = z_yz_l
            col += 1
            if i == 0:
                var_names.append('Plume right boundary; y in yz-plane (m)')
            data[i,col] = y_yz_r
            col += 1
            if i == 0:
                var_names.append('Plume right boundary; z in yz-plane (m)')
            data[i,col] = z_yz_r
            col += 1
            if i == 0:
                var_names.append('Plume velocity along centerline (m/s)')
            data[i,col] = self.q_local.V
            col += 1
            if i == 0:
                var_names.append('Plume half-width (radius, m)')
            data[i,col] = b
            col += 1
            Q = self.q_local.V * np.pi * b**2
            
            # Store the dissolved concentrations of tracked chemicals in the 
            # plume
            for chem in track_chems:
                if chem in self.q_local.chem_names:
                    md = Q * self.q_local.c_chems[
                        self.q_local.chem_names.index(chem)]
                else:
                    md = 0.
                if i == 0:
                    var_names.append('Mass flux of %s in the plume (kg/s)' \
                        % (chem))
                data[i,col] = md
                col += 1
            
            # Store the results for each plume particle
            for j in range(num_p):
                
                # Store the position
                if i == 0:
                    var_names.append('x-coordinate (m) of particle %3.3d' \
                        % j)
                data[i,col] = self.q_local.x_p[j,0]
                col += 1
                if i == 0:
                    var_names.append('y-coordinate (m) of particle %3.3d' \
                        % j)
                data[i,col] = self.q_local.x_p[j,1]
                col += 1
                if i == 0:
                    var_names.append('z-coordinate (m) of particle %3.3d' \
                        % j)
                data[i,col] = self.q_local.x_p[j,2]
                col += 1
                
                # Store the mass fluxes of tracked chemicals
                for chem in track_chems:
                    if chem in self.particles[j].composition:
                        Mpf = self.q_local.M_p[j][ \
                            self.particles[j].composition.index(chem)] \
                            / self.particles[j].nbe * self.particles[j].nb0
                    else:
                        Mpf = 0.
                    # Mass flux cannot be negative
                    if Mpf < 0.:
                        Mpf = 0.
                    if i == 0:
                        var_names.append('Mass flux of %s (kg/s) in particle' \
                            ' group %3.3d' % (chem, j))
                    data[i,col] = Mpf                       
                    col += 1
        
        # Return the data and header information
        return (data, var_names, num_p, num_c)
    
    def save_sim(self, fname, profile_path, profile_info):
        """
        Save the current simulation results

        Save the current simulation results and the model parameters so that
        all information needed to rebuild the class object is stored in a
        file.  The output data are stored in netCDF4-classic format.

        Parameters
        ----------
        fname : str
            File name of the netCDF file to write
        profile_path : str
            String stating the file path to the ambient profile data relative
            to the directory where `fname` will be saved.
        profile_info : str
            Single line of text describing the ambient profile data.

        See Also
        --------
        dispersed_phases.save_particle_to_nc_file

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
        title = 'Simulation results for the TAMOC Bent Plume Model'
        nc = model_share.tamoc_nc_file(fname, title, profile_path,
            profile_info)

        # Create variables for the dimensions
        t = nc.createDimension('t', None)
        p = nc.createDimension('profile', 1)
        ns = nc.createDimension('ns', len(self.q_local.q0))
        params = nc.createDimension('params', 1)

        # Save the names of the chemicals in the tracers and particle objects
        nc.tracers = ' '.join(self.tracers)
        nc.chem_names = ' '.join(self.chem_names)

        # Create variables to store the initial conditions
        x0 = nc.createVariable('x0', 'f8', ('params',))
        x0.long_name = 'Initial value of the x-coordinate'
        x0.standard_name = 'x0'
        x0.units = 'm'

        y0 = nc.createVariable('y0', 'f8', ('params',))
        y0.long_name = 'Initial value of the y-coordinate'
        y0.standard_name = 'y0'
        y0.units = 'm'

        z0 = nc.createVariable('z0', 'f8', ('params',))
        z0.long_name = 'Initial depth below the water surface'
        z0.standard_name = 'depth'
        z0.units = 'm'
        z0.axis = 'Z'
        z0.positive = 'down'

        D = nc.createVariable('D', 'f8', ('params',))
        D.long_name = 'Orifice diameter'
        D.standard_name = 'diameter'
        D.units = 'm'

        Vj = nc.createVariable('Vj', 'f8', ('params',))
        Vj.long_name = 'Discharge velocity'
        Vj.standard_name = 'Vj'
        Vj.units = 'm'

        phi_0 = nc.createVariable('phi_0', 'f8', ('params',))
        phi_0.long_name = 'Discharge vertical angle to horizontal'
        phi_0.standard_name = 'phi_0'
        phi_0.units = 'rad'

        theta_0 = nc.createVariable('theta_0', 'f8', ('params',))
        theta_0.long_name = 'Discharge horizontal angle to x-axis'
        theta_0.standard_name = 'theta_0'
        theta_0.units = 'rad'

        Sj = nc.createVariable('Sj', 'f8', ('params',))
        Sj.long_name = 'Discharge salinity'
        Sj.standard_name = 'Sj'
        Sj.units = 'psu'

        Tj = nc.createVariable('Tj', 'f8', ('params',))
        Tj.long_name = 'Discharge temperature'
        Tj.standard_name = 'Tj'
        Tj.units = 'K'

        cj = nc.createVariable('cj', 'f8', ('params',))
        cj.long_name = 'Discharge tracer concentration'
        cj.standard_name = 'cj'
        cj.units = 'nondimensional'

        Ta = nc.createVariable('Ta', 'f8', ('params',))
        Ta.long_name = 'ambient temperature at the release point'
        Ta.standard_name = 'Ta'
        Ta.units = 'K'

        Sa = nc.createVariable('Sa', 'f8', ('params',))
        Sa.long_name = 'ambient salinity at the release point'
        Sa.standard_name = 'Sa'
        Sa.units = 'psu'

        P = nc.createVariable('P', 'f8', ('params',))
        P.long_name = 'ambient pressure at the release point'
        P.standard_name = 'P'
        P.units = 'Pa'

        # Create variables for the simulation setup
        track = nc.createVariable('track', 'i4', ('params',))
        track.long_name = 'SBM Status (0: false, 1: true)'
        track.standard_name = 'track'
        track.units = 'boolean'

        dt_max = nc.createVariable('dt_max', 'f8', ('params',))
        dt_max.long_name = 'Simulation maximum duration'
        dt_max.standard_name = 'dt_max'
        dt_max.units = 's'

        sd_max = nc.createVariable('sd_max', 'f8', ('params',))
        sd_max.long_name = 'Maximum distance along centerline s/D'
        sd_max.standard_name = 'sd_max'
        sd_max.units = 'nondimensional'

        # Create a variable for the independent variable
        t = nc.createVariable('t', 'f8', ('t', 'profile',))
        t.long_name = 'time along the plume centerline'
        t.standard_name = 'time'
        t.units = 's'
        t.axis = 'T'
        t.n_times = len(self.t)

        # Create a variable for the model state space
        q = nc.createVariable('q', 'f8', ('t', 'ns',))
        q.long_name = 'Lagranian plume model state space'
        q.standard_name = 'q'
        q.units = 'variable'

        # Store the model initial conditions
        x0[0] = self.X[0]
        y0[0] = self.X[1]
        z0[0] = self.X[2]
        D[0] = self.D
        Vj[0] = self.Vj
        phi_0[0] = self.phi_0
        theta_0[0] = self.theta_0
        Sj[0] = self.Sj
        Tj[0] = self.Tj
        cj[0] = self.cj
        Ta[0], Sa[0], P[0] = self.profile.get_values(np.max(self.X[2]),
            ['temperature', 'salinity', 'pressure'])

        # Store the model setup
        if self.track:
            track[0] = 1
        else:
            track[0] = 0
        dt_max[0] = self.dt_max
        sd_max[0] = self.sd_max

        # Save the dispersed phase particles
        dispersed_phases.save_particle_to_nc_file(nc, self.chem_names,
            self.particles, self.K_T0)

        # Save the tracked particles if they exist
        for i in range(len(self.particles)):
            if self.particles[i].farfield:
                fname_sbm = fname.split('.nc')[0] + '%3.3d.nc' % i
                self.particles[i].sbm.save_sim(fname_sbm, profile_path,
                    profile_info)

        # Store the plume simulation solution
        t[:,0] = self.t[:]
        for i in range(len(nc.dimensions['ns'])):
            q[:,i] = self.q[:,i]

        # Store any single bubble model simulations


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

        See Also
        --------
        save_sim, load_sim, stratified_plume_model.Model.save_txt,
        single_bubble_model.Model.save_txt

        Notes
        -----
        The output will be organized in columns, with each column as follows:

            0   : time (s)
            1-n : state space

        The header to the output file will give the extact organization of
        each column of the output data.

        These output files are written using the `numpy.savetxt` method.

        Note, also, that this only saves the state space solution, and the
        data saved by this function is inadequate to rebuild the `Model`
        object by reloading a saved solution.  To have seamless saving and
        loading of `Model` objects, use the `save_sim` and `load_sim`
        commands.

        """
        if self.sim_stored is False:
            print('No simulation results to store...')
            print('Saved nothing to text file.\n')
            return

        # Create the header string that contains the column descriptions
        # for the Lagrangian plume state space
        p_list = ['Lagrangian Plume Model ASCII Output File \n']
        p_list.append('Created: ' + datetime.today().isoformat(' ') + '\n\n')
        p_list.append('Simulation based on CTD data in:\n')
        p_list.append(profile_path)
        p_list.append('\n\n')
        p_list.append(profile_info)
        p_list.append('\n\n')
        p_list.append('Column Descriptions:\n')
        p_list.append('    0: time (s)\n')
        p_list.append('    1: mass (kg)\n')
        p_list.append('    2: salinity (psu)\n')
        p_list.append('    3: heat (J)\n')
        p_list.append('    4: x-direction momentum (kg m/s)\n')
        p_list.append('    5: y-direction momentum (kg m/s)\n')
        p_list.append('    6: z-direction momentum (kg m/s)\n')
        p_list.append('    7: relative thickness h/V (s)\n')
        p_list.append('    8: x-coordinate (m)\n')
        p_list.append('    9: y-coordinate (m)\n')
        p_list.append('    10: z-coordinate (m)\n')
        p_list.append('    11: distance along plume centerline (m)\n')
        idx = 11
        for i in range(len(self.particles)):
            for j in range(len(self.particles[i].m0)):
                idx += 1
                p_list.append(
                    '    %d: Total mass flux of %s in particle %d (kg/s)\n' %
                    (idx, self.particles[i].composition[j], i))
            idx += 1
            p_list.append('    %d: Total heat flux of particle %d (J/s)\n' %
                          (idx, i))
            idx += 1
            p_list.append('    %d: Time since release of particle %d (s)\n' %
                          (idx, i))
            idx += 1
            p_list.append('    %d: Lambda coordinate of particle %d (m)\n' %
                          (idx, i))
            idx += 1
            p_list.append('    %d: Eta coordinate of particle %d (m)\n' %
                          (idx, i))
            idx += 1
            p_list.append('    %d: Csi coordinate of particle %d (m)\n' %
                          (idx, i))
        for i in range(len(self.chem_names)):
            idx += 1
            p_list.append('    %d: Mass of dissolved %s (kg)\n' %
                          (idx, self.chem_names[i]))
        for i in range(len(self.tracers)):
            idx += 1
            p_list.append('    %d: Mass of %s in (kg)\n' %
                          (idx, self.tracers[i]))
        header = ''.join(p_list)

        # Assemble and write the state space data
        data = np.hstack((np.atleast_2d(self.t).transpose(), self.q))
        np.savetxt(base_name + '.txt', data)
        with open(base_name + '_header.txt', 'w') as dat_file:
            dat_file.write(header)

        # Save the tracked particles if they exist
        for i in range(len(self.particles)):
            if self.particles[i].farfield:
                fname_sbm = base_name + '%3.3d' % i
                self.particles[i].sbm.save_txt(fname_sbm, profile_path,
                    profile_info)

    def save_derived_variables(self, fname, track_chems=None):
        """
        Save an ASCII text file of derived simulation results
        
        The bent plume model state space does not contain many of the derived
        variables that one may want to analyze (e.g., the plume velocity, 
        width, temperature, concentrations, etc.).  While all of these may be
        computed from the state space variables saved through either `save_sim`
        or `save_txt`, new functions would need to be used to compute these
        derived results.  This method uses the built-in conversion tools in 
        the bent plume model `LagElement` class to compute many common derived
        results, stores these in an array, and saves them to a text file.  
        See the notes below and the file text header for details on which 
        variables are saved, their meaning and dimensions.
        
        Parameters
        ----------
        fname : str
            File name with absolute or relative file path for the ASCII data
            file to write. Include the full file name including any needed
            relative or absolute file path information.  This method uses
            `np.savetxt` to create the output file.
        track_chems : list, default=None
            A list of string names for the chemicals to include in the output
            file.  The default is `None`, which will cause this method to save
            all tracked chemicals.
        
        Returns
        -------
        data : ndarray
            The array of output data written to disk
        header : str
            The string header describing the data written to disk
        
        Notes
        -----
        The simulation results are stored in the model attributes `self.t` (the
        vector of simulation times) and `self.q` (the vector of state-space)
        variables. Although the model computes the solution as a function of
        time in a Lagrangian reference frame, the data are steady-state, and
        the output of this method records the data as a function of depth (m).
        
        Because the bent plume model may overshoot the level of neutral 
        buoyancy and come to rest at a location deeper than the maximum height
        of plume rise, the depth data in the output file may not be 
        monotonically increasing, and there may be multiple output values at
        the same heights (once as the plume ascends through a given depth
        and again as the plume descends to a location of neutral buoyancy).
        If one chooses to interpolate between data points in this file, care
        should be used that the right set of output depths are used.
        
        """
        # Get the derived variables
        data, var_names, num_p, num_c = self.get_derived_variables(track_chems)
        
        # Build an output header
        from datetime import date
        header = 'Derived output data from the TAMOC Bent Plume Model\n'
        header += 'Created on: ' + \
            datetime.now().strftime("%Y-%m-%d %H:%M:%S") \
            + '\n\n'
        header += 'Data are stored in the following order:\n'
        col = 0
        for name in var_names:
            header += '    Col %3.3d:  ' % (col) + name + '\n'
            col += 1
        header += '\nThere are %3.3d particle groups in this output.\n' % num_p
        header += 'There are %3.3d chemicals tracked in this output.\n' % \
            num_c
        
        # Write the data to a file
        ext = fname.split('.')[-1]
        if len(ext) != 3:
            fname += '.txt'
        np.savetxt(fname, data, header=header)
        
        # Return the results
        return (data, header)
        
    def load_sim(self, fname):
        """
        Load in a saved simulation result file for post-processing

        Load in a saved simulation result file and rebuild the `Model`
        object attributes.  The input file must be in netCDF4-classic data
        format.

        Parameters
        ----------
        fname : str
            File name of the file to read

        See Also
        --------
        save_sim

        Notes
        -----
        This method will attempt to load the ambient profile data from the
        `profile_path` attribute of the `fname` netCDF file.  If the load
        fails, a warning will be reported to the terminal, but the other
        steps of loading the `Model` object attributes will be performed.

        """
        # Open the netCDF dataset object containing the simulation results
        nc = Dataset(fname)

        # Try to get the profile data
        self.profile = model_share.profile_from_model_savefile(nc, fname)
        if self.profile is not None:
            self.p = ModelParams(self.profile)
            self.got_profile = True
        else:
            self.p = None
            self.got_profile = False

        # Get the release location of the plume
        self.X = np.zeros(3)
        self.X[0] = nc.variables['x0'][0]
        self.X[1] = nc.variables['y0'][0]
        self.X[2] = nc.variables['z0'][0]

        # Create the Particle objects
        self.particles, self.chem_names = \
            dispersed_phases.load_particle_from_nc_file(nc)

        # Extract the remaining model constants
        self.D = nc.variables['D'][0]
        self.Vj = nc.variables['Vj'][0]
        self.phi_0 = nc.variables['phi_0'][0]
        self.theta_0 = nc.variables['theta_0'][0]
        self.Sj = nc.variables['Sj'][0]
        self.Tj = nc.variables['Tj'][0]
        self.cj = nc.variables['cj'][0]
        if nc.variables['track'][0] == 1:
            self.track = True
        else:
            self.track = False
        self.dt_max = nc.variables['dt_max'][0]
        self.sd_max = nc.variables['sd_max'][0]

        # Compute the dimensions of the arrayed data
        ns = len(nc.dimensions['ns'])
        nt = nc.variables['t'].n_times

        # Extract the arrayed data
        self.tracers = nc.tracers.split()
        self.K_T0 = np.array([self.particles[i].K_T for i in
                              range(len(self.particles))])
        self.t = np.zeros(nt)
        self.t[:] = nc.variables['t'][0:nt,0]
        self.q = np.zeros((nt, ns))
        for i in range(ns):
            self.q[:,i] = nc.variables['q'][0:nt,i]

        # Create the local Lagrangian plume element
        self.q_local = LagElement(self.t[0], self.q[0,:], self.D,
                       self.profile, self.p, self.particles, self.tracers,
                       self.chem_names)

        # Load in any farfield tracking results
        for i in range(len(self.particles)):
            if self.particles[i].farfield:
                fname_sbm = fname.split('.nc')[0] + '%3.3d.nc' % i
                self.particles[i].sbm = \
                    single_bubble_model.Model(simfile=fname_sbm)

        # Close the netCDF dataset
        nc.close()
        self.sim_stored = True 
    
    def report_mass_fluxes(self, idx, stage=0, chems=None, fp_type=-1):
        """
        Report the particulate mass fluxes at a given point
        
        Compute the mass flux of each compound in a mixture for each tracked
        particle and report these data at a requested simulation index. The
        user may request output either from the near-field plume (stage = 0)
        or the Lagrangian particle tracking phase of transport (stage = 1).
        The list of compounds to track can be specified with chems, and the
        particle phase can be limited using fp_type. To track all particles,
        set fp_type = -1. To track only particles that were initially gas,
        set fp_type = 0. To track particles that were initially liquid, set
        fp_type = 1.
        
        This method returns the actual mass fluxes for the requested
        particles at the requested index to the solution (mp) and the
        difference between these current values and their initial values
        (md). This latter variable should be equal to the mass that has
        dissolved and biodegraded from the release to the present simulation
        step.
        
        Parameters
        ----------
        idx : int
            Index to the simulation step for which data should be reported.
            This can be any acceptable numpy index to an array, but not a
            slice. The purpose of this method is to output data at a single
            point in the solution.  Zero will yield the initial condition, -1 
            will give the last computed point, and any other integer will 
            index that respective point in the output arrays.
        stage : int, default=0
            Flag indicating whether the user wants data from the near-field
            plume (stage = 0) or the Lagrangian particle tracking stage of
            transport (stage = 1). The idx value will index the respective
            model simulation vector that matches the chosen stage of the
            solution.
        chems : list, default=None
            A list of chemical compounds to track.  This may be a list of 
            chemical names or a list of index-numbers corresponding to 
            chemicals in the mixture composition. If `None`, then all 
            compounds in the simulation will be included.
        fp_type : int
            A flag indicating which particles should be included. 0 -- will
            include only particles that were initially gas-phase, 1 -- will
            include only particles that were initially liquid-phase particles,
            and -1 -- will include all particles.
        
        Returns
        -------
        mp : ndarray
            Mass flux (kg/s) of compounds for each selected particle at the
            given point in the simulation. These data are stored with the
            row-index corresponding to a given particle and the column-index
            corresponding to a mixture compound in the chems list
        md : ndarray
            Difference (kg/s) between the current mass flux at the given point
            in the solution and the initial mass flux for the selected 
            particles and compounds.
        
        """
        # Check whether the simulation data exist
        if not self.sim_stored:
            print('\nERROR:  Run a simulation before interrogating results.')
            print('    --> Execute Model.simulate() to proceed.\n')
            return(0, 0)
        
        # Get the indices to the chems we want to track
        c_idx = chem_idx_list(chems, self.particles[0].particle.composition)
        
        # Initialize matrices to hold the model results
        mp = np.zeros((len(self.particles), len(c_idx)))
        md = np.zeros((len(self.particles), len(c_idx)))
        
        # Get the conditions at the release (initial conditions)
        self.q_local.update(self.t[0], self.q[0], self.profile, self.p, 
            self.particles)
        
        # Store all the initial particle fluxes
        m0 = np.zeros((len(self.particles), len(c_idx)))
        for particle in self.particles:
            m0[self.particles.index(particle), :] = particle.nb0 * \
                particle.m[c_idx]
        
        # Update the LagElement at the selected index
        if stage == 0:
            # Use the index to the near-field plume specified by the user
            self.q_local.update(self.t[idx], self.q[idx], self.profile, 
                self.p, self.particles)
        else:
            # Update the LagElement to the last point in the plume
            self.q_local.update(self.t[-1], self.q[-1], self.profile, self.p,
                self.particles)
        
        # Add up the contributions from each particle
        for particle in self.particles:
            
            if fp_type < 0 or particle.particle.fp_type == fp_type:
                # Include this particle in the mass balance
                
                if stage == 0:
                    # Get results from the near-field plume
                    n_dot = particle.nb0
                    m = particle.m[c_idx]
                
                else:
                    # Get results from the Lagrangian particle-tracking
                    n_dot = particle.nb0
                    if self.track:
                        if particle.tracked:
                            m = particle.sbm.y[idx, 3:-1][c_idx]
                        else:
                            m = particle.m[c_idx]
                    else:
                        m = particle.m[c_idx]
                
                # Compute the mass flux
                mp[self.particles.index(particle), :] = n_dot * m
                md[self.particles.index(particle), :] = \
                    m0[self.particles.index(particle), :] - n_dot * m
        
        return (mp, md)
    
    def report_surfacing_fluxes(self, chems=None, fp_type=-1):
        """
        Report the mass fluxes reaching the sea surface
        
        Compute the mass fluxes of the listed compounds (chems) as they reach
        the sea surface. These mass fluxes include the fluxes of selected
        particles plus all dissolved-phase masses if the near-field plume
        reaches the surface. Set fp_type=-1 to track all particles, fp_type=0
        to track only initially gas-phase particles, and fp_type=1 to track
        only initially liquid-phase particles. The results report both the
        surfacing mass flux and the time for which that portion of the mass
        reaches the sea surface. The sea surface is defined as the region
        within 50 m of the surface.
        
        Parameters
        ----------
        chems : list
            A list of chemical compounds to track.  This may be a list of 
            chemical names or a list of index-numbers corresponding to 
            chemicals in the mixture composition.
        fp_type : int
            A flag indicating which particles should be included. 0 -- will
            include only particles that were initially gas-phase, 1 -- will
            include only particles that were initially liquid-phase particles,
            and -1 -- will include all particles.
        
        Returns
        -------
        mp : ndarray
            Mass flux (kg/s) of compounds for each selected particle at the
            given point in the simulation. These data are stored with the
            row-index corresponding to a given particle and the column-index
            corresponding to a mixture compound in the chems list
        mc : ndarray
            Mass flux (kg/s) for each compound reaching the sea surface in the
            dissolved phase due to the near-field plume surfacing.  If the 
            near-field plume traps below 50 m depth, this array will contain
            zeros.
        tp : ndarray
            Array of surfacing times (s) for each particle in the mp array.
        tc : float
            Surfacing time (s) for the near-field plume.  Will equal np.nan 
            if the near-field plume does not surface.
        
        """
        # Check whether the simulation data exist
        if not self.sim_stored:
            print('\nERROR:  Run a simulation before interrogating results.')
            print('    --> Execute Model.simulate() to proceed.\n')
            return(0, 0, 0, 0)
        
        # Get the indices to the chems we want to trac
        c_idx = chem_idx_list(chems, self.particles[0].particle.composition)
        
        # Determine whether the near-field plume surfaced
        if self.q[-1,9] <= 50.:
            # The plume surfaced...get the dissolved mass fluxes
            plume_surfaced = True
            self.q_local.update(self.t[-1], self.q[-1], self.profile, self.p, 
                self.particles)
            V = self.q_local.V
            b = self.q_local.b
            Q = np.pi * b**2 * V
            mc = self.q_local.c_chems[c_idx] * Q
            tc = self.t[-1]
        else:
            # The near-field plume trapped
            plume_surfaced = False
            mc = np.zeros(len(c_idx))
            tc = np.nan
        
        # Initialize an array to store the particle surfacing times
        tp = np.zeros(len(self.particles))
        
        # Get the mass fluxes of the particles as they surface
        if not plume_surfaced and not self.track:
            print('\nERROR:  The plume did not surface, and the particles')
            print('        were not tracked.  Hence, we cannot compute the')
            print('        surfacing fluxes of the particles.')
            print('    --> Rerun the simulation with track=True.\n')
            return (0, 0, 0, 0)
        
        elif plume_surfaced:
            # Particle masses are from the plume solution
            mp, md = self.report_mass_fluxes(-1, stage=0, chems=c_idx, 
                fp_type=fp_type)
            for particle in self.particles:
                # Only track particles of the right type
                if fp_type < 0 or particle.particle.fp_type == fp_type:
                    if particle.z >= 50.:
                        # This particle did not surface
                        tp[self.particles.index(particle)] = np.nan
                        mp[self.particles.index(particle), :] = \
                            np.zeros(len(c_idx))   
                    else:
                        # This particle did surface
                        tp[self.particles.index(particle)] = particle.t
            
            # Warn the user about untracked particles
            if not self.track:
                print('\nWARNING:  Plume surfaced, but particles were not')
                print('          tracked.  If particles exited the plume ')
                print('          before surfacing, they will not be counted.')
                print('      --> Proceeding with calculation...')
                print('      --> To fix, re-run simulation with track=True.')
        
        else:
            # Particle masses are from the SBM solution
            mp, md = self.report_mass_fluxes(-1, stage=1, chems=c_idx, 
                fp_type=fp_type)
            for particle in self.particles:
                # Only track particles of the right type
                if fp_type < 0 or particle.particle.fp_type == fp_type:
                    if particle.sbm.y[-1,2] >= 50.:
                        # This particle did not surface
                        tp[self.particles.index(particle)] = np.nan
                        mp[self.particles.index(particle), :] = \
                            np.zeros(len(c_idx))
                    else:
                        # This particle did surface
                        tp[self.particles.index(particle)] = particle.t + \
                            particle.sbm.t[-1]
        
        # Return all the results
        return (mp, mc, tp, tc)

    def report_watercolumn_particle_fluxes(self, chems=None, fp_type=-1):
        """
        Report the mass fluxes of particles leaving the plume subsurface
        
        Compute the mass fluxes of the listed compounds (chems) within
        particles that leave the plume before surfacing. Set fp_type=-1 to
        track all particles, fp_type=0 to track only initially gas-phase
        particles, and fp_type=1 to track only initially liquid-phase
        particles. The results report both the exiting mass flux and the time
        at which the particle left the plume. For surfacing plumes, the sea
        surface is defined as the region within 50 m of the surface.
        
        Parameters
        ----------
        chems : list
            A list of chemical compounds to track.  This may be a list of 
            chemical names or a list of index-numbers corresponding to 
            chemicals in the mixture composition.
        fp_type : int
            A flag indicating which particles should be included. 0 -- will
            include only particles that were initially gas-phase, 1 -- will
            include only particles that were initially liquid-phase particles,
            and -1 -- will include all particles.
        
        Returns
        -------
        mp : ndarray    
            Mass flux (kg/s) of compounds for each selected particle at moment
            it leaves the bent plume. These data are stored with the row-index
            corresponding to a given particle and the column-index
            corresponding to a mixture compound in the chems list
        tp : ndarray
            Array of exiting times (s) for each particle in the mp array.
        
        """
        # Check whether the simulation data exist
        if not self.sim_stored:
            print('\nERROR:  Run a simulation before interrogating results.')
            print('    --> Execute Model.simulate() to proceed.\n')
            return(0, 0)
        
        # Get the indices to the chems we want to trac
        c_idx = chem_idx_list(chems, self.particles[0].particle.composition)
        
        # Initialize an array to store the particle exiting times
        tp = np.zeros(len(self.particles))
        
        # Get the particle fluxes as they exit the plume
        mp, md = self.report_mass_fluxes(-1, stage=0, chems=c_idx, 
            fp_type=fp_type)
        
        # Only track particles of the right type and location
        for particle in self.particles:
            if fp_type < 0 or particle.particle.fp_type == fp_type:
                if particle.z >= 50.:
                    # This particle is considered subsurface
                    tp[self.particles.index(particle)] = particle.t
                else:
                    # This particle is at the surface
                    mp[self.particles.index(particle)] = np.zeros(len(c_idx))
            else:
                # We are not interested in this particle
                mp[self.particles.index(particle)] = np.zeros(len(c_idx))
        
        # Return all the results
        return (mp, tp)
    
    def report_psds(self, loc, stage):
        """
        Return the gas and liquid particle distributions
        
        Extracts the gas bubble and liquid droplet particle size distributions
        at the position in the simulation solution vector given by `loc` (see
        parameter description below for details). The `stage` flag is used to
        indicate whether the results should be taken from the nearfield plume
        simulation (`stage` = 0) or the farfield particle tracking (`stage` =
        1).
        
        Parameters
        ----------
        loc : int or float
            Location where the particle size distribution should be computed.
            If `stage` = 0, this is an index to the solution vector of the
            nearfield plume in a bent plume model simulation. If `stage` = 1,
            this is a depth (m) where the output data should be extracted.
        stage : int
            Flag indicating whether to return results for the nearfield plume
            solution (`stage` = 0) or the farfield particle tracking (`stage` =
            1)
        
        Returns
        -------
        d_gas : float
            Diameters (m) of the gas bubbles
        v_gas : float
            Volume fraction (--) corresponding to each gas bubble size
        d_liq : float
            Diameters (m) of the liquid droplets
        v_liq : float
            Volume fraction (--) corresponding to each liquid droplet size
        
        """
        # Check whether the simulation data exist
        if not self.sim_stored:
            print('\nERROR:  Run a simulation before interrogating results.')
            print('    --> Execute Model.simulate() to proceed.\n')
            return(0, 0)
        
        # Initialize lists to hold the model results
        d_gas = []
        v_gas = []
        d_liq = []
        v_liq = []
        
        if stage == 0:
            # Get results from the near-field plume...
            # Update the plume element to the desired index
            self.q_local.update(self.t[loc], self.q[loc], self.profile, 
                self.p, self.particles)
        
            # Get the conditions at desired index point
            for particle in self.particles:
                mp = particle.m
                Tp = particle.T
                Ta = self.q_local.Ta
                Sa = self.q_local.Sa
                Pa = self.q_local.Pa
                de = particle.diameter(mp, Tp, Pa, Sa, Ta)
                Vp = 4./3. * np.pi * (de/2.)**3
                Vf = Vp * particle.nb0
                if particle.particle.fp_type == 0:
                    d_gas.append(de)
                    v_gas.append(Vf)
                else:
                    d_liq.append(de)
                    v_liq.append(Vf)
        
        elif stage == 1:
            # Get results from the far-field Lagrantian particle tracking...
            if not self.track:
                print('ERROR:  The far-field tracking was not activated.')
                print('    --> Plot results from the near-field plume')
                print('        (stage = 0) or rerun the simulation with')
                print('        track = True.')
                return (-1, -1, -1, -1)
            for particle in self.particles:
                # Interpolate the SBM output to the requested vertical level
                z = particle.sbm.y[:,2]
                i0 = np.max(np.where(z>=loc))
                if i0 + 1 == len(z):
                    i1 = i0 - 1
                else:
                    i1 = i0 + 1
                z0 = z[i0]
                z1 = z[i1]
                y0 = particle.sbm.y[i0,:]
                y1 = particle.sbm.y[i1,:]
                y = (y1 - y0) / (z1  - z0) * (loc - z0) + y0
                
                # Extract the particle properties
                zp = loc
                mp = y[3:-1]
                Tp = y[-1] / (np.sum(mp) * particle.cp)
                
                # Get the ambient data
                Ta, Sa, Pa = self.profile.get_values(zp, ['temperature',
                    'salinity', 'pressure'])
                
                # Get the diameter
                de = particle.diameter(mp, Tp, Pa, Sa, Ta)
                Vp = 4./3. * np.pi * (de/2.)**3
                Vf = Vp * particle.nb0
                if particle.particle.fp_type == 0:
                    d_gas.append(de)
                    v_gas.append(Vf)
                else:
                    d_liq.append(de)
                    v_liq.append(Vf)
        
        else:
            print('\nERROR:  Requested simulation data unknown.')
            print('    --> Choose a simulation stage = 0 (nearfield plume)')
            print('        or 1 (farfield particle tracking)\n')
            return (-1, -1, -1, -1)
        
        # Convert lists to arrays
        d_gas = np.array(d_gas)
        v_gas = np.array(v_gas)
        d_liq = np.array(d_liq)
        v_liq = np.array(v_liq)
    
        # Convert the volume distributions to volume fraction
        v_gas = v_gas / np.sum(v_gas)
        v_liq = v_liq / np.sum(v_liq)
    
        # Return the results
        return (d_gas, v_gas, d_liq, v_liq)
    
    def plot_psds(self, fig, loc=0, stage=0, clear=True):
        """
        Plot bubble and droplet size distributions
        
        Plot the gas bubble and liquid droplet size distributions
        
        Parameters
        ----------
        fig : int or MPL Figure object
              MPL Figure() on which to plot
              or
              Number of the figure window in which to draw the plot
              (Figure will be created for you with the provided fig number)
        loc : int or float (default=0)
            Position where the size distributions should be computed. This
            parameter is passed to `report_psds`. If `stage` = 0, this is an
            index to the solution vector of the nearfield plume in a bent plume
            model simulation. If `stage` = 1, this is a depth (m) where the
            output data should be extracted.
        stage : int
            Flag indicating whether to return results for the nearfield plume
            solution (`stage` = 0) or the farfield particle tracking (`stage` =
            1)
        clear : bool
            Flag indicating whether or not to clear the contents of the
            requested figure number before plotting

        Returns
        -------
        fig : MPL Figure
            The MPL figure of the created plot

        """
        import matplotlib.pyplot as plt
        
        # Get the particle size distributions from the model
        d_gas, vf_gas, d_liq, vf_liq = self.report_psds(loc, stage)
        
        # Prepare the figure for plotting
        # Prepare the figure for plotting
        try:
            if clear:
                fig.clear()
            axes = fig.subplots(2, 1)
        except AttributeError:  # not a figure object already
            fig = plt.figure(fig)
            if clear:
                fig.clear()
            fig.set_size_inches(8, 7)
            axes = fig.subplots(2, 1)
        
        # Plot the gas
        # ax = plt.subplot(211)
        ax = axes[0]
        ax.semilogx(d_gas * 1.e6, vf_gas, '.-')
        ax.set_xlabel('Gas bubble diameter, (um)')
        ax.set_ylabel('Volume fraction, (--)')
        ax.grid(True, which='major')
        ax.grid(True, which='minor')
    
        # Plot the oil
        # ax = plt.subplot(212)
        ax = axes[1]
        ax.semilogx(d_liq * 1.e6, vf_liq, '.-')
        ax.set_xlabel('Liquid droplet diameter, (um)')
        ax.set_ylabel('Volume fraction, (--)')
        ax.grid(True, which='major')
        ax.grid(True, which='minor')
    
        fig.set_tight_layout
        fig.canvas.draw_idle()

        return fig
    
    def plot_state_space(self, fig):
        """
        Plot the simulation state space

        Plot the standard set of state space variables used to evaluate
        the quality of the model solution

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


        See Also
        --------
        plot_all_variables

        """
        if self.sim_stored is False:
            print('No simulation results available to plot...')
            print('Plotting nothing.\n')
            return

        # Plot the results
        print('Plotting the state space...')
        fig = plot_state_space(self.t, self.q, self.q_local, self.profile, self.p,
                               self.particles, fig)
        print('Done.\n')
        return fig

    def plot_all_variables(self, fig):
        """
        Plot a comprehensive suite of simulation results

        Generate a comprehensive suite of graphs showing the state and
        derived variables along with ambient profile data in order to
        view the model output for detailed analysis.

        Parameters
        ----------
        fig : int
            Number of the figure window in which to draw the plot

        See Also
        --------
        plot_state_space

        """
        if self.sim_stored is False:
            print('No simulation results available to plot...')
            print('Plotting nothing.\n')
            return

        # Plot the results
        print('Plotting the full variable suite...')
        plot_all_variables(self.t, self.q, self.q_local, self.profile,
            self.p, self.particles, self.track, fig)
        print('Done.\n')
    
    def plot_fractions_dissolved(self, fig, chems=None, stage=1, fp_type=-1,
                                 clear=True):
        """
        Plot the fraction of each chem dissolved subsea
        
        Plot the fraction of each compound (chems) that is dissolved subsea 
        within either the near-field plume (stage=0) or after the far-field
        particle tracking (stage=1).
        
        Parameters
        ----------
        fig : int
            The figure number to plot
        chems : list, default=None
            A list of chemical compounds to track.  This may be a list of 
            chemical names or a list of index-numbers corresponding to 
            chemicals in the mixture composition.  If None, then all tracked
            compounds are included.
        stage : int, default=1
            Flag indicating whether the user wants data from the near-field
            plume (stage = 0), the Lagrangian particle tracking stage of
            transport (stage = 1), or the whole simulation. The idx value
            will index the respective model simulation vector that matches 
            the chosen stage of the solution.
        fp_type : int, default=-11
            A flag indicating which particles should be included. 0 -- will
            include only particles that were initially gas-phase, 1 -- will
            include only particles that were initially liquid-phase particles,
            and -1 -- will include all particles.
        clear : bool, default=True
            Flag indicating whether the figure should be cleared before 
            plotting
        
        """
        import matplotlib.pyplot as plt
        
        # Prepare the figure for plotting
        plt.figure(fig, figsize=(9,6))
        if clear:
            plt.clf()

        # Get the composition and indices
        composition = self.particles[0].particle.composition
        c_idx = chem_idx_list(chems, composition)
        
        # Get the mass fluxes at the beginning and end of the specified stage
        # of transport
        mp0, md0 = self.report_mass_fluxes(0, 0, chems, fp_type)
        mpp, mdp = self.report_mass_fluxes(-1, 0, chems, fp_type)
        if self.track:
            mps, mds = self.report_mass_fluxes(-1, 1, chems, fp_type)
        if stage == 0:
            mp1 = mpp
        elif stage == 1:
            mp1 = mp0 - (mpp - mps)
        else:
            mp1 = mps
        
        # Compute the percentage changes
        npart, ncomp = mp0.shape
        m0 = np.zeros(ncomp)
        m1 = np.zeros(ncomp)
        for i in range(npart):
            for j in range(ncomp):
                if mp1[i,j] >= 0.:
                    m1[j] += mp1[i,j]
                if mp0[i,j] >= 0.:
                    m0[j] += mp0[i,j]
        f = np.zeros(ncomp)
        for i in range(ncomp):
            if m0[i] != 0:
                f[i] = (m0[i] - m1[i]) / m0[i] * 100.
            else:
                f[i] = 0.
            if f[i] < 0.:
                f[i] = 0.
        
        # Get the composition names
        compounds = [composition[i] for i in c_idx]
        
        # Plot the data
        ax = plt.subplot(111)
        ax.plot(range(ncomp), f, '-o')
        ax.set_xticks(range(ncomp))
        ax.set_xticklabels(compounds, rotation=90)
        ax.set_ylabel('Fraction dissolved, (%)')
        plt.tight_layout()
        plt.show()
    
    def plot_mass_balance(self, fig, chems=None, fp_type=-1, t_max=-1, 
        clear=True):
        """
        Plot the time-history of the mass balance
        
        Plot the total mass in the ocean system as a function of time and 
        location, up to the duration of the simulation or a user-specified 
        time.  The figure plots the mass subsea, the mass at the surface,
        and the mass biodegraded.  The subsea mass includes all dissolved,
        bubble, and droplet mass that has not surfaced.  The surface mass
        only includes mass that has reached the surface.  The biodegraded 
        mass reports the fraction of subsea mass that has biodegraded.  Hence,
        the total subsea mass includes the mass that has biodegraded.
        
        Parameters
        ----------
        fig : int
            The figure number to plot
        chems : list, default=None
            A list of chemical compounds to track.  This may be a list of 
            chemical names or a list of index-numbers corresponding to 
            chemicals in the mixture composition.  If None, then all tracked
            compounds are included.
        fp_type : int, default=-11
            A flag indicating which particles should be included. 0 -- will
            include only particles that were initially gas-phase, 1 -- will
            include only particles that were initially liquid-phase particles,
            and -1 -- will include all particles.
        t_max : float, default=-1
            The maximum time to plot the mass history (days).  If -1, then 
            the maximum surfacing time in the simulation is used.
        clear : bool, default=True
            Flag indicating whether the figure should be cleared before 
            plotting
        
        """
        import matplotlib.pyplot as plt
        
        # Prepare the figure for plotting
        plt.figure(fig, figsize=(8,5))
        if clear:
            plt.clf()
        
        # Get the initial mass fluxes
        mp0, md0 = self.report_mass_fluxes(0, 0, chems, fp_type)
        npart, ncomp = mp0.shape
        
        # Get the mass fluxes at the surface
        mp, mc, tp, tc = self.report_surfacing_fluxes(chems, fp_type)
        
        # Find the maximum surfacing time in the dataset
        t_max_p = np.nanmax(tp)
        if t_max_p < np.nanmax(tc):
            t_max_p = tc
        
        # Make sure this time is more than a few seconds
        if t_max_p < 12.*3600.:
            t_max_p = 12.*3600.
        
        # Create a time series with 500 points within this range
        if t_max < 0:
            t = np.linspace(1., t_max_p, num=500)
        else:
            t = np.linspace(1., t_max, num=500)
        
        # Track the masses released and surfaced
        m0 = np.zeros(t.shape)
        ms = np.zeros(t.shape)
        md = np.zeros(t.shape)
        for i in range(len(t)):
            
            # Get the particle mass fluxes for this time step
            for j in range(npart):
                
                # ...at the release
                m0[i] += np.sum(mp0[j,:]) * t[i]
                
                # ...and at the surface
                if t[i] > tp[j]:
                    ms[i] += np.sum(mp[j,:]) * (t[i] - tp[j])
                
                # ...and the total amount of biodegradation
                particle = self.particles[j]
                for k in range(ncomp):
                    if particle.lag_time:
                        t_bio = particle.particle.t_bio[k]
                    else:
                        t_bio = 0.
                    if t[i] >= t_bio:
                        k_bio = particle.particle.k_bio[k]
                        if k_bio > 0.:
                            md[i] += mp0[j,k] * ((t[i] - t_bio) - 
                                1./k_bio * (1. - np.exp(-k_bio * (t[i] -
                                t_bio))))
            
            # And the dissolved plume flux (if applicable)
            if not np.isnan(tc):
                if t[i] >= tc:
                    ms[i] += np.sum(mc) * (t[i] - tc)

        # Plot the results
        f_surf = ms / m0 * 100.
        f_sub = (m0 - ms) / m0 * 100.
        f_bio = md / m0 * 100.
        plt.plot(t / 3600. / 24., f_surf)
        plt.plot(t / 3600. / 24., f_sub)
        plt.plot(t / 3600. / 24., f_bio)
        plt.legend(('Fraction surfacing', 'Fraction subsea', 
            'Fraction biodegraded'))
        plt.xlabel('Time, (days)')
        plt.ylabel('Mass, (%)')
        plt.tight_layout()
        plt.grid(True)
        
        print('\nMass balance statistics')
        print('---------------------')
        print('    Fraction surfacing =   %g' % f_surf[-1])
        print('    Fraction subsea =      %g' % f_sub[-1])
        print('    Fraction biodegraded = %g' % f_bio[-1])
        plt.show()


# ----------------------------------------------------------------------------
# Model Parameters objects
# ----------------------------------------------------------------------------

class ModelParams(single_bubble_model.ModelParams):
    """
    Fixed model parameters for the bent plume model

    This class stores the set of model parameters that should not be adjusted
    by the user and that are needed by the bent plume model.

    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.

    Attributes
    ----------
    alpha_j : float
        Jet shear entrainment coefficient.
    alpha_Fr : float
        Plume entrainment coefficient in Froude-number expression.
    gamma : float
        Momentum amplification factor
    Fr_0 : float
        Initial plume Froude number for the Wuest et al. (1992) multiphase
        plume initial conditions
    rho_r : float
        Reference density (kg/m^3) evaluated at mid-depth of the water body.
    g : float
        Acceleration of gravity (m/s^2)
    Ru : float
        Ideal gas constant (J/mol/K)

    Notes
    -----
    This object inherits all of the parameters of the
    `single_bubble_model.ModelParams` object.

    """
    def __init__(self, profile):
        super(ModelParams, self).__init__(profile)

        # Set the model parameters to the values in Jirka (2004)
        self.alpha_j = 0.057  # Changed from 0.055 on 11/20/2018
        self.alpha_Fr = 0.544
        self.gamma = 1.10

        # Set some of the multiphase plume model parameters
        self.Fr_0 = 1.6


# ----------------------------------------------------------------------------
# Particle object that handles tracking and exiting the plume
# ----------------------------------------------------------------------------

class Particle(dispersed_phases.PlumeParticle):
    """
    Special model properties for tracking inside a Lagrangian plume object

    This new `Particle` class is needed to allow dispersed phase particles to
    be tracked within the Lagrangian plume element during the solution and
    to exit the plume at the right time.

    This object inherits the `dispersed_phases.PlumeParticle` object and
    adds functionality for three-dimensional positioning and particle
    tracking.  All behavior not associated with tracking is identical to
    that in the `dispersed_phases.PlumeParticle` object.  Moreover, this
    object can be used in a `stratified_plume_model.Model` simulation.

    Parameters
    ----------
    x : float
        Initial position of the particle in the x-direction (m)
    y : float
        Initial position of the particle in the y-direction (m)
    z : float
        Initial position of the particle in the z-direction (m)
    dbm_particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Object describing the particle properties and behavior
    m0 : ndarray
        Initial masses of one particle for the components of the
        `dbm_particle` object (kg)
    T0 : float
        Initial temperature of the of `dbm` particle object (K)
    nb0 : float
        Initial number flux of particles at the release (#/s)
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
    fdis : float, default = 1.e-6
        Fraction (--) of the initial mass of each component of the mixture
        when that component should be considered totally dissolved.
    t_hyd : float, default = 0.
        Hydrate film formation time (s).  Mass transfer is computed by clean
        bubble methods for t less than t_hyd and by dirty bubble methods
        thereafter.  The default behavior is to assume the particle is dirty
        or hydrate covered from the release.
    lag_time : bool, default = True.
        Flag that indicates whether (True) or not (False) to use the
        biodegradation lag times data.

    Attributes
    ----------
    particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Stores the `dbm_particle` object passed to at creation.
    composition : str list
        Copy of the `composition` attribute of the `dbm_particle` object.
    m0 : ndarray
        Initial masses (kg) of one particle for the particle components
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
    t_hyd : float
        Formation time for a hydrate skin (s)
    nb0 : float
        Initial number flux of particles at the release (#/s)
    nbe : float
        Number of particles associated with a Lagrangian element (#).  This
        number with the mass per particle sets the total mass of particles
        inside the Lagrangian element at any given time.  This value is set
        by `lmp.bent_plume_ic`.
    lambda_1 : float
        Spreading rate of the dispersed phase in a plume (--)
    m : ndarray
        Masses of the particle components for a single particle (kg)
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
    T : float
        Temperature of the particle (K)
    integrate : bool
        Flag indicating whether or not the particle is still inside the plume,
        where its trajectory should continue to be integrated.
    t : float
        Current time since the particle was released (s)
    x : float
        Current position of the particle in the x-direction (m)
    y : float
        Current position of the particle in the y-direction (m)
    z : float
        Current position of the particle in the z-direction (m)
    p_fac : float
        Buoyant force reduction factor due to a reduced buoyancy as the
        particle moves to the edge of the plume (--)
    b_local : float
        Width of the bent plume model at the location where the particle
        exited the plume.
    sbm : `single_bubble_model.Model` object
        Model object for tracking the particle outside the plume


    See Also
    --------
    dispersed_phases.SingleParticle, dispersed_phases.PlumeParticle

    """
    def __init__(self, x, y, z, dbm_particle, m0, T0, nb0, lambda_1,
                 P, Sa, Ta, K=1., K_T=1., fdis=1.e-6, t_hyd=0.,
                 lag_time=True):
        super(Particle, self).__init__(dbm_particle, m0, T0, nb0, lambda_1,
                                       P, Sa, Ta, K, K_T, fdis, t_hyd,
                                       lag_time)

        # Particles start inside the plume and should be integrated
        self.integrate = True
        self.sim_stored = False
        self.farfield = False

        # Store the initial particle locations
        self.t = 0.
        self.x = x
        self.y = y
        self.z = z

        # Update the particle with its current properties
        self.update(m0, T0, P, Sa, Ta, self.t)

    def track(self, t_p, X_cl, X_p, q_local, Ainv=None):
        """
        Track the particle in the Lagragian plume model

        Track the location of the particle within a Lagrangian plume model
        element and stop the integration when the particle exits the plume.

        Parameters
        ----------
        t_p : float
            Time since the particle was released (s)
        X_cl : ndarray
            Array of Cartesian coordinates (x,y,z) for the plume centerline
            (m).
        X_p : ndarray
            Array of local plume coordinates (l,n,m) for the current
            particle position (m)().  This method converts these coordinates,
            which are solved by the bent plume model state space solution, to
            Cartesian coordinates.
        q_local : `LagElement` object
            Object that translates the bent plume model state space `t` and
            `q` into the comprehensive list of derived variables.
        Ainv : ndarray, default = None
            Coordinate transformation matrix from the local plume coordinates
            (l,n,m) to Cartesian coordinates (x,y,z).  If `Ainv` is known, it
            can be passed to this function; otherwise, this function can
            solve for `Ainv` using q_local.

        Returns
        -------
        xp : ndarray
            Array of Cartesian coordinates (x,y,z) for the current particle
            position (m).

        """
        if self.integrate:
            # Compute the transformation matrix from local plume coordinates
            # (l,n,m) to Cartesian coordinates (x,y,z) if needed
            if Ainv is None:
                A = lmp.local_coords(q_local, q_local, 0.)
                Ainv = inv(A)

            # Update the particle age
            self.t = t_p
            tp = self.t

            # Get the particle position
            xp = np.dot(Ainv, X_p) + X_cl
            self.x = xp[0]
            self.y = xp[1]
            self.z = xp[2]

            # Compute the particle offset from the plume centerline
            lp = np.sqrt(X_p[0]**2 + X_p[1]**2 + X_p[2]**2)

            # Compute the buoyant force reduction factor
            self.p_fac = (q_local.b - lp)**4 / q_local.b**4
            if self.p_fac < 0.:
                self.p_fac = 0.
            
            # Store the plume width
            self.b_local = q_local.b

            # Check if the particle exited the plume
            if lp > self.b_local:
                self.p_fac = 0.

        else:
            # Return the time and position when the particle exited the plume
            tp = self.te
            xp = np.array([self.xe, self.ye, self.ze])
            self.p_fac = 0.

        # Return the particle position as a matrix
        return (tp, xp)

    def outside(self, Ta, Sa, Pa):
        """
        Remove the effect of particles if they are outside the plume

        Sets all of the particle properties that generate forces or
        dissolution to zero effect if the particle is outside the plume.

        Parameters
        ----------
        Ta : float
            Local temperature surrounding the particle (K)
        Sa : float
            Local salinity surrounding the particle (psu)
        Pa : float
            Local pressure (Pa)

        """
        self.us = 0.
        self.rho_p = seawater.density(Ta, Sa, Pa)
        self.A = 0.
        self.Cs = np.zeros(len(self.composition))
        self.beta = np.zeros(len(self.composition))
        self.beta_T = 0.
        self.T = Ta

    def run_sbm(self, profile):
        """
        Run the `single_bubble_model` to track particles outside the plume

        Continues the simulation of the particle is outside of the plume using
        the `single_bubble_model`.  The object containing the simulation
        and simulation results will be added to the attributes of this
        Particle object.

        Parameters
        ----------
        profile : `ambient.Profile` object
            Ambient CTD data for the model simulation

        """
        # Create the simulation object
        self.sbm = single_bubble_model.Model(profile)

        # Create the inputs to the sbm.simulate method
        X0 = np.array([self.x, self.y, self.z])
        Ta, Sa, P = profile.get_values(X0[2], ['temperature', 'salinity',
                    'pressure'])
        de = self.diameter(self.m, self.T, P, Sa, Ta)
        if self.particle.issoluble:
            yk = self.particle.mol_frac(self.m)
        else:
            yk = 1.
        if self.t_hyd > 0.:
            if self.t > self.t_hyd:
                t_hyd = 0.
            else:
                t_hyd = self.t_hyd - self.t
        else:
            t_hyd = 0.

        # Run the simulation
        if not np.isnan(de):
            self.sbm.simulate(self.particle, X0, de, yk, self.T, self.K,
                self.K_T, self.fdis, t_hyd, self.lag_time, delta_t=1000.)

            # Set flag indicating that far-field solution was computed
            self.farfield = True
        
            # Prepare for computing far-field concentrations
            self._create_concentration_model()
        else:
            print('Particle component masses (d = %g mm):' % (de * 1000.))
            for i in range(len(self.m)):
                print('    %s:  %g (kg)' % (self.composition[i], self.m[i]))
            
    
    def _create_concentration_model(self, k=1000):
        """
        Internal method to setup parameters to compute far-field concentrations
        
        The far-field concentration model relies on an analytical solution for
        a continuous line source. This method creates the variables that remain
        constant in this solution for the present `Particle`. These values are
        stored in the attributes described below. This method automatically is
        computed after running the `single_bubble_model` particle tracking
        (e.g., `Particle.run_sbm`) and only needs to be computed once before
        making far-field concentration calculations using `point_concentration`
        or `field_concentrations`.
        
        Parameters
        ----------
        k : int
            Number of random particle locations to include in the analytical
            model.
        
        Attributes
        ----------
        xp : ndarray
            Array of `k` random x-coordinates for the particle locations in the 
            analytical model
        yp : ndarray
            Array of `k` random y-coordinates for the particle locations in the
            analytical model
        xh : scipy.interpolate.interp1d
            An interpolation function that returns the particle x- and
            y-coordinate along its trajectory as a function of depth `z`
        currents : scipy.interpolate.interp1d
            An interpolation function that returns the current speed (m/s) and
            angle from the x-axis (rad, -pi to pi) as a function of depth `z`
        age : scipy.interpolate.interp1d     
            An interpolation function that returns the time (s) since the start
            of the particle trajectory as a function of depth `z`
        md_p : scipy.interpolate.interp1d
            An interpolation function that returns the mass flux (kg/m/s) of
            each compound in the `Particle` composition as a function of depth
            `z`
        solubility : scipy.interpolate.interp1d
            An interpolation function that returns the solubility (kg/m^3) of
            each compound in the `Particle` composition as a function of depth
            `z`
        z_min : float
            Shallowest depth included in the particle trajectory; hence, the
            shallowest depth available within the interpolation functions.
        z_max : float    
            Deepest depth included in the particle trajectory; hence, the
            deepest depth available within the interpolation functions.
        sigma_0 : float
            The width of the near-field plume at the start of the far-field
            trajectory for this `Particle`
        
        """
        # Get random positions for k realizations of particle position assuming
        # a unit standard deviation
        from scipy.stats import norm
        self.xp = norm.rvs(0., scale=1., size=k)
        self.yp = norm.rvs(0., scale=1., size=k)
        
        # Store the particle trajectory and ambient currents
        t = np.zeros(len(self.sbm.t))
        z = np.zeros(len(self.sbm.t))
        xh = np.zeros((len(self.sbm.t), 2))
        currents = np.zeros((len(self.sbm.t), 2))
        for i in range(len(self.sbm.t)):
            t[i] = self.sbm.t[i]
            z[i] = self.sbm.y[i,2]
            xh[i,0:2] = self.sbm.y[i,0:2]
            ua, va = self.sbm.profile.get_values(z[i], ['ua', 'va'])
            currents[i,0] = np.sqrt(ua**2 + va**2)
            currents[i,1] = np.arctan2(va, ua)
        
        # Create interpolation functions to access these data at any height
        from scipy.interpolate import interp1d
        self.xh = interp1d(z, xh, axis=0)
        self.currents = interp1d(z, currents, axis=0)
        self.age = interp1d(z, t)
        self.z_min = np.min(z)
        self.z_max = np.max(z)
            
        # Compute the dissolution rate for each chemical along the particle
        # trajectory
        md_p = np.zeros((len(self.sbm.t), len(self.particle.composition)))
        Cs = np.zeros((len(self.sbm.t), len(self.particle.composition)))
        if not isinstance(self.sbm.particle.particle, dbm.InsolubleParticle):
            for i in range(len(self.sbm.t)):
                m = self.sbm.y[i,3:-1]
                T = self.sbm.y[i,-1] / (np.sum(m) * self.sbm.particle.cp)
                Ta, Sa, Pa = self.sbm.profile.get_values(z[i], ['temperature',
                    'salinity', 'pressure'])
                (us, rho_p, A, Cs[i,:], beta, beta_t, T) = \
                    self.sbm.particle.properties(m, T, Pa, Sa, Ta, t[i])
                Ca = self.sbm.profile.get_values(z[i], 
                    self.particle.composition)
                md_p[i,:] = A * self.nb0 * beta / us * (Cs[i,:] - Ca)
        
        # When the atmospheric gases are stripped from the water column, 
        # the dissolution rate appears negative...it should be zero
        md_p[md_p < 0.] = 0.
        
        # Store the dissolution rates in an interpolation function
        self.md_p = interp1d(z, md_p, axis=0)
        self.solubility = interp1d(z, Cs, axis=0)
        
        # Store the Gaussian standard-deviation for the plume width at the point
        # where this particle exited the plume
        self.sigma_0 = self.b_local / 2.
        
    def point_concentration(self, x, max_C=True):
        """
        Compute the dissolved concentration at a point downstream of particle
        
        Compute the dissolved concentration in seawater at a point downstream
        of the present particle. This method utilizes the analytical solution
        for a continuous line source and computes the concentrations of all
        compounds in the particle composition. Because each 'Particle' is a
        stream of bubbles or droplets that are spreading out, this method
        simulates the random locations of 1000 particles and superposes their
        concentration fields to get the downstream concentration.
        
        Parameters
        ----------
        x : ndarray
            Array containing the three coordinates of the x, y, and z position
            (m) where the concentration is to be computed
        max_C : bool, default=True
            A flag indicating how to interpret the horizontal coordinates x and
            y. The maximum concentration will occur along a line parallel to
            the currents, which may change direction with height. If `max_C` is
            `True`, then the x-coordinate is taken as the distance along the
            current direction and the y-coordinate is taken as the distance
            perpendicular to x. Thus, if y=0, then this will always return the
            maximum concentration a distance x from the particle center of
            mass. If `max_C` is `False`, then the x- and y-coordinates are to
            be taken as absolute coordinates in the reference frame of the
            release. If they are located upstream of the particle center, the
            concentration will be zero; otherwise, the analytical solution is
            computed and the corresponding concentrations returned.
        
        Returns
        -------
        Cp : ndarray
            Array of the concentrations (kg/m^3) at the given point. Each
            element of the array corresponds to a compound in the `Particle`
            composition.
        
        """
        # Check whether the desired location is within the vertical bounds of this
        # particle
        if x[2] < self.z_min or x[2] > self.z_max:
            return np.zeros(len(self.particle.composition))
        
        # Extract the currents
        Ua = self.currents(x[2])[0]
        theta = self.currents(x[2])[1]
        if theta < 0.:
            theta = 2. * np.pi + theta
        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),
            np.cos(theta)]])
        
        # Get the desired point in a currents-based and particle-center based
        # Coordinate system
        if max_C:
            # The user specified the downstream distance and lateral offset
            Lx = x[0]
            Ly = x[1]
        else:
            # The user has provided absolute coordinates in the model 
            # coordinate system
            alpha = np.arctan2(x[1], x[0])
            if alpha < 0.:
                alpha = 2. * np.pi + alpha
            if np.abs(theta - alpha) < np.pi / 2.:
                # The chosen point is in the downstream direction: a non-zero
                # solution for concentration exists
                xs = np.matmul(R.transpose(), x[0:2])
                Lx = xs[0]
                Ly = xs[1]
                
            else:
                # The chosen point is upstream of the particles: the concentrations
                # will all be zero
                return np.zeros(len(self.particle.composition))
        
        # Get the downstream distance to the current particle center
        xp = np.matmul(R, self.xh(x[2]))
        Lp = np.sqrt(xp[0]**2 + xp[1]**2)
        
        # Get the apparent turbulent diffusion coefficient for bubble
        # spreading from the Okubo diagram using equations (9.56) and (9.57)
        # in Chen (2013)
        Et = (0.0108 / 4. * self.age(x[2])**1.34) / 100.**2
        
        # Get the local width of the bubble cloud using Equation (9-57) in 
        # Chen (2013)
        sigma = np.sqrt(4. * Et * self.age(x[2]) + self.sigma_0**2)
        
        # Get locations for the random particle positions
        sx = Lp + sigma * self.xp
        sy = 0. + sigma * self.yp
        k = len(self.xp)
        
        # Compute length to advection-dominated region downstream of a 
        # Single bubble using the diffusivity measured by Wang et al. (2020)
        alpha = 10.
        Ua = self.currents(x[2])[0]
        Et_0 = 5.0e-4
        L_D = Et_0 / (alpha * Ua)
        
        # Compute the contributions from each random particle position
        md_p = self.md_p(x[2])
        Cp = np.zeros(len(md_p))
        for j in range(k):
            # Get distance along s from particle to plane of interest
            L = Lx - sx[j]
            H = Ly - sy[j]
                        
            # Get travel time to this position...along L
            if L > 0:
                # Compute the apparent diffusion coefficient using the Okubo
                # diagram and the travel time with the currents
                t = L / self.currents(x[2])[0]
                Et = (0.0108 / 4. * t**1.34) / 100.**2
            else:
                # L is negative...use the diffusivity for a single bubble
                Et = 5.0e-4
                t = 0.
            
            # Only include this particle if in advection-dominated region
            if L > L_D:
                Cp += md_p / np.float64(k) / np.sqrt(4. * np.pi * L * Ua * Et) * \
                    np.exp(-(Ua * H**2) / (4. * Et * L))
        
        # Concentration cannot be higher than saturation
        Cs = self.solubility(x[2])
        for i in range(len(self.particle.composition)):
            if Cp[i] > Cs[i]:
                Cp[i] = Cs[i]
        
        return Cp
    
    def grid_concentrations(self, x, max_C=True):
        """
        Compute the dissolved concentrations for points in the far-field
        
        Computes the dissolved concentrations in seawater for each compound in
        the Particle composition at each point in a vector of locations, x.
        This method uses `point_concentration` to compute the concentration,
        iteratively cycling through all the points in the given `x` array.
        
        Parameters
        ----------
        x : ndarray
            Array of three-dimensional positions where the concentrations should
            be computed. Each row of `x` corresponds to a different point, with
            the columns of `x` giving the x-, y-, and z-coordinates of each 
            point.
        max_C : bool, default=True            
            A flag indicating how to interpret the horizontal coordinates x and
            y. The maximum concentration will occur along a line parallel to
            the currents, which may change direction with height. If `max_C` is
            `True`, then the x-coordinate is taken as the distance along the
            current direction and the y-coordinate is taken as the distance
            perpendicular to x. Thus, if y=0, then this will always return the
            maximum concentration a distance x from the particle center of
            mass. If `max_C` is `False`, then the x- and y-coordinates are to
            be taken as absolute coordinates in the reference frame of the
            release. If they are located upstream of the particle center, the
            concentration will be zero; otherwise, the analytical solution is
            computed and the corresponding concentrations returned.
        
        Returns
        -------
        C_vals : ndarray
            A two-dimensional array of concentrations at the given points. Each
            row of `C_vals` corresponds to a row of `x`; the columns of `C_vals`
            each correspond to a compound in the `Particle` composition.
         
        """
        # Set up a matrix to store solutions
        C_vals = np.zeros((x.shape[0], len(self.particle.composition)))
        
        # Loop through all the points
        for i in range(x.shape[0]):
            C_vals[i,:] = self.point_concentration(x[i,:], max_C)
         
        # Return the solution set
        return C_vals


# ----------------------------------------------------------------------------
# Object to translate the state space into all the derived variables
# ----------------------------------------------------------------------------

class LagElement(object):
    """
    Manages the Lagragian plume element state space and derived variables

    Translates the state space variables for a Lagrangian plume element into
    its individual parts and derived quantitites.

    Parameters
    ----------
    t0 : float
        Initial time of the simulation (s)
    q0 : ndarray
        Initial values of the simulation state space, q
    D : float
        Diameter for the equivalent circular cross-section of the release (m)
    profile : `ambient.Profile`
        Ambient CTD data
    p : `ModelParams`
        Container for the fixed model parameters
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the
        simulation
    tracers : string list
        List of passive tracers in the discharge.  These can be chemicals
        present in the ambient `profile` data, and if so, entrainment of these
        chemicals will change the concentrations computed for these tracers.
        However, none of these concentrations are used in the dissolution of
        the dispersed phase.  Hence, `tracers` should not contain any
        chemicals present in the dispersed phase particles.
    chem_names : string list
        List of chemical parameters to track for the dissolution.  Only the
        parameters in this list will be used to set background concentration
        for the dissolution, and the concentrations of these parameters are
        computed separately from those listed in `tracers` or inputed from
        the discharge through `cj`.

    Attributes
    ----------
    t0 : float
        Initial time of the simulation (s)
    q0 : ndarray
        Initial values of the simulation state space, q
    D : float
        Diameter for the equivalent circular cross-section of the release (m)
    tracers : string list
        List of passive tracers in the discharge.  These can be chemicals
        present in the ambient `profile` data, and if so, entrainment of these
        chemicals will change the concentrations computed for these tracers.
        However, none of these concentrations are used in the dissolution of
        the dispersed phase.  Hence, `tracers` should not contain any
        chemicals present in the dispersed phase particles.
    chem_names : string list
        List of chemical parameters to track for the dissolution.  Only the
        parameters in this list will be used to set background concentration
        for the dissolution, and the concentrations of these parameters are
        computed separately from those listed in `tracers` or inputed from
        the discharge through `cj`.
    len : int
        Number of variables in the state space q (--)
    ntracers : int
        Number of passive chemical tracers (--)
    nchems : int
        Number of chemicals tracked for dissolution of the dispersed phase
        particles (--)
    np : int
        Number of dispersed phase particles (--)
    t : float
        Independent variable for the current time (s)
    q : ndarray
        Dependent variable for the current state space
    M : float
        Mass of the Lagrangian element (kg)
    Se : float
        Salt in the Lagrangian element (psu kg)
    He : float
        Heat of the Lagrangian element (J)
    Jx : float
        Dynamic momentum of the Lagrangian element in the x-direction
        (kg m/s)
    Jy : float
        Dynamic momentum of the Lagrangian element in the y-direction
        (kg m/s)
    Jz : float
        Dynamic momentum of the Lagrangian element in the z-direction
        (kg m/s)
    H : float
        Relative thickness of the Lagrangian element h/V (s)
    x : float
        Current x-position of the Lagrangian element (m)
    y : float
        Current y-position of the Lagrangian element (m)
    z : float
        Current z-position of the Lagrangian element (m)
    s : float
        Current s-position along the centerline of the plume for the
        Lagrangian element (m)
    M_p : dict of ndarrays
        For integer key: the total mass (kg) of each component in a particle
        that are in the Lagrangian element.  The mass per particle is 
        M_p / nbe, where nbe is the number of particles in the Lagrangian 
        element.  The total mass flux in particles of M_p / nbe * nb0, 
        where nb0 is the initial number flux of particles.
    H_p : ndarray
        Total heat flux for each particle (J/s)
    t_p : ndarray
        Time since release for each particle (s)
    X_p : ndarray
        Position of each particle in local plume coordinates (l,n,m) (m).
    cpe : ndarray
        Masses of the chemical components involved in dissolution (kg)
    cte : ndarray
        Masses of the passive tracers in the plume (concentration kg)
    Pa : float
        Ambient pressure at the current element location (Pa)
    Ta : float
        Ambient temperature at the current element location (K)
    Sa : float
        Ambient salinity at the current element location (psu)
    ua : float
        Crossflow velocity in the x-direction at the current element location
        (m/s)
    ca_chems : ndarray
        Ambient concentration of the chemical components involved in
        dissolution at the current element location (kg/m^3)
    ca_tracers :
        Ambient concentration of the passive tracers in the plume at the
        current element location (concentration)
    rho_a : float
        Ambient density at the current element location (kg/m^3)
    S : float
        Salinity of the Lagrangian element (psu)
    T : float
        Temperature of the Lagrangian element (T)
    c_chems :
        Concentration of the chemical components involved in dissolution for
        the Lagrangian element (kg/m^3)
    c_tracers :
        Concentration of the passive tracers in the Lagrangian element
        (concentration)
    u : float
        Velocity in the x-direction of the Lagrangian element (m/s)
    v : float
        Velocity in the y-direction of the Lagrangian element (m/s)
    w : float
        Velocity in the z-direction of the Lagrangian element (m/s)
    hvel : float
        Velocity in the horizontal plane for the Lagrangian element (m/s)
    V : float
        Velocity in the s-direction of the Lagrangian element (m/s)
    h : float
        Current thickness of the Lagrangian element (m)
    rho : float
        Density of the entrained seawater in the Lagrangian element (kg/m^3)
    b : float
        Half-width of the Lagrangian element (m)
    sin_p : float
        The sine of the angle phi (--)
    cos_p : float
        The cosine of the angle phi (--)
    sin_t : float
        The sine of the angle theta (--)
    cos_t : float
        The cosine of the angle theta (--)
    phi : float
        The vertical angle from horizontal of the current plume trajectory
        (rad in range +/- pi/2).  Since z is positive down (depth), phi =
        pi/2 point down and -pi/2 points up.
    theta : float
        The lateral angle in the horizontal plane from the x-axis to the
        current plume trajectory (rad in range 0 to 2 pi)
    mp : ndarray
        Masses of each of the dispersed phase particles in the `particles`
        variable
    fb : ndarray
        Buoyant force for each of the dispersed phase particles in the
        `particles` variable as density difference (kg/m^3)
    x_p : ndarray
    Mp : float
        Total mass of dispersed phases in the Lagrangian element (kg)
    Fb : float
        Total buoyant force as density difference of the dispersed phases in
        the Lagrangian element (kg/m^3)

    """
    def __init__(self, t0, q0, D, profile, p, particles, tracers,
                 chem_names):
        super(LagElement, self).__init__()

        # Store the inputs to stay with the Lagrangian element
        self.t0 = t0
        self.q0 = q0
        self.D = D
        self.tracers = tracers
        self.chem_names = chem_names
        self.len = q0.shape[0]
        self.ntracers = len(self.tracers)
        self.nchems = len(self.chem_names)
        self.np = len(particles)

        # Extract the state variables and compute the derived quantities
        self.update(t0, q0, profile, p, particles)

    def update(self, t, q, profile, p, particles=[]):
        """
        Update the `LagElement` object with the current local conditions

        Extract the state variables and compute the derived quantities given
        the current local conditions.

        Parameters
        ----------
        t : float
            Current time of the simulation (s)
        q : ndarray
            Current values of the simulation state space, q
        profile : `ambient.Profile`
            Ambient CTD data
        p : `ModelParams`
            Container for the fixed model parameters
        particles : list of `Particle` objects
            List of `Particle` objects describing each dispersed phase in the
            simulation

        """
        # Save the current state space
        self.t = t
        self.q = q

        # Extract the state-space variables from q
        self.M = q[0]
        self.Se = q[1]
        self.He = q[2]
        self.Jx = q[3]
        self.Jy = q[4]
        self.Jz = q[5]
        self.H = q[6]
        self.x = q[7]
        self.y = q[8]
        self.z = q[9]
        self.s = q[10]
        idx = 11
        M_p = {}
        H_p = []
        t_p = []
        X_p = []
        for i in range(self.np):
            M_p[i] = q[idx:idx + particles[i].particle.nc]
            idx += particles[i].particle.nc
            H_p.extend(q[idx:idx + 1])
            idx += 1
            t_p.extend(q[idx:idx + 1])
            idx += 1
            X_p.append(q[idx:idx + 3])
            idx += 3
        self.M_p = M_p
        self.H_p = np.array(H_p)
        self.t_p = np.array(t_p)
        self.X_p = np.array(X_p)
        self.cpe = q[idx:idx + self.nchems]
        idx += self.nchems
        if self.ntracers >= 1:
            self.cte = q[idx:]
        else:
            self.cte = np.array([])

        # Get the local ambient conditions
        self.Pa, self.Ta, self.Sa, self.ua, self.va, self.wa = \
            profile.get_values(self.z, ['pressure', 'temperature',
            'salinity', 'ua', 'va', 'wa'])
        self.ca_chems = profile.get_values(self.z, self.chem_names)
        self.ca_tracers = profile.get_values(self.z, self.tracers)
        self.rho_a = seawater.density(self.Ta, self.Sa, self.Pa)

        # Compute the derived quantities
        self.S = self.Se / self.M
        self.T = self.He / (self.M * seawater.cp())
        self.rho = seawater.density(self.T, self.S, self.Pa)
        self.c_chems = self.cpe / (self.M / self.rho)
        self.c_tracers = self.cte / (self.M / self.rho)
        self.u = self.Jx / self.M
        self.v = self.Jy / self.M
        self.w = self.Jz / self.M
        self.hvel = np.sqrt(self.u**2 + self.v**2)
        self.V = np.sqrt(self.hvel**2 + self.w**2)
        self.h = self.H * self.V
        self.b = np.sqrt(self.M / (self.rho * np.pi * self.h))
        self.sin_p = self.w / self.V
        self.cos_p = self.hvel / self.V
        if self.hvel == 0.:
            # if hvel = 0, flow is purely along z; let theta = 0
            self.sin_t = 0.
            self.cos_t = 1.
        else:
            self.sin_t = self.v / self.hvel
            self.cos_t = self.u / self.hvel
        self.phi = np.arctan2(self.w, self.hvel)
        self.theta = np.arctan2(self.v, self.u)

        # Compute the transformation matrix from the local plume coordinates
        # (l,n,m) to Cartesian coordinates (x,y,z)
        Ainv = inv(lmp.local_coords(self, self, 0.))

        # Get the particle characteristics
        self.mp = np.zeros(self.np)
        self.fb = np.zeros(self.np)
        self.x_p = np.zeros((self.np, 3))
        self.de = np.zeros(self.np)
        
        for i in range(self.np):
            # If this is a post-processing call, update the status of the
            # integration flag
            if particles[i].sim_stored:
                if np.isnan(self.X_p[i][0]):
                    particles[i].integrate = False
                else:
                    particles[i].integrate = True

            # Update the particles with their current properties
            m_p = self.M_p[i] / particles[i].nbe
            T_p = self.H_p[i] / (np.sum(self.M_p[i]) * particles[i].cp)
            particles[i].update(m_p, T_p, self.Pa, self.S, self.T,
                                self.t_p[i])
            
            # Store the particle diameters (not used by model, but important
            # diagnostic parameter)
            self.de[i] = particles[i].diameter(m_p, T_p, self.Pa, self.Sa,
                self.Ta)

            # Store biodegradation rates to use with dissolved phase
            if particles[i].particle.issoluble:
                self.k_bio = particles[i].k_bio
            else:
                self.k_bio = 0.

            # Track the particle in the plume
            self.t_p[i], self.x_p[i,:] = particles[i].track(self.t_p[i],
                            np.array([self.x, self.y, self.z]),
                            self.X_p[i], self, Ainv)

            # Get the mass of particles following this Lagrangian element
            self.mp[i] = np.sum(m_p) * particles[i].nbe

            # Compute the buoyant force coming from this set of particles
            self.fb[i] = self.rho / particles[i].rho_p * self.mp[i] * \
                         (self.rho_a - particles[i].rho_p) * \
                         particles[i].p_fac

            # Force the particle mass and bubble force to zero if the bubble
            # has dissolved
            if self.rho == particles[i].rho_p:
                self.mp[i] = 0.
                self.fb[i] = 0.

        # Compute the net particle mass and buoyant force
        self.Fb = np.sum(self.fb)


# ----------------------------------------------------------------------------
# Functions to plot output from the simulations
# ----------------------------------------------------------------------------

def plot_state_space(t, q, q_local, profile, p, particles, fig):
    """
    Plot the Lagrangian model state space

    Plot the standard set of state space variables used to evaluate the
    quality of the model solution

    Parameters
    ----------
    t : ndarray
        Array of times computed in the solution (s)
    q : ndarray
        Array of state space values computed in the solution
    q_local : `LagElement` object
        Object that translates the `Model` state space `t` and `q` into the
        comprehensive list of derived variables.
    profile : `ambient.Profile`
        Ambient CTD data
    p : `ModelParams`
        Container for the fixed model parameters
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the
        simulation
    fig : int or MPL Figure object
        MPL Figure() on which to plot
        or
        Number of the figure window in which to draw the plot
        (Figure will be created for you with the provided fig number)

    Returns
    -------
    fig : MPL Figure
        The MPL figure of the created plot


    Notes
    -----
    Plots the trajectory of the jet centerline, the trajectory of the
    simulated particles, and the Lagrangian element mass.

    """
    import matplotlib.pyplot as plt

    # Extract the trajectory variables
    x = q[:,7]
    y = q[:,8]
    z = q[:,9]
    s = q[:,10]
    M = q[:,0]

    # Extract the particle positions from the q state space
    xp = np.zeros((len(t),3*len(particles)))
    for i in range(len(t)):
        q_local.update(t[i], q[i,:], profile, p, particles)
        for j in range(len(particles)):
            xp[i,j*3:j*3+3] = q_local.x_p[j,:]

    # Set up the figure
    try:
        # If it's already a Figure object
        axes = fig.subplots(2, 2)
    except AttributeError:  # not a figure object already
        fig = plt.figure(fig)
        # this fig size works OK with all the axis labels
        width = 9.5
        fig.set_size_inches(width, width * 0.75)
        axes = fig.subplots(2, 2)
    ax1, ax2, ax3, ax4 = axes.flat

    # Plot the figure
    # x-z plane
    ax1.plot(x, z)
    for i in range(len(particles)):
        ax1.plot(xp[:, i * 3], xp[:, i * 3 + 2], '.--')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('Depth (m)')
    ax1.invert_yaxis()
    ax1.grid(which='major', color='0.65', linestyle='-')

    # y-z plane
    ax2.plot(y, z)
    for i in range(len(particles)):
        ax2.plot(xp[:, i * 3 + 1], xp[:, i * 3 + 2], '.--')
    ax2.set_xlabel('y (m)')
    ax2.set_ylabel('Depth (m)')
    ax2.invert_yaxis()
    ax2.grid(which='major', color='0.65', linestyle='-')

    # x-y plane
    ax3.plot(x, y)
    for i in range(len(particles)):
        ax3.plot(xp[:, i * 3], xp[:, i * 3 + 1], '.--')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.grid(which='major', color='0.65', linestyle='-')

    # M-s plane
    ax4.plot(s, M)
    ax4.set_xlabel('s (m)')
    ax4.set_ylabel('M (kg)')
    ax4.grid(which='major', color='0.65', linestyle='-')

    fig.canvas.draw_idle()

    return fig

def plot_all_variables(t, q, q_local, profile, p, particles,
                       tracked, fig):
    """
    Plot a comprehensive suite of simulation results

    Generate a comprehensive suite of graphs showing the state and
    derived variables along with ambient profile data in order to
    view the model output for detailed analysis.

    Parameters
    ----------
    t : ndarray
        Array of times computed in the solution (s)
    q : ndarray
        Array of state space values computed in the solution
    q_local : `LagElement` object
        Object that translates the `Model` state space `t` and `q` into the
        comprehensive list of derived variables.
    profile : `ambient.Profile`
        Ambient CTD data
    p : `ModelParams`
        Container for the fixed model parameters
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the
        simulation
    tracked : bool
        Flag indicating whether or not the `single_bubble_model` was run to
        track the particles.
    fig : int
        Number of the figure window in which to draw the plot
    """
    # Don't offset any of the axes
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    formatter = mpl.ticker.ScalarFormatter(useOffset=False)

    # Create a second Lagrangian element in order to compute entrainment
    q0_local = LagElement(t[0], q[0,:], q_local.D, profile, p, particles,
        q_local.tracers, q_local.chem_names)
    n_part = q0_local.np
    pchems = 1
    for i in range(n_part):
        if len(particles[i].composition) > pchems:
            pchems = len(particles[i].composition)

    # Store the derived variables
    M = np.zeros(t.shape)
    S = np.zeros(t.shape)
    T = np.zeros(t.shape)
    Mpf = np.zeros((len(t), n_part, pchems))
    Hp = np.zeros((len(t), n_part))
    Mp = np.zeros((len(t), n_part))
    Tp = np.zeros((len(t), n_part))
    xp = np.zeros((len(t), 3*n_part))
    u = np.zeros(t.shape)
    v = np.zeros(t.shape)
    w = np.zeros(t.shape)
    V = np.zeros(t.shape)
    h = np.zeros(t.shape)
    x = np.zeros(t.shape)
    y = np.zeros(t.shape)
    z = np.zeros(t.shape)
    s = np.zeros(t.shape)
    rho = np.zeros(t.shape)
    b = np.zeros(t.shape)
    cos_p = np.zeros(t.shape)
    sin_p = np.zeros(t.shape)
    cos_t = np.zeros(t.shape)
    sin_t = np.zeros(t.shape)
    rho_a = np.zeros(t.shape)
    Sa = np.zeros(t.shape)
    Ta = np.zeros(t.shape)
    ua = np.zeros(t.shape)
    E = np.zeros(t.shape)

    for i in range(len(t)):
        if i > 0:
            q0_local.update(t[i-1], q[i-1,:], profile, p, particles)
        q_local.update(t[i], q[i,:], profile, p, particles)
        M[i] = q_local.M
        S[i] = q_local.S
        T[i] = q_local.T
        for j in range(n_part):
            Mpf[i,j,0:len(q_local.M_p[j])] = q_local.M_p[j][:]
            Mp[i,j] = np.sum(particles[j].m[:])
            Tp[i,j] = particles[j].T
            xp[i,j*3:j*3+3] = q_local.x_p[j,:]
        Hp[i,:] = q_local.H_p
        u[i] = q_local.u
        v[i] = q_local.v
        w[i] = q_local.w
        V[i] = q_local.V
        h[i] = q_local.h
        x[i] = q_local.x
        y[i] = q_local.y
        z[i] = q_local.z
        s[i] = q_local.s
        rho[i] = q_local.rho
        b[i] = q_local.b
        cos_p[i] = q_local.cos_p
        sin_p[i] = q_local.sin_p
        cos_t[i] = q_local.cos_t
        sin_t[i] = q_local.sin_t
        rho_a[i] = q_local.rho_a
        Sa[i] = q_local.Sa
        Ta[i] = q_local.Ta
        ua[i] = q_local.ua
        E[i] = lmp.entrainment(q0_local, q_local, p)

    # Compute the unit vector along the plume axis
    Sz = sin_p
    Sx = cos_p * cos_t
    Sy = cos_p * sin_t

    # Plot cross-sections through the plume
    plt.figure(fig)
    plt.clf()
    plt.show()
    fig += 1

    ax1 = plt.subplot(221)
    ax1.plot(x, z, 'b-')
    [x1, z1, x2, z2] = width_projection(Sx, Sz, b)
    ax1.plot(x + x1, z + z1, 'b--')
    ax1.plot(x + x2, z + z2, 'b--')
    for i in range(len(particles)):
        ax1.plot(particles[i].x, particles[i].z, 'o')
        ax1.plot(xp[:,i*3], xp[:,i*3+2], '.--')
        if particles[i].farfield:
            if particles[i].integrate is False and particles[i].z > 0.:
                ax1.plot(particles[i].sbm.y[:,0], particles[i].sbm.y[:,2],
                         '.:')
            elif particles[i].z > 0:
                ax1.plot(particles[i].sbm.y[:,0], particles[i].sbm.y[:,2],
                         '.:')
    ax1.invert_yaxis()
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('z (m)')
    ax1.grid(which='major', color='0.5', linestyle='-')

    ax2 = plt.subplot(222)
    ax2.plot(y, z, 'b-')
    [y1, z1, y2, z2] = width_projection(Sy, Sz, b)
    ax2.plot(y + y1, z + z1, 'b--')
    ax2.plot(y + y2, z + z2, 'b--')
    for i in range(len(particles)):
        ax2.plot(particles[i].y, particles[i].z, 'o')
        ax2.plot(xp[:,i*3+1], xp[:,i*3+2], '.--')
        if particles[i].farfield:
            if particles[i].integrate is False and particles[i].z > 0.:
                ax2.plot(particles[i].sbm.y[:,1], particles[i].sbm.y[:,2],
                         '.:')
            elif particles[i].z > 0.:
                ax2.plot(particles[i].sbm.y[:,1], particles[i].sbm.y[:,2],
                        '.:')
    ax2.invert_yaxis()
    ax2.set_xlabel('y (m)')
    ax2.set_ylabel('z (m)')
    ax2.grid(which='major', color='0.5', linestyle='-')
    
    ax3 = plt.subplot(223)
    ax3.plot(x, y, 'b-')
    [x1, y1, x2, y2] = width_projection(Sx, Sy, b)
    ax3.plot(x + x1, y + y1, 'b--')
    ax3.plot(x + x2, y + y2, 'b--')
    for i in range(len(particles)):
        ax3.plot(particles[i].x, particles[i].y, 'o')
        ax3.plot(xp[:,i*3], xp[:,i*3+1], '.--')
        if particles[i].farfield:
            if particles[i].integrate is False and particles[i].z > 0.:
                ax3.plot(particles[i].sbm.y[:,0], particles[i].sbm.y[:,1],
                         '.:')
            elif particles[i].z > 0:
                ax3.plot(particles[i].sbm.y[:,0], particles[i].sbm.y[:,1],
                         '.:')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.grid(which='major', color='0.5', linestyle='-')
    
    ax4 = plt.subplot(224)
    ax4.plot(s, np.zeros(s.shape), 'b-')
    ax4.plot(s, b, 'b--')
    ax4.plot(s, -b, 'b--')
    ax4.set_xlabel('s (m)')
    ax4.set_ylabel('r (m)')
    ax4.grid(which='major', color='0.5', linestyle='-')
    
    plt.tight_layout()
    plt.draw()

    # Plot the Lagrangian element height and entrainment rate
    plt.figure(fig)
    plt.clf()
    plt.show()
    fig += 1

    ax1 = plt.subplot(121)
    ax1.plot(s, h, 'b-')
    ax1.set_xlabel('s (m)')
    ax1.set_ylabel('h (m)')
    ax1.grid(which='major', color='0.5', linestyle='-')

    ax2 = plt.subplot(122)
    ax2.plot(s, E, 'b-')
    ax2.set_xlabel('s (m)')
    ax2.set_ylabel('E (kg/s)')
    ax2.grid(which='major', color='0.5', linestyle='-')

    plt.draw()

    # Plot the velocities along the plume centerline
    plt.figure(fig)
    plt.clf()
    plt.show()
    fig += 1

    ax1 = plt.subplot(221)
    ax1.plot(s, u, 'b-')
    ax1.plot(s, ua, 'g--')
    ax1.set_xlabel('s (m)')
    ax1.set_ylabel('u (m/s)')
    ax1.grid(which='major', color='0.5', linestyle='-')

    ax2 = plt.subplot(222)
    ax2.plot(s, v, 'b-')
    ax2.set_xlabel('s (m)')
    ax2.set_ylabel('v (m/s)')
    ax2.grid(which='major', color='0.5', linestyle='-')

    ax3 = plt.subplot(223)
    ax3.plot(s, w, 'b-')
    ax3.set_xlabel('s (m)')
    ax3.set_ylabel('w (m/s)')
    ax3.grid(which='major', color='0.5', linestyle='-')

    ax4 = plt.subplot(224)
    ax4.plot(s, V, 'b-')
    ax4.set_xlabel('s (m)')
    ax4.set_ylabel('V (m/s)')
    ax4.grid(which='major', color='0.5', linestyle='-')

    plt.draw()

    # Plot the salinity, temperature, and density in the plume
    plt.figure(fig)
    plt.clf()
    plt.ticklabel_format(useOffset=False, axis='y')
    plt.show()
    fig += 1

    ax1 = plt.subplot(221)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.plot(s, S, 'b-')
    ax1.plot(s, Sa, 'g--')
    if np.max(S) - np.min(S) < 1.e-6:
        ax1.set_ylim([S[0] - 1, S[0] + 1])
    ax1.set_xlabel('s (m)')
    ax1.set_ylabel('Salinity (psu)')
    ax1.grid(which='major', color='0.5', linestyle='-')

    ax2 = plt.subplot(222)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.plot(s, T - 273.15, 'b-')
    ax2.plot(s, Ta - 273.15, 'g--')
    if np.max(T) - np.min(T) < 1.e-6:
        ax2.set_ylim([T[0] - 273.15 - 1., T[0] - 273.15 + 1.])
    ax2.set_xlabel('s (m)')
    ax2.set_ylabel('Temperature (deg C)')
    ax2.grid(which='major', color='0.5', linestyle='-')

    ax3 = plt.subplot(223)
    ax3.yaxis.set_major_formatter(formatter)
    ax3.plot(s, rho, 'b-')
    ax3.plot(s, rho_a, 'g--')
    if np.max(rho) - np.min(rho) < 1.e-6:
        ax3.set_ylim([rho[0] - 1, rho[0] + 1])
    ax3.set_xlabel('s (m)')
    ax3.set_ylabel('Density (kg/m^3)')
    ax3.grid(which='major', color='0.5', linestyle='-')

    plt.draw()

    # Plot the particle mass and temperature
    if n_part > 0:
        plt.figure(fig)
        plt.clf()
        plt.ticklabel_format(useOffset=False, axis='y')
        plt.show()
        fig += 1

        ax1 = plt.subplot(121)
        ax1.yaxis.set_major_formatter(formatter)
        ax1.plot(s, Mp / 1.e-6, 'b-')
        ax1.set_xlabel('s (m)')
        ax1.set_ylabel('m (mg)')
        ax1.grid(which='major', color='0.5', linestyle='-')

        ax2 = plt.subplot(122)
        ax2.yaxis.set_major_formatter(formatter)
        ax2.plot(s, Tp - 273.15, 'b-')
        ax2.set_xlabel('s (m)')
        ax2.set_ylabel('Temperature (deg C)')
        ax2.grid(which='major', color='0.5', linestyle='-')

        plt.draw()


def width_projection(Sx, Sy, b):
    """
    Find the location of the plume width in x, y, z space

    Converts the width b and plume orientation phi and theta into an
    (x, y, z) location of the plume edge.  This function provides a two-
    dimensional result given the unit vector along the plume centerline
    (Sx, Sy) along two dimensions in the (x, y, z) space

    Parameters
    ----------
    Sx : float
        Unit vector projection of the plume trajectory on one of the
        coordinate axes in (x, y, z) space.
    Sy : float
        Unit vector projection of the plume trajectory on another of the
        coordinate axes in (x, y, z) space.
    b : float
        Local plume width

    Returns
    -------
    x1 : float
        Plume edge for Sx projection to left of plume translation direction
    y1 : float
        Plume edge for Sy projection to left of plume translation direction
    x2 : float
        Plume edge for Sx projection to right of plume translation direction
    y1 : float
        Plume edge for Sy projection to right of plume translation direction

    Notes
    -----
    The values of S in the (x, y, z) sytem would be::

        Sz = sin ( phi )
        Sx = cos ( phi ) * cos ( theta )
        Sy = cos ( phi ) * sin ( theta )

    Any two of these coordinates of the unit vector can be provided to this
    function as input.

    """
    # Get the angle to the s-direction in the x-y plane
    alpha = np.arctan2(Sy, Sx)

    # Get the coordinates of the plume edge to the right of the s-vector
    # moving with the plume
    x1 = b * np.cos(alpha - np.pi/2.)
    y1 = b * np.sin(alpha - np.pi/2.)

    # Get the coordinates of the plume edge to the left of the s-vector
    # moving with the plume
    x2 = b * np.cos(alpha + np.pi/2.)
    y2 = b * np.sin(alpha + np.pi/2.)

    return (x1, y1, x2, y2)


def chem_idx_list(chems, composition):
    """
    Find the indices corresponding to chem in the composition list
    
    Parameters
    ----------
    chems : list
        A list of chemical compounds to track.  This may be a list of 
        chemical names or a list of index-numbers corresponding to chemicals
        in the mixture composition.
    composition : list
        A list of chemical compounds in the mixture.
    
    Returns
    -------
    c_idx : list
        A list of array indices corresponding to the positions of the variables
        in the chems list within the composition list.
    
    """
    # Make sure the chems variable contains a list of names or indices
    if isinstance(chems, type(None)):
        # Report all compounds in the mixture
        chems = composition[:]
    elif not isinstance(chems, type(list)):
        chems = list(chems)
    else:
        # Report all compounds, but warn user
        print('\nWARNING:  Input chems list not recognized.')
        print('      --> Proceeding with full composition list...\n')
        chems = composition[:]
    
    # Find the desired list of array indices to track chemicals
    if isinstance(chems[0], str):
        # Report the compounds provided by name
        c_idx = []
        for comp in composition:
            if comp in chems:
                c_idx.append(composition.index(comp))
    else:
        # Report the compounds provided by index
        c_idx = chems[:]
    
    return c_idx
