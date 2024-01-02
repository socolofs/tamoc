"""
subsurface_fracture_model
-------------------------

Simulate the evolution of a petroleum fluid transiting a single pathway
through a subsurface fracture.

WARNING!!!
----------

THIS MODULE IS UNDER CONSTRUCTION AND DOES NOT YET REPRESENT A COMPLETE
SIMULATION OF THE OCEAN SUBSURFACE. PLEASE DO NOT USE THIS MODULE FOR ANY
CRITICAL CALCULATIONS, DESIGN, OR DECISION-MAKING PURPOSES. Scott A.
Socolofsky, 10/28/2022.

Notes
-----

- What does the water and gas saturation in the UT model data mean, and what do
  we do when the gas saturation goes to zero, which I think would mean that the
  aperture is closed off to gas. See _gen_fracture_path.

- What about the fraction filled with water?

- How do we adapt the single fracture model to a pipe fracture?

"""
# S. Socolofsky, Texas A&M University, October 2022, <socolofs@tamu.edu>

from tamoc import seawater, ambient, dbm, dbm_utilities
from tamoc import chemical_properties
from tamoc import lfm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Model(object):
    """
    Master class object for subsurface fracture simulations
    
    Parameters
    ----------
    profile : `ambient.Profile`
        An `ambient.Profile` object that contains the ambient property data
        from a UT fracture model simulation. The minimum data fields needed for
        a simulation are temperature, salinity, pressure
    fracture : `SlotFracture`
        A `SlotFracture` object that contains all of the fracture geometric
        information
    p : `ModelParams`, default=None
        Object containing the fixed model parameters for the simulation.  If
        `None`, then the default parameters will be used.
    
    Attributes
    ----------
    profile : `ambient.Profile`
        An `ambient.Profile` object that contains the ambient property data
        from a UT fracture model simulation. The minimum data fields needed for
        a simulation are temperature, salinity, pressure
    fracture : `SlotFracture`
        A `SlotFracture` object that contains all of the fracture geometric
        information
    x0 : ndarray
        Position of the origin of the fracture network (x, y, z) in meters
    p : `ModelParams`
        Object containing the fixed model parameters for the simulation
    sim_stored : bool
        Flag indicating whether or not the simulation has been run so that data
        would be available for post-processing
    
    """
    def __init__(self, profile, fracture, p=None):
        super(Model, self).__init__()
        
        # Store the input variables
        self.profile = profile
        self.fracture = fracture
        
        # Extract some useful attributes from the fracture object
        self.x0 = fracture.x0
        
        # Get the model parameters object
        if isinstance(p, type(None)):
            self.p = ModelParams()
        else:
            self.p = p
        
        # Set the simulation flag to false
        self.sim_stored = False
        
    def simulate(self, t0, s0, mass_frac, T0, fluid, dt_max, s_max):
        """
        Simulate the petroleum fluid migration along the fracture network
        
        Parameters
        ----------
        t0 : float
            Time at the beginning of the simulation (s)
        s0 : float
            Initial position along the fracture pathway (m)
        mass_frac : ndarray
            Array of mass fractions (--) for each component in the petroleum
            fluid model
        T0 : float
            Initial temperature of the released oil (K).  Set to `None` if 
            the ambient temperature at the initial point should be used.
        fluid : `dbm.FluidMixture`
            A discrete bubble model (`dbm`) fluid mixture object that provides 
            the thermodynamic properties of thhe petroleum fluid
        dt_max : float, default=86400
            The maximum time step to take in the model output (s).  This is not
            the numerical time step used by the ODE solver, but rather, the 
            maximum time step desiredi in the output data.
        s_max : float, default=None
            Maximum distance along the fracture to compute (m). If `None`, then
            the simulation will proceed to the seafloor or the end of the given
            fracture network
        
        """
        # Store the input data for this simulation
        self.t0 = t0
        self.s0 = s0
        self.mass_frac = mass_frac
        self.T0 = T0
        self.fluid = fluid
        self.dt_max = dt_max
        self.s_max = s_max
        
        # If the user set s_max = None, use the full fracture pathway
        if isinstance(self.s_max, type(None)):
            self.s_max = self.fracture.s_max
        
        # Create an initial state-space vector from the given input variables
        self.y0, self.m0_dot = lfm.main_ic(self.s0, self.T0, self.fracture, 
            self.mass_frac, self.fluid, self.profile, self.p)
        
        # Create a local LagParcel object to translate the state space
        self.y_local = LagParcel(self.t0, self.y0, self.m0_dot, self.fracture, 
            self.fluid, self.profile, self.p)
        
        # Compute the evolution along this flow path
        print('\n-- TEXAS A&M OIL-SPILL CALCULATOR (TAMOC) --')
        print('-- Subsurface Fracture Model              --\n')
        print('\nMaximum path length to compute: %g (m)\n' % self.s_max)
        self.t, self.y, self.derived_vars = lfm.calculate(
                self.t0, self.y0, self.y_local, self.p, self.dt_max, 
                self.s_max
            )
        
        # Set the simulation flag to true
        self.fracture.sim_stored = True
        self.sim_stored = True
    
    def plot_state_space(self, fig=2):
        """
        docstring for plot_state_space
        
        """
        plot_state_space(self.t, self.y, self.y_local, self.fracture, 
            self.p, fig)
    
    def plot_component_map(self, comps=None, norm_comp=None, fig=3):
        """
        Plot the model solution for each component
        
        Parameters
        ----------
        comps : list, default=None
            A list of string names of the modeled chemical components and
            pseudo-components that should be included in each plot.  If `None`,
            then all modeled components will be plotted.
        norm_comp : str
            A component that should be used to normalize the mass losses.  If
            `None`, then a normalized plot will not be produced.
        fig : int
            The figure number to start plotting figures.
        
        """
        # If no composition specified, plot all components
        if isinstance(comps, type(None)):
            comps = self.y_local.composition
        
        # Create the plot
        plot_component_map(self.t, self.y, self.derived_vars, self.y_local, 
            self.fracture, self.p, comps, norm_comp, fig)

class ModelParams(object):
    """
    Fixed parameters used in the subsurface fracture model
    
    Fixed model parameters that the user should not adjust and that are used
    by the subsurface fracture model.  These include parameters such as 
    entrainment coefficients and other model constants that have been fitted
    to data and are not considered further calibration coefficients.
    
    Attributes
    ----------
    t_diss : float, default=None
        Time-scale for the diffusion-limited mass transfer coefficients
        in days. If `None`, then the default value will be used.
    
    """
    def __init__(self, t_diss=None):
        super(ModelParams, self).__init__()
        
        # The maximum vertical distance (m) a fluid parcel may rise before
        # recomputing the flash equilibrium of the fluid
        self.dz_equil = 5. 
        
        # Heat capacity of oil mixture
        self.cp = seawater.cp() * 0.5
        
        # Dissolution fraction (kg/kg) below which any component of the mixture 
        # will be considered to be fully dissolved
        self.fdis = 1.e-6
        
        # Time-scale for the diffusion-limited mass transfer coefficients
        if isinstance(t_diss, type(None)):
            self.t_diss = 60. * 60. * 24. * 14.
        else:
            self.t_diss = 60. * 60. * 24. * t_diss
        
        # Mass-transfer reduction factor
        self.K = 1.
        
        # Heat-transfer reduction factor while heat transfer is active
        self.K_T = 1.
        
        # Set a boolean flag stating whether or not to include biodegradation
        self.no_bio = True
    
                
class LagParcel(object):
    """
    Lagrangian element for a parcel within the fracture path
    
    A Lagrangian element object for unpacking the simulation state-space and
    relating state variables to properties of the Lagrangian parcel at a 
    single point in the fracture model simulation.
    
    Parameters
    ----------
    t0 : float
        Initial time (s) that the simulation starts
    y0 : float
        Initial values of the state-space vector
    m0_dot : float
        Net mass flux (kg/s) of the petroleum fluid at the initial condition
    fracture : `SlotFracture`
        A `SlotFracture` object containing the fracture pathway and properties
    fluid : `dbm.FluidMixture`
        A `dbm.FluidMixture` object that contains the equations of state for 
        the petroleum fluid
    profile : `ambient.Profile`
        An `ambient.Profile` object that contains ambient property data for
        the boundary conditions controlling the simulation.  This database must
        at least contain temperature (K), pressure (Pa), and salinity (psu).
        It may also include data for the concentrations of dissolved components
        in the surrounding rock pore water.  Properties such as
        fracture aperture are taken directly from the `fracture` object.
    p : `ModelParams`
        Object containing the fixed model parameters of the simulation
    
    Notes
    -----
    The state space vector contain the position s along the fracture pathway, 
    the masses of each pseudo-component of the oil model in the present
    Lagrangian parcle, and the heat of the Lagrangian parcel.
    
    """
    def __init__(self, t0, y0, m0_dot, fracture, fluid, profile, p):
        super(LagParcel, self).__init__()
        
        # Store the initial values of the input variables
        self.t0 = t0
        self.y0 = y0
        self.m0_dot = m0_dot
        self.fracture = fracture
        self.fluid = fluid
        self.profile = profile
        self.p = p
        
        # Extract some additional variables for quick and easy-to-read
        # access
        self.composition = self.fluid.composition
        self.m0 = self.y0[1:-1]
        self.diss_indices = self.m0 > 0.
        self.x0 = self.fracture.get_x(self.y0[0])
        
        # Store the initial mass and heat transfer reduction factors
        self.K = self.p.K
        self.K_T = self.p.K_T
        
        # Store variables to track the equilibrium calculations
        self.flash = False
        self.xe, self.ye, self.ze = self.x0[:]
        
        # Update the parcel with the present state space
        self.update(t0, y0)
    
    def update(self, t, y):
        """
        Update the LagParcel object with the given state-space solution
        
        Parameters
        ----------
        t : float
            Present simulation time (s)
        y : ndarray
            Present values of the state-space vector
        
        """
        # Save the current state-space vector
        self.t = t
        self.y = y
        
        # Extract the state-space variables from the state-space vector
        self.s = y[0]
        self.m = y[1:-1]
        self.h = y[-1]
        
        # Get the present position in Cartesian space
        self.xp = self.fracture.get_x(self.s) 

        # Get the local ambient conditions
        self.Ta, self.Sa, self.Pa = self.profile.get_values(self.xp[2], 
            ['temperature', 'salinity', 'pressure'])
        self.Ca = self.profile.get_values(self.xp[2], self.composition)
        self.rho_a = seawater.density(self.Ta, self.Sa, self.Pa)

        # Get the temperature of the Lagrangian element
        if np.sum(self.m) == 0:
            # The masses are zero...only contains ambient fluid
            self.T = self.Ta
        elif self.K_T == 0:
            # Heat transfer it turned off...set temperature to ambient fluid
            self.T = self.Ta
        else:
            # Compute from the parcel heat 
            self.T = self.h / (np.sum(self.m) * self.p.cp)
        
        # If we are getting out of equilibrium, perform a new flash
        # calculation
        self.update_fluid_state(self.xp, self.m, self.T)
        
        # Update the system densities
        self.rho_gas, self.rho_liq, self.alpha, self.rho = \
            self.density(self.T, self.Pa)
        
        # Compute the solubilities at the current conditions
        self.Cs = self.solubility(self.T, self.Pa, self.Sa)
        
        # Get the geometric properties of the fracture at the current location
        self.V = np.sum(self.m) / self.rho
        self.xs, self.ws, self.Ax, self.As = self.fracture.return_all(self.s,
            self.V)
        
        # Compute the updated mass flow rate
        self.m_dot = self.m0_dot * np.sum(self.m) / np.sum(self.m0)
        
        # Get the advection speed along the fracture
        if self.rho > 0.:
            self.us = self.m_dot / self.Ax / self.rho
        else:
            self.us = 0.
        
        # Compute the correlations for mass transfer coefficient
        self.D = self.fluid.diffusivity(self.T, self.Sa, self.Pa)
        self.mu = self.viscosity(self.T, self.Pa)
        self.kh = seawater.k(self.Ta, self.Sa, self.Pa) / \
            (seawater.density(self.Ta, self.Sa, self.Pa) * seawater.cp())
        Re = self.rho * self.ws * self.us / self.mu
        Sc = self.mu / (self.rho * self.D)
        Pr = self.mu / (self.rho * self.kh) 
        Sh = 3.0 + 0.7 * Re**(1./2.) * Sc**(1./3.)
        # Set maximum Sh from Panga et al. (2005), p. 3236
        for i in range(len(Sh)):
            if Sh[i] > 4.36:
                Sh[i] = 4.36
        
        # Get the equivalent correlation for heat-transfer coefficient
        Nu = 3.0 + 0.7 * Re**(1./2.) * Pr**(1./3.)
        
        # Select the appropriate mass and heat transfer coefficients
        if self.p.t_diss == 0.:
            # Use the empirical heat and mass transfer coefficients
            self.beta = Sh * self.D /  self.ws
            self.beta_T = Nu * self.kh / self.ws
        else:
            # Use the diffusion-limited mass transfer rates
            self.beta = np.sqrt(self.D / (np.pi * self.p.t_diss))
            self.beta_T = np.sqrt(self.kh / (np.pi * self.p.t_diss))
        
        # Turn off dissolution for dissolved components
        frac_diss = np.ones(np.size(self.m))
        frac_diss[self.diss_indices] = \
            self.m[self.diss_indices] / self.m0[self.diss_indices]
        self.beta[frac_diss < self.p.fdis] = 0.
        self.beta[np.where(np.isnan(self.beta))] = 0.
        
        # Turn off heat transfer when at equilibrium
        if self.K_T > 0. and np.abs(self.Ta - self.T) < 0.5:
            # Parcel temperature is close enough to neglect heat transfer
            self.K_T = 0.
        
        # Apply the mass transfer and heat transfer reduction factors
        self.beta = self.K * self.beta
        self.beta_T = self.K_T * self.beta_T
    
        # Get the biodegradation rates
        self.k_bio = self.fluid.k_bio
        if self.p.no_bio:
            self.k_bio = np.zeros(self.k_bio.shape)
        
    
    def update_fluid_state(self, x, m, T):
        """
        Perform an equilibrium calculation to update the fluid state
        
        Perform a new flash equilibrium calculation to compute the gas and 
        liquid composition of the petroleum mixture.  This method only 
        updates the solution if the fluid is far enough from the previous
        flash calculation to warrant a new evaluation.  This is judged based
        on the model parameter p.dz_equil, the maximum vertical distance a
        fluid may rise before computing a new flash equilibrium
        
        Parameters
        ----------
        x : ndarray
            Vector position (x, y, z in meters) of the present Lagrangian
            element
        m : ndarray
            Array containing the present total masses (kg) of each component of
            the petreoleum fluid in the mixture
        T : float
            Temperature (K) of the petroleum mixture
        
        """
        # Check if we need to compute a new equilibrium
        if not self.flash or np.abs(x[2] - self.ze) > self.p.dz_equil:
            
            # Perform the flash equilibrium
            self.xe, self.ye, self.ze = x[:]
            self.Te = T
            self.Se, self.Pe = self.profile.get_values(self.ze, ['salinity',
                'pressure'])
            if not self.flash:
                self.me, xe, self.Ke = self.fluid.equilibrium(m,
                    self.Te, self.Pe)
            else:
                self.me, xe, self.Ke = self.fluid.equilibrium(m,
                    self.Te, self.Pe, self.Ke)
            self.flash = True
        
        
    def density(self, Tp, Pp):
        """
        Compute the petroleum fluid densities at the given thermodynamic
        state
        
        Parameters
        ----------
        Tp : float
            Temperature (K) of the petroleum fluid
        Pp : float
            Pressure (Pa) of the petroleum fluid
        
        Returns
        -------
        rho_gas : float
            Density (kg/m^3) of the gas-phase fluid
        rho_liq : float
            Density (kg/m^3) of the liquid-phase fluid
        alpha : float
            Fraction (--) of the fluid in the gas phase
        rho : float
            System-average density (kg/m^3) of the mixture
        
        Notes
        -----
        This method assumes that the most recent equilibrium calculation is
        still valid; hence, it does not compute a new equilibrium
                
        """        
        # Density of each phase and the gas fraction
        if np.sum(self.me[0,:]) == 0.:
            # Single-phase liquid
            rho_gas = 0.
            rho_liq = self.fluid.density(self.me[1,:], Tp, Pp)[1,0]
            alpha = 0.
            rho = rho_liq
        elif np.sum(self.me[1,:]) == 0:
            # Single-phase gas
            rho_liq = 0.
            rho_gas = self.fluid.density(self.me[0,:], Tp, Pp)[0,0]
            alpha = 1.
            rho = rho_gas
        else:
            # Mixed-phase fluid
            rho_gas = self.fluid.density(self.me[0,:], Tp, Pp)[0,0]
            rho_liq = self.fluid.density(self.me[1,:], Tp, Pp)[1,0]   
            Vg = np.sum(me[0,:]) / rho_gas
            Vl = np.sum(me[1,:]) / rho_liq
            alpha = Vg / (Vg + Vl)
            rho = (np.sum(self.me[0,:]) + np.sum(self.me[1,:])) / (Vg + Vl)
        
        return (rho_gas, rho_liq, alpha, rho)
    
    def solubility(self, Tp, Pa, Sa):
        """
        Solubility of the petroleum fluid mixture at the given thermodynamic
        state and salinity
        
        Parameters
        ----------
        Tp : float
            Temperature (K) of the petroleum fluid
        Pa : float
            Pressure (Pa) of the surrounding fluid
        Sa : float
            Salinity (psu) of the surrounding fluid
        """
        # Use the masses of the correct fluids to compute solubilities
        if np.sum(self.me[0,:]) == 0.:
            # Single-phase liquid
            Cs = self.fluid.solubility(self.m, Tp, Pa, Sa)[1,:]
        elif np.sum(self.me[1,:]) == 0:
            # Single-phase gas
            Cs = self.fluid.solubility(self.m, Tp, Pa, Sa)[0,:]
        else:
            # Two-phase mixture in equilibrium...return the gas-values
            Cs = self.fluid.solubility(self.m, Tp, Pa, Sa)[0,:]
        
        return Cs
    
    def viscosity(self, Tp, Pa):
        """
        Return the viscosity of the petroleum fluid mixture
        
        """
        if np.sum(self.me[0,:]) == 0.:
            # Single-phase liquid
            mu = self.fluid.viscosity(self.me[1,:], Tp, Pa)[1,0]
        elif np.sum(self.me[1,:]) == 0.:
            # Single-phase gas
            mu = self.fluid.viscosity(self.me[0,:], Tp, Pa)[0,0]
        else:
            # Compute the viscosity and density of each phase
            mu_p = np.zeros(2)
            rho_p = np.zeros(2)
            for i in range(len(mu_p)):
                mu_p[i] = self.fluid.viscosity(self, self.me[i,:], T, P)[i,0]
                rho_p[i] = self.fluid.density(self, self.me[i,:], T, P)[i,0]
    
            # Return a volume-weighted average
            mu = (mu_p[0] * np.sum(self.me[0,:]) / rho_p[0] + mu_p[1] * 
                np.sum(self.me[1,:]) / rho_p[1]) / (np.sum(self.me[0,:]) / 
                rho_p[0] + np.sum(self.me[1,:]) / rho_p[1])
        
        return mu
    
    
class SlotFracture(object):
    """
    Master class controlling generation and behavior a pipe fracture network
    
    Parameters
    ----------
    profile : `ambient.Profile`
        An `ambient.Profile` object that contains ambient property data for
        the boundary conditions controlling the simulation. This database
        must at least contain entries for temperature (K), pressure (Pa), and
        salinity (psu). It may also include data for fracture aperture (m),
        gas saturation (--), and the concentrations of dissolved compounds in
        the pore waters.
    x0 : np.array
        Planar (x, y) coordinate (m) for the origin of the fracture path
    H : float
        Depth (m) at the seafloor
    Hs : float
        Thickness (m) of the subsurface layer between the petroleum reservoir
        supplying the fracture and the seafloor.    
    Lx : float
        Length of the fracture path simulated in the UT model (m)
    lc : float
        Correlation length scale (m) of the rock strength.  Values in 
        Santillian et al. (2017) are between 0.05 m and 0.4 m.
    Cv : float
        Coefficient of variation of the rock strength data.  Values in 
        Santillian et al. (2017) are between 0.1 and 0.3
    delta_s : float
        Step-size to use when building the network. This scale should
        normally be larger than the in situ bedrock scale. The model
        constructs a network that matches this scale.
    
    """
    def __init__(self, profile, x0, H, Hs, Lx, lc, Cv, delta_s):
    
        super(SlotFracture, self).__init__()
        
        # Record the input parameters
        self.profile = profile
        self.x0 = x0
        self.H = H
        self.Hs = Hs
        self.Lx = Lx
        self.lc = lc
        self.Cv = Cv
        self.delta_s = delta_s
        
        # Specify the model equations that can simulate this fracture
        self.derivs = lfm.slot_derivs
        
        # Generate the associated fracture network
        self._gen_fracture_path()
        
    def _gen_fracture_path(self):
        """
        Generate a random-walk fracture path 
        
        Generates the three-dimensional coordinates of a fracture pathway 
        in the subsurface.  Creates interpolation functions xs ans ws, which
        return the (x, y, z) coordinate of the centerline of the fracture and
        the aperature slot-width given any distance s along a fracture 
        pathway.  Results of this function are stored in new `SlotFracture`
        object attributes.
        
        Attributes
        ----------
        Du : float
            Pseudo-diffusivity of the random walk model, equal to D / u and
            taken from a fit to the data in Santillian et al. (2017)
        xp : ndarray
            Coordinates of the nodes of the centerline of each point in the
            random-walk model of the slot path
        sp : ndarray
            Path length (m) along the trajectory xp
        ap : ndarray
            Aperture size (m) at each point along the trajectory xp
        s_max : float
            Total length of the random-walk path
        xs : `interp1d`
            Interpolation function that returns the x, y, z coordinate of a 
            point given its distance s along the random-walk pathway
        ws : `interp1d`
            Interpolation function that returns the slot-width (aperture, m)
            of a point given its distance s along the random-walk pathway
        sim_stored : bool
            Flag indicating that a simulation has not yet been run
        
        """
        # Echo the progress to the screen
        print('\n-- Generating Fracture Pathway for a Slot Fracture --')
        print('\nGenerating slot path from %g (m) to %g (m)' % 
            ((self.Hs + self.H, self.H)))
        
        # Compute the psuedo-diffusivity D/u
        self.Du = 340 * self.lc * self.Cv**2 + 0.65
        
        # Create a random-walk network of line segments
        self.xp = slot_fracture_network(self.x0, self.H, self.Hs, self.Du,
            self.delta_s)
        
        # Generate a path-length coorinate system and aperture size
        sp = np.zeros(self.xp.shape[0])
        ap = np.zeros(self.xp.shape[0])
        
        # Fill the first point with s = 0 and a = aperature in database
        ap[0] = self.profile.get_values(self.xp[0,2],
            ['aperture_filled_by_oil'])
        
        print('\nFilling in properties of the fracture pathway:')
        psteps = 1000
        for i in range(len(sp) - 1):
            
            # Set the base of this segment at the end of the previous segment
            sp[i+1] = sp[i]
            
            # Add the length of the current segment
            sp[i+1] += np.sqrt(np.sum((self.xp[i+1,:] - self.xp[i,:])**2))
        
            # Look up the aperture size at this depth
            ap[i+1] = self.profile.get_values(self.xp[i+1,2],
                ['aperture_filled_by_oil'])
                        
            # Echo progress to the screen
            if i % psteps == 0.:
                print('    Depth : %g (m), Node:  %d of %d, a: %g (mm)' 
                    % (self.xp[i+1,2], i, len(sp) - 1, 1000 * ap[i+1]))
        
        # Store the results in the object attributes
        self.sp = sp
        self.ap = ap
        self.s_max = self.sp[-1]
        
        # Create an interpolator for the x-coorindate of any s-value
        from scipy.interpolate import interp1d
        fill_value = (self.xp[0,:], self.xp[-1,:])
        self.xs = interp1d(self.sp, self.xp, axis=0, fill_value=fill_value,
            bounds_error=False)

        # Create an interpolator for the area of any segment
        fill_value = (self.ap[0], self.ap[-1])
        self.ws = interp1d(self.sp, self.ap, axis=0, fill_value=fill_value,
            bounds_error=False)
        
        # Set a flag indicating that no simulation has yet been conducted 
        # on this fracture pathway
        self.sim_stored = False
    
    def get_x(self, s):
        """
        Return the x, y, z coordinates (m) of a segment at position s along the
        fracture path
        
        """
        return self.xs(s)

    def get_xsec_area(self, s):
        """
        Return the cross-sectional area (m^2) of a segment at position s along
        the fracture path
        
        """
        return self.ws(s) * self.Lx
    
    def get_width(self, s):
        """
        Return the characteristic width (m) of the segment at position s
        
        """
        return self.ws(s)

    def get_surface_area(self, s, V):
        """
        Return the surface area (m^2) of a fracture element with volume V
        at position s
        
        """
        w = self.ws(s)
        h = V / (w * self.Lx)
        
        return 2. * (w + self.Lx) * h

    def return_all(self, s, V):
        """
        Return all geometric properties of the current location 
        
        Returns all geometric properties (position, surface area, width,
        and cross-sectional area) while minimizing calls to the interpolation
        functions.
        
        Parameters
        ----------
        s : float
            Position (m) along the fracture pathway in path-coordinates
        V : float
            Volume of the present Lagrangian element
        
        Returns
        -------
        xs : ndarray
            Vector of position (x, y, z in meters) of the present path point
            in Cartesian space
        ws : float
            Width (m) of the fracture aperture at the present location
        Ax : float
            Cross-sectional area (m^2) of the fracture at the present location 
        As : float
            Surface area of exposed surface (m^2) between the fracture and the
            exposed rock for the given element volume
        
        """
        # Interpolate the position
        xs = self.xs(s)
        
        # Interpolate the slot width
        ws = self.ws(s)
        
        # Compute the cross-sectional area
        Ax = ws * self.Lx
        
        # Compute the surface area exposed to rock
        h = V / (ws * self.Lx)
        As = 2. * (ws + self.Lx) * h
        
        return xs, ws, Ax, As
        
    def regenerate_network(self):
        """
        Create a new fracture network with the same network properties
        
        Because the random-walk network pathways are generated using 
        random number generators, there are an infinite number of networks
        with the same properties.  This method generates a new network 
        pathway with the present properties for the fracture network
        
        """
        # Generate a new fracture path
        self._gen_fracture_path()
            
    def show_network(self, fig=1, clearFig=True):
        """
        Plot the fracture network
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to create
        clearFig : bool, default=True
            Boolean parameter stating whether or not to clear the figure
            data before plotting the network
        
        """
        show_network(self.xp, fig, clearFig)
        
    
# -----------------------------------------------------------------------------
# ---------- Functions used by subsurface fracture model classes --------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Functions to read in the ambient property data computed by a regional model
# -----------------------------------------------------------------------------

def load_UT_fracture_data(fname):
    """
    Load the ambient fracture conditions and properties from the UT model
    
    Load the simulation results the ambient conditions within a single fracture
    computed by the UT fracture model.
    
    Parameters
    ----------
    fname : str
        String path to the data file that contains the output from the UT 
        fracture model
    
    Returns
    -------
    profile : `ambient.Profile`
        An `ambient.Profile` object that contains the vertical profile data
        (temperature, salinity, pressure, aperture, etc.) loaded from the UT 
        model output
    H : float
        The water depth (m) at the seafloor
    Hs : float
        The thickness (m) for the subsurface layer
    Lx : float
        The length of the UT model domain (m)
    
    """
    # Load the header information
    header = []
    with open(fname) as ut_data:
        for line in ut_data:
            if line[0] == '#':
                header.append(line)
    
    # Load the dataset
    ut_model = np.loadtxt(fname)
    
    # Parse the simulation parameters from the header block
    for line in header:
        if 'Seafloor' in line:
            data = line.strip().split(' ')
            # Get the water depth in m
            H = float(data[data.index('=') + 1])
            H_units = data[data.index('=') + 2]
            H, H_units = ambient.convert_units(H, H_units)
            H = float(H)
        
        if 'fracture length' in line:
            data = line.strip().split(' ')
            # Get the length of fracture in the simulation
            Lx = float(data[data.index('=') + 1])
            Lx_units = data[data.index('=') + 2]
            Lx, Lx_units = ambient.convert_units(Lx, Lx_units)
            Lx = float(Lx)
    
    # Read the profile variable names and units from the header block
    ut_vars = []
    ut_units = []
    for line in header:
        if 'Col' in line:
            v0 = line.index(':') + 2
            v1 = line.index('(') - 1
            var_name = line[v0:v1].strip().split(' ')
            ut_vars.append('_'.join(var_name).lower())
            u0 = line.index('(') + 1
            u1 = line.index(')')
            ut_units.append(line[u0:u1])
    
    # Convert the depth below seafloor to depth
    for var in ut_vars:
        if var == 'depth_below_seafloor':
            Hs = ut_model[-1, ut_vars.index(var)]
            ut_model[:, ut_vars.index(var)] += H
            ut_vars[ut_vars.index(var)] = 'z'
    
    # Create the ambient Profile object using this data
    profile = ambient.Profile(ut_model, ztsp_units=ut_units[0:4],
        chem_names=ut_vars[4:], chem_units=ut_units[4:], err=0., 
        stabilize_profile=False)
    
    # Return the profile data and model settings
    return (profile, H, Hs, Lx)
    

# -----------------------------------------------------------------------------
# Functions to create a wandering fracture pathway
# -----------------------------------------------------------------------------

def slot_fracture_network(x0, H, Hs, Du, delta_s):
    """
    docstring for slot_fracture_network
    
    """
    # Create an empty list to hold the x, y, z coordinate of the centerline
    # of the fracture path
    x = []
    
    # Add the origin
    x.append(np.array([x0[0], x0[1], H + Hs]))
    
    # Import the random number generator
    from scipy.stats import norm
    mu = 0.
    sigma = 1.
    
    # Find points along the fracture network until we reach the seabed
    psteps = 1000
    k = 0
    while x[-1][2] > H:
        
        # Generate the next point along the trajectory...first, the random
        # step
        x_new = np.zeros(3)
        r = norm.rvs(mu, scale=sigma, size=3)
        x_new[:2] = x[-1][:2] + r[:2] * np.sqrt(Du * delta_s)
        
        # ...then, the deterministic, pseudo-advection step
        x_new[2] = x[-1][2] - delta_s
        
        # Append this point to the network
        x.append(x_new)
        k += 1
        
        # Print the progress to screen
        if k % psteps == 0:
            print('    -> Adding step %d at depth %g (m)' % (k, x[-1][2]))
    
    # Convert to a numpy array
    x = np.array(x)
    
    # Ensure the final point is exactly at the mudline
    dl = (H - x[-2,2]) / (x[-1,2] - x[-2,2])
    if dl != 0:
        x[-1,:] = dl * (x[-1,:] - x[-2,:]) + x[-2,:]
    
    # Echo progress to screen
    print('    = Created network with %d nodes' % k)
    
    return (x)


# -----------------------------------------------------------------------------
# Functions to read an oil composition file and fill in missing properties 
# -----------------------------------------------------------------------------

def read_pvtsim_data(stout, sol=None, user_data={}, tamoc_names={}):
    """
    Read in psuedo-component property data exported from PVT Sim
    
    Read in the standard output and optional solubility data for pure 
    substances and psuedo-components exported from PVT sim.  This function
    uses `pandas` to read the Excel files natively.  
    
    This function returns a `tamoc` `dbm` object with the component names
    matching those in the PVT Sim output data.
    
    Parameters
    ----------
    stout : str
        File name for the standard output property data
    sol : str
        File name for solubilities predicted by PVT Sim
    user_data : dict, default={}
        A dictionary of property data for pure compounds.  The default is an 
        empty dictionary, which will force this function to replace all missing
        data with computed values
    tamoc_names : dict
        A dictionary of chemical property names used by the `tamoc` default
        database or used in the `user_data` database.  This dictionary should 
        have key names matching those in `data` with the stored values matching
        the names in `tamoc`.  An empty dictionary indicates that none of the
        present psuedo-components either have name different from those in 
        `tamoc` or they are not present in `tamoc` databases.
        
    Returns
    -------
    composition : list
        A list of string names for each component in the composition matching
        those names used in PVT Sim.
    mass_frac : ndarray
        An array of mass fractions (--) for each component in the mixture
    dbm_obj : `dbm.FluidMixture`
        A `dbm.FluidMixture` object containing the equations of state for this
        fluid in a format used by `tamoc`.
    data : dict
        Dictionary of chemical properties data organized with the compound
        name as the primary key and a nested dictionary of chemical property
        data as the key value.  The chemical property data is stored using 
        names used in TAMOC when a given variable is part of the TAMOC Discrete
        Bubble Model (DBM).
    units : dict
        A dictionary of units corresponding to each property in the data
        dictionary.
    
    """
    # Get the PVT Sim data from the default Excel spreadsheets
    composition, mol_frac, data, units, delta = extract_pvtsim_data(stout)
    
    # If we have solubility data, also read that from PVT Sim
    data, units = extract_pvtsim_solubilities(composition, data, units, sol)   
    
    # Add the missing properties
    data, units = fill_missing_properties(data, units, user_data, tamoc_names)
    
    # Create the `dbm.FluidMixture` object
    dbm_obj = dbm.FluidMixture(composition, delta=delta, user_data=data)
    mass_frac = dbm_obj.mass_frac(mol_frac)
    
    return (composition, mass_frac, dbm_obj, data, units)

    
def extract_pvtsim_data(stout):
    """
    Extract the chemical property data output from PVT Sim
    
    Create a data dictionary and corresponding dictionary of units that contains
    the data read from a PVT Sim Excel output file.
    
    Parameters
    ----------
    stout : str
        A string file name for the Excel spreadsheet in which the standard
        output data from PVT Sim are stored
        
    Returns
    -------
    composition : list
        A list of string names describing the pseudo-components used in the
        PVT Sim output data
    mol_frac : ndarray
        An array of mole fractions for each component in the composition
    data : dict
        Dictionary of chemical properties data organized with the compound
        name as the primary key and a nested dictionary of chemical property
        data as the key value.  The chemical property data is stored using 
        names used in TAMOC when a given variable is part of the TAMOC Discrete
        Bubble Model (DBM).
    units : dict
        A dictionary of units corresponding to each property in the data
        dictionary.
    delta : ndarray
        A filled `numpy` array of binary interaction coefficients
    
    Notes
    -----
    This function uses the `convert_units` function in the `chemical_propertes` 
    module of `tamoc` to convert the units provided from the PVT Sim software
    to the standard SI units used in `tamoc`.

    """
    # Each PVT Sim output has two sheets:  Sheet1 contains the pseudo-component
    # property data, and Sheet2 contains the binary interaction coefficients
    df = pd.read_excel(stout, sheet_name='Sheet1')
    
    # Write the column name of the data to extract from the PVT Sim output
    pvtsim_names = ['Molecular Weight', 'Liquid Density kg/m³',
        'Critical Temperature K', 'Critical Pressure kPa', 'Acentric Factor', 
        'Normal Tb K', 'Weight Av. Molecular Weight', 'Critical Volume m³/mol',
        'Cpen m³/mol', 'CpenT m³/(mol K)', 'Href J/mol']
    
    # Give the units of each property listed above using unit names recognized
    # by TAMOC.  These should be the same as the units in the original PVT Sim
    # output; these will be converted to standard TAMOC units in the last 
    # step below.
    pvtsim_units = ['(g/mol)', '(kg/m3)', '(K)', '(kPa)', '(--)',
        '(K)', '(g/mol)', '(m3/mol)', '(m3/mol)', '(m3/(mol K))', '(J/mol)']
    
    # Give the TAMOC name (if there is one) for each of the properties included
    # in the `pvtsim_names` list
    tamoc_names = ['M', 'rho_p', 'Tc', 'Pc', 'omega', 'Tb', 'M_ave', 'Vc', 
        'C_pen', 'C_pen_T', 'dH']
    
    # Find the number of pseudo-components contained in this dataset
    m = len(df)
    
    # Create an empty dictionary to hold the property data
    data = {}
    
    # Create empty lists to contain the composition and mol fraction
    composition = []
    mol_frac = []
    
    # Loop through pseudo-component (row) of the PVT Sim output
    for i in range(m):
        
        # Create an empty dictionary to hold the data for this component
        comp_data = {}

        # Loop through property (column) of the PVT Sim output that we want
        # to extract
        for j in range(len(pvtsim_names)):
            
            # Store this property data
            comp_data[tamoc_names[j]] = df[pvtsim_names[j]][i]
        
        # Store this dataset with the component name of the pseudo-component
        data[df['Component'][i]] = comp_data
        
        # Store the composition and mole fraction
        composition.append(df['Component'][i])
        mol_frac.append(df['Mol %'][i] / 100.)
    
    # Convert the data to standard TAMOC units
    data, units = chemical_properties.convert_units(data, tamoc_names,
        pvtsim_units)
    
    # Convert the mol_frac to a numpy array
    mol_frac = np.array(mol_frac)
    mol_frac = mol_frac / np.sum(mol_frac)  # Remove round-off error
    
    # Read in the binary interaction coefficients
    delta = extract_pvtsim_delta(stout)
        
    return (composition, mol_frac, data, units, delta)


def extract_pvtsim_delta(stout):
    """
    Get the binary interaction coefficients
    
    Extract a `numpy` array of the binary interaction coefficients from the
    Excel property data file exported from PVT Sim.  
    
    Parameters
    ----------
    stout : str
        The string file name of the output file containing the standard 
        output data from PVT Sim
    
    Returns
    -------
    delta : ndarray
        A filled `numpy` array of binary interaction coefficients
    
    Notes
    -----
    
    The PVT Sim software package saves the binary interaction coefficients with
    the psuedo-component name in the first column of the spreadsheet and in the
    first row. Pandas interprets the first row to be column headers so that the
    `DataFrame` has `m` rows and `m+1` columns of data, where `m` is the number
    of pseudo-components (i.e., the row of column headers is not included as
    data in the data frame, but rather is used to name each column of data in
    the data frame dictionary). 
    
    Also, the data are stored in the spreadsheet with numbers in the lower,
    triangular matrix and empty cells in the upper, triangular part of the
    dataset. This function fills the matrix and then extracts the data (numbers
    only) to return as the binary interaction matrix.
    
    """# Each PVT Sim output has two sheets:  Sheet1 contains the pseudo-component
    # property data, and Sheet2 contains the binary interaction coefficients
    df = pd.read_excel(stout, sheet_name='Sheet2')
    
    # Find the number of components
    m = len(df)
    
    # Fill in blank cells
    for i in range(m):
        for j in range(m):
            if i == j:                
                # Diagonal values should be zero.  
                df.iloc[i,j+1] = 0.
                
            else:
                # Off-diagonal values should be transpose
                df.iloc[i,j+1] = df.iloc[j,i+1]
    
    # Export the binary interaction coefficients to numpy
    delta = df.iloc[:,1:].to_numpy()
    
    return delta

 
def extract_pvtsim_solubilities(composition, data, units, sol_fname):
    """
    Extract the predicted solubilities from PVT Sim
    
    Read the predicted solubilities from the PVT Sim solubility output, convert
    them to units used in `tamoc` (kg/m^3), and store the results in the
    properties database.
    
    Parameters
    ----------
    composition : list
        A list of string names describing the pseudo-components used in the
        PVT Sim output data
    data : dict
        Dictionary of chemical properties data organized with the compound
        name as the primary key and a nested dictionary of chemical property
        data as the key value.  The chemical property data is stored using 
        names used in TAMOC when a given variable is part of the TAMOC Discrete
        Bubble Model (DBM).
    units : dict
        A dictionary of units corresponding to each property in the data
        dictionary.
    sol_fname : str
        A string file name were the PVT Sim solubility output data are stored
    
    Returns
    -------
    data : dict
        The `data` dictionary with 'P_sol' (pressure, Pa, at which the solubility
        data were calculated), 'T_sol' (temperature, K, at which the solubility
        data were calculated), and 'solubility' (the solubility data read from
        the file and converted to kg/m^3) added to the dictionary.
    units : dict
        The `units` dictionary with the units for `P_sol`, `T_sol`, and 
        `solubility` added to the dictionary
    
    """
    # Read the PT Conditions used in the flash calculation
    df_PT = pd.read_excel(sol_fname, sheet_name='Compositions', usecols=[1, 2, 3], 
        nrows=3)
    P = df_PT.iloc[1,1] * 1000.    # Pa
    T = df_PT.iloc[2,1]            # K
    
    # Read the mol % results at this PT condition for the aqueous phase
    df_sol = pd.read_excel(sol_fname, sheet_name='Compositions', header=5,
        usecols=[1, 2, 3, 4, 5], nrows=len(composition) + 2)
    
    # Put the data into a usable dictionary
    sol_data = {}
    for i in range(len(composition) + 2):

        # Create an array to store the total, vapor, liquid, aqueous data
        comp_data = np.zeros(4)
        
        # Extract the data
        for j in range(4):
            comp_data[j] = df_sol.iloc[i,j+1]
        
        # Add the data to the dictionary
        sol_data[df_sol.iloc[i,0]] = comp_data
    
    # Read the bulk properties
    df_prop = pd.read_excel(sol_fname, sheet_name='Properties',
        header=11, usecols=[1, 2, 3, 4, 5], nrows=21)
    Vm = df_prop.iloc[2,4]  # molar volume in m^3/mol
    
    # Calculate the solubility of each compound in kg/m^3
    for comp in composition:
        data[comp]['solubility'] = 1. / Vm * sol_data[comp][-1] / 100. * \
            data[comp]['M']
        data[comp]['P_sol'] = P
        data[comp]['T_sol'] = T
    units['solubility'] = '(kg/m^3)'
    units['P_sol'] = '(Pa)'
    units['T_sol'] = '(K)'
    
    return data, units
    

def fill_missing_properties(data, units, user_data={}, tamoc_names={}):
    """
    Fill missing data used by `tamoc`
    
    Some of the data used by the `tamoc` `dbm` module is not available in the
    PVT Sim output datasets.  Some of that data is no longer relevant (e.g.,
    `B` and `dE` used in the out-dated methods for diffusivity) or has been 
    replaced by data from PVT Sim (e.g., `Cpen` and `CpenT` are used instead
    of `Vb` and `nu_bar`).  However, to maintain compatibility with the current
    versions of the `tamoc`, we still add these data to the properties 
    dictionaries before calling the `dbm` `Fluid` classes.
    
    This method adds the following properties to the properties data::
       . `Vb` - specific volume at the boiling point (m^3/kg)
       . `nu_bar` - specific volume at infinite dilution (m^3/kg)
       . `K_salt` -  the Setschenow salting out constant (m^3/mol)
       . `-dH_solR` - enthalpy of solution divided by the ideal gas constant (K)
       . `kh_0` - Henry's law constant at 298.15 K and atmospheric pressure
         (kg/(m^3 Pa))
       . `B` - parameter of the old diffusivity algorithm -- no longer used
       . `dE` - parameter of the old diffusivity algorithm -- no longer used
    
    Parameters
    ----------
    data : dict
        The present dictionary of property data for each pseudo-component
    units : dict
        A dictionary of corresponding units for each property
    user_data : dict, default={}
        A dictionary of property data for pure compounds.  The default is an 
        empty dictionary, which will force this function to replace all missing
        data with computed values
    tamoc_names : dict
        A dictionary of chemical property names used by the `tamoc` default
        database or used in the `user_data` database.  This dictionary should 
        have key names matching those in `data` with the stored values matching
        the names in `tamoc`.  An empty dictionary indicates that none of the
        present psuedo-components either have name different from those in 
        `tamoc` or they are not present in `tamoc` databases.
    
    Returns
    -------
    data : dict
        An updated dictionary of property data for each pseudo-component with 
        the required data added.
    units : dict
        An updated dictionary of units including the units for the new property
        data
    
    """
    # Get the default property data supplied with `tamoc`
    default_db, default_units, bio_db, bio_units, pj_data, pj_units = \
        chemical_properties.tamoc_data()
    
    # List the properties that need to be added
    missing_vars = ['Vb', 'kh_0', '-dH_solR', 'nu_bar', 'K_salt', 'B',
        'dE']
    missing_units = ['(m^3/mol)', '(kg/(m^3 Pa))', '(K)', '(m^3/mol)', 
        '(m^3/mol)', '(mm^2/sec)', '(J/mol)']
    
    # Fill in missing property data needed by TAMOC
    for component in data:
        
        # Check for the `tamoc` name of this compound
        if component in tamoc_names:
            # Set `db_name` to the `tamoc` name for searching `tamoc` property
            # data
            db_name = tamoc_names[component]
            
        else:
            # Assume the component name itself is an appropriate name for 
            # searching `tamoc` property data
            db_name = component
        
        # Get the missing data from the most reliable source
        if db_name in user_data:
            # Use the data provided by the user
            for var in missing_vars:
                data[component][var] = user_data[db_name][var]
        
        elif db_name in default_db:
            # Use the database provided with `tamoc`
            for var in missing_vars:
                data[component][var] = default_db[db_name][var]

        else:
            # Estimate the properties from various algorithms...some should use
            # algorithms built into the `tamoc` `dbm` module; set a flag to 
            # invoke those methods.
            data[component]['Vb'] = -9999.
            data[component]['nu_bar'] = -9999.
            data[component]['K_salt'] = -9999.
            data[component]['B'] = -9999.
            data[component]['dE'] = -9999.
        
            # Estimate the enthalpy of solution
            data[component]['-dH_solR'] = est_neg_dH_solR(
                data[component]['M'],
                data[component]['Tc'], 
                data[component]['Pc'],
                data[component]['Tb'])
            
            # Estimate the Henry's constant
            data[component]['kh_0'] = est_kh_0(data)
    
    # Add these properties to the units database
    for i in range(len(missing_vars)):
        units[missing_vars[i]] = missing_units[i]

    return (data, units)

def est_neg_dH_solR(M, Tc, Pc, Tb):
    """
    Estimate the enthalpy of dissolution
    
    Use correlation methods to estimate -dh_sol / R.  This function follows an 
    equation in Gros et al. (2018).
    
    Parameters
    ----------
    M : float
        Molecular weight, kg/mol
    Tc : float
        Critical point temperature, K
    Pc : float
        Critical point pressure, Pa
    Tb : float
        Boiling point temperature, K
    
    Returns
    -------
    neg_dH_solR : float
        Negative of the enthalpy of solution divided by the ideal gas constant, K
    
    """
    # Estimate nu_bar using method in the `tamoc` `dbm` module
    if Tb < 273.15 + 10.:
        # Assume this is a gas...use Lyckman formula
        nu_bar = (0.095 + 2.35 * (298.15 * Pc / (2.2973e9 * Tc))) * \
            8.314510 * Tc / Pc
    else:
        # Use empirical equation from Jonas Gros
        nu_bar = (1.148236984 * M*1000. + 6.789136822) / 100.**3
    
    # Use equation from Gros et al. (2018)
    neg_dH_solR = 2.637 * Tc + 22.48e6 * nu_bar
    
    return neg_dH_solR

def est_kh_0(data):
    """
    Estimate tne Henry's coefficient at standard conditions
    
    """
    # Get the temperature and pressure where we know solubilities
    return 0


# -----------------------------------------------------------------------------
# Functions plot data in the module classes
# -----------------------------------------------------------------------------

def show_network(xp, fig, clearFig):
    """
    Plot the coordinates of the fracture network defined by xp
    
    """
    # Create the figure
    plt.figure(fig, figsize=(11,6))
    if clearFig:
        plt.clf()
    
    # Set up some formatting to show the start and end of each path
    marker_fmt = {'markerfacecolor':'w', 'label':'_no_legend_'}
    
    # Create two subplots...first the x-z and y-z plane
    ax = plt.subplot(121)
    ax.plot(xp[:,0], xp[:,2], 'b-', label='Easterly path')
    ax.plot(xp[:,1], xp[:,2], 'g-', label='Northerly path')
    ax.legend()
    # Plot the start and end points
    ax.plot(xp[0,0], xp[0,2], 'ko', **marker_fmt)
    ax.plot(xp[-1,0], xp[-1,2], 'ko', **marker_fmt)
    ax.plot(xp[0,1], xp[0,2], 'ko', **marker_fmt)
    ax.plot(xp[-1,1], xp[-1,2], 'ko', **marker_fmt)
    # Format and label the axes
    ax.invert_yaxis()
    ax.set_xlabel('Distance, (m)')
    ax.set_ylabel('Depth, (m)')
    
    # and also the x-y plane
    ax = plt.subplot(122)
    ax.plot(xp[:,0], xp[:,1], 'm-')
    # Plot the start and end points
    ax.plot(xp[0,0], xp[0,1], 'ko', **marker_fmt)
    ax.plot(xp[-1,0], xp[-1,1], 'ko', **marker_fmt)
    # Format and label the axes
    ax.set_xlabel('Easterly distance, (m)')
    ax.set_ylabel('Northerly distance, (m)')
    
    plt.show()

def plot_state_space(t, y, y_local, fracture, p, fig):
    """
    docstring for plot_state_space
    
    """
    # Extract the state-space variables
    s = y[:,0]
    m = y[:,1:-1]
    h = y[:,-1]
    
    # Convert heat to temperature
    T = np.zeros(h.shape)
    for i in range(len(T)):
        T[i] = h[i] / (p.cp * np.sum(m[i,:]))
    
    # Plot the variables
    plt.figure(fig, figsize=(11,9))
    plt.clf()
    
    # Plot position
    ax = plt.subplot(131)
    ax.plot(t / 3600., s)
    ax.set_xlabel('Time, (hrs)')
    ax.set_ylabel('Distance, (m)')
    
    # Plot the masses
    ax = plt.subplot(132)
    ax.semilogx(m, s)
    ax.set_xlabel('Mass, (kg)')
    ax.legend(y_local.composition)
    
    # Plot the temperature
    ax = plt.subplot(133)
    ax.plot(T - 273.15, s)
    ax.set_xlabel('Temperature, (deg C)')
    
    plt.show()

def plot_component_map(t, y, derived_vars, parcel, fracture, p, comps, 
    norm_comp, fig):
    """
    docstring for plot_component_map
    
    """
    # Import tool for shading lines by data values
    from matplotlib.collections import LineCollection
    
    # Extract the state-space variables
    s = y[:,0]
    h = y[:,-1]
    
    # Get the x,y,z coordinates
    x = np.zeros((len(t), 3))
    for i in range(len(t)):
        x[i,:] = fracture.get_x(s[i])
    
    # Get the indices to the components
    im = [parcel.composition.index(comp) for comp in comps if comp in 
        parcel.composition]
    
    # Get the component masses for the selected components
    m = np.zeros((len(t), len(comps)))
    for i in range(len(t)):
        m[i,:] = y[i,1:-1][im]
    
    # Compute the fraction remaining for each component
    mf = np.zeros((len(t), len(comps)))
    for i in range(len(t)):
        mf[i,:] = 1. - (m[0,:] - m[i,:]) / m[0,:]
    
    # Get the fraction of gas
    alpha = derived_vars[:]
    
    # Figure out the figure size and number of subplots
    if len(comps) + 1 >= 5:
        cols = 5
    else:
        cols = len(comps) + 1
    if cols == 5:
        rows = int((len(comps) + 1) / cols)
        if (len(comps) + 1) % cols > 0:
            rows += 1
    else:
        rows = 1
    figsize = (2.5 * cols, 4 * rows)
    
    # Plot each component one at a time
    figure = plt.figure(fig, figsize=figsize)
    plt.clf()
    
    # Component masses
    add_bar = True
    for i in range(len(comps)):
        ax = plt.subplot(rows, cols, i+1)
        for j in range(2):
            points = np.array([x[:,j], x[:,2]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm= plt.Normalize(0, np.max(m))
            if j == 0:
                lc = LineCollection(segments, cmap='viridis', norm=norm, 
                    label=comps[i])
            else:
                lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(m[:,i])
            line = ax.add_collection(lc)
            ax.set(xlim=(np.min(x[:,j]), np.max(x[:,j])), 
                ylim=(np.min(x[:,2]), np.max(x[:,2])))
            if add_bar:
                if cols / (i+1) == 1:
                    figure.colorbar(line, ax=ax, label='Mass, (kg)')
                    add_bar = False
            ax.set_xlabel('Distance, (m)')
            if i % cols == 0:
                ax.set_ylabel('Depth, (m)')
            ax.invert_yaxis()
            ax.legend()

    plt.tight_layout()
    plt.show()
    
    # Plot each component one at a time
    figure = plt.figure(fig+1, figsize=figsize)
    plt.clf()
    
    # Fraction remaining
    add_bar = True
    for i in range(len(comps)):
        ax = plt.subplot(rows, cols, i+1)
        for j in range(2):
            points = np.array([x[:,j], x[:,2]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm= plt.Normalize(0, 1)
            if j == 0:
                lc = LineCollection(segments, cmap='viridis', norm=norm, 
                    label=comps[i])
            else:
                lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(mf[:,i])
            line = ax.add_collection(lc)
            ax.set(xlim=(np.min(x[:,j]), np.max(x[:,j])), 
                ylim=(np.min(x[:,2]), np.max(x[:,2])))
            if add_bar:
                if cols / (i+1) == 1:
                    figure.colorbar(line, ax=ax, 
                        label='Fraction remaining, (--)')
                    add_bar = False
            ax.set_xlabel('Distance, (m)')
            if i % cols == 0:
                ax.set_ylabel('Depth, (m)')
            ax.invert_yaxis()
            ax.legend()
    
    # And plot the fraction of gas
    ax = plt.subplot(rows, cols, i+2)
    add_bar = True
    for j in range(2):
        points = np.array([x[:,j], x[:,2]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if j == 0:
            norm = plt.Normalize(0, np.max(alpha))
            lc = LineCollection(segments, cmap='viridis', norm=norm, 
                label='Fraction gas, (--)')
        else:
            lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(alpha)
        line = ax.add_collection(lc)
        ax.set(xlim=(np.min(x[:,j]), np.max(x[:,j])), 
            ylim=(np.min(x[:,2]), np.max(x[:,2])))
        if add_bar:
            figure.colorbar(line, ax=ax, label='(--)')
            add_bar = False
        ax.set_xlabel('Distance, (m)')
        if (i+1) % cols == 0:
            ax.set_ylabel('Depth, (m)')
        ax.invert_yaxis()
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Normalized fraction remaining
    if not isinstance(norm_comp, type(None)):
        
        # Plot the normalized losses one at a time
        figure = plt.figure(fig+2, figsize=figsize)
        plt.clf()
        
        # Get the index to the component to use for normalization
        ic = comps.index(norm_comp)
        
        # Compute the fraction lost of each component
        fl = np.zeros((len(t), len(comps)))
        for i in range(len(t)):
            fl[i,:] = 1. - mf[i,:]
        
        # Normalize by the selected component and plot
        for i in range(len(comps)):
            fl[:,i] = fl[:,i] / fl[:,ic]
            
            ax = plt.subplot(rows, cols, i+1)
            ax.plot(fl[:,i] * 100., x[:,2], label=comps[i])
            ax.set_xlabel('Normalized loss, (%)')
            ax.set_ylabel('Depth, (m)')
            ax.invert_yaxis()
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
