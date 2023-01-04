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

- What does the water and gas saturation in the UT model data mean, and what do we do when the gas saturation goes to zero, which I think would mean that the aperture is closed off to gas.  See _gen_fracture_path.

- What about the fraction filled with water?

- How do we adapt the single fracture model to a pipe fracture?

"""
# S. Socolofsky, Texas A&M University, October 2022, <socolofs@tamu.edu>

from tamoc import seawater, dbm
from tamoc import lfm

import numpy as np
import matplotlib.pyplot as plt


class Model(object):
    """

    Master class object for subsurface fracture simulations
    
    Master class object for the subsurface fracture module. This model class
    generates a fracture network and can simulate transport, transformation,
    and chemical reactions of petroleum fluids flowing through those
    fractures.
    
    Parameters
    ----------
    profile : `ambient.Profile`
        An `ambient.Profile` object that contains ambient property data for
        the boundary conditions controlling the simulation. This database
        must at least contain entries for temperature (K), pressure (Pa), and
        salinity (psu). It may also include data for fracture aperture (m),
        gas saturation (--), and the concentrations of dissolved compounds in
        the pore waters.
    fracture : `PipeFracture`
        A `PipeFracture` object in which we want to run simulations.
    
    Attributes
    ----------
    profile : `ambient.Profile`
        An `ambient.Profile` object that contains ambient property data for
        the boundary conditions controlling the simulation. This database
        must at least contain entries for temperature (K), pressure (Pa), and
        salinity (psu). It may also include data for fracture aperture (m),
        gas saturation (--), and the concentrations of dissolved compounds in
        the pore waters.
    H : float
        Depth (m) at the seafloor
    Hs : float
        Thickness (m) of the subsurface layer between the petroleum reservoir
        supplying the fracture and the seafloor.    
    x0 : float
        Position of the origin of the fracture network.
    p : `ModelParams`
        Object containing the fixed model parameters for the model
    sim_stored : bool
        Boolean flag indicating whether or not a simulation has been 
        completed for the current model settings.
    
    """
    def __init__(self, profile, fracture):
        super(Model, self).__init__()
        
        # Store the input variables
        self.profile = profile
        self.fracture = fracture
        
        # Extract some useful attributes from the fracture object
        self.x0 = np.zeros(3)
        self.x0[0:2] = fracture.x0
        self.x0[2] = fracture.z0
        
        # Get the model parameters object
        self.p = ModelParams()
        
        # Set the simulation flag to false
        self.sim_stored = False
    
    def simulate(self, m_dot, mass_frac, fluid, dt_max=60., s_max=None):
        """
        Simulate the petroleum fluid migration along the fracture network
        
        Parameters
        ----------
        m_dot : float
            Mass flux (kg/s) of petroleum fluid at the origin of the
            fracture pathway
        mass_frac : np.array
            Array of mass fractions (--) for each component in the petroleum
            fluid model
        fluid : `dbm.FluidMixture`
            A discrete bubble model (`dbm`) fluid mixture object that
            provides the thermodynamic properties of the petroleum fluid
        dt_max : float, default=60    
            The maximum time-step to take in the model output (s). This is
            not the numerical time-step used by the ODE solver, but rather
            the maximum time step desired in the output data.
        s_max : float, default=None
            Maximum distance along the fracture to compute (m)
            
        Returns
        -------
        t : np.array
            Array of times corresponding to the travel time of a fluid 
            along the fracture path (s)
        y : np.array
            Array of state space values for each time in the simulations
            domain. The state space consists of y[0]=distance (m) along the
            fracture path, y[1:-1]=masses (kg) of each pseudo-component of
            the oil mixture at each position, and y[-1]=the heat (J)
            contained in the whole petroleum fluid at each position.
        
        
        """
        # Store the input data for this simulation
        self.m_dot = m_dot
        self.mass_frac = mass_frac
        self.fluid = fluid
        self.dt_max = dt_max
        if isinstance(s_max, type(None)):
            self.s_max = self.fracture.s_max
        else:
            self.s_max = s_max
        
        # Create an initial state space vector from the given input variables
        t0, y0 = lfm.main_ic(self.x0[2], self.fracture,
            self.mass_frac, self.fluid, self.profile, self.p)
        
        # Create a Lagrangian parcel object to handle the state space vector
        # and its relation to the properties of the Lagrangian element
        self.y_local = LagParcel(t0, y0, self.m_dot, self.fracture,
             self.fluid, self.profile, self.p)
        
        # Compute the evolution along this flow path
        print('\n-- TEXAS A&M OIL-SPILL CALCULATOR (TAMOC) --')
        print('-- Subsurface Fracture Model              --\n')
        self.t, self.y, self.derived_vars = lfm.calculate(
            t0, y0, self.y_local, self.p, self.dt_max, self.s_max
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
    
    def plot_component_map(self, comps=None, fig=3):
        """
        docstring for plot_component_map
        
        """
        # If no composition specified, plot all components
        if isinstance(comps, type(None)):
            comps = self.y_local.composition
        
        # Create the plot
        plot_component_map(self.t, self.y, self.derived_vars, self.y_local, 
            self.fracture, self.p, comps, fig)

class ModelParams(object):
    """
    Fixed parameters used in the subsurface fracture model
    
    Fixed model parameters that the user should not adjust and that are used
    by the subsurface fracture model. These include parameters such as
    entrainment coefficients and other model constants that have been fitted
    to data and are not considered calibration coefficients.
    
    Parameters
    ----------
        
    """
    def __init__(self):
        super(ModelParams, self).__init__()
        
        # Heat capacity of the petroleum fluid.  Currently, this is not 
        # predicted by the equations of state in the `dbm` model
        self.cp = seawater.cp() * 0.5
        
        # Set a dissolution fraction below which any pseudo-component of an 
        # oil model will be considered to be fully dissolved
        self.fdis = 1.e-6
        
        # By default, we expect the fraction fluid to be in thermal
        # equilibrium...To save simulation effort, ignore heat transfer
        self.K_T = 0.
        
        # Reduce the mass transfer rate by a constant factor
        self.K_m = 0.0
        
        # Or, use diffusion-limited dissolution with a characteristic system
        # time
        self.t_diss = 60. * 60. * 24. * 365.25 * 10.
        
        # Decide whether to turn on biodegradation
        self.no_bio = True
        
        # Decide when to make an equilibrium calculation
        self.delta_z_equil = 5.    # m
        

class LagParcel(object):
    """
    Lagrangian element for a parcel within the fracture path
    
    A Lagrangian element object for unpacking the simulation state space and
    relating state variables to properties of the Lagrangian parcel at a
    single point in the fracture model simulation.
    
    Parameters
    ----------
    t0 : float
        Initial time in the simulation
    y0 : float
        Initial value of the state space vector
    m_dot0 : float
        Net mass flux (kg/s) of the petroleum fluid at the initial condition
    fracture : `PipeFracture`
        A `PipeFracture` object containing the fracture pathway and 
        properties
    fluid : `dbm.FluidMixture`
        A `dbm.FluidMixture` object that contains the equations of state for
        the petroleum fluid
    profile : `ambient.Profile`
        An `ambient.Profile` object that contains ambient property data for
        the boundary conditions controlling the simulation. This database
        must at least contain entries for temperature (K), pressure (Pa), and
        salinity (psu). It may also include data for fracture aperture (m),
        gas saturation (--), and the concentrations of dissolved compounds in
        the pore waters.
    p : `ModelParams`
        Object containing the fixed model parameters for the model
    
    """
    def __init__(self, t0, y0, m_dot0, fracture, fluid, profile, p):
        super(LagParcel, self).__init__()
        
        # Store the initial values of the input variables
        self.t0 = t0
        self.y0 = y0
        self.m_dot0 = m_dot0
        self.fracture = fracture
        self.fluid = fluid
        self.profile = profile
        self.p = p
        
        # Extract some additional parameters
        self.composition = self.fluid.composition
        self.m0 = self.y0[1:-1]
        self.diss_indices = self.m0 > 0.
        self.x0, self.y0, self.z0 = fracture.get_x_position(self.y0[0])
        
        # Perform an initial equilibrium calculation
        self.xe, self.ye, self.ze = (self.x0, self.y0, self.z0)
        self.Te, self.Se, self.Pe = self.profile.get_values(self.z0,
            ['temperature', 'salinity', 'pressure'])
        self.me, self.xe, self.K = self.fluid.equilibrium(self.m0, 
            self.Te, self.Pe)
        
        # Get the density of all phases
        self.rho_gas, self.rho_liq, self.alpha, self.rho = \
            self.density(self.Te, self.Pe)
        
        # Set the mass transfer and heat transfer reduction factor
        self.K_T = p.K_T
        self.K_m = p.K_m
        self.t_diss = p.t_diss
        
        # Update the parcel with the present state space
        self.update(t0, y0)
    
    def update(self, t, y):
        """
        Update the LagParcel object with a given solution for the state space
        
        """
        # Save the current state-space vector
        self.t = t
        self.y = y
        
        # Extract the state-space variables from the state-space solution
        self.s = y[0]
        self.m = y[1:-1]
        self.h = y[-1]
        
        # Get the current position in space and the fracture properties
        self.xp, self.yp, self.zp = self.fracture.get_x_position(self.s)
        self.Ap = self.fracture.get_xsec_area(self.s)
        
        # Get the local ambient conditions
        self.Ta, self.Sa, self.Pa = self.profile.get_values(self.zp, 
            ['temperature', 'salinity', 'pressure'])
        self.Ca = self.profile.get_values(self.zp, self.composition)
        self.rho_a = seawater.density(self.Ta, self.Sa, self.Pa)
        
        # If we are getting out of equilibrium, perform a new flash 
        # calculation
        if (self.zp - self.ze) > self.p.delta_z_equil:
            # We need to make a new equilibrium calculation
            self.xe, self.ye, self.ze = (self.xp, self.yp, self.zp)
            self.Te, self.Se, self.Pe = (self.Ta, self.Sa, self.Pa)
            self.me, self.xe, self.K = self.fluid.equilibrium(self.m, 
                self.Ta, self.Pa, self.K)
        
        # Get the updated densities of each phase
        self.rho_gas, self.rho_liq, self.alpha, self.rho = \
            self.density(self.Ta, self.Pa)
        
        # Compute the solubilities at the current conditions
        self.Cs = self.solubility(self.Ta, self.Pa, self.Sa)
        
        # Get the temperature of the Lagrangian element
        if np.sum(self.m) == 0:
            self.T = self.Ta
        elif self.K_T == 0:
            self.T = self.Ta
        else:
            self.T = self.h / (np.sum(self.m) * self.p.cp)
        
        # Get the updated mass flow rate
        self.m_dot = self.m_dot0 * np.sum(self.m0 - self.m) / np.sum(self.m0)
        
        # Get the advection speed along the fracture
        if self.rho > 0:
            self.us = self.m_dot / self.Ap / self.rho
        else:
            self.us = 0.
        
        # Get the dimensions of the Lagrangian element
        self.ds = self.fracture.get_width(self.s)
        self.V = np.sum(self.m) / self.rho
        self.As = self.fracture.get_surface_area(self.s, self.V)
        
        # Compute the mass transfer coefficient
        self.D = self.fluid.diffusivity(self.T, self.Sa, self.Pa)
        self.mu = self.viscosity(self.Ta, self.Pa)
        self.kh = seawater.k(self.Ta, self.Sa, self.Pa) / \
            (seawater.density(self.Ta, self.Sa, self.Pa) * seawater.cp())
        Re = self.rho * self.ds * self.us / self.mu
        Sc = self.mu / (self.rho * self.D)
        Pr = self.mu / (self.rho * self.kh) 
        Sh = 3.0 + 0.7 * Re**(1./2.) * Sc**(1./3.)
        # Set maximum Sh from Panga et al. (2005), p. 3236
        for i in range(len(Sh)):
            if Sh[i] > 4.36:
                Sh[i] = 4.36

        if self.K_m > 0:
            # Use the correlations for mass transfer coefficients
            self.beta = self.p.K_m * Sh * self.D /  self.ds
        else:
            # Of, use diffusion-limited mass transfer rates
            self.beta = np.sqrt(self.D / (np.pi * self.p.t_diss))    
        
        # Get the equivalent heat-transfer coefficient
        Nu = 3.0 + 0.7 * Re**(1./2.) * Pr**(1./3.)
        self.beta_T = Nu * self.kh / self.ds

        # Get the biodegradation rates
        self.k_bio = self.fluid.k_bio
        if self.p.no_bio:
            self.k_bio = np.zeros(self.m.shape)

        # Turn off dissolution for dissolved components
        frac_diss = np.ones(np.size(self.m))
        frac_diss[self.diss_indices] = \
            self.m[self.diss_indices] / self.m0[self.diss_indices]
        self.beta[frac_diss < self.p.fdis] = 0.
        self.beta[np.where(np.isnan(self.beta))] = 0.
        
        # Turn off heat transfer when at equilibrium
        if self.beta_T > 0. and np.abs(self.Ta - self.T) < 0.5:
            # Parcel temperature is close enough to neglect heat transfer
            self.K_T = 0.
        if self.K_T == 0:
            # Set the heat transfer coefficient to zero
            self.beta_T = 0.
        if self.beta_T == 0.:
            # Make sure you use the ambient temperature for the element
            self.T = self.Ta
        
    def viscosity(self, T, P):
        """
        Compute the viscosity of the petroleum fluid
        
        """
        # Get the mass fractions of gas and liquid
        m = self.me[0,:] + self.me[1,:]
        f_gas = np.sum(self.me[0,:]) / np.sum(m)
        f_liq = np.sum(self.me[1,:]) / np.sum(m)
        
        # Compute the viscosities of gas and liquid
        if f_gas <= 0.001:
            # This is essentially pure liquid
            mu = self.fluid.viscosity(self.me[1,:], T, P)[1,0]
            
        elif f_liq <= 0.001:
            # This is essentially pure gas
            mu = self.fluid.density(self.me[0,:], T, P)[0,0]
            
        else:
            # This is a gas/liquid mixture...compute a mass-weighted average
            mu_gas = self.fluid.density(self.me[0,:], T, P)[0,0]
            mu_liq = self.fluid.density(self.me[1,:], T, P)[1,0]
            rho_gas = self.fluid.density(self.me[0,:], T, P)[0]
            rho_liq = self.fluid.density(self.me[1,:], T, P)[1]
            mu = (mu_gas * np.sum(self.me[0,:]) / rho_gas + mu_liq * 
                np.sum(self.me[1,:]) / rho_liq) / (np.sum(self.me[0,:]) / \
                rho_gas + np.sum(self.me[1,:]) / rho_liq)
        
        return mu

    def density(self, T, P):
        """
        Compute the density of the petroleum fluid
        
        """
        # Compute the mixture density
        rho_gas, rho_liq, alpha, rho = lfm.mixture_density(self.me, T, P, 
            self.fluid)
            
        return (rho_gas, rho_liq, alpha, rho)
    
    def solubility(self, T, P, S):
        """
        Compute the solubilities of each component in the petroleum fluid
        
        """
        m = self.me[0,:] + self.me[1,:]
        # Get the mass fractions of gas and liquid
        f_gas = np.sum(self.me[0,:]) / np.sum(m)
        f_liq = np.sum(self.me[1,:]) / np.sum(m)
        
        # Compute the solubilities for gas and liquid
        if f_gas <= 0.001:
            # This is essentially pure liquid
            Cs = self.fluid.solubility(self.me[1,:], T, P, S)[1,:]
            
        elif f_liq <= 0.001:
            # This is essentially pure gas
            Cs = self.fluid.solubility(self.me[0,:], T, P, S)[0,:]
            
        else:
            # This is a gas/liquid mixture...the solubilities of gas and
            # liquid are equal at equilibrium.  Return the gas-phase 
            # solubilities
            Cs = self.fluid.solubility(self.me[0,:], T, P, S)[0,:]
        
        return Cs


class PipeFracture(object):
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
    dx : np.array
        A three-dimensional array of characteristic distances (m) that each
        straight segment of the random-walk path should have (dx, dy, dz).
        Anisotropy is introduced by providing different values of dx, dy, and
        dz. These sizes should represent the actual scales within the bedrock.
    du : np.array
        A three-dimensional, normalized pseudo-velocity vector (--) that
        allows the fracture network to connect from point-to-point. This
        parameter encapsulates the advection speed normalized by the velocity
        along the mean free path of the diffusion step. This must have a
        positive vertical component so that the network will eventually reach
        the seafloor.
    delta_s : float
        Step-size to use when building the network. This scale should
        normally be larger than the in situ bedrock scale. The model
        constructs a network that matches this scale.
    n_pipes : float
        Number of pipes to consider occupying one fracture of the UT model
        domain (--)
    Lf : int
        Length of one fracture in the UT model domain (m)
    mu_A : float, default=None
        Arithmetic average of the cross-sectional areas of each segment of
        the fracture network (m^2).  If the aperture size is provided in the
        profile data, this parameter it taken from that data.
    sigma_A : float, default=1.
        Normalized sample standard deviation (--) of the cross-sectional 
        areas of each segment of the fracture network, with the normalization
        factor taken as mu_A.
    
    
    """
    def __init__(self, profile, x0, H, Hs, dx, du, delta_s, Lf, n_pipes, 
        mu_A=None, sigma_A=1.):
        super(PipeFracture, self).__init__()
        
        self.profile = profile
        self.x0 = x0
        self.H = H
        self.Hs = Hs
        self.z0 = H + Hs
        self.dx = dx
        self.du = du
        self.delta_s = delta_s
        self.Lf = Lf
        self.n_pipes = float(n_pipes)
        
        # Convert pseudo-velocity to a coordinate system with z positive 
        # downward
        self.du[2] *= -1.
        
        # Check whether the aperture data are available if needed
        if isinstance(mu_A, type(None)):
            # Get the average aperture from the profile data
            z0 = H + Hs
            z1 = H
            z = np.linspace(0.99 * z0, 1.01 * z1, num=250)
            data = self.profile.get_values(z, ['aperture', 
                'gas_saturation'])
            a = data[:,0]
            gas_sat = data[:,1]
            if a[0] == 0.:
                print('\nWarning: average fracture size mu_A not provided')
                print('         and not found in profile data...using')
                print('         default value of 0.1 mm.\n')
                self.mu_A = 0.0001
            else:
                # Reduce the aperture by the fraction filled with gas
                a *= gas_sat
                self.mu_A = np.mean(a)
        self.sigma_A = sigma_A
        
        # Generate the associated fracture network
        self._gen_fracture_path()
        
        # Set the derivs() function that computes the subsurface fraction
        # simulation for this geometry
        self.derivs = lfm.pipe_derivs
    
    def _gen_fracture_path(self):
        """
        Generate a random-walk model of the fracture path
        
        """
        # Echo progress to the screen
        print('\n-- Generating Fracture Pathway for a Pipe Fracture --')
        print('\nGenerating pipe path from %g (m) to ...' % 
            (self.Hs + self.H))
            
        # Create a random-walk network of line segments
        self.xp = pipe_fracture_network(self.H, self.Hs, self.dx, self.du,
            self.delta_s)
        
        print('%g (m) depth' % self.H)
        
        # Shift network to origin
        self.xp[:,:2] = self.x0 + self.xp[:,:2]
        
        # Generate a path-length coorinate system and aperture size
        from scipy.stats import lognorm
        sp = np.zeros(self.xp.shape[0])
        As = np.zeros(self.xp.shape[0])
        psteps = 1000
        print('\nFilling in properties of the fracture pathway:')
        for i in range(len(sp) - 1):
            
            # Set the base of this segment at the end of the previous segment
            sp[i+1] = sp[i]
            # Add the length of the current segment
            sp[i+1] += np.sqrt(np.sum((self.xp[i+1,:] - self.xp[i,:])**2))
        
            # Look up the aperture size at this depth
            zp = (self.xp[i,2] + self.xp[i+1,2]) / 2.
            a, gas_sat = self.profile.get_values(zp, ['aperture',
                'gas_saturation'])
            
            # Get the cross-sectional area of this segment
            if a == 0.:
                # The aperture is not in the profile data
                mu_A = self.mu_A
            else:
                # Reduce the aperture by the fraction filled with gas
                a *= gas_sat
                # Compute the area of the fracture assigned to this pipe
                mu_A = a * self.Lf / self.n_pipes
                
            # Set the standard deviation at this point
            sigma_A = self.sigma_A * mu_A
            
            # Convert mu and sigma to lognormal coordinates
            mu = np.log(mu_A / np.sqrt(1. + (sigma_A / mu_A)**2))
            sigma = np.sqrt(np.log(1. + (sigma_A / mu_A)**2))
            
            # Generate a pipe area for this point
            As[i] = lognorm.rvs(sigma, scale=np.exp(mu), size=1)
            
            # Echo progress to the screen
            if i % psteps == 0.:
                print('    Depth : %g (m), Node:  %d of %d, a: %g (mm)' 
                    % (zp, i, len(sp) - 1, np.sqrt(4. * mu_A / np.pi)))
        
        self.sp = sp
        self.As = As
        self.s_max = self.sp[-1]
        
        # Create an interpolator for the x-coorindate of any s-value
        from scipy.interpolate import interp1d
        fill_value = (self.xp[0,:], self.xp[-1,:])
        self.xs = interp1d(self.sp, self.xp, axis=0, fill_value=fill_value,
            bounds_error=False)

        # Create an interpolator for the area of any segment
        fill_value = (self.As[0], self.As[-1])
        self.As = interp1d(self.sp, self.As, axis=0, fill_value=fill_value,
            bounds_error=False)
        
        # Set a flag indicating that no simulation has yet been conducted 
        # on this fracture pathway
        self.sim_stored = False
    
    def get_x_position(self, s):
        """
        Return the x, y, z coordinates of a segment at position s along the
        fracture path
        
        """
        return self.xs(s)

    def get_xsec_area(self, s):
        """
        Return the cross-sectional area of a segment at position s along the
        fracture path
        
        """
        return self.As(s)
    
    def get_width(self, s):
        """
        Return the characteristic width of the segment
        
        """
        # For a pipe, return the diameter
        Ap = self.get_xsec_area(s)
        dp = np.sqrt(Ap / np.pi) * 2.
        
        return dp

    def get_surface_area(self, s, V):
        """
        Return the surface area of a cylinder with volume V
        
        """
        Ap = self.get_xsec_area(s)
        hs = V / Ap
        dp = np.sqrt(Ap / np.pi) * 2.
        As =  np.pi * dp * hs
        
        return As

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
        
    
# Functions used by subsurface fracture model classes ------------------------

def pipe_fracture_network(H, Hs, dx, du, delta_s):
    """
    Generate the fracture network for the given pipe network parameters
    
    Parameters
    ----------
    H : float
        Depth (m) at the seafloor
    Hs : float
        Thickness (m) of the subsurface layer between the petroleum reservoir
        supplying the fracture and the seafloor.    
    dx : np.array
        A three-dimensional array of characteristic distances (m) that each
        straight segment of the random-walk path should have (dx, dy, dz).
        Anisotropy is introduced by providing different values of dx, dy, and
        dz. These sizes should represent the actual scales within the bedrock.
    du : np.array
        A three-dimensional, normalized pseudo-velocity vector (--) that
        allows the fracture network to connect from point-to-point. This
        parameter encapsulates the advection speed normalized by the velocity
        along the mean free path of the diffusion step. This must have a
        positive vertical component so that the network will eventually reach
        the seafloor.
    delta_s : float
        Step-size to use when building the network. This scale should
        normally be larger than the in situ bedrock scale. The model
        constructs a network that matches this scale.
    
    """
    # Create an empty list to hold the x, y, z coordinates of the pipe
    # segments of the fracture path
    x = []
    
    # Add the origin of the fracture network
    x.append(np.array([0., 0., H + Hs]))
    
    # Compute the effective diffusivities for the random-walk model of the
    # fracture network
    ds = np.sqrt(np.sum(dx**2))
    D = dx**2 / ds
    
    # Import a random number generator to create random-walk steps...here, 
    # we use a normal distribution with zero mean and standard deviation 
    # of one.
    from scipy.stats import norm
    mu = 0.
    sigma = 1.
    
    # Find points along the fracture network until we reach the seabed
    psteps = 1000
    k = 0
    while x[-1][2] > H:
        
        # Generate the next point along the trajectory...first, the random 
        # step
        r = norm.rvs(mu, scale=sigma, size=3)
        x_new = x[-1] + r * np.sqrt(D * delta_s)
        
        # ...then, the deterministic, pseudo-advection step
        x_new += du * delta_s
        
        # Append this point to the network
        x.append(x_new)
        k += 1
            
    # Convert x to a numpy array
    x = np.array(x)
    
    # For the final point to be exactly at the mudline
    dl = (H - x[-2,2]) / (x[-1,2] - x[-2,2])
    x[-1,:] = dl * (x[-1,:] - x[-2,:]) + x[-2,:]
    
    # Return the positions along each link of the network
    return x

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

def plot_component_map(t, y, derived_vars, parcel, fracture, p, comps, fig):
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
        x[i,:] = fracture.get_x_position(s[i])
    
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
    
    # Plot each component one at a time
    figure = plt.figure(fig+1, figsize=figsize)
    plt.clf()
    
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
