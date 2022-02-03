"""
subsurface_fracture_model.py
----------------------------

Simulate the evolution of a petroleum fluid transiting a single pathway
through a subsurface fracture.

WARNING!!!
THIS MODULE IS UNDER CONSTRUCTION AND DOES NOT YET REPRESENT AN ACCURATE
SIMULATION OF THE OCEAN SUBSURFACE.  PLEASE DO NOT USE THIS FOR ANY PURPOSES
AT THIS TIME.  Scott Socolofsky, 02/03/2022.

"""
# S. Socolofsky, January 2021, <socolofs@tamu.edu>

from __future__ import (absolute_import, division, print_function)

from tamoc import seawater, dbm
from tamoc import lfm

import numpy as np
import matplotlib.pyplot as plt


class Model(object):
    """
    Master class object for controlling and post-processing the model
    
    Master class object for the subsurface fracture model. This model class
    generates a fracture network and can simulate the transport,
    transformation, and chemical reactions of petroleum fluids flowing
    through those fractures.
    
    Parameters
    ----------
    profile : `ambient.Profile` object, default = None
        An object containing the ambient temperature, salinity, pressure, and
        ambient concentration data and associated methods to interpolate that
        data to different subsurface depths. If stored as a netCDF dataset,
        the `ambient.Profile` object may be open or closed at instantiation.
        If open, the initializer will close the file since this model does
        not support changing the ambient data once initialized.
    
    """
    def __init__(self, H, Hs, dx, du, mu_d, sigma_d, 
        x0=np.zeros(2), delta_s=1.):
        super(Model, self).__init__()
        
        # Store the model parameters
        self.H = H
        self.Hs = Hs
        self.dx = dx
        self.du = du
        self.mu_d = mu_d
        self.sigma_d = sigma_d
        self.x0 = x0
        self.delta_s = delta_s
        
        # Compute the derived quantities
        self.mu_A = np.pi * (self.mu_d / 2.)**2
        self.sigma_A = np.pi * (self.sigma_d / 2.)**2
        
        # Adjust to a z = depth coordinate system
        self.du[2] *= -1.
        
        # Set the fixed model parameters for the simulation that the user
        # cannot adjust
        self.p = ModelParams()
        
        # Generate the fracture network
        self._gen_fracture_path()
        
        # Set the simulation flag to false
        self.ran_sim = False

    def _gen_fracture_path(self):
        """
        Generate a fracture path for this model
        
        Parameters
        ----------
        H : float
            Water depth at the outlet of the fracture network (m)
        Hs : float
            Thickness of the subsurface layer from the petroleum reservoir
            to the sea floor (m)
        dx : ndarray
            Array of coordinate in the x-, y-, and z-directions giving the
            average fluctuating displacements along each straight tube
            segment of the network (m).
        du : ndarray
            Average pseudo-advection step in the x-, y-, and z-directions in
            length per line segment length (m)
        mu_A : float
            Arithmetic average of the cross-sectional areas of each segment
            of the fracture network (m^2)
        sigma_A : float
            Sample standard deviation of the cross-sectional areas of each
            segment of the fracture network (m^2)
        x0 : ndarray, default=np.zeros(3)
            Planar coordinates of the origin of this fracture network at the 
            petroleum source in the x- and y-directions (easterly and 
            northerly; meters)
        delta_s : float
            Spatial step to take when building the fracture network.
        
        Attributes
        ----------
        xp : ndarray
            Two-dimensional array of vertex positions for each segment in 
            the fracture network.  Each row contains a different point, with
            the three columns reporting the x-, y-, and z-coordinates of the 
            vertex (easterly, northerly, and depth; meters).
        
        """        
        # Create random-walk network 
        self.xp = fracture_network(self.H, self.Hs, self.dx, self.du, 
            self.delta_s)
        
        # Shift network to origin
        self.xp[:,:2] = self.x0 + self.xp[:,:2]
        
        # Select diameters for each segment of the network using a log-normal
        # distribution
        self.As = fracture_areas(self.xp.shape[0], self.mu_A, self.sigma_A)
        
        # Generate a path-length coordinate system
        sp = np.zeros(self.xp.shape[0])
        for i in range(len(sp)-1):
            # Set the base of this segment at the end of the previous segment
            sp[i+1] = sp[i]
            for j in range(3):
                # Add the length of this segment
                sp[i+1] += np.sqrt((self.xp[i+1,j] - self.xp[i,j])**2)
        self.sp = sp
        
        # Create some interpolators
        from scipy.interpolate import interp1d
        fill_value = (self.xp[0,:], self.xp[-1,:])
        self.x = interp1d(self.sp, self.xp, axis=0, 
            fill_value=fill_value, bounds_error=False)
    
    def simulate_pipe_flow(self, u0, mass_frac, fluid, profile, dt_max=60.):
        """
        Simulate the gas migration assuming tubes are full of reservoir fluid
        
        """
        # Store the input variables in the model attributes
        self.u0 = u0
        self.mass_frac = mass_frac
        self.fluid = fluid
        self.profile = profile
        self.dt_max = dt_max
        
        # Choose a heat capacity value for the petroleum fluid
        self.cp = seawater.cp() * 0.5
        
        # Create an initial state space vector from the given input variables
        t0, y0, self.m_dot = lfm.main_ic(self.x(0)[2], self.u0, 
            self.get_A_seg(0), self.mass_frac, self.fluid, self.cp,
            self.profile)
        
        # Create a Lagrangian Parcel object to handle the properties of the
        # Lagrangian element
        self.y_local = PipeParcel(t0, y0, self.p, self.m_dot, self.fluid,
            self.cp, self.x, self.get_A_seg, self.profile)
        
        # Compute the evolution along this flow path
        print('\n-- TEXAS A&M OIL-SPILL CALCULATOR (TAMOC) --')
        print('-- Subsurface Fracture Model              --\n')
        self.t, self.y = lfm.calculate_pipe(np.max(self.sp), self.y_local,
            t0, y0, self.dt_max)
        
    
    def get_A_seg(self, s):
        """
        Return the cross-sectional area for the segment at this path position
        
        """
        # Find the index to this path point
        if s > self.sp[-1]:
            # ODE is trying to solve a point just outside the network...
            ip = -1
        elif s < self.sp[0]:
            # ODE is trying to solve a point just before the network...
            ip = 0
        else:
            # This is within the network
            ip = np.max(np.where(self.sp <= s)) + 1
        
        # Return the area
        return self.As[ip]
        
        
    def new_fracture_path(self, H, Hs, dx, ds, du, mu_d, sigma_d, 
        x0=np.zeros(2), delta_s=1.):
        """ 
        Generate a new fracture path for this model
        
        """
        # Store the new model parameters
        self.H = H
        self.Hs = Hs
        self.dx = dx
        self.ds = ds
        self.du = du
        self.mu_d = mu_d
        self.sigma_d = sigma_d
        self.x0 = x0
        self.delta_s = delta_s
        
        # Compute the derived quantities
        self.mu_A = np.pi * (self.mu_d / 2.)**2
        self.sigma_A = np.pi * (self.sigma_d / 2.)**2
        
        # Create the new path
        self._gen_fracture_path()
        
        # Reset the simulation flag to False
        self.ran_sim = False
    
    def show_network(self, fig=1):
        """
        Plot the fracture network
        
        """
        show_network(self.xp, fig)
    
    def plot_state_space(self, fig=2):
        """
        Create a default plot of the state space solution
        
        """
        plot_state_space(self.t, self.y, self.y_local, fig)
    
    def plot_component_map(self, comps=None, fig=3):
        """
        docstring for plot_component_map
        
        """
        # If no composition specified, plot all components
        if isinstance(comps, type(None)):
            comps = self.y_local.composition
        
        # Create the plot
        plot_component_map(self.t, self.y, self.y_local, comps, fig)


class ModelParams(object):
    """
    Fixed parameters used in the subsurface fracture model
    
    Fixed model parameters that the user should not adjust and that are used
    by the subsurface fracture model.  These include parameters such as
    entrainment coefficients and other model constants that have been fit to 
    data and are not considered calibration coefficients.
    
    Parameters
    ----------
        
    """
    def __init__(self):
        super(ModelParams, self).__init__()
        
        # Set the model parameters
        pass


class PipeParcel(object):
    """
    Lagrangian element for a slice of fluid in a pipe-flow
    
    """
    def __init__(self, t0, y0, p, m_dot, fluid, cp, x, A, profile):
        super(PipeParcel, self).__init__()
        
        # Store the initial values of the input variables
        self.t0 = t0
        self.y0 = y0
        self.p = p
        self.m_dot = m_dot
        self.fluid = fluid
        self.cp = cp
        self.x = x
        self.A = A
        self.profile = profile
        
        # Extract some additional parameters
        self.composition = self.fluid.composition
        
        # Update the parcel with the present state space
        self.update(t0, y0)
    
    def update(self, t, y):
        """
        Extract the derived quantities from the state space vector
        
        """
        # Save the current state-space vector
        self.t = t
        self.y = y
        
        # Extract the state-space variables from the state-space
        self.s = y[0]
        self.m = y[1:-1]
        self.h = y[-1]
        
        # Get the current position in space and the pipe properties
        self.xp, self.yp, self.zp = self.x(self.s)
        self.Ap = self.A(self.s)
        
        # Get the local ambient conditions
        self.Pa, self.Ta, self.Sa = self.profile.get_values(self.zp, 
            ['pressure', 'temperature', 'salinity'])
        self.Ca = self.profile.get_values(self.zp, self.composition)
        self.rho_a = seawater.density(self.Ta, self.Sa, self.Pa)
        
        # Compute the derived quantities
        self.T = self.h / (np.sum(self.m) * self.cp)
        self.rho = self.fluid.density(self.m, self.T, self.Pa)[0]
        self.us = self.m_dot / self.Ap / self.rho
        self.V = np.sum(self.m) / self.rho
        self.hs = self.V / self.Ap
        self.ds = np.sqrt(self.Ap / np.pi) * 2.
        self.As = np.pi * self.ds


def fracture_network(H, Hs, dx, du, delta_s):
    """
    Generate the fracture network for the given parameters
    
    """
    # Set the origin of the fracture network
    x = [np.array([0., 0., H + Hs])]
    
    # Compute the effective diffusivities for the random-walk model of the
    # fracture network
    ds = np.sqrt(np.sum(dx**2))
    D = dx**2 / ds
    
    # Import a random number generator to create the steps
    from scipy.stats import norm
    mu = 0.
    sigma = 1.
    
    # Find points along the fracture network until we reach the seabed
    while x[-1][2] > H:
        
        # Generate the next point along the trajectory
        x_new = np.zeros(3)
        for i in range(len(x_new)):
            # First, the random step
            r = norm.rvs(mu, scale=sigma, size=1)
            x_new[i] = x[-1][i] + r * np.sqrt(D[i]* delta_s)
            
            # Then, the deterministic, pseudo-advection step
            x_new[i] += du[i] * delta_s
        
        x.append(x_new)
    
    # Convert x to a numpy array
    x = np.array(x)
    
    # Set the final point to be at the mud line
    dl = (H - x[-2,2]) / (x[-1,2] - x[-2,2])
    x[-1,:] = dl * (x[-1,:] - x[-2,:]) + x[-2,:]
    
    # Return the positions
    return x

def fracture_areas(n_A, mu_A, sigma_A, dist='lognorm'):
    """
    Generate a set of cross-sectional areas for each segment of a network
    
    Generate a set of cross-sectional areas with mean mu_A and standard
    deviation sigma_A for each segment of a fracture network given by the 
    coordinate points x, y, and z.  
    
    Parameters
    ----------
    n_A : int
        Number of segments in the fracture network
        mu_A : float
            Arithmetic average of the cross-sectional areas of each segment
            of the fracture network (m^2)
        sigma_A : float
            Sample standard deviation of the cross-sectional areas of each
            segment of the fracture network (m^2)
    """
    # Generate the areas from a probability density function
    if dist == 'lognorm':
        from scipy.stats import lognorm
        mu = np.log(mu_A / np.sqrt(1. + (sigma_A / mu_A)**2))
        sigma = np.sqrt(np.log(1. + (sigma_A / mu_A)**2))
        seg_A = lognorm.rvs(sigma, scale=np.exp(mu), size=n_A)
    
    # Return the areas
    return seg_A

def show_network(xp, fig):
    """
    docstring for show_network
    
    """
    # Create the figure
    plt.figure(fig, figsize=(11,6))
    plt.clf()
    
    # Some formatting commands
    marker_fmt = {'markerfacecolor':'w', 'label':'_no_legend_'}
    
    # Create two subplots
    ax = plt.subplot(121)
    ax.plot(xp[:,0], xp[:,2], 'b-')
    ax.plot(xp[:,1], xp[:,2], 'g-')
    ax.legend(('Easterly path', 'Northerly path'))
    
    ax.plot(xp[0,0], xp[0,2], 'ko', **marker_fmt)
    ax.plot(xp[-1,0], xp[-1,2], 'ko', **marker_fmt)
    ax.plot(xp[0,1], xp[0,2], 'ko', **marker_fmt)
    ax.plot(xp[-1,1], xp[-1,2], 'ko', **marker_fmt)
    ax.invert_yaxis()
    ax.set_xlabel('Distance, (m)')
    ax.set_ylabel('Depth, (m)')
    
    ax = plt.subplot(122)
    ax.plot(xp[:,0], xp[:,1], 'm-')
    ax.plot(xp[0,0], xp[0,1], 'ko', **marker_fmt)
    ax.plot(xp[-1,0], xp[-1,1], 'ko', **marker_fmt)
    
    ax.set_xlabel('Easterly distance, (m)')
    ax.set_ylabel('Northerly distance, (m)')
    
    plt.show()

def plot_state_space(t, y, parcel, fig):
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
        T[i] = h[i] / (parcel.cp * np.sum(m[i,:]))
    
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
    ax.legend(parcel.composition)
    
    # Plot the temperature
    ax = plt.subplot(133)
    ax.plot(T - 273.15, s)
    ax.set_xlabel('Temperature, (deg C)')
    
    plt.show()

def plot_component_map(t, y, parcel, comps, fig):
    """
    docstring for plot_component_map
    
    """
    from matplotlib.collections import LineCollection
    
    # Extract the state-space variables
    s = y[:,0]
    h = y[:,-1]
    
    # Get the x,y,z coordinates
    x = np.zeros((len(t), 3))
    for i in range(len(t)):
        x[i,:] = parcel.x(s[i])
    
    # Get the indices to the components
    im = [parcel.composition.index(comp) for comp in comps if comp in 
        parcel.composition]
    
    # Get the component masses
    m = np.zeros((len(t), len(comps)))
    for i in range(len(t)):
        m[i,:] = y[i,1:-1][im]
    
    # Figure out the figure size and number of subplots
    if len(comps) >= 5:
        cols = 5
    else:
        cols = len(comps)
    if cols == 5:
        rows = int(len(comps) / cols)
        if len(comps) % cols > 0:
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
#        ax.plot(m[:,i], s, label=comps[i])
#        ax.legend()
#        ax.set_xlabel(comps[i] + ' mass, (kg)')
#        plt.setp(ax.get_xticklabels(), rotation=30,                
#            horizontalalignment='right')
#        if i % cols == 0:
#            ax.set_ylabel('Distance, (m)')
#    plt.tight_layout()
#    plt.show()
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
    
        