"""
Lagrangian Fracture Model
-------------------------

Simulate the transport and transformation of a Lagrangian parcel as it
transits through a fracture network in the marine subsurface.


WARNING!!!
THIS SCRIPT IS UNDER CONSTRUCTION AND DOES NOT YET REPRESENT AN ACCURATE
SIMULATION OF THE OCEAN SUBSURFACE.  PLEASE DO NOT USE THIS FOR ANY PURPOSES
AT THIS TIME.  Scott Socolofsky, 02/03/2022.

"""
# S. Socolofsky, January 2021, <socolofs@tamu.edu>

from __future__ import (absolute_import, division, print_function)



import numpy as np
import matplotlib.pyplot as plt


def pipe_derivs(t, y, parcel):
    """
    Coupled equations for the Lagrangian pipe-flow model
    
    Parameters
    ----------
    t : float
        Current time in the simulation (s)
    y : ndarray
        Array of present values for the model state space
    profile : ambient.Profile object
        Ambient temperature, salinity, pressure, and dissolved components
        data stored as a function of depth (m) from the ocean surface
    parcel : LagrangianParcel object
        Object describing the properties and behavior of the Lagrangian 
        parcel
    p : ModelParams object
        Object containing all of the model constants that are not adjustable
        by the user
    
    """
    # Initialize an output variable with stationary forcing
    yp = np.zeros(y.shape)
    
    # Update the Lagrangian parcel with this ambient data
    parcel.update(t, y)
    
    # Extract the state space variables for speed and ease of reading code
    s = y[0]
    m = y[1:-1]
    T = y[-1] / (np.sum(m) * parcel.cp)
    
    # Compute the advection step
    yp[0] = parcel.us
    
    # Compute the mass transfer
    yp[1:-1] = 0.
    
    # Compute the heat transfer
    yp[-1] = 0.
    
    # Return the derivatives
    return yp

def calculate_pipe(s_max, parcel, t0, y0, delta_t):
    """
    Simulate the fracture tube as continuous pipe flow
    
    Parameters
    ----------
    s_max : float
        Final position in the along-pipe (s) coordinate system to stop
        integration (m)
    profile : ambient.Profile object
        Ambient temperature, salinity, pressure, and dissolved components
        data stored as a function of depth (m) from the ocean surface
    parcel : LagrangianParcel object
        Object describing the properties and behavior of the Lagrangian 
        parcel
    p : ModelParams object
        Object containing all of the model constants that are not adjustable
        by the user
    t0 : float
        Time corresponding to the initial time of the simulation (s)
    y0 : ndarray
        Array of initial conditions for the model state space
    delta_t : float
        Maximum step size to use in the adaptive-step-size ODE solver (s)
    
    Returns
    -------
    t : ndarray
        List of simulation times (s)
    y : ndarray
        Two-dimensional vector of state space solutions.  Each row 
        presents the solution at one simulation time
    
    Notes
    -----
    The model state space is organized as follows. The y[0] is the position
    in the along-pipe (s) coordinate system, y[1:-1] are the masses of the
    components and psuedo-components of all components in the petroleum
    mixture, and y[-1] is the heat content of the Lagrangian parcel.
    
    """
    # Import the ODE solver library
    from scipy import integrate
    
    # Create the integrator object:  use "vode" with "backward 
    # differentiation formula" for stiff ODEs
    r = integrate.ode(pipe_derivs).set_integrator('vode', method='bdf',
         atol=1.e-6, rtol=1.e-3, order=5, max_step=delta_t)
    
    # Initialize the state space
    t0 = 0.
    r.set_initial_value(y0, t0)
    
    # Set passing variables for derivs method
    r.set_f_params(parcel)
    
    # Create vectors (using the list data type) to store the solution 
    t = [t0]
    y = [y0]
    
    # Integrate to the top of the subsurface region (parcel.z_min)
    k = 0
    k_limit = 300000
    t_limit = 14. * 24 * 60 * 60.
    psteps = 10
    stop = False
    while r.successful() and not stop:
        
        # Print progress to the screen
        if k % psteps == 0:
            print('   Distance: %g (m), t: %g (s), k: %d, m: %g' %
                (r.y[0], r.t, k, np.sum(r.y[1:-1])))
        
        # Perform one step of the integration
        r.integrate(t[-1] + delta_t, step=True)
        
        # Store the results
        t.append(r.t)
        y.append(r.y)
        k += 1
        
        # Evaluate the stop criteria
        if r.successful():
            
            # Check if we reached the end of the pipe
            if r.y[0] >= s_max:
                stop = True
            
            # Check if the control volume has lost all its mass
            if np.sum(r.y[1:-1]) <= 0.:
                stop = True
            
            # Make sure the iterations eventually stop if we are stuck
            if k > k_limit:
                stop = True
                print('\nWARNING!! Model stopped because it exceeded')
                print('          k = %d iterations' % k_limit)
            
            # Set a maximum time to simulate
            if r.t > t_limit:
                stop = True
                print('\nWARNING!! Model stopped because it reached the')
                print('          maximum allowable time of %d (days)' % 
                    (t_limit / 24. / 60. / 60.))
    
    # Convert the solution vectors to numpy arrays
    t = np.array(t)
    y = np.array(y)
    
    # Print the final position to the screen
    print('   Distance: %g (m), t: %g (s), k: %d, m: %g' %
        (r.y[0], r.t, k, np.sum(r.y[1:-1])))
    
    # Return the results
    return (t, y)

def main_ic(z0, u0, Ap, mass_frac, fluid, cp, profile):
    """
    Compute the initial conditions from the given information
    
    """
    # Get the ambient conditions at this point
    Ta, Sa, Pa = profile.get_values(z0, ['temperature', 'salinity', 
        'pressure'])
    
    # Compute the fluid density
    rho = fluid.density(mass_frac, Ta, Pa)[0]
    print(rho)
    
    # Compute the mass flux
    m_dot = u0 * Ap * rho
    
    # Get a mass for this element such that the initial Lagrangian element is 
    # shorter than it is wide
    dp = np.sqrt(Ap / np.pi) * 2.
    hp = dp / 3.
    Vp = Ap * hp
    mp = Vp * rho
    m = mass_frac * mp
    
    # Set the initial conditions
    t0 = 0.
    y0 = np.zeros((2 + len(m)))
    y0[0] = 0.
    y0[1:-1] = m
    y0[-1] = Ta * np.sum(m) * cp
    
    # Return the results
    return (t0, y0, m_dot)
    
    