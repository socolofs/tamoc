"""
Lagrangian Fracture Model
-------------------------

Simulate the transport and transformation of a Lagrangian parcel as it
transits through a fracture network in the marine subsurface.

WARNING!!!
----------

THIS MODULE IS UNDER CONSTRUCTION AND DOES NOT YET REPRESENT A COMPLETE
SIMULATION OF THE OCEAN SUBSURFACE. PLEASE DO NOT USE THIS MODULE FOR ANY
CRITICAL CALCULATIONS, DESIGN, OR DECISION-MAKING PURPOSES. Scott A.
Socolofsky, 10/28/2022.

"""
# S. Socolofsky, Texas A&M University, October 2022, <socolofs@tamu.edu>

from __future__ import (absolute_import, division, print_function)

import numpy as np
import matplotlib.pyplot as plt

def pipe_derivs(t, y, parcel, p):
    """
    Coupled equations for the subsurface Lagrangian pipe-flow model
    
    Parameters
    ----------
    t : float
        Current time in the simulation (s)
    y : ndarray
        Array of present values for the model state space
    parcel : LagParcel object
        Object describing the properties and behavior of the Lagrangian 
        parcel
    p : `ModelParams`
        Object containing the fixed model parameters for the model
    
    """
    # Initialize an output variable with stationary forcing
    yp = np.zeros(y.shape)
    
    # Update the Lagrangian parcel with this ambient data
    parcel.update(t, y)
    
    # Extract the state space variables
    s = parcel.s
    m = parcel.m
    T = parcel.T
    
    # Compute the advection step
    yp[0] = parcel.us
    
    # Compute the mass transfer and biodegradation
    md_diss = - parcel.As * parcel.beta * (parcel.Cs - parcel.Ca)
    md_biodeg = parcel.k_bio * parcel.m
    yp[1:-1] = md_diss + md_biodeg
    
    # Compute the heat transfer
    if parcel.beta_T == 0.:
        # Parcel is at thermal equilibrium; hence, heat transfer has ceased
        yp[-1] = 0.
    else:
        # Contribution from conduction and convection
        dH = - parcel.rho * p.cp * parcel.As * parcel.K_T * parcel.beta_T * \
            (T - parcel.Ta)
        yp[-1] = dH
        
    # and contribution from mass loss
    yp[-1] += p.cp * np.sum(md_diss + md_biodeg) * T
    
    # Return the derivatives
    return yp


def slot_derivs(t, y, parcel, p):
    """
    Coupled equations for the subsurface Lagrangian slot-fracture model
    
    Parameters
    ----------
    t : float
        Current time in the simulation (s)
    y : ndarray
        Array of present values for the model state space
    parcel : LagParcel object
        Object describing the properties and behavior of the Lagrangian 
        parcel
    p : `ModelParams`
        Object containing the fixed model parameters for the model
    
    """
    # Initialize an output variable with stationary forcing
    yp = np.zeros(y.shape)
    
    # Update the Lagrangian parcel with this ambient data
    parcel.update(t, y)
    
    # Extract the state space variables
    s = parcel.s
    m = parcel.m
    T = parcel.T
    
    # Compute the advection step
    yp[0] = parcel.us
    
    # Compute the mass transfer and biodegradation
    md_diss = - parcel.As * parcel.K * parcel.beta * (parcel.Cs - parcel.Ca)
    md_biodeg = parcel.k_bio * parcel.m
    yp[1:-1] = md_diss + md_biodeg
    
    # Compute the heat transfer
    if parcel.beta_T == 0.:
        # Parcel is at thermal equilibrium; hence, heat transfer has ceased
        yp[-1] = 0.
    else:
        # Contribution from conduction and convection
        dH = - parcel.rho * p.cp * parcel.As * parcel.K_T * parcel.beta_T * \
            (T - parcel.Ta)
        yp[-1] = dH
        
    # and contribution from mass loss
    yp[-1] += p.cp * np.sum(md_diss + md_biodeg) * T
    
    # Return the derivatives
    return yp


def calculate(t0, y0, parcel, p, delta_t, s_max):
    """
    Simulate the subsurface fracture
    
    Parameters
    ----------
    t0 : float
        Time corresponding to the initial time of the simulation (s)
    y0 : ndarray
        Array of initial conditions for the model state space
    parcel : LagParcel object
        Object describing the properties and behavior of the Lagrangian 
        parcel
    p : `ModelParams`
        Object containing the fixed model parameters for the model
    delta_t : float
        Maximum step size to use in the adaptive-step-size ODE solver (s)
    s_max : float
        Maximum along-path distance (m) to compute
    
    Returns
    -------
    t : ndarray
        List of simulation times (s)
    y : ndarray
        Two-dimensional vector of state space solutions.  Each row 
        presents the solution at one simulation time
    derived_vars : ndarray
        Two-dimensional vector of derived variables. Each row presents the
        solution at one simulation time. Derived variables are
        non-state-space quantities that are computed every time-step and may
        be of interest for post-processing, such as gas fraction.
    
    """
    # Import the ODE solver library
    from scipy import integrate
    
    # Create the integrator object:  use "vode" with "backward 
    # differentiation formula" for stiff ODEs
    r = integrate.ode(parcel.fracture.derivs).set_integrator('vode',
        method='bdf', atol=1.e-6, rtol=1.e-3, order=5, max_step=delta_t)
    
    # Initialize the state space
    r.set_initial_value(y0, t0)
    
    # Update the parcel at the initial position
    parcel.update(t0, y0)
    
    # Set passing variables for derivs method
    r.set_f_params(parcel, p)
    
    # Create vectors (using the list data type) to store the solution 
    t = [t0]
    y = [y0]
    
    # Create vectors to save some of the computed variables during a 
    # simulation
    derived_vars = [parcel.alpha]
    
    # Integrate to the top of the subsurface region
    k = 0
    k_limit = 300000
    psteps = 1
    stop = False
    while r.successful() and not stop:
        
        # Print progress to the screen
        if k % psteps == 0:
            print('   Distance: %g (m), t: %g (hr), k: %d, m: %g' %
                (r.y[0], r.t/60./60., k, np.sum(r.y[1:-1])))
        
        # Perform one step of the integration
        r.integrate(t[-1] + delta_t, step=True)
        
        # Store the results
        if parcel.K_T == 0:
            # Make the state-space heat correct
            r.y[-1] = np.sum(r.y[1:-1]) * p.cp * parcel.Ta
        for i in range(len(r.y[1:-1])):
            if r.y[i+1] < 0.:
                # Concentration should not overshoot zero
                r.y[i+1] = 0.
        t.append(r.t)
        y.append(r.y)
        derived_vars.append(parcel.alpha)
        k += 1
        
        # Evaluate the stop criteria
        if r.successful(): 
            
            if r.y[0] >= s_max:
                # We integrated up to the maximum allowable distance
                stop = True
                print('\n -> Reached maximum downstream distance...')
            
            if np.sum(parcel.beta) == 0:
                # Mass transfer coefficients are all zero...stop integration
                stop = True
                print('\n -> All petroleum components dissolved...')
            
            if np.sum(np.where(np.isnan(r.y))) > 0:
                # The solution is no longer valid...stop integration
                stop = True
                print('\n -> State space contains NaN values...')
            
            if k > k_limit:
                # Reached iteration limit
                stop = True
                print('\n -> Reached maximum number of iterations...')
            
            if parcel.xp[2] <= parcel.fracture.H:
                # Reached the top of the subsurface layer
                stop = True
                print('\n -> Reached the mud line...')
        
        else:
            print('\n -> ODE Solver stopped automatically...')
    
    # Convert the solution vectors to numpy arrays
    t = np.array(t)
    y = np.array(y)
    derived_vars = np.array(derived_vars)
    
    # Print the final position to the screen
    print('   Done.  \n\nFinal position:')
    print('   Distance: %g (m), t: %g (hr), k: %d, m: %g' %
        (r.y[0], r.t/60./60., k, np.sum(r.y[1:-1])))
    
    # Return the results
    return (t, y, derived_vars)


def main_ic(s0, T0, fracture, mass_frac, fluid, profile, p):
    """
    Create the initial state space vector for a fracture simulation
    
    Parameters
    ----------
    s0 : float
        Initial position along the fracture pathway (m)
    T0 : float
        Initial temperature (K) of the fracture fluid.  If set to `None`, then
        the local temperature at the initial point will be used.
    fracture : `SlotFracture`
        A `SlotFracture` object that can report all of the geometric 
        properties of the fracture pathway
    mass_frac : ndarray
        Array of masses fraction specifying the initial composition of the
        simulated fluid
    fluid : `dbm.FluidMixture`
        A discrete bubble model (`dbm`) `FluidMixture` object containing
        the equations of state of the tracked fluid
    profile : `ambient.Profile`
        An `ambient.Profile` object that contains the ambient properties
        of the water surrounding the fracture
    p : `ModelParams`
        A `ModelParams` object that contains the fixed model parameters
        of the present simulation
    
    Returns
    -------
    y0 : ndarray
        Array of initial state space variables.  The array is organized as 
        follows:  position along the fracture pathway (m), masses of each
        component of the simulated fluid (kg) in the initial Lagrangian
        element, and heat content (J) of the initial Lagrangian element.
    m0_dot : float
        Total mass flux (kg/s) of the fluid in the fracture at the initial
        position.  Note:  this value is taken directly from the UT model
        data stored in the `ambient.Profile` object.
    
    """
    # Get the Cartesian coordinates of the initial position
    x0 = fracture.get_x(s0)
    
    # Get the properties of the fracture fluid from the UT model
    w0, u0, rho_0 = profile.get_values(x0[2], ['aperture_filled_by_oil', 
        'oil_velocity', 'oil_density'])
    
    # Compute the initial mass flux
    m0_dot = rho_0 * (u0 * w0 * fracture.Lx)
    
    # Create the initial Lagrangian element
    if isinstance(T0, type(None)):
        T0 = profile.get_values(x0[2], 'temperature')
    
    # Make the initial Lagrangian element 10 times taller than the 
    # fracture aperture
    h0 = 10. * w0
    V0 = w0 * fracture.Lx * h0
    
    # Compute the initial mass in the Lagrangian element
    m0 = rho_0 * V0
    
    # Get the mass composition and heat of this element
    m = mass_frac * m0
    h = np.sum(m) * p.cp * T0
    
    # Fill the state-space vector
    y0 = np.zeros(len(m)+2)
    y0[0] = s0
    y0[1:-1] = m
    y0[-1] = h
    
    # Return the state space and the initial mass flux
    return (y0, m0_dot)
