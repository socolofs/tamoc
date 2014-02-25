"""
Stratified Integral Multiphase Plume
====================================

This module contains the numerical solution for the `stratified_plume_model`
module.  While these functions make use of the objects and methods defined
in the `stratified_plume_model`, these functions could be converted to 
Fortran to have a stand alone numerical solution to the integral plume 
model.  In that case, the `stratified_plume_model` module would be the 
interface sending input to this module and post-processing the output from 
this module.

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.
from tamoc import seawater

import numpy as np
from scipy import integrate
from scipy.optimize import fsolve

# ----------------------------------------------------------------------------
# Derivatives of the system of ODEs
# ----------------------------------------------------------------------------

def derivs_inner(z, y, yi, yo, particles, profile, p, neighbor):
    """
    Calculate the derivatives for the system of ODEs for the inner plume
    
    Calculates the right-hand-side of the system of ODEs for the inner plume
    state space.  These equations follow Socolofsky et al. (2008) very 
    closely, with the exception that multiple dispersed phase particles are
    allowed within the inner plume.  Heat transfer between the dispersed
    and continuous phase is also added.
    
    Parameters
    ----------
    z : float
        Current value for the independent variable (depth in m).
    y : ndarray
        Current value for the inner plume state space vector.
    yi : `InnerPlume`
        Object for manipulating the inner plume state space
    yo : `OuterPlume`
        Object for manipulating the outer plume state space
    particles : list of `Particle` objects
        List of `Particle` objects containing the dispersed phase local
        conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    neighbor : `scipy.interpolate.interp1d` object
        Container holding the latest solution for the outer plume state
        space
    
    Returns
    -------
    yp : ndarray
        A vector of the derivatives of the inner plume state space.
    
    See Also
    --------
    stratified_plume_model.InnerPlume, stratified_plume_model.OuterPlume, 
    stratified_plume_model.inner_main, calculate
    
    Notes
    -----
    It is important that the inner plume entrains fluid from either the 
    ambient water (whenever the outer plume is not present) or the outer 
    plume (whenever it is shrouding the inner plume).  This is accomplished
    in `stratified_plume_model.OuterPlume`:  if there is no outer plume 
    segment, then the ambient conditions are stored in the outer plume 
    variables.  Thus, `yo.c[i]` is equivalent to `ca[i]` when there is no 
    outer plume.  This behavior is true for temperature, salinity, density
    and concentration.
    
    """
    # Set up the output from the fuction to have the right size and type
    yp = np.zeros((yi.len, 1))
    
    # Update the inner plume object with the corrent solution
    yi.update(z, y, particles, profile, p)
    
    # Update the outer plume object at the current depth
    if z < np.min(neighbor.x):
        # This plume is above any existing outer plumes
        yo.update(z, np.zeros(yo.len), profile, p, yi.b)
    else:
        # Interpolate the outer plume solution to the current depth
        yo.update(z, neighbor(z), profile, p, yi.b)
    
    # Conservation of mass
    yp[0] = 2. * np.pi * yi.b * (p.alpha_1 * (yi.u + p.c1 * yo.u) + \
            p.alpha_2 * yo.u) + yi.Ep
            
    # Conservation of momentum
    yp[1] = 1. / p.gamma_i * (np.pi * p.g * yi.b**2 / p.rho_r * \
            (yi.Fb + p.lambda_2**2 * (1. - yi.Xi) * (yi.rho_a - \
            yi.rho)) + 2. * np.pi * yi.b * (p.alpha_1 * (yi.u + p.c1 * \
            yo.u) * yo.u + p.alpha_2 * yo.u * yi.u) + yi.Ep * yi.u)
            
    # Conservation of salinity
    yp[2] = 2. * np.pi * yi.b * (p.alpha_1 * (yi.u + p.c1 * yo.u) * \
            yo.s + p.alpha_2 * yo.u * yi.s) + yi.Ep * yi.s
            
    # Conservation of continuous phase fluid heat
    yp[3] = p.rho_r * seawater.cp() * (2. * np.pi * yi.b * (p.alpha_1 * \
            (yi.u + p.c1 * yo.u) * yo.T + p.alpha_2 * yo.u * yi.T) + \
            yi.Ep * yi.T)
            
    # Conservation equations for each dispersed phase
    idx = 4
    
    # Track the mass dissolving into the continuous phase
    delDiss = np.zeros(yi.nchems)
    for i in range(yi.np):
        delDiss_p = np.zeros(yi.nchems)
        
        if particles[i].particle.issoluble:
            for j in range(yi.nchems):
                
                # Conservation of particle mass for soluble particles 
                yp[idx] = -(particles[i].A * particles[i].nb0 / (yi.u + 
                            particles[i].us) * particles[i].beta[j] * 
                            (particles[i].Cs[j] - yi.c[j]))
                delDiss[j] += yp[idx]
                delDiss_p[j] += yp[idx]
                
                # Update continuous phase temperature with heat of solution
                yp[3] += yp[idx] * particles[i].particle.dH_solR[j] * p.Ru / \
                        particles[i].particle.M[j]
                idx += 1
            
        else:
            # Conservation of particle mass for insoluble particles
            yp[idx] = 0.
            idx += 1
        
        # Conservation of particle heat including dissolution mass transfer
        yp[idx] = -particles[i].A * particles[i].nb0 / (yi.u + 
                  particles[i].us) * particles[i].rho_p * particles[i].cp * \
                  particles[i].beta_T * (particles[i].T - yi.T) + \
                  np.sum(delDiss_p) * particles[i].cp * particles[i].T
        
        # Take the heat leaving the particle and put it in the continuous 
        # phase fluid
        yp[3] -= yp[idx] 
        idx += 1
    
    # Conservation equations for the dissolved constituents.
    for i in range(yi.nchems):
        yp[idx] = 2. * np.pi * yi.b * (p.alpha_1 * (yi.u + p.c1 * yo.u) * \
                  yo.c[i] + p.alpha_2 * yo.u * yi.c[i]) + yi.Ep * \
                  yi.c[i] - delDiss[i]
        idx += 1
    
    # z is positive downward (depth)
    return -yp

def derivs_outer(z, y, yi, yo, particles, profile, p, neighbor):
    """
    Calculate the derivatives for the system of ODEs for the outer plume
    
    Calculates the right-hand-side of the system of ODEs for the outer plume
    state space.  These equations follow those in Socolofsky et al. (2008).
    
    Parameters
    ----------
    z : float
        Current value for the independent variable (depth in m).
    y : ndarray
        Current value for the outer plume state space vector.
    yi : `InnerPlume`
        Object for manipulating the inner plume state space
    yo : `OuterPlume`
        Object for manipulating the outer plume state space
    particles : list of `Particle` objects
        List of `Particle` objects containing the dispersed phase local
        conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    neighbor : `scipy.interpolate.interp1d` object
        Container holding the latest solution for the outer plume state
        space
    
    Returns
    -------
    yp : ndarray
        A vector of the derivatives of the outer plume state space.
    
    See Also
    --------
    stratified_plume_model.InnerPlume, stratified_plume_model.OuterPlume, 
    stratified_plume_model.outer_main, calculate
    
    """
    # Set up the output from the function to have the correct size and type
    yp = np.zeros((yo.len,1))
    
    # Update the inner plume object at the current depth
    if z > np.max(neighbor.x):
        yi.update(z, np.zeros(yi.len), particles, profile, p)
    else:
        yi.update(z, neighbor(z), particles, profile, p)
    
    # Update the outer plume object with the current solution
    yo.update(z, y, profile, p, yi.b)
    
    # Conservation of Mass:
    yp[0] = 2. * np.pi * yi.b * (p.alpha_1 * (yi.u + p.c1 * yo.u) + \
            p.alpha_2 * yo.u) + 2. * np.pi * yo.b * p.alpha_3 * yo.u + yi.Ep
    
    # Conservation of Momentum:
    yp[1] = 1. / p.gamma_o * (-np.pi * p.g * (yo.b**2 - yi.b**2) / p.rho_r * \
            (yo.rho_a - yo.rho) + 2. * np.pi * yi.b * (p.alpha_1 * (yi.u + \
            p.c1 * yo.u) * yo.u + p.alpha_2 * yo.u * yi.u) + yi.Ep * yi.u)
    
    # Conservation of Salinity:
    yp[2] = 2. * np.pi * yi.b * (p.alpha_1 * (yi.u + p.c1 * yo.u) * yo.s + \
            p.alpha_2 * yo.u * yi.s) + 2. * np.pi * yo.b * p.alpha_3 * \
            yo.u * yo.Sa + yi.Ep * yi.s
    
    # Conservation of Heat:
    yp[3] = p.rho_r * seawater.cp() * (2. * np.pi * yi.b * (p.alpha_1 * \
            (yi.u + p.c1 * yo.u) * yo.T + p.alpha_2 * yo.u * yi.T) + \
            2. * np.pi * yo.b * p.alpha_3 * yo.u * yo.Ta + yi.Ep * yi.T)
    
    # Conservation of tracked chemical constituents:
    idx = 4
    for i in range(yo.nchems):
        yp[idx] = 2. * np.pi * yi.b * (p.alpha_1 * (yi.u + p.c1 * yo.u) * \
                yo.c[i] + p.alpha_2 * yo.u * yi.c[i]) + 2. * np.pi * yo.b * \
                p.alpha_3 * yo.u * yo.ca[i] + yi.Ep * yi.c[i]
        idx += 1
    
    # z is positive downward (depth)
    return yp


# ----------------------------------------------------------------------------
# Main integration controller
# ----------------------------------------------------------------------------

def calculate(yi, yo, particles, profile, p, neighbor, derivs, z0, y0, zf, 
              z_dir, delta_z):
    """
    Integrate an inner or outer plume segment from `z0` to `zf`
    
    Integrate the inner or outer plume over the range from `z0` to `zf`, 
    integrating in the direction (positive or negative) given by `z_dir`.
    
    Parameters
    ----------
    yi : `InnerPlume`
        Object for manipulating the inner plume state space
    yo : `OuterPlume`
        Object for manipulating the outer plume state space
    particles : list of `Particle` objects
        List of `Particle` objects containing the dispersed phase local
        conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    neighbor : `scipy.interpolate.interp1d` object
        Container holding the latest solution for the outer plume state
        space
    derivs : function handle
        Pointer to the function where the derivatives of the ODE system are
        stored.  Should be either `simp.derivs_inner` or `simp.derivs_outer`.
    z0 : float
        Initial depth (m)
    y0 : ndarray
        Initial values of the state space vector
    zf : float
        Final depth to calculate (m)
    z_dir : float
        Direction (+1 or -1) to integrate the vertical coordinate.  The inner
        plume integrates in the negative z-direction (to shallower depths),
        and the outer plume integrates in the positive z-direction (to 
        greater depths).
    delta_z : float
        Maximum step size to use in the simulation (m).  The ODE solver 
        in `calculate` is set up with adaptive step size integration, so 
        in theory this value determines the largest step size in the 
        output data, but not the numerical stability of the calculation.
    
    Returns
    -------
    z : ndarray
        Vector of elevations where the inner plume solution is obtained (m).
    y : ndarray
        Matrix of inner plume state space solutions.  Each row corresponds to
        a depth in z.
    
    See Also
    --------
    derivs_inner, derivs_outer, stratified_plume_model.Model, 
    stratified_plume_model.inner_main, stratified_plume_model.outer_main, 
    stratified_plume_model.InnerPlume, stratified_plume_model.OuterPlume
    
    """
    # Create the integrator object:  use "vode" with "backward 
    # differentiation formula" for stiff ODEs
    r = integrate.ode(derivs).set_integrator('vode', method='bdf', atol=1.e-6, 
        rtol=1e-3, order=5, max_step=delta_z)
    
    # Initialize the state space
    r.set_initial_value(y0, z0)
    
    # Set passing variables for derivs method
    r.set_f_params(yi, yo, particles, profile, p, neighbor)
    
    # Create vectors (using the list data type) to store the solution
    z = [z0]
    y = [y0]
    
    # Integrate to zf unless the solution stops naturally earlier
    k = 0
    psteps = 30.
    stop = False
    while r.successful() and not stop:
        
        # Print progress to the screen
        if np.remainder(np.float(k), psteps) == 0.:
            print '    Depth:  %g (m), k: %d' % (z[-1], k)
        
        # Perform one step of the integration
        r.integrate(zf, step=True)
        
        # Store the results
        if derivs == derivs_inner:
            # Store the correct temperature for the particles after heat 
            # transfer turns off
            r = correct_temperature(r, yi, particles, profile, p)
        z.append(r.t)
        y.append(r.y)
        k += 1
        
        # Evaluate the stop criteria
        if r.successful():
            # Check if we reached the free surface
            if z[-1] * z_dir >= zf * z_dir:
                stop = True
            # Check if the momentum went negative
            if y[-1][1] < 0.:
                stop = True
    
    # Convert solution to numpy arrays.
    z = np.array(z)
    y = np.array(y)
    
    # Remove any part of the solution with a negative momentum
    rows = y[:,1] >= 0
    z = z[rows,:]
    y = y[rows,:]
    
    # Return the solution
    if np.remainder(np.float(k), psteps) == 0.:
        print '    Depth:  %g (m), k: %d' % (z[-1], k)
    return (z, y)

def correct_temperature(r, yi, particles, profile, p):
    """
    Make sure the correct temperature is stored in the state space solution
    
    When the dispersed phase particles equilibrate to their surrounding 
    temperature, heat transfer is turned off by the methods in 
    `single_bubble_model.Particle`.  This is needed to prevent numerical
    oscillations as the particles become small.  Unfortunately, it is not as
    easy to make the numerical solution output the correct result once 
    particle temperature effectively stops being a state space variable.  
    
    Once heat transfer is turned off, all of the model methods use the 
    correct temperature (e.g., the ambient temperature) in all of the 
    equations coupled to the heat transfer equation and in all equations 
    involving particle temperature.  
    
    In order to prevent the state space variable for particle temperature 
    from blowing up as the mass goes to zero, we also continue to adjust the
    particle heat in the ODE solution to maintain a constant temperature. 
    This is done by setting `beta_T = 0`.  This is merely a numerical trick, 
    as all equations using the particle temperature know to use the ambient
    temperature when this is the case.  
    
    Hence, the purpose of this function is to simply overwrite the state 
    space solution containing the particle heat (returned by the ODE solver
    to maintain a constant particle temperature) with the correct particle
    heat yielding the ambient temperature for the particle temperature.
    
    Parameters
    ----------
    r : `scipy.integrate.ode` object
        ODE solution containing the currect values of the state space (e.g., 
        `r.y`).
    yi : `InnerPlume`
        Object for manipulating the inner plume state space
    particles : list of `Particle` objects
        List of `Particle` objects containing the dispersed phase local
        conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    
    Returns
    -------
    r : `scipy.integrate.ode` object
        Returns the original ODE object with the corrected solution stored
        in the public x and y.
    
    """
    # Update the inner plume state space with the current solution.  This 
    # will set the correct particle temperature in the attributes 
    # yi.particles[].T.  If heat transfer is still turned on, the answer will
    # be the value computed from the state space r.x and r.y; if heat 
    # transfer is turned off, the answer will be the ambient fluid
    # temperature (e.g., Ti).
    yi.update(r.x, r.y, particles, profile, p)
    
    # Find the heat conservation equation in the inner plume state space and
    # replace the heat with the correct value so that r.y always yields the
    # particle temperature determined above.
    idx = 4
    for i in range(len(particles)):
        idx += particles[i].particle.nc
        r.y[idx] = np.sum(particles[i].m) * particles[i].nb0 * \
                       particles[i].cp * particles[i].T
        idx += 1
    
    # Return the corrected solution
    return r

# ----------------------------------------------------------------------------
# General tools used throughout the model simulation
# ----------------------------------------------------------------------------

def bf_average(particles, rho, g, rho_r, parm):
    """
    Compute a buoyancy-flux-weighted average of `parm`
    
    Computes a weighted average of the values in `parm` using the kinematic
    buoyancy flux of each particle containing parm as the weight in the 
    average calculation.  
    
    Parameters
    ----------
    particles : list of `single_particle_model.Particle` objects
        Iterable list of dbm class objects describing each
        dispersed phase.
    rho : float
        Local density of ambient fluid outside plume (kg/m^3).
    g : float
        Acceleration of gravity (m/s^2).
    rho_r : float
        Model reference density (kg/m^3).
    parm : ndarray
        Numpy array of parameters to average, one value for each 
        dispersed phase entry (same as elements in parm).
    
    Returns
    -------
    parm_ave : float
        The weighted average of `parm`.
    
    """
    # Compute the total buoyancy flux of each dispersed phase particle in the 
    # simulation
    F = np.zeros(len(particles))
    for i in range(len(particles)):
        # Get the total particle volume flux
        Q = np.sum(particles[i].m) * particles[i].nb0 / particles[i].rho_p
        # Compute the particle kinematic buoyancy flux
        F[i] = g * (rho - particles[i].rho_p) / rho_r * Q
    
    # Return the buoyancy-flux-weighted value of parm
    if np.sum(F) == 0.:
        parm = 0.
    else:
        parm = np.sum(F * parm) / np.sum(F)
    
    return parm

def cp_model(epsilon, particles, rho_a, rho, g, rho_r, b, u):
    """
    Continuous peeling model of Crounse (2000)
    
    Computes the local peeling flux from the continuous model of Crounse
    (2000)
    
    Parameters
    ----------
    epsilon : float
        Continuous peeling model calibration factor (--).
    particles : list of `single_particle_model.Particle` objects
        Iterable list of dbm class objects describing each dispersed phase.
    rho_a : float
        Local density of ambient fluid outside plume (kg/m^3).
    rho : float
        Local density of plume fluid (kg/m^3)
    g : float
        Acceleration of gravity (m/s^2).
    rho_r : float
        Model reference density (kg/m^3).
    b : float
        Local plume half-width (m)
    u : float
        Local plume fluid velocity (m/s)
    
    """
    # Compute a buoyancy flux weighted average of the slip velocity
    us = np.array([particles[i].us for i in range(len(particles))])
    us = bf_average(particles, rho, g, rho_r, us)
    
    # Return the peeling flux
    return epsilon * (us / u)**2 * g * (rho_a - rho) / rho_r * \
           np.pi * b**2 / u

# ----------------------------------------------------------------------------
# Functions to compute inner plume initial conditions
# ---------------------------------------------------------------------------

def main_ic(profile, particles, p, z0, R):
    """
    Compute the initial conditions for the inner plume state space
    
    Compute the initial conditions at the release location for the inner
    plume state space
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the single bubble model simulation.
    particles : list of `Particle` objects
        List of `Particle` objects containing the dispersed phase initial
        conditions
    p : `stratified_plume_model.ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    z0 : float
        Depth of the release point (m)
    R : float
        Radius of the release port (m)
    
    """
    # Get the initial volume flux and related variables at the source
    Q, J, z, Ta, Sa, P, rho = flux_ic(profile, particles, p, z0, R)
    
    # Sequentially build the inner plume state space
    y = [Q, J, Sa * Q, p.rho_r * seawater.cp() * Ta * Q]
    
    # Include the dispersed phase variables (mass fluxes followed by heat 
    # flux, one particle at a time)
    chem_names = []
    for i in range(len(particles)):
        
        # Total mass flux
        y.extend(particles[i].m0 * particles[i].nb0)
        
        # Total heat flux
        y.append(np.sum(particles[i].m0) * particles[i].nb0 * 
                 particles[i].cp * particles[i].T0)
        
        # Chemical composition
        if particles[i].particle.issoluble:
            chem_names = chem_names + [chem for chem in 
                                       particles[i].composition if chem 
                                       not in chem_names]
    
    # Get the mass fluxes of dissolved components
    ca = profile.get_values(z, chem_names)
    C = ca * Q
    y.extend(C)
    
    # Return the initial depth, state space, and list of chem_names
    return (z, np.array(y), chem_names)

def flux_ic(profile, particles, p, z0, R):
    """
    Initial volume and momentum flux for the inner plume
    
    Uses the Wueest et al. (1992) plume Froude number method to estimate
    the amount of entrainment at the source of a dispersed phase plume with
    zero continuous phase flux (e.g., a pure bubble, droplet, or particle 
    plume)
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the single bubble model simulation.
    particles : list of `Particle` objects
        List of `Particle` objects containing the dispersed phase initial
        conditions
    p : `stratified_plume_model.ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    z0 : float
        Depth of the release point (m)
    R : float
        Radius of the release port (m)
    
    """
    # Get the ambient conditions at the discharge
    Ta, Sa, P = profile.get_values(z0, ['temperature', 'salinity', 
                                   'pressure'])
    rho = seawater.density(Ta, Sa, P)
    
    # Get the individual particle properties
    lambda_1 = np.zeros(len(particles))
    us = np.zeros(len(particles))
    rho_p = np.zeros(len(particles))
    Q = np.zeros(len(particles))
    for i in range(len(particles)):
        lambda_1[i] = particles[i].lambda_1
        us[i], rho_p[i] = particles[i].properties(particles[i].m, 
                          particles[i].T, P, Sa, Ta)[0:2]
        Q[i] = np.sum(particles[i].m) * particles[i].nb0 / rho_p[i]
    
    # Compute the buoyancy flux weighted average of lambda_1
    lambda_ave = bf_average(particles, rho, p.g, p.rho_r, lambda_1)
    
    # Calculate the initial velocity of entrained ambient fluid
    u_0 = np.sum(Q) / (np.pi * (lambda_ave * R)**2)
    u = wuest_ic(u_0, particles, lambda_1, lambda_ave, us, rho_p, rho, Q, R, 
                 p.g, p.Fr_0)
    
    # The initial width of the plume is the source radius
    b = R
    
    # Calcualte the flux variables
    Q = np.pi * b**2 * u
    J = Q * u
    
    # The initial condition is valid at the diffuser (e.g., no virtual point
    # source for the Wuest et al 1992 initial conditions)
    z = z0
    
    return (Q, J, z, Ta, Sa, P, rho)

def wuest_ic(u_0, particles, lambda_1, lambda_ave, us, rho_p, rho, Q, R, 
             g, Fr_0):
    """
    Compute the initial velocity of entrained ambient fluid
    
    Computes the initial velocity of the entrained ambient fluid following 
    the method in Wueest et al. (1992).  This method is implicit; thus, an 
    initial guess for the velocity and a root-finding approach is required.
    
    Parameters
    ----------
    u_0 : float
        Initial guess for the entrained fluid velocity (m/s)
    particles : list of `Particle` objects
        List of `Particle` objects containing the dispersed phase initial
        conditions
    lambda_1 : ndarray
        Spreading rate of the each dispersed phase particle in a plume (--)
    lambda_ave : float
        Buoyancy flux averaged value of lambda_1 (--)
    us : ndarray
        Slip velocity of each of the dispersed phase particles (m/s)
    rho_p : ndarray
        Density of each of the dispersed phase particles (kg/m^3)
    rho : float
        Density of the local ambient continuous phase fluid (kg/m^3)
    Q : ndarray
        Total volume flux of particles for each dispersed phase (m^3/s)
    R : float
        Radius of the release port (m)
    g : float
        Acceleration of gravity (m/s^2)
    Fr_0 : float
        Desired initial plume Froude number (--)
    
    Returns
    -------
    u : float
        The converged value of the entrained fluid velocity in m/s at the 
        release location in order to achieve the specified value of Fr_0.
    
    """
    # The Wuest et al. (1992) initial condition is implicit; define the 
    # residual for use in a root-finding algorithm
    def residual(u):
        """
        Compute the residual of the Wueest et al. (1992) initial condition
        using the current guess for the initial velocity u.
        
        Parameters
        ----------
        u : float
            the current guess for the initial velocity (m/s)
        
        Notes
        -----
        All parameters of `wuest_ic` are global to this function since it is
        a subfunction of `wuest_ic`.
        
        """
        # Get the void fraction for the current estimate of the mixture of 
        # dispersed phases and entrained ambient water
        xi = np.zeros(len(particles))
        for i in range(len(particles)):
            xi[i] = Q[i] / (np.pi * lambda_1[i]**2 * R**2 * (us[i] + 
                    2. * u / (1. + lambda_1[i]**2)))
        
        # Get the mixed-fluid plume density
        rho_m = np.sum(xi * rho_p) + (1. - np.sum(xi)) * rho
        
        # Calculate the deviation from the desired Froude number
        return Fr_0 - u / np.sqrt(2. * lambda_ave * R * g * 
                                  (rho - rho_m) / rho_m)
    
    return fsolve(residual, u_0)[0]

# ----------------------------------------------------------------------------
# Functions to compute outer plume initial conditions
# ---------------------------------------------------------------------------

def outer_surf(yi, p):
    """
    Compute the initial condition for the outer plume at the sea surface
    
    Computes the initial conditions for the first outer plume segment after
    the inner plume impinges on the free surface of the water body.  It is 
    assumed that the inner plume had significant volume flux and that this 
    first outer plume segment will be viable.
    
    Parameters
    ----------
    yi : `stratified_plume_model.InnerPlume` object
        Object for manipulating the inner plume state space
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    
    Returns
    -------
    z0 : float
        Initial depth of the outer plume segment (m).
    y0 : ndarray
        Initial dependent variables state space for the outer plume segment.
    
    """
    # The outer plume is a mixture of inner plume fluid and ambient fluid
    # entrained from the water surface
    Q = (1. + p.fe) * yi.Q
    T = (yi.T + yi.Ta * p.fe) * yi.Q / Q
    s = (yi.s + yi.Sa * p.fe) * yi.Q / Q
    c = (yi.c + yi.ca * p.fe) * yi.Q / Q
    rho = seawater.density(T, s, yi.P)
    
    # Use a Froude number approach to set the initial width and velocity
    u = outer_fr(yi.u, Q, yi.b, yi.rho_a, rho, p.g, p.Fro_0)
    
    # Calculate the outer plume state space variables
    y0 = []
    Q = -Q
    y0.append(Q)
    y0.append(Q * (-u))
    y0.append(s * Q)
    y0.append(p.rho_r * seawater.cp() * T * Q)
    y0.extend(c * Q)
    
    # Return the outer plume initial condition
    return (yi.z, np.array(y0))

def outer_dis(yi, particles, profile, p, neighbor, z_0):
    """
    Compute the initial condition for the outer plume at the DMPR
    
    Computes the initial conditions for the an outer plume segment at the 
    depth of maximum plume rise (DMPR) following full dissolution of the 
    dispersed phases.  
    
    Parameters
    ----------
    yi : `stratified_plume_model.InnerPlume` object
        Object for manipulating the inner plume state space.
    particles : list of `Particle` objects
        List of `Particle` objects containing the dispersed phase local
        conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    neighbor : `scipy.interpolate.interp1d` object
        Container holding the latest solution for the inner plume state
        space.
    z_0 : float
        Top of the inner plume calculation (m).
    
    Returns
    -------
    z0 : float
        Initial depth of the outer plume segment (m).
    y0 : ndarray
        Initial dependent variables state space for the outer plume segment.
    
    """
    # Search for the maximum flux near the top of the plume
    Qmax = neighbor.y[0,0]
    imax = 1
    while Qmax < neighbor.y[imax,0]:
        Qmax = neighbor.y[imax,0]
        imax += 1
    
    # Since most of this fluid will be regained as the outer plume descends
    # through Ep, take the initial volume flux as a small fraction (given
    # by model parameter qdis_ic)
    Q = p.qdis_ic * Qmax
    
    # Get the local plume properties at the top of the plume
    yi.update(z_0, neighbor(z_0), particles, profile, p)
    rho = yi.rho
    
    # Use a Froude number approach to set the initial width and velocity
    yi.u, -Q, yi.b, yi.rho_a, rho, p.g, p.Fro_0
    u = outer_fr(0.05, Q, yi.b, yi.rho_a, rho, p.g, p.Fro_0)
    
    # Calculate the outer plume state space variables
    y0 = []
    Q = -Q
    y0.append(Q)
    y0.append(Q * (-u))
    y0.append(yi.s * Q)
    y0.append(p.rho_r * seawater.cp() * yi.T * Q)
    y0.extend(yi.c * Q)
    
    # Return the outer plume initial condition
    return (yi.z, np.array(y0))

def outer_cpic(yi, yo, particles, profile, p, neighbor, z_0):
    """
    Compute the initial condition for the outer plume at depth
    
    Computes the initial conditions for the an outer plume segment within the 
    reservoir body.  Part of the calculation determines whether or not the 
    computed initial condition has enough downward momentum to be viable as 
    an initial condition (e.g., whether or not it will be overwhelmed by the
    upward drag of the inner plume).
    
    Parameters
    ----------
    yi : `stratified_plume_model.InnerPlume` object
        Object for manipulating the inner plume state space.
    yo : `stratified_plume_model.OuterPlume` object
        Object for manipulating the outer plume state space.
    particles : list of `Particle` objects
        List of `Particle` objects containing the dispersed phase local
        conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    neighbor : `scipy.interpolate.interp1d` object
        Container holding the latest solution for the inner plume state
        space.
    z_0 : float
        Top of the inner plume calculation (m).
    
    Returns
    -------
    z0 : float
        Initial depth of the outer plume segment (m).
    y0 : ndarray
        Initial dependent variables state space for the outer plume segment.
    flag : bool
        Outer plume viability flag:  `True` means the outer plume segment is
        viable and should be integrated; `False` means the outer plume 
        segment is too weak and should be discarded, moving down the inner 
        plume to calculate the next outer plume initial condition.
    
    Notes
    -----
    The iteration required to find a viable outer plume segment is conducted 
    by the `stratified_plume_model.outer_main` function.  This function 
    computes the initial conditions for one attempt to find an outer plume
    segment and reports back (through `flag`) on the success.
    
    There is one caveat to the above statement.  The model parameter 
    `p.nwidths` determines the vertical scale over which this function may
    integrate to find the start to an outer plume, given as a integer number
    of times of the inner plume half-width.  This function starts by searching
    one half-width.  If `p.nwidths` is greater than one, it will continue to
    expand the search region.  The physical interpretation of `p.nwidths` is
    to set a reasonable upper bound on the diameter of eddies shed from the
    inner plume in the peeling region into the outer plume.  While the 
    integral model does not have "eddies" per se, the search window size 
    should still be representative of this type of length scale.  
    
    """
    # Start the iteration counters
    iter = 0
    done = False
    
    # Compute the outer plume initial conditions until the outer plume is 
    # viable or until the maximum number of widths is integrated
    while not done and iter < p.nwidths:
        
        # Update iteration counter
        iter += 1
        
        # Get the inner plume properties at the top of this peeling region
        yi.update(z_0, neighbor(z_0), particles, profile, p)
        
        # Set the range to integrate to get the current peeling flux
        z_upper = z_0
        z_lower = z_0 + iter * yi.b
        
        # Check if the bottom of the reservoir is encountered.
        if z_lower > profile.z_max:
            z_lower = profile.z_max
        
        # Find the indices in the raw data for the inner plume solution close
        # to where z_upper and z_lower occur
        i_upper = np.min(np.where(neighbor.x >= z_upper)[0])
        i_lower = np.max(np.where(neighbor.x <= z_lower)[0])
        
        # Get the grid of elevations where we will integrate the solution to 
        # obtain the initial flux for the outer plume.  This is needed 
        # because the solution is so stiff:  if we integrated over a fixed
        # step size, we could easily miss dramatic changes in the solution.
        # Hence, we integrate over the steps in the numerical solution 
        # itself.
        n_grid = i_lower - i_upper + 3
        zi = np.zeros(n_grid)
        zi[0] = z_upper
        zi[-1] = z_lower
        zi[1:-1] = neighbor.x[i_upper:i_lower+1]
        
        # Integrate the peeling fluid over this grid to get the total 
        # contributions going into the outer plume
        Q = 0.
        tracer_vars = np.zeros(2 + yi.nchems)
        for i in range(len(zi)-1):
            yi.update(zi[i], neighbor(zi[i]), particles, profile, p)
            dz = zi[i+1] - zi[i]
            Q = Q + yi.Ep * dz
            tracer_vars = tracer_vars + np.hstack((yi.s, yi.T * p.rho_r * \
                          seawater.cp(), yi.c)) * yi.Ep * dz
        
        # Get the initial velocity of the peeling fluid using the modified 
        # outer plume Froude number condition
        T = tracer_vars[1] / (Q * p.rho_r * seawater.cp())
        s = tracer_vars[0] / Q
        c = tracer_vars[2:] / Q
        rho = seawater.density(T, s, yi.P)
        u = outer_fr(0.05, -Q, yi.b, yi.rho_a, rho, p.g, p.Fro_0)
        b = np.sqrt(Q**2 / (np.pi * (-Q) * u) + yi.b**2)
        dQdz = 2. * np.pi * yi.b * (p.alpha_1 * (yi.u + p.c1 * (-u)) + \
                p.alpha_2 * (-u)) + 2. * np.pi * b * p.alpha_3 * (-u) + yi.Ep
        
        # Check whether this outer plume segment will be viable
        if dQdz > 0:
            # This outer plume segment is not viable
            flag = False
            z0 = np.array([z_0, z_lower])
            y0 = np.array([np.zeros(yo.len), np.zeros(yo.len)])
        
        else:
            # This outer plume segmet is viable...stop integrating widths
            done = True
            
            # Check where the diffuser is
            if z_lower >= yi.z0:
                # This outer plume segment should not exist
                flag = False
                z0 = np.array([z_0, z_lower])
                y0 = np.array([np.zeros(yo.len), np.zeros(yo.len)])
            
            else:
                # This is the next outer plume segment to integrate
                flag = True
                z0 = z_lower
                y0 = []
                y0.append(Q)
                y0.append(Q * (-u))
                y0.append(s * Q)
                y0.append(p.rho_r * seawater.cp() * T * Q)
                y0.extend(c * Q)
    
    # Return the results of the initial conditions search
    return (z0, np.array(y0), flag)

def outer_fr(u_0, Q, bi, rho_a, rho, g, Fr_0):
    """
    Compute the outer plume initial width and velocity
    
    Computes the initial velocity of an outer plume segment using a Froude
    number condition analogous to Wueest et al. (1992) and calibrated and 
    reported in Socolofsky et al. (2008).
    
    Parameters
    ----------
    u_0 : float
        Initial guess for the outer plume velocity (m/s).
    Q : float
        Flow rate in the outer plume (m^3/s).
    bi : float
        Radius of the inner plume (m).
    rho_a : float
        Density of the ambient water (kg/m^3).
    rho : float
        Density of the continuous outer plume fluid (kg/m^3).
    g : float
        Acceleration of gravity (m/s^2).
    Fr_0 : float
        Equilibrium plume Froude number for the outer plume (--).
    
    Returns
    -------
    u0 : float
        The initial velocity of the outer plume (m/s).
    
    """
    # The Froude number condition is implicit; define the residual for use in
    # a root-finding algorithm
    def residual(u):
        """
        Computes the residual of a modified Froude number condition for 
        starting outer plume segments using the current guess for the initial
        velocity u.
        
        Parameters
        ----------
        u : float
            the current guess for the initial velocity (m/s).
        
        Notes
        -----
        All parameters of `outer_Fr` are global to this function since it is
        a subfunction of `outer_Fr`.
        """
        # Compute the outer plume width from Qo
        b = np.sqrt(Q**2 / (np.pi * Q * u) + bi**2)
        
        # Calculate the deviation from the desired Froude number
        return u - Fr_0 * np.sqrt(np.abs((b - bi) * g * (rho_a - rho) / 
               rho))
    
    return fsolve(residual, u_0)[0]

