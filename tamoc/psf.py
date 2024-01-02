"""
Particle Size Functions
=======================

This module provides empirical functions for computing particle sizes (bubble
and droplets) from jet releases into water.  Model equations are from::

    Johansen et al. (2013)
    Li et al. (2017)
    Wang et al. (2018)

Utilities are also included to generate log-normal or Rosin-Rammler size
distributions and to apply the d_95 rule (the idea that the d_95 value of a
volume-size distribution should not exceed the maximum stable particle size).

The functions in this module are used by the `particle_size_models` module
and are not intended to be called directly.  In the future, these functions
could, for example, be ported to Fortran or another language and wrapped 
in Python for use by the `particle_size_models` class objects.  

See Also
--------
particle_size_models.ModelBase, particle_size_models.PureJet, 
particle_size_models.Model

Notes
-----
Particle size distributions are computed from the following sources::

* Johansen, O., Brandvik, P. J., and Farooq, U. (2013) "Droplet breakup in 
  subsea oil releases - Part 2: Predictions of droplet size distributions 
  with and without injection of chemical dispersants." Mar Pollut Bull, 
  73(1), 327-335. This reference is intended for liquid oil breakup only.

* Li, Z., Spaulding, M., French McCay, D., Crowley, D., and Payne, J. R. 
  (2017) "Development of a unified oil droplet size distribution model with 
  application to surface breaking waves and subsea blowout releases 
  considering dispersant effects." Mar Pollut Bull, 114(1), 247-257.  The
  authors apply this reference for breakup of gas bubble or oil droplets.

* Wang, B., Socolofsky, S. A., Lai, C. C. K., Adams, E. E., and Boufadel, M. 
  C. (2018). "Behavior and dynamics of bubble breakup in gas pipeline leaks 
  and accidental subsea oil well blowouts." Mar Pollut Bull, 131, 72-86.  
  This reference is only intended for gas bubble breakup.

The maximum stable droplet size of an immiscible liquid in seawater is taken
from::

* Clift, R., Grace, J., and Weber, M. E. (1978) *Bubbles, Drops, and 
  Particles*, Dover Publications, Inc., Mineola, New York.

For gas bubbles in water, the maximum stable bubble size is taken from a 
method in Grace et al::

* Grace, J.R., Wairegi, T., Brophy, J., (1978) "Break-up of drops and bubbles 
  in stagnant media," Can. J. Chem. Eng. 56 (1), 3-8.

"""
# S. Socolofsky, December 2019, Texas A&M University <socolofs@tamu.edu>

from __future__ import (absolute_import, division, print_function)

from tamoc import dbm

import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize

# Use SI units throughout
G = 9.81

# General Utilities ----------------------------------------------------------

def mass2vol(m, rho):
    """
    Convert a mass or mass flux to a volume or volume flux
    
    Parameters
    ----------
    m : ndarray
        Array of masses (kg) or mass fluxes (kg/s) for each component of a 
        given fluid
    rho : float
        In-situ density of a given fluid (kg/m^3)
    
    Returns
    -------
    q : float
        Corresponding volume (m^3) or volume flux (m^3/s) of a given fluid
    
    """
    # Compute volume handling zero-fluxes correctly
    if np.sum(m) > 0:
        q = np.sum(m) / rho
    else:
        q = 0.
    
    return q


# Probabiity Density Functions -----------------------------------------------

def rosin_rammler(nbins, d50, k, alpha):
    """
    Return the volume size distribution from the Rosin-Rammler distribution
    
    Returns the fluid particle diameters in the selected number of bins on
    a volume-fraction basis from the Rosin Rammler distribution with 
    parameters d_50, k, and alpha.  
    
    Parameters
    ----------
    nbins : int
        Desired number of size bins in the particle volume size distribution
    d50 : float
        Volume mean particle diameter (m)
    k : float
        Scale parameter of the Rosin-Rammler distribution (=log(0.5) for d_50)
    alpha : float
        Shape parameter of the Rosin-Rammler distribution
    
    Returns
    -------
    de : ndarray
        Array of particle sizes at the center of each bin in the distribution
        (m)
    vf : ndarray
        Volume fraction in each bin (--)
    
    """
    # Get the de/d50 ratio for the edges of each bin in the distribution 
    # using a log-spacing
    a99 = (np.log(1. - 0.995) / k)**(1. / alpha)
    a01 = (np.log(1. - 0.01) / k)**(1. / alpha)
    bin_edges = np.logspace(np.log10(a01), np.log10(a99), nbins + 1)
    
    # Find the logarithmic average location of the center of each bin
    bin_centers = np.zeros(len(bin_edges) - 1)
    
    for i in range(len(bin_centers)):
        bin_centers[i] = np.exp(np.log(bin_edges[i]) + 
                         (np.log(bin_edges[i+1]) - np.log(bin_edges[i])) / 2.)
    
    # Compute the actual diameters of each particle
    de = d50 * bin_centers
    
    # Get the volume fraction within each bin
    if d50 == 0:
        vf = np.zeros(len(bin_centers))
    else:
        vn = 1. - np.exp(k * bin_edges**alpha)
        vf = np.zeros(len(bin_centers))
        for i in range(len(bin_edges) - 1):
            vf[i] = vn[i+1] - vn[i]
        vf = vf / np.sum(vf)
    
    # Return the particle size distribution
    return (de, vf)


def log_normal(nbins, d50, sigma):
    """
    Return the volume size distribution from the Log-normal distribution
    
    Returns the fluid particle diameters in the selected number of bins on
    a volume-fraction basis from the Log-normal distribution with parameters
    d50 and sigma_x.
    
    Parameters
    ----------
    nbins : int
        Desired number of size bins in the particle volume size distribution
    d50 : float
        Volume mean particle diameter (m)
    sigma : float
        Standard deviation of the Log-normal distribution in logarithmic 
        units.
    
    Returns
    -------
    de : ndarray
        Array of particle sizes at the center of each bin in the distribution
        (m)
    vf : ndarray
        Volume fraction in each bin (--)
    
    Notes
    -----
    This function uses the log-normal distribution defined at::
    
        https://en.wikipedia.org/wiki/Log-normal_distribution
    
    last accessed on 1/10/2020.  The relationship between mu_x and sigma_x 
    of the real x values and mu and sigma of the log(x) values is from the
    notes for ENGR 102 posted here:
        
        https://ceprofs.civil.tamu.edu/ssocolofsky/ENGR102/
        Downloads/19c/Week_12/week_12.pdf
    
    last accessed on 1/10/2020.
    
    """
    # Get the de/d50 ratio for the edges of each bin in the distribution 
    # using a log-spacing
    a0 = np.exp(np.log(d50) - 2.8 * sigma) / d50
    a1 = np.exp(np.log(d50) + 2.3 * sigma) / d50
    bin_edges = np.logspace(np.log10(a0), np.log10(a1), nbins + 1)
    
    # Find the logarithmic average location of the center of each bin
    bin_centers = np.zeros(len(bin_edges) - 1)
    
    for i in range(len(bin_centers)):
        bin_centers[i] = np.exp(np.log(bin_edges[i]) + 
                         (np.log(bin_edges[i+1]) - np.log(bin_edges[i])) / 2.)
    
    # Compute the actual diameters of each particle
    de = d50 * bin_edges[1:]
    
    # Compute log-normal parameters for de/d50 distribution
    mu = np.log(1.)
    
    # Get the volume fraction within each bin
    if d50 == 0:
        vf = np.zeros(len(bin_centers))
    else:
        vf = np.zeros(len(bin_centers))
        for i in range(len(bin_centers)):
            vf[i] = 1. / bin_centers[i] * 1. / (sigma * 
                np.sqrt(2. * np.pi)) * np.exp(-(np.log(bin_centers[i]) - 
                mu)**2 / (2. * sigma**2)) * (bin_edges[i+1] - bin_edges[i]) 
        vf = vf / np.sum(vf)
    
    # Return the particle size distribution
    return (de, vf)


def ln2rr(d50, sigma):
    """
    Convert the parameters of a log-normal distribution to Rosin-Rammler
    
    Parameters
    ----------
    d50 : float
        The median particle size of a volume distribution
    sigma : float
        Standard deviation of the Log-normal distribution in logarithmic 
        units.
    
    Returns
    -------
    d50 : float
        Volume mean particle diameter (m)
    k : float
        Scale parameter of the Rosin-Rammler distribution (=log(0.5) for d_50)
    alpha : float
        Shape parameter of the Rosin-Rammler distribution
    
    """
    # Compute d95 of the log-normal distribution
    mu = np.log(d50)
    mu_95 = mu + 1.6449 * sigma
    d95 = np.exp(mu_95)
    
    # Find parameters of Rosin-Rammler with same d50 and d95
    k = np.log(0.5)
    alpha = np.log(np.log(1. - 0.95) / k) / np.log(d95 / d50)
    
    return (d50, k, alpha)


def rr2ln(d50, k, alpha):
    """
    Convert the parameters of a Rosin-Rammler distribution to log-normal
    
    Parameters
    ----------
    d50 : float
        Volume mean particle diameter (m)
    k : float
        Scale parameter of the Rosin-Rammler distribution (=log(0.5) for d_50)
    alpha : float
        Shape parameter of the Rosin-Rammler distribution
    
    Returns
    -------
    d50 : float
        The median particle size of a volume distribution
    sigma : float
        Standard deviation of the Log-normal distribution in logarithmic 
        units.
    
    """
    # Compute d95 of the Rosin-Rammler distribution
    k = np.log(0.5)
    d95 = d50 * (np.log(1. - 0.95) / k)**(1. / alpha)
    
    # Find the parameters of log-normal with same d50 and d95
    sigma = (np.log(d95) - np.log(d50)) / 1.6449
    
    return (d50, sigma)


def rosin_rammler_fit(d50, d_max, alpha=1.8):
    """
    Return d_50, k, and alpha for the Rosin-Rammler distribution
    
    Parameters
    ----------
    d_50 : float
        Volume median diameter (m)
    d_max : float
        Maximum stable diameter (m)
    alpha : float, default=1.8
    
    Returns
    -------
    d_50 : float
        Volume median diameter (m)
    k : float
        Scale parameter for the Rosin-Rammler size distribution (--)
    alpha : float
        Shape parameter for the Rosin-Rammler size distribution (--)
    
    Notes
    -----
    This function follows the idea of Sintef to not let d_95 of the Rosin-
    Rammler distribution exceed the maximum stable particle size.  If the 
    original d_50 and d_max result in d_95 exceeding d_max, the d_50 is 
    shifted downward such that d_95 will equal d_max.  Otherwise, the original
    d_50 is preserved.
    
    Uses the Rosin-Rammler distribution equations at:
    
        https://en.wikipedia.org/wiki/Particle-size_distribution
    
    last access on 03/17/20.
    
    """
    # k parameter for d50
    k = np.log(0.5)
    
    # Adjust down if d95 exceeds the de_max
    if isinstance(d_max, type(None)):
        d50 = d50
    
    else:
        # Compute d95 for the given d50
        d95 = d50 * (np.log(1. - 0.95) / k)**(1. / alpha)
        
        # Adjust d50 so that d95 does not exceed d_max
        if d95 > d_max:
            print('\nPredicted size distribution exceeds d50...')
            print('    --> Adjusting size distribution down.')
            print('        Original d50 = %g (mm)' % (d50 * 1000.))
            print('        d_max = %g (mm)' % (d_max * 1000.))
            d95 = d_max
            k95 = np.log(0.05)
            d50 = d95 * (np.log(1. - 0.5) / k95)**(1. / alpha)
            print('        New d50 = %g (mm)\n' % (d50 * 1000.))
    
    # Return the final distribution fit
    return (d50, k, alpha)


def log_normal_fit(d50, d_max, sigma=0.27):
    """
    Return d_50 and sigma for the log-normal distribution
    
    Parameters
    ----------
    d_50 : float
        Volume median diameter (m)
    d_max : float
        Maximum stable diameter (m)
    sigma : float, default=0.27
        Standard deviation of the Log-normal distribution in logarithmic 
        units.
    
    Returns
    -------
    d_50 : float
        Volume median diameter (m)
    sigma : float
        Standard deviation of the Log-normal distribution in logarithmic 
        units.
    
    Notes
    -----
    This function follows the idea of Sintef to not let d_95 of the particle
    size distribution exceed the maximum stable particle size.  If the 
    original d_50 and d_max result in d_95 exceeding d_max, the d_50 is 
    shifted downward such that d_95 will equal d_max.  Otherwise, the original
    d_50 is preserved.
    
    """
    # Adjust down if d50 exceeds the de_max
    if d_max == None:
        # Do not adjust the fit
        d50 = d50
        
    else:
        # Comnpute d95 for the given d50 and sigma
        mu = np.log(d50)
        mu_95 = mu + 1.6449 * sigma
        d95 = np.exp(mu_95)
        
        # Adjust d_50 so that d_95 does not exceed d_max
        if d95 > d_max:
            print('\nPredicted size distribution exceeds d50...')
            print('    --> Adjusting size distribution down.')
            print('        Original d50 = %g (mm)' % (d50 * 1000.))
            print('        d_max = %g (mm)' % (d_max * 1000.))
            d50 = np.exp(np.log(d_max) - 1.6449 * sigma)
            print('        New d50 = %g (mm)\n' % (d50 * 1000.))
    
    # Return the final distribution fit
    return (d50, sigma)


# Functions for computing maximum stable particle size -----------------------

def de_max_oil(rho_p, sigma, rho):
    """
    Calculate the maximum stable oil droplet size
    
    Predicts the maximum stable liquid particle size per Clift et al. (1978) 
    via the equation:
    
    d_max = 4. * np.sqrt(sigma / (g (rho - rho_p)))
    
    Parameters
    ----------
    rho_p : float
        Density of the phase undergoing breakup (kg/m^3)
    sigma : float
        Interfacial tension between the phase undergoing breakup and the 
        ambient receiving continuous phase (N/m)
    rho : float
        Density of the ambient receiving continuous phase (kg/m^3) 
    
    Returns
    -------
    d_max : float
        Maximum stable particle size (m)
    
    """
    return 4. * np.sqrt(sigma / (G * (rho - rho_p)))


def grow_rate(n, k, nu_c, nu_d, sigma, g, dp, rho_c, rho_d, K):
    """
    Compute the instability growth rate on a gas bubble
    
    Write instability growth rate equation in Grace et al. as a root
    problem for n = f(k)
    
    Returns
    -------
    res : float
        The residual of the growth-rate equation expressed as a root-finding
        problem.
    
    Notes
    -----
    This function is used by the `grace()` function for maximum stable 
    particle size.  It should not be called directly.
    
    """
    # Compute the kinematic viscosity from the dynamic viscosity
    mu_c = nu_c * rho_c

    # Compute more derived variables
    if n < 0.:
        m_c = k
        m_d = k
    else:
        m_c = np.sqrt(k**2 + n / nu_c)
        m_d = np.sqrt(k**2 + n / nu_d)
    
    # Compute the residual of the root function
    res = (sigma * k**3 - g * k * dp + n**2 * (rho_c + rho_d)) * \
          (k + m_c + K * (k + m_d)) + 4 * n * k * mu_c * (k + K * m_d) * \
          (K * k + m_c)
    
    # Return the residual
    return res

def grow_time(lam, de, U, nu_c, nu_d, sigma, g, dp, rho_c, rho_d, K, c_0):
    """
    Compare the available and needed disturbance growth times for instability
    
    Compares the time available for a disturbance to grow to the time needed
    for that disturbance to break a bubble.  
    
    Returns
    -------
    t_cr : float
        The critical time (s) for which the required grow time equals the
        available time
    
    Notes
    -----
    This function is used by the `grace()` function for maximum stable 
    particle size.  It should not be called directly.
    
    """
    # Compute the derived variables
    k = 2. * np.pi / lam
    
    # Consider disturbances with a node at the nose
    theta_1 = lam / (2. * de)
    
    # Compute the available time for disturbance growth
    t_a = de / 2 / U * (1. + 3. / 2. * K) * np.log(1. / np.tan(
          theta_1 / 2.))
    
    # Compute the grwoth rate of this disturbance
    n0 = 1. / t_a  # Value at the critical point
    n = fsolve(grow_rate, 5. * n0, args=(k, nu_c, nu_d, sigma, g, dp, rho_c, 
                                 rho_d, K)
              )[0]
    
    # Relate n to t_e
    t_e = 1. / n
    
    # Return the critical growth time
    return c_0 * t_e - t_a

def find_de(de, rho_d, rho_c, mu_d, mu_c, sigma, nu_d, nu_c, g, dp, K, 
            lam_crit, c_0):
    """
    Search for the critical stable bubble size
    
    Search for the maximum stable bubble size of a gas bubble in water using
    the method in Grace et al.
    
    Returns
    -------
    t_min : float
        The minimum time required for a disturbance of the given size to 
        break the fluid particle.  
    
    Notes
    -----
    This function is used by the `grace()` function for maximum stable 
    particle size.  It should not be called directly.
    
    """
    # Select the best available equations of state module
    try:
        from tamoc import dbm_f
    except ImportError:
        from tamoc import dbm_p as dbm_f

    # Time available for growth, t_a
    # The travel time from the position where disturance starts to the equator
    # Compute the rise velocity of this bubble size
    shape = dbm_f.particle_shape(de, rho_d, rho_c, mu_c, sigma)
    if shape == 1:
        U = dbm_f.us_sphere(de, rho_d, rho_c, mu_c)
    elif shape == 2:
        U = dbm_f.us_ellipsoid(de, rho_d, rho_c, mu_d, mu_c, sigma, -1)
    else:
        U = dbm_f.us_spherical_cap(de, rho_d, rho_c)
    
    # lam_max is upper limit on leading interface disturbance size
    lam_max = np.pi * de / 2.
    
    # Find the wave length that corresponds to the maximum growth rate
    delta = 2. * np.finfo(np.float64()).eps
    lam = minimize(grow_time, lam_crit, args=(de, U, nu_c, nu_d,
        sigma, g, dp, rho_c, rho_d, K, c_0),
        bounds=[((1. + delta) * lam_crit, lam_max)]
    ).x[0]
    
    t_min = grow_time(lam, de, U, nu_c, nu_d,sigma, g, dp, rho_c, rho_d, K, 
                      c_0)
    
    # Return the growth time
    return t_min


def grace(rho_c, rho_d, mu_c, mu_d, sigma, fp_type=0):
    """
    Implement the Grace et al. algorithm for maximum stable particle size
    
    Computes the maximum stable particle size of an immiscible particle 
    rising in stagnant water following a method in Grace et al. 
    
    Parameters
    ----------
    rho_c : float
        Density of the continuous-phase ambient fluid (kg/m^3)
    rho_d : float
        Density of the immiscible fluid particle subject to breakup (kg/m^3)
    mu_c : float
        Dynamic viscosity of the continuous-phase ambient fluid (Pa s)
    mu_d : float
        Dynamic viscosity of the immiscible fluid particle subject to breakup 
        (Pa s)
    sigma : float
        Interfacial tension between the continuous phase ambient fluid and
        the immiscible fluid particle subject to breakup (N/m)
    fp_type : int, default=0
        Phase of the immiscible fluid particle; 0 = gas, 1 = liquid.
    
    Returns
    -------
    de_max : float
        Equivalent spherical diameter of the maximum stable fluid particle
        subject to breakup in stagnant water (m)
    
    See Also
    --------
    grow_rate, grow_time, find_de
    
    Notes
    -----
    Implements the method in * Grace, J.R., Wairegi, T., Brophy, J., (1978)
    "Break-up of drops and bubbles in stagnant media," Can. J. Chem. Eng. 56
    (1), 3-8.
    
    """
    # Set the fit parameter
    if fp_type == 0:
        # This is gas
        c_0 = 3.8
    else:
        # This is liquid
        c_0 = 1.4
    
    # Compute the derived properties
    dp = np.abs(rho_c - rho_d)
    K = mu_d / mu_c
    nu_c = mu_c / rho_c
    nu_d = mu_d / rho_d
    
    # Region of instability.
    # lam_crit is lower limit on unstable wavelengths
    lam_crit = 2. * np.pi * np.sqrt(sigma / (G * dp))
    
    # Lower limit on maximum stable diameter is lam_crit = lam_max
    de_max_star = 2. / np.pi * lam_crit
    
    # Choose Initialize the search near this minimum
    de = 1.01 * de_max_star
    
    # Find the maximum stable bubble size
    de_max = fsolve(find_de, de, args=(rho_d, rho_c, mu_d, mu_c, sigma, nu_d, 
                                       nu_c, G, dp, K, lam_crit, c_0)
                   )[0]
    
    # Return the result
    return de_max


# SINTEF Model Equations -----------------------------------------------------

def sintef(d0, m_gas, rho_gas, m_oil, rho_oil, mu_p, sigma, rho, mu, 
           fp_type=1, use_d95=True):
    """
    Compute characteristic values for jet breakup
    
    Computes the characteristic particle sizes for jet breakup using the
    equations in Johansen et al. (2013) (sintef model equations) and using
    the model coefficients updated in technical reports to API.
    
    Parameters
    ----------
    d0 : float
        Equivalent circular diameter of the release (m)
    m_gas : np.array
        Mass fluxes of each pseudo-component of the gas-phase fluid at the
        release (kg/s)
    rho_gas : float
        Density of the gas-phase petroleum fluid at the release (kg/m^3)
    m_oil : np.array
        Mass fluxes of each pseudo-component of the liquid-phase petroleum
        fluid at the release (kg/s)
    rho_oil : float
        Density of the liquid-phase petroleum fluid at the release (kg/m^3)
    mu_p : float
        Dynamic viscosity of the fluid phase of interest at the release 
        (Pa s)
    sigma : float
        Interfacial tension between the fluid phase of interest and water at
        the release (N/m)
    rho : float
        Density of seawater at the release (kg/m^3)
    mu : float
        Dynamic viscosity of seawater at the release (Pa s)
    fp_type : int, default=1
        Fluid phase to compute breakup; 0 = gas, 1 = liquid.  The SINTEF
        equation authors do not recommend using this method for gas; hence,
        fp_type should normally equal 1.
    use_d95 : bool, default=True
        Flag indicating whether or not to implement the d_95 rule (see
        module documentation above); `True` means to use the rule.
    
    Returns
    -------
    d50 : float
        Volume median diameter of the fluid phase of interest (m)
    de_max : float
        Maximum stable particle size of the fluid phase of interest (m)
    k : float
        Scale parameter for the Rosin-Rammler size distribution (--)
    alpha : float
        Shape parameter for the Rosin-Rammler size distribution (--)
    
    """
    # Convert mass-flux to volume flux
    if np.sum(m_gas) > 0:
        q_gas = mass2vol(m_gas, rho_gas)
    else:
        q_gas = 0.
    if np.sum(m_oil) > 0:
        q_oil = mass2vol(m_oil, rho_oil)
    else:
        q_oil = 0.
    
    # Get the void-fraction adjusted characteristic exit velocity
    n = q_gas / (q_gas + q_oil)
    if q_oil == 0.:
        # This is gas only
        Un = 4. * q_gas / (np.pi * d0**2)
        rho_m = rho_gas
    elif q_gas == 0:
        # This is oil only
        Un = 4. * q_oil / (np.pi * d0**2)
        rho_m = rho_oil
    else:
        # This is oil and gas
        Un = 4. * q_oil / (np.pi * d0**2) / (1. - n)**(1./2.)
        rho_m = rho_oil * (1. - n) + rho_gas * n
    
    Fr = Un / (G * (rho - rho_m) / rho * d0)**(1./2.)
    Uc = Un * (1. + 1./Fr)
    
    # Compute the particle size distribution parameters
    if fp_type == 0:
        d50, de_max, k, alpha = sintef_model(Uc, d0, q_gas, rho_gas, mu_p, 
                                             sigma, rho, mu, is_gas=True, 
                                             use_d95=use_d95)
    else:
        d50, de_max, k, alpha = sintef_model(Uc, d0, q_oil, rho_oil, mu_p, 
                                             sigma, rho, mu, is_gas=False, 
                                             use_d95=use_d95)
    
    return (d50, de_max, k, alpha)


def sintef_model(Uc, d0, q, rho_p, mu_p, sigma, rho, mu, is_gas=False, 
                 use_d95=True):
    """
    Computes the particle size for the Sintef equation
    
    Evaluates the parameters of the particle size distribution for the 
    SINTEF equation and implements the d_95 rule as appropriate.  This
    function returns the parameters of the Rosin-Rammler size distribution
    with the spreading rates as reported in Johansen et al.
    
    Returns
    -------
    d50 : float
        Volume median diameter of the fluid phase of interest (m)
    de_max : float
        Maximum stable particle size of the fluid phase of interest (m)
    k : float
        Scale parameter for the Rosin-Rammler size distribution (--)
    alpha : float
        Shape parameter for the Rosin-Rammler size distribution (--)
    
    Notes
    -----
    This function is called by the `sintef()` function after several 
    intermediate parameters are computed.  This function should not be 
    called directly.
    
    """
    if q > 0.:
        
        # Compute d_50 from the We model
        d50 = sintef_d50(Uc, d0, rho_p, mu_p, sigma, rho)
        
        # Get an estimate of de_max
        if is_gas:
            de_max = grace(rho, rho_p, mu, mu_p, sigma, fp_type=0)
        else:
            de_max = de_max_oil(rho_p, sigma, rho)
        
        # Get the adjusted particle size distribution
        d50_from95, k, alpha = rosin_rammler_fit(d50, de_max)
        
        # Return the desired value for d50
        if use_d95:
            # Use the d_95 rule
            d50 = d50_from95
        elif d50 > de_max:
            # Truncate the distribution
            d50 = de_max
    
    else:
        
        # Return an empty set of particles
        de_max = None
        d50, k, alpha = rosin_rammler_fit(0., de_max)
    
    return (d50, de_max, k, alpha)


def sintef_d50(u0, d0, rho_p, mu_p, sigma, rho):
    """
    Compute d_50 from the SINTEF equations
    
    Returns
    -------
    d50 : float
        Volume median diameter of the fluid phase of interest (m)
    
    Notes
    -----
    This function is called by the `sintef()` function after several 
    intermediate parameters are computed.  This function should not be 
    called directly.
    
    """
    # Compute the non-dimensional constants
    We = rho_p * u0**2 * d0 / sigma
    Vi = mu_p * u0 / sigma
    
    if We > 350.:
        # Atomization...use the the We model
        A = 24.8
        B = 0.08
        
        # Solve for the volume mean diameter from the implicit equation
        def residual(dp):
            """
            Compute the residual of the SINTEF modified Weber number model
            
            Evaluate the residual of the non-dimensional diameter 
            dp = de_50 / D for the SINTEF droplet break-up model.
            
            Input variables are:
                We, Vi, A, B = constant and global from above
                dp = Non-dimensional diameter de_50 / D (--)
            
            """
            # Compute the non-dimensional diameter and return the residual
            return dp - A * (We / (1. + B * Vi * dp**(1./3.)))**(-3./5.)
        
        # Find the gas and liquid fraction for the mixture
        dp = fsolve(residual, 5.)[0]
        
        # Compute the final d_50
        d50 = dp * d0
    
    else:
        # Sinuous wave breakup...use the pipe diameter
        d50 = 1.2 * d0
    
    # Return the result
    return d50


# Li et al. Equations --------------------------------------------------------

def li_etal(d0, m_gas, rho_gas, m_oil, rho_oil, mu_p, sigma, rho, mu, 
            fp_type=1):
    """
    Compute characteristic values for jet breakup
    
    Computes the characteristic particle sizes for jet breakup using the
    equations in Li et al. (2017) (li_etal model equations).  The authors
    provide different equation fit parameters for gas or liquid breakup, 
    and this function selects the correct parameters for the fluid phase 
    of interest.
    
    Parameters
    ----------
    d0 : float
        Equivalent circular diameter of the release (m)
    m_gas : np.array
        Mass fluxes of each pseudo-component of the gas-phase fluid at the
        release (kg/s)
    rho_gas : float
        Density of the gas-phase petroleum fluid at the release (kg/m^3)
    m_oil : np.array
        Mass fluxes of each pseudo-component of the liquid-phase petroleum
        fluid at the release (kg/s)
    rho_oil : float
        Density of the liquid-phase petroleum fluid at the release (kg/m^3)
    mu_p : float
        Dynamic viscosity of the fluid phase of interest at the release 
        (Pa s)
    sigma : float
        Interfacial tension between the fluid phase of interest and water at
        the release (N/m)
    rho : float
        Density of seawater at the release (kg/m^3)
    mu : float
        Dynamic viscosity of seawater at the release (Pa s)
    fp_type : int, default=1
        Fluid phase to compute breakup; 0 = gas, 1 = liquid.  The SINTEF
        equation authors do not recommend using this method for gas; hence,
        fp_type should normally equal 1.
    
    Returns
    -------
    d50 : float
        Volume median diameter of the fluid phase of interest (m)
    de_max : float
        Maximum stable particle size of the fluid phase of interest (m)
    k : float
        Scale parameter for the Rosin-Rammler size distribution (--)
    alpha : float
        Shape parameter for the Rosin-Rammler size distribution (--)
    
    """
    # Convert mass-flux to volume flux
    q_gas = mass2vol(m_gas, rho_gas)
    q_oil = mass2vol(m_oil, rho_oil)
    
    # Get the void-fraction adjusted characteristic exit velocity
    n = q_gas / (q_gas + q_oil)
    if fp_type == 0:
        Uc = 4. * q_gas / (np.pi * d0**2) / n
    elif fp_type == 1:
        Uc = 4. * q_oil / (np.pi * d0**2) / (1. - n)
    
    # Compute the particle size distribution for gas and oil
    if fp_type == 0:
        d50, de_max, k, alpha = li_etal_model(Uc, d0, q_gas, rho_gas, mu_p, 
                                              sigma, rho, mu, is_gas=True)
    else:
        d50, de_max, k, alpha = li_etal_model(Uc, d0, q_oil, rho_oil, mu_p, 
                                              sigma, rho, mu, is_gas=False)
    
    return (d50, de_max, k, alpha)


def li_etal_model(Uc, d0, q, rho_p, mu_p, sigma, rho, mu, is_gas=False):
    """
    Computes the particle size for the Li et al. equation
    
    Evaluates the parameters of the particle size distribution for the Li et
    al. equation and implements the d_95 rule. This function returns the
    parameters of the Rosin-Rammler size distribution with the spreading
    rates as reported in Li et al.
    
    Returns
    -------
    d50 : float
        Volume median diameter of the fluid phase of interest (m)
    de_max : float
        Maximum stable particle size of the fluid phase of interest (m)
    k : float
        Scale parameter for the Rosin-Rammler size distribution (--)
    alpha : float
        Shape parameter for the Rosin-Rammler size distribution (--)
    
    Notes
    -----
    This function is called by the `li_etal()` function after several 
    intermediate parameters are computed.  This function should not be 
    called directly.
    
    """
    if q > 0.:
        
        # Compute d_50 from the We model
        d50 = li_etal_d50(Uc, d0, rho_p, mu_p, sigma, rho, is_gas)
        
        # Get an estimate of de_max
        if is_gas:
            de_max = grace(rho, rho_p, mu, mu_p, sigma, fp_type=0)
        else:
            de_max = de_max_oil(rho_p, sigma, rho)
        
        # Get the adjusted particle size distribution
        d50, k, alpha = rosin_rammler_fit(d50, None)
    
    else:
        
        # Return an empty set of particles
        de_max = None
        d50, k, alpha = rosin_rammler_fit(0., de_max)
    
    return (d50, de_max, k, alpha)


def li_etal_d50(Uc, d0, rho_p, mu_p, sigma, rho, is_gas=True):
    """
    Compute d_50 from the Li et al. equations
    
    Returns
    -------
    d50 : float
        Volume median diameter of the fluid phase of interest (m)
    
    Notes
    -----
    This function is called by the `li_etal()` function after several 
    intermediate parameters are computed.  This function should not be 
    called directly.
    
    """
    # Get the constants of the fit for a jet release
    p = 0.460
    q = -0.518
    if is_gas:
        r = 2.988
    else:
        r = 14.05
    
    # The Li et al. (2017) model wrongly uses the oil equation for the 
    # maximum stable particle size of gas
    de_max = de_max_oil(sigma, rho_p, rho)
    if de_max < d0:
        dc = de_max
    else:
        dc = d0
    
    # Compute dimensionless groups
    We = rho * Uc**2 * dc / sigma
    Oh = mu_p / np.sqrt(rho_p * sigma * dc)
    
    # Compute the non-dimensional d50
    ds = r * (1 + 10 * Oh)**p * We**q
    d50 = ds * dc
    
    return d50


# Wang et al. Equations ------------------------------------------------------

def wang_etal(d0, m_g, rho_g, mu_g, sigma_g, rho, mu, 
              m_l=0., rho_l=None, P=4.e6, T=288.15):
    """
    Compute characteristic values for gas jet breakup
    
    Computes the characteristic gas bubble sizes for jet breakup using the
    equations in Wang et al. (2018) (wang_etal model equations).  
    
    Parameters
    ----------
    d0 : float
        Equivalent circular diameter of the release (m)
    m_g : np.array
        Mass fluxes of each pseudo-component of the gas-phase fluid at the
        release (kg/s)
    rho_g : float
        Density of the gas-phase  fluid at the release (kg/m^3)
    mu_g : float
        Dynamic viscosity of the gas-phase fluid at the release (Pa s)
    sigma_g : float
        Interfacial tension between the gas-phase fluid and water at the
        release (N/m)
    rho : float
        Density of seawater at the release (kg/m^3)
    mu : float
        Dynamic viscosity of seawater at the release (Pa s)
    m_l : np.array
        Mass fluxes of each pseudo-component of the liquid-phase fluid at the 
        release (kg/s)
    rho_l : float
        Density of the liquid-phase fluid at the release (kg/m^3)
    P : float, default=4.e6
        Pressure in the receiving fluid (Pa); used to compute the speed of
        sound in the released gas.
    T : float, default=288.15
        Temperature of the gas phase at the release (K); used to compute the
        speed of sound in the released gas.
    
    Returns
    -------
    d50_gas : float
        Volume median diameter of the gas bubbles (m)
    m_gas : float
        Mass fluxes of each pseudo-component of the gas-phase fluid at the
        release (kg/s).  This may be different from the input value in the 
        case of choked flow at the orifice.
    m_oil : float
        Mass fluxes of each pseudo-component of the liquid-phase fluid at the
        release (kg/s).  This may be different from the input value in the 
        case of choked flow at the orifice.
    de_max : float
        Maximum stable particle size of the fluid phase of interest (m)
    sigma : float
        Standard deviation of the Log-normal distribution in logarithmic 
        units.
    
    """
    # Convert mass-flux to volume flux
    Qg = mass2vol(m_g, rho_g)
    if np.sum(m_l) == 0.:
        Ql = 0.
    else:
        Ql = mass2vol(m_l, rho_l)
    
    # Compute the exit velocity assuming no choked flow and single exit
    # velocity
    n = Qg / (Qg + Ql)
    A = np.pi * d0**2 / 4.
    Ug = (Qg + Ql) / A
    
    # Check for choked flow using methane for speed of sound
    ch4 = dbm.FluidMixture(['methane'])
    delta_rho = ch4.density(np.array([1.]), T, P)[0,0] - \
                ch4.density(np.array([1.]), T, 1.01 * P)[0,0]
    a = np.sqrt((P - 1.01 * P) / delta_rho)
    if 10. * Ug < a:
        U_E = Ug
    else:
        # Compute the cp / cv ratio
        cp_ch4 = 35.69  # J/mol/K;  CO2 = 37.13
        cv_ch4 = cp_ch4 - 8.31451  # From Poling et al. for ideal gases
        kappa = cp_ch4 / cv_ch4    # Assume approximately ok for petroleum
        
        # Get the Mach number
        Ma = Ug / a
        
        # Correct the exit velocity for choked flow
        if Ma < np.sqrt((kappa + 1.) / 2.):
            U_E = a * (-1. + np.sqrt(1. + 2. * (kappa - 1.) * Ma**2.)) / \
                  ((kappa - 1.) * Ma)
        else:
            U_E = a * np.sqrt(2. / (kappa + 1.))
    
    # Update the gas and oil exit velocities
    if Qg > 0:
        Ug = U_E
    else:
        Ug = 0
    if Ql > 0:
        Ul = U_E
    else:
        Ul = 0
    
    # Compute the particle size distribution for gas
    d50_gas, m_gas, m_oil, de_max, sigma = wang_etal_model(A, n, Ug, rho_g, 
                                                           mu_g, sigma_g, Ul, 
                                                           rho_l, rho, mu)
    
    return (d50_gas, m_gas, m_oil, de_max, sigma)


def wang_etal_model(A, n, Ug, rho_g, mu_g, sigma_g, Ul, rho_l, rho, mu):
    """
    Computes the particle size for the Wang et al. equation
    
    Evaluates the parameters of the gas bubble size distribution for the Wang
    et al. equation and implements the d_95 rule. This function returns the
    parameters of the log-normal size distribution with the spreading
    rates as reported in Wang et al.
    
    Returns
    -------
    d50_gas : float
        Volume median diameter of the gas bubbles (m)
    m_gas : float
        Mass fluxes of each pseudo-component of the gas-phase fluid at the
        release (kg/s).  This may be different from the input value in the 
        case of choked flow at the orifice.
    m_oil : float
        Mass fluxes of each pseudo-component of the liquid-phase fluid at the
        release (kg/s).  This may be different from the input value in the 
        case of choked flow at the orifice.
    de_max : float
        Maximum stable particle size of the fluid phase of interest (m)
    sigma_ln : float
        Standard deviation of the Log-normal distribution in logarithmic 
        units.
    
    Notes
    -----
    This function is called by the `wang_etal()` function after several 
    intermediate parameters are computed.  This function should not be 
    called directly.
    
    """
    if Ug > 0:
        
        # Compute d50 from the model
        (d, m_gas, m_oil) = wang_etal_d50(A, n, Ug, rho_g, mu_g, sigma_g, Ul, 
                                      rho_l, rho, mu)
        
        # Compute the maximum stable bubble size
        de_max = grace(rho, rho_g, mu, mu_g, sigma_g, fp_type=0)
        
        # Get the adjusted particle size distribution
        d50_gas, sigma_ln = log_normal_fit(d, de_max, sigma=0.27)
        
    else:
        
        # Return an empty set of particles
        m_gas = 0.
        m_oil = rho_l * A * Ul
        de_max = None
        d50_gas, sigma_ln = log_normal_fit(0., de_max)
    
    return (d50_gas, m_gas, m_oil, de_max, sigma_ln)


def wang_etal_d50(A, n, Ug, rho_g, mu_g, sigma_g, Ul, rho_l, rho, mu):
    """
    Compute d_50 from the Wang et al. equations
    
    Returns
    -------
    d50 : float
        Volume median diameter of the gas bubbles (m)
    
    Notes
    -----
    This function is called by the `wang_etal()` function after several 
    intermediate parameters are computed.  This function should not be 
    called directly.
    
    """
    # Compute the total dynamic momentum and buoyancy fluxes
    Ag = A * n
    Al = A * (1. - n)
    mg = rho_g * Ag * Ug**2
    bg = (rho - rho_g) * G * Ag * Ug
    if n == 1:
        ml = 0.
        bl = 0.
    else:
        ml = rho_l * Al * Ul**2
        bl = (rho - rho_l) * G * Al * Ul
    mo = mg + ml
    bo = bg + bl
    
    # The kinematic momentum and buoyancy fluxes are
    M = mo / rho
    B = bo / rho
    
    # Jet-to-plume transition length scale
    l_M = M**(3./4.) / B**(1./2.)
    
    # Characteristic velocity scale
    Ua = np.sqrt(mo / (rho * A))
    
    # Compute the mixture density
    if n == 1:
        rho_l = 0.
    rho_m = n * rho_g + (1. - n) * rho_l
    
    # Get the modified Weber number
    We_m = rho_m * Ua**2 * l_M / sigma_g
    
    # Compute the characteristic droplet size
    d = 4.3 * We_m**(-3./5.) * l_M
        
    # Compute the actual gas and oil flow rate
    m_g = rho_g * Ag * Ug
    m_l = rho_l * Al * Ul
    
    # Return the characteristic size
    return (d, m_g, m_l)
