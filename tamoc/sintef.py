"""
Sintef
======

Evaluate initial bubble and droplet sizes from the SINTEF model

This module is deprecated and has been replaced by the `particle_size_models`
and `psf` modules.  In order to retain compatibility with previous versions, 
we retain the original API of the `sintef` model, but replace all calculations
with calls to the appropriate functions in `particle_size_models` and `psf`.

"""
# S. Socolofsky, September 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import psf

import numpy as np
from copy import copy
from scipy.optimize import fsolve

def modified_We_model(D, rho_gas, m_gas, mu_gas, sigma_gas, rho_oil, m_oil, 
                      mu_oil, sigma_oil, rho):
    """
    This function is deprecated:  Use psf.sintef() instead.
    
    Compute the initial oil droplet and gas bubble sizes from the SINTEF model
    
    Apply the SINTEF modified Weber number model to estimate the initial 
    oil and gas particle sizes.  This function calculates the adjusted exit
    velocity based on a void fraction and buoyancy adjustment per the method
    suggested by SINTEF.
    
    Parameters
    ----------
    D : float
        Diameter of the release point (m)
    rho_gas : float
        In-situ density of the gas (kg/m^3)
    m_gas : ndarray
        Array of mass fluxes for each component of the gas object (kg/s)
    mu_gas : float
        Dynamic viscosity of gas (Pa s)
    sigma_gas : float
        Interfacial tension between gas and seawater (N/m)
    rho_oil : float
        In-situ density of the oil
    m_oil : ndarray
        Array of mass fluxes for each component of the oil object (kg/s)
    mu_oil : float
        Dynamic viscosity of oil (Pa s)
    sigma_oil : float
        Interfacial tension between oil and seawater (N/m)
    rho : float
        Density of the continuous phase fluid (kg/m^3)
    
    Returns
    -------
    A tuple containing:
        de_gas : float
            The volume mean diameter of the gas bubbles (m)
        de_oil : float
            The volume mean diameter of the oil droplets (m)
    """
    # Make sure the masses are in arrays
    if not isinstance(m_gas, np.ndarray):
        if not isinstance(m_gas, list):
            m_gas = np.array([m_gas])
        else:
            m_gas = np.array(m_gas)
    if not isinstance(m_oil, np.ndarray):
        if not isinstance(m_oil, list):
            m_oil = np.array([m_oil])
        else:
            m_oil = np.array(m_oil)
    
    # Call the psf functions
    mu = 0.0012   # pass mu of seawater (not used)
    de_gas, de_max, k, alpha = psf.sintef(D, m_gas, rho_gas, m_oil, rho_oil, 
                                          mu_gas, sigma_gas, rho, mu, 
                                          fp_type=0, use_d95=False)
    
    de_oil, de_max, k, alpha = psf.sintef(D, m_gas, rho_gas, m_oil, rho_oil, 
                                          mu_oil, sigma_oil, rho, mu, 
                                          fp_type=1, use_d95=False)
    
    # Return the bubble and droplet sizes
    return (de_gas, de_oil)


# Provide tool to estimate the maximum stable particle size
def de_max(sigma, rho_p, rho):
    """
    This function is deprecated:  Use psf.de_max_oil() instead.
    
    Calculate the maximum stable particle size
    
    Predicts the maximum stable particle size per Clift et al. (1978) via 
    the equation:
    
    d_max = 4. * np.sqrt(sigma / (g (rho - rho_p)))
    
    Parameters
    ----------
    sigma : float
        Interfacial tension between the phase undergoing breakup and the 
        ambient receiving continuous phase (N/m)
    rho_p : float
        Density of the phase undergoing breakup (kg/m^3)
    rho : float
        Density of the ambient receiving continuous phase (kg/m^3) 
    
    Returns
    -------
    d_max : float
        Maximum stable particle size (m)
    
    """
    return psf.de_max_oil(rho_p, sigma, rho)

def de_50(U, D, rho_p, mu_p, sigma, rho):
    """
    This function is deprecated:  Use psf.sintef_d50() instead.
    
    Predict the volume mean diameter from a modified Weber number model
    
    Calculates the SINTEF modified Weber number model for the volume mean 
    diameter of a blowout release.
    
    Parameters
    ----------
    U : float
        Effective exit velocity after void fraction and buoyancy correction 
        of the phase undergoing breakup (m/s)
    D : float
        Diameter of the discharge (m)
    rho_p : float
        Density of the phase undergoing breakup (kg/m^3)
    mu_p : float
        Dynamic viscosity of the phase undergoing breakup (Pa s)
    sigma : float
        Interfacial tension between the phase undergoing breakup and the 
        ambient receiving continuous phase (N/m)
    
    Returns
    -------
    de_50 : float
        The volume mean diameter of the phase undergoing breakup
    
    Notes
    -----
    This function first checks the We number.  If the We is less than the 
    critical value of 350 required for atomization, then the fluid particle 
    diameter is estimated as 1.2 D.  Otherwise, the SINTEF modified We number 
    model is used.  In both cases, the resulting particle diameter is compared
    to the maximum stable particle size per Clif et al. (1978) of 
        
        d_max = 4 (sigma/ (g (rho - rho_p)))^(1/2).  
        
    The function returns the lesser of the estimated particle size or the 
    maximum stable particle size.
    
    """
    # Call the psf function
    de = psf.sintef_d50(U, D, rho_p, mu_p, sigma, rho)
    
    # Require the diameter to be less than the maximum stable size
    dmax = de_max(sigma, rho_p, rho)
    if de > dmax:
        de = dmax
    
    # Return the result
    return de

def rosin_rammler(nbins, d50, md_total, sigma, rho_p, rho):
    """
    This function is deprecated:  Use psf.rosin_rammler() instead.
    
    Return the volume size distribution from the Rosin Rammler distribution
    
    Returns the fluid particle diameters in the selected number of bins on
    a volume basis from the Rosin Rammler distribution with parameters 
    k = -ln(0.5) and alpha = 1.8.  
    
    Parameters
    ----------
    nbins : int
        Desired number of size bins in the particle volume size distribution
    d50 : float
        Volume mean particle diameter (m)
    md_total : float
        Total particle mass flux (kg/s)
    sigma : float
        Interfacial tension between the phase undergoing breakup and the 
        ambient receiving continuous phase (N/m)
    rho_p : float
        Density of the phase undergoing breakup (kg/m^3)
    rho : float
        Density of the ambient receiving continuous phase (kg/m^3) 
    
    Returns
    -------
    de : ndarray
        Array of particle sizes at the center of each bin in the distribution
        (m)
    md : ndarray
        Total mass flux of particles corresponding to each bin (kg/s)
    
    Notes
    -----
    This method computes the un-truncated volume size distribution from the
    Rosin-Rammler distribution and then enforces that all particle sizes
    be less than the maximum stable size by moving mass in larger sizes to 
    the maximum stable size bin.  
    
    References
    ----------
    Johansen, Brandvik, and Farooq (2013), "Droplet breakup in subsea oil
    releases - Part 2: Predictions of droplet size distributions with and 
    without injection of chemical dispersants." Marine Pollution Bulletin,
    73: 327-335.  doi:10.1016/j.marpolbul.2013.04.012.
    
    """
    # Get the maximum stable size
    dmax = de_max(sigma, rho_p, rho)
    
    # Define the parameters of the distribution
    k = np.log(0.5)
    alpha = 1.8
    
    de, V_frac = psf.rosin_rammler(nbins, d50, k, alpha)
    
    # Compute the mass fraction for each diameter
    md = V_frac * md_total
    
    # Truncate the distribution at the maximum stable droplet size
    imax = -1
    for i in range(len(de)):
        if de[i] > dmax:
            if imax < 0:
                imax = i
                de[i] = dmax
            else:
                md[imax] += md[i]
                md[i] = 0.
    
    # Return the particle size distribution
    return (de, md)

