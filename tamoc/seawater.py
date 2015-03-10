"""
Equations of State for Seawater
===============================

This module provides a few simple equations of state that can be used for
seawater.  Some are very good (e.g., density) and others are just taken from 
tables in the Handbook of Physics and Chemistry for pure water (e.g., surface
tension).  Eventually, these should be replaced with routines from the 
official seawater equation of state.

"""
# S. Socolofsky, March 2013, Texas A&M University <socolofs@tamu.edu>.

import numpy as np

# Define some universal constants
g = 9.81       # Acceleration of gravity (m/s^2)

def density(T, S, P):
    """
    Computes the density of seawater from Gill (1982)
    
    Computes the density of seawater using the equation of state in Gill
    (1982), *Ocean-Atmosphere Dynamics*, Academic Press, New York.  The
    equations for this code are taken from Appendix B in Crounse (2000).
    
    Parameters
    ----------
    T : float
        temperature (K)
    S : float
        salinity (psu)
    P : float
        pressure (Pa)
    
    Returns
    -------
    rho : float
        seawater density (kg/m^3)
    
    """
    # Convert T to dec C and P to bar
    T = T - 273.15
    P = P * 1.e-5
    
    # Compute the density at atmospheric pressure
    rho_sw_0 = (
                999.842594 + 6.793952e-2 * T - 9.095290e-3 * T**2 
                + 1.001685e-4 * T**3 - 1.120083e-6 * T**4 + 6.536332e-9 * T**5 
                + 8.24493e-1 * S - 5.72466e-3 * S**(3./2.) + 4.8314e-4 * S**2 
                - 4.0899e-3 * T*S + 7.6438e-5 * T**2 * S - 8.2467e-7 * T**3 * 
                S + 5.3875e-9 * T**4 * S + 1.0227e-4 * T * S**(3./2.) 
                - 1.6546e-6 * T**2 * S**(3./2.)
                )
    
    # Compute the pressure correction coefficient
    K = (
         19652.21 + 148.4206 * T - 2.327105 * T**2 + 1.360477e-2 * T**3 
         - 5.155288e-5 * T**4 + 3.239908 * P + 1.43713e-3 * T * P 
         + 1.16092e-4 * T**2 * P - 5.77905e-7 * T**3 * P 
         + 8.50935e-5 * P**2 - 6.12293e-6 * T * P**2 
         + 5.2787e-8 * T**2 * P**2 + 54.6746 * S - 0.603459 * T * S 
         + 1.09987e-2 * T**2 * S - 6.1670e-5 * T**3 * S 
         + 7.944e-2 * S**(3./2.) + 1.64833e-2 * T * S**(3./2.) 
         - 5.3009e-4 * T**2 * S**(3./2.) + 2.2838e-3 * P * S 
         - 1.0981e-5 * T * P * S - 1.6078e-6 * T**2 * P * S 
         + 1.91075e-4 * P * S**(3./2.) - 9.9348e-7 * P**2 * S 
         + 2.0816e-8 * T * P**2 * S + 9.1697e-10 * T**2 * P**2 * S
         )
    
    return rho_sw_0 / (1 - P / K)

def mu(T, S, P):
    """
    Compute the viscosity of seawater
    
    Evaluates the viscosity of seawater as a function of temperature, 
    salinity, and pressure.  The equation accounting for temperature and 
    salinity is from Sharqawy et al. (2010).  The pressure correction is 
    from McCain (1990), equation B-75 on page 528. 
    
    Parameters
    ----------
    T : float
        temperature (K)
    S : float
        salinity (psu)
    P : float
        pressure (Pa)
    
    Returns
    -------
    mu : float
        dynamic viscosity of seawater (Pa s)
    
    """
    # The following equations use Temperature in deg C
    T = T - 273.15
    
    # Get the fit coefficients
    a = np.array([1.5700386464E-01, 6.4992620050E+01, -9.1296496657E+01,
                  4.2844324477E-05, 1.5409136040E+00, 1.9981117208E-02,
                  -9.5203865864E-05, 7.9739318223E+00, -7.5614568881E-02,
                  4.7237011074E-04])
                  
    # Compute the viscosity of pure water at given temperature
    mu_w = a[3] + 1./(a[0] * (T + a[1])**2 + a[2])
    
    # Correct for salinity
    S = S / 1000.
    A = a[4] + a[5] * T + a[6] * T**2
    B = a[7] + a[8] * T + a[9] * T**2
    mu_0 = mu_w * (1. + A * S + B * S**2)
    
    # And finally for pressure
    P = P * 0.00014503773800721815
    mu = mu_0 * (0.9994 + 4.0295e-5 * P + 3.1062e-9 * P**2)
    
    # Return the in situ dynamic viscosity
    return mu

def sigma(T):
    """
    Compute the surface tension of seawater
    
    Evaluates the surface tension of seawater as a function of temperature per 
    data in the CRC handbook and fit following methods in Leifer et al. 
    (2000).  Note that Leifer et al. (2000) do not present a correlation for
    sigma; rather, this function just follows their methodology for viscosity.
    
    Parameters
    ----------
    T : float
        temperature (K)
    
    Returns
    -------
    sigma : float
        interfacial tension of air in seawater (N/m)
    
    """
    T = T - 273.15
    return ((-0.00035) * T**2 - 0.1375 * T + 75.64) * 1e-3 

def k():
    """
    Compute the thermal conductivity of seawater
    
    Evaluates the thermal conductivity of seawater, evaluating it as a 
    constant.
    
    Returns
    -------
    k : float
        thermal conductivity of seawater (m^2/s)
    
    Notes
    -----
    TODO (S. Socolofsky, 4/16/13): Get the correct equation of state for 
    seawater.  This constant value is what was used in the Matlab SIMP 
    Ver. 1.0 calculations and should be retained until validation against
    that code is complete.
    
    """
    return 1.46e-7
    
def cp():
    """
    Compute the heat capacity of seawater at constant pressure
    
    Evaluates the heat capacity of seawater at constant pressure, evaluating 
    it as a constant.
    
    Returns
    -------
    cp : float
        heat capacity of seawater (J/(kg K))
    
    Notes
    -----
    TODO (S. Socolofsky, 4/16/13): Get the correct equation of state for 
    seawater.  This constant value is what was used in the Matlab SIMP 
    Ver. 1.0 calculations and should be retained until validation against
    that code is complete.
    
    """
    return 4185.5

