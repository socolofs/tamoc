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

from __future__ import (absolute_import, division, print_function)

import numpy as np

# Define some universal constants
g = 9.81       # Acceleration of gravity (m/s^2)

def density(T, S, P):
    """
    Computes the density of seawater from Gill (1982)
    
    Computes the density of seawater. For temperatures less than 40 deg C, this
    function uses the equation of state in Gill (1982), *Ocean-Atmosphere
    Dynamics*, Academic Press, New York; the equations for this code are taken from
    Appendix B in Crounse (2000). For higher temperatures, this function uses the
    equations in Sun et al. (2008), Deep-Sea Research I, Volume 55, pages 1304-1310.
    
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
    if T < 273.15 + 40:
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
        
        rho = rho_sw_0 / (1 - P / K)
    
    else:
        # Convert T to deg C and P to MPa
        T = T - 273.15
        P = P / 1.e6
        
        # Summations
        left_col = 9.9920571e2 + 9.5390097e-2 * T - 7.6186636e-3 * T**2 + \
            3.1305828e-5 * T**3 - 6.1737704e-8  * T**4 + 4.3368858e-1 * P + \
            2.5495667e-5 * P*T**2 - 2.8988021e-7 * P*T**3 + \
            9.5784313e-10 * P*T**4 + 1.7627497e-3 * P**2 - 1.2312703e-4 * P**2*T \
            + 1.3659381e-6 * P**2*T**2 + 4.0454583e-9 * P**2*T**3 - 1.4673241e-5 \
            * P**3 + 8.8391585e-7 * P**3*T - 1.1021321e-9 * P**3*T**2 + \
            4.2472611e-11 * P**3*T**3 - 3.9591772e-14 * P**3*T**4
        right_col = -7.99992230e-1 * S + 2.40936500e-3 * S*T - 2.58052775e-5 * \
            S*T**2 + 6.85608405e-8 * S*T**3 + 6.29761106e-4 * P*S - \
            9.36263713e-7 * P**2*S
        
        rho = left_col - right_col
    
    return rho

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

def sigma(T, S):
    """
    Compute the surface tension of seawater
    
    Evaluates the surface tension of seawater as a function of temperature and
    salinity following equations in Sharqawy et al. (2010), Table 6.
    
    Parameters
    ----------
    T : float
        temperature (K)
    S : float
        salinity (psu)
    
    Returns
    -------
    sigma : float
        interfacial tension of air in seawater (N/m)
    
    """
    # Equations in Sharqawy using deg C and g/kg
    T = T - 273.15
    S = S / 1000.
    
    # Use equations (27) for pure water surface tension (N/m)
    sigma_w = 0.2358 * (1. - (T + 273.15) / 647.096)**1.256 * (1. - 0.625 * 
              (1. - (T + 273.15) / 647.096))
    
    # Equation (28) gives the salinity correction
    if T < 40:
        # Salinity correction only valid [0, 40] deg C
        sigma = sigma_w * (1. + (0.000226 * T + 0.00946) * np.log(1. + 0.0331 * 
                S))
    else:
        # No available salinity correction for hot cases
        sigma = sigma_w
    
    return sigma

def k(T, S, P):
    """
    Compute the thermal conductivity of seawater
    
    Evaluates the thermal conductivity of seawater as a function of 
    temperature, pressure, and salinity following equations in Sharqawy et 
    al. (2010), Table 4.
    
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
    k : float
        thermal conductivity of seawater (W/(mK))
    
    Notes
    -----
    Table 4 provides three equations.  Equation (14) is valid for temperatures
    up to 60 deg C, but I could not reproduce the values in the paper.  Hence,
    the slightly less-accurate equation (13) is used for temperatures above
    30 deg C.  
    
    """
    # Thermal conductivity equations use T_68 in deg C
    T_68 = (T - 0.0682875) / (1.0 - 0.00025)
    T_68  = T_68 - 273.15
    
    # Salinity is in g/kg and pressure in MPa
    S = S / 1000.
    P = P * 1e-6
    
    # Compute the thermal conductivity from Table 4
    if T_68 < 30.:
        # Equation (15)
        k_sw = 0.55286 + 3.4025e-4 * P + 1.8364e-3 * T_68 - 3.3058e-7 * \
               T_68**3
        
    else:
        # Equation (13)
        k_sw = 10.**(np.log10(240. + 0.0002 * S) + 0.434 * (2.3 - (343.5 + 
               0.037 * S) / (T_68 + 273.15)) * (1. - (T_68 + 273.15) / 
               (647. + 0.03 * S)) ** 0.333) / 1000.
    
    return k_sw
    
def cp():
    """
    Compute the heat capacity of seawater at fixed conditions
    
    Per Figure 5 in Sharqawy et al. (2010), the heat capacity of seawater 
    only varies +/- 5 percent over practical temperatures and salinities 
    for deepwater blowouts.  If we let heat capacity depend on temperature
    and or salinity, computing the temperature of water given the total 
    heat becomes an implicit calculation.  This is a problem for the plume
    models.  As a result, we choose to set the heat capacity to that of
    seawater at 10 deg C and 34.5 psu.  
        
    Returns
    -------
    cp : float
        heat capacity of seawater (J/(kg K))
    
    Notes
    -----
    This approximation is valid since we have treated cp to be a constant in
    derivation of the governing equations.  If we let cp vary with T and S, 
    then the governing equations will contain a lot of new terms coming from
    gradients of cp due to spatial variation of T and S.  This complexity is
    unnecessary due to the small variation of cp over the environmental 
    range.  In addition, the temperature T will become an implicit equation
    of the heat, H, since H = rho(T) cp(T) T.  Note that we have also used
    the reference density to define rho in the relation for heat:  rho(T) -> 
    rho_0.  This is known as the Boussinesq approximation.  
    
    """
    return 3997.4

def pH(co2, Ta, alk=0.002300, ph_guess=9):
    """ 
    Compute the pH given the DIC in kg/m^3 of CO2
    
    Compute the pH of a solution of CO2 in water given the DIC measured in
    kg/m^3 of CO2. This algorithm assumes an ocean alkalinity of 2,300 ueq/l
    and converts the given `co2` input in kg/m^3 to DIC in mol/L using the
    molecular weight of CO2. The solution method and solubility constants come
    from the EPA document:
    
        https://www.epa.gov/sites/default/files/2018-05/ 
        documents/wasp-ph-release-notes.pdf
    
    The non-linear solution algorithm is sensitive to the initial guess for the
    pH, which can be adjusted by the optional input parameter.
    
    Parameters
    ----------
    co2 : float
        Total dissolved inorganic carbon in kg/m^3 of CO2.
    Ta : float
        Ambient temperature (K)
    alk : float, default=0.002300
        Alkalinity in eq/l.  The average ocean value is 2300 ueq/l, which is the
        value supplied by default.
    ph_guess : float, default=9        
        Initial guess for the pH. It seems the algorithm easily diverges if the
        initial guess is too close to the final answer. Approaching the
        solution from a higher pH seems to be stable.
    
    Returns
    -------
    ph : float
        Converged value of the pH.
    
    """
    # Convert kg/m^3 of co2 to mol / l
    cT = co2 / (12.011 / 1000.) / 1000.
    
    def h_residual(h):
        """
        Compute the residual of the pH calculation
        
        Following the EPA documentation, which is based on Stumm and Morgan
        (1996), the pH is computed from a non-linear, root-finding problem. This
        function returns the residual of the root-finding equation.
        
        """
        # Compute the temperature-dependent solubility constants
        pKw = 4787.3 / Ta + 7.1321 * np.log10(Ta) + 0.010365 * Ta - 22.80
        log_K1 = -356.3094 - 0.06091964 * Ta + 21834.37 / Ta + \
            126.8339 * np.log10(Ta) - 1684915. / Ta**2
        log_K2 = -107.8871 - 0.03252849 * Ta + 5151.79 / Ta + \
            38.92561 * np.log10(Ta) - 563713.9 / Ta**2
        
        # Convert these constants from their log-values to actual numbers
        Kw = 10. ** (-pKw)
        K1 = 10. ** log_K1
        K2 = 10. ** log_K2
        
        # Compute the coefficients of the root-finding equation
        a1 = K1 * h / (h**2 + K1 * h + K1 * K2)
        a2 = K1 * K2 / (h**2 + K1 * h * K1 * K2)
        
        # Return the residual of the root-finding equation
        return (a1 + 2. * a2) * cT + Kw / h - h - alk
    
    # Use a root-finding algorithm to converge on the correct pH
    from scipy.optimize import fsolve
    h = fsolve(h_residual, 10**-ph_guess)[0]
    
    # Convert the hydrogen ion concentration to a pH value
    ph = -np.log10(h)
    
    # Return the result
    return ph

