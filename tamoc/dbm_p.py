"""
dbm_p.py
========

Recreate all of the functions in the Fortran files using Python

This module recreates all of the functions in the Fortran files dbm_eos.f95, dbm_phys.f95, and math_funcs.f95 using regular Python code.  This will not be as fast as the compiled Fortran code, but will make installing TAMOC easier in environments where a modern Fortran compiler is not available or not compatible with the f2py modules used in the setup.py module of TAMOC.

"""
# S. Socolofsky, April 2022, Texas A&M University <socolofs@tamu.edu>

from __future__ import (absolute_import, division, print_function)

import numpy as np

# Declare some global variables
G = 9.81
PI = 3.141592653589793
RU = 8.314510
P_ATM = 101325.
M_SEA = 0.06835

# Functions from math_funcs.f95 ----------------------------------------------

def cubic_roots(p):
    """
    Computes the roots of a cubic polynomial with coefficients p().
    
    Computes the roots of a 3rd-order polynomial with real-valued 
    coefficients specified in p().  The order of the coefficients in 
    p() are given by
    
        p(1) * x**3 + p(2) * x**2 + p(3) * x + p(4) = 0
    
    Parameters
    ----------
    p : ndarray
        Array (order 4) of polynomial coefficients (must be real)
    
    Returns
    -------
    x0 : ndarray
        Array (order 3) of roots (real or complex)
    
    """
    # Use the numpy built-in function
    if np.any(np.isnan(p)):
        x0 = np.zeros(3) + np.nan
    else:
        x0 = np.roots(p)
    
    return x0


# Functions from dbm_phys.f95 ------------------------------------------------

def eotvos(de, rho_p, rho, sigma):
    """
    
    Calculate the Eotvos number per Clift et al. page 26
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    rho_p : float
        Dispersed phase density (kg/m^3)
    rho : float
        Continuous phase density (kg/m^3)
    sigma : float
        Interfacial tension (N/m)
    
    Returns
    -------
    Eo : float
        The non-dimensional Eo number
    
    """
    Eo = G * (rho - rho_p) * de**2 / sigma
    
    return Eo


def morton(rho_p, rho, mu, sigma):
    """
    
    Calculate the Morton number per Clift et al. page 26
    
    Parameters
    ----------
    rho_p : float
        Dispersed phase density (kg/m^3)
    rho : float
        Continuous phase density (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    sigma : float
        Interfacial tension (N/m)
    
    Returns
    -------
    M : float
        The non-dimensional M number
    
    """
    M = G * mu**4 * (rho - rho_p) / (rho**2 * sigma**3)
    
    return M


def reynolds(de, us, rho, mu):
    """
    
    Calculate the Reynolds number per Clift et al. page 26
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    us : float
        Slip velocity of the dispersed phase (m/s)
    rho : float
        Continuous phase density (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    
    Returns
    -------
    Re : float
        The non-dimensional Re number
    
    """ 
    Re = rho * de * us / mu
    
    return Re


def h_parameter(Eo, M, mu):
    """
    Calculate H in equation (7-7) of Clift et al. page 176
    
    Parameters
    ----------
    Eo : float
        Non-dimensional Eotvos number
    M : float
        Non-dimensional Morton number
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    
    Returns
    -------
    H : float
        The non-dimensional parameter H in equation (7-7) of Clift et 
        al. 1978 page 176
    
    """
    H = 4. / 3. * Eo * M**(-0.149) * (mu / 0.0009)**(-0.14)
    
    return H


# ---------------------------------
# Slip velocity and shape functions
# ---------------------------------

def particle_shape(de, rho_p, rho, mu, sigma):
    """
    Calculate the shape of a fluid particle
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    rho_p : float
        Dispersed phase density (kg/m^3)
    rho : float
        Continuous phase density (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    sigma : float
        Interfacial tension (N/m)
    
    Returns
    -------
    shape_p : int
        An integer flag, where 1=sphere, 2=ellipsoid, or 3=spherical cap
    
    """
    # Calculate the non-dimensional variables
    Eo = eotvos(de, rho_p, rho, sigma)
    M = morton(rho_p, rho, mu, sigma)
    H = h_parameter(Eo, M, mu)
    
    # Select the appropriate shape classification
    if (H < 2.):
        shape_p = 1
    elif (Eo < 40. and M < 0.001 and H < 1000.):
        shape_p = 2
    else:
        shape_p = 3
    
    return shape_p


def theta_w_sc(de, us, rho, mu):
    """
    Compute the wake angle for a spherical cap bubble
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    us : float
        Slip velocity of the dispersed phase (m/s)
    rho : float
        Continuous phase density (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    
    Returns
    -------
    theta_w : float
        The wake angle (rad) for a spherical cap bubble using equation 
        (8-1) in Clift et al. (1978) p. 204.
    
    """
    # Get the Reynolds number
    Re = reynolds(de, us, rho, mu)
    
    # Compute the wake angle
    theta_w = PI * (50. + 190. * np.exp(-0.62 * Re**(0.4))) / 180.
    
    return theta_w


def surface_area_sc(de, theta_W):
    """
    Compute the surface area for a spherical cap fluid particle
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    theta_w : float
        Wake angle for the partial sphere model (rad)
    
    Returns
    ------- 
    area : float
        The surface area (m^2) of a spherical cap fluid particle per the 
        model sketched in figure 8.1 of Clift et al. (1978) p. 204.
    
    """
    # Match the volume
    V = 4. / 3. * PI * (de / 2.)**3
    
    # Find the radius at the bottom of the partial sphere
    r_sc = (V / PI / (2. / 3. - np.cos(theta_W) + np.cos(theta_W)**3 / 
        3.))**(1. / 3.)
    
    # Surface area of the frontal sphere
    Af = 2. * PI * r_sc**2 * (1. - np.cos(theta_W))
    
    # Surface area of the real bottom of the partial sphere
    Ar = PI * (r_sc * np.sin(theta_W))**2
    
    # Compute the surface area
    area = Af + Ar
    
    return area


def surface_area_sphere(de):
    """
    Compute the surface area for a spherical cap fluid particle
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    theta_w : float
        Wake angle for the partial sphere model (rad)
    
    Returns 
    -------
    area : float
        The surface area (m^2) of a spherical cap fluid particle per the 
        model sketched in figure 8.1 of Clift et al. (1978) p. 204.
    
    """
    # Compute the surface area
    area = PI * de**2
    
    return area


def us_sphere(de, rho_p, rho, mu):
    """
    Compute the slip velocity of a rigid sphere
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    rho_p : float
        Dispersed phase density (kg/m^3)
    rho : float
        Continuous phase density (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    
    Returns
    -------
    u_slip : float
        The slip velocity (m/s) of a rigid spherical particle per 
        equation (5-15) and following in Clift et al. (1978) p. 133ff.
    
    """
    # Compute the non-dimensional independent parameters
    Nd = 4. * rho * np.abs(rho - rho_p) * G * de**3 / (3. * mu**2)
    W = np.log10(Nd)
    
    # Compute the Reynolds number from the correlations
    if (Nd <= 73.):
        Re = (Nd / 24. - 1.7569e-4 * Nd**2 + 6.9252e-7 * Nd**3
            - 2.3027e-10 * Nd**4)
    elif (Nd <= 580.):
        Re = 10.**(-1.7095 + 1.33438 * W - 0.11591 * W**2)
    elif (Nd <= 1.55e7):
        Re = 10.**(-1.81391 + 1.34671 * W - 0.12427 * W**2 +
           0.006344 * W**3)
    elif (Nd <= 5.0e10):
        Re = 10.**(5.33283 - 1.21728 * W + 0.19007 * W**2 -
           0.007005 * W**3)
    else:
        print('US_SPHERE: Outside range of Nd -- RE set to zero')
        Re = 0.
    
    # Compute the slip velocity
    u_slip = mu / (rho * de) * Re
    
    return u_slip


def us_ellipsoid(de, rho_p, rho, mu_p, mu, sigma, status):
    """
    Compute the slip velocity of an elliptical-wobbling fluid particle
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    rho_p : float 
        Dispersed phase density (kg/m^3)
    rho : float
        Continuous phase density (kg/m^3)
    mu_p : float
        Dynamic viscosity of the dispersed phase (Pa s)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    sigma : float
        Interfacial tension (N/m)
    status : int
        Flag indicating whether the interface is clean (status = 1)
             or dirty (status = -1)
    
    Returns
    -------
    u_slip : float
        The slip velocity (m/s) of an elliptical-wobbling fluid 
        particle per equation (7-4) and following in Clift et al. (1978) 
        p. 175ff.
    
    """
    # Calculate the non-dimensional variables
    Eo = eotvos(de, rho_p, rho, sigma)
    M = morton(rho_p, rho, mu, sigma)
    H = h_parameter(Eo, M, mu)
    
    # Compute the correlation equations
    if (H > 59.3):
        J = 3.42 * H**(0.441)
    else:
        J = 0.94 * H**(0.757)
    
    # Calculate the Reynolds number
    Re = M**(-0.149) * (J - 0.857)
    
    # Compute the dirty-bubble the slip velocity
    us_dirty =  mu / (rho * de) * Re
    
    # Return the correct slip velocity
    if (status > 0):
        # Compute the clean-bubble correction from Figure 7.7 and Eqn. 7-10 
        # in Clift et al. (1978)
        kappa = mu_p / mu
        Xi = Eo * (1. + 0.15 * kappa) / (1. + kappa)
        gamma = 2. * np.exp(-(np.log10(Xi) + 0.6383)**2 / 
            (0.2598 + 0.2*(np.log10(Xi) + 1.))**2)
        u_slip = us_dirty * (1. + gamma/(1. + kappa))
    else:
        u_slip = us_dirty
    
    return u_slip


def us_spherical_cap(de, rho_p, rho):
    """
    Compute the slip velocity of a spherical cap fluid particle
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    rho_p : float
        Dispersed phase density (kg/m^3)
    rho : float
        Continuous phase density (kg/m^3)
    
    Returns
    -------
    u_slip : float
        The slip velocity (m/s) of a spherical cap fluid particle using
        equation (8-11) in Clift et al. (1978) p. 206.  This is the equation 
        also suggested by Zheng and Yapa (2000).  This is strictly valid for
        Re > 150 and Eo > 40, though these limits are not tested here.
    
    """
    # Compute the slip velocity
    u_slip = 0.711 * np.sqrt(G * de * (rho - rho_p) / rho)
    
    return u_slip


# --------------------------
# Mass Transfer Coefficients
# --------------------------

def xfer_kumar_hartland(de, us, rho, mu, D, sigma, mu_p):
    """
    Compute the mass transfer coefficient for clean fluid particles
    
    Computes the mass transfer coefficients for clean fluid particles 
    following the formulas in Kumar and Hartland (1999), "Correlations for
    prediction of mass transfer coefficients in single drop systems and 
    liquid-liquid extraction columns," Trans. IChemE., 77, Part A, 372--
    384.
    
    This equation performs similarly to the Johnson et al. (1969) equation
    in the ellipsoidal bubble regime and for large spherical particles and
    follows the Clift et al. (1978) formulation for solid particles for 
    smaller spherical particles.  For small spherical particles, the Johnson
    et al. (1969) formula gets very small; hence, either the Clift et al.
    (1978) or the Kumar and Hartland (1999) equations are preferred for 
    small spherical particles.
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    us : float
        Slip velocity of the dispersed phase (m/s)
    rho : float
        Continuous phase density (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    D : ndarray
        Diffusion coefficients of the dispersed phase components in the
        continuous phase fluid (m^2/s)
    sigma : float
        Interfacial tension between seawater and the dispersed 
        phase (N/m)
    mu_p : float
        Viscosity of the dispersed phase (Pa s)
    
    Returns
    -------
    beta : ndarray
        The mass transfer coefficients (m/s) for each component in the
        dispersed phase (assuming rigid spheres)
    
    """
    # Compute the Reynolds, Schmidt, and Peclet numbers
    Re = de * us * rho / mu
    Sc = mu / (rho * D)
    Pe = de * us / D
    
    # Constants for the formulas
    C1 = 50.
    C2 = 5.26e-2
    n1 = (1. / 3.0) + 6.59e-2 * Re**0.25
    n2 = 1. / 3.
    n3 = 1. / 3.
    n4 = 1.1
    
    # Compute equation 16
    Sh_rigid = 2.43 + 0.775 * Re**0.5 * Sc**n2 + 0.0103 * Re * Sc**n2
    
    # Compute equation 50
    Sh_infty = C1 + 2. / np.sqrt(PI) * Pe**0.5
    
    # Compute lambda as the RHS of equation 51
    lam = C2 * Re**n1 * Sc**n2 * (us * mu / sigma)**n3 * 1. / \
        (1. + (mu_p / mu)**n4)
    
    # Compute the in situ Sherwood number
    Sh = (Sh_infty * lam + Sh_rigid) / (1. + lam)
    
    # Convert Sherwood number to mass transfer coefficient
    beta = Sh * D / de
    
    return beta


def xfer_johnson(de, us, D):
    """
    Compute the mass transfer coefficient for clean particles
    
    Computes the mass transfer coefficient for clean particles given by 
    equation (42) in Johnson et al. (1969), Canadian Journal of Chemical
    Engineering, vol. 47, pp. 559-564.
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    us : float
        Slip velocity of the dispersed phase (m/s)
    D : ndarray
        Diffusion coefficients of the dispersed phase components in the
        continuous phase fluid (m^2/s)
    
    Returns
    -------
    beta : ndarray
        The mass transfer coefficients (m/s) for each component in the
        dispersed phase (assuming clean particles)
    
    """
    # Compute equation (42) in Johnson et al. (1969)
    beta = 1.13 * np.sqrt(D * us * 100.**3 / (0.45 + 
        0.2 * de * 100.)) / 100.
    
    return beta


def xfer_clift(de, us, rho, mu, D):
    """
    Compute the mass transfer coefficients for a rigid sphere
    
    Computes the mass transfer coefficients for a rigid sphere in water from
    equations in Clift et al. (1978).  For Re < 1, uses equation (3-49) on 
    page 49 for creeping flow past a sphere.  For higher Reynolds numbers,
    for Re < 100, equation (5-25) page 121 is used; for Re < 1e5, equations 
    in Table 5.4 on page 123 are used.  All of these equations assume high 
    Schmidt number.
    
    These equations may be used for fluid particles in contaminated systems
    when slight impurities result in the bubble or droplet having no 
    internal circulations.
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    us : float
        Slip velocity of the dispersed phase (m/s)
    rho : float 
        Continuous phase density (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    D : ndarray
        Diffusion coefficients of the dispersed phase components in the
        continuous phase fluid (m^2/s)
    status : int
        Flag indicating whether the interface is clean (status = 1)
        or dirty (status = -1)

    Returns
    -------
    beta : ndarray
        The mass transfer coefficients (m/s) for each component in the
        dispersed phase.
    
    """
    # Compute the non-dimensional governing parameters
    Sc = mu / (rho * D)
    Pe = us * de / D
    Re = reynolds(de, us, rho, mu)
    
    # Compute the Sherwood Number
    Sh = np.zeros(D.shape)
    for i in range(len(D)):
        if (D[i] > 0.):
            if (Re < 1.):
                Sh[i] = 1. + (1. + Pe[i])**(1./3.)
            elif (Re < 100.):
                Sh[i] = 1. + (1. + 1. / Pe[i])**(1./3.) * \
                    Re**0.41 * Sc[i]**(1./3.)
            elif (Re < 2000.):
                Sh[i] = 1. + 0.724 * Re**(0.48) * Sc[i]**(1./3.)
            else:
                Sh[i] = 1. + 0.425 * Re**0.55 * Sc[i]**(1./3.)
        else:
            Sh[i] = 0.
    
    # Compute the dimensional mass transfer coefficient
    beta = Sh * D / de
    
    return beta


def xfer_sphere(de, us, rho, mu, D, sigma, mu_p, fp_type, status):
    """
    Compute the mass transfer coefficients for a spherical fluid particle
    
    This function computes the mass transfer for a spherical fluid particle
    using equations in Johnson et al. (1969) for clean bubbles and equations
    for solid particles in Clift et al. (1978) for dirty bubbles.
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    us : float
        Slip velocity of the dispersed phase (m/s)
    rho : float
        Continuous phase density (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    D : ndarray
        Diffusion coefficients of the dispersed phase components in the
        continuous phase fluid (m^2/s)
    sigma : float
        Interfacial tension between seawater and the dispersed 
        phase (N/m)
    mu_p : flaot
        Viscosity of the dispersed phase (Pa s)
    fp_type : int
        Flag indicating whether the fluid is 0: gas or 1: 
              liquid
    status : int
        Flag indicating whether the interface is clean (status = 1)
        or dirty (status = -1)
    
    Returns
    -------
    beta : ndarra
        The mass transfer coefficients (m/s) for each component in the
        dispersed phase (assuming rigid spheres).
    
    """
    # Compute the correct mass transfer coefficients
    if (status > 0):
        # This is a clean particle
        
        if (fp_type > 0):
            # This is a liquid particle: use Kumar and Hartland
            beta = xfer_kumar_hartland(de, us, rho, mu, D, sigma, mu_p)
            
        else:
            # This is gas: use larger of Johnson or Clift
            beta_j = xfer_johnson(de, us, D)
            beta_c = xfer_clift(de, us, rho, mu, D)
            
            # For small particles, the Johnson formula is too low
            beta = np.zeros(D.shape)
            for i in range(len(D)):
                if (beta_j[i] > beta_c[i]):
                    # Johnson is ok
                    beta[i] = beta_j[i]
                else:
                    # Clift is better
                    beta[i] = beta_c[i]
    
    else:
        # For dirty particles, use Clift et al. (1978)
        beta = xfer_clift(de, us, rho, mu, D)
    
    return beta


def xfer_ellipsoid(de, us, rho, mu, D, sigma, mu_p, fp_type, status):
    """
    Compute the mass transfer coefficients for an ellipsoidal fluid particle
    
    Clift is not very clear on what equations should be used for ellipsoidal
    fluid particles (drops and bubbles), but indicates that in contaminated
    liquids, the mass transfer is close to that of rigid particles (i.e.,
    there is no internal circulation due to the contamination).  Thus, this
    subroutine currently returns the result for rigid spheres if the 
    particles are dirty.  For clean fluid particles, this function returns
    the result from equation (42) in Johnson et al. (1969).
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    us : float
        Slip velocity of the dispersed phase (m/s)
    rho : float
        Continuous phase density (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    D : ndarray
        Diffusion coefficients of the dispersed phase components in the
        continuous phase fluid (m^2/s)
    sigma : float
        Interfacial tension between seawater and the dispersed 
        phase (N/m)
    mu_p : float
        Viscosity of the dispersed phase (Pa s)
    status : int
        Flag indicating whether the interface is clean (status = 1)
        or dirty (status = -1)
    
    Returns
    -------
    beta : ndarray
        The mass transfer coefficients (m/s) for each component in the
        dispersed phase (assuming rigid spheres).
    
    """
    # Compute the correct mass transfer coefficients
    if (status > 0):
        # This is a clean particle
        
        if (fp_type > 0):
            # This is a liquid particle: use Kumar and Hartland
            beta = xfer_kumar_hartland(de, us, rho, mu, D, sigma, mu_p)
            
        else:
            # This is gas: use larger of Johnson or Clift
            beta_j = xfer_johnson(de, us, D)
            beta_c = xfer_clift(de, us, rho, mu, D)
            
            # For small particles, the Johnson formula is too low
            beta = np.zeros(D.shape)
            for i in range(len(D)):
                if (beta_j[i] > beta_c[i]):
                    # Johnson is ok
                    beta[i] = beta_j[i]
                else:
                    # Clift is better
                    beta[i] = beta_c[i]
    
    else:
        # For dirty particles, use Clift et al. (1978)
        beta = xfer_clift(de, us, rho, mu, D)
    
    return beta


def xfer_spherical_cap(de, us, rho, rho_p, mu, D, status):
    """
    Compute the mass transfer coefficients for spherical cap fluid particles
    
    Computes the mass transfer coefficient for spherical-cap bubbles or
    droplets.  If the particles are clean, it uses equation (42) in 
    Johnson et al. (1969).  If the particles are dirty, it uses equation 
    (8-28) in Clift et al. (1978), p. 214.
    
    Parameters
    ----------
    de : float
        Equivalent spherical diameter (m)
    us : float
        Slip velocity of the dispersed phase (m/s)
    rho : float
        Continuous phase density (kg/m^3)
    rho_p : float
        Dispersed phase density (kg/m^3)
    mu : float
        Dynamic viscosity of the continuous phase (Pa s)
    D : ndarray
        Diffusion coefficients of the dispersed phase components in the
        continuous phase fluid (m^2/s)
    status : int
        Flag indicating whether the interface is clean (status = 1)
        or dirty (status = -1)
    
    Returns
    -------
    beta : ndarray
        The mass transfer coefficients (m/s) for each component in the
        dispersed phase.
    
    """
    # Compute the correct mass transfer coefficients
    if (status > 0):
        # Use the Johnson et al. (1969) equation for clean bubbles
        beta = xfer_johnson(de, us, D)
        
    else:
        # Use the Clift et al. (1978) equation for spherical cap bubbles
        # Compute the wake angle for the partial sphere model (equation 8-1)
        theta_w = theta_w_sc(de, us, rho, mu)
        
        # Compute the surface area of the spherical cap and equivalent sphere
        A = surface_area_sc(de, theta_w)
        Ae = 4. * PI * (de / 2.)**2
        
        # Compute the mass transfer (equation 8-28)
        beta = (1.25 * (G * (rho - rho_p) / rho)**(0.25) * np.sqrt(D) / \
            de**(0.25)) * Ae / A
    
    return beta

# Functions from dbm_eos.f95 -------------------------------------------------

# ---------------------------------------------------------
# Peng-Robinson Equations of State for Density and Fugacity
# ---------------------------------------------------------

def density(T, P, mass, Mol_wt, Pc, Tc, Vc, omega, delta, Aij, 
            Bij, delta_groups, calc_delta):
    """
    
    Computes the liquid and gas density of a mixture from the P-R EOS
    
    Computes the density of a mixture using the Peng-Robinson equation 
    of state as described in McCain (1990), Properties of Petroleum 
    Fluids, 2nd Edition, PennWell Publishing Company, Tulsa, Oklahoma.
    
    Parameters
    ----------
    T : float
        Temperature (K)
    P : float
        Pressure (Pa)
    mass : ndarray
        Array of masses for each component in the mixture (kg)
    Mol_wt : ndarray
        Array of molecular weights for each component (kg/mol)
    Pc : ndarray
        Array of critical point pressures for each component (Pa)
    Tc : ndarray
        Array of critical point temperatures for each component (K)
    omega : ndarray
        Array of Pitzer acentric factors for each component (--)
    delta : ndarray
        Matrix of binary interaction coefficients (--)
    Aij : ndarray
        Group contribution matrix A in Privat and Jaubert (2012) (Pa)
    Bij : ndarray
        Ggroup contribution matrix B in Privat and Jaubert (2012) (Pa)
    delta_groups : ndarray
        Group contribution numbers (normalized) for each 
        component in the mixture (--)
    calc_groups : ndarray
        Flag indicating whether or not delta_groups has 
        been provided (1 = yes, -1 = no)
    
    Returns
    -------
    rho : ndarray
        Array of the density [gas, liquid] of the mixture 
        (kg/m^3)
    
    """
    # Get the z-factor using the Peng-Robinson equation of state
    z, A, B, Ap, Bp, yk = z_pr(T, P, mass, Mol_wt, Pc, Tc, omega, delta, 
        Aij, Bij, delta_groups, calc_delta)
    
    # Convert the masses to mole fraction
    yk = mole_fraction(mass, Mol_wt)
    
    # Compute the volume translation coefficient
    vt = volume_trans(T, P, mass, Mol_wt, Pc, Tc, Vc)
    
    # Compute the molar volume
    nu = z * RU * T / P - np.sum(yk * vt)
    
    # Compute and return the density
    rho = 1. / nu * np.sum(yk * Mol_wt)
    
    return rho


def fugacity(T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij,
             delta_groups, calc_delta):
    """
    Computes the liquid and gas fugacity of a mixture from the P-R EOS
    
    Computes the gas and liquid fugacity of a mixture using the Peng-
    Robinson equation of state as described in McCain (1990), Properties of 
    Petroleum Fluids, 2nd Edition, PennWell Publishing Company, Tulsa, 
    Oklahoma.
    
    Parameters
    ----------
    T : float
        Temperature (K)
    P : float
        Pressure (Pa)
    mass : ndarray
        Array of masses for each component in the mixture (kg)
    Mol_wt : ndarray
        Array of molecular weights for each component (kg/mol)
    Pc : ndarray
        Array of critical point pressures for each component (Pa)
    Tc : ndarray
        Array of critical point temperatures for each component (K)
    omega : ndarray
        Array of Pitzer acentric factors for each component (--)
    delta : ndarray
        Matrix of binary interaction coefficients (--)
    Aij : ndarray
        Group contribution matrix A in Privat and Jaubert (2012) (Pa)
    Bij : ndarray
        Ggroup contribution matrix B in Privat and Jaubert (2012) (Pa)
    delta_groups : ndarray
        Group contribution numbers (normalized) for each 
        component in the mixture (--)
    calc_groups : ndarray
        Flag indicating whether or not delta_groups has 
        been provided (1 = yes, -1 = no)
    
    Returns
    -------
    fug : ndarray 
        Array of the fugacities [gas, liquid] of the mixture (Pa)
        The first row of f are the gas component fugacities and the 
        second row of f are the liquid component fugacities
    
    """
    # Get the z-factor using the Peng-Robinson equation of state
    z, A, B, Ap, Bp, yk =  z_pr(T, P, mass, Mol_wt, Pc, Tc, omega, delta, 
        Aij, Bij, delta_groups, calc_delta)
    
    fug = np.zeros((2,len(mass)))
    for i in range(2):
        fug[i,:] = np.exp((z[i,0] - 1.) * Bp - np.log(z[i,0] - B) - A / 
                   (2.**(1.5) * B) * (Ap - Bp) * np.log((z[i,0] + 
                   (np.sqrt(2.) + 1.) * B) / (z[i,0] - (np.sqrt(2.) - 
                   1.) * B))) * yk * P
    
    return fug


def volume_trans(T, P, mass, Mol_wt, Pc, Tc, Vc):
    """
    Computes the volume translation parameter to correct the density
    
    Computes the volume translation parameter to correct the density from
    the Peng-Robinson Equation of State based on Lin and Duan (2005), 
    "Empirical correction to the Peng-Robinson equation of state for the
    saturated region," Fluid Phase Equilibria, 233: 194-203.  The volume
    translation parameter has a value for each component in the mixture.
    
    Parameters
    ----------
    T : float
        Temperature (K)
    P : float
        Pressure (Pa)
    Pc : ndarray
        Array of critical point pressures for each component (Pa)
    Tc : ndarray
        Array of critical point temperatures for each component (K)
    Vc : ndarray
        Array of critical point molar volumes for each component 
         (m^3/mol)
    mass : ndarray
        Array of masses for each component in the mixture (kg)
    Mol_wt : ndarray
        Array of molecular weights for each component (kg/mol)
    
    Returns
    -------
    vt : ndarray 
        Volume translation parameter (m^3/mol)
    
    """
    # Compute the compressibility factor (--) for each component of the 
    # mixture
    Zc = Pc * Vc / (RU * Tc)
    
    # Calculate the parameters in the Lin and Duan (2005) paper:  beta is 
    # from equation (12)
    beta = -2.8431 * np.exp(-64.2184 * (0.3074 - Zc)) + 0.1735
    
    # and gamma is from Equation (13)
    gamma = -99.2558 + 301.6201 * Zc
    
    # Account for the temperature dependence (equation 10)
    f_Tr = beta + (1. - beta) * np.exp(gamma * np.abs(1. - T / Tc))
    
    # Compute the volume translation for the critical point (equation 9)
    cc = (0.3074 - Zc) * RU * Tc / Pc
    
    # Finally, the volume translation at the given state is (equation 8)
    vt = f_Tr * cc
    
    return vt


def z_pr(T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij,
         delta_groups, calc_delta):
    """
    Computes the z-factor for gas and liquid of a mixture using the P-R EOS
    
    Computes the z-factor of a mixture using the Peng-Robinson equation of
    state as described in McCain (1990), Properties of Petroleum Fluids, 2nd
    Edition, PennWell Publishing Company, Tulsa, Oklahoma.
    
    The approach results in a cubic equation for the z-factor in which the
    largest root is for the liquid phase and the smallest root is for the 
    gas phase; the middle root is discarded.  If the temperature is above
    the critical temperature, only one real root is obtained for the 
    critical state.
    
    Parameters
    ----------
    nc : int
        Number of components in the mixture
    T : float
        Temperature (K)
    P : float
        Pressure (Pa)
    mass : ndarray
        Array of masses for each component in the mixture (kg)
    Mol_wt : ndarray
        Array of molecular weights for each component (kg/mol)
    Pc : ndarray
        Array of critical point pressures for each component (Pa)
    Tc : ndarray
        Array of critical point temperatures for each component (K)
    omega : ndarray
        Array of Pitzer acentric factors for each component (--)
    delta : ndarray
        Matrix of binary interaction coefficients (--)
    Aij : ndarray
        Group contribution matrix A in Privat and Jaubert (2012) (Pa)
    Bij : ndarray
        Group contribution matrix B in Privat and Jaubert (2012) (Pa)
    delta_groups : ndarray
        Group contribution numbers (normalized) for each 
        component in the mixture (--)
    calc_groups : ndarray
        Flag indicating whether or not delta_groups has 
        been provided (1 = yes, -1 = no)
        
    Returns
    -------
    z : ndarray
        Array of the z-factor (gas, liquid) for the mixture (--)
    A : float
        aT coefficient in P-R EOS
    B : float
        b coefficient in P-R EOS
    Ap : ndarray
        Non-dimensional array of mixture aT-coefficients
    Bp : ndarray
        Non-dimensional array of mixture b-coefficients
    
    """
    # Compute the coefficients of the polynomial for z-factor
    A, B, Ap, Bp, yk = coefs(T, P, mass, Mol_wt, Pc, Tc, omega, delta, 
        Aij, Bij, delta_groups, calc_delta)
    p_coefs = np.zeros(4)
    p_coefs[0] = 1.
    p_coefs[1] = B - 1.
    p_coefs[2] = A - 2. * B - 3. * B**2
    p_coefs[3] = B**3 + B**2 - A * B
    
    # Find the roots of the cubic equation of state
    z_roots = cubic_roots(p_coefs)
    
    # Extract the correct z-factors
    z_max = 0.
    for i in range(3):
        if (np.imag(z_roots[i]) == 0.0):
            if (np.real(z_roots[i]) > z_max):
                z_max = np.real(z_roots[i])
    
    z_min = z_max
    for i in range(3):
        if (np.imag(z_roots[i]) == 0.0):
            if ((np.real(z_roots[i]) < z_min) and (np.real(z_roots[i]) > 0.)):
                z_min = np.real(z_roots[i])
    
    # Return the z-factors in z
    z = np.zeros((2,1))
    z[0,0] = z_max
    z[1,0] = z_min
    
    return (z, A, B, Ap, Bp, yk)


def coefs(T, P, mass, Mol_wt, Pc, Tc, omega, delta_in, Aij, Bij,
          delta_groups, calc_delta):
    """
    Computes the mixture coefficients for the P-R EOS
    
    Computes the mixing rules for the coefficients of the Peng-Robinson
    equation of state as described in McCain (1990), Properties of Petroleum
    Fluids, 2nd Edition, PennWell Publishing Company, Tulsa, Oklahoma.
    
    Parameters
    ----------
    nc : int
        Number of components in the mixture
    T : float
        Temperature (K)
    P : float
        Pressure (Pa)
    mass : ndarray
        Array of masses for each component in the mixture (kg)
    Mol_wt : ndarray
        Array of molecular weights for each component (kg/mol)
    Pc : ndarray
        Array of critical point pressures for each component (Pa)
    Tc : ndarray
        Array of critical point temperatures for each component (K)
    omega : ndarray
        Array of Pitzer acentric factors for each component (--)
    delta_in : ndarray
        Matrix of binary interaction coefficients (--)
    Aij : ndarray
        Group contribution matrix A in Privat and Jaubert (2012) (Pa)
    Bij : ndarray
        Group contribution matrix B in Privat and Jaubert (2012) (Pa)
    delta_groups : ndarray
        Group contribution numbers (normalized) for each 
        component in the mixture (--)
    calc_delta : int
        Flag indicating whether or not delta_groups has 
        been provided (1 = yes, -1 = no)
    
    Returns
    -------
    A : float
        aT coefficient in P-R EOS
    B : float
        b coefficient in P-R EOS
    Ap : ndarray
        Non-dimensional array of mixture aT-coefficients
    Bp : ndarray
        Non-dimensional array of mixture b-coefficients
    yk : ndarray
        Mole fractions of each component of the mixture
    
    """
    # Convert the masses to mole fraction
    yk = mole_fraction(mass, Mol_wt)
    nc = len(mass)
    
    # Compute the coefficient values for each gas in the mixture.  Use the 
    # modified Peng-Robinson (1978) equations for mu
    mu = np.zeros(nc)
    for i in range(nc):
        if (omega[i] > 0.49):
            mu[i] = 0.379642 + 1.48503 * omega[i] - 0.164423 * \
                    omega[i]**2 + 0.016666 * omega[i]**3
        else:
            mu[i] = 0.37464 + 1.54226 * omega[i] - 0.26992 * omega[i]**2
    
    alpha = (1. + mu * (1. - (T / Tc)**(1./2.)))**2
    aTk = 0.45724 * RU**2 * Tc**2 / Pc * alpha
    bk = 0.0778 * RU * Tc / Pc
    
    # Initialize the output vector for delta to the input values
    delta = delta_in
    
    # Get the temperature-dependent binary interaction coefficients (if 
    # the user provided the group contributions)
    if (calc_delta > 0):
        for j in range(1,nc):
            for i in range(j-1):
                sum1 = 0.
                for l in range(15):
                    for k in range(15):
                        sum_term =  (delta_groups[i,k] -  
                                   delta_groups[j,k]) * (delta_groups[i,l] - 
                                   delta_groups[j,l]) * Aij[k, l] * \
                                   (298.15 / T) ** (Bij[k,l] / Aij[k,l] - 
                                   1.)
                        if (~np.isnan(sum_term)):
                            sum1 = sum1 + sum_term
                
                delta[i, j] = - (0.5 * sum1 + (np.sqrt(aTk[i]) / bk[i] - 
                              np.sqrt(aTk[j]) / bk[j]) ** 2) / \
                              (2. * np.sqrt(aTk[i] * aTk[i]) /
                              (bk[i] * bk[j]))
                delta[j, i] = delta[i,j]
    
    # Use the mixing rules in McCain (1990)
    bd = np.sum(yk * bk)
    aT = 0.
    for j in range(nc):
        for i in range(nc):
            aT = aT + yk[i] * yk[j] * (aTk[i] * aTk[j])**(1./2.) * \
                 (1. - delta[i,j])
    
    # Compute the coefficients of the polynomials for z-factor and fugacity
    A = aT * P / (RU**2 * T**2)
    B = bd * P / (RU * T)
    Bp = bk / bd
    Ap = np.zeros(nc)
    for i in range(nc):
        Ap[i] = 1. / aT * (2. * aTk[i]**(1./2.) * \
                np.sum(yk * aTk**(1./2.) * (1. - delta[:,i])))
    
    return (A, B, Ap, Bp, yk)


def mole_fraction(mass, Mol_wt):
    """
    Compute the mole fraction of a mixture from the mass
    
    Converts the masses of each component in a mixture to the mole fraction
    of each component in the mixture.
    
    Parameters
    ----------
    nc = number of components in the mixture
    mass = array of masses for each component in the mixture (kg)
    Mol_wt = array of molecular weights for each component (kg/mol)
    
    Returns
    -------
    yk : ndarray
        the mole fractions (--) of the mixture.
    
    """
    # Compute the total number of moles
    n_moles = mass / Mol_wt
    
    # Compute the mole fraction
    yk = n_moles / np.sum(n_moles)
    
    return yk


# -----------------------------------------------------------
# Other Fluid Properties (viscosity, surface tension, etc...)
# -----------------------------------------------------------

def viscosity(T, P, mass, Mol_wt, Pc, Tc, Vc, omega, delta, Aij,
              Bij, delta_groups, calc_delta):
    """
    Computes the viscosity of a petroleum fluid
    
    Computes the viscosity of the given fluid mixture for the gas and 
    liquid phases following the method in Pedersen et al. "Phase Behavior
    of Petroleum Reservoir Fluids", 2nd edition, Chapeter 10.
    
    This method correlates the viscosity of the mixture to the viscosity
    of methane taken at a specialized corresponding state.  The function
    has the properties of methane hard-wired so that any mixture can be
    evaluated.
    
    Parameters
    ----------
    nc : int
        Number of components in the mixture
    T : float
        Temperature (K)
    P : float
        Pressure (Pa)
    mass : ndarray
        Array of masses for each component in the mixture (kg)
    Mol_wt : ndarray
        Array of molecular weights for each component (kg/mol)
    Pc : ndarray
        Array of critical point pressures for each component (Pa)
    Tc : ndarray
        Array of critical point temperatures for each component (K)
    Vc : ndarray
        Array of critical point molar volumes for each component 
         (m^3/mol)
    omega : ndarray
        Array of Pitzer acentric factors for each component (--)
    delta : ndarray
        Matrix of binary interaction coefficients (--)
    Aij : ndarray
        Group contribution matrix A in Privat and Jaubert (2012) (Pa)
    Bij : ndarray
        Group contribution matrix B in Privat and Jaubert (2012) (Pa)
    delta_groups : ndarray
        Group contribution numbers (normalized) for each 
        component in the mixture (--)
    calc_groups : ndarray
        Flag indicating whether or not delta_groups has 
        been provided (1 = yes, -1 = no)
    
    Returns
    -------
    mu : ndarray
        Array of the viscosity [gas, liquid] of the mixture
        (Pa s)    
    
    """
    # Count the number of chemical components
    nc = len(mass)
    
    # Enter the parameter values from Table 10.1
    GV = np.array([-2.090975e5, 2.647269e5, -1.472818e5, 4.716740e4,
        -9.491872e3, 1.219979e3, -9.627993e1, 4.274152, -8.141531e-2])
    A = 1.696985927
    B = -0.133372346
    C = 1.4
    F = 168.
    jc = np.array([-10.3506, 17.5716, -3019.39, 188.73, 0.0429036, 
        145.29, 6127.68])
    kc = np.array([-9.74602, 18.0834, -4126.66, 44.6055, 0.976544,
        81.8134, 15649.9])
    
    # Enter the properties for the reference fluid (methane)
    M0 = np.array([16.043e-3])
    Tc0 = np.array([190.56])
    Pc0 = np.array([4599000.])
    omega0 = np.array([0.011])
    Vc0 = np.array([9.86e-5])
    delta0 = np.array([[0.]])
    delta_groups0 = np.array([[0., 0., 0., 0., 1., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0.,
                              0.]])
    rho_c0 = 162.84
    
    # 1.  Prepare the variables to determine the corresponding states between
    #    the given mixture and the reference fluid (methane) ----------------
    
    # Get the mole fraction of the components of the mixture
    z = mole_fraction(mass, Mol_wt)
    
    # Compute equation (10.19)
    numerator = 0.
    denominator = 0.
    for i in range(nc):
        for j in range(nc):
            numerator += z[i] * z[j] * ((Tc[i] / Pc[i])  
                        **(1./3.) + (Tc[j] / Pc[j])**(1./3.)) \
                        **3 * np.sqrt(Tc[i] * Tc[j])
            denominator += z[i] * z[j] * ((Tc[i] / Pc[i]) 
                        **(1./3.) + (Tc[j] / Pc[j])**(1./3.)) \
                        **3
    
    Tc_mix = numerator / denominator
    
    # Compute equation (10.22)
    Pc_mix = 8. * numerator / denominator**2
    
    # Get the density of methane at TTc0/Tc_mix and PPc0/Pc_mix
    rho0 = density(T * Tc0 / Tc_mix, P * Pc0 / Pc_mix, 
        np.array([1.]), M0, Pc0, Tc0, Vc0, omega0, delta0, Aij, Bij,
        delta_groups0, -1)
    
    # Compute equation (10.27)
    rho_r = np.zeros((2,1))
    rho_r[:,0] = rho0[:,0] / rho_c0
    
    # Compute equation (10.23), where M is in g/mol
    M = Mol_wt * 1.0e3
    M_bar_n = np.sum(z * M)
    M_bar_w = np.sum(z * M**2) / M_bar_n
    M_mix = 1.304e-4 * (M_bar_w**2.303 - M_bar_n**2.303) + M_bar_n
    
    # Compute equation (10.26), where M is in g/mol
    M0 = M0 * 1.0e3
    alpha_mix = np.zeros((2,1))
    alpha0 = np.zeros((2,1))
    alpha_mix[:,0] = 1. + 7.378e-3 * rho_r[:,0]**1.847 * M_mix**0.5173
    alpha0[:,0] = 1. + 7.378e-3 * rho_r[:,0]**1.847 * M0**0.5173
    
    # 2.  Compute the viscosity of methane at the corresponding state --------
    
    # Corresponding state
    T0 = T * Tc0 / Tc_mix * alpha0[:,0] / alpha_mix[:,0]
    P0 = P * Pc0 / Pc_mix * alpha0[:,0] / alpha_mix[:,0]
    
    # Compute each state separately
    theta = np.zeros((2,1))
    delta_eta_p = np.zeros((2,1))
    delta_eta_pp = np.zeros((2,1))
    eta_ch4 = np.zeros((2,1))
    for i in range(2):
        
        # Get the density of methane at T0 and P0.  Be sure to use molecular
        # weight in kg/mol
        rho0 = density(T0[i], P0[i], np.array([1.]), M0*1.0e-3, Pc0, Tc0, 
            Vc0, omega0, delta0, Aij, Bij, delta_groups0, -1)
        
        # Compute equation (10.10)
        theta[:,0] = (rho0[:,0] - rho_c0) / rho_c0
        
        # Equation (10.9) with T in K and rho in g/cm^3
        rho0[:,0] = rho0[:,0] * 1.0e-3
        
        delta_eta_p[:,0] = np.exp(jc[0] + jc[3] / T0[i]) * (np.exp(rho0[:,0] 
                           **0.1 * (jc[1] + jc[2] / T0[i]**1.5) + 
                           theta[:,0] * rho0[:,0]**0.5 * (jc[4] + jc[5] 
                           / T0[i] + jc[6] / T0[i]**2)) - 1.)
        
        # Equation (10.28)
        delta_eta_pp[:,0] = np.exp(kc[0] + kc[3] / T0[i]) * (np.exp(rho0[:,0]
                            **0.1 * (kc[1] + kc[2] / T0[i]**1.5) + 
                            theta[:,0] * rho0[:,0]**0.5 * (kc[4] + kc[5] 
                            / T0[i] + kc[6] / T0[i]**2)) - 1.)
        
        # Equation (10.7)
        eta_0 = GV[0] / T0[i] + GV[1] / T0[i]**(2./3.) + GV[2] / \
                T0[i]**(1./3.) + GV[3] + GV[4] * T0[i]**(1./3.) \
                + GV[5] * T0[i]**(2./3.) + GV[6] * T0[i] + GV[7] * \
                T0[i]**(4./3.) + GV[8] * T0[i]**(5./3.)
        
        # Equation (10.8)
        eta_1 = A + B * (C - np.log(T0[i] / F))**2
        
        # Equation (10.32)
        delta_T = T0[i] - 91.
        
        # Equation (10.31)
        htan = (np.exp(delta_T) - np.exp(-delta_T)) / (np.exp(delta_T) +
            np.exp(-delta_T))
        
        # Viscosity of methane (Equation 10.29) -- reported in (Pa s)
        eta_ch4[i,0] = (eta_0 + eta_1 + (htan + 1.) / 2. * 
                       delta_eta_p[i,0] + (1. - htan) / 2. * 
                       delta_eta_pp[i,0]) * 1.0e-7
    
    # Compute the viscosity of the mixture at the given T and P
    mu = np.zeros((2,1))
    mu[:,0] = (Tc_mix / Tc0)**(-1./6.) * (Pc_mix / Pc0)** \
              (2./3.) * (M_mix / M0)**(0.5) * alpha_mix[:,0] / \
              alpha0[:,0] * eta_ch4[:,0]
    
    return mu


# ------------------------------------------------
# Modified Henry's Law for Solubility Calculations
# ------------------------------------------------

def kh_insitu(T, P, S, kh_0, dH_solR, nu_bar, Mol_wt, K_salt):
    """
    Compute the in-situ Henry's law constant
    
    Compute the in-situ Henry's law constant per the algorithm in McGinnis
    et al. (2006).  This involves adjustment from Henry's coefficients at 
    STP to the appropriate values at ambient temperature, pressure, and 
    continuous phase salinity.  The conditions at STP are specified in the
    source documentation for the input Henry's coefficients.  Adjustments
    for temperature and pressure are per appropriate thermodynamic equations
    of state.  The adjust for salinity is taken from detailed calculations 
    for dissolution of CO2 in seawater.  The form of the equation is 
    likely correct for a wide range of chemicals; however, the fit 
    coefficients used here were derived for CO2 and should be adjusted 
    when applied to other components.  No available method for adjustment
    is provided in this function.
    
    Parameters
    ----------
    T : float
        Temperature (K)
    P : float
        Pressure (Pa)
    S : float
        Salinity (psu)   
    kh_0 : ndarray
        Henry's Law constant at 298.15 K (kg/(m^3 Pa))
    dH_solR : ndarray
        Enthalpy of solution / R (K)
    nu_bar : ndarray
        Partial molar volume at infinite dilution (m^3/mol)
    Mol_wt : ndarray
        Array of molecular weights for each component (kg/mol) 
    K_salt : ndarray
        Setschenow constant (m^3/mol)
    
    Returns
    -------
    kh : ndarray
        An array of Henry's law coefficients (kg/(m^3 Pa))
    
    """
    kh = np.zeros(kh_0.shape)
    for i in range(len(kh_0)):
        
        if (kh_0[i] < 0.):
            # These are low solubility compounds for which we do not know 
            # the solubility...set kh to zero.
            kh[i] = 0.
        
        else:
            # Adjust from STP to ambient temperature
            kh[i] = kh_0[i] * np.exp(dH_solR[i] * (1. / T - 1. / 298.15))
            
            # Adjust to the ambient pressure
            kh[i] =  kh[i] * np.exp((P_ATM - P) * nu_bar[i] / (RU * T))
            
            # Adjust for the salting out effect of salinity
            kh[i] = kh[i] * 10. ** (-S / M_SEA * K_salt[i])
    
    return kh


def sw_solubility(f, kh):
    """
    """
    Cs = f * kh
    
    return Cs


def diffusivity(mu, Vb):
    """
    Compute the diffusivity of each component in a mixture into seawater
    
    Computes the diffusivity of each component in a fluid mixture into 
    seawater at the given temperature.  The calculation is from Hayduk and
    Laudie (1974), AIChE J., vol. 20, pp. 611-615.
    
    Parameters
    ----------
    mu : float
        Viscosity of seawater at the ambient conditions (Pa s)
    Vb : ndarray
        Molar volume of each compound at its boiling point (m^3/mol)
    
    Returns
    -------
    D : ndarray
        An array of diffusivities (m^2/s) for each component into water.
    
    """
    D = np.zeros(Vb.shape)
    for i in range(len(Vb)):
        
        if (Vb[i] < 0.):
            # For some insoluble compounds, we do not know the inputs...
            # set diffusivity to zero
            D[i] = 0.
            
        else:
            # Use the Hayduk and Laudie formula
            D[i] = 13.26e-9 / ((mu * 1.0e3)**1.14 * (Vb[i] * 1.0e6)
                   **0.589)
    
    return D


# -------------------------------------------------------
# Hydrate formation predictions from Sloan and Koh (2008)
# -------------------------------------------------------

def kvsi_hydrate(T_in, P_in, mass):
    """
    Determine whether or not hydrate is stable for the given conditions
    
    Solve the K_vsi method for hydrate partition coefficient to determine
    whether the given gas composition and thermodynamic state yields a 
    stable hydrate.  The masses of each component of the gas must be 
    organized in the following order:
    
        CH4
        C2H6
        C3H8
        i-C4H10
        n-C4H10
        N2
        CO2
        H2S
    
    Parameters
    ----------
    T_in : float
        Temperature of the gas mixture (K)
    P_in : float
        Pressure (Pa)
    mass : ndarray
        Array of masses for each component in the mixture (kg)   
    
    Returns
    -------
    K_vsi : ndarray
        Hydrate partition coefficient
    yk : ndarray
        Mole fractions
    
    """
    # Define the molecular weight of each gas component
    Mol_wt = np.array([16.0426, 30.0694, 44.0962, 58.123, 58.123, 28.0134,
             44.0098, 34.0818])
    
    # Fill the parameter matrix coef with data from Table 4.4a, p. 223
    coef = np.array([
        [1.63636, 6.41934, -7.8499, -2.17137, -37.211, 1.78857, 9.0242,
         -4.7071],
        [0., 0., 0., 0., 0.86564, 0., 0., 0.06192],
         [0., 0., 0., 0., 0., -0.001356, 0., 0.],
         [31.6621, -290.283, 47.056, 0., 732.2, -6.187, -207.033, 82.627],
         [-49.3534, 2629.1, 0., 0., 0., 0., 0., 0.],
         [-5.31e-6, 0., -1.17e-6, 0., 0., 0., 4.66e-5, -7.39e-6],
         [0., 0., 7.145e-4, 1.251e-3, 0., 0., -6.992e-3, 0.],
         [0., -9.0e-8, 0., 1.0e-8, 9.37e-6, 2.5e-7, -2.89e-6, 0.],
         [0.128525, 0.129759, 0., 0.166097, -1.07657, 0., -6.223e-3,
          0.240869],
         [-0.78338, -1.19703, 0.12348, -2.75945, 0., 0., 0., -0.64405],
         [0., -8.46e4, 1.669e4, 0., 0., 0., 0., 0.],
         [0., -71.0352, 0., 0., -66.221, 0., 0., 0.],
         [0., 0.596404, 0.23319, 0., 0., 0., 0.27098, 0.],
         [-5.3569, -4.7437, 0., 0., 0., 0., 0., -12.704],
         [0., 7.82e4, -4.48e4, -8.84e2, 9.17e5, 5.87e5, 0., 0.],
         [-2.3e-7, 0., 5.5e-6, 0., 0., 0., 8.82e-5, -1.3e-6],
         [-2.0e-8, 0., 0., -5.4e-7, 4.98e-6, 1.0e-8, 2.55e-6, 0.],
         [0., 0., 0., -1.0e-8, -1.26e-6, 1.1e-7, 0., 0.]
     ])
    
    # Convert the masses to mole fraction
    yk = mole_fraction(mass, Mol_wt)
    
    # Convert pressure to psia and temperature to deg F
    P = 0.0001450377377 * P_in
    T = 9. / 5. * (T_in- 273.15) + 32.
    
    # Compute equation 4.2 in Sloan and Koh
    K_vsi = np.zeros(8)
    for i in range(8):
        K_vsi[i] = np.exp(coef[0,i] + coef[1,i] * T + coef[2,i] * P
                   + coef[3,i] / T + coef[4,i] / P + coef[5,i] * P * T
                   + coef[6,i] * T**2 + coef[7,i] * P**2
                   + coef[8,i] * P / T + coef[9,i] * np.log(P / T)
                   + coef[10,i] / P**2 + coef[11,i] * T / P
                   + coef[12,i] * T**2 / P + coef[13,i] * P / T**2
                   + coef[14,i] * T / P**3 + coef[15,i] * T**3
                   + coef[16,i] * P**3 / T**2 + coef[17,i] * T**4)
    
    return (K_vsi, yk)