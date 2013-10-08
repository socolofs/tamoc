"""
Unit tests for the `dbm` module of ``TAMOC``

Provides testing of the functions defined in ``dbm_f``.

``dbm_f`` is a library created by ``f2py`` from the source code::

    dbm_eos.f95
    dbm_phys.f95
    math_funcs.f95

It can be compiled using::

    f2py -c -m dbm_f dbm_eos.f95 dbm_phys.f95 math_funcs.f95

This module tests direct calling of the ``dbm_f`` functions from Python as 
well as calls to the functions implemented in `dbm`.

Notes
-----
All of the tests defined herein check the general behavior of each of the
programmed function--this is not a comparison against measured data. The
results of the hand calculations entered below as sample solutions have been
ground-truthed for their reasonableness. However, passing these tests only
means the programs and their interfaces are working as expected, not that they
have been validated against measurements.

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from tamoc import dbm_f
from tamoc import dbm

import numpy as np
from numpy.testing import *

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def slip_velocity(de, rho_p, rho, sigma, mu, Eo, M, H, shape, us, Re):
    """
    Test the governing parameters, shape, and slip velocity calculations in 
    the fortran file dbm_phys.f95.
    
    """
    
    # Test eotvos()
    Eo_calc = dbm_f.eotvos(de, rho_p, rho, sigma) 
    assert_approx_equal(Eo_calc, Eo, significant = 6)
    
    # Test morton()
    M_calc = dbm_f.morton(rho_p, rho, mu, sigma)
    assert_approx_equal(M_calc, M, significant = 6)
    
    # Test h_parameter()
    H_calc = dbm_f.h_parameter(Eo, M, mu)
    assert_approx_equal(H_calc, H, significant = 6)
    
    # Test particle_shape()
    shape_names = {1: 'sphere', 2: 'ellipsoid', 3:'spherical_cap'}
    shape_calc = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
    assert shape_names[shape_calc] == shape
    
    if shape_calc == 1:
        # Test us_sphere()
        us_calc = dbm_f.us_sphere(de, rho_p, rho, mu)
        assert_approx_equal(us_calc, us, significant = 6)
        
    elif shape_calc == 2:
        # Test us_ellipsoid()
        us_calc = dbm_f.us_ellipsoid(de, rho_p, rho, mu, sigma)
        assert_approx_equal(us_calc, us, significant = 6)
    
    else:
        # Test us_spherical_cap()
        us_calc = dbm_f.us_spherical_cap(de, rho_p, rho)
        assert_approx_equal(us_calc, us, significant = 6)
    
    # Test reynolds()
    Re_calc = dbm_f.reynolds(de, us, rho, mu)
    assert_approx_equal(Re_calc, Re, significant = 6)

def mass_transfer(de, rho_p, rho, sigma, mu, D, theta_w, area, beta):
    """
    Test the mass transfer calculations in the fortran file 
    dbm_phys.f95.  Note that heat transfer calls the same function; thus, 
    it does not have to be tested separately.  
    
    """
    
    # Get the shape and make the appropriate calculations
    shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
    
    if shape == 1:
        # Test xfer_sphere()
        us = dbm_f.us_sphere(de, rho_p, rho, mu)
        beta_calc = dbm_f.xfer_sphere(de, us, rho, mu, D)
        for i in range(len(beta_calc)):
            assert_approx_equal(beta_calc[i], beta[i], significant = 6)
        
        area_calc = dbm_f.surface_area_sphere(de)
        assert_approx_equal(area_calc, np.pi * de**2, significant = 6)        
    
    elif shape == 2:
        # Test xfer_ellipsoid()
        us = dbm_f.us_ellipsoid(de, rho_p, rho, mu, sigma)
        beta_calc = dbm_f.xfer_ellipsoid(de, us, rho, mu, D)
        for i in range(len(beta_calc)):
            assert_approx_equal(beta_calc[i], beta[i], significant = 6)
        
        area_calc = dbm_f.surface_area_sphere(de)
        assert_approx_equal(area_calc, np.pi * de**2, significant = 6)
    
    else:
        # Test xfer_spherical_cap()
        us = dbm_f.us_spherical_cap(de, rho_p, rho)
        beta_calc = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, D)
        for i in range(len(beta_calc)):
            assert_approx_equal(beta_calc[i], beta[i], significant = 6)
        
        # Test theta_w_calc()
        theta_w_calc = dbm_f.theta_w_sc(de, us, rho, mu)
        assert_approx_equal(theta_w_calc, theta_w, significant = 6)
        
        # Test surface_area_sc()
        area_calc = dbm_f.surface_area_sc(de, theta_w_calc)
        assert_approx_equal(area_calc, area, significant = 6)

def density(T, P, mass, Mol_wt, Pc, Tc, omega, delta, rho):
    """
    Test the functions for computing the mixture density in the fortran 
    file dbm_eos.f95.
    
    """
    
    # Test density()
    rho_calc = dbm_f.density(T, P, mass, Mol_wt, Pc, Tc, omega, delta)[0]
    assert_approx_equal(rho_calc, rho, significant = 6)
    
    rho_calc = dbm_f.density(T, P, mass, Mol_wt, Pc, Tc, omega, delta)[1]
    assert_approx_equal(rho_calc, rho, significant = 6)

def fugacity(T, P, mass, Mol_wt, Pc, Tc, omega, delta, f):
    """
    Test the functions for computing the mixture fugacity in the fortran 
    file dbm_eos.f95
    
    """
    
    # Test fugacity()
    f_calc = dbm_f.fugacity(T, P, mass, Mol_wt, Pc, Tc, omega, delta)
    for i in range(len(mass)):
        for j in range(2):
            assert_approx_equal(f_calc[j,i], f[j,i], significant = 6)

def solubility(T, P, S, mass, Mol_wt, Pc, Tc, omega, delta, kh_0, dH_solR, 
               nu_bar, Cs):
    """
    Test the functions for computing the solubility of a mixture from the
    gas phase into seawater in the fortran file dbm_eos.f95
    
    """
    
    # Test sw_solubility()
    f = dbm_f.fugacity(T, P, mass, Mol_wt, Pc, Tc, omega, delta)
    kh = dbm_f.kh_insitu(T, P, S, kh_0, dH_solR, nu_bar)
    Cs_calc = dbm_f.sw_solubility(f[0,:], kh)
    for i in range(len(mass)):
        assert_approx_equal(Cs_calc[i], Cs[i], significant = 6)

def mixture_obj_funcs(dbm_obj, m, T, P, S, T_m, mk, y, yk, rho_m, fug, D, Cs):
    """
    Test that the methods defined for a FluidMixture return the correct 
    results as passed through the above list.
    
    """
    m_calc = dbm_obj.masses(y)
    assert_array_almost_equal(m_calc, m, decimal = 4)
    mk_calc = dbm_obj.mass_frac(y)
    assert_array_almost_equal(mk_calc, mk, decimal = 4)
    y_calc = dbm_obj.moles(m)
    assert_array_almost_equal(y_calc, y, decimal = 4)
    yk_calc = dbm_obj.mol_frac(m)
    assert_array_almost_equal(yk_calc, yk, decimal = 4)
    Pk_calc = dbm_obj.partial_pressures(m, P)
    assert_array_almost_equal(Pk_calc, P * yk, decimal = 4)
    rho_m_calc = dbm_obj.density(m, T, P)
    assert_approx_equal(rho_m_calc[0], rho_m, significant = 6)
    assert_approx_equal(rho_m_calc[1], rho_m, significant = 6)
    fug_calc = dbm_obj.fugacity(m, T, P)
    assert_array_almost_equal(fug_calc, fug, decimal = 4)
    Cs_calc = dbm_obj.solubility(m, T, P, S)
    assert_array_almost_equal(Cs_calc[0,:], Cs, decimal = 4)
    assert_array_almost_equal(Cs_calc[1,:], Cs, decimal = 4)    
    D_calc = dbm_obj.diffusivity(T)
    assert_array_almost_equal(D_calc, D, decimal = 4)

def particle_obj_funcs(dbm_obj, m, T, P, S, T_m, mk, y, yk, rho_p, fug, D, 
                       Cs, de, us, shape, A, beta, beta_T):
    """
    Test that the methods defined for a FluidMixture return the correct 
    results as passed through the above list.
    
    """
    m_calc = dbm_obj.masses(y)
    assert_array_almost_equal(m_calc, m, decimal = 4)
    
    mk_calc = dbm_obj.mass_frac(y)
    assert_array_almost_equal(mk_calc, mk, decimal = 4)
    
    y_calc = dbm_obj.moles(m)
    assert_array_almost_equal(y_calc, y, decimal = 4)
    
    yk_calc = dbm_obj.mol_frac(m)
    assert_array_almost_equal(yk_calc, yk, decimal = 4)
    
    Pk_calc = dbm_obj.partial_pressures(m, P)
    assert_array_almost_equal(Pk_calc, P * yk, decimal = 4)
    
    rho_p_calc = dbm_obj.density(m, T, P)
    assert_approx_equal(rho_p_calc, rho_p, significant = 6)
    
    fug_calc = dbm_obj.fugacity(m, T, P)
    assert_array_almost_equal(fug_calc, fug[0,:], decimal = 4)
    
    Cs_calc = dbm_obj.solubility(m, T, P, S)
    assert_array_almost_equal(Cs_calc, Cs, decimal = 4)
    
    D_calc = dbm_obj.diffusivity(T)
    assert_array_almost_equal(D_calc, D, decimal = 4)
    
    m_calc = dbm_obj.masses_by_diameter(de, T, P, yk)
    assert_array_almost_equal(m_calc, m, decimal = 4)
    
    de_calc = dbm_obj.diameter(m, T, P)
    assert_approx_equal(de_calc, de, significant = 6)
    
    us_calc = dbm_obj.slip_velocity(m, T, P, S, T)
    assert_approx_equal(us_calc, us, significant = 6)
    
    shape_calc, de_x, rho_p_x, rho_x, mu_x, sigma_x = \
        dbm_obj.particle_shape(m, T, P, S, T)
    assert shape_calc == shape
    
    A_calc = dbm_obj.surface_area(m, T, P, S, T)
    assert_approx_equal(A_calc, A, significant = 6)
    
    beta_calc = dbm_obj.mass_transfer(m, T, P, S, T)
    assert_array_almost_equal(beta_calc, beta, decimal = 4)
    
    beta_T_calc = dbm_obj.heat_transfer(m, T, P, S, T)
    assert_array_almost_equal(beta_T_calc, beta_T, decimal = 4)
    
    shape_c, de_c, rho_p_c, us_c, A_c, Cs_c, beta_c, beta_T_c = \
        dbm_obj.return_all(m, T, P, S, T)
    assert shape_c == shape_calc
    assert de_c == de_calc
    assert rho_p_c == rho_p_calc
    assert us_c == us_calc
    assert A_c == A_calc
    assert_array_equal(Cs_c, Cs_calc)
    assert_array_equal(beta_c, beta_calc)
    assert_array_equal(beta_T_c, beta_T_calc)

def inert_obj_funcs(oil, T, P, Sa, Ta, rho_p, de, m, shape, us, A, beta_T):
    """
    Test that the methods defined for an inert fluid particle return the 
    correct results as passed through the above list.
    """
    rho_p_calc = oil.density(T, P, Sa, Ta)
    assert_approx_equal(rho_p_calc, rho_p, significant = 6)
    
    de_calc = oil.diameter(m, T, P, Sa, Ta)
    assert_approx_equal(de_calc, de, significant = 6)
    
    m_calc = oil.mass_by_diameter(de, T, P, Sa, Ta)
    assert_approx_equal(m_calc, m, significant = 6)
    
    shape_calc, de_x, rho_p_x, rho_x, mu_x, sigma_x = \
        oil.particle_shape(m, T, P, Sa, Ta)
    assert_approx_equal(shape_calc, shape, significant = 6)
    
    us_calc = oil.slip_velocity(m, T, P, Sa, Ta)
    assert_approx_equal(us_calc, us, significant = 6)
    
    A_calc = oil.surface_area(m, T, P, Sa, Ta)
    assert_approx_equal(A_calc, A, significant = 6)
    
    beta_T_calc = oil.heat_transfer(m, T, P, Sa, Ta)
    assert_approx_equal(beta_T_calc, beta_T, significant = 6)

# ----------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------

def test_sphere():
    """
    This function sets up the parameters for a spherical air bubble in water
    and runs through the full suite of tests defined in:
        slip_velocity()
        mass_transfer()
        density()
        fugacity()
        solubility()
        mixture_obj_funcs()
        
    """
    # Set up the input parameters for the dbm_f functions
    de = 0.0001
    rho_p = 1.32138841912
    rho = 998.257215941
    sigma = 0.07275
    mu = 0.00100012110033
    D = np.array([2.50541350979e-09, 2.22998583836e-09, 2.22596225609e-09,
                  5.18756387566e-09])
    T = 293.15
    P = 111117.85929002732
    S = 0.0
    mass = np.array([5.22489199e-13, 1.60097383e-13, 8.91233206e-15,
                  3.78443750e-16])
    Mol_wt = np.array([0.0280134, 0.0319988, 0.039948, 0.0440098])
    Pc = np.array([3399806.156, 5042827.464, 4870458.464, 7384287.96 ])
    Tc = np.array([126.2, 154.57777778, 150.81666667, 304.21111111])
    omega = np.array([0.0372, 0.0216, -0.004, 0.2667])
    delta = np.zeros((4,4))
    kh_0 = np.array([1.74176580e-07, 4.10544683e-07, 5.51958549e-07,
                     1.47676605e-05])
    dH_solR = np.array([1300., 1650., 1300., 2400.])
    nu_bar = np.array([3.30000000e-05, 3.20000000e-05, 3.30000000e-05,
                       3.30000000e-05])
    
    # Specify the correct outputs for the governing variables, shape, and 
    # slip velocity
    Eo = 0.00134432
    M = 2.55013e-11
    Re = 0.51401578378930768
    H = 0.0669038
    shape = 'sphere'
    us = 0.0051497552239961834
    theta_w = np.NaN
    area = np.NaN
    beta = np.array([0.00016578604765643502, 0.00015249575958119796, 
                     0.00015229853037034425, 0.00028089050557286602])
    beta_T = 0.003755119084887344
    fug = np.array([[8.67166530e+04, 2.32493850e+04, 1.03670859e+03,
                     3.98301694e+01], 
                    [8.67166530e+04,2.32493850e+04, 1.03670859e+03,
                     3.98301694e+01]])
    Cs = np.array([0.016267937144682276, 0.010488404509247566, 
                   0.000616315915439951, 0.0006746724569526987])
    
    # Test whether the slip velocity functions produce the correct outputs
    slip_velocity(de, rho_p, rho, sigma, mu, Eo, M, H, shape, us, Re)
    
    # Test whether the mass transfer functions produce the correct outputs
    mass_transfer(de, rho_p, rho, sigma, mu, D, theta_w, area, beta)
    
    # Test whether the density calculations are correct
    density(T, P, mass, Mol_wt, Pc, Tc, omega, delta, rho_p)
    
    # Test whether the fugacity calculations are correct
    fugacity(T, P, mass, Mol_wt, Pc, Tc, omega, delta, fug)
    
    # Test whether the solubility calculations are correct
    solubility(T, P, S, mass, Mol_wt, Pc, Tc, omega, delta, kh_0, dH_solR, 
               nu_bar, Cs)
    S = 35.
    Cs = np.array([0.013819958114035266, 0.008910122389559611, 
                   0.0005235734587752693, 0.0005731485819440912])
    solubility(T, P, S, mass, Mol_wt, Pc, Tc, omega, delta, kh_0, dH_solR, 
               nu_bar, Cs)
    
    # Test function wrapping in the dbm objects
    air = dbm.FluidMixture(['nitrogen', 'oxygen', 'argon', 'carbon_dioxide'])
    S = 0.
    Cs = np.array([0.0162892182087, 0.0104929588798, 0.000616406296083,
                   0.000674762281553])
    mk = mass / np.sum(mass)
    y = mass / Mol_wt
    yk = y / np.sum(y)
    mixture_obj_funcs(air, mass, T, P, S, T, mk, y, yk, rho_p, fug, D, Cs)
    
    bub = dbm.FluidParticle(['nitrogen', 'oxygen', 'argon', 'carbon_dioxide'])
    A = np.pi * de**2
    particle_obj_funcs(bub, mass, T, P, S, T, mk, y, yk, rho_p, fug, D, Cs, 
                       de, us, 1, A, beta, beta_T)

def test_ellipsoid():
    """
    This function sets up the parameters for an ellipsoidal air bubble in 
    water and runs through the full suite of tests defined in:
        slip_velocity()
        mass_transfer()
        density()
        fugacity()
        solubility()
        
    """
    # Set up the input parameters for the dbm_f functions
    de = 0.00055
    rho_p = 1.32138841912
    rho = 998.257215941
    sigma = 0.07275
    mu = 0.00100012110033
    D = np.array([2.50541350979e-09, 2.22998583836e-09, 2.22596225609e-09,
                  5.18756387566e-09])
    T = 293.15
    P = 111117.85929002732
    S = 0.
    mass = np.array([8.69291405e-11, 2.66362021e-11, 1.48278925e-12,
                     6.29635790e-14])
    Mol_wt = np.array([0.0280134, 0.0319988, 0.039948, 0.0440098])
    Pc = np.array([3399806.156, 5042827.464, 4870458.464, 7384287.96 ])
    Tc = np.array([126.2, 154.57777778, 150.81666667, 304.21111111])
    omega = np.array([0.0372, 0.0216, -0.004, 0.2667])
    delta = np.zeros((4,4))
    kh_0 = np.array([1.74176580e-07, 4.10544683e-07, 5.51958549e-07,
                     1.47676605e-05])
    dH_solR = np.array([1300., 1650., 1300., 2400.])
    nu_bar = np.array([3.30000000e-05, 3.20000000e-05, 3.30000000e-05,
                       3.30000000e-05])
    
    # Specify the correct outputs for the governing variables, shape, and 
    # slip velocity
    Eo = 0.0406657
    M = 2.55013e-11
    Re = 28.255026439940735
    H = 2.02384
    shape = 'ellipsoid'
    us = 0.051468695427127696
    theta_w = np.NaN
    area = np.NaN
    beta = np.array([0.000136620688077, 0.000126254013204, 0.000126099657194,
                     0.0002239804085])
    beta_T = 0.0022538521067439303
    fug = np.array([[8.67166530e+04, 2.32493850e+04, 1.03670859e+03,
                     3.98301694e+01], 
                    [8.67166530e+04,2.32493850e+04, 1.03670859e+03,
                     3.98301694e+01]])
    Cs = np.array([0.016267937144682276, 0.010488404509247566, 
                   0.000616315915439951, 0.0006746724569526987])
    
    # Test whether the functions produce the correct outputs
    slip_velocity(de, rho_p, rho, sigma, mu, Eo, M, H, shape, us, Re)
    
    # Test whether the mass transfer functions produce the correct outputs
    mass_transfer(de, rho_p, rho, sigma, mu, D, theta_w, area, beta)
    
    # Test whether the density calculations are correct
    density(T, P, mass, Mol_wt, Pc, Tc, omega, delta, rho_p)
    
    # Test whether the fugacity calculations are correct
    fugacity(T, P, mass, Mol_wt, Pc, Tc, omega, delta, fug)
    
    # Test whether the solubility calculations are correct
    solubility(T, P, S, mass, Mol_wt, Pc, Tc, omega, delta, kh_0, dH_solR, 
               nu_bar, Cs)
    S = 35.
    Cs = np.array([0.013819958114035266, 0.008910122389559611, 
                   0.0005235734587752693, 0.0005731485819440912])
    solubility(T, P, S, mass, Mol_wt, Pc, Tc, omega, delta, kh_0, dH_solR, 
               nu_bar, Cs)
               
    # Test function wrapping in the dbm objects
    air = dbm.FluidMixture(['nitrogen', 'oxygen', 'argon', 'carbon_dioxide'])
    S = 0.
    Cs = np.array([0.016267937144552966, 0.0104929588798, 0.000616406296083,
                   0.000674762281553])
    mk = mass / np.sum(mass)
    y = mass / Mol_wt
    yk = y / np.sum(y)
    mixture_obj_funcs(air, mass, T, P, S, T, mk, y, yk, rho_p, fug, D, Cs) 
    
    bub = dbm.FluidParticle(['nitrogen', 'oxygen', 'argon', 'carbon_dioxide'])
    A = np.pi * de**2
    particle_obj_funcs(bub, mass, T, P, S, T, mk, y, yk, rho_p, fug, D, Cs, 
                       de, us, 2, A, beta, beta_T)

def test_spherical_cap():
    """
    This function sets up the parameters for a spherical cap air bubble in 
    water and runs through the full suite of tests defined in:
        slip_velocity()
        mass_transfer()
        density()
        fugacity()
        solubility()
    
    """
    # Set up the input parameters for the dbm_f functions
    de = 0.0123
    rho_p = 1.32138841912
    rho = 998.257215941
    sigma = 0.07275
    mu = 0.00100012110033
    D = np.array([2.50541350979e-09, 2.22998583836e-09, 2.22596225609e-09,
                  5.18756387566e-09])
    T = 293.15
    P = 111117.85929002732
    S = 0.
    mass = np.array([9.72282909e-07, 2.97919937e-07, 1.65846646e-08,
                     7.04233487e-10])
    Mol_wt = np.array([0.0280134, 0.0319988, 0.039948, 0.0440098])
    Pc = np.array([3399806.156, 5042827.464, 4870458.464, 7384287.96 ])
    Tc = np.array([126.2, 154.57777778, 150.81666667, 304.21111111])
    omega = np.array([0.0372, 0.0216, -0.004, 0.2667])
    delta = np.zeros((4,4))
    kh_0 = np.array([1.74176580e-07, 4.10544683e-07, 5.51958549e-07,
                     1.47676605e-05])
    dH_solR = np.array([1300., 1650., 1300., 2400.])
    nu_bar = np.array([3.30000000e-05, 3.20000000e-05, 3.30000000e-05,
                       3.30000000e-05])
    
    # Specify the correct outputs for the governing variables, shape, and 
    # slip velocity
    Eo = 20.3382
    M = 2.55013e-11
    Re = 3030.1491966325693
    H = 1012.19
    shape = 'spherical_cap'
    us = 0.24681356947709357
    theta_w = 0.872665369958
    area = 0.00080418635487435703
    beta = np.array([0.000196449205412, 0.000185336783495, 0.000185169505644,
                     0.000282678187328])
    beta_T = 0.0014996399347451304
    fug = np.array([[8.67166530e+04, 2.32493850e+04, 1.03670859e+03,
                     3.98301694e+01], 
                    [8.67166530e+04,2.32493850e+04, 1.03670859e+03,
                     3.98301694e+01]])
    Cs = np.array([0.016267937144682276, 0.010488404509247566, 
                   0.000616315915439951, 0.0006746724569526987])
    
    # Test whether the functions produce the correct outputs
    slip_velocity(de, rho_p, rho, sigma, mu, Eo, M, H, shape, us, Re)
    
    # Test whether the mass transfer functions produce the correct outputs
    mass_transfer(de, rho_p, rho, sigma, mu, D, theta_w, area, beta)
    
    # Test whether the density calculations are correct
    density(T, P, mass, Mol_wt, Pc, Tc, omega, delta, rho_p)
    
    # Test whether the fugacity calculations are correct
    fugacity(T, P, mass, Mol_wt, Pc, Tc, omega, delta, fug)
    
    # Test whether the solubility calculations are correct
    solubility(T, P, S, mass, Mol_wt, Pc, Tc, omega, delta, kh_0, dH_solR, 
               nu_bar, Cs)
    S = 35.
    Cs = np.array([0.013819958114035266, 0.008910122389559611, 
                   0.0005235734587752693, 0.0005731485819440912])
    solubility(T, P, S, mass, Mol_wt, Pc, Tc, omega, delta, kh_0, dH_solR, 
               nu_bar, Cs)
    
    # Test function wrapping in the dbm objects
    air = dbm.FluidMixture(['nitrogen', 'oxygen', 'argon', 'carbon_dioxide'])
    S = 0.
    Cs = np.array([0.0162892182087, 0.0104929588798, 0.000616406296083,
                   0.000674762281553])
    mk = mass / np.sum(mass)
    y = mass / Mol_wt
    yk = y / np.sum(y)
    mixture_obj_funcs(air, mass, T, P, S, T, mk, y, yk, rho_p, fug, D, Cs)
    
    bub = dbm.FluidParticle(['nitrogen', 'oxygen', 'argon', 'carbon_dioxide'])
    A = area
    particle_obj_funcs(bub, mass, T, P, S, T, mk, y, yk, rho_p, fug, D, Cs, 
                       de, us, 3, A, beta, beta_T)

def test_rigid():
    """
    This function tests the algorithms in the dbm object InsolubleParticle.
    This object does not use any new dbm_f functions; rather, it provides 
    capability for particles not defined by the Peng-Robinson equation of 
    state.
    
    """
    # Set up the inputs to the InsolubleParticle object
    isfluid = True
    iscompressible = False 
    rho_p = 870.
    gamma = 29., 
    beta = 0.0001 
    co= 1.0e-9
    
    # Set the thermodynamic state
    T = 293.15
    P = 101325.0 * 100.0
    Sa = 35.0
    Ta = 279.15
    
    # Test a default incompressible fluid particle
    oil = dbm.InsolubleParticle(isfluid, iscompressible=False)
    de = 0.001
    m = 4.86946861306418e-07
    shape = 1
    us = 0.018979716423733476
    A = 3.1415926535897963e-06
    beta_T = 0.0010500339198291767
    inert_obj_funcs(oil, T, P, Sa, Ta, 930., de, m, shape, us, A, beta_T)
    
    # Test a default oil-like particle
    oil = dbm.InsolubleParticle(isfluid, iscompressible=True)
    rho_p = 872.61692306815519
    de = 0.001
    m = 4.5690115248484099e-07
    shape = 1
    us = 0.022759095242958018
    A = 3.1415926535897963e-06
    beta_T = 0.0011171386749182551
    inert_obj_funcs(oil, T, 101325., 0., Ta, rho_p, de, m, shape, us, A, 
                    beta_T)
    
    # Test a default sand-like particle
    oil = dbm.InsolubleParticle(isfluid=False, iscompressible=False)
    de = 0.001
    m = 4.86946861306418e-07
    shape = 4
    us = 0.018979716423733476
    A = 3.1415926535897963e-06
    beta_T = 0.0010500339198291767
    inert_obj_funcs(oil, T, P, Sa, Ta, 930., de, m, shape, us, A, beta_T)
    
    # Test a user-defined oil-like particle
    oil = dbm.InsolubleParticle(isfluid=True, iscompressible=True, gamma=29.,
                                beta=0.0001, co=1.0e-9)
    rho_p = 889.27848851245381
    de = 0.02
    m = 0.0037250010220082142
    shape = 2
    us = 0.13492474667673468
    A = 0.0012566370614359179
    beta_T = 0.00042971102013021731
    inert_obj_funcs(oil, T, P, Sa, Ta, rho_p, de, m, shape, us, A, 
                    beta_T)

def test_cubic_solver():
    """
    This function test the cubic equation solver in the fortran file
    math_funcs.f95
    
    """
    
    # Test the Math_funcs solver for cubic equations
    p = np.array([1., -6., 11., -6.])
    r = np.array([1., 2., 3.])
    
    r_calc = np.sort(dbm_f.cubic_roots(p))
    for i in range(3):
        assert_approx_equal(r_calc[i], r[i], significant = 6)

def test_equilibrium():
    """
    Test the `FluidMixture.equilibrium` method for determining the gas/
    liquid partitioning at equilibrium.  This test follows the example 
    problem in McCain (1990), example problem 15-2, p. 431.
    
    """
    # Define the thermodynamic state
    T = 273.15 + 5./9. * (160. - 32.)
    P = 1000. * 6894.76
    
    # Define the properties of the complete mixture (gas + liquid)
    composition = ['methane', 'n-butane', 'n-decane']
    oil = dbm.FluidMixture(composition)
    yk = np.array([0.5301, 0.1055, 0.3644])
    m = oil.masses(yk)
    
    # Compute the mass partitioning at equilibrium
    (mk, xi, K) = oil.equilibrium(m, T, P)
    
    # Check the result
    mk_ans = np.array([[0.00586382, 0.00080885, 0.00011871],
                       [ 0.00264036, 0.00532313, 0.0517295 ]])
    assert_array_almost_equal(mk, mk_ans, decimal=6)
    xi_ans = np.array([[ 0.9612089 ,  0.03659695,  0.00219414],
                       [ 0.26555477,  0.1477816 ,  0.58666364]])
    assert_array_almost_equal(xi, xi_ans, decimal=6)
    K_ans = np.array([3.61962584, 0.24764217, 0.00374003])
    assert_array_almost_equal(K, K_ans, decimal=6)

