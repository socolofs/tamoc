"""
Unit tests for the `dbm_c` module of ``TAMOC``

These are the same tests as for the fortran :-)

Provides testing of the functions defined in the source code of the
`dbm_f` module.

`dbm_f` is a library created from three source-code files::

    dbm_eos.f95
    dbm_phys.f95
    math_funcs.f95

It can be compiled using::

    f2py -c -m dbm_f dbm_eos.f95 dbm_phys.f95 math_funcs.f95

This module tests direct calling of the ``dbm_f`` functions from Python.  Normal usage of these functions in ``TAMOC`` should be through the `dbm` module, which defines class objects that interface with these functions.

Notes
-----
All of the tests defined herein check the general behavior of each of the
programmed function--this is not a comparison against measured data. The
results of the hand calculations entered below as sample solutions have been
ground-truthed for their reasonableness. However, passing these tests only
means the programs and their interfaces are working as expected, not that they
have been validated against measurements.

"""
from __future__ import (absolute_import, division, print_function)

import pytest
pytestmark = pytest.mark.skipif(True, reason="WIP: not ready for full suite.")

from tamoc import dbm_c as dbm_f

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_approx_equal


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def base_state():
    """
    docstring for base_state

    """
    # Choose a thermodynamic state and composition
    T = 273.15 + 15.
    S = 0.0
    P = 101325. + 9.81 * 1000. * 10.
    composition = ['nitrogen', 'oxygen', 'argon', 'carbon_dioxide']
    yk = np.array([0.78084, 0.20946, 0.009340, 0.00036])

    # List the physical constants that should be in the ChemData
    # database
    Mol_wt = np.array([0.0280134, 0.0319988, 0.039948, 0.04401])
    Pc = np.array([3399806.156, 5042827.464, 4870458.464, 7373999.99902408])
    Tc = np.array([126.2, 154.57777778, 150.81666667, 304.12])
    Vc = np.array([9.01E-05, 7.34E-05, 7.46E-05, 0.00009407])
    Vb = np.array([0.0000348, 0.0000279, 0.00002856, 3.73000000e-05])
    omega = np.array([0.0372, 0.0216, -0.004, 0.225])
    delta = np.zeros((4,4))
    kh_0 = np.array([1.74176580e-07, 4.10544683e-07, 5.51958549e-07,
                     1.47433500e-05])
    neg_dH_solR = np.array([1300., 1650., 1300., 2368.988311])
    nu_bar = np.array([3.30000000e-05, 3.20000000e-05, 6.83500659e-06,
                       3.20000000e-05])
    Aij = np.zeros((15,15))
    Bij = np.zeros((15,15))
    delta_groups = np.zeros((4,15))
    calc_delta = -1
    K_salt = np.array([0.0001834, 0.000169, 0.0001694,
                       0.0001323])

    # Give the various thermodynamic properties at this state
    rho = 999.194667977339
    mu = 0.0011383697567284897
    rho_p = 2.4134402697361361
    Cs = np.array([0.03147573, 0.02070961, 0.00119098, 0.0013814])
    sigma = 0.041867943708892935
    mu_p = 1.8438154276225057e-05
    D = np.array([1.41620605e-09, 1.61034760e-09, 1.56772544e-09,
         1.35720048e-09])

    return (T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta,
            kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta,
            K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D)


# ----------------------------------------------------------------------------
# Test dbm_f module functions
# ----------------------------------------------------------------------------

def test_cubic_roots():
    """
    Test the cubic equation solver in the math_funcs module.
    """
    # Choose the coefficients of a polynomial
    p = np.array([1.3, 2.4, 0.6, 5.7])

    # Compute the solution and record the answer
    xr = dbm_f.cubic_roots(p)
    ans = np.array([-2.40971422+0.j, 0.28178023-1.31915051j,
        0.28178023+1.31915051j])

    # Check the solution
    assert_array_almost_equal(xr, ans, decimal=6)


def test_eotvos():
    """
    Compute the Eotvos number using the function in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the Eotvos number and record the solution
    de = 0.005
    Eo = dbm_f.eotvos(de, rho_p, rho, sigma)
    ans = 5.838848876721049

    # Check the solution
    assert_approx_equal(Eo, ans, significant=6)

def test_morton():
    """
    Compute the Morton number using the function in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the Morton number and record the solution
    Mo = dbm_f.morton(rho_p, rho, mu, sigma)
    ans = 2.2410788876702557e-10

    # Check the solution
    assert_approx_equal(Mo, ans, significant=6)

def test_reynolds():
    """
    Compute the Reynolds number using the function in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the Reynolds number and record the solution
    de = 0.005
    us = 0.215
    Re = dbm_f.reynolds(de, us, rho, mu)
    ans = 943.5723864999244

    # Check the solution
    assert_approx_equal(Re, ans, significant=6)

def test_h_parameter():
    """
    Compute the h-parameter for evaluating bubble shape from Clift et al.
    (1978) using the function in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the h-parameter and record the solution
    Eo = 5.838848876721049
    M = 2.2410788876702557e-10
    h = dbm_f.h_parameter(Eo, M, mu)
    ans = 206.42492684498671

    # Check the solution
    assert_approx_equal(h, ans, significant=6)

def test_particle_shape():
    """
    Test the particle_shape() function in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Check the particle shape for different bubble sizes
    de = 0.0004
    shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
    ans = 1
    assert shape == ans

    de = 0.0005
    shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
    ans = 2
    assert shape == ans

    de = 0.02
    shape = dbm_f.particle_shape(de, rho_p, rho, mu, sigma)
    ans = 3
    assert shape == ans

def test_theta_w_sc():
    """
    Compute theta_w for a spherical cap bubble using the function in dbm_phys
    module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute theta_w for a spherical-cap bubble
    de = 0.05
    us = 0.4973521249888033
    theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
    ans = 0.8726646259971722

    # Check the solution
    assert_approx_equal(theta_w, ans, significant=6)

def test_surface_area_sc():
    """
    Compute the surface area of a spherical cap bubble using the function in
    the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the surface area for a spherical-cap bubble
    de = 0.05
    theta_w = 0.8726646259971722
    surface_area = dbm_f.surface_area_sc(de, theta_w)
    ans = 0.013288829280658505

    # Check the solution
    assert_approx_equal(surface_area, ans, significant=6)

def test_surface_area_sphere():
    """
    Compute the surface area of a sphere using the function in the dbm_phys
    module.
    """
    # Compute the surface area for a spherical bubble
    de = 0.0005
    surface_area = dbm_f.surface_area_sphere(de)
    ans = 7.853981633974482e-07

    # Check the solution
    assert_approx_equal(surface_area, ans, significant=6)

def test_us_sphere():
    """
    Compute the slip velocity of a sphere using the function in the dbm_phys
    module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute and the slip velocity for a spherical bubble
    de = 0.0004
    us = dbm_f.us_sphere(de, rho_p, rho, mu)
    ans = 0.0383009640907029

    # Check the solution
    assert_approx_equal(us, ans, significant=6)

def test_us_ellipsoid():
    """
    Compute the slip velocity of an ellipsoid using the function in the
    dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute and check the slip velocity for a clean ellipsoidal bubble
    de = 0.001
    us = dbm_f.us_ellipsoid(de, rho_p, rho, mu_p, mu, sigma, 1)
    ans = 0.3511735400683849
    assert_approx_equal(us, ans, significant=6)

    # Compute and check the slip velocity for a dirty ellipsoidal bubble
    de = 0.001
    us = dbm_f.us_ellipsoid(de, rho_p, rho, mu_p, mu, sigma, -1)
    ans = 0.1183154621133231
    assert_approx_equal(us, ans, significant=6)

def test_us_spherical_cap():
    """
    Compute the slip velocity of a spherical cap bubble using the function
    in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute and the slip velocity for a spherical bubble
    de = 0.05
    us = dbm_f.us_spherical_cap(de, rho_p, rho)
    ans = 0.4973521249888033

    # Check the solution
    assert_approx_equal(us, ans, significant=6)

def test_xfer_kumar_hartland():
    """
    Compute the mass transfer coefficient using the equation from Kumar and
    Hartland and the function in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the mass transfer coefficients from Kumar and Hartland
    de = 0.001
    us = 0.3511735400683849
    beta = dbm_f.xfer_kumar_hartland(de, us, rho, mu, D, sigma, mu_p, 4)
    ans = np.array([0.00071983, 0.00076744, 0.00075723, 0.00070475])

    # Check the solution
    assert_array_almost_equal(beta, ans, decimal=6)

def test_xfer_johnson():
    """
    Compute the mass transfer coefficient using the equation from Johnson
    et al. and the function in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute and the mass transfer coefficients from Johnson
    de = 0.001
    us = 0.3511735400683849
    beta = dbm_f.xfer_johnson(de, us, D, 4)
    ans = np.array([0.00036758, 0.00039197, 0.00038675, 0.00035984])

    # Check the solution
    assert_array_almost_equal(beta, ans, decimal=6)

def test_xfer_clift():
    """
    Compute the mass transfer coefficient using equations in Clift et al. and
    the function in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute and the mass transfer coefficients from Clift et al.
    de = 0.0004
    us = 0.0383009640907029
    beta = dbm_f.xfer_clift(de, us, rho, mu, D, 4)
    ans = np.array([9.91108368e-05, 1.08142556e-04, 1.06190553e-04,
        9.62898245e-05])

    # Check the solution
    assert_array_almost_equal(beta, ans, decimal=6)

def test_xfer_sphere():
    """
    Get the mass transfer coefficient for a sphere using the functions in the
    dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute and check the mass transfer coefficients for a clean sphere
    de = 0.0004
    us = 0.0383009640907029
    beta = dbm_f.xfer_sphere(de, us, rho, mu, D, sigma, mu_p, 0, 1, 4)
    ans = np.array([0.00012297, 0.00013113, 0.00012939, 0.00012039])
    assert_array_almost_equal(beta, ans, decimal=6)

    # Compute and check the mass transfer coefficients for a dirty sphere
    de = 0.0004
    us = 0.0383009640907029
    beta = dbm_f.xfer_sphere(de, us, rho, mu, D, sigma, mu_p, 0, -1, 4)
    ans = np.array([9.91108368e-05, 1.08142556e-04, 1.06190553e-04,
        9.62898245e-05])
    assert_array_almost_equal(beta, ans, decimal=6)

def test_xfer_ellipsoid():
    """
    Get the mass transfer coefficient for an ellipsoidal bubble using the
    functions in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute and check the mass transfer coefficients for a clean
    # ellipsoidal bubble
    de = 0.001
    us = 0.3511735400683849
    beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, D, sigma, mu_p, 0, 1, 4)
    ans = np.array([0.00036758, 0.00039197, 0.00038675, 0.00035984])
    assert_array_almost_equal(beta, ans, decimal=6)

    # Compute and check the mass transfer for a dirty ellipsoidal bubble
    de = 0.001
    us = 0.1183154621133231
    beta = dbm_f.xfer_ellipsoid(de, us, rho, mu, D, sigma, mu_p, 0, -1, 4)
    ans = np.array([8.99773552e-05, 9.80906597e-05, 9.63380316e-05,
        8.74410374e-05])
    assert_array_almost_equal(beta, ans, decimal=6)

def test_xfer_spherical_cap():
    """
    Get the mass transfer coefficient for a spherical cap bubble using the
    functions in the dbm_phys module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute and check the mass transfer coefficientsfor a clean spherial
    # cap bubble
    de = 0.05
    us = 0.4973521249888033
    beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, D, 1, 4)
    ans = np.array([0.00024905, 0.00026557, 0.00026204, 0.00024381])
    assert_array_almost_equal(beta, ans, decimal=6)

    # Compute and check the mass transfer coefficients for a dirty spherical
    # cap bubble
    de = 0.05
    us = 0.4973521249888033
    beta = dbm_f.xfer_spherical_cap(de, us, rho, rho_p, mu, D, -1, 4)
    ans = np.array([0.00010399, 0.00011089, 0.00010941, 0.0001018 ])
    assert_array_almost_equal(beta, ans, decimal=6)

def test_density():
    """
    Compute the fluid density using the function in the dbm_eos module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the density
    mass = np.array([9.54296597e-13, 2.92408701e-13, 1.62778641e-14,
        6.91208976e-16])
    density = dbm_f.density(T, P, mass, Mol_wt, Pc, Tc, Vc, omega, delta,
        Aij, Bij, delta_groups, calc_delta)

    # Check the solution
    assert density.shape[0] == 2
    assert density.shape[1] == 1
    assert density[0,0] == density[1,0]
    assert_approx_equal(density[0,0], rho_p, significant=6)

def test_fugacity():
    """
    Compute the component fugacities using the function in the dbm_eos
    module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the fugacity
    mass = np.array([9.54296597e-13, 2.92408701e-13, 1.62778641e-14,
        6.91208976e-16])
    fugacity = dbm_f.fugacity(T, P, mass, Mol_wt, Pc, Tc, omega, delta,
        Aij, Bij, delta_groups, calc_delta)
    ans = np.array([[1.55545112e+05, 4.16841748e+04, 1.85873462e+03,
        7.12092269e+01], [1.55545112e+05, 4.16841748e+04, 1.85873462e+03,
        7.12092269e+01]])

    # Check the solution
    assert fugacity.shape[0] == 2
    assert fugacity.shape[1] == 4
    assert_array_almost_equal(fugacity[0,:], fugacity[1,:], decimal=6)
    for i in range(4):
        assert_approx_equal(fugacity[0,i], ans[0,i], significant=6)

    # TODO:   Figure out why this does not work:
    # assert_array_almost_equal(fugacity[0,:], ans[0,:], decimal=6)

def test_volume_trans():
    """
    Compute the volume-translation correction using the function in the
    dbm_eos module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the volume translation correction to density
    mass = np.array([9.54296597e-13, 2.92408701e-13, 1.62778641e-14,
        6.91208976e-16])
    vol_trans = dbm_f.volume_trans(T, P, mass, Mol_wt, Pc, Tc, Vc)
    ans = np.array([-4.19831425e-06, -3.18588657e-06, -3.37047932e-06,
        3.66998629e-06])

    # Check the solution
    assert_array_almost_equal(vol_trans, ans, decimal=6)

def test_z_pr():
    """
    Compute the z-factor in the Peng-Robinson equation of state using the
    function in the dbm_eos module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the z factor
    mass = np.array([9.54296597e-13, 2.92408701e-13, 1.62778641e-14,
        6.91208976e-16])
    z, A, B, Ap, Bp, z_yk = dbm_f.z_pr(T, P, mass, Mol_wt, Pc, Tc, omega,
        delta, Aij, Bij, delta_groups, calc_delta)
    z_ans = np.array([[0.99867356],[0.99867356]])
    A_ans = 0.003263624928010674
    B_ans = 0.0019227426778496003
    Ap_ans = np.array([1.95713762, 2.14915791, 2.15406876, 4.18622307])
    Bp_ans = np.array([1.03950027, 0.85840524, 0.86715933, 1.15494488])

    # Check the solution
    assert z.shape[0] == 2
    assert z.shape[1] == 1
    assert_array_almost_equal(z, z_ans, decimal=6)
    assert_approx_equal(A, A_ans, significant=6)
    assert_approx_equal(B, B_ans, significant=6)
    assert_array_almost_equal(Ap, Ap_ans, decimal=6)
    assert_array_almost_equal(Bp, Bp_ans, decimal=6)
    assert_array_almost_equal(z_yk, yk, decimal=6)

def test_coefs():
    """
    Compute the mixture coefficients for the Peng-Robinson equation of
    state using the function in the dbm_eos module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the coefficients for the Peng-Robinson equation of state
    mass = np.array([9.54296597e-13, 2.92408701e-13, 1.62778641e-14,
        6.91208976e-16])
    A, B, Ap, Bp, z_yk = dbm_f.coefs(T, P, mass, Mol_wt, Pc, Tc, omega,
        delta, Aij, Bij, delta_groups, calc_delta)
    A_ans = 0.003263624928010674
    B_ans = 0.0019227426778496003
    Ap_ans = np.array([1.95713762, 2.14915791, 2.15406876, 4.18622307])
    Bp_ans = np.array([1.03950027, 0.85840524, 0.86715933, 1.15494488])

    # Check the solution
    assert_approx_equal(A, A_ans, significant=6)
    assert_approx_equal(B, B_ans, significant=6)
    assert_array_almost_equal(Ap, Ap_ans, decimal=6)
    assert_array_almost_equal(Bp, Bp_ans, decimal=6)
    assert_array_almost_equal(z_yk, yk, decimal=6)

def test_mole_fraction():
    """
    Convert mass fraction to mole fraction using the function in the
    dbm_eos module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the mole fractions from mass fraction
    mass = np.array([9.54296597e-13, 2.92408701e-13, 1.62778641e-14,
        6.91208976e-16])
    z_yk = dbm_f.mole_fraction(mass, Mol_wt)

    # Check the solution
    assert_array_almost_equal(z_yk, yk, decimal=6)

def test_viscosity():
    """
    Compute the fluid viscosity using the function in the dbm_eos module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the viscosity
    mass = np.array([9.54296597e-13, 2.92408701e-13, 1.62778641e-14,
        6.91208976e-16])
    mu = dbm_f.viscosity(T, P, mass, Mol_wt, Pc, Tc, Vc, omega, delta,
        Aij, Bij, delta_groups, calc_delta)
    ans = np.array([[1.84381543e-05], [1.84381543e-05]])

    # Check the solution
    assert mu.shape[0] == 2
    assert mu.shape[1] == 1
    assert_array_almost_equal(mu, ans, decimal=6)

def test_kh_insitu():
    """
    Compute the in situ Henry's coefficient using the function in the
    dbm_eos module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the Henry's coefficients at in situ conditions
    kh = dbm_f.kh_insitu(T, P, S, kh_0, neg_dH_solR, nu_bar, Mol_wt, K_salt,
        4)
    ans = np.array([2.02357544e-07, 4.96821922e-07, 6.41950275e-07,
        1.93991261e-05])

    # Check the solution
    assert_array_almost_equal(kh, ans, decimal=6)

def test_sw_solubility():
    """
    Compute the component solubilities from the function in the dbm_eos
    module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the solubilities at in situ conditions
    f = np.array([1.55545112e+05, 4.16841748e+04, 1.85873462e+03,
        7.12092269e+01])
    kh = np.array([2.02357544e-07, 4.96821922e-07, 6.41950275e-07,
        1.93991261e-05])
    Cs = dbm_f.sw_solubility(f, kh, 4)
    ans = np.array([0.03147573, 0.02070961, 0.00119322, 0.0013814 ])

    # Check the solution
    assert_array_almost_equal(Cs, ans, decimal=6)

def test_diffusivity():
    """
    Compute the component diffusivities in seawater using the function in the
    dbm_eos module.
    """
    # Load the base thermodynamic state
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, mu_p, D = base_state()

    # Compute the diffusivities at in situ conditions
    D_fun = dbm_f.diffusivity(mu, Vb, 4)
    ans = np.array([0.03147573, 0.02070961, 0.00119322, 0.0013814 ])

    # Check the solution
    assert_array_almost_equal(D_fun, D, decimal=6)

def test_Kvsi_hydrate():
    """
    Compute the hydrate stability using the K_vsi method and the function in
    the dbm_eos module.
    """
    # Load the base thermodynamic state
    T = 273.15 + 4.14
    P = 101325. * 100.

    # Check hydrate stabilty using the K_vsi method
    mass = np.array([0., 0., 0., 0., 0., 9.54296597e-13, 6.91208976e-16, 0.])
    K_vsi, yk_f = dbm_f.kvsi_hydrate(T, P, mass)
    ans = np.array([3.45832745e-01, 9.72276343e-03, 1.03133054e-02,
        5.95436244e-03, 2.63753287e+00, 1.60492007e+00, 1.02531488e+00,
        2.40385659e-03])
    yk_ans = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 9.99539168e-01, 4.60831586e-04,
        0.00000000e+00])

    # Check the solution
    assert_array_almost_equal(K_vsi, ans, decimal=6)
    assert_array_almost_equal(yk_f, yk_ans, decimal=6)


