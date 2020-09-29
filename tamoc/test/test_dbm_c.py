"""
Unit tests for the `dbm` module of ``TAMOC``

THIS is a copy of the test_dbm written to test the Fortran code.

This version tests the Cython version instead.

Making a copy, as this is an experiment at this point.

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
# Cython port by Christopher Barker, May 2010 Chris.Barker@noaa.gov

from __future__ import (absolute_import, division, print_function)

from tamoc import dbm_c  # for the ones that are new
from tamoc import dbm_c as dbm_f

from tamoc import dbm

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_approx_equal

import pytest

# ------------------------------------------------------------
# Test the individual functions that are in the Cython version
# ------------------------------------------------------------


def test_trial_nparray():
    result = dbm_c.trial_nparray(4,
                                 np.array([1, 2, 3, 4], dtype=np.float64),
                                 np.arange(12, dtype=np.float64).reshape((3,4)),
                                 )

    assert result == 10


def test_sum():
    assert dbm_c.sum(np.array((1.2, 1.3, 1.4))) == 3.9


def test_sum_mult():
    arr1 = np.array((1, 2, 3), dtype=np.float64)
    arr2 = np.array((2, 3, 4), dtype=np.float64)
    assert dbm_c.sum_mult(arr1, arr2) == (arr1 * arr2).sum()


## kinda overkill, but nice for testing if we change the code
@pytest.mark.parametrize(
    "mass, Mol_wt, expected",
    [((1.0, 2.0, 3.0), (78.11, 28.05, 32.12), (0.072125, 0.401689, 0.526186)),
     ((2.0, 2.0), (10, 10), (0.5, 0.5)),
     ((1.1,), (234.15,), (1.0,)),  # one component -- should be 100%
     ])
def test_mole_fraction(mass, Mol_wt, expected):
    mass = np.array(mass, dtype=np.float64)
    Mol_wt = np.array(Mol_wt, dtype=np.float64)

    mf = dbm_c.mole_fraction(mass, Mol_wt)

    print(mf)

    # Note: this has not been checked!
    assert_array_almost_equal(mf, expected)


def test_mole_fraction_wrong_size():
    mass = np.array((1, 2, 3), dtype=np.float64)
    Mol_wt = np.array((1, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        mf = dbm_c.mole_fraction(mass, Mol_wt)


def test_eotvos():
    """
    not really sure what I should get here, but at least it works
    """
    # are these expected numbers?
    Eo = dbm_c.eotvos(0.1, 0.8, 1.2, 3.0)

    # no idea if this is correct!
    assert Eo == 0.013080000000000001


def test_morton():
    # no idea if these are reasonable
    M = dbm_c.morton(0.8, 1.2, 0.15, 1.5)

    assert M == 0.00040875


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

    # Give the properties of seawater at this state
    rho = 999.194667977339
    mu = 0.0011383697567284897

    # Give the particle properties that should come back from the dbm_f
    # and dbm object function calls
    rho_p = 2.4134402697361361
    Cs = np.array([0.03147573, 0.02070961, 0.00119098, 0.0013814 ])
    sigma = 0.041867943708892935

    # At seawater salinity:
    S_S = 35.
    rho_S = 1026.0612427871006
    mu_S = 0.001220538626793113

    # Give the particle properties that should come back from the dbm_f
    # and dbm object function calls
    rho_p_S = 2.4134402697361361
    Cs_S = np.array([ 0.02535491, 0.01696806, 0.00097535, 0.00118188])
    sigma_S = 0.043023878167355374

    # Also, give the fortran function answers for this state
    mu_p = 1.8438154276225057e-05
    D = np.array([1.41620605e-09, 1.61034760e-09, 1.56772544e-09,
         1.35720048e-09])
    D_S = np.array([1.30803909e-09, 1.48735249e-09, 1.44798573e-09,
         1.25354024e-09])

    return (T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta,
            kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta,
            K_salt, rho, mu, rho_p, Cs, sigma, S_S, rho_S, mu_S, rho_p_S,
            Cs_S, sigma_S, mu_p, D, D_S)


def particle_obj_funcs(obj, mass, yk, Mol_wt, fp_type, T, P, Sa, Ta, rho_p,
                       us, A, Cs, beta, beta_T, de, shape, sigma):
    """
    Test that the methods defined for a FluidParticle return the correct
    results as passed through the above list.

    """
    # Compute some of the composition data
    y = mass / Mol_wt
    mf = mass / np.sum(mass)

    # Test the particle composition data
    assert_array_almost_equal(mass, obj.masses(y), decimal = 4)
    assert_array_almost_equal(mf, obj.mass_frac(y), decimal = 4)
    assert_array_almost_equal(y, obj.moles(mass), decimal = 4)
    assert_array_almost_equal(yk, obj.mol_frac(mass), decimal = 4)
    assert_array_almost_equal(P * yk, obj.partial_pressures(mass, P),
        decimal = 4)

    # Test the particle attributes
    assert_approx_equal(rho_p, obj.density(mass, T, P), significant = 6)
    assert_approx_equal(us, obj.slip_velocity(mass, T, P, Sa, Ta),
        significant = 6)
    assert_approx_equal(A, obj.surface_area(mass, T, P, Sa, Ta),
        significant = 6)
    assert_array_almost_equal(Cs, obj.solubility(mass, T, P, Sa),
        decimal = 6)
    assert_array_almost_equal(beta, obj.mass_transfer(mass, T, P, Sa, Ta),
        decimal = 6)
    assert_array_almost_equal(beta_T, obj.heat_transfer(mass, T, P, Sa, Ta),
        decimal = 6)
    assert_approx_equal(de, obj.diameter(mass, T, P), significant = 6)
    assert_array_almost_equal(mass, obj.masses_by_diameter(de, T, P, yk),
        decimal = 6)
    assert_approx_equal(sigma, obj.interface_tension(mass, T, Sa, P),
        significant = 6)

    # Test that the return_all function agrees with the above
    shape_c, de_c, rho_p_c, us_c, A_c, Cs_c, beta_c, beta_T_c = \
        obj.return_all(mass, T, P, Sa, Ta)
    assert_approx_equal(rho_p, rho_p_c, significant = 6)
    assert_approx_equal(us, us_c, significant = 6)
    assert_approx_equal(A, A_c, significant = 6)
    assert_array_almost_equal(Cs, Cs_c, decimal = 4)
    assert_array_almost_equal(beta, beta_c, decimal = 4)
    assert_array_almost_equal(beta_T, beta_T_c, decimal = 4)
    assert_approx_equal(de, de_c, significant = 6)
    assert_approx_equal(shape_c, shape, significant = 6)


def particle_fortran_funcs(mass, T, P, Sa, Ta, Mol_wt, fp_type, Pc, Tc, Vc,
                           Vb, omega, delta, kh_0, neg_dH_solR, nu_bar, Aij,
                           Bij, delta_groups, calc_delta, K_salt, rho, mu,
                           sigma, shape, rho_p, us, A, Cs, beta, beta_T, de,
                           mu_p, D):
    """
    Test that the methods defined for in dbm_f are consistent with how they
    are used in the FluidParticle objects.
    """
    # Test the items in dbm_eos
    assert_approx_equal(rho_p, dbm_f.density(T, P, mass, Mol_wt, Pc, Tc, Vc,
        omega, delta, Aij, Bij, delta_groups, calc_delta)[fp_type, 0],
        significant = 6)
    if fp_type == 0:
        m_g = mass
        m_o = np.zeros(len(mass))
    else:
        m_g = np.zeros(len(mass))
        m_o = mass
    print("in particle_fortran_funcs, mu_p", mu_p)

    assert_approx_equal(mu_p,
                        dbm_f.viscosity(T, P, mass, Mol_wt, Pc, Tc, Vc,
                                        omega, delta, Aij, Bij, delta_groups,
                                        calc_delta)[fp_type, 0],
                        significant = 6)

    f = dbm_f.fugacity(T, P, mass, Mol_wt, Pc, Tc, omega, delta, Aij, Bij,
                       delta_groups, calc_delta)[fp_type, :]
    kh = dbm_f.kh_insitu(T, P, Sa, kh_0, neg_dH_solR, nu_bar,
                         Mol_wt, K_salt)
    assert_array_almost_equal(Cs, dbm_f.sw_solubility(f, kh), decimal = 4)
    assert_array_almost_equal(D, dbm_f.diffusivity(mu, Vb), decimal = 4)

    # Test the items in dbm_phys
    # expecting an integer, so should be exactly equal
    assert shape == dbm_f.particle_shape(de, rho_p, rho, mu, sigma)

    # if shape == 1:
    #     assert_approx_equal(us, dbm_f.us_sphere(de, rho_p, rho, mu),
    #         significant = 6)
    #     assert_approx_equal(A, dbm_f.surface_area_sphere(de),
    #         significant = 6)
    #     assert_array_almost_equal(beta, dbm_f.xfer_sphere(de, us, rho, mu, D,
    #         sigma, mu_p, fp_type, -1), decimal = 4)
    # elif shape == 2:
    #     assert_approx_equal(us, dbm_f.us_ellipsoid(de, rho_p, rho, mu_p,
    #         mu, sigma, -1), significant = 6)
    #     assert_approx_equal(A, dbm_f.surface_area_sphere(de),
    #         significant = 6)
    #     assert_array_almost_equal(beta, dbm_f.xfer_ellipsoid(de, us, rho, mu,
    #         D, sigma, mu_p, fp_type, -1), decimal = 4)
    # else:
    #     # this isn't getting run
    #     assert_approx_equal(us, dbm_f.us_spherical_cap(de, rho_p, rho),
    #         significant = 6)
    #     theta_w = dbm_f.theta_w_sc(de, us, rho, mu)
    #     assert_array_almost_equal(A, dbm_f.surface_area_sc(de, theta_w),
    #         decimal = 4)
    #     assert_array_almost_equal(beta, dbm_f.xfer_spherical_cap(de, us, rho,
    #         rho_p, mu, D, -1), decimal = 4)


# def inert_obj_funcs(oil, T, P, Sa, Ta, rho_p, de, m, shape, us, A, beta_T):
#     """
#     Test that the methods defined for an inert fluid particle return the
#     correct results as passed through the above list.
#     """
#     assert_approx_equal(oil.density(T, P, Sa, Ta), rho_p, significant = 5)
#     assert_approx_equal(oil.diameter(m, T, P, Sa, Ta), de, significant = 5)
#     assert_approx_equal(oil.mass_by_diameter(de, T, P, Sa, Ta), m,
#         significant = 5)
#     shape_calc, de_x, rho_p_x, rho_x, mu_p_x, mu_x, sigma_x = \
#         oil.particle_shape(m, T, P, Sa, Ta)
#     assert_approx_equal(shape_calc, shape, significant = 5)
#     assert_approx_equal(oil.slip_velocity(m, T, P, Sa, Ta), us,
#         significant = 5)
#     assert_approx_equal(oil.surface_area(m, T, P, Sa, Ta), A,
#         significant = 5)
#     assert_approx_equal(oil.heat_transfer(m, T, P, Sa, Ta), beta_T,
#         significant = 5)


# # ----------------------------------------------------------------------------
# # Unit Tests
# # ----------------------------------------------------------------------------

def test_sphere():
    """
    This function tests the dbm object function and dbm_f fortran functions
    for the properties of a FluidParticle with spherical shape for air.

    """
    # Set the variable inputs that may change is physical properties are
    # updated.
    de = 0.0001
    mass = np.array([9.54296597e-13, 2.92408701e-13, 1.62778641e-14,
                     6.91208976e-16])
    fp_type = 0

    # Choose a thermodynamic state and composition
    T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
        kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
        K_salt, rho, mu, rho_p, Cs, sigma, S_S, rho_S, mu_S, rho_p_S, Cs_S, \
        sigma_S, mu_p, D, D_S = base_state()

    # Give the particle properties that should come back from the dbm_f
    # and dbm object function calls
    shape = 1
    us = 0.004577842602244148
    A = 3.1415926535897957e-08
    beta = np.array([0.00011146, 0.00012211, 0.0001198, 0.00010814])
    beta_T = 0.0037826310843623603

    # Give the particle properties that should come back from the dbm_f
    # and dbm object function calls
    shape_S = 1
    us_S =  0.004399537169232339
    A_S = 3.1415926535897957e-08
    beta_S = np.array([0.00010414, 0.00011409, 0.00011193, 0.00010104])
    beta_T_S = 0.0036760250874190332

    # Perform the tests on the dbm_object
    bub = dbm.FluidParticle(['nitrogen', 'oxygen', 'argon', 'carbon_dioxide'],
                            fp_type=fp_type)

    particle_obj_funcs(bub, mass, yk, Mol_wt, fp_type, T, P, S, T, rho_p,
        us, A, Cs, beta, beta_T, de, shape, sigma)

    particle_obj_funcs(bub, mass, yk, Mol_wt, fp_type, T, P, S_S, T, rho_p_S,
        us_S, A_S, Cs_S, beta_S, beta_T_S, de, shape, sigma_S)


    # Check the implementation of the dbm_f functions
    particle_fortran_funcs(mass, T, P, S, T, Mol_wt, fp_type, Pc, Tc, Vc, Vb,
        omega, delta, kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups,
        calc_delta, K_salt, rho, mu, sigma, shape, rho_p, us, A, Cs, beta,
        beta_T, de, mu_p, D)


    particle_fortran_funcs(mass, T, P, S_S, T, Mol_wt, fp_type, Pc, Tc, Vc,
        Vb, omega, delta, kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups,
        calc_delta, K_salt, rho_S, mu_S, sigma_S, shape, rho_p_S, us_S, A_S,
        Cs_S, beta_S, beta_T_S, de, mu_p, D_S)


# def test_ellipsoid():
#     """
#     This function tests the dbm object function and dbm_f fortran functions
#     for the properties of a FluidParticle with ellipsoidal shape for air.

#     """
#     # Set the variable inputs that may change is physical properties are
#     # updated.
#     de = 0.00055
#     mass = np.array([1.58771096e-10, 4.86494976e-11, 2.70822963e-12,
#         1.14999893e-13])
#     fp_type = 0

#     # Choose a thermodynamic state and composition
#     T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
#         kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
#         K_salt, rho, mu, rho_p, Cs, sigma, S_S, rho_S, mu_S, rho_p_S, Cs_S, \
#         sigma_S, mu_p, D, D_S = base_state()

#     # Give the particle properties that should come back from the dbm_f
#     # and dbm object function calls
#     shape = 2
#     us = 0.05804556426362543
#     A = 9.5033177771091327e-07
#     beta = np.array([9.64924845e-05, 1.05243812e-04, 1.03352837e-04,
#                      9.37579986e-05])
#     beta_T = 0.0023227561695939032

#     # Give the particle properties that should come back from the dbm_f
#     # and dbm object function calls
#     shape_S = 2
#     us_S = 0.056179706827748754
#     A_S = 9.5033177771091327e-07
#     beta_S = np.array([8.99749784e-05, 9.81341113e-05, 9.63711082e-05,
#                        8.74255049e-05])
#     beta_T_S = 0.0022462557308422313

#     # Perform the tests on the dbm_object
#     bub = dbm.FluidParticle(['nitrogen', 'oxygen', 'argon', 'carbon_dioxide'],
#                             fp_type=fp_type)
#     particle_obj_funcs(bub, mass, yk, Mol_wt, fp_type, T, P, S, T, rho_p,
#         us, A, Cs, beta, beta_T, de, shape, sigma)
#     particle_obj_funcs(bub, mass, yk, Mol_wt, fp_type, T, P, S_S, T, rho_p_S,
#         us_S, A_S, Cs_S, beta_S, beta_T_S, de, shape, sigma_S)

#     # Check the implementation of the dbm_f functions
#     particle_fortran_funcs(mass, T, P, S, T, Mol_wt, fp_type, Pc, Tc, Vc, Vb,
#         omega, delta, kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups,
#         calc_delta, K_salt, rho, mu, sigma, shape, rho_p, us, A, Cs, beta,
#         beta_T, de, mu_p, D)
#     particle_fortran_funcs(mass, T, P, S_S, T, Mol_wt, fp_type, Pc, Tc, Vc,
#         Vb, omega, delta, kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups,
#         calc_delta, K_salt, rho_S, mu_S, sigma_S, shape, rho_p_S, us_S, A_S,
#         Cs_S, beta_S, beta_T_S, de, mu_p, D_S)


# def test_spherical_cap():
#     """
#     This function tests the dbm object function and dbm_f fortran functions
#     for the properties of a FluidParticle with spherical cap shape for air.

#     """
#     # Set the variable inputs that may change is physical properties are
#     # updated.
#     de = 0.0123
#     mass = np.array([1.77581904e-06, 5.44133701e-07, 3.02909400e-08,
#         1.28624797e-09])
#     fp_type = 0

#     # Choose a thermodynamic state and composition
#     T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta, \
#         kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta, \
#         K_salt, rho, mu, rho_p, Cs, sigma, S_S, rho_S, mu_S, rho_p_S, Cs_S, \
#         sigma_S, mu_p, D, D_S = base_state()

#     # Give the particle properties that should come back from the dbm_f
#     # and dbm object function calls
#     shape = 3
#     us = 0.24667863205554724
#     A = 0.0008041858466133636
#     beta = np.array([0.00014766, 0.00015745, 0.00015536, 0.00014455])
#     beta_T = 0.0014943559651345536

#     # Give the particle properties that should come back from the dbm_f
#     # and dbm object function calls
#     shape_S = 3
#     us_S = 0.2466864514005606
#     A_S = 0.0008041855794806015
#     beta_S = np.array([0.00014191, 0.00015132, 0.00014931, 0.00013892])
#     beta_T_S = 0.0014746858094273542

#     # Perform the tests on the dbm_object
#     bub = dbm.FluidParticle(['nitrogen', 'oxygen', 'argon', 'carbon_dioxide'],
#                             fp_type=fp_type)
#     particle_obj_funcs(bub, mass, yk, Mol_wt, fp_type, T, P, S, T, rho_p,
#         us, A, Cs, beta, beta_T, de, shape, sigma)
#     particle_obj_funcs(bub, mass, yk, Mol_wt, fp_type, T, P, S_S, T, rho_p_S,
#         us_S, A_S, Cs_S, beta_S, beta_T_S, de, shape, sigma_S)

#     # Check the implementation of the dbm_f functions
#     particle_fortran_funcs(mass, T, P, S, T, Mol_wt, fp_type, Pc, Tc, Vc, Vb,
#         omega, delta, kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups,
#         calc_delta, K_salt, rho, mu, sigma, shape, rho_p, us, A, Cs, beta,
#         beta_T, de, mu_p, D)
#     particle_fortran_funcs(mass, T, P, S_S, T, Mol_wt, fp_type, Pc, Tc, Vc,
#         Vb, omega, delta, kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups,
#         calc_delta, K_salt, rho_S, mu_S, sigma_S, shape, rho_p_S, us_S, A_S,
#         Cs_S, beta_S, beta_T_S, de, mu_p, D_S)


# def test_rigid():
#     """
#     This function tests the algorithms in the dbm object InsolubleParticle.
#     This object does not use any new dbm_f functions; rather, it provides
#     capability for particles not defined by the Peng-Robinson equation of
#     state.

#     """
#     # Set up the inputs to the InsolubleParticle object
#     isfluid = True
#     iscompressible = False
#     rho_p = 870.
#     gamma = 29.,
#     beta = 0.0001
#     co= 1.0e-9

#     # Set the thermodynamic state
#     T = 293.15
#     P = 101325.0 * 100.0
#     Sa = 35.0
#     Ta = 279.15

#     # Test a default incompressible fluid particle
#     oil = dbm.InsolubleParticle(isfluid, iscompressible=False)
#     de = 0.001
#     m = 4.86946861306418e-07
#     shape = 1
#     us = 0.01790261590633024
#     A = 3.1415926535897963e-06
#     beta_T = 0.00097737605668318946
#     inert_obj_funcs(oil, T, P, Sa, Ta, 930., de, m, shape, us, A, beta_T)

#     # Test a default oil-like particle
#     oil = dbm.InsolubleParticle(isfluid, iscompressible=True)
#     rho_p = 872.61692306815519
#     de = 0.001
#     m = 4.5690115248484099e-07
#     shape = 1
#     us = 0.02280639990893376
#     A = 3.1415926535897963e-06
#     beta_T = 0.0010911313518431477
#     inert_obj_funcs(oil, T, 101325., 0., Ta, rho_p, de, m, shape, us, A,
#                     beta_T)

#     # Test a default sand-like particle
#     oil = dbm.InsolubleParticle(isfluid=False, iscompressible=False)
#     de = 0.001
#     m = 4.86946861306418e-07
#     shape = 4
#     us = 0.01790261590633024
#     A = 3.1415926535897963e-06
#     beta_T = 0.00097737605668318946
#     inert_obj_funcs(oil, T, P, Sa, Ta, 930., de, m, shape, us, A, beta_T)

#     # Test a user-defined oil-like particle
#     oil = dbm.InsolubleParticle(isfluid=True, iscompressible=True, gamma=29.,
#                                 beta=0.0001, co=1.0e-9)
#     rho_p = 889.27848851245381
#     de = 0.02
#     m = 0.0037250010220082142
#     shape = 2
#     us = 0.13615999681339813
#     A = 0.0012566370614359179
#     beta_T = 0.00040708256713750879
#     inert_obj_funcs(oil, T, P, Sa, Ta, rho_p, de, m, shape, us, A,
#                     beta_T)


def test_cubic_solver():
    """
    This function test the cubic equation solver in the fortran file
    math_funcs.f95

    """

    # Test the Math_funcs solver for cubic equations
    p = np.array([1., -6., 11., -6.])
    r = np.array([1., 2., 3.])

    # for the output:
    r_calc = dbm_c.cubic_roots(p)
    r_calc = np.sort(r_calc)
    print("expected roots:", r)
    print("calculated roots:", r_calc)
    for i in range(3):
        assert_approx_equal(r_calc[i], r[i], significant=6)

# def test_equilibrium():
#     """
#     Test the `FluidMixture.equilibrium` method for determining the gas/
#     liquid partitioning at equilibrium.  This test follows the example
#     problem in McCain (1990), example problem 15-2, p. 431.

#     """
#     # Define the thermodynamic state
#     T = 273.15 + 5./9. * (160. - 32.)
#     P = 1000. * 6894.76

#     # Define the properties of the complete mixture (gas + liquid)
#     composition = ['methane', 'n-butane', 'n-decane']
#     oil = dbm.FluidMixture(composition)
#     yk = np.array([0.5301, 0.1055, 0.3644])
#     m = oil.masses(yk)

#     # Compute the mass partitioning at equilibrium
#     (mk, xi, K) = oil.equilibrium(m, T, P)

#     # Check the result
#     mk_ans = np.array([[0.00587539, 0.00081076, 0.00011848],
#                        [0.00262901, 0.00532121, 0.05173017]])
#     assert_array_almost_equal(mk, mk_ans, decimal=6)
#     xi_ans = np.array([[0.9612035, 0.03661098, 0.00218551],
#                        [0.26474152, 0.14790347, 0.58735501]])
#     assert_array_almost_equal(xi, xi_ans, decimal=6)
#     K_ans = np.array([ 3.63072448, 0.24753296, 0.00372094])
#     assert_array_almost_equal(K, K_ans, decimal=6)
