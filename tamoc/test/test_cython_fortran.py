"""
custom test file for checking Cython vs Fortran implimentations
"""

from __future__ import (absolute_import, division, print_function)

import numpy as np
from numpy.testing import assert_approx_equal

from tamoc import dbm_c
from tamoc import dbm_f


# Some test data:
def base_state():
    """
    returns a configuration with everything preset.
    """

    # Choose a thermodynamic state and composition
    T = 273.15 + 15.  # 15C
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

    D = np.array([1.41620605e-09, 1.61034760e-09, 1.56772544e-09, 1.35720048e-09])

    D_S = np.array([1.30803909e-09, 1.48735249e-09, 1.44798573e-09, 1.25354024e-09])

    return (T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta,
            kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta,
            K_salt, rho, mu, rho_p, Cs, sigma, S_S, rho_S, mu_S, rho_p_S,
            Cs_S, sigma_S, mu_p, D, D_S)


def test_viscosity():
    """
    a test, but also a way to run the fortran and Cython versions side by side
    """

    (T, S, P, composition, yk, Mol_wt, Pc, Tc, Vc, Vb, omega, delta,
     kh_0, neg_dH_solR, nu_bar, Aij, Bij, delta_groups, calc_delta,
     K_salt, rho, mu, rho_p, Cs, sigma, S_S, rho_S, mu_S, rho_p_S,
     Cs_S, sigma_S, mu_p, D, D_S) = base_state()

    fp_type = 0

    mass = np.array([9.54296597e-13, 2.92408701e-13, 1.62778641e-14,
                     6.91208976e-16])

    print("expected viscosity:", mu_p)

    viscosity_c = dbm_c.viscosity(T, P, mass, Mol_wt, Pc, Tc, Vc,
                                  omega, delta, Aij, Bij, delta_groups,
                                  calc_delta)

    viscosity_f = dbm_f.viscosity(T, P, mass, Mol_wt, Pc, Tc, Vc,
                                  omega, delta, Aij, Bij, delta_groups,
                                  calc_delta)

    assert_approx_equal(mu_p, viscosity_f[fp_type, 0])

    assert_approx_equal(mu_p, viscosity_c[fp_type, 0])




