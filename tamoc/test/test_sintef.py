"""
Unit tests for the `sintef` module of ``TAMOC``

Provides testing of all of the functions in the `sintef` module.  Since this
is a simple module, the unit tests are stand-alone and do not require 
external data source or disk access.

Notes
-----
All of the tests defined herein check the general behavior of each of the
programmed functions--this is not a comparison against measured data. The
results of the hand calculations entered below as sample solutions have been
ground-truthed for their reasonableness. However, passing these tests only
means the programs and their interfaces are working as expected, not that they
have been validated against measurements.

"""
# S. Socolofsky, February 2014, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import sintef

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_approx_equal

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def check_We_model(D, rho_gas, Q_gas, mu_gas, sigma_gas, rho_oil, Q_oil, 
               mu_oil, sigma_oil, rho, de_gas, de_oil):
    """
    Check the solution for a modified We-number model
    
    """
    # Get the mass fluxes of gas and oil
    md_gas = Q_gas * rho_gas
    md_oil = Q_oil * rho_oil
    
    # Compute the sizes from the model
    d50_gas, d50_oil = sintef.modified_We_model(D, rho_gas, md_gas, mu_gas, 
                       sigma_gas, rho_oil, md_oil, mu_oil, sigma_oil, rho)
    
    # Check the model result
    if de_gas:
        assert_approx_equal(d50_gas, de_gas, significant=6)
    else:
        assert d50_gas == 0.
    
    if de_oil:
        assert_approx_equal(d50_oil, de_oil, significant=6)
    else:
        assert d50_oil == 0.


# ----------------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------------

def test_sintef_module():
    """
    Test the results of the `sintef` module
    
    Test the output of the modified We-number model for a pure gas release,
    pure oil release, and a mixed oil and gas release.  Also, test cases that 
    are and are not limited by the maximum stable bubble/droplet size.
    Finally, test the function for creating a droplet size distribution.
    
    """
    # Enter a set of base-case physical fluid properties
    rho_gas = 100.
    rho_oil = 850.
    sigma_gas = 0.06
    sigma_oil = 0.03
    mu_gas = 0.00002
    mu_oil = 0.0005
    rho = 1035.
    
    # Enter the maximum allowable bubble and droplet size
    dmax_gas = 0.04697373994474605
    dmax_oil = 0.016262984601677299
    
    # Test a strongly atomized source of oil
    D = 0.03
    Q_oil = 0.030
    Q_gas = 0.0
    check_We_model(D, rho_gas, Q_gas, mu_gas, sigma_gas, rho_oil, Q_oil, 
                   mu_oil, sigma_oil, rho, None, 0.00014463009911995408)
    
    
    # Test a strongly atomized source of gas
    D = 0.03
    Q_oil = 0.0
    Q_gas = 0.030
    check_We_model(D, rho_gas, Q_gas, mu_gas, sigma_gas, rho_oil, Q_oil, 
                   mu_oil, sigma_oil, rho, 0.0007809790559321155, None)
    
    # Test a strongly atomized source of gas and oil
    D = 0.05
    Q_oil = 0.030
    Q_gas = 0.030
    check_We_model(D, rho_gas, Q_gas, mu_gas, sigma_gas, rho_oil, Q_oil, 
                   mu_oil, sigma_oil, rho, 0.002124110280893527, 
                   0.00038937956648815573)
    
    # Test a gas source limited by the maximum stable bubble size
    D = 0.3
    Q_oil = 0.0
    Q_gas = 0.100
    check_We_model(D, rho_gas, Q_gas, mu_gas, sigma_gas, rho_oil, Q_oil, 
                   mu_oil, sigma_oil, rho, dmax_gas, None)
    
    # Test an oil source limited by the maximum stable droplet size
    D = 0.3
    Q_oil = 0.010
    Q_gas = 0.0
    check_We_model(D, rho_gas, Q_gas, mu_gas, sigma_gas, rho_oil, Q_oil, 
                   mu_oil, sigma_oil, rho, None, dmax_oil)
    
    # Test an oil leak that is non-atomizing
    D = 0.01
    Q_oil = 0.00001
    Q_gas = 0.0
    check_We_model(D, rho_gas, Q_gas, mu_gas, sigma_gas, rho_oil, Q_oil, 
                   mu_oil, sigma_oil, rho, None, 0.012)
    
    # Get the droplet size distribution for oil for a strongly atomized
    # source
    D = 0.03
    Q_oil = 0.030
    md_oil = Q_oil * rho_oil
    d50, _ = sintef.modified_We_model(D, rho_oil, md_oil, mu_oil, 
                       sigma_oil, 0., 0., 0., 0., rho)
    
    de, md = sintef.rosin_rammler(30, d50, md_oil, sigma_oil, rho_oil, rho)
    assert_array_almost_equal(de, np.array([1.45878944e-05, 1.63832188e-05, 
        1.83994927e-05, 2.06639084e-05, 2.32070044e-05, 2.60630780e-05, 
        2.92706470e-05, 3.28729698e-05, 3.69186285e-05, 4.14621843e-05, 
        4.65649130e-05, 5.22956318e-05, 5.87316271e-05, 6.59596969e-05, 
        7.40773213e-05, 8.31939774e-05, 9.34326155e-05, 1.04931317e-04,       
        1.17845158e-04, 1.32348298e-04, 1.48636332e-04, 1.66928926e-04, 
        1.87472779e-04, 2.10544953e-04, 2.36456607e-04, 2.65557195e-04, 
        2.98239176e-04, 3.34943311e-04, 3.76164604e-04, 4.22458979e-04,]), 
        decimal = 6)
    assert_array_almost_equal(md, np.array([0.05977836, 0.07347575, 
        0.09025703, 0.11078833, 0.13586501, 0.16642905, 0.20358426, 
        0.2486062,  0.30294156, 0.36818918, 0.44605145, 0.53824003,         
        0.6463149,  0.77143036, 0.91395899, 1.07296736, 1.24553354, 
        1.42593496, 1.60480996, 1.76851643, 1.89907518, 1.97525133, 
        1.97538766, 1.88236042, 1.69023873, 1.41080284, 1.07645928, 
        0.73561661, 0.43909252, 0.22204271]), decimal = 6)


