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

from tamoc import sintef

import numpy as np
from numpy.testing import *

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
    assert d50_gas == de_gas
    
    if d50_gas:
        assert_approx_equal(d50_gas, de_gas, significant=6)
    else:
        assert d50_gas == None
    
    if d50_oil:
        assert_approx_equal(d50_oil, de_oil, significant=6)
    else:
        assert d50_oil == None


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
    dmax_gas = 0.010230463432826563
    dmax_oil = 0.016262984601677299
    
    # Test a strongly atomized source of oil
    D = 0.03
    Q_oil = 0.030
    Q_gas = 0.0
    check_We_model(D, rho_gas, Q_gas, mu_gas, sigma_gas, rho_oil, Q_oil, 
                   mu_oil, sigma_oil, rho, None, 9.121025713149752e-05)
    
    # Test a strongly atomized source of gas
    D = 0.03
    Q_oil = 0.0
    Q_gas = 0.030
    check_We_model(D, rho_gas, Q_gas, mu_gas, sigma_gas, rho_oil, Q_oil, 
                   mu_oil, sigma_oil, rho, 0.00047308368655330852, None)
    
    # Test a strongly atomized source of gas and oil
    D = 0.05
    Q_oil = 0.030
    Q_gas = 0.030
    check_We_model(D, rho_gas, Q_gas, mu_gas, sigma_gas, rho_oil, Q_oil, 
                   mu_oil, sigma_oil, rho, 0.0012859271644586957, 
                   0.00024164352702629014)
    
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
    assert_array_almost_equal(de, np.array([9.84865702e-06, 1.14826911e-05, 
        1.33878349e-05, 1.56090695e-05, 1.81988389e-05, 2.12182883e-05,
        2.47387079e-05, 2.88432159e-05, 3.36287207e-05, 3.92082098e-05, 
        4.57134165e-05, 5.32979306e-05, 6.21408248e-05, 7.24508825e-05, 
        8.44715273e-05, 9.84865702e-05, 1.14826911e-04, 1.33878349e-04,
        1.56090695e-04, 1.81988389e-04, 2.12182883e-04, 2.47387079e-04, 
        2.88432159e-04, 3.36287207e-04, 3.92082098e-04, 4.57134165e-04,
        5.32979306e-04, 6.21408248e-04, 7.24508825e-04, 8.44715273e-04]), 
        decimal = 6)
    assert_array_almost_equal(md, np.array([0.09731316, 0.1248589, 0.16082887,  
        0.20765644, 0.26837291, 0.34667545, 0.44693616, 0.574086, 0.73326503, 
        0.92907423, 1.16420417, 1.43718547, 1.73908916, 2.04935713, 
        2.33179654, 2.53329678, 2.58970422, 2.44386835, 2.07655548, 
        1.53821072, 0.9531004, 0.46944624, 0.17428515, 0.04898427, 
        0.01498327, 0.00970565,  0.00929966,  0.00928682,  0.00928669,  
        0.00928669]), decimal = 6)

