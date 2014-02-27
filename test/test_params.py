"""
Unit tests for the `params` module of ``TAMOC``

Provides testing of all of the classes and methods in the `params` module.
This module uses the `ambient` module to control ambient data and the `dbm`
module to predict particle behavior. These tests check the behavior of the
`params.Scales` object and the results of its methods.

The ambient data used here are from the `ctd_BM54.cnv` dataset, stored as::

    ./test/output/test_BM54.nc

This netCDF file is written by the `test_ambient.test_from_ctd` function, 
which is run in the following test script as needed to ensure the dataset is 
available.

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

from tamoc import ambient
import test_sbm
from tamoc import dbm
from tamoc import stratified_plume_model
from tamoc import params

import numpy as np
from numpy.testing import *

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def get_sim_data():
    """
    Get the input data to the `params.Scales` object
    
    Create an `ambient.Profile` object and a list of 
    `stratified_plume_model.Particle` objects as the required input for 
    the `params.Scales` object.  
    
    Returns
    -------
    profile : `ambient.Profile` object
        profile object from the BM54 CTD data
    disp_phases: list
        list of `stratified_plume_model.Particle` objects describing the 
        blowout dispersed phases.
    z0 : float
        depth at the plume model origin (m)
    
    """
    # Get the netCDF file
    nc = test_sbm.make_ctd_file()
    
    # Create a profile object with all available chemicals in the CTD data
    profile = ambient.Profile(nc, chem_names='all')
    
    # Create the stratified plume model object
    spm = stratified_plume_model.Model(profile)
    
    # Set the release conditions
    T0 = 273.15 + 35.   # Release temperature in K
    R = 0.15            # Radius of leak source in m
    
    # Create the gas phase particles
    composition = ['methane', 'ethane', 'propane', 'oxygen']
    yk = np.array([0.93, 0.05, 0.02, 0.0])
    gas = dbm.FluidParticle(composition)
    z0 = 1000.
    disp_phases = []
    
    # Larger free gas bubbles
    mb0 = 8.         # total mass flux in kg/s
    de = 0.025       # bubble diameter in m
    lambda_1 = 0.85
    disp_phases.append(stratified_plume_model.particle_from_mb0(profile, z0, 
        gas, yk, mb0, de, lambda_1, T0))
    
    # Smaller free gas bubbles (note, it is not necessary to have more than
    # one bubble size)
    mb0 = 2.         # total mass flux in kg/s
    de = 0.0075      # bubble diameter in m
    lambda_1 = 0.9
    disp_phases.append(stratified_plume_model.particle_from_mb0(profile, z0, 
        gas, yk, mb0, de, lambda_1, T0))
    
    # Liquid hydrocarbon.  This could either be a dissolving phase (mixture
    # of liquid phases) or an inert phase.  We demonstrate here the simple
    # case of an inert oil phase
    oil = dbm.InsolubleParticle(True, True, rho_p=890., gamma=30., 
                                beta=0.0007, co=2.90075e-9)
    mb0 = 10.        # total mass flux in kg/s
    de = 0.004       # bubble diameter in m
    lambda_1 = 0.9
    disp_phases.append(stratified_plume_model.particle_from_mb0(profile, z0, 
        oil, np.array([1.]), mb0, de, lambda_1, T0))
    
    # Return the simulation data
    return (profile, disp_phases, z0)

def check_get_variables(model, z0, u_inf, B_ans, N_ans, us_ans, ua_ans):
    """
    Check the results of the `Scales.get_variables()` method
    """
    # Get the model answer
    (B, N, us, ua) = model.get_variables(z0, 0.15)
    
    # Check the results
    assert_approx_equal(B, B_ans, significant=6)
    assert_approx_equal(N, N_ans, significant=6)
    assert_approx_equal(us, us_ans, significant=6)
    assert_approx_equal(ua, ua_ans, significant=6)


# ----------------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------------

def test_params_module():
    """
    Test the class and methods in the `params` module
    
    Test the `params.Scales` class object and the output from its methods
    for a typical plume simulation of a subsea oil well blowout.  This test
    function tests the same calculations as stored in `./bin/params.py`.
    
    """
    # Get the inputs required by the Scales object
    (profile, disp_phases, z0) = get_sim_data()
    
    
    # Test that the governing parameters are computed correctly
    # First, test a single dispersed phase
    model = params.Scales(profile, disp_phases[1])
    check_get_variables(model, z0, 0.15, 0.21352660994152442, 
                        0.001724100901081246, 0.23066264207937315, 0.15)
    
    # Second, try a list of dispersed phases, where the dominant phase is 
    # not the first one
    particles = [disp_phases[1], disp_phases[0], disp_phases[2]]
    model = params.Scales(profile, particles)
    check_get_variables(model, z0, 0.15, 1.0829392838486933, 
                        0.001724100901081246, 0.33740973034294336, 0.15)
    
    # Third, make sure we get the same answer as the previous case if the 
    # particles are in a different order (i.e., the original order)
    model = params.Scales(profile, disp_phases)
    check_get_variables(model, z0, 0.15, 1.0829392838486933, 
                        0.001724100901081246, 0.33740973034294336, 0.15)
    
    # Using the latest Scales object, check that the other methods return
    # the correct results.  Since these methods only depend on the values 
    # of B, N, and us computed by the get_variables() method, only one case
    # needs to be tested
    assert_approx_equal(model.h_T(z0), 344.84410830994869, significant=6)
    assert_approx_equal(model.h_P(z0), 625.03783498854557, significant=6)
    assert_approx_equal(model.h_S(z0, 0.15), 290.90093637637739, 
        significant=6)
    assert_approx_equal(model.lambda_1(z0, 0), 0.74468472200979985, 
        significant=6)
    assert_approx_equal(model.u_inf_crit(z0), 0.062897786762886404, 
        significant=6)

