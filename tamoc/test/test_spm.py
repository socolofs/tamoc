"""
Unit tests for the `stratified_plume_model` module of ``TAMOC``

Provides testing of the objects, methods and functions defined in the
`stratified_plume_model` module of ``TAMOC``.  This module uses the `ambient`
module to control ambient data and the `dbm` module to predict particle
behavior.  These tests check the behavior of the object, the results of
the simulations, and the read/write algorithms.

The ambient data used here are from the `ctd_BM54.cnv` dataset, stored as::

    ./test/output/test_BM54.nc

This netCDF file is written by the `test_ambient.test_from_ctd` function,
which is run in the following as needed to ensure the dataset is available.

Notes
-----
All of the tests defined herein check the general behavior of each of the
programmed function--this is not a comparison against measured data. The
results of the hand calculations entered below as sample solutions have been
ground-truthed for their reasonableness. However, passing these tests only
means the programs and their interfaces are working as expected, not that they
have been validated against measurements.

"""
# S. Socolofsky, August 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

import os

from tamoc import seawater
from tamoc import ambient
from tamoc.test import test_sbm
from tamoc import dbm
from tamoc import dispersed_phases
from tamoc import stratified_plume_model
from tamoc import smp as simp

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_approx_equal

DATA_DIR = os.path.realpath(os.path.join(
    os.path.dirname(__file__),'../data'))
OUTPUT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__),'output'))

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

def get_profile():
    """
    Create an `ambient.Profile` object from a netCDF file

    Create the `ambient.Profile` object needed by the `stratified_plume_model`
    simulation from the netCDF file `./test/output/test_bm54.nc`.  This
    function calls `test_sbm.make_ctd_file` to create the netCDF dataset
    before using it to create the `ambient.Profile` object.

    Returns
    -------
    profile : `ambient.Profile` object
        Return a profile object from the BM54 CTD data

    """
    # Get the netCDF file
    nc = test_sbm.make_ctd_file()

    # Return a profile object with all available chemicals in the CTD data
    profile = ambient.Profile(nc, chem_names='all')
    profile.close_nc()
    
    return profile

def get_sim_data():
    """
    Create the data needed to initialize a simulation

    Performs the steps necessary to set up a stratified plume model simulation
    and passes the input variables to the `Model` object and
    `Model.simulate()` method.

    Returns
    -------
    profile : `ambient.Profile` object
        Return a profile object from the BM54 CTD data
    particles : list of `PlumeParticle` objects
        List of `PlumeParticle` objects containing the dispersed phase initial
        conditions
    z : float
        Depth of the release port (m)
    R : float
        Radius of the release port (m)
    maxit : float
        Maximum number of iterations to converge between inner and outer
        plumes
    toler : float
        Relative error tolerance to accept for convergence (--)
    delta_z : float
        Maximum step size to use in the simulation (m).  The ODE solver
        in `calculate` is set up with adaptive step size integration, so
        in theory this value determines the largest step size in the
        output data, but not the numerical stability of the calculation.

    """
    # Get the ambient CTD data
    profile = get_profile()

    # Specify the release location and geometry and initialize a particle
    # list
    z0 = 300.
    R = 0.15
    particles = []

    # Add a dissolving particle to the list
    composition = ['oxygen', 'nitrogen', 'argon']
    yk = np.array([1.0, 0., 0.])
    o2 = dbm.FluidParticle(composition)
    Q_N = 150. / 60. / 60.
    de = 0.005
    lambda_1 = 0.85
    particles.append(stratified_plume_model.particle_from_Q(profile, z0, o2,
                     yk, Q_N, de, lambda_1))

    # Add an insoluble particle to the list
    composition = ['inert']
    yk = np.array([1.])
    oil = dbm.InsolubleParticle(True, True)
    mb0 = 50.
    de = 0.01
    lambda_1 = 0.8
    particles.append(stratified_plume_model.particle_from_mb0(profile, z0,
                     oil, [1.], mb0, de, lambda_1))

    # Set the other simulation parameters
    maxit = 2
    toler = 0.2
    delta_z = 1.0

    # Return the results
    return (profile, particles, z0, R, maxit, toler, delta_z)


def check_sim(particles, R, maxit, toler, delta_z, spm):
    """
    Check the results of a simulation against known values

    Check the simulation results for the simulation defined above in
    `get_sim_data()`.  This is used by the `test_simulate()` and
    `test_files()` functions below.

    Parameters
    ----------
    particles : list of `PlumeParticle` objects
        List of `PlumeParticle` objects containing the dispersed phase initial
        conditions
    R : float
        Radius of the release port (m)
    maxit : float
        Maximum number of iterations to converge between inner and outer
        plumes
    toler : float
        Relative error tolerance to accept for convergence (--)
    delta_z : float
        Maximum step size to use in the simulation (m).  The ODE solver
        in `calculate` is set up with adaptive step size integration, so
        in theory this value determines the largest step size in the
        output data, but not the numerical stability of the calculation.
    spm : `stratified_plume_model.Model` object
        Object containing the model simulation results

    """
    # Check the object attributes are set correctly
    assert len(spm.particles) == len(particles)
    assert_array_almost_equal(spm.K_T0, np.array([particles[i].K_T
                              for i in range(len(particles))]),
                              decimal=6)
    assert spm.R == R
    assert spm.maxit == maxit
    assert spm.toler == toler
    assert spm.delta_z == delta_z

    # Check the model output
    assert spm.sim_stored == True
    assert spm.zi[0] == 300.
    ans = np.array([7.15956185e-02, 7.25171119e-02, 2.54372237e+00,
         8.42522400e+07, 5.87744590e-02, 0.00000000e+00,
         0.00000000e+00, 3.35412989e+04, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         5.00000000e+01, 2.85339070e+07, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         2.93018840e-04, 0.00000000e+00, 0.00000000e+00])
    for i in range(len(ans)):
        assert_approx_equal(spm.yi[0,i], ans[i], significant=2)
    assert_approx_equal(spm.zi[5], 299.9851495542245)
    ans = np.array([7.27017077e-02, 7.33950786e-02, 2.58302067e+00, 
        8.55538638e+07, 5.87732529e-02, 0.00000000e+00, 0.00000000e+00,
        3.35406110e+04, 1.20102075e-02, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.00000000e+01, 2.85339073e+07, 1.28560570e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.98752765e-04,
        0.00000000e+00, 0.00000000e+00])
    for i in range(len(ans)):
        assert_approx_equal(spm.yi[5,i], ans[i], significant=2)
    assert spm.zo[0] == 0.
    ans = np.array([-2.43596803e+00,  4.82171573e-02, -8.82547235e+01, 
        -2.99464970e+09, -1.77268357e-02, -0.00000000e+00, -0.00000000e+00])
    assert spm.zo[-1] >= 300.
    assert len(spm.yo[0,:]) == 7
    assert spm.yo[0,0] < 0.
    assert spm.yo[0,1] > 0.
    assert spm.yo[0,2] < 0.
    assert spm.yo[0,3] < 0.
    assert spm.yo[0,4] < 0.
    assert spm.yo[0,5] <= 0.
    assert spm.yo[0,6] <= 0.
    assert_array_almost_equal(spm.yo[-1,:],
        np.array([0., 0., 0., 0., 0., 0., 0.]),
        decimal=6)

# ----------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------

def test_modelparams_obj():
    """
    Test the behavior of the `ModelParams` object

    Test the instantiation and attribute data for the `ModelParams object of
    the `stratified_plume_model` module.

    """
    # Get the ambient CTD data
    profile = get_profile()

    # Initialize the ModelParams object
    p = stratified_plume_model.ModelParams(profile)

    # Check if the attributes are set correctly
    assert_approx_equal(p.rho_r, 1031.035855535142, significant=6)
    assert p.alpha_1 == 0.055
    assert p.alpha_2 == 0.110
    assert p.alpha_3 == 0.110
    assert p.lambda_2 == 1.00
    assert p.epsilon == 0.015
    assert p.qdis_ic == 0.1
    assert p.c1 == 0.
    assert p.c2 == 1.
    assert p.fe == 0.1
    assert p.gamma_i == 1.10
    assert p.gamma_o == 1.10
    assert p.Fr_0 == 1.6
    assert p.Fro_0 == 0.1
    assert p.nwidths == 1
    assert p.naverage == 1
    assert p.g == 9.81
    assert p.Ru == 8.314510

def test_particle_obj():
    """
    Test the object behavior for the `PlumeParticle` object

    Test the instantiation and attribute data for the `PlumeParticle` object
    of the `stratified_plume_model` module.

    """
    # Set up the base parameters describing a particle object
    T = 273.15 + 15.
    P = 150e5
    Sa = 35.
    Ta = 273.15 + 4.
    composition = ['methane', 'ethane', 'propane', 'oxygen']
    yk = np.array([0.85, 0.07, 0.08, 0.0])
    de = 0.005
    lambda_1 = 0.85
    K = 1.
    Kt = 1.
    fdis = 1e-6

    # Compute a few derived quantities
    bub = dbm.FluidParticle(composition)
    nb0 = 1.e5
    m0 = bub.masses_by_diameter(de, T, P, yk)

    # Create a `PlumeParticle` object
    bub_obj = dispersed_phases.PlumeParticle(bub, m0, T, nb0, lambda_1, P,
                                             Sa, Ta, K, Kt, fdis)

    # Check if the initialized attributes are correct
    for i in range(len(composition)):
        assert bub_obj.composition[i] == composition[i]
    assert_array_almost_equal(bub_obj.m0, m0, decimal=6)
    assert bub_obj.T0 == T
    assert_array_almost_equal(bub_obj.m, m0, decimal=6)
    assert bub_obj.T == T
    assert bub_obj.cp == seawater.cp() * 0.5
    assert bub_obj.K == K
    assert bub_obj.K_T == Kt
    assert bub_obj.fdis == fdis
    for i in range(len(composition)-1):
        assert bub_obj.diss_indices[i] == True
    assert bub_obj.diss_indices[-1] == False
    assert bub_obj.nb0 == nb0
    assert bub_obj.lambda_1 == lambda_1

    # Including the values after the first call to the update method
    us_ans = bub.slip_velocity(m0, T, P, Sa, Ta)
    rho_p_ans = bub.density(m0, T, P)
    A_ans = bub.surface_area(m0, T, P, Sa, Ta)
    Cs_ans = bub.solubility(m0, T, P, Sa)
    beta_ans = bub.mass_transfer(m0, T, P, Sa, Ta)
    beta_T_ans = bub.heat_transfer(m0, T, P, Sa, Ta)
    assert bub_obj.us == us_ans
    assert bub_obj.rho_p == rho_p_ans
    assert bub_obj.A == A_ans
    assert_array_almost_equal(bub_obj.Cs, Cs_ans, decimal=6)
    assert_array_almost_equal(bub_obj.beta, beta_ans, decimal=6)
    assert bub_obj.beta_T == beta_T_ans

    # No need to test the properties or diameter objects since they are
    # inherited from the `single_bubble_model` and tested in `test_sbm`.

    # Check functionality of insoluble particle
    drop = dbm.InsolubleParticle(isfluid=True, iscompressible=True)
    m0 = drop.mass_by_diameter(de, T, P, Sa, Ta)
    drop_obj = dispersed_phases.PlumeParticle(drop, m0, T, nb0,
               lambda_1, P, Sa, Ta, K, fdis=fdis, K_T=Kt)
    assert len(drop_obj.composition) == 1
    assert drop_obj.composition[0] == 'inert'
    assert_array_almost_equal(drop_obj.m0, m0, decimal=6)
    assert drop_obj.T0 == T
    assert_array_almost_equal(drop_obj.m, m0, decimal=6)
    assert drop_obj.T == T
    assert drop_obj.cp == seawater.cp() * 0.5
    assert drop_obj.K == K
    assert drop_obj.K_T == Kt
    assert drop_obj.fdis == fdis
    assert drop_obj.diss_indices[0] == True
    assert drop_obj.nb0 == nb0
    assert drop_obj.lambda_1 == lambda_1

    # Including the values after the first call to the update method
    us_ans = drop.slip_velocity(m0, T, P, Sa, Ta)
    rho_p_ans = drop.density(T, P, Sa, Ta)
    A_ans = drop.surface_area(m0, T, P, Sa, Ta)
    beta_T_ans = drop.heat_transfer(m0, T, P, Sa, Ta)
    assert drop_obj.us == us_ans
    assert drop_obj.rho_p == rho_p_ans
    assert drop_obj.A == A_ans
    assert drop_obj.beta_T == beta_T_ans

def test_plume_objs():
    """
    Test the object behavior for the `InnerPlume` and `OuterPlume` objects

    Test the instantiation and attribute data for the `InnerPlume` object of
    the `stratified_plume_model` module and the `OuterPlume` object.

    This test does many of the calculations in `Model.Simulate`, but does
    not perform the simulation.  These calculations are needed to create
    reasonable `InnerPlume` and `OuterPlume` objects that can be tested.

    """
    # Get the model parameters
    (profile, particles, z0, R, maxit, toler, delta_z) = get_sim_data()
    profile.close_nc()
    p = stratified_plume_model.ModelParams(profile)

    # Get the initial conditions for this plume
    z0, y0, chem_names = simp.main_ic(profile, particles, p, z0, R)

    # Create the `InnerPlume` object:
    yi = stratified_plume_model.InnerPlume(z0, y0, profile, particles, p,
                                           chem_names)
    
    # Validate the values in yi
    assert yi.z0 == z0
    assert_array_almost_equal(yi.y0, y0, decimal=6)
    for i in range(len(chem_names)):
        assert yi.chem_names[i] == chem_names[i]
    assert yi.len == y0.shape[0]
    assert yi.nchems == len(chem_names)
    assert yi.np == len(particles)
    assert yi.z == z0
    assert_array_almost_equal(yi.y, y0, decimal=6)
    assert_approx_equal(yi.Q, 0.071595618494002422, significant=6)
    assert_approx_equal(yi.J, 0.072517111934455133, significant=6)
    assert_approx_equal(yi.S, 2.54372236947938290, significant=6)
    assert_approx_equal(yi.H, 84252240.034169987, significant=6)
    assert_array_almost_equal(yi.M_p[0], np.array([0.0587744, 0., 0.]),
        decimal=6)
    assert_array_almost_equal(yi.M_p[1], np.array([50.]), decimal=6)
    assert_array_almost_equal(yi.H_p, np.array([33541.29892424, 
        28533906.99805339]), decimal=2)
    assert_array_almost_equal(yi.C, np.array([0.00029302, 0., 0.]), decimal=6)
    assert_approx_equal(yi.Ta, 285.52466101019053, significant=6)
    assert_approx_equal(yi.Sa, 35.52902290651307, significant=6)
    assert_approx_equal(yi.P, 3123785.3190075322, significant=6)
    assert_approx_equal(yi.rho_a, 1028.32228185795, significant=6)
    assert_array_almost_equal(yi.ca, np.array([ 0.00409269, 0., 0.]),
                              decimal=6)
    assert_approx_equal(yi.u, 1.0128708077370672, significant=6)
    assert_approx_equal(yi.b, 0.14999999999999999, significant=6)
    assert_approx_equal(yi.s, 35.52902290651307, significant=6)
    assert_approx_equal(yi.T, 285.52466101019047, significant=6)
    assert_array_almost_equal(yi.c, np.array([ 0.00409269, 0., 0.]),
                              decimal=6)
    assert_approx_equal(yi.rho, 1028.32228185795, significant=6)
    assert_array_almost_equal(yi.xi, np.array([0.01896933, 0.9054897]),
                              decimal=6)
    assert_array_almost_equal(yi.fb, np.array([13.50017158, 83.04871004]),
                              decimal=4)
    assert_approx_equal(yi.Xi, 0.92445903017296149, significant=6)
    assert_approx_equal(yi.Fb, 96.548881624595595, significant=6)
    assert_approx_equal(yi.Ep, 0.0, significant=6)

    # Get similar initial conditions for the outer plume
    y0[0] = 100. * y0[0]
    yi = stratified_plume_model.InnerPlume(z0, y0, profile, particles, p,
                                           chem_names)
    yo_z0, yo_y0 = simp.outer_surf(yi, p)

    # Create the outer plume object
    yo = stratified_plume_model.OuterPlume(yo_z0, yo_y0, profile, p,
                                           chem_names, 0.15)

    # Validate the values in yi
    assert yo.z0 == yo_z0
    assert_array_almost_equal(yo.y0, yo_y0, decimal=6)
    for i in range(len(chem_names)):
        assert yo.chem_names[i] == chem_names[i]
    assert yo.len == yo_y0.shape[0]
    assert yo.nchems == len(chem_names)
    assert yo.z == yo_z0
    assert_array_almost_equal(yo.y, yo_y0, decimal=6)
    assert_approx_equal(yo.Q, -7.8755174713019168, significant=6)
    assert_approx_equal(yo.J, 1.6304312871005346, significant=6)
    assert_approx_equal(yo.S, -27.980944063852967, significant=6)
    assert_approx_equal(yo.H, -926774574.0759038, significant=6)
    assert_array_almost_equal(yo.C, np.array([-0.00322347, -0., -0.]),
                              decimal=6)
    assert_approx_equal(yo.Ta, 285.5246610101905, significant=6)
    assert_approx_equal(yo.Sa, 35.52902290651307, significant=6)
    assert_approx_equal(yo.P, 3123788.4472934473, significant=6)
    assert_approx_equal(yo.rho_a, 1028.3222832497443, significant=6)
    assert_array_almost_equal(yo.ca, np.array([0.00409303, 0., 0.]),
                              decimal=6)
    assert_approx_equal(yo.u, -0.20702528983546334, significant=6)
    assert_approx_equal(yo.b, 3.483020467914242, significant=6)
    # The next four lines give non-physical results due to the way y0 was
    # manipulated above.  Nonetheless, the test is still checking the right
    # behavior.
    assert_approx_equal(yo.s, 3.5529022906513066, significant=6)
    assert_approx_equal(yo.T, 28.55246610101905, significant=6)
    assert_array_almost_equal(yo.c, np.array([0.0004093, 0., 0.]),
                              decimal=6)
    assert_approx_equal(yo.rho, -10624.710049181911, significant=6)

    # Turn off the inner plume flow rate
    y0[0] = 0.
    yi.update(z0, y0, particles, profile, p)
    assert_approx_equal(yi.Q, 0.0, significant=8)
    assert_approx_equal(yi.J, 0.07251710156563534, significant=6)
    assert_approx_equal(yi.S, 2.5437223694793829, significant=6)
    assert_approx_equal(yi.H, 84252240.034169987, significant=6)
    assert_array_almost_equal(yi.M_p[0], np.array([0.05877446, 0., 0.]),
                              decimal=6)
    assert_array_almost_equal(yi.M_p[1], np.array([50.]), decimal=6)
    assert_array_almost_equal(yi.H_p, np.array([33541.29948138, 
        28533906.99805339]), decimal=6)
    assert_array_almost_equal(yi.C, np.array([0.00029302, 0., 0.]), decimal=6)
    assert_approx_equal(yi.Ta, 285.52466101019053, significant=6)
    assert_approx_equal(yi.Sa, 35.52902290651307, significant=6)
    assert_approx_equal(yi.P, 3123785.3190075322, significant=6)
    assert_approx_equal(yi.rho_a, 1028.32228185795, significant=6)
    assert_array_almost_equal(yi.ca, np.array([ 0.00409269, 0., 0.]),
                              decimal=6)
    assert_approx_equal(yi.u, 0., significant=6)
    assert_approx_equal(yi.b, 0., significant=6)
    assert_approx_equal(yi.s, 35.52902290651307, significant=6)
    assert_approx_equal(yi.T, 285.52466101019053, significant=6)
    assert_array_almost_equal(yi.c, np.array([0.00409269, 0., 0.]),
                              decimal=6)
    assert_approx_equal(yi.rho, 1028.32228185795, significant=6)
    assert_array_almost_equal(yi.xi, np.array([0., 0.]),
                              decimal=6)
    assert_array_almost_equal(yi.fb, np.array([0., 0.]),
                              decimal=6)
    assert_approx_equal(yi.Xi, 0., significant=6)
    assert_approx_equal(yi.Fb, 0., significant=6)
    assert_approx_equal(yi.Ep, 0., significant=6)

    # Turn off the outer plume flow rate
    yo_y0[0] = 0.
    yo.update(yo_z0, yo_y0, profile, p, 0.15)
    assert_approx_equal(yo.Q, 0., significant=6)
    assert_approx_equal(yo.J, 1.6304314421135921, significant=6)
    assert_approx_equal(yo.S, -27.980946064273212, significant=6)
    assert_approx_equal(yo.H, -926774640.37586987, significant=6)
    assert_array_almost_equal(yo.C, np.array([-0.00322321, -0., -0.]),
                              decimal=6)
    assert_approx_equal(yo.Ta, 285.52466101019053, significant=6)
    assert_approx_equal(yo.Sa, 35.52902290651307, significant=6)
    assert_approx_equal(yo.P, 3123785.3190075322, significant=6)
    assert_approx_equal(yo.rho_a, 1028.32228185795, significant=6)
    assert_array_almost_equal(yo.ca, np.array([0.00409269, 0., 0.]),
                              decimal=6)
    assert_approx_equal(yo.u, 0., significant=6)
    assert_approx_equal(yo.b, 0., significant=6)
    assert_approx_equal(yo.s, 35.52902290651307, significant=6)
    assert_approx_equal(yo.T, 285.52466101019053, significant=6)
    assert_array_almost_equal(yo.c, np.array([0.00409269, 0., 0.]),
                              decimal=6)
    assert_approx_equal(yo.rho, 1028.32228185795, significant=6)

def test_simulate():
    """
    Test the `Model.simulate()` method of the Stratified Plume Model

    Run a simulation to test the operation of the `Model.simulate()` method
    of the Stratified Plume Model.

    """
    # Get the model parameters
    (profile, particles, z0, R, maxit, toler, delta_z) = get_sim_data()

    # Initialize a stratified plume model `Model` object
    spm = stratified_plume_model.Model(profile)

    # Run the simulation
    spm.simulate(particles, z0, R, maxit, toler, delta_z, False)

    # Check that the results are correct
    assert spm.sim_stored == True
    check_sim(particles, R, maxit, toler, delta_z, spm)

def test_files():
    """
    Test the input and output of model simulation data

    Test the methods `save_sim`, `save_txt`, and `load_sim` of the Stratified
    Plume Model `Model` object.

    """
    # Get the model parameters
    (profile, particles, z0, R, maxit, toler, delta_z) = get_sim_data()

    # Initialize a stratified plume model `Model` object
    spm = stratified_plume_model.Model(profile)

    # Run the simulation
    spm.simulate(particles, z0, R, maxit, toler, delta_z, False)

    # Save the simulation to a netCDF file
    fname = os.path.join(OUTPUT_DIR, 'spm_data.nc')
    profile_path = './test_BM54.nc'
    profile_info = 'Results of ./test_spm.py script'
    spm.save_sim(fname, profile_path, profile_info)

    # Save the simulation to a text file
    base_name = os.path.join(OUTPUT_DIR, 'spm_data')
    spm.save_txt(base_name, profile_path, profile_info)

    # Save the simulation derived variables to to text files
    base_name = os.path.join(OUTPUT_DIR, 'spm_derived_vars')
    spm.save_derived_variables(base_name)
    
    # Load the simulation data from the netCDF file
    spm.load_sim(fname)
    assert spm.sim_stored

    # Initialize a Model object from the netCDF file
    spm_load = stratified_plume_model.Model(simfile=fname)
    assert spm.sim_stored

