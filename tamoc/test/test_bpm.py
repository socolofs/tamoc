"""
Unit tests for the `bent_plume_model` module of ``TAMOC``

Provides testing of the classes, methods and functions defined in the
`bent_plume_model` module of ``TAMOC``. These tests check the behavior of the
object, the results of the simulations, and the read/write algorithms.

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
# S. Socolofsky, September 2015, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

import os

from tamoc import seawater
from tamoc import ambient
from tamoc.test import test_sbm
from tamoc import dbm
from tamoc import dispersed_phases
from tamoc import bent_plume_model
from tamoc import lmp

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

    Create the `ambient.Profile` object needed by the `bent_plume_model`
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

    # Create profile object
    profile = ambient.Profile(nc, chem_names='all')
    
    # Add crossflow
    z = profile.interp_ds.coords['z'].values
    ua = np.zeros(len(z))
    for i in range(len(z)):
        ua[i] = 0.15

    # Add this crossflow profile to the Profile dataset
    data = np.vstack((z, ua)).transpose()
    symbols = ['z', 'ua']
    units = ['m', 'm/s']
    comments = ['measured', 'synthetic']
    profile.append(data, symbols, units, comments, 0)
    
    # Close the netCDF dataset
    profile.close_nc()
    
    # Return a profile object
    return profile


def get_sim_data():
    """
    Create the data needed to initialize a simulation

    Performs the steps necessary to set up a bent plume model simulation
    and passes the input variables to the `Model` object and
    `Model.simulate()` method.

    Returns
    -------
    profile : `ambient.Profile` object
        Return a profile object from the BM54 CTD data
    z0 : float
        Depth of the release port (m)
    D : float
        Diameter of the release port (m)
    Vj : float
        Initial velocity of the jet (m/s)
    phi_0 : float
        Vertical angle from the horizontal for the discharge orientation
        (rad in range +/- pi/2)
    theta_0 : float
        Horizontal angle from the x-axis for the discharge orientation.
        The x-axis is taken in the direction of the ambient current.
        (rad in range 0 to 2 pi)
    Sj : float
        Salinity of the continuous phase fluid in the discharge (psu)
    Tj : float
        Temperature of the continuous phase fluid in the discharge (T)
    cj : ndarray
        Concentration of passive tracers in the discharge (user-defined)
    tracers : string list
        List of passive tracers in the discharge.  These can be chemicals
        present in the ambient `profile` data, and if so, entrainment of
        these chemicals will change the concentrations computed for these
        tracers.  However, none of these concentrations are used in the
        dissolution of the dispersed phase.  Hence, `tracers` should not
        contain any chemicals present in the dispersed phase particles.
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the
        simulation
    dt_max : float
        Maximum step size to take in the storage of the simulation
        solution (s)
    sd_max : float
        Maximum number of orifice diameters to compute the solution along
        the plume centerline (m/m)

    """
    # Get the ambient CTD data
    profile = get_profile()

    # Specify the release location and geometry and initialize a particle
    # list
    z0 = 300.
    D = 0.3
    particles = []

    # Add a dissolving particle to the list
    composition = ['oxygen', 'nitrogen', 'argon']
    yk = np.array([1.0, 0., 0.])
    o2 = dbm.FluidParticle(composition)
    Q_N = 1.5 / 60. / 60.
    de = 0.009
    lambda_1 = 0.85
    (m0, T0, nb0, P, Sa, Ta) = dispersed_phases.initial_conditions(
        profile, z0, o2, yk, Q_N, 1, de)
    particles.append(bent_plume_model.Particle(0., 0., z0, o2, m0, T0,
        nb0, lambda_1, P, Sa, Ta, K=1., K_T=1., fdis=1.e-6, t_hyd=0.))

    # Add an insoluble particle to the list
    composition = ['inert']
    yk = np.array([1.])
    oil = dbm.InsolubleParticle(True, True)
    mb0 = 10.
    de = 0.01
    lambda_1 = 0.8
    (m0, T0, nb0, P, Sa, Ta) = dispersed_phases.initial_conditions(
        profile, z0, oil, yk, mb0, 2, de)
    particles.append(bent_plume_model.Particle(0., 0., z0, oil, m0, T0,
        nb0, lambda_1, P, Sa, Ta, K=1., K_T=1., fdis=1.e-6, t_hyd=0.))

    # Set the other simulation parameters
    Vj = 0.
    phi_0 = -np.pi/2.
    theta_0 = 0.
    Sj = 0.
    Tj = Ta
    cj = np.array([1.])
    tracers = ['tracer']
    dt_max = 60.
    sd_max = 3000.

    # Return the results
    return (profile, np.array([0., 0., z0]), D, Vj, phi_0, theta_0, Sj, Tj,
        cj, tracers, particles, dt_max, sd_max)

def check_sim(X0, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers,
        particles, dt_max, sd_max, bpm):
    """
    Check the results of a simulation against known values

    Check the simulation results for the simulation defined above in
    `get_sim_data()`.  This is used by the `test_simulate()` and
    `test_files()` functions below.

    Parameters
    ----------
    X0 : ndarray
        Release location in (x, y, z) (m)
    D : float
        Diameter of the release port (m)
    Vj : float
        Initial velocity of the jet (m/s)
    phi_0 : float
        Vertical angle from the horizontal for the discharge orientation
        (rad in range +/- pi/2)
    theta_0 : float
        Horizontal angle from the x-axis for the discharge orientation.
        The x-axis is taken in the direction of the ambient current.
        (rad in range 0 to 2 pi)
    Sj : float
        Salinity of the continuous phase fluid in the discharge (psu)
    Tj : float
        Temperature of the continuous phase fluid in the discharge (T)
    cj : ndarray
        Concentration of passive tracers in the discharge (user-defined)
    tracers : string list
        List of passive tracers in the discharge.  These can be chemicals
        present in the ambient `profile` data, and if so, entrainment of
        these chemicals will change the concentrations computed for these
        tracers.  However, none of these concentrations are used in the
        dissolution of the dispersed phase.  Hence, `tracers` should not
        contain any chemicals present in the dispersed phase particles.
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the
        simulation
    dt_max : float
        Maximum step size to take in the storage of the simulation
        solution (s)
    sd_max : float
        Maximum number of orifice diameters to compute the solution along
        the plume centerline (m/m)
    bpm : bent_plume_model.Model
        Bent plume model Model object that contains the whole simulation

    """
    # Check the object attributes are set correctly
    assert_array_almost_equal(bpm.X, X0, decimal=6)
    assert bpm.D == D
    assert bpm.Vj == Vj
    assert bpm.phi_0 == phi_0
    assert bpm.theta_0 == theta_0
    assert bpm.Sj == Sj
    assert bpm.Tj == Tj
    assert bpm.cj == cj
    for i in range(len(tracers)):
        assert bpm.tracers[i] == tracers[i]
    assert len(bpm.particles) == len(particles)
    assert bpm.track == True
    assert bpm.dt_max == dt_max
    assert bpm.sd_max == sd_max
    assert_array_almost_equal(bpm.K_T0, np.array([particles[i].K_T
        for i in range(len(particles))]), decimal=6)

    # Check the model output
    assert bpm.sim_stored == True
    assert bpm.t[0] == 0.
    ans = np.array([4.36126913e+00,  1.54951631e+02,  4.97776191e+06,  
        1.41352957e-16, 0.00000000e+00, -2.30846897e+00, 1.13354847e-01,
        0.00000000e+00, 0.00000000e+00, 3.00000000e+02, 0.00000000e+00,
        6.66236993e-05, 0.00000000e+00, 0.00000000e+00, 3.80206888e+01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.13354847e+00, 6.46891333e+05, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 1.73577225e-05, 0.00000000e+00,
        0.00000000e+00, 4.24115008e-03])
    for i in range(len(ans)):
        assert_approx_equal(bpm.q[0,i], ans[i], significant=6)
    assert_approx_equal(bpm.t[-1], 1147.9493273189328, significant=6)
    ans = np.array([2.61953650e+03, 9.31317467e+04, 2.99118513e+09, 
        3.92276284e+02, 0.00000000e+00, 1.01029373e+01, 1.13354847e-01,
        1.71199411e+02, 0.00000000e+00, 2.93084183e+02, 1.74083221e+02,
        6.58401430e-05, 0.00000000e+00, 0.00000000e+00, 3.75906304e+01,
        1.74241653e+01, np.nan, np.nan, np.nan, 1.13354847e+00, 6.47185740e+05,
        2.84433399e+01, np.nan, np.nan, np.nan, 1.03485668e-02, 0.00000000e+00,
        0.00000000e+00, 4.24115008e-03])
    for i in range(len(ans)):
        assert_approx_equal(bpm.q[-1,i], ans[i], significant=5)

    # Check the tracking data for the particles that left the plume
    assert bpm.particles[0].farfield == True
    ans = np.array([2.04762227e+02, 0.00000000e+00, 1.49786084e+00, 
        3.01711218e-06, 0.00000000e+00, 0.00000000e+00, 1.81160226e+00])
    for i in range(len(ans)):
        assert_approx_equal(bpm.particles[0].sbm.y[-1,i], ans[i],
            significant=6)
    ans = np.array([1.93463857e+02, 0.00000000e+00, 1.11491145e+02, 
        4.63392573e-04, 2.70044428e+02])
    for i in range(len(ans)):
        assert_approx_equal(bpm.particles[1].sbm.y[-1,i], ans[i],
            significant=6)

# ----------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------

def test_modelparams_obj():
    """
    Test the behavior of the `ModelParams` object

    Test the instantiation and attribute data for the `ModelParams object of
    the `bent_plume_model` module.

    """
    # Get the ambient CTD data
    profile = get_profile()

    # Initialize the ModelParams object
    p = bent_plume_model.ModelParams(profile)

    # Check if the attributes are set correctly
    assert_approx_equal(p.rho_r, 1031.035855535142, significant=6)
    assert p.g == 9.81
    assert p.Ru == 8.314510
    assert p.alpha_j == 0.057
    assert p.alpha_Fr == 0.544
    assert p.gamma == 1.10
    assert p.Fr_0 == 1.6

def test_particle_obj():
    """
    Test the object behavior for the `Particle` object

    Test the instantiation and attribute data for the `Particle` object of
    the `bent_plume_model` module.

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
    x = 0.
    y = 0.
    z = 0.

    # Compute a few derived quantities
    bub = dbm.FluidParticle(composition)
    nb0 = 1.e5
    m0 = bub.masses_by_diameter(de, T, P, yk)

    # Create a `PlumeParticle` object

    bub_obj = bent_plume_model.Particle(x, y, z, bub, m0, T, nb0,
        lambda_1, P, Sa, Ta, K, Kt, fdis)

    # Check if the initialized attributes are correct
    assert bub_obj.integrate == True
    assert bub_obj.sim_stored == False
    assert bub_obj.farfield == False
    assert bub_obj.t == 0.
    assert bub_obj.x == x
    assert bub_obj.y == y
    assert bub_obj.z == z
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

    # Test the bub_obj.outside() method
    bub_obj.outside(Ta, Sa, P)
    assert bub_obj.us == 0.
    assert bub_obj.rho_p == seawater.density(Ta, Sa, P)
    assert bub_obj.A == 0.
    assert_array_almost_equal(bub_obj.Cs, np.zeros(len(composition)))
    assert_array_almost_equal(bub_obj.beta, np.zeros(len(composition)))
    assert bub_obj.beta_T == 0.
    assert bub_obj.T == Ta

    # No need to test the properties or diameter objects since they are
    # inherited from the `single_bubble_model` and tested in `test_sbm`.

    # No need to test the bub_obj.track(), bub_obj.run_sbm() since they will
    # be tested below for the simulation cases.

    # Check functionality of insoluble particle
    drop = dbm.InsolubleParticle(isfluid=True, iscompressible=True)
    m0 = drop.mass_by_diameter(de, T, P, Sa, Ta)
    drop_obj = bent_plume_model.Particle(x, y, z, drop, m0, T, nb0,
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
    Test the object behavior for the `LagElement` object

    Test the instantiation and attribute data for the `LagElement` object of
    the `bent_plume_model` module.

    This test does many of the calculations in `Model.Simulate`, but does
    not perform the simulation.

    """
    # Get the model parameters
    profile, X0, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers, particles, \
        dt_max, sd_max = get_sim_data()
    p = bent_plume_model.ModelParams(profile)

    # Get the initial conditions for this plume
    t0, q0, chem_names = lmp.main_ic(profile, particles, X0, D, Vj, phi_0,
        theta_0, Sj, Tj, cj, tracers, p)

    # Create the `LagElement` object:
    q_local = bent_plume_model.LagElement(t0, q0, D, profile, p, particles,
        tracers, chem_names)
    
    # Validate the values in q_local
    assert q_local.t0 == t0
    assert_array_almost_equal(q_local.q0, q0, decimal = 6)
    assert q_local.D == D
    for i in range(len(tracers)):
        assert q_local.tracers[i] == tracers[i]
    for i in range(len(chem_names)):
        assert q_local.chem_names[i] == chem_names[i]
    assert q_local.len == q0.shape[0]
    assert q_local.nchems == len(chem_names)
    assert q_local.np == len(particles)
    assert q_local.t == t0
    assert_array_almost_equal(q_local.q, q0, decimal=6)
    assert q_local.M == q0[0]
    assert q_local.Se == q0[1]
    assert q_local.He == q0[2]
    assert q_local.Jx == q0[3]
    assert q_local.Jy == q0[4]
    assert q_local.Jz == q0[5]
    assert q_local.H == q0[6]
    assert q_local.x == q0[7]
    assert q_local.y == q0[8]
    assert q_local.z == q0[9]
    assert q_local.s == q0[10]
    assert_array_almost_equal(q_local.M_p[0], np.array([6.66236993e-05,
        0.00000000e+00, 0.00000000e+00]), decimal=6)
    assert_array_almost_equal(q_local.M_p[1], np.array([1.13354847]),
        decimal=6)
    assert_approx_equal(q_local.H_p[0], 3.80206888e+01, significant=6)
    assert_approx_equal(q_local.H_p[1], 6.46891333e+05, significant=6)
    assert_array_almost_equal(q_local.t_p, np.array([0., 0.]), decimal=6)
    assert_array_almost_equal(q_local.X_p[0], np.array([0., 0., 0.]),
        decimal=6)
    assert_array_almost_equal(q_local.X_p[1], np.array([0., 0., 0.]),
        decimal=6)
    assert_array_almost_equal(q_local.cpe, np.array([1.73577225e-05,
        0.00000000e+00, 0.00000000e+00]), decimal=6)
    assert_array_almost_equal(q_local.cte, np.array([0.00424115]), decimal=6)
    assert_approx_equal(q_local.Ta, 285.52466101019053, significant=6)
    assert_approx_equal(q_local.Sa, 35.52902290651307, significant=6)
    assert_approx_equal(q_local.Pa, 3123785.3190075322, significant=6)
    assert_approx_equal(q_local.ua, 0.15, significant=6)
    assert_approx_equal(q_local.va, 0., significant=6)
    assert_approx_equal(q_local.wa, 0., significant=6)
    assert_approx_equal(q_local.rho_a, 1028.32228185795, significant=6)
    assert_array_almost_equal(q_local.ca_chems, np.array([0.00409269, 0.,
        0.]), decimal=6)
    assert_approx_equal(q_local.S, 35.52902290651307, significant=6)
    assert_approx_equal(q_local.T, 285.52466101019053, significant=6)
    assert_approx_equal(q_local.rho, 1028.32228185795, significant=6)
    assert_array_almost_equal(q_local.c_chems, np.array([0.00409269, 0., 0.]),
        decimal=6)
    assert_array_almost_equal(q_local.c_tracers, np.array([1.]),
        decimal=6)
    assert_approx_equal(q_local.u, 3.241096864109426e-17, significant=6)
    assert_approx_equal(q_local.v, 0., significant=6)
    assert_approx_equal(q_local.w, -0.5293112865466196, significant=6)
    assert_approx_equal(q_local.hvel, 3.241096864109426e-17, significant=6)
    assert_approx_equal(q_local.V, 0.5293112865466196, significant=6)
    assert_approx_equal(q_local.h, 0.060000000000000005, significant=6)
    assert_approx_equal(q_local.b, 0.15000000000000002, significant=6)
    assert_approx_equal(q_local.sin_p, -1., significant=6)
    assert_approx_equal(q_local.cos_p, 6.123233995736766e-17, significant=6)
    assert_approx_equal(q_local.sin_t, 0., significant=6)
    assert_approx_equal(q_local.cos_t, 1., significant=6)
    assert_approx_equal(q_local.phi, -1.5707963267948966, significant=6)
    assert_approx_equal(q_local.theta, 0., significant=6)
    assert_approx_equal(q_local.fb[0], 1.55882441, significant=6)
    assert_approx_equal(q_local.fb[1], 188.75058126, significant=6)
    assert_array_almost_equal(q_local.mp, np.array([6.66236993e-05,
        1.13354847e+00]), decimal=6)
    assert_array_almost_equal(q_local.x_p, np.array([[0., 0., 300.],[0., 0.,
        300.]]), decimal=6)
    assert_array_almost_equal(q_local.t_p, np.array([ 0.,  0.]), decimal=6)
    assert_approx_equal(q_local.Fb, 190.3094056738958, significant=6)

def test_simulate():
    """
    Test the `Model.simulate()` method of the Bent Plume Model

    Run a simulation to test the operation of the `Model.simulate()` method
    of the Bent Plume Model.

    """
    # Get the model parameters
    profile, X0, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers, particles, \
        dt_max, sd_max = get_sim_data()

    # Initialize a stratified plume model `Model` object
    bpm = bent_plume_model.Model(profile)

    # Run the simulation
    bpm.simulate(X0, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers,
                 particles, track=True, dt_max=dt_max, sd_max=sd_max)

    # Check that the results are correct
    check_sim(X0, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers, particles,
        dt_max, sd_max, bpm)

def test_files():
    """
    Test the input and output of model simulation data

    Test the methods `save_sim`, `save_txt`, and `load_sim` of the Bent
    Plume Model `Model` object.

    """
    # Get the model parameters
    profile, X0, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers, particles, \
        dt_max, sd_max = get_sim_data()

    # Initialize a stratified plume model `Model` object
    bpm = bent_plume_model.Model(profile)

    # Run the simulation
    bpm.simulate(X0, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers,
        particles, track=True, dt_max=dt_max, sd_max=sd_max)

    # Save the simulation to a netCDF file
    fname = os.path.join(OUTPUT_DIR, 'bpm_data.nc')
    profile_path = os.path.join(OUTPUT_DIR, 'test_BM54.nc')
    profile_info = 'Results of test_bpm.py script'
    bpm.save_sim(fname, profile_path, profile_info)

    # Save the simulation state-space to a text file
    base_name = os.path.join(OUTPUT_DIR, 'bpm_data')
    bpm.save_txt(base_name, profile_path, profile_info)

    # Save the simulation derived variables to a text file
    base_name = os.path.join(OUTPUT_DIR, 'bpm_derived_vars')
    bpm.save_derived_variables(base_name)
    
    # Load the simulation data from the netCDF file
    bpm.load_sim(fname)
    print('Checking simulation...')
    check_sim(X0, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers, particles,
        dt_max, sd_max, bpm)

    # Initialize a Model object from the netCDF file
    bpm_load = bent_plume_model.Model(simfile = fname)
    print('Checking simulation...')
    check_sim(X0, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers, particles,
        dt_max, sd_max, bpm)

