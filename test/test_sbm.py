"""
Unit tests for the `single_bubble_model` module of ``TAMOC``

Provides testing of the objects, methods and functions defined in the 
`single_bubble_model` module of ``TAMOC``.  This module uses the `ambient`
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
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from tamoc import seawater
from tamoc import ambient
import test_ambient
from tamoc import dbm
from tamoc import dispersed_phases
from tamoc import single_bubble_model 

from netCDF4 import Dataset

import os
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_approx_equal

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def make_ctd_file():
    """
    Creates a netCDF dataset of ambient CTD data
    
    Creates a netCDF dataset of ambient CTD data for the data in the file
    `./data/ctd_BM54.cnv`.  This function first tries to load the data from
    `./test/output/test_bm54.nc`.  If that fails, it builds the dataset
    using `test_ambient.test_from_ctd`.  
    
    Returns
    -------
    nc : netCDF4.Dataset object
        A netCDF4 dataset object containing the CTD data.
    
    """
    # Get the correct netCDF4 dataset
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    'output'))
    nc_file = os.path.join(__location__,'test_BM54.nc')
    
    # Be sure to start with the original, unedited CTD data
    test_ambient.test_from_ctd()
    
    # Load the data into a netCDF file
    nc = Dataset(nc_file, 'a')
    
    return nc
    
def get_profile():
    """
    Create an `ambient.Profile` object from a netCDF file
    
    Create the `ambient.Profile` object needed by the `single_bubble_model`
    simulation from the netCDF file `./test/output/test_bm54.nc`.  This 
    function calls `make_ctd_file` to create the netCDF dataset before 
    using it to create the `ambient.Profile` object.
    
    Returns
    -------
    profile : `ambient.Profile` objecty
        Return a profile object from the BM54 CTD data
    
    """
    # Create the netCDF file
    nc = make_ctd_file()
    
    # Return a profile object with all available chemicals in the CTD data
    return ambient.Profile(nc, chem_names='all')


# ----------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------

def test_modelparams_obj():
    """
    Test the object behavior for the `ModelParams` object
    
    Test the instantiation and attribute data for the `ModelParams` object of 
    the `single_bubble_model` module.
    
    """
    # Get the ambient CTD data
    profile = get_profile()
    
    # Initialize a ModelParams object
    p = single_bubble_model.ModelParams(profile)
    
    # Check if the reference density is correct
    assert_approx_equal(p.rho_r, 1031.035855535142, significant=6) 


def test_particle_obj():
    """
    Test the object behavior for the `Particle` object
    
    Test the instantiation and attribute data for the `Particle` object of 
    the `single_bubble_model` module.
    
    """
    # Set up the base parameters describing a particle object
    T = 273.15 + 15.
    P = 150e5
    Sa = 35.
    Ta = 273.15 + 4.
    composition = ['methane', 'ethane', 'propane', 'oxygen']
    yk = np.array([0.85, 0.07, 0.08, 0.0])
    de = 0.005
    K = 1.
    Kt = 1.
    fdis = 1e-6
    
    # Compute a few derived quantities
    bub = dbm.FluidParticle(composition)
    m0 = bub.masses_by_diameter(de, T, P, yk)
    
    # Create a `SingleParticle` object
    bub_obj = dispersed_phases.SingleParticle(bub, m0, T, K, fdis=fdis, 
              K_T=Kt)
    
    # Check if the initial attributes are correct
    for i in range(len(composition)):
        assert bub_obj.composition[i] == composition[i]
    assert_array_almost_equal(bub_obj.m0, m0, decimal=6)
    assert bub_obj.T0 == T
    assert bub_obj.cp == seawater.cp() * 0.5
    assert bub_obj.K == K
    assert bub_obj.K_T == Kt
    assert bub_obj.fdis == fdis
    for i in range(len(composition)-1):
        assert bub_obj.diss_indices[i] == True
    assert bub_obj.diss_indices[-1] == False
    
    # Check if the values returned by the `properties` method match the input
    (us, rho_p, A, Cs, beta, beta_T, T_ans) = bub_obj.properties(m0, T, P, 
        Sa, Ta, 0.)
    us_ans = bub.slip_velocity(m0, T, P, Sa, Ta)
    rho_p_ans = bub.density(m0, T, P)
    A_ans = bub.surface_area(m0, T, P, Sa, Ta)
    Cs_ans = bub.solubility(m0, T, P, Sa)
    beta_ans = bub.mass_transfer(m0, T, P, Sa, Ta)
    beta_T_ans = bub.heat_transfer(m0, T, P, Sa, Ta)
    assert us == us_ans
    assert rho_p == rho_p_ans
    assert A == A_ans
    assert_array_almost_equal(Cs, Cs_ans, decimal=6)
    assert_array_almost_equal(beta, beta_ans, decimal=6)
    assert beta_T == beta_T_ans
    assert T == T_ans
    
    # Check that dissolution shuts down correctly
    m_dis = np.array([m0[0]*1e-10, m0[1]*1e-8, m0[2]*1e-3, 1.5e-5])
    (us, rho_p, A, Cs, beta, beta_T, T_ans) = bub_obj.properties(m_dis, T, P, 
        Sa, Ta, 0)
    assert beta[0] == 0.
    assert beta[1] == 0.
    assert beta[2] > 0.
    assert beta[3] > 0.
    m_dis = np.array([m0[0]*1e-10, m0[1]*1e-8, m0[2]*1e-7, 1.5e-16])
    (us, rho_p, A, Cs, beta, beta_T, T_ans) = bub_obj.properties(m_dis, T, P, 
        Sa, Ta, 0.)
    assert np.sum(beta[0:-1]) == 0.
    assert us == 0.
    assert rho_p == seawater.density(Ta, Sa, P)
    
    # Check that heat transfer shuts down correctly
    (us, rho_p, A, Cs, beta, beta_T, T_ans) = bub_obj.properties(m_dis, Ta, P, 
        Sa, Ta, 0)
    assert beta_T == 0.
    (us, rho_p, A, Cs, beta, beta_T, T_ans) = bub_obj.properties(m_dis, T, P, 
        Sa, Ta, 0)
    assert beta_T == 0.
    
    # Check the value returned by the `diameter` method
    de_p = bub_obj.diameter(m0, T, P, Sa, Ta)
    assert_approx_equal(de_p, de, significant=6)
    
    # Check functionality of insoluble particle 
    drop = dbm.InsolubleParticle(isfluid=True, iscompressible=True)
    m0 = drop.mass_by_diameter(de, T, P, Sa, Ta)
    
    # Create a `Particle` object
    drop_obj = dispersed_phases.SingleParticle(drop, m0, T, K, fdis=fdis, 
               K_T=Kt)
    
    # Check if the values returned by the `properties` method match the input
    (us, rho_p, A, Cs, beta, beta_T, T_ans) = drop_obj.properties(
        np.array([m0]), T, P, Sa, Ta, 0)
    us_ans = drop.slip_velocity(m0, T, P, Sa, Ta)
    rho_p_ans = drop.density(T, P, Sa, Ta)
    A_ans = drop.surface_area(m0, T, P, Sa, Ta)
    beta_T_ans = drop.heat_transfer(m0, T, P, Sa, Ta)
    assert us == us_ans
    assert rho_p == rho_p_ans
    assert A == A_ans
    assert beta_T == beta_T_ans
    
    # Check that heat transfer shuts down correctly
    (us, rho_p, A, Cs, beta, beta_T, T_ans) = drop_obj.properties(m_dis, Ta, P, 
        Sa, Ta, 0)
    assert beta_T == 0.
    (us, rho_p, A, Cs, beta, beta_T, T_ans) = drop_obj.properties(m_dis, T, P, 
        Sa, Ta, 0)
    assert beta_T == 0.
    
    # Check the value returned by the `diameter` method
    de_p = drop_obj.diameter(m0, T, P, Sa, Ta)
    assert_approx_equal(de_p, de, significant=6)


def test_ic():
    """
    Test the initial conditions function for the single bubble model
    
    Test that the initial conditions returned by `sbm_ic` are correct based
    on the input and expected output
    
    """
    # Set up the inputs
    profile = get_profile()
    T0 = 273.15 + 15.
    z0 = 1500.
    P = profile.get_values(z0, ['pressure'])
    composition = ['methane', 'ethane', 'propane', 'oxygen']
    bub = dbm.FluidParticle(composition)
    yk = np.array([0.85, 0.07, 0.08, 0.0])
    de = 0.005
    K = 1.
    K_T = 1.
    fdis = 1.e-4
    t_hyd = 0.
    
    # Get the initial conditions
    (bub_obj, y0) = single_bubble_model.sbm_ic(profile, bub, 
                    np.array([0., 0., z0]), de, yk, T0, K, K_T, fdis, t_hyd)
    
    # Check the initial condition values
    assert y0[0] == 0.
    assert y0[1] == 0.
    assert y0[2] == z0
    assert y0[-1] == T0 * np.sum(y0[3:-1]) * seawater.cp() * 0.5
    assert_approx_equal(bub.diameter(y0[3:-1], T0, P), de, significant=6)
    
    # Check the bub_obj parameters
    for i in range(len(composition)):
        assert bub_obj.composition[i] == composition[i]
    assert bub_obj.T0 == T0
    assert bub_obj.cp == seawater.cp() * 0.5
    assert bub_obj.K == K
    assert bub_obj.K_T == K_T
    assert bub_obj.fdis == fdis
    assert bub_obj.t_hyd == t_hyd
    for i in range(len(composition)-1):
        assert bub_obj.diss_indices[i] == True
    assert bub_obj.diss_indices[-1] == False


def test_model_obj():
    """
    Test the object behavior for the `Model` object
    
    Test the instantiation and attribute data for the `Model` object of 
    the `single_bubble_model` module.
    
    Notes
    -----
    This test function only tests instantiation from a netCDF file of ambient
    CTD data and does not test any of the object methods.  Instantiation 
    from simulation data and testing of the object methods is done in the 
    remaining test functions.
    
    See Also
    --------
    test_simulation
    
    """
    # Get the ambient profile data
    profile = get_profile()
    
    # Initialize a Model object
    sbm = single_bubble_model.Model(profile)
    
    # Check the model attributes
    assert_approx_equal(sbm.p.rho_r, 1031.035855535142, significant=6) 
    (T, S, P) = profile.get_values(1000., ['temperature', 'salinity', 
                                   'pressure'])
    (Tp, Sp, Pp) = sbm.profile.get_values(1000., ['temperature', 'salinity', 
                                   'pressure'])
    assert Tp == T
    assert Sp == S
    assert Pp == P


def test_simulation():
    """
    Test the output from the `Model.simulate` method
    
    Test the output of the `Model.simulate` method of the 
    `single_bubble_model` module.  These tests include a single component 
    bubble, multi-component bubbles and drops, and multi-component fluid
    particles with gas stripping from the ambient dissolved chemicals.
    
    """
    # Get the ambient profile data
    profile = get_profile()
    
    # Initialize a Model object
    sbm = single_bubble_model.Model(profile)
    
    # Set up the initial conditions
    composition = ['methane', 'ethane', 'propane', 'oxygen']
    bub = dbm.FluidParticle(composition)
    mol_frac = np.array([0.90, 0.07, 0.03, 0.0])
    de = 0.005
    z0 = 1000.
    T0 = 273.15 + 30.
    
    # Run the simulation
    sbm.simulate(bub, np.array([0., 0., z0]), de, mol_frac, T0, K_T=1, 
        fdis=1e-8, delta_t=10.)
    
    # Check the solution
    assert sbm.y.shape[0] == 1117
    assert sbm.y.shape[1] == 8
    assert sbm.t.shape[0] == 1117
    assert_approx_equal(sbm.t[-1], 11016.038751523512, significant = 6)
    assert_array_almost_equal(sbm.y[-1,:], np.array([0.00000000e+00,   
        0.00000000e+00, 3.36934635e+02, 4.31711152e-14, 6.66318106e-15,  
        -2.91389824e-13, -1.08618680e-15, -1.37972400e-07]), decimal = 6)
    
    # Write the output files
    sbm.save_sim('./output/sbm_data.nc', './test_BM54.nc', 
                 'Results of ./test_sbm.py script')
    
    sbm.save_txt('./output/sbm_data', './test_BM54.nc', 
                 'Results of ./test_sbm.py script')
    
    # Reload the simulation
    sbm_f = single_bubble_model.Model(simfile = './output/sbm_data.nc')
    
    # Check that the attributes are loaded correctly
    assert sbm_f.y[0,0] == sbm.y[0,0]  # x0
    assert sbm_f.y[0,1] == sbm.y[0,1]  # y0
    assert sbm_f.y[0,2] == sbm.y[0,2]  # z0
    
    assert_array_almost_equal(sbm_f.particle.m0, sbm.particle.m0, decimal = 6)
    assert sbm_f.particle.T0 == sbm.particle.T0
    print sbm_f.particle.K_T, sbm.particle.K_T
    assert sbm_f.particle.K == sbm.particle.K
    assert sbm_f.particle.K_T == sbm.particle.K_T
    assert sbm_f.particle.fdis == sbm.particle.fdis
    assert sbm_f.K_T0 == sbm.K_T0
    assert sbm_f.delta_t == sbm.delta_t
    (T, S, P) = profile.get_values(1000., ['temperature', 'salinity', 
                                   'pressure'])
    (Tp, Sp, Pp) = sbm_f.profile.get_values(1000., ['temperature', 'salinity', 
                                   'pressure'])
    assert Tp == T
    assert Sp == S
    assert Pp == P
    
    # Check that the results are still correct
    assert sbm.y.shape[0] == 1117
    assert sbm.y.shape[1] == 8
    assert sbm.t.shape[0] == 1117
    assert_approx_equal(sbm.t[-1], 11016.038751523512, significant = 6)
    assert_array_almost_equal(sbm.y[-1,:], np.array([0.00000000e+00,   
        0.00000000e+00, 3.36934635e+02, 4.31711152e-14, 6.66318106e-15,  
        -2.91389824e-13, -1.08618680e-15, -1.37972400e-07]), decimal = 6)
    
    # Load the data in the txt file and check the solution
    data = np.loadtxt('./output/sbm_data.txt')
    assert sbm.y.shape[0] == 1117
    assert sbm.y.shape[1] == 8
    assert sbm.t.shape[0] == 1117
    assert_approx_equal(sbm.t[-1], 11016.038751523512, significant = 6)
    assert_array_almost_equal(sbm.y[-1,:], np.array([0.00000000e+00,   
        0.00000000e+00, 3.36934635e+02, 4.31711152e-14, 6.66318106e-15,  
        -2.91389824e-13, -1.08618680e-15, -1.37972400e-07]), decimal = 6)
    
    # Create an inert particle that is compressible
    oil = dbm.InsolubleParticle(True, True, rho_p=840.)
    mol_frac = np.array([1.])
    
    # Specify the remaining particle initial conditions
    de = 0.03
    z0 = 1000.
    T0 = 273.15 + 30.
    
    # Simulate the trajectory through the water column and plot the results
    sbm.simulate(oil, np.array([0., 0., z0]), de, mol_frac, T0, K_T=1, 
        delta_t=10.)
    ans = np.array([0.00000000e+00, 0.00000000e+00, 1.16067764e-01,
        1.26136097e-02, 7.57374681e+03])
    for i in range(5):
        assert_approx_equal(sbm.y[-1,i], ans[i], significant = 6)
    

