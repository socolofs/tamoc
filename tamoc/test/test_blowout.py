"""
Unit tests for the `blowout` module of ``TAMOC``

Provides testing of the class definition, methods, and functions in the
`blowout` module of ``TAMOC``.  These tests check the behavior of the class
object, the results of simulations, and the related object methods.

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
# S. Socolofsky, March 2020, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient, blowout

from tamoc.test import test_sbm

import numpy as np
from numpy.testing import assert_approx_equal
from numpy.testing import assert_array_almost_equal

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def get_ctd():
    """
    Provide the ambient CTD data

    Load the required CTD data from the ./test/output/test_BM54.nc dataset
    and include water currents.

    Returns
    -------
    profile : `ambient.Profile` object
        An `ambient.Profile` object containing the required CTD data and
        currents for a `bent_plume_model` simulation.

    """
    # Get the CTD data from the requested file
    nc = test_sbm.make_ctd_file()
    profile = ambient.Profile(nc, chem_names='all')

    # Add the ambient currents
    z = profile.nc.variables['z'][:]
    ua = np.zeros(z.shape) + 0.09
    data = np.vstack((z, ua)).transpose()
    symbols = ['z', 'ua']
    units = ['m', 'm/s']
    comments = ['measured', 'arbitrary crossflow velocity']
    profile.append(data, symbols, units, comments, 0)
    profile.close_nc()

    # Return the profile object
    return profile

def get_blowout():
    """
    Create the `blowout.Blowout` object for a basic case

    Define the parameters for a basic, synthetic accidental subsea oil well
    blowout

    Returns
    -------
    z0 : float, default=100
        Depth of the release point (m)
    d0 : float, default=0.1
        Equivalent circular diameter of the release (m)
    substance : str or list of str, default=['methane']
        The chemical composition of the released petroleum fluid.  If using
        the chemical property data distributed with TAMOC, this should be a
        list of TAMOC chemical property names.  If using an oil from the
        NOAA OilLibrary, this should be a string containing the Adios oil
        ID number (e.g., 'AD01554' for Louisiana Light Sweet).
    q_oil : float, default=20000.
        Release rate of the dead oil composition at the release point in
        stock barrels of oil per day.
    gor : float, default=0.
        Gas to oil ratio at standard surface conditions in standard cubic
        feet per stock barrel of oil
    x0 : float, default=0
        x-coordinate of the release (m)
    y0 : float, default=0
        y-coordinate of the release (m)
    u0 : float, default=None
        Exit velocity of continuous-phase fluid at the release.  This is
        only used when produced water exits.  For a pure oil and gas release,
        this should be zero or None.
    phi_0 : float, default=-np.pi / 2. (vertical release)
        Vertical angle of the release relative to the horizontal plane; z is
        positive down so that -pi/2 represents a vertically upward flowing
        release (rad)
    theta_0 : float, default=0.
        Horizontal angle of the release relative to the x-direction (rad)
    num_gas_elements : int, default=10
        Number of gas bubble sizes to include in the gas bubble size
        distribution
    num_oil_elements : int, default=25
        Number of oil droplet sizes to include in the oil droplet size
        distribution
    water : various
        Data describing the ambient water temperature and salinity profile.
        See Notes below for details.
    current : various
        Data describing the ambient current velocity profile.  See Notes
        below for details.

    """
    # Define some typical blowout values
    z0 = 100.
    d0 = 0.2
    substance = {'composition' : ['methane', 'ethane', 'propane',
                                    'toluene', 'benzene'],
                 'masses' : np.array([0.2, 0.03, 0.02, 0.25, 0.5])}
    q_oil = 20000.
    gor = 1000.
    x0 = 0.
    y0 = 0.
    u0 = 0.
    phi_0 = -np.pi / 2.
    theta_0 = 0.
    num_gas_elements = 10
    num_oil_elements = 10
    water = None
    current = np.array([0.09, 0.0, 0.0])

    return (z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0,
            num_gas_elements, num_oil_elements, water, current)

def check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill):
    """
    Check that the attributes in the `blowout.Blowout` object contain the
    correct values

    Parameters
    ----------
    z0 : float, default=100
        Depth of the release point (m)
    d0 : float, default=0.1
        Equivalent circular diameter of the release (m)
    substance : str or list of str, default=['methane']
        The chemical composition of the released petroleum fluid.  If using
        the chemical property data distributed with TAMOC, this should be a
        list of TAMOC chemical property names.  If using an oil from the
        NOAA OilLibrary, this should be a string containing the Adios oil
        ID number (e.g., 'AD01554' for Louisiana Light Sweet).
    q_oil : float, default=20000.
        Release rate of the dead oil composition at the release point in
        stock barrels of oil per day.
    gor : float, default=0.
        Gas to oil ratio at standard surface conditions in standard cubic
        feet per stock barrel of oil
    x0 : float, default=0
        x-coordinate of the release (m)
    y0 : float, default=0
        y-coordinate of the release (m)
    u0 : float, default=None
        Exit velocity of continuous-phase fluid at the release.  This is
        only used when produced water exits.  For a pure oil and gas release,
        this should be zero or None.
    phi_0 : float, default=-np.pi / 2. (vertical release)
        Vertical angle of the release relative to the horizontal plane; z is
        positive down so that -pi/2 represents a vertically upward flowing
        release (rad)
    theta_0 : float, default=0.
        Horizontal angle of the release relative to the x-direction (rad)
    num_gas_elements : int, default=10
        Number of gas bubble sizes to include in the gas bubble size
        distribution
    num_oil_elements : int, default=25
        Number of oil droplet sizes to include in the oil droplet size
        distribution
    water : various
        Data describing the ambient water temperature and salinity profile.
        See Notes below for details.
    current : various
        Data describing the ambient current velocity profile.  See Notes
        below for details.
    spill : `blowout.Blowout` object
        A `blowout.Blowout` object that contains the specified input data

    """
    # Check each of the object attributes in the __init__() method
    assert spill.z0 == z0
    assert spill.d0 == d0
    for i in range(len(substance['composition'])):
        assert spill.substance['composition'][i] == \
            substance['composition'][i]
        assert spill.substance['masses'][i] == substance['masses'][i]
    assert spill.q_oil == q_oil
    assert spill.gor == gor
    assert spill.x0 == x0
    assert spill.y0 == y0
    assert spill.u0 == u0
    assert spill.phi_0 == phi_0
    assert spill.theta_0 == theta_0
    assert spill.num_gas_elements == num_gas_elements
    assert spill.num_oil_elements == num_oil_elements
    if water == None:
        assert spill.water == water
    else:
        assert isinstance(spill.water, ambient.Profile)
    if isinstance(current, float):
        assert spill.current == current
    else:
        assert_array_almost_equal(spill.current, current, decimal=6)


def check_simulation(spill):
    """
    Compare the simulation solution stored in spill to the expected solution

    Parameters
    ----------
    spill : `blowout.Blowout` object
        A `blowout.Blowout` object that contains a simulation run already
        completed

    """
    # Check the model parameters
    assert spill.update == False
    assert spill.bpm.sim_stored == True

    # Check that the object attributes are set properly
    assert_array_almost_equal(spill.bpm.X, np.array([spill.x0, spill.y0,
        spill.z0]), decimal=6)
    assert spill.bpm.D == spill.d0
    assert spill.bpm.Vj == spill.u0
    assert spill.bpm.phi_0 == spill.phi_0
    assert spill.bpm.theta_0 == spill.theta_0
    assert spill.bpm.Sj == spill.Sj
    assert spill.bpm.Tj == spill.Tj
    assert spill.bpm.cj == spill.cj
    for i in range(len(spill.tracers)):
        assert spill.bpm.tracers[i] == spill.tracers[i]
    assert len(spill.bpm.particles) == len(spill.disp_phases)
    assert spill.bpm.track == spill.track
    assert spill.bpm.dt_max == spill.dt_max
    assert spill.bpm.sd_max == spill.sd_max
    assert_array_almost_equal(spill.bpm.K_T0,
        np.array([spill.disp_phases[i].K_T
        for i in range(len(spill.disp_phases))]), decimal=6)

    # Check the model simulation results
    q0 = np.array([
        1.29037789e+00,  4.52019374e+01,  1.47755395e+06,  3.69108722e-15,
        0.00000000e+00, -6.02800289e+01,  8.56255653e-04,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+02,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.09302629e-04,  1.11480306e-05,  3.64591502e-06,  2.07244046e-07,
        1.42962222e-06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  7.19858668e+01,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.06225097e-04,
        3.12326136e-05,  1.02144907e-05,  5.80620330e-07,  4.00526692e-06,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.01677484e+02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  6.61440772e-04,  6.74618908e-05,
        2.20631185e-05,  1.25412960e-06,  8.65130542e-06,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.35619783e+02,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.10149149e-03,  1.12343693e-04,  3.67415169e-05,
        2.08849098e-06,  1.44069427e-05,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  7.25433789e+02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.41420165e-03,  1.44237732e-04,  4.71723241e-05,  2.68140735e-06,
        1.84970309e-05,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  9.31382280e+02,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.39985011e-03,
        1.42773985e-04,  4.66936120e-05,  2.65419601e-06,  1.83093201e-05,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        9.21930467e+02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.06829633e-03,  1.08958040e-04,
        3.56342540e-05,  2.02555105e-06,  1.39727671e-05,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  7.03571710e+02,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  6.28553057e-04,  6.41075959e-05,  2.09661109e-05,
        1.19177261e-06,  8.22115101e-06,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  4.13960188e+02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.85122937e-04,  2.90803549e-05,  9.51060382e-06,  5.40609424e-07,
        3.72926150e-06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.87779763e+02,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.97154314e-05,
        1.01702100e-05,  3.32612302e-06,  1.89066171e-07,  1.30422661e-06,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        6.56718128e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  4.98777791e-06,  3.72494741e-06,
        4.63112076e-06,  8.36251140e-05,  1.66674352e-04,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.50943077e+02,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  9.21816811e-06,  6.88426631e-06,  8.55901174e-06,
        1.54551861e-04,  3.08039416e-04,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  2.78965640e+02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.68511998e-05,  1.25847289e-05,  1.56462341e-05,  2.82527315e-04,
        5.63109034e-04,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  5.09960945e+02,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.01813740e-05,
        2.25399032e-05,  2.80232180e-05,  5.06021096e-04,  1.00855753e-03,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        9.13366541e+02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  5.20333042e-05,  3.88592526e-05,
        4.83125992e-05,  8.72390687e-04,  1.73877374e-03,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.57466254e+03,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  8.35592123e-05,  6.24032740e-05,  7.75842087e-05,
        1.40095425e-03,  2.79226096e-03,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  2.52871816e+03,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.17656634e-04,  8.78677406e-05,  1.09243453e-04,  1.97263183e-03,
        3.93167932e-03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  3.56059446e+03,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.30166694e-04,
        9.72104406e-05,  1.20858965e-04,  2.18237556e-03,  4.34972240e-03,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        3.93918125e+03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  9.32233035e-05,  6.96205624e-05,
        8.65572573e-05,  1.56298246e-03,  3.11520159e-03,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.82117859e+03,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  3.11693104e-05,  2.32777089e-05,  2.89405108e-05,
        5.22584843e-04,  1.04157096e-03,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  9.43264054e+02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.29037789e+00])
    qn = np.array([ 
        5.13446051e+02, 1.79514291e+04, 5.92952938e+08, 4.60940105e+01,
        0.00000000e+00, -1.19996563e+03, 8.56255653e-04, 4.30447790e+00,
        0.00000000e+00, -5.02658383e+00, 1.05121062e+02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.03192528e-04, 1.04079855e-05, 3.48092795e-06, 4.22943393e-09,
        1.60501483e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 6.76174348e+01, 4.35204659e+01, 0.00000000e+00,
        2.91032647e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.92374138e-04,
        2.95532013e-05, 9.84089672e-06, 2.29194433e-08, 8.29546826e-08,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.91632333e+02, 4.35823334e+01, 0.00000000e+00, 2.87233761e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 6.28551396e-04, 6.34122009e-05,
        2.11461661e-05, 3.36314665e-08, 1.40721865e-07, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.11867919e+02,
        4.35662510e+01, 0.00000000e+00, 2.90550330e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.04422375e-03, 1.05230143e-04, 3.51161538e-05,
        8.17294213e-09, -1.01586808e-08, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 6.84004809e+02, 4.35284591e+01,
        0.00000000e+00, 2.95489341e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.33984495e-03, 1.34952898e-04, 4.50395667e-05, -1.29117803e-09,
        -2.17284742e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 8.77591772e+02, 4.34476786e+01, 0.00000000e+00,
        3.03639648e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.30136266e-03,
        1.30427131e-04, 4.38315686e-05, -5.18177726e-08, -6.64680532e-07,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        8.52060331e+02, 4.32124931e+01, 0.00000000e+00, 3.22069284e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 9.83687405e-04, 9.83546818e-05,
        3.31705979e-05, -4.01571080e-08, -3.49550436e-07, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.43951432e+02,
        4.29034986e+01, 0.00000000e+00, 3.43599977e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.64027119e-04, 5.60691947e-05, 1.90963943e-05,
        7.72354973e-09, 4.32600594e-08, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 3.69115144e+02, 4.24253646e+01,
        0.00000000e+00, 3.70370233e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.60364477e-04, 2.59903152e-05, 8.79429763e-06, 4.03972899e-09,
        2.41652607e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.70442683e+02, 4.20396249e+01, 0.00000000e+00,
        3.95022755e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.24622810e-05,
        9.26338972e-06, 3.11651509e-06, 1.52770676e-09, 9.91826459e-09,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        6.05450845e+01, 4.16356124e+01, 0.00000000e+00, 4.21051189e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 1.58972982e-06, 3.10260211e-06,
        4.48708496e-06, 8.29233234e-05, 1.60590418e-04, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.45911289e+02,
        4.82599816e+01, 0.00000000e+00, 1.50120408e-02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 4.24615716e-06, 6.08045116e-06, 8.37719910e-06,
        1.53664242e-04, 3.00304893e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 2.72933066e+02, 4.80937056e+01,
        0.00000000e+00, 2.43853561e-02, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        9.98991456e-06, 1.15722255e-05, 1.54208391e-05, 2.81424711e-04,
        5.53465675e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.03441280e+02, 4.78444242e+01, 0.00000000e+00,
        3.85096090e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.13979431e-05,
        2.13285164e-05, 2.77566167e-05, 5.04714611e-04, 9.97099761e-04,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        9.07883497e+02, 4.75290806e+01, 0.00000000e+00, 5.64824124e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 4.13725581e-05, 3.74530194e-05,
        4.80053768e-05, 8.70883053e-04, 1.72552849e-03, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.57246768e+03,
        4.70288999e+01, 0.00000000e+00, 8.51401019e-02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 7.17407787e-05, 6.08909490e-05, 7.72554275e-05,
        1.39933913e-03, 2.77805437e-03, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 2.53332454e+03, 4.65320117e+01,
        0.00000000e+00, 1.13853536e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.06544228e-04, 8.64757771e-05, 1.08941873e-04, 1.97114921e-03,
        3.91862716e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 3.57526305e+03, 4.60119426e+01, 0.00000000e+00,
        1.44168731e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.22130007e-04,
        9.62180545e-05, 1.20644446e-04, 2.18132038e-03, 4.34042793e-03,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        3.96156171e+03, 4.54471865e+01, 0.00000000e+00, 1.77390568e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 8.97627886e-05, 6.91977300e-05,
        8.64660103e-05, 1.56253347e-03, 3.11124501e-03, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.84047084e+03,
        4.53757193e+01, 0.00000000e+00, 1.81376690e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 3.04926253e-05, 2.31955598e-05, 2.89228013e-05,
        5.22497681e-04, 1.04080266e-03, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 9.50389977e+02, 4.54682223e+01,
        0.00000000e+00, 1.75923738e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        5.33889044e-04, 6.73109464e-05, 1.54138879e-05, 2.36193111e-05,
        1.82894425e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.29037789e+00])
    assert spill.bpm.t[0] == 0.
    for i in range(len(q0)):
        assert_approx_equal(spill.bpm.q[0,i], q0[i], significant=6)
    assert_approx_equal(spill.bpm.t[-1], 48.52956775079609, significant=6)
    for i in range(len(qn)):
        assert_approx_equal(spill.bpm.q[-1,i], qn[i], significant=3)

    # Check tracking data for a particle outside the plume
    assert spill.bpm.particles[0].farfield == False


# ----------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------

def test_Blowout_inst():
    """
    Test instantiation of a `blowout.Blowout` object

    Test that the initializer of the `blowout.Blowout` class creates the
    correct object

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    print(water)
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Check the object attributes set by the __init__() method
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

    # Check the attributes not accessible to the user
    assert spill.new_oil == False
    assert spill.Sj == 0.
    assert spill.cj == 1.
    assert spill.tracers[0] == 'tracer'
    assert len(spill.disp_phases) == num_gas_elements + num_oil_elements
    assert spill.track == True
    assert spill.dt_max == 5. * 3600.
    assert spill.sd_max == 300. * z0 / d0
    assert spill.update == True

    # Check the CTD data
    T, S, P, ua, va = spill.profile.get_values(z0, ['temperature',
                      'salinity', 'pressure', 'ua', 'va'])
    assert_approx_equal(spill.T0, T, significant=6)
    assert_approx_equal(spill.S0, S, significant=6)
    assert_approx_equal(spill.P0, P, significant=6)
    return spill
    assert_approx_equal(T, 286.45, significant=6)
    assert_approx_equal(S, 35.03, significant=6)
    assert_approx_equal(P, 1107655.378995259, significant=6)
    assert_approx_equal(ua, 0.09, significant=6)
    assert_approx_equal(va, 0., significant=6)

    # Check the bubble and droplet sizes
    de_gas = np.array([0.00367319, 0.00421546, 0.0048378 , 0.00555201,
       0.00637166, 0.00731231, 0.00839184, 0.00963073, 0.01105253,
       0.01268423])
    vf_gas = np.array([0.01545088, 0.0432876 , 0.09350044, 0.15570546,
        0.19990978, 0.19788106, 0.15101303, 0.08885147, 0.04030462,
        0.01409565])
    de_oil = np.array([0.00044767, 0.00063414, 0.00089826, 0.00127239,
        0.00180236, 0.00255306, 0.00361644, 0.00512273, 0.0072564 ,
        0.01027876])
    vf_oil = np.array([0.00876514, 0.01619931, 0.02961302, 0.05303846,
        0.09143938, 0.14684062, 0.20676085, 0.22874507, 0.16382356,
        0.05477458])
    assert_array_almost_equal(spill.d_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.d_liq, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_liq, vf_oil, decimal=6)

    # Check the mass fluxes of each of the particles in the disp_phases
    # particle list
    m0 = np.array([5.289671808056377e-07, 7.995318667820805e-07,
        1.2084893528298558e-06, 1.8266270258633492e-06,
        2.760939749945933e-06, 4.173149852104387e-06, 6.307700009918694e-06,
        9.534064393845064e-06, 1.4410701796700701e-05,
        2.1781720543811073e-05, 4.085231065881823e-08,
        1.1611175556114503e-07, 3.300165783053678e-07, 9.379837677075682e-07,
        2.665967731077995e-06, 7.5772995096915136e-06,
        2.1536445167832175e-05, 6.121157938574416e-05,
        0.00017397752608186881, 0.0004944845384698018])

    for i in range(len(m0)):
        assert_approx_equal(np.sum(spill.disp_phases[i].m), m0[i],
        significant=6)

def test_update_release_depth():
    """
    Check that the Blowout.update_release_depth() method works as
    anticipated and with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    z0 = 200.
    spill.update_release_depth(z0)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_orifice_diameter():
    """
    Check that the Blowout.update_orifice_diameter() method works as
    anticipated and with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    d0 = 0.1
    spill.update_orifice_diameter(d0)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_substance():
    """
    Check that the Blowout.update_substance() method works as
    anticipated and with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    substance = {'composition' : ['methane'],
                 'masses' : np.array([1.0])}
    spill.update_substance(substance)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_q_oil():
    """
    Check that the Blowout.update_q_oil() method works as anticipated and
    with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    q_oil = 30000.
    spill.update_q_oil(q_oil)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_gor():
    """
    Check that the Blowout.update_gor() method works as anticipated and with
    no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    gor = 500.
    spill.update_gor(gor)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_produced_water():
    """
    Check that the Blowout.update_produced_water() method works as
    anticipated and with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    u0 = 1.3
    spill.update_produced_water(u0)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_vertical_orientation():
    """
    Check that the Blowout.update_vertical_orientation() method works as
    anticipated and with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    phi_0 = -np.pi / 4.
    spill.update_vertical_orientation(phi_0)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_horizontal_orientation():
    """
    Check that the Blowout.update_horizontal_orientation() method works as
    anticipated and with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    theta_0 = np.pi
    spill.update_horizontal_orientation(theta_0)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_num_gas_elements():
    """
    Check that the Blowout.update_num_gas_elements() method works as
    anticipated and with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    num_gas_elements = 5
    spill.update_num_gas_elements(num_gas_elements)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_num_oil_elements():
    """
    Check that the Blowout.update_num_oil_elements() method works as
    anticipated and with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    num_oil_elements = 20
    spill.update_num_oil_elements(num_oil_elements)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_water_data():
    """
    Check that the Blowout.update_water_data() method works as
    anticipated and with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    water = get_ctd()
    spill.update_water_data(water)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_update_current_data():
    """
    Check that the Blowout.update_current_data() method works as
    anticipated and with no side effects

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Update the release depth
    current = 0.1
    spill.update_current_data(current)

    # Check that things were done correctly
    assert spill.update == False
    assert spill.bpm.sim_stored == False
    check_attributes(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                     theta_0, num_gas_elements, num_oil_elements,
                     water, current, spill)

def test_simulate():
    """
    Check that the Blowout.simulate() method works and produces the expected
    output.

    """
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0,
                            theta_0, num_gas_elements, num_oil_elements,
                            water, current)

    # Run the simulation
    # Get the input parameters for a typical blowout
    z0, d0, substance, q_oil, gor, x0, y0, u0, phi_0, theta_0, \
        num_gas_elements, num_oil_elements, water, current = \
        get_blowout()

    # Create the blowout.Blowout object
    spill.simulate()

    # Check the simulation
    check_simulation(spill)

    # Re-run the simulation and make sure the results are unchanged
    spill.simulate()
    check_simulation(spill)




