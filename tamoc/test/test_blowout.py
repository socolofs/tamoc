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
        1.29037789e+00, 4.52019374e+01, 1.47755395e+06, 3.69108722e-15,
        0.00000000e+00, -6.02800289e+01, 8.56255653e-04, 0.00000000e+00,
        0.00000000e+00, 1.00000000e+02, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        5.71594861e-05, 5.82982963e-06, 1.90662047e-06, 1.08377661e-07,
        7.47616707e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 3.76447960e+01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.83781098e-04,
        1.87442638e-05, 6.13023010e-06, 3.48459491e-07, 2.40376232e-06,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.21036811e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 4.55568169e-04, 4.64644627e-05,
        1.51960008e-05, 8.63783351e-07, 5.95859756e-06, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.00033677e+02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 8.70654618e-04, 8.88001001e-05, 2.90416872e-05,
        1.65081104e-06, 1.13877150e-05, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 5.73406405e+02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.28285759e-03, 1.30841645e-04, 4.27911920e-05, 2.43237150e-06,
        1.67791181e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 8.44880100e+02, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.45730681e-03,
        1.48634129e-04, 4.86101466e-05, 2.76313722e-06, 1.90608243e-05,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        9.59771009e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 1.27633167e-03, 1.30176052e-04,
        4.25735125e-05, 2.41999798e-06, 1.66937625e-05, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.40582181e+02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 8.61819078e-04, 8.78989428e-05, 2.87469676e-05,
        1.63405835e-06, 1.12721507e-05, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 5.67587387e+02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.48651027e-04, 4.57589672e-05, 1.49652715e-05, 8.50668054e-07,
        5.86812490e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 2.95478101e+02, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.80069946e-04,
        1.83657548e-05, 6.00644036e-06, 3.41422935e-07, 2.35522236e-06,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.18592675e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 4.98777791e-06, 3.72494741e-06,
        4.63112076e-06, 8.36251140e-05, 1.66674352e-04, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.50943077e+02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 9.21816811e-06, 6.88426631e-06, 8.55901174e-06,
        1.54551861e-04, 3.08039416e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 2.78965640e+02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.68511998e-05, 1.25847289e-05, 1.56462341e-05, 2.82527315e-04,
        5.63109034e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.09960945e+02, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.01813740e-05,
        2.25399032e-05, 2.80232180e-05, 5.06021096e-04, 1.00855753e-03,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        9.13366541e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 5.20333042e-05, 3.88592526e-05,
        4.83125992e-05, 8.72390687e-04, 1.73877374e-03, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.57466254e+03,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 8.35592123e-05, 6.24032740e-05, 7.75842087e-05,
        1.40095425e-03, 2.79226096e-03, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 2.52871816e+03, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.17656634e-04, 8.78677406e-05, 1.09243453e-04, 1.97263183e-03,
        3.93167932e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 3.56059446e+03, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.30166694e-04,
        9.72104406e-05, 1.20858965e-04, 2.18237556e-03, 4.34972240e-03,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        3.93918125e+03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 9.32233035e-05, 6.96205624e-05,
        8.65572573e-05, 1.56298246e-03, 3.11520159e-03, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.82117859e+03,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 3.11693104e-05, 2.32777089e-05, 2.89405108e-05,
        5.22584843e-04, 1.04157096e-03, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 9.43264054e+02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.25663706e-03])        
    qn = np.array([
        4.98889119e+02, 1.74439416e+04, 5.76032293e+08, 4.47838867e+01,
        0.00000000e+00, -1.13287054e+03, 8.56255653e-04, 4.26784997e+00,
        0.00000000e+00, -2.91360221e+00, 1.03007821e+02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        5.42822293e-05, 5.48119956e-06, 1.82899795e-06, 3.14478442e-09,
        1.13944669e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 3.55666225e+01, 4.31278981e+01, 0.00000000e+00,
        2.89543024e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.75211135e-04,
        1.76973469e-05, 5.89528057e-06, 1.56711009e-08, 6.26446450e-08,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.14817597e+02, 4.31559787e+01, 0.00000000e+00, 2.88828709e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 4.33691209e-04, 4.37634545e-05,
        1.45828921e-05, 3.07165899e-08, 1.48827912e-07, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.84164290e+02,
        4.31461204e+01, 0.00000000e+00, 2.91396155e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 8.26389716e-04, 8.32912103e-05, 2.77805089e-05,
        1.55931626e-08, 1.08770876e-07, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 5.41282299e+02, 4.30936290e+01,
        0.00000000e+00, 2.97381010e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.21497632e-03, 1.22352035e-04, 4.08376599e-05, -3.03029291e-08,
        -1.66142100e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 7.95635845e+02, 4.29863932e+01, 0.00000000e+00,
        3.07444000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.34581586e-03,
        1.34660203e-04, 4.53671704e-05, -9.74538952e-08, -1.40850458e-06,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        8.80892110e+02, 4.26760451e+01, 0.00000000e+00, 3.30419465e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 1.15596900e-03, 1.15143438e-04,
        3.90761728e-05, -1.80896789e-07, -3.30725011e-06, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.56391473e+02,
        4.22682700e+01, 0.00000000e+00, 3.55471601e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 7.80141262e-04, 7.77168380e-05, 2.63832148e-05,
        9.24651335e-09, 6.39320943e-08, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 5.10527977e+02, 4.18510481e+01,
        0.00000000e+00, 3.80014235e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.13010156e-04, 4.13074471e-05, 1.39351030e-05, 5.59816213e-09,
        3.38246000e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 2.70352047e+02, 4.14594369e+01, 0.00000000e+00,
        4.05193862e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.68095433e-04,
        1.68677742e-05, 5.66068967e-06, 3.13854789e-09, 1.55685182e-08,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.10060825e+02, 4.10492224e+01, 0.00000000e+00, 4.31725818e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 1.60827880e-06, 3.10819342e-06,
        4.48848489e-06, 8.29300832e-05, 1.60648097e-04, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.45935484e+02,
        4.78267169e+01, 0.00000000e+00, 1.50228243e-02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 4.27907833e-06, 6.08785608e-06, 8.37896749e-06,
        1.53672781e-04, 3.00378505e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 2.72952885e+02, 4.76609228e+01,
        0.00000000e+00, 2.44023188e-02, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.00417082e-05, 1.15817362e-05, 1.54230378e-05, 2.81435345e-04,
        5.53558027e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.03441645e+02, 4.74124173e+01, 0.00000000e+00,
        3.85312790e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.14705499e-05,
        2.13400568e-05, 2.77592236e-05, 5.04727246e-04, 9.97210093e-04,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        9.07831897e+02, 4.70980772e+01, 0.00000000e+00, 5.65081178e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 4.14662736e-05, 3.74665979e-05,
        4.80083990e-05, 8.70897729e-04, 1.72565710e-03, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.57231501e+03,
        4.65994058e+01, 0.00000000e+00, 8.51767986e-02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 7.18489707e-05, 6.09056856e-05, 7.72586750e-05,
        1.39935493e-03, 2.77819314e-03, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 2.53300479e+03, 4.61040573e+01,
        0.00000000e+00, 1.13896227e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.06649056e-04, 8.64894567e-05, 1.08944867e-04, 1.97116379e-03,
        3.91875547e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 3.57473569e+03, 4.55856393e+01, 0.00000000e+00,
        1.44212688e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.22207565e-04,
        9.62278913e-05, 1.20646589e-04, 2.18133082e-03, 4.34051996e-03,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        3.96091909e+03, 4.50227155e+01, 0.00000000e+00, 1.77430528e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 8.97964087e-05, 6.92019066e-05,
        8.64669172e-05, 1.56253789e-03, 3.11128401e-03, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.83997801e+03,
        4.49510952e+01, 0.00000000e+00, 1.81443693e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 3.04992346e-05, 2.31963706e-05, 2.89229770e-05,
        5.22498538e-04, 1.04081023e-03, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 9.50218463e+02, 4.50433120e+01,
        0.00000000e+00, 1.75990774e-01, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        5.75797031e-04, 7.26002746e-05, 1.66788195e-05, 2.37344944e-05,
        1.85538499e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.25663706e-03])

    assert spill.bpm.t[0] == 0.
    for i in range(len(q0)):
        assert_approx_equal(spill.bpm.q[0,i], q0[i], significant=6)
    assert_approx_equal(spill.bpm.t[-1], 48.09487839432009, significant=6)
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
    assert_approx_equal(T, 286.45, significant=6)
    assert_approx_equal(S, 35.03, significant=6)
    assert_approx_equal(P, 1107655.378995259, significant=6)
    assert_approx_equal(ua, 0.09, significant=6)
    assert_approx_equal(va, 0., significant=6)

    # Check the bubble and droplet sizes
    de_gas = np.array([0.00393499, 0.00451592, 0.00518261, 0.00594773, 
        0.0068258, 0.0078335, 0.00898997, 0.01031717, 0.01184031, 0.0135883])
    vf_gas = np.array([0.00807999, 0.02597907, 0.06439855, 0.12307465, 
        0.18134315, 0.20600307, 0.18042065, 0.12182567, 0.06342075, 0.02545446])
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
    m0 = np.array([6.50327663e-07, 9.82967771e-07, 1.48575202e-06,  
        2.24570848e-06, 3.39437977e-06, 5.13059201e-06, 7.75487017e-06,
        1.17214565e-05, 1.77169366e-05, 2.67790818e-05, 4.08523077e-08,
        1.16111747e-07, 3.30016555e-07, 9.37983700e-07, 2.66596754e-06,
        7.57729896e-06, 2.15364436e-05, 6.12115750e-05, 1.73977514e-04,
        4.94484503e-04])

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




