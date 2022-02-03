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
        1.29037789e+00,  4.52019374e+01,  1.47755395e+06,  3.69108728e-15,
        0.00000000e+00, -6.02800299e+01,  8.56255639e-04,  0.00000000e+00,
        0.00000000e+00,  1.00000000e+02,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.09302627e-04,  1.11480315e-05,  3.64591565e-06,  2.07244080e-07,
        1.42962247e-06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  7.19858670e+01,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.06225093e-04,
        3.12326162e-05,  1.02144924e-05,  5.80620425e-07,  4.00526763e-06,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.01677484e+02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  6.61440764e-04,  6.74618963e-05,
        2.20631223e-05,  1.25412981e-06,  8.65130694e-06,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.35619784e+02,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.10149148e-03,  1.12343702e-04,  3.67415233e-05,
        2.08849132e-06,  1.44069453e-05,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  7.25433791e+02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.41420163e-03,  1.44237744e-04,  4.71723323e-05,  2.68140779e-06,
        1.84970341e-05,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  9.31382283e+02,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.39985009e-03,
        1.42773997e-04,  4.66936201e-05,  2.65419644e-06,  1.83093233e-05,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        9.21930470e+02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.06829632e-03,  1.08958049e-04,
        3.56342602e-05,  2.02555138e-06,  1.39727695e-05,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  7.03571713e+02,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  6.28553049e-04,  6.41076011e-05,  2.09661145e-05,
        1.19177281e-06,  8.22115246e-06,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  4.13960189e+02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.85122933e-04,  2.90803573e-05,  9.51060546e-06,  5.40609513e-07,
        3.72926216e-06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.87779764e+02,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.97154302e-05,
        1.01702108e-05,  3.32612359e-06,  1.89066202e-07,  1.30422683e-06,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        6.56718131e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  4.98777605e-06,  3.72494650e-06,
        4.63112014e-06,  8.36251107e-05,  1.66674345e-04,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.50943069e+02,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  9.21816467e-06,  6.88426463e-06,  8.55901059e-06,
        1.54551855e-04,  3.08039404e-04,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  2.78965625e+02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.68511935e-05,  1.25847258e-05,  1.56462320e-05,  2.82527304e-04,
        5.63109011e-04,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  5.09960919e+02,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.01813628e-05,
        2.25398977e-05,  2.80232143e-05,  5.06021076e-04,  1.00855749e-03,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        9.13366494e+02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  5.20332848e-05,  3.88592431e-05,
        4.83125927e-05,  8.72390652e-04,  1.73877367e-03,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.57466246e+03,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  8.35591811e-05,  6.24032587e-05,  7.75841983e-05,
        1.40095419e-03,  2.79226085e-03,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  2.52871803e+03,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.17656590e-04,  8.78677191e-05,  1.09243438e-04,  1.97263175e-03,
        3.93167916e-03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  3.56059427e+03,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.30166645e-04,
        9.72104168e-05,  1.20858949e-04,  2.18237547e-03,  4.34972222e-03,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        3.93918104e+03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  9.32232687e-05,  6.96205454e-05,
        8.65572457e-05,  1.56298240e-03,  3.11520147e-03,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.82117844e+03,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  3.11692988e-05,  2.32777032e-05,  2.89405069e-05,
        5.22584823e-04,  1.04157092e-03,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  9.43264006e+02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  7.92154297e-06,  0.00000000e+00,
        0.00000000e+00,  1.29037789e+00])
    qn = np.array([ 
        5.11749000e+02,  1.78922408e+04,  5.90981238e+08,  4.59412760e+01,
        0.00000000e+00, -1.19432857e+03,  8.56255639e-04,  4.29638520e+00,
        0.00000000e+00, -4.82707871e+00,  1.04921379e+02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.03198605e-04,  1.04087319e-05,  3.48110030e-06,  4.17493931e-09,
        1.46801528e-08,  0.00000000e+00,  4.70613021e-07,  0.00000000e+00,
        0.00000000e+00,  6.78910327e+01,  4.34390947e+01,  0.00000000e+00,
        2.90606145e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.92387077e-04,
        2.95547988e-05,  9.84126498e-06,  2.29089428e-08,  8.18127781e-08,
        0.00000000e+00,  1.07274177e-06,  0.00000000e+00,  0.00000000e+00,
        1.92255842e+02,  4.35007755e+01,  0.00000000e+00,  2.86818297e-01,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  6.28565796e-04,  6.34139759e-05,
        2.11465810e-05,  3.39603514e-08,  1.43580030e-07,  0.00000000e+00,
        4.43153785e-06,  0.00000000e+00,  0.00000000e+00,  4.14429900e+02,
        4.34851670e+01,  0.00000000e+00,  2.90086793e-01,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  1.04423696e-03,  1.05231826e-04,  3.51165697e-05,
        7.91153584e-09, -1.13665791e-08,  0.00000000e+00,  8.17596554e-06,
        0.00000000e+00,  0.00000000e+00,  6.88720681e+02,  4.34478035e+01,
        0.00000000e+00,  2.94985225e-01,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.33979534e-03,  1.34944653e-04,  4.50384565e-05, -1.89573079e-08,
        -2.95449127e-07,  0.00000000e+00,  1.04020702e-05,  0.00000000e+00,
        0.00000000e+00,  8.83546406e+02,  4.33674516e+01,  0.00000000e+00,
        3.03240012e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.30136650e-03,
        1.30425453e-04,  4.38321508e-05, -5.17158212e-08, -6.57617089e-07,
        0.00000000e+00,  1.14506002e-05,  0.00000000e+00,  0.00000000e+00,
        8.58656541e+02,  4.31327917e+01,  0.00000000e+00,  3.21642401e-01,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  9.83802579e-04,  9.83704985e-05,
        3.31739906e-05,  1.02559125e-08, -1.01660765e-07,  0.00000000e+00,
        8.42263034e-06,  0.00000000e+00,  0.00000000e+00,  6.48885360e+02,
        4.28253573e+01,  0.00000000e+00,  3.42791001e-01,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  5.64111231e-04,  5.60796643e-05,  1.90989240e-05,
        -7.11516786e-09,  4.55158681e-08,  0.00000000e+00,  4.90612277e-06,
        0.00000000e+00,  0.00000000e+00,  3.71993511e+02,  4.23474469e+01,
        0.00000000e+00,  3.69695461e-01,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.60394404e-04,  2.59940642e-05,  8.79519918e-06, -4.33037159e-10,
        -1.76184914e-08,  0.00000000e+00,  1.88446173e-06,  0.00000000e+00,
        0.00000000e+00,  1.71531054e+02,  4.19627597e+01,  0.00000000e+00,
        3.94282317e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.24705025e-05,
        9.26442578e-06,  3.11676329e-06,  1.22809043e-09, -1.85975913e-09,
        0.00000000e+00,  5.55134158e-07,  0.00000000e+00,  0.00000000e+00,
        6.08639999e+01,  4.15598300e+01,  0.00000000e+00,  4.20242980e-01,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  1.59369947e-06,  3.10382114e-06,
        4.48739009e-06,  8.29247853e-05,  1.60602766e-04,  0.00000000e+00,
        3.29893438e-08,  0.00000000e+00,  0.00000000e+00,  1.45938556e+02,
        4.81698984e+01,  0.00000000e+00,  1.49919949e-02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  4.25317490e-06,  6.08203837e-06,  8.37757757e-06,
        1.53666066e-04,  3.00320505e-04,  0.00000000e+00,  5.35870283e-08,
        0.00000000e+00,  0.00000000e+00,  2.72973785e+02,  4.80039303e+01,
        0.00000000e+00,  2.43514918e-02,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.00007638e-05,  1.15742187e-05,  1.54212985e-05,  2.81426939e-04,
        5.53484959e-04,  0.00000000e+00,  8.04690474e-08,  0.00000000e+00,
        0.00000000e+00,  5.03497741e+02,  4.77551102e+01,  0.00000000e+00,
        3.84548812e-02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.14128053e-05,
        2.13308757e-05,  2.77571471e-05,  5.04717203e-04,  9.97122402e-04,
        0.00000000e+00,  1.10109361e-07,  0.00000000e+00,  0.00000000e+00,
        9.07953673e+02,  4.74403425e+01,  0.00000000e+00,  5.64015871e-02,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  4.13914304e-05,  3.74557471e-05,
        4.80059793e-05,  8.70886012e-04,  1.72555455e-03,  0.00000000e+00,
        1.39878260e-07,  0.00000000e+00,  0.00000000e+00,  1.57254646e+03,
        4.69411142e+01,  0.00000000e+00,  8.50149195e-02,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  7.17622446e-05,  6.08938631e-05,  7.72560624e-05,
        1.39934227e-03,  2.77808220e-03,  0.00000000e+00,  1.60088743e-07,
        0.00000000e+00,  0.00000000e+00,  2.53339845e+03,  4.64451679e+01,
        0.00000000e+00,  1.13684065e-01,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        1.06564733e-04,  8.64784388e-05,  1.08942446e-04,  1.97115204e-03,
        3.91865257e-03,  0.00000000e+00,  1.53979478e-07,  0.00000000e+00,
        0.00000000e+00,  3.57531021e+03,  4.59260790e+01,  0.00000000e+00,
        1.43952693e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.22144970e-04,
        9.62199364e-05,  1.20644845e-04,  2.18132235e-03,  4.34044591e-03,
        0.00000000e+00,  1.13084846e-07,  0.00000000e+00,  0.00000000e+00,
        3.96156897e+03,  4.53623843e+01,  0.00000000e+00,  1.77123523e-01,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  8.97692448e-05,  6.91985200e-05,
        8.64661738e-05,  1.56253428e-03,  3.11125257e-03,  0.00000000e+00,
        4.92476725e-08,  0.00000000e+00,  0.00000000e+00,  2.84045137e+03,
        4.52910199e+01,  0.00000000e+00,  1.81105659e-01,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  3.04938874e-05,  2.31957105e-05,  2.89228313e-05,
        5.22497829e-04,  1.04080411e-03,  0.00000000e+00,  9.69825254e-09,
        0.00000000e+00,  0.00000000e+00,  9.50378255e+02,  4.53833522e+01,
        0.00000000e+00,  1.75660892e-01,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        5.33530219e-04,  6.72656653e-05,  1.54018665e-05,  2.35857251e-05,
        1.82592878e-04,  0.00000000e+00,  3.51972607e-03,  0.00000000e+00,
        0.00000000e+00,  1.29037789e+00])
    assert spill.bpm.t[0] == 0.
    for i in range(len(q0)):
        assert_approx_equal(spill.bpm.q[0,i], q0[i], significant=6)
    assert_approx_equal(spill.bpm.t[-1], 48.439163984340084, significant=6)
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




