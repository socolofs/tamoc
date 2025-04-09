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
    q0 = np.array([1.290377888e+00, 4.520193743e+01, 1.477553950e+06,
        4.221220995e-15, 0.000000000e+00, -6.893777044e+01, 7.487204070e-04,
        0.000000000e+00, 0.000000000e+00, 1.000000000e+02, 0.000000000e+00,
        6.652118720e-09, 2.516529226e-10, 7.344836926e-11, 7.471205406e-13,
        5.358697359e-13, 5.614682270e-05, 7.123086067e-06, 3.163404326e-06,
        2.702767083e-07, 1.854695512e-06, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 3.925550696e+01, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 2.138811531e-08,
        8.091229206e-10, 2.361536613e-10, 2.402167031e-12, 1.722946355e-12,
        1.805251483e-04, 2.290238533e-05, 1.017108373e-05, 8.690027412e-07,
        5.963279243e-06, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 1.262156231e+02, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 5.301820836e-08, 2.005704896e-09,
        5.853925807e-10, 5.954643048e-12, 4.270948025e-12, 4.474971163e-04,
        5.677187632e-05, 2.521272346e-05, 2.154138769e-06, 1.478215246e-05,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        3.128712422e+02, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 1.013252266e-07, 3.833183151e-09, 1.118767263e-09,
        1.138015739e-11, 8.162380240e-12, 8.552297049e-04, 1.084990121e-04,
        4.818504803e-05, 4.116861086e-06, 2.825076504e-05, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 5.979407921e+02,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        1.492966712e-07, 5.647966495e-09, 1.648436761e-09, 1.676798241e-11,
        1.202677991e-11, 1.260129905e-03, 1.598668159e-04, 7.099779116e-05,
        6.065948994e-06, 4.162581544e-05, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 8.810300543e+02, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 1.695987594e-07,
        6.416004474e-09, 1.872599217e-09, 1.904817429e-11, 1.366223999e-11,
        1.431488504e-03, 1.816062838e-04, 8.065241644e-05, 6.890826268e-06,
        4.728629647e-05, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 1.000836810e+03, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 1.485371965e-07, 5.619235193e-09,
        1.640051135e-09, 1.668268340e-11, 1.196559947e-11, 1.253719602e-03,
        1.590535707e-04, 7.063662418e-05, 6.035091408e-06, 4.141406421e-05,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        8.765482393e+02, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 1.002969623e-07, 3.794283409e-09, 1.107413838e-09,
        1.126466978e-11, 8.079547128e-12, 8.465507001e-04, 1.073979472e-04,
        4.769605863e-05, 4.075082536e-06, 2.796407186e-05, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 5.918727954e+02,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        5.221320376e-08, 1.975251176e-09, 5.765042434e-10, 5.864230429e-12,
        4.206099873e-12, 4.407025216e-04, 5.590987772e-05, 2.482990482e-05,
        2.121431296e-06, 1.455770692e-05, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 3.081207462e+02, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 2.095621809e-08,
        7.927840362e-10, 2.313849330e-10, 2.353659284e-12, 1.688154335e-12,
        1.768797448e-04, 2.243990995e-05, 9.965695706e-06, 8.514546840e-07,
        5.842860791e-06, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 1.236669097e+02, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 2.132428812e-10, 5.931488616e-11,
        6.590559464e-11, 1.563147574e-12, 1.792311549e-12, 1.799786887e-06,
        1.679049732e-06, 2.839114576e-06, 7.809133165e-05, 1.547570875e-04,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        1.369295473e+02, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 3.941051029e-10, 1.096228825e-10, 1.218035088e-10,
        2.888933183e-12, 3.312462875e-12, 3.326278431e-06, 3.103137904e-06,
        5.247113224e-06, 1.443245942e-04, 2.860145087e-04, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 2.530665176e+02,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        7.204407360e-10, 2.003952488e-10, 2.226619470e-10, 5.281091601e-12,
        6.055321725e-12, 6.080577143e-06, 5.672666857e-06, 9.591944092e-06,
        2.638314402e-04, 5.228465749e-04, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 4.626162585e+02, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 1.290346779e-09,
        3.589182994e-10, 3.987991124e-10, 9.458709362e-12, 1.084539573e-11,
        1.089062950e-05, 1.016004099e-05, 1.717967008e-05, 4.725358130e-04,
        9.364453728e-04, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 8.285697478e+02, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 2.224584154e-09, 6.187824659e-10,
        6.875378005e-10, 1.630700779e-11, 1.869768336e-11, 1.877566729e-05,
        1.751611780e-05, 2.961810149e-05, 8.146613758e-04, 1.614450915e-03,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        1.428471138e+03, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 3.572413907e-09, 9.936900263e-10, 1.104102803e-09,
        2.618708818e-11, 3.002622487e-11, 3.015145766e-05, 2.812877307e-05,
        4.756309959e-05, 1.308247936e-03, 2.592613496e-03, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 2.293952400e+03,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        5.030183810e-09, 1.399178150e-09, 1.554646295e-09, 3.687306970e-11,
        4.227881599e-11, 4.245515165e-05, 3.960708433e-05, 6.697184025e-05,
        1.842095502e-03, 3.650563113e-03, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 3.230029477e+03, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 5.565027407e-09,
        1.547948355e-09, 1.719946937e-09, 4.079366703e-11, 4.677418929e-11,
        4.696927417e-05, 4.381838082e-05, 7.409274503e-05, 2.037959713e-03,
        4.038715988e-03, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 3.573468335e+03, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 3.985583599e-09, 1.108615848e-09,
        1.231798481e-09, 2.921577171e-11, 3.349892607e-11, 3.363864274e-05,
        3.138202334e-05, 5.306403865e-05, 1.459554143e-03, 2.892463779e-03,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        2.559260853e+03, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 1.332584104e-09, 3.706668846e-10, 4.118531287e-10,
        9.768324262e-12, 1.120040147e-11, 1.124711589e-05, 1.049261279e-05,
        1.774201761e-05, 4.880034759e-04, 9.670983328e-04, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 8.556915812e+02,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 1.256637061e-03,])
        
    qn = np.array([4.535820639e+02, 1.586090853e+04, 5.236124088e+08,
        4.070625174e+01, 0.000000000e+00, -1.074298264e+03, 7.487204070e-04,
        3.984056044e+00, 0.000000000e+00, -1.256360237e+00, 1.013402116e+02,
        6.049729093e-09, 2.247139189e-10, 6.805503334e-11, 7.189761228e-13,
        5.073065468e-13, 5.106268001e-05, 6.360580966e-06, 2.931116150e-06,
        1.477862960e-09, 8.810861695e-09, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 3.484596533e+01, 4.028365354e+01,
        0.000000000e+00, 2.735820684e-01, 0.000000000e+00, 1.981828272e-08,
        7.387980890e-10, 2.221249451e-10, 2.329049235e-12, 1.648685982e-12,
        1.672758364e-04, 2.091185016e-05, 9.566874246e-06, 7.237267628e-09,
        3.261960541e-08, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 1.141783395e+02, 4.033706752e+01, 0.000000000e+00,
        2.702301534e-01, 0.000000000e+00, 4.986743898e-08, 1.864361829e-09,
        5.572787587e-10, 5.808095326e-12, 4.122078412e-12, 4.209047865e-04,
        5.277117977e-05, 2.400187585e-05, 3.478555267e-08, 1.240414473e-07,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        2.873800164e+02, 4.039133673e+01, 0.000000000e+00, 2.668375575e-01,
        0.000000000e+00, 9.645341840e-08, 3.614391232e-09, 1.075350487e-09,
        1.115350883e-11, 7.932207701e-12, 8.141119129e-04, 1.023061368e-04,
        4.631511023e-05, 1.343234673e-07, 4.610696832e-07, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 5.560887369e+02,
        4.044622952e+01, 0.000000000e+00, 2.634179578e-01, 0.000000000e+00,
        1.429236461e-07, 5.360015188e-09, 1.591012036e-09, 1.646623634e-11,
        1.172051327e-11, 1.206341682e-03, 1.517163859e-04, 6.852453693e-05,
        3.020104547e-07, 1.141355647e-06, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 8.243388229e+02, 4.048176243e+01,
        0.000000000e+00, 2.618151815e-01, 0.000000000e+00, 1.624565780e-07,
        6.090164427e-09, 1.806978813e-09, 1.870048557e-11, 1.330941383e-11,
        1.371208164e-03, 1.723834028e-04, 7.782617579e-05, 3.038006983e-07,
        1.199985674e-06, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 9.368437765e+02, 4.048735074e+01, 0.000000000e+00,
        2.628933081e-01, 0.000000000e+00, 1.421663817e-07, 5.326184485e-09,
        1.580546098e-09, 1.636544503e-11, 1.164366789e-11, 1.199949328e-03,
        1.507587802e-04, 6.807376786e-05, 1.381629890e-07, 3.878080431e-07,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        8.193062852e+02, 4.045696313e+01, 0.000000000e+00, 2.670024298e-01,
        0.000000000e+00, 9.514163558e-08, 3.555210808e-09, 1.058385074e-09,
        1.100206107e-11, 7.812956570e-12, 8.030390754e-04, 1.006309996e-04,
        4.558440992e-05, 1.327527166e-08, -3.381417170e-07, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 5.479720882e+02,
        4.033538400e+01, 0.000000000e+00, 2.781746587e-01, 0.000000000e+00,
        4.858565366e-08, 1.806676711e-09, 5.416543077e-10, 5.677023295e-12,
        4.016187504e-12, 4.100856195e-04, 5.113838204e-05, 2.332893304e-05,
        1.833357333e-08, 5.946249273e-08, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 2.797568061e+02, 4.008487655e+01,
        0.000000000e+00, 2.968578533e-01, 0.000000000e+00, 1.915256208e-08,
        7.091895767e-10, 2.140641301e-10, 2.260611360e-12, 1.593853544e-12,
        1.616568274e-04, 2.007377496e-05, 9.219696914e-06, -8.163725557e-09,
        -3.067369747e-07, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 1.102275289e+02, 3.972563359e+01, 0.000000000e+00,
        3.196847588e-01, 0.000000000e+00, 3.287828813e-11, 4.407265376e-11,
        6.260703678e-11, 1.545734086e-12, 1.774438424e-12, 2.775109805e-07,
        1.247614612e-06, 2.697047436e-06, 7.706247931e-05, 1.460145323e-04,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        1.311964889e+02, 4.481481563e+01, 0.000000000e+00, 5.366453971e-03,
        0.000000000e+00, 1.073571521e-10, 8.917349911e-11, 1.175273515e-10,
        2.866099084e-12, 3.289001618e-12, 9.061470270e-07, 2.524316892e-06,
        5.062941669e-06, 1.429931552e-04, 2.746288734e-04, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 2.459527331e+02,
        4.474421251e+01, 0.000000000e+00, 9.408731030e-03, 0.000000000e+00,
        2.906170866e-10, 1.734882319e-10, 2.171686653e-10, 5.251180957e-12,
        6.024547640e-12, 2.452929242e-06, 4.911065783e-06, 9.355351928e-06,
        2.621223653e-04, 5.081658318e-04, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 4.542587895e+02, 4.462543596e+01,
        0.000000000e+00, 1.621974302e-02, 0.000000000e+00, 7.004172553e-10,
        3.257129612e-10, 3.921440803e-10, 9.421248934e-12, 1.080677866e-11,
        5.911757799e-06, 9.220169944e-06, 1.689304225e-05, 4.704662875e-04,
        9.186046598e-04, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 8.202531238e+02, 4.447074952e+01, 0.000000000e+00,
        2.512984764e-02, 0.000000000e+00, 1.488165472e-09, 5.804804547e-10,
        6.799620236e-10, 1.626206005e-11, 1.865122351e-11, 1.256051198e-05,
        1.643198933e-05, 2.929181792e-05, 8.123063910e-04, 1.594096864e-03,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        1.422611869e+03, 4.427729656e+01, 0.000000000e+00, 3.547321819e-02,
        0.000000000e+00, 2.709611424e-09, 9.509507575e-10, 1.095714733e-09,
        2.613393300e-11, 2.997111878e-11, 2.286970322e-05, 2.691905117e-05,
        4.720183120e-05, 1.305639617e-03, 2.570035397e-03, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 2.293013426e+03,
        4.389014285e+01, 0.000000000e+00, 5.875966526e-02, 0.000000000e+00,
        4.186844788e-09, 1.358920682e-09, 1.546791433e-09, 3.681815365e-11,
        4.222165950e-11, 3.533769645e-05, 3.846761047e-05, 6.663353683e-05,
        1.839652750e-03, 3.629393632e-03, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 3.237782157e+03, 4.346968695e+01,
        0.000000000e+00, 8.329363378e-02, 0.000000000e+00, 4.929670576e-09,
        1.518372663e-09, 1.714198853e-09, 4.074776612e-11, 4.672619310e-11,
        4.160712692e-05, 4.298125170e-05, 7.384517906e-05, 2.036171960e-03,
        4.023210840e-03, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 3.588909681e+03, 4.304201138e+01, 0.000000000e+00,
        1.084394800e-01, 0.000000000e+00, 3.684355578e-09, 1.094835591e-09,
        1.229127458e-09, 2.919023595e-11, 3.347208255e-11, 3.109640999e-05,
        3.099197845e-05, 5.294899955e-05, 1.458723340e-03, 2.885254392e-03,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00,
        2.573733633e+03, 4.258624528e+01, 0.000000000e+00, 1.354484454e-01,
        0.000000000e+00, 1.267938662e-09, 3.677444049e-10, 4.112876975e-10,
        9.761458254e-12, 1.119314284e-11, 1.070153758e-05, 1.040989315e-05,
        1.771766489e-05, 4.878275936e-04, 9.655715438e-04, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 8.613108460e+02,
        4.217100640e+01, 0.000000000e+00, 1.600238221e-01, 0.000000000e+00,
        4.559801006e-08, 2.118863554e-09, 4.272285946e-10, 2.210002192e-12,
        2.258218920e-12, 3.848471079e-04, 5.997423785e-05, 1.840052541e-05,
        4.884425562e-05, 3.677640364e-04, 0.000000000e+00, 0.000000000e+00,
        0.000000000e+00, 0.000000000e+00, 1.256637061e-03])

    assert spill.bpm.t[0] == 0.
    
    for i in range(len(q0)):
        assert_approx_equal(spill.bpm.q[0,i], q0[i], significant=6)
    assert_approx_equal(spill.bpm.t[-1], 44.90897795309906, significant=6)
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
    de_gas = np.array([0.0035, 0.004017, 0.00461, 0.005291, 0.006072, 0.006968,
        0.007997, 0.009178, 0.010533, 0.012088])
    vf_gas = np.array([0.00807999, 0.02597907, 0.06439855, 0.12307465, 
        0.18134315, 0.20600307, 0.18042065, 0.12182567, 0.06342075, 0.02545446])
    de_oil = np.array([0.000266, 0.000377, 0.000534, 0.000756, 0.001071, 
        0.001518, 0.00215, 0.003045, 0.004313, 0.00611])
    vf_oil = np.array([0.00876514, 0.01619931, 0.02961302, 0.05303846,
        0.09143938, 0.14684062, 0.20676085, 0.22874507, 0.16382356,
        0.05477458])
    assert_array_almost_equal(spill.d_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.d_liq, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_liq, vf_oil, decimal=6)

    # Check the mass fluxes of each of the particles in the disp_phases
    # particle list
    m0 = np.array([1.9307513609810385e-07, 2.9183232834711973e-07,
        4.4110344599331093e-07, 6.667261683076455e-07, 1.0077540484980383e-06,
        1.5232163825847852e-06, 2.302335725301921e-06, 3.479971626228537e-06,
        5.259963777770762e-06, 7.950415093885464e-06, 8.474626782785078e-09,
        2.4086857698030127e-08, 6.846044417480355e-08, 1.945804834889151e-07,
        5.530429288204161e-07, 1.5718764576699594e-06, 4.467637988694678e-06,
        1.2698064851492751e-05, 3.609084965718672e-05, 0.00010257857746131524])

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




