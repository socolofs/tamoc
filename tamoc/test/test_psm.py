"""
Unit tests for the `particle_size_model` module of ``TAMOC``

Provides testing of the classes, methods and functions defined in the
`particle_size_models` module of ``TAMOC``.  These tests check the behavior
of the class objects, the results of simulations, and the related object
methods.

The ambient data used here to compute oil properties at `in situ` conditions
are from the `ctd_BM54.cnv` dataset, stored as::

    ./test/output/test_BM54.nc

This netCDF file is written by the `test_ambient.test_from_ctd` function,
which is run in the following as needed to ensure the dataset is available.

Since the `particle_size_models` module relies on the functions in the
`psf` module to compute results, this set of tests checks the performance
of both modules.

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
from tamoc import dbm_utilities
from tamoc import particle_size_models as psm

from tamoc.test import test_sbm

from datetime import datetime
from netCDF4 import date2num

import os
import numpy as np
from numpy.testing import assert_approx_equal
from numpy.testing import assert_array_almost_equal

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def get_blowout_model():
    """
    Compute the inputs defining a synthetic blowout

    Create the `ambient.Profile` object, `dbm.FluidMixture` object, and
    other parameters defining a synthetic blowout scenario.

    Returns
    -------
    profile : `ambient.Profile` object
        Profile containing ambient CTD data
    oil : `dbm.FluidMixture` object
        A `dbm.FluidMixture` object that contains the chemical description
        of an oil mixture.
    mass_flux : ndarray
        An array of mass fluxes (kg/s) of each pseudo-component in the live-
        oil composition.
    z0 : float
        Release point of a synthetic blowout (m)
    Tj : float
        Temperature of the released fluids (K)

    """
    # Get the CTD data from the requested file
    nc = test_sbm.make_ctd_file()
    profile = ambient.Profile(nc, chem_names='all')
    profile.close_nc()

    # Define an oil substance to use
    substance={
        'composition' : ['n-hexane', '2-methylpentane', '3-methylpentane',
                         'neohexane', 'n-heptane', 'benzene', 'toluene',
                         'ethylbenzene', 'n-decane'],
        'masses' : np.array([0.04, 0.07, 0.08, 0.09, 0.11, 0.12, 0.15, 0.18,
                             0.16])
    }

    # Define the atmospheric gases to track
    ca = ['oxygen']

    # Define the oil flow rate, gas to oil ratio, and orifice size
    q_oil = 20000.   # bbl/d
    gor = 500.       # ft^3/bbl at standard conditions
    z0 = 100.       # release depth (m)
    Tj = profile.get_values(z0, 'temperature') # release temperature (K)

    # Import the oil with the desired gas to oil ratio
    oil, mass_flux = dbm_utilities.get_oil(substance, q_oil, gor, ca)

    return (profile, oil, mass_flux, z0, Tj)


def get_blowout_properties():
    """
    Return the properties for the base blowout case

    Return the fluid properties and initial conditions for the base case of
    a blowout from the Model Inter-comparison Study for the case of
    20,000 bbl/d, 2000 m depth, GOR of 2000, and 30 cm orifice.

    """
    # Get the properties for the base case
    d0 = 0.30
    m_gas = 7.4
    m_oil = 34.5
    rho_gas = 131.8
    mu_gas = 0.00002
    sigma_gas = 0.06
    rho_oil = 599.3
    mu_oil = 0.0002
    sigma_oil = 0.015
    rho = 1037.1
    mu = 0.002

    return (d0, m_gas, m_oil, rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil,
            sigma_oil, rho, mu)


def get_blowout_ans():
    """
    Report the correct answer for the base blowout case

    """
    de_max_gas = 0.0313470341427564
    de_max_oil = 0.007475379384955715

    return (de_max_gas, de_max_oil)


def check_properties(rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil, sigma_oil,
                     rho, mu, psd_obj):
    """
    Check that the given properties match the properties inside the object

    """
    assert psd_obj.rho_gas == rho_gas
    assert psd_obj.mu_gas == mu_gas
    assert psd_obj.sigma_gas == sigma_gas
    assert psd_obj.rho_oil == rho_oil
    assert psd_obj.mu_oil == mu_oil
    assert psd_obj.sigma_oil == sigma_oil
    assert psd_obj.rho == rho
    assert psd_obj.mu == mu


# ----------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------

def test_psm_ModelBase():
    """
    Test the `ModelBase` and `PureJet` classes

    Test all of the functionality in the `ModelBase` and `PureJet` classes.
    These classes used fixed values of the fluid properties and do not rely
    on the `dbm` module.  Tests include jet of oil and gas, pure oil, and
    pure gas.

    """
    # Get properties for a blowout
    d0, m_gas, m_oil, rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil, \
        sigma_oil, rho, mu = get_blowout_properties()

    # Load the correct answer to the model properties
    de_max_gas, de_max_oil = get_blowout_ans()

    # Create a ModelBase object
    spill = psm.ModelBase(rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil,
                               sigma_oil, rho, mu)

    # Check that parameters are correct
    check_properties(rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil, sigma_oil,
                     rho, mu, spill)

    # Try setting the properties by using the .update() method
    spill.update_properties(rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil,
                      sigma_oil, rho, mu)

    # Check that the parameters are still correct
    check_properties(rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil, sigma_oil,
                     rho, mu, spill)

    # Try using the method to get a maximum gas and droplet size
    de_max_gas_model = spill.get_de_max(0)
    de_max_oil_model = spill.get_de_max(1)

    # Compare results to the correct answer
    assert_approx_equal(de_max_gas_model, de_max_gas, significant = 6)
    assert_approx_equal(de_max_oil_model, de_max_oil, significant = 6)

    # Simulate a given release condition
    spill.simulate(d0, m_gas, m_oil)

    # Create the particle size distributions
    nbins_gas = 10
    nbins_oil = 10
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    # Check that the object stores the correct attributes --------------------
    assert spill.d0 == d0
    assert spill.nbins_gas == nbins_gas
    assert np.sum(spill.m_gas) == np.sum(spill.m_gas)
    assert spill.nbins_oil == nbins_oil
    assert np.sum(spill.m_oil) == np.sum(spill.m_oil)
    assert spill.model_gas == 'wang_etal'
    assert spill.model_oil == 'sintef'
    assert spill.pdf_gas == 'lognormal'
    assert spill.pdf_oil == 'rosin-rammler'

    # Check that the model stores the right solution -------------------------
    assert_approx_equal(spill.d50_gas, 0.01134713688939418, significant=6)
    assert_approx_equal(spill.d50_oil, 0.0033149657926870454,significant=6)
    assert_approx_equal(spill.de_max_gas, 0.0313470341427564,significant=6)
    assert_approx_equal(spill.de_max_oil, 0.007475379384955715,
        significant=6)
    assert spill.sigma_ln_gas == 0.27
    assert_approx_equal(spill.k_oil, -0.6931471805599453,significant=6)
    assert spill.alpha_oil == 1.8

    de_gas = np.array([0.0057077 , 0.00655033, 0.00751736, 0.00862716,
        0.0099008, 0.01136247, 0.01303992, 0.01496502, 0.01717432,
        0.01970979])
    vf_gas = np.array([0.01545088, 0.0432876 , 0.09350044, 0.15570546,
        0.19990978, 0.19788106, 0.15101303, 0.08885147, 0.04030462,
        0.01409565])
    de_oil = np.array([0.00037551, 0.00053191, 0.00075346, 0.00106728,
        0.00151182, 0.0021415 , 0.00303346, 0.00429693, 0.00608665,
        0.00862181])
    vf_oil = np.array([0.00876514, 0.01619931, 0.02961302, 0.05303846,
        0.09143938, 0.14684062, 0.20676085, 0.22874507, 0.16382356,
        0.05477458])
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    # Try Li et al. for gas --------------------------------------------------
    spill.simulate(d0, m_gas, m_oil, model_gas='li_etal')

    # Create the particle size distributions
    nbins_gas = 10
    nbins_oil = 10
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    # Check whether the values were updated correctly
    assert_approx_equal(spill.d50_gas, 0.0065216270047474025, significant=6)
    assert_approx_equal(spill.k_gas, -0.6931471805599453,significant=6)
    assert spill.alpha_gas == 1.8

    de_gas = np.array([0.00185333, 0.00238478, 0.00306863, 0.00394856,
        0.00508083, 0.00653777, 0.00841249, 0.0108248 , 0.01392884,
        0.01792298])
    vf_gas = np.array([0.02515921, 0.06286577, 0.12110766, 0.17987423,
        0.20597111, 0.18183759, 0.12376591, 0.0649469 , 0.02627579,
        0.00819583])
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    # Try Li et al. for oil --------------------------------------------------
    # Simulate a given release condition
    spill.simulate(d0, m_gas, m_oil, model_oil='li_etal')

    # Create the particle size distributions
    nbins_gas = 10
    nbins_oil = 10
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    # Check whether the values were updated correctly
    assert_approx_equal(spill.d50_oil, 0.014962419470081935, significant=6)
    assert_approx_equal(spill.k_oil, -0.6931471805599453,significant=6)
    assert spill.alpha_oil == 1.8

    de_gas = np.array([0.0057077 , 0.00655033, 0.00751736, 0.00862716,
        0.0099008, 0.01136247, 0.01303992, 0.01496502, 0.01717432,
        0.01970979])
    vf_gas = np.array([0.01545088, 0.0432876 , 0.09350044, 0.15570546,
        0.19990978, 0.19788106, 0.15101303, 0.08885147, 0.04030462,
        0.01409565])
    de_oil = np.array([0.00169489, 0.00240083, 0.00340081, 0.00481728,
        0.00682373, 0.00966589, 0.01369183, 0.01939462, 0.02747269,
        0.03891536])
    vf_oil = np.array([0.00876514, 0.01619931, 0.02961302, 0.05303846,
        0.09143938, 0.14684062, 0.20676085, 0.22874507, 0.16382356,
        0.05477458])
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    # Try to run a case of a pure oil release using Sintef model -------------
    spill.update_properties(None, None, None, rho_oil, mu_oil,
               sigma_oil, rho, mu)
    m_gas = np.array([0.])
    spill.simulate(d0, m_gas, m_oil)

    # Create the particle size distributions
    nbins_gas = 0
    nbins_oil = 10
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    # Check whether the values were updated correctly
    assert_approx_equal(spill.d50_oil, 0.0033149657926870454, significant=6)
    assert_approx_equal(spill.k_oil, -0.6931471805599453,significant=6)
    assert spill.alpha_oil == 1.8

    de_gas = np.array([])
    vf_gas = np.array([])
    de_oil = np.array([0.00037551, 0.00053191, 0.00075346, 0.00106728,
        0.00151182, 0.0021415 , 0.00303346, 0.00429693, 0.00608665,
        0.00862181])
    vf_oil = np.array([0.00876514, 0.01619931, 0.02961302, 0.05303846,
        0.09143938, 0.14684062, 0.20676085, 0.22874507, 0.16382356,
        0.05477458])
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    # Rerun this last case using the PureJet class object --------------------
    jet = psm.PureJet(rho_oil, mu_oil, sigma_oil, rho, mu, 1)
    jet.simulate(d0, m_oil)

    # Create the particle size distributions
    nbins = 10
    de_oil_model, vf_oil_model = jet.get_distributions(nbins)

    # Check whether the values were updated correctly
    assert_approx_equal(jet.d50_oil, 0.0033149657926870454, significant=6)
    assert_approx_equal(jet.k_oil, -0.6931471805599453,significant=6)
    assert jet.alpha_oil == 1.8
    assert_array_almost_equal(jet.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(jet.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(jet.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(jet.vf_oil, vf_oil, decimal=6)

    # Try to run a case of a pure oil release using Li et al model -----------
    spill.update_properties(None, None, None, rho_oil, mu_oil,
               sigma_oil, rho, mu)
    m_gas = np.array([0.])
    spill.simulate(d0, m_gas, m_oil, model_oil='li_etal')

    # Create the particle size distributions
    nbins_gas = 0
    nbins_oil = 10
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    de_gas = np.array([])
    vf_gas = np.array([])
    de_oil = np.array([0.00343099, 0.00486004, 0.0068843 , 0.00975168,
        0.01381336, 0.01956677, 0.02771654, 0.03926078, 0.05561331,
        0.07877685])
    vf_oil = np.array([0.00876514, 0.01619931, 0.02961302, 0.05303846,
        0.09143938, 0.14684062, 0.20676085, 0.22874507, 0.16382356,
        0.05477458])
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    # Rerun this last case using the PureJet class object --------------------
    jet = psm.PureJet(rho_oil, mu_oil, sigma_oil, rho, mu, 1)
    jet.simulate(d0, m_oil, model='li_etal')

    # Create the particle size distributions
    nbins = 10
    de_oil_model, vf_oil_model = jet.get_distributions(nbins)

    # Check whether the values were updated correctly
    assert_array_almost_equal(jet.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(jet.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(jet.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(jet.vf_oil, vf_oil, decimal=6)

    # Try to run a case of a pure gas release --------------------------------
    spill.update_properties(rho_gas, mu_gas, sigma_gas, None, None,
               None, rho, mu)
    m_gas = 7.4
    m_oil = np.array([0.])
    spill.simulate(d0, m_gas, m_oil, model_gas='li_etal')

    # Create the particle size distributions
    nbins_gas = 10
    nbins_oil = 0
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    de_gas = np.array([0.00385017, 0.00495422, 0.00637486, 0.00820287,
        0.01055507, 0.01358177, 0.01747638, 0.02248779, 0.02893623,
        0.03723378])
    vf_gas = np.array([0.02515921, 0.06286577, 0.12110766, 0.17987423,
        0.20597111, 0.18183759, 0.12376591, 0.0649469 , 0.02627579,
        0.00819583])
    de_oil = np.array([])
    vf_oil = np.array([])
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    # Rerun this last case using the PureJet class object --------------------
    jet = psm.PureJet(rho_gas, mu_gas, sigma_gas, rho, mu, 0)
    jet.simulate(d0, m_gas, model='li_etal', pdf='lognormal')

    # Create the particle size distributions
    nbins = 10
    de_gas_model, vf_gas_model = jet.get_distributions(nbins)

    # Check whether the values were updated correctly
    assert_array_almost_equal(jet.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(jet.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(jet.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(jet.vf_oil, vf_oil, decimal=6)

    # Try to run a case of pure gas relese using Wang et al. -----------------
    spill.update_properties(rho_gas, mu_gas, sigma_gas, None, None,
               None, rho, mu)
    m_gas = 7.4
    m_oil = np.array([0.])
    spill.simulate(d0, m_gas, m_oil)

    # Create the particle size distributions
    nbins_gas = 10
    nbins_oil = 0
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    de_gas = np.array([0.01011325, 0.01160628, 0.01331973, 0.01528614, 
        0.01754285, 0.02013273, 0.02310494, 0.02651596, 0.03043054, 
        0.03492304])
    vf_gas = np.array([0.01545088, 0.0432876 , 0.09350044, 0.15570546,
        0.19990978, 0.19788106, 0.15101303, 0.08885147, 0.04030462,
        0.01409565])
    de_oil = np.array([])
    vf_oil = np.array([])
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    # Rerun this last case using the PureJet class object --------------------
    jet = psm.PureJet(rho_gas, mu_gas, sigma_gas, rho, mu, 0)
    jet.simulate(d0, m_gas, model='wang_etal', pdf='lognormal')

    # Create the particle size distributions
    nbins = 10
    de_gas_model, vf_gas_model = jet.get_distributions(nbins)

    # Check whether the values were updated correctly
    assert_array_almost_equal(jet.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(jet.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(jet.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(jet.vf_oil, vf_oil, decimal=6)


def test_psm_Model():
    """
    Test the `Model` class

    Test all of the functionality in the `Model` class. This class uses fluid
    property values computed by the `dbm` module. The main thing that needs
    to be tested is that the connections to the `BaseModel` class are
    implemented correctly.

    """
    # Get the TAMOC objects for a typical spill
    profile, oil, mass_flux, z0, Tj = get_blowout_model()

    # Create a psm.Model object
    spill = psm.Model(profile, oil, mass_flux, z0, Tj)

    # Simulate breakup from a blowout ----------------------------------------
    d0 = 0.15
    spill.simulate(d0, model_gas='wang_etal', model_oil='sintef')

    # Create the particle size distributions
    nbins_gas = 10
    nbins_oil = 15
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    de_max_gas = 0.029811727894630175
    de_max_oil = 0.019433783423489368
    d50_gas = 0.004757196222196337
    d50_oil = 0.004183783481056991
    de_gas = np.array([0.00239291, 0.00274617, 0.00315159, 0.00361687, 
        0.00415083, 0.00476362, 0.00546688, 0.00627396, 0.0072002,
        0.00826317])
    vf_gas = np.array([0.01545088, 0.0432876 , 0.09350044, 0.15570546,
        0.19990978, 0.19788106, 0.15101303, 0.08885147, 0.04030462,
        0.01409565])
    de_oil = np.array([0.0004472 , 0.00056405, 0.00071143, 0.00089732,
        0.00113177, 0.00142749, 0.00180047, 0.00227091, 0.00286426,
        0.00361265, 0.00455658, 0.00574714, 0.00724879, 0.00914279,
        0.01153166])
    vf_oil = np.array([0.00522565, 0.00788413, 0.01185467, 0.01773296,
        0.02631885, 0.03859967, 0.05559785, 0.07791868, 0.10476347,
        0.13228731, 0.15193437, 0.15128424, 0.12160947, 0.0710618 ,
        0.02592687])
    assert_approx_equal(spill.get_de_max(0), de_max_gas)
    assert_approx_equal(spill.get_de_max(1), de_max_oil)
    assert_approx_equal(spill.get_d50(0), d50_gas)
    assert_approx_equal(spill.get_d50(1), d50_oil)
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    # Switch oil model to li_etal --------------------------------------------
    spill.simulate(d0, model_gas='wang_etal', model_oil='li_etal')

    # Create the particle size distributions
    nbins_gas = 10
    nbins_oil = 15
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    d50_oil = 0.0022201887727817814
    de_oil = np.array([0.00023732, 0.00029932, 0.00037753, 0.00047618,
        0.00060059, 0.00075752, 0.00095545, 0.00120509, 0.00151996,
        0.00191711, 0.00241802, 0.00304981, 0.00384668, 0.00485176,
        0.00611945])
    vf_oil = np.array([0.00522565, 0.00788413, 0.01185467, 0.01773296,
        0.02631885, 0.03859967, 0.05559785, 0.07791868, 0.10476347,
        0.13228731, 0.15193437, 0.15128424, 0.12160947, 0.0710618 ,
        0.02592687])
    assert_approx_equal(spill.get_de_max(0), de_max_gas)
    assert_approx_equal(spill.get_de_max(1), de_max_oil)
    assert_approx_equal(spill.get_d50(0), d50_gas)
    assert_approx_equal(spill.get_d50(1), d50_oil)
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    # Switch gas model to li_etal --------------------------------------------
    spill.simulate(d0, model_gas='li_etal')

    # Create the particle size distributions
    nbins_gas = 10
    nbins_oil = 15
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    de_max_gas = 0.029811727894630175
    d50_gas = 0.0004202930757153284
    d50_oil = 0.004183783481056991
    de_gas = np.array([0.00011944, 0.00015369, 0.00019776, 0.00025447, 
        0.00032744, 0.00042133, 0.00054215, 0.00069762, 0.00089766,
        0.00115507])
    vf_gas = np.array([0.02515921, 0.06286577, 0.12110766, 0.17987423,
        0.20597111, 0.18183759, 0.12376591, 0.0649469 , 0.02627579,
        0.00819583])
    de_oil = np.array([0.0004472 , 0.00056405, 0.00071143, 0.00089732,
        0.00113177, 0.00142749, 0.00180047, 0.00227091, 0.00286426,
        0.00361265, 0.00455658, 0.00574714, 0.00724879, 0.00914279,
        0.01153166])
    vf_oil = np.array([0.00522565, 0.00788413, 0.01185467, 0.01773296,
        0.02631885, 0.03859967, 0.05559785, 0.07791868, 0.10476347,
        0.13228731, 0.15193437, 0.15128424, 0.12160947, 0.0710618 ,
        0.02592687])
    assert_approx_equal(spill.get_de_max(0), de_max_gas)
    assert_approx_equal(spill.get_de_max(1), de_max_oil)
    assert_approx_equal(spill.get_d50(0), d50_gas)
    assert_approx_equal(spill.get_d50(1), d50_oil)
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    # Try a case with no gas -------------------------------------------------
    spill.update_z0(1000.)
    spill.simulate(d0, model_gas='wang_etal', model_oil='sintef')
    # Create the particle size distributions
    nbins_gas = 10
    nbins_oil = 15
    de_gas_model, vf_gas_model, de_oil_model, vf_oil_model = \
        spill.get_distributions(nbins_gas, nbins_oil)

    de_max_oil = 0.017327034580027646
    d50_gas = 0.0
    d50_oil = 0.007683693892124441
    de_gas = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan])
    vf_gas = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    de_oil = np.array([0.00082131, 0.00103591, 0.00130657, 0.00164796,
        0.00207855, 0.00262164, 0.00330664, 0.00417061, 0.00526033,
        0.00663478, 0.00836835, 0.01055487, 0.0133127 , 0.01679111,
        0.02117837])
    vf_oil = np.array([0.00522565, 0.00788413, 0.01185467, 0.01773296,
        0.02631885, 0.03859967, 0.05559785, 0.07791868, 0.10476347,
        0.13228731, 0.15193437, 0.15128424, 0.12160947, 0.0710618 ,
        0.02592687])
    assert_approx_equal(spill.get_de_max(1), de_max_oil)
    assert_approx_equal(spill.get_d50(0), d50_gas)
    assert_approx_equal(spill.get_d50(1), d50_oil)
    assert_array_almost_equal(spill.de_gas, de_gas, decimal=6)
    assert_array_almost_equal(spill.vf_gas, vf_gas, decimal=6)
    assert_array_almost_equal(spill.de_oil, de_oil, decimal=6)
    assert_array_almost_equal(spill.vf_oil, vf_oil, decimal=6)

    return spill