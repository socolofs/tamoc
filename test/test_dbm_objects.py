"""
Unit tests for the `dbm` module of ``TAMOC``

Provides testing of instantiation of the class objects defined in ``dbm.py``.

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

from tamoc import dbm

import numpy as np
from numpy.testing import *

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def mixture_attributes(dbm_obj, composition, nc):
    """
    Test that the object attributes stored in a dbm object match those
    specified in the arguments passed through the above list.
    """
    assert dbm_obj.composition == composition
    assert dbm_obj.nc == nc
    assert dbm_obj.issoluble == True

def chem_properties(dbm_obj, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, 
                    nu_bar, B, dE, K_salt):
    """
    Test that the chemical properties stored in a dbm object match the 
    values specified in the arguments passed through the above list.
    """
    assert_array_almost_equal(dbm_obj.delta, delta, decimal=6)
    assert_array_almost_equal(dbm_obj.M, M, decimal=6)
    assert_array_almost_equal(dbm_obj.Pc, Pc, decimal=6)
    assert_array_almost_equal(dbm_obj.Tc, Tc, decimal=6)
    assert_array_almost_equal(dbm_obj.omega, omega, decimal=6)
    assert_array_almost_equal(dbm_obj.kh_0, kh_0, decimal=6)
    assert_array_almost_equal(dbm_obj.neg_dH_solR, neg_dH_solR, decimal=6)
    assert_array_almost_equal(dbm_obj.nu_bar, nu_bar, decimal=6)
    assert_array_almost_equal(dbm_obj.B, B, decimal=6)
    assert_array_almost_equal(dbm_obj.dE, dE, decimal=6)
    assert_array_almost_equal(dbm_obj.K_salt, K_salt, decimal=6)

def inert_attributes(dbm_obj, isfluid, iscompressible, rho_p, gamma, beta, 
                     co):
    """
    Test that parameters stored in a dbm object match the values specified 
    in the arguments passed through the above list and the default values
    hard-wired in the creator method.
    """
    assert dbm_obj.isfluid is isfluid
    assert dbm_obj.iscompressible is iscompressible
    assert dbm_obj.rho_p == rho_p
    assert dbm_obj.gamma == gamma
    assert dbm_obj.beta == beta
    assert dbm_obj.co == co
    assert dbm_obj.issoluble is False
    assert dbm_obj.nc == 1
    assert dbm_obj.composition == ['inert']

# ----------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------

def test_objects():
    """
    Test the class instantiation functions to ensure proper creations of class
    instances.
    """
    # Define the properties of a simple fluid mixture
    comp = ['oxygen', 'nitrogen', 'carbon_dioxide']
    delta = np.zeros((3, 3))
    M = np.array([0.031998800000000001, 0.028013400000000001, 
                  0.04401])
    Pc = np.array([5042827.4639999997, 3399806.1560000004, 7373999.99902408])
    Tc = np.array([154.57777777777773, 126.19999999999999, 
                   304.12])
    omega = np.array([0.0216, 0.0372, 0.225])
    kh_0 = np.array([4.1054468295090059e-07, 1.7417658031088084e-07, 
                     1.47433500e-05])
    neg_dH_solR = np.array([1650.0, 1300.0, 2368.988311])
    nu_bar = np.array([3.20000000e-05, 3.3000000000000e-05, 
                       3.20000000e-05])
    B = np.array([4.2000000000000004e-06, 7.9000000000000006e-06, 
                  5.00000000e-06])
    dE = np.array([18380.044045116938, 19636.083501503061, 
                   16747.19275181])
    K_salt = np.array([0.000169, 0.0001834, 0.0001323])
    
    # Initiate a simple mixture from a composition list
    air = dbm.FluidMixture(comp)
    mixture_attributes(air, comp, 3)
    chem_properties(air, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
                    B, dE, K_salt)
    
    bub = dbm.FluidParticle(comp)
    mixture_attributes(bub, comp, 3)
    chem_properties(bub, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
                    B, dE, K_salt)
    
    # Initiate a simple mixture from a composition list with delta specified
    air = dbm.FluidMixture(comp, delta)
    mixture_attributes(air, comp, 3)
    chem_properties(air, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
                    B, dE, K_salt)
    
    bub = dbm.FluidParticle(comp, delta)
    mixture_attributes(bub, comp, 3)
    chem_properties(bub, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
                    B, dE, K_salt)
    
    # Define the properties of a single-component mixture
    comp = 'oxygen'
    delta = np.zeros((1,1))
    M = np.array([0.031998800000000001])
    Pc = np.array([5042827.4639999997])
    Tc = np.array([154.57777777777773])
    omega = np.array([0.021600000000000001])
    kh_0 = np.array([4.1054468295090059e-07])
    neg_dH_solR = np.array([1650.0])
    nu_bar = np.array([3.1999999999999999e-05])
    B = np.array([4.2000000000000004e-06])
    dE = np.array([18380.044045116938])
    K_salt = np.array([0.000169])
    
    # Initiate a single-component mixture from a list
    o2 = dbm.FluidMixture([comp])
    mixture_attributes(o2, [comp], 1)
    chem_properties(o2, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
                    B, dE, K_salt)
    
    bub = dbm.FluidParticle([comp])
    mixture_attributes(bub, [comp], 1)
    chem_properties(bub, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
                    B, dE, K_salt)
    
    # Initiate a single-componment mixture from a string with delta specified
    o2 = dbm.FluidMixture(comp, delta)
    mixture_attributes(o2, [comp], 1)
    chem_properties(o2, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
                    B, dE, K_salt)
    
    bub = dbm.FluidParticle(comp, delta)
    mixture_attributes(bub, [comp], 1)
    chem_properties(bub, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
                    B, dE, K_salt)
    
    # Initiate a single-componet mixture from a string with scalar delta
    o2 = dbm.FluidMixture(comp, 0.)
    mixture_attributes(o2, [comp], 1)
    chem_properties(o2, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
                    B, dE, K_salt)
    
    bub = dbm.FluidParticle(comp, 0.)
    mixture_attributes(bub, [comp], 1)
    chem_properties(bub, delta, M, Pc, Tc, omega, kh_0, neg_dH_solR, nu_bar, 
                    B, dE, K_salt)
    
    # Define the properties of an inert fluid particle
    isfluid = True
    iscompressible = False 
    rho_p = 870.
    gamma = 29., 
    beta = 0.0001 
    co= 1.0e-9
    
    # Initiate an inert fluid particle with different combinations of input
    # variables
    oil = dbm.InsolubleParticle(isfluid, iscompressible)
    inert_attributes(oil, isfluid, iscompressible, 930., 30., 0.0007,
                     2.90075e-9)
    
    oil = dbm.InsolubleParticle(isfluid, iscompressible, rho_p = rho_p)
    inert_attributes(oil, isfluid, iscompressible, rho_p, 30., 0.0007,
                     2.90075e-9)
    
    oil = dbm.InsolubleParticle(isfluid, iscompressible, gamma = gamma)
    inert_attributes(oil, isfluid, iscompressible, 930., gamma, 0.0007,
                     2.90075e-9)
    
    oil = dbm.InsolubleParticle(isfluid, iscompressible, beta = beta)
    inert_attributes(oil, isfluid, iscompressible, 930., 30., beta,
                     2.90075e-9)
    
    oil = dbm.InsolubleParticle(isfluid, iscompressible, co = co)
    inert_attributes(oil, isfluid, iscompressible, 930., 30., 0.0007,
                     co)
    
    oil = dbm.InsolubleParticle(isfluid, iscompressible, rho_p, gamma, beta,
                                co)
    inert_attributes(oil, isfluid, iscompressible, rho_p, gamma, beta, co)
    
    oil = dbm.InsolubleParticle(isfluid, iscompressible, beta = beta, 
                                rho_p = rho_p, gamma = gamma, co = co)
    inert_attributes(oil, isfluid, iscompressible, rho_p, gamma, beta, co)

