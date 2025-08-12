"""
eos_tools.py
------------

Tools to assist with setting up `dbm` objects and in using `dbm` objects to
perform equations-of-state tasks for other simulation modules.

"""
# S. Socolofsky, Texas A&M University, August 2025, <socolofs@tamu.edu>

from tamoc import dbm_utilities

import numpy as np
from scipy.optimize import fsolve


def adjust_mass_frac_for_gor(dbm_mixture, mass_frac, gor):
    """
    Adjust a fluid mixture composition to match a specified GOR
    
    Adjust the mass fractions for a fluid mixture to match a specified
    gas-to-oil ratio, std ft^3 / bbl.  This function can operate on two 
    different types of fluid compositions.
    
    First, if the fluid mixture is a dead oil that does not contain C1-C4
    compounds, then these will be added to the mixture using the `natural_gas`
    function in the `dbm_utilities`.  Second, if C1-C4 compounds are prresent,
    a flash equilibrium calculation at standard conditions will be done to
    get the initial composition of gas and liquid petroleum.
    
    Once a live-oil mixture is created by one of the methods above, the 
    mass fractions for gas and oil at standard conditions will be considered
    fixed, and a ratio of gas to oil adjusted by iteration until the 
    desired gas-to-oil ratio is reached.  This search uses the function
    'gas_fraction' in the 'dbm_utilities' to find the correct gas fraction.
    
    Finally, a new set of mixture mass fractions is created that yields the
    desired gas-to-oil ratio at standard conditions, and this mass fraction
    along with the final `dbm.FluidMixture` object is returned.
    
    Parameters
    ----------
    dbm_mixture : dbm.FluidMixture
        A discrete bubble model `FluidMixture` object that contains the 
        chemical components and thermodynamic properties of the original
        fluid mixture.
    mass_frac : ndarray
        Array of mass fractions for each chemical component in the mixture
        for the initial composition.  
    gor : float
        Desired gas to oil ratio in std ft^3 / bbl.
    
    Returns
    -------
    dbm_mixture : dbm.FluidMixture
        A `dbm.FluidMixture` object that has potentially been updated with
        C1-C4 compounds.
    mass_frac : ndarray
        Array of adjusted mass fractions for each component of the fluid 
        mixture such that the flash equilibrium at standard conditions
        gives the desired gas-to-oil ratio    
    
    """
    # Get a general natural gas composition from TAMOC
    gas_components, gas_mass_frac, gas_delta_groups = \
        dbm_utilities.natural_gas()
    
    # Check whether the present mixture contains gas
    add_gas = True
    for gas in gas_components:
        if gas in dbm_mixture.composition:
            # The given mixture contains gas components...do not add gas
            add_gas = False
    
    # If we need to add gas, create a new dbm_mixture
    if add_gas:
        # Extract the dead-oil composition
        dead_composition = dbm_mixture.composition
        user_data = dbm_mixture.user_data
        delta = dbm_mixture.delta
        delta_groups = dbm_mixture.delta_groups
    
        # Add gas to the composition
        composition = gas_components + dead_composition
        delta_groups = np.vstack((gas_groups, delta_groups))
        
        # Create a new fluid mixture object with gas added
        dbm_mixture = dbm.FluidMixture(composition, delta=delta,
            delta_groups=delta_groups, user_data=user_data)
    
    # Set up standard conditions
    T_std, P_std = pete_stp()
    
    # Get an initial composition of gas- and liquid-phase petroleum
    if add_gas:
        # We had to add gas to a dead oil...assume all gas is in gas
        mf_gas = np.zeros(len(dbm_mixture.composition))
        mf_liq = np.zeros(len(dbm_mixture.composition))
        mf_gas[0:len(gas_fractions)] = gas_mass_frac
        mf_liq[len(gas_fractions):] = mass_frac
    else:
        # Do an initial equilibrium calculation
        m, xi, K = dbm_mixture.equilibrium(mass_frac, T_std, P_std)
        mf_gas = m[0,:]
        mf_liq = m[1,:]
    
    # Get a mass and volume of gas and liquid at standard conditions for
    # this guess of the composition
    v_gas = gor * 0.0283168465924   # ft^3 to m^3
    v_liq = 0.1589872949288          # 1 bbl to m^3
    m_gas = dbm_mixture.density(mf_gas, T_std, P_std)[0,0] * v_gas
    m_liq = dbm_mixture.density(mf_liq, T_std, P_std)[1,0] * v_liq
    beta_0 = m_gas / (m_gas + m_liq)       
    
    # Use the function in the dbm_utilities to find the correct beta
    beta = fsolve(dbm_utilities.gas_fraction, beta_0, args=(gor, 
        dbm_mixture, mf_gas, mf_liq, T_std, P_std))
    
    # Compute the final mixture mass fraction
    mass_frac = beta * mf_gas + (1. - beta) * mf_liq
    mass_frac = mass_frac / np.sum(mass_frac)
    
    # Return the results
    return (dbm_mixture, mass_frac)

def mass_flowratefrom_volume_flowrate(dbm_mixture, mass_frac, q0):
    """
    Commpute the mass flux from a dead-oil volume flux
    
    Convert barrels per day of oil to a total mixture mass flow rate in kg/s.
    Be sure to perform the `adjust_mass_frac_for_gor` function first so that
    the `mass_frac` composition will yield the correct gas-to-oil ratio. This
    function performs a flash equilibrium at standard conditions and then
    determines the total mixture mass flow rate such that the liquid mass flow
    rate will match the given volume flow rate in barrels per day.
    
    Parameters
    ----------
    dbm_mixture : dbm.FluidMixture
        A discrete bubble model `FluidMixture` object that contains the 
        chemical components and thermodynamic properties of the fluid mixture.
    mass_frac : ndarray
        Array of mass fractions for each chemical component in the mixture.
    q0 : float
        Desired liquid volume flow rate at standard conditions in barrels per
        day
    
    Returns
    -------
    m0 : ndarray
        Array of mass flow rates (kg/s) for each component of the mixture such
        that the liquid-phase mass flow rate at standard conditions will match
        the given volume flow rate in barrels per day.
    
    """
    # Get the standard petroleum engineering temperature and pressure
    T_std, P_std = pete_stp()

    # Perform an equilibrium calculation at these conditions
    m, xi, K = dbm_mixture.equilibrium(mass_frac, T_std, P_std)
    
    # Get the density of the liquid-phase petroleum
    rho_liq = dbm_mixture.density(m[1,:], T_std, P_std)[1,0]
    
    # Compute the mass per day to give the desired volume per day
    md_liq = q0 * 0.1589872949288 / (3600. * 24.) * rho_liq
    
    # Determine the scale factor to apply to the liquid and gas mass fractions
    # to achieve the desired mass flow rate of liquid
    m_adjust = md_liq / np.sum(m[1,:])
    
    # Get the adjusted total mixture mass flow rates by component
    m0 = m[0,:] * m_adjust + m[1,:] * m_adjust
    
    # Return the component mass flow rates
    return m0
    
def pete_stp():
    """
    Standard conditions for petroleum engineering 
    
    Standard temperature and pressure for petroleum engineering applications
    of 15 deg C and atmospheric pressure.
    
    Returns
    -------
    T_std : float
        Standard temperature, K
    P_std : float
        Standard pressure, Pa
    
    """
    return (273.15 + 15., 101325.)

def get_flowrate_gor(dbm_mixture, m0, report_result=False):
    """
    Compute the volume flow rate and GOR for a given mass flow rate
    
    Compute the liquid flow rate (bbl/d) and the gas-to-oil ratio 
    (std ft^3/bbl) given the mass flow rate (kg/s) of each component of a 
    fluid mixture.
    
    Parameters
    ----------
    dbm_mixture : dbm.FluidMixture
        A discrete bubble model `FluidMixture` object that contains the 
        chemical components and thermodynamic properties of the fluid mixture.
    mass_frac : ndarray
        Array of mass flowrates (kg/s) for each chemical component in the
        mixture.
    report_result : bool, default=False
        A boolean flag indicating whether to print the results to the screen
    
    Returns
    -------
    q0 : float
        Volume flowrate of the liquid-phase in bbl/d
    gor : float
        Gas-to-oil ratio in std ft^3 / bbl
    
    """
    # Report the voluem flow rate and GOR...get the flash equilibrium
    T_stp, P_stp = pete_stp()
    m, xi, K = dbm_mixture.equilibrium(m0, T_stp, P_stp)
    
    # Compute the densities of gas and liquid
    rho_gas = dbm_mixture.density(m[0,:], T_stp, P_stp)[0,0]
    rho_liq = dbm_mixture.density(m[1,:], T_stp, P_stp)[1,0]
    
    # Compute the volume flow rate in m^3/s
    vf_gas = np.sum(m[0,:]) / rho_gas
    vf_liq = np.sum(m[1,:]) / rho_liq
    
    # Convert gas volume to ft^3 and liquid volume to bbl
    vf_gas = vf_gas * 35.314666721
    vf_liq = vf_liq * 6.2898107704
    
    # Get the GOR
    gor = vf_gas / vf_liq
    
    # Get the liquid flow rate in bbl/day
    q_liq = vf_liq * 3600. * 24.
    
    if report_result:
        print('\nComponent mass flowrates give volume flowrates of:')
        print('    Q_0 = %g bbl/d' % q_liq)
        print('    GOR = %g ft^3/bbl\n' % gor)
    
    return (q_liq, gor)