"""
Blowout Module:  Blowout Simulation
===================================

This script demonstrates using the `blowout.Blowout` class to simulate a
synthetic subsea accidental oil well blowout.  This class does much of the 
work that would otherwise be done through scripting interfaces to the ``TAMOC`` suite of modules and functions.  

Here, we simulate a synthetic blowout with a dead oil flow rate of 20000
bbl/d and a gas-to-oil ratio of 2000 released through a 0.2 m diameter
orifice at 1000 m depth. We use the `dbm.FluidMixture` object to compute the
oil and gas equilibrium and properties, and we use the `particle_size_models`
to compute the gas bubble and oil droplet size distributions.

"""
# S. Socolofsky, March 2020, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient, dbm_utilities, blowout
from tamoc import bent_plume_model as bpm

import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    # We still need ambient CTD data, and the blowout.Blowout class can 
    # accept a netCDF4 dataset as input...load the dataset and include
    # uniform currents
    try:
        # Open the BM54 CTD profile if it exists
        nc = '../../test/output/test_BM54.nc'
        ctd = ambient.Profile(nc, chem_names='all')
        
        # Insert a constant crossflow velocity
        z = ctd.nc.variables['z'][:]
        ua = np.zeros(z.shape) + 0.09
        data = np.vstack((z, ua)).transpose()
        symbols = ['z', 'ua']
        units = ['m', 'm/s']
        comments = ['measured', 'arbitrary crossflow velocity']
        ctd.append(data, symbols, units, comments, 0)
        ctd.close_nc()
        
        # Prepare to send this .nc file to blowout.Blowout
        water = ctd
        current = None
        print('\nUsing CTD data in BM54.nc')
    
    except RuntimeError:
        
        # Tell the user to create the dataset
        print('\nCTD data not available; run test cases in ./test first.')
        
        # Use the world-ocean average instead
        print('Using the world-ocean average data instead')
        water = None
        current = np.array([0.09, 0., 0.])
    
    # Jet initial conditions
    x0 = 0.
    y0 = 0.
    u0 = 0.              # no produced water
    phi_0 = -np.pi / 2.  # vertical release
    theta_0 = 0.
    Tj = 273.15 + 35.
    
    # Create an oil mixture using the dbm_utilities functions...
    
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
    gor = 2000.      # ft^3/bbl at standard conditions
    z0 = 1000.       # release depth (m)
    d0 = 0.2         # orifice diameter (m)
    
    # Import the oil with the desired gas to oil ratio
    oil, mass_flux = dbm_utilities.get_oil(substance, q_oil, gor, ca)
    
    # Decide on the number of Lagrangian elements to use for gas bubbles
    # and oil droplets
    num_gas_elements = 15
    num_oil_elements = 15
    
    # Initialize the blowout.Blowout object with these conditions
    spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0,
                                phi_0, theta_0, num_gas_elements, 
                                num_oil_elements, water, current)
    
    # Run the simulation
    spill.simulate()
    
    # Plot the trajectory of the plume and the bubbles and droplets in the
    # plume
    spill.plot_state_space(1)
    
    # Plot all of the simulation variables
    spill.plot_all_variables(10)
    
    # Change the oil and gas flow rate
    spill.update_q_oil(30000.)
    spill.update_gor(1500.)
    
    # Re-run the simulation and re-plot the results
    spill.simulate()
    spill.plot_state_space(100)
    spill.plot_all_variables(110)
    
    # Try saving the bent plume model solution to the disk
    save_file = './output/blowout_obj.nc'
    ctd_file = '../../../test/output/test_BM54.nc'
    ctd_data = 'Default data in the BM54.nc dataset distributed with TAMOC'
    print('\nSaving present simulation to: ', save_file)
    spill.save_sim(save_file, ctd_file, ctd_data)
    
    # Try saving an ascii text file for output
    text_file = './output/blowout_obj.dat'
    print('\nSaving text output...')
    spill.save_txt(text_file, ctd_file, ctd_data)
    
    # We cannot load a saved simulation back into a Blowout object, but
    # we could load it into a bent_plume_model.Model object...try this now
    print('\nLoading saved simulation file...')
    sim = bpm.Model(simfile=save_file)
    sim.plot_state_space(200)
    
    print('\nDone.')
