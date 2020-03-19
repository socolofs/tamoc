"""
Single Bubble Model:  Bubble simulations
========================================

Use the ``TAMOC`` `single_bubble_model` to simulate the trajectory of a 
natural gas bubble rising through the water column.  This script demonstrates
the typical steps involved in running the single bubble model.

It uses the ambient data stored in the file `../test/output/test_bm54.nc`,
created by the `test_ambient` module.  Please make sure all tests have 
passed before running this script or modify the script to use a different
source of ambient data.

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient
from tamoc import dbm
from tamoc import seawater
from tamoc import single_bubble_model

import numpy as np

if __name__ == '__main__':
    
    # Open an ambient profile object from the netCDF dataset
    nc = '../../test/output/test_bm54.nc'
    bm54 = ambient.Profile(nc, chem_names='all')
    bm54.close_nc()
    
    # Initialize a single_bubble_model.Model object with this data
    sbm = single_bubble_model.Model(bm54)
    
    # Create a natural gas particle to track
    composition = ['methane', 'ethane', 'propane', 'oxygen']
    gas = dbm.FluidParticle(composition, fp_type=0.)
    
    # Set the mole fractions of each component at release.  Note that oxygen
    # is listed so that stripping from the water column can be simulated, but
    # that the initial mole fraction of oxygen is zero.  This is the normal
    # behavior:  any component not listed in the composition, even if it is
    # present in the ambient CTD data, will not be simulated.  The 
    # `composition` variable is the only means to tell the single bubble 
    # model what chemicals to track.
    mol_frac = np.array([0.90, 0.07, 0.03, 0.0])
    
    # Specify the remaining particle initial conditions
    de = 0.005
    z0 = 1000.
    T0 = 273.15 + 30.
    
    # Simulate the trajectory through the water column and plot the results
    sbm.simulate(gas, z0, de, mol_frac, T0, K=0.5, K_T=1, fdis=1e-8, 
                 delta_t=1.)
    sbm.post_process()
    
    # Save the simulation to a netCDF file
    sbm.save_sim('./bubble.nc', '../../test/output/test_bm54.nc', 
                 'Results of ./bubbles.py script')
    
    # Save the data for importing into Matlab
    sbm.save_txt('./bubble', '../../test/output/test_bm54.nc', 
                 'Results of ./bubbles.py script')

