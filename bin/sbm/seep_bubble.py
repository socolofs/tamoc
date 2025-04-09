"""
Single Bubble Model:  Natural seep bubble simulations
=====================================================

Use the ``TAMOC`` `single_bubble_model` to simulate the trajectory of a light
hydrocarbon bubble rising through the water column. This script demonstrates
the typical steps involved in running the single bubble model for a natural
seep bubble.

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
from tamoc import dispersed_phases

import numpy as np

if __name__ == '__main__':
    
    # Open an ambient profile object from the netCDF dataset
    nc = '../../tamoc/test/output/test_bm54.nc'
    bm54 = ambient.Profile(nc, chem_names='all')
    bm54.close_nc()
    
    # Initialize a single_bubble_model.Model object with this data
    sbm = single_bubble_model.Model(bm54)
    
    # Create a light gas bubble to track
    composition = ['methane', 'ethane', 'propane', 'oxygen']
    bub = dbm.FluidParticle(composition, fp_type=0.)
    
    # Set the mole fractions of each component at release.
    mol_frac = np.array([0.95, 0.03, 0.02, 0.])
    
    # Specify the remaining particle initial conditions
    de = 0.005
    z0 = 1000.
    T0 = 273.15 + 30.
    fdis = 1.e-15
    
    # Also, use the hydrate model from Jun et al. (2015) to set the 
    # hydrate shell formation time
    P = bm54.get_values(z0, 'pressure')
    m = bub.masses_by_diameter(de, T0, P, mol_frac)
    t_hyd = dispersed_phases.hydrate_formation_time(bub, z0, m, T0, bm54)
    
    # Simulate the trajectory through the water column and plot the results
    sbm.simulate(bub, z0, de, mol_frac, T0, K_T=1, fdis=fdis, t_hyd=t_hyd, 
                 delta_t=10.)
    sbm.post_process()
    
    # Save the simulation to a netCDF file
    sbm.save_sim('./seep_bubble.nc', '../../test/output/test_bm54.nc', 
                 'Results of ./seep_bubble.py script')
    
    # Save the data for importing into Matlab
    sbm.save_txt('./seep_bubble', '../../test/output/test_bm54.nc', 
                 'Results of ./seep_bubble.py script')

