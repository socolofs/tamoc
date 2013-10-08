"""
Single Bubble Model:  Droplet simulations
=========================================

Use the ``TAMOC`` `single_bubble_model` to simulate the trajectory of a light
oil droplet rising through the water column. This script demonstrates the
typical steps involved in running the single bubble model.

It uses the ambient data stored in the file `../test/output/test_bm54.nc`,
created by the `test_ambient` module.  Please make sure all tests have 
passed before running this script or modify the script to use a different
source of ambient data.

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

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
    
    # Create a light oil droplet particle to track
    composition = ['benzene', 'toluene', 'ethylbenzene']
    drop = dbm.FluidParticle(composition, fp_type=1.)
    
    # Set the mole fractions of each component at release.
    mol_frac = np.array([0.4, 0.3, 0.3])
    
    # Specify the remaining particle initial conditions
    de = 0.02
    z0 = 1000.
    T0 = 273.15 + 30.
    
    # Simulate the trajectory through the water column and plot the results
    sbm.simulate(drop, z0, de, mol_frac, T0, K_T=1, fdis=1e-8, delta_t=10.)
    sbm.post_process()
    
    # Save the simulation to a netCDF file
    sbm.save_sim('./drop.nc', '../../test/output/test_bm54.nc', 
                 'Results of ./drops.py script')
    
    # Save the data for importing into Matlab
    sbm.save_txt('./drop.txt', '../../test/output/test_bm54.nc', 
                 'Results of ./drops.py script')

