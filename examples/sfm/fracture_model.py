"""
fracture_model.py
-----------------

Demonstrate features of the subsurface fracture model.

WARNING!!!
THIS SCRIPT IS UNDER CONSTRUCTION AND DOES NOT YET REPRESENT AN ACCURATE
SIMULATION OF THE OCEAN SUBSURFACE.  PLEASE DO NOT USE THIS FOR ANY PURPOSES
AT THIS TIME.  Scott Socolofsky, 02/03/2022.

"""
# S. Socolofsky, January 2021, <socolofs@tamu.edu>

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient, dbm
from tamoc import subsurface_fracture_model as sfm

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    # Define a profile
    nc = '../../tamoc/test/output/test_bm54.nc'
    profile = ambient.Profile(nc, chem_names='all')
    
    # Set the simulation parameters
    H = profile.z_max   # water depth (m)
    Hs = 200.        # Subsurface layer thickness (m)
    x0 = np.array([0., 0.])
                     # Planar origin of the fracture network (m)
    dx = np.array([0.10, 0.15, 0.01])
                     # Average x-, y-, and z-displacement for the fluctuating
                     # component of each straight segment (m/segment)
    du = np.array([0., 0., 0.03])
                     # Pseudo-velocity of each straight-segement (m/m)
    mu_D = 0.01      # Average diameter of each segment (m)
    sigma_D = 0.005  # Standard deviation of the diameters of each segment (m)
    delta_s = 1.     # Average displacement to use in building fracture 
                     # network (m)
    
    # Define a fluid
    composition = ['methane', 'ethane', 'propane']
    mass_frac = np.array([0.93, 0.05, 0.02])
    gas = dbm.FluidMixture(composition)
    
    # Extend the profile deeper
    nc = '../../tamoc/test/output/BM54_subsurface.nc'
    profile.extend_profile_deeper(H + Hs, nc)
    profile.close_nc()
    
    # Create a model object with the fracture network
    frac = sfm.Model(profile, H, Hs, dx, du, mu_D, sigma_D, x0, delta_s)
    
    # Show the network
    frac.show_network()
    
    # Simulate the gas transport through a filled pipe network.
    frac.simulate_pipe_flow(1.0, mass_frac, gas)
    
    # Plot default results
    frac.plot_state_space()
    frac.plot_component_map()