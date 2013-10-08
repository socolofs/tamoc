"""
Stratified Plume Model:  Lake simulation
========================================

Use the ``TAMOC`` `stratified_plume_model` to simulate an inert lake plume. 
This script demonstrates the typical steps involved in running the single 
bubble model with a non-reactive (inert) particle.

This simulation uses the ambient data stored in the file
`./tamoc/data/lake.dat`. This module first organizes this data and stores the
necessary netCDF file in `../test/output`. Please make sure this directory
exists before running this file.

"""
# S. Socolofsky, August 2013, Texas A&M University <socolofs@tamu.edu>.
from tamoc import ambient
from tamoc import dbm
from tamoc import stratified_plume_model
import lake_bub

from datetime import datetime
from netCDF4 import date2num

import numpy as np


if __name__ == '__main__':

    # Get the ambient CTD profile data
    nc = '../../test/output/lake.nc'
    try:
        # Open the lake dataset as a Profile object if it exists
        lake = ambient.Profile(nc, chem_names=['oxygen', 'nitrogen', 'argon'])
        
    except RuntimeError:
        # Create the lake netCDF dataset and get the Profile object
        lake = lake_bub.get_lake_data()
    
    # Create the stratified plume model object
    spm = stratified_plume_model.Model(lake)
        
    # Create the dispersed phase particles
    sphere = dbm.InsolubleParticle(False, False, rho_p=15.)
    z0 = 46.
    particles = []
    
    # Small particle
    Q_N = 100. / 60. / 60. 
    de = 0.01
    lambda_1 = 0.7
    particles.append(stratified_plume_model.particle_from_Q(lake, z0, sphere, 
                   1., Q_N,  de, lambda_1))
    
    # Initialize a simulation
    R = 6.5 / 2.
    spm.simulate(particles, z0, R, maxit=5, delta_z = 0.2)
    
    # Save the model results
    spm.save_sim('../../test/output/spm_sphere.nc', 
        '../../test/output/lake.nc', 
        'Lake data from McGinnis et al. (2006) in ./test/output/lake.nc')
        
    # Demonstrate how to read the data back in from the hard drive
    spm.load_sim('../../test/output/spm_sphere.nc')
    spm.plot_state_space(1)
    
    # Plot the full suite of model variables
    spm.plot_all_variables(1)


