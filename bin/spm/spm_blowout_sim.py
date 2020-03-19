"""
Stratified Plume Model:  Blowout simulation
===========================================

Use the ``TAMOC`` `stratified_plume_model` to simulate a subsea accidental 
oil spill plume. This script demonstrates the typical steps involved in 
running the single bubble model with petroleum fluids in the ocean.

This simulation uses the ambient data stored in the file
`./test/output/test_BM54.nc`. This dataset is created by the test files in the
`./test` directory. Please be sure that all of the tests pass using ``py.test
-v`` at the command prompt before trying to run this simulation.

"""
# S. Socolofsky, August 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient
from tamoc import dbm
from tamoc import stratified_plume_model

from datetime import datetime
from netCDF4 import date2num

import numpy as np

if __name__ == '__main__':

    # Get the ambient CTD profile data
    nc = '../../test/output/test_BM54.nc'
    try:
        # Open the lake dataset as a Profile object if it exists
        ctd = ambient.Profile(nc, chem_names='all')
        
    except RuntimeError:
        # Tell the user to create the dataset
        print('CTD data not available; run test cases in ./test first.')
    
    # Create the stratified plume model object
    spm = stratified_plume_model.Model(ctd)
    
    # Set the release conditions
    T0 = 273.15 + 35.   # Release temperature in K
    R = 0.15            # Radius of leak source in m
    
    # Create the gas phase particles
    composition = ['methane', 'ethane', 'propane', 'oxygen']
    yk = np.array([0.93, 0.05, 0.02, 0.0])
    gas = dbm.FluidParticle(composition)
    z0 = 1000.
    disp_phases = []
    
    # Larger free gas bubbles
    mb0 = 8.         # total mass flux in kg/s
    de = 0.025       # bubble diameter in m
    lambda_1 = 0.85
    disp_phases.append(stratified_plume_model.particle_from_mb0(ctd, z0, gas, 
                       yk, mb0, de, lambda_1, T0))
    
    # Smaller free gas bubbles (note, it is not necessary to have more than
    # one bubble size)
    mb0 = 2.         # total mass flux in kg/s
    de = 0.0075      # bubble diameter in m
    lambda_1 = 0.9
    disp_phases.append(stratified_plume_model.particle_from_mb0(ctd, z0, gas, 
                       yk, mb0, de, lambda_1, T0))
    
    # Liquid hydrocarbon.  This could either be a dissolving phase (mixture
    # of liquid phases) or an inert phase.  We demonstrate here the simple
    # case of an inert oil phase
    oil = dbm.InsolubleParticle(True, True, rho_p=890., gamma=30., 
                                beta=0.0007, co=2.90075e-9)
    mb0 = 10.        # total mass flux in kg/s
    de = 0.004       # bubble diameter in m
    lambda_1 = 0.9
    disp_phases.append(stratified_plume_model.particle_from_mb0(ctd, z0, oil,
                       np.array([1.]), mb0, de, lambda_1, T0))
    
    # Run the simulation
    spm.simulate(disp_phases, z0, R, maxit=15, toler=0.2, delta_z = 1.)
    
    # Save the model results
    spm.save_sim('../../test/output/spm_blowout.nc', 
        '../../test/output/test_BM54.nc', 
        'CTD data from Brooks McCall in file ./test/output/test_BM54.nc')
    
    # Demonstrate how to read the data back in from the hard drive
    spm.load_sim('../../test/output/spm_blowout.nc')
    spm.plot_state_space(1)
    
    # Plot the full suite of model variables
    spm.plot_all_variables(1)

