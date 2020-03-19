"""
Bent Plume Model:  Blowout simulation
=====================================

Use the ``TAMOC`` `bent_plume_model` to simulate a subsea accidental
oil spill plume. This script demonstrates the typical steps involved in
running the bent bubble model with petroleum fluids in the ocean.

This simulation uses the ambient data stored in the file
`./test/output/test_BM54.nc`. This dataset is created by the test files in the
`./test` directory. Please be sure that all of the tests pass using ``py.test
-v`` at the command prompt before trying to run this simulation.

"""
# S. Socolofsky, December 2014, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient
from tamoc import dbm
from tamoc import dispersed_phases
from tamoc import bent_plume_model

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
    
    # Insert a constant crossflow velocity
    z = ctd.nc.variables['z'][:]
    ua = np.zeros(z.shape) + 0.09
    data = np.vstack((z, ua)).transpose()
    symbols = ['z', 'ua']
    units = ['m', 'm/s']
    comments = ['measured', 'arbitrary crossflow velocity']
    ctd.append(data, symbols, units, comments, 0)
    
    # Jet initial conditions
    z0 = 1000.
    U0 = 0.
    phi_0 = -np.pi / 2.
    theta_0 = 0.
    D = 0.3
    Tj = 273.15 + 35.
    Sj = 0.
    cj = 1.
    chem_name = 'tracer'
    
    # Create the stratified plume model object
    bpm = bent_plume_model.Model(ctd)
        
    # Create the gas phase particles
    composition = ['methane', 'ethane', 'propane', 'oxygen']
    yk = np.array([0.93, 0.05, 0.02, 0.0])
    gas = dbm.FluidParticle(composition)
    disp_phases = []
    
    # Larger free gas bubbles
    mb0 = 5.         # total mass flux in kg/s
    de = 0.005       # bubble diameter in m
    lambda_1 = 0.85
    (m0, T0, nb0, P, Sa, Ta) = dispersed_phases.initial_conditions(
        ctd, z0, gas, yk, mb0, 2, de, Tj)
    disp_phases.append(bent_plume_model.Particle(0., 0., z0, gas, m0, T0, 
        nb0, lambda_1, P, Sa, Ta, K=1., K_T=1., fdis=1.e-6, t_hyd=0.,
        lag_time=False))
    
    # Smaller free gas bubbles
    mb0 = 5.         # total mass flux in kg/s
    de = 0.0005       # bubble diameter in m
    lambda_1 = 0.95
    (m0, T0, nb0, P, Sa, Ta) = dispersed_phases.initial_conditions(
        ctd, z0, gas, yk, mb0, 2, de, Tj)
    disp_phases.append(bent_plume_model.Particle(0., 0., z0, gas, m0, T0, 
        nb0, lambda_1, P, Sa, Ta, K=1., K_T=1., fdis=1.e-6, t_hyd=0., 
        lag_time=False))
    
    # Larger oil droplets
    oil = dbm.InsolubleParticle(True, True, rho_p=890., gamma=30., 
                                beta=0.0007, co=2.90075e-9, 
                                k_bio=3.000e-6, 
                                t_bio=86400.)
    mb0 = 10.         # total mass flux in kg/s
    de = 0.005       # bubble diameter in m
    lambda_1 = 0.9
    (m0, T0, nb0, P, Sa, Ta) = dispersed_phases.initial_conditions(
        ctd, z0, oil, yk, mb0, 2, de, Tj)
    disp_phases.append(bent_plume_model.Particle(0., 0., z0, oil, m0, T0, 
        nb0, lambda_1, P, Sa, Ta, K=1., K_T=1., fdis=1.e-6, t_hyd=0.,
        lag_time=False))
    
    # Run the simulation
    bpm.simulate(np.array([0., 0., z0]), D, U0, phi_0, theta_0,
        Sj, Tj, cj, chem_name, disp_phases, track=True, dt_max=60.,
        sd_max = 2000.)
    
    # Plot the full suite of model variables
    bpm.plot_all_variables(1)

