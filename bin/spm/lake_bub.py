"""
Stratified Plume Model:  Lake simulation
========================================

Use the ``TAMOC`` `stratified_plume_model` to simulate a lake aeration plume. 
This script demonstrates the typical steps involved in running the single 
bubble model with reactive (dissolving) particles.

This simulation uses the ambient data stored in the file
`./tamoc/data/lake.dat`. This module first organizes this data and stores the
necessary netCDF file in `../tamoc/test/output`. Please make sure this
directory exists before running this file.

"""
# S. Socolofsky, August 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient
from tamoc import dbm
from tamoc import stratified_plume_model

from datetime import datetime
from netCDF4 import date2num

import numpy as np

def get_lake_data():
    """
    Create the netCDF dataset of CTD data for a lake simulation
    
    Creates the ambient.Profile object and netCDF dataset of CTD data for 
    a lake simualtion from the `./data/lake.dat` text file, digitized from
    the data in McGinnis et al. (2002) for Lake Hallwil.
    
    """
    
    # Read in the lake CTD data
    fname = '../../tamoc/data/lake.dat'
    raw = np.loadtxt(fname, skiprows=9)
    variables = ['z', 'temperature', 'salinity', 'oxygen']
    units = ['m', 'deg C', 'psu', 'kg/m^3']
    
    # Convert the units to mks
    profile, units = ambient.convert_units(raw, units)
    
    # Calculate the pressure data
    P = ambient.compute_pressure(profile[:,0], profile[:,1], profile[:,2], 0)
    profile = np.hstack((profile, np.atleast_2d(P).transpose()))
    variables = variables + ['pressure']
    units = units + ['Pa']
    
    # Set up a netCDF dataset object
    summary = 'Default lake.dat dataset provided by TAMOC'
    source = 'Lake CTD data digitized from figures in McGinnis et al. 2004'
    sea_name = 'Lake Hallwil, Switzerland'
    lat = 47.277166666666666
    lon = 8.217294444444445
    date = datetime(2002, 7, 18)
    t_units = 'seconds since 1970-01-01 00:00:00 0:00'
    calendar = 'julian'
    time = date2num(date, units=t_units, calendar=calendar)
    nc = ambient.create_nc_db('../../tamoc/test/output/lake.nc', summary,
        source, sea_name, lat, lon, time)
    
    # Insert the measured data
    comments = ['digitized from measured data'] * 4
    comments = comments + ['computed from z, T, S']
    nc = ambient.fill_nc_db(nc, profile, variables, units, comments, 0)
    
    # Insert an additional column with data for nitrogen and argon equal to 
    # their saturation concentrations at the free surface.
    composition = ['nitrogen', 'oxygen', 'argon']
    yk = np.array([0.78084, 0.209476, 0.009684])
    air = dbm.FluidMixture(composition)
    m = air.masses(yk)
    Cs = air.solubility(m, profile[0,1], 101325., profile[0,2])
    N2 = np.zeros((profile.shape[0], 1))
    Ar = np.zeros((profile.shape[0], 1))
    N2 = N2 + Cs[0,0]
    Ar = Ar + Cs[0,2]
    z = np.atleast_2d(profile[:,0]).transpose()
    comments = ['calculated potential saturation value']*3
    nc = ambient.fill_nc_db(nc, np.hstack((z, N2, Ar)), 
                           ['z', 'nitrogen', 'argon'], 
                           ['m', 'kg/m^3', 'kg/m^3'], 
                           comments, 0)
    
    # Create an ambient.Profile object
    lake = ambient.Profile(nc, chem_names=['oxygen', 'nitrogen', 'argon'])
    lake.close_nc()
    
    # Return the Profile object
    return lake

if __name__ == '__main__':

    # Get the ambient CTD profile data
    nc = '../../test/output/lake.nc'
    try:
        # Open the lake dataset as a Profile object if it exists
        lake = ambient.Profile(nc, chem_names=['oxygen', 'nitrogen', 'argon'])
        
    except:
        # Create the lake netCDF dataset and get the Profile object
        lake = get_lake_data()
    
    # Create the stratified plume model object
    spm = stratified_plume_model.Model(lake)
    
    # Create the dispersed phase particles
    composition = ['oxygen', 'nitrogen', 'argon']
    yk = np.array([1.0, 0., 0.])
    o2 = dbm.FluidParticle(composition, isair=True)
    z0 = 46.
    bubbles = []
    
    # Small bubble
    Q_N = 30. / 60. / 60. 
    de = 0.001
    lambda_1 = 0.9
    bubbles.append(stratified_plume_model.particle_from_Q(lake, z0, o2, yk, 
                   Q_N,  de, lambda_1, t_hyd=0.))
    
    # Medium bubble
    Q_N = 70. / 60./ 60.
    de = 0.002
    lambda_1 = 0.8
    bubbles.append(stratified_plume_model.particle_from_Q(lake, z0, o2, yk, 
                  Q_N,  de, lambda_1, t_hyd=0.))
    
    # Initialize a simulation
    R = 6.5 / 2.
    spm.simulate(bubbles, z0, R, maxit=50, delta_z = 0.2)
    
    # Save the model results
    spm.save_sim('../../tamoc/test/output/spm_gas.nc',
        '../../test/output/lake.nc', 
        'Lake data from McGinnis et al. (2006) in ./test/output/lake.nc')
    
    # Demonstrate how to read the data back in from the hard drive
    spm.load_sim('../../tamoc/test/output/spm_gas.nc')
    spm.plot_state_space(1)
    
    # Plot the full suite of model variables
    spm.plot_all_variables(1)


