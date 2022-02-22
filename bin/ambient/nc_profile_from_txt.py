""" 
Create a profile from arbitrary ASCII text files
================================================

Script profile_from_txt
-----------------------

Use the TAMOC ambient module to create profiles in netCDF format for use by
TAMOC from data in text files. This file demonstrates working with data
digitized from plots of temperature and salinity in the SINTEF DeepSpill 
report

Notes
-----
Much of the input data coded in the script (e.g., columns to extract, column
names, lat and lon location data, date and time, etc.) must be known by the
user (e.g., from the SINTEF report) and is hand-coded in the script code.

Requires
--------
This script read data from the files::

    ./Profiles/Raw_Data/C.dat
    ./Profiles/Raw_Data/T.dat

Returns
-------
This script generates a `ambient.Profile` object, whose netCDF file is written
to the file::

    ./Profiles/Profiles/DS.nc

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from tamoc import ambient
from tamoc import seawater

from netCDF4 import date2num, num2date
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    
    # Get the path to the input file
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    '../../tamoc/data'))
    C_file = os.path.join(__location__,'C.dat')
    T_file = os.path.join(__location__,'T.dat')
    
    # Load in the data using numpy.loadtxt
    C_raw = np.loadtxt(C_file, comments = '%')
    T_raw = np.loadtxt(T_file, comments = '%')
    
    # Clean the profiles to remove depth reversals
    C_data = ambient.extract_profile(C_raw, 1, 25.0)
    T_data = ambient.extract_profile(T_raw, 1, 25.0)
    
    # Convert the data to standard units
    C_profile, C_units = ambient.convert_units(C_data, ['psu', 'm'])
    T_profile, T_units = ambient.convert_units(T_data, ['deg C', 'm'])
        
    # Create an empty netCDF4-classic dataset to store this CTD data
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    '../../tamoc/test/output'))
    nc_file = os.path.join(__location__,'DS.nc')
    summary = 'Dataset created by profile_from_txt in the ./bin directory' \
              + ' of TAMOC'
    source = 'Digitized data from the average CTD profile in the SINTEF ' + \
             'DeepSpill Report'
    sea_name = 'Norwegian Sea'
    p_lat = 64.99066
    p_lon = 4.84725 
    p_time = date2num(datetime(2000, 6, 27, 12, 0, 0), 
                      units = 'seconds since 1970-01-01 00:00:00 0:00', 
                      calendar = 'julian')
    nc = ambient.create_nc_db(nc_file, summary, source, sea_name, p_lat, 
                              p_lon, p_time)
    
    # Insert the CTD data into the netCDF dataset
    comments = ['digitized', 'digitized']
    nc = ambient.fill_nc_db(nc, C_profile, ['salinity', 'z'], C_units, 
                            comments, 1)
    nc = ambient.fill_nc_db(nc, T_profile, ['temperature', 'z'], T_units, 
                            comments, 1)
    
    # Calculate and insert the pressure data
    nc.set_auto_mask(False)
    z = nc.variables['z'][:]
    T = nc.variables['temperature'][:]
    S = nc.variables['salinity'][:]
    P = ambient.compute_pressure(z, T, S, 0)
    P_data = np.vstack((z, P)).transpose()
    nc = ambient.fill_nc_db(nc, P_data, ['z', 'pressure'], ['m', 'Pa'], 
                            ['measured', 'computed'], 0)
    
    # Create an ambient.Profile object for this dataset
    ds = ambient.Profile(nc)
    
    # Close the netCDF dataset
    ds.nc.close()
    
    # Since the netCDF file is now fully stored on the hard drive in the 
    # correct format, we can initialize an ambient.Profile object directly
    # from the netCDF file
    ds = ambient.Profile(nc_file, chem_names='all')
    
    # Plot the density profile using the interpolation function
    z = np.linspace(ds.z_min, 
                    ds.z_max, 250)
    rho = np.zeros(z.shape)
    tsp = ds.get_values(z, ['temperature', 'salinity', 'pressure'])
    for i in range(len(z)):
        rho[i] = seawater.density(tsp[i,0], tsp[i,1], tsp[i,2])
    
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.plot(rho, z)
    ax1.set_xlabel('Density (kg/m^3)')
    ax1.set_ylabel('Depth (m)')
    ax1.invert_yaxis()
    ax1.set_title('Computed data')
    plt.show()
    
    # Close the netCDF dataset
    ds.close_nc()

