""" 
Create a profile from arbitrary ASCII text files
================================================

Script profile_from_txt
-----------------------

Use the TAMOC ambient module to create profiles for use by
TAMOC from data in text files. This file demonstrates working with data
digitized from plots of temperature and salinity in the SINTEF DeepSpill 
report.

This script demonstrates the new version of the `ambient.Profile` object, which uses `xarray`.  For the older version, which used netCDF datasets, see the script with the same file name but prepended by 'nc'.  

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

from netCDF4 import date2num
from datetime import datetime

import xarray as xr

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
    
    # These datasets have different z-coordinates.  Choose one as the base
    # dataset
    ds = xr.Dataset()
    ds.coords['z'] = C_data[:,1]
    ds['salinity'] = (('z'), C_data[:,0])
    ztsp = ['z', 'salinity']
    ztsp_units = ['m', 'psu']
    
    # Create a Profile object with these data.  Note: this is not a complete
    # profile as the minimum requirements are to provide temperature and
    # salinity.  We will use this profile to merge the two dataset, then
    # we will re-build the profile.
    profile1 = ambient.Profile(ds, ztsp, ztsp_units=ztsp_units, err=0.)
    
    # Insert the temperature data
    profile1.append(T_data, ['temperature', 'z'], ['deg C', 'm'], z_col=1)
    
    # Get the dataset out of this profile object
    ds = profile1.ds
    
    # Add some options attributes to the dataset
    ds.attrs['summary'] = 'Dataset created by profile_from_txt in the ./bin'\
        ' directory of TAMOC'
    ds.attrs['source'] = 'Digitized data from the average CTD profile in'\
        ' the SINTEF DeepSpill Report'
    ds.attrs['sea_name'] = 'Norwegian Sea'
    ds.coords['lat'] = 64.99066
    ds.coords['lon'] = 4.84725
    ds.coords['time'] = date2num(datetime(2000, 6, 27, 12, 0, 0), 
        units = 'seconds since 1970-01-01 00:00:00 0:00', 
        calendar = 'julian')
    
    # Create a "complete" profile with these merged data...Note: this Profile
    # initializer will fill in the pressure data
    ztsp = ['z', 'temperature', 'salinity', 'pressure']
    ztsp_units = ['m']
    ztsp_units += [ds['temperature'].attrs['units']]
    ztsp_units += [ds['salinity'].attrs['units']]
    ztsp_units += ['Pa']
    profile = ambient.Profile(ds, ztsp, ztsp_units=ztsp_units, err=0.)
    
    # Plot the density profile using the interpolation function
    z = np.linspace(profile.z_min, 
                    profile.z_max, 250)
    rho = np.zeros(z.shape)
    tsp = profile.get_values(z, ['temperature', 'salinity', 'pressure'])
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
    
