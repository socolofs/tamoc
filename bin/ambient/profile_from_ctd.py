"""
Create a Profile from an ASCII CTD datafile
===========================================

Use the TAMOC ambient module to create profiles in netCDF format for use by 
TAMOC from data in text files downloaded from a CTD.  This file demonstrates
working with the data from the R/V Brooks McCall at Station BM 54 on May 30,
2010, stored in the file /Raw_Data/ctd_BM54.cnv.

Notes
-----
Much of the input data in the script (e.g., columns to extract, column names,
lat and lon location data, date and time, etc.) is read by the user manually
from the header file of the CTD text file. These data are then hand-coded in
the script text. While it would be straightforward to automate this process
for a given format of CTD files, this step is left to the user to customize to
their own data sets.

Requires
--------
This script read data from the text file::

    ./Profiles/Raw_Data/ctd_BM54.dat

Returns
-------
This script generates a `ambient.Profile` object, whose netCDF file is written
to the file::

    ./Profiles/Profiles/BM54.nc

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

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
    dat_file = os.path.join(__location__,'ctd_BM54.cnv')
    
    # Load in the data using numpy.loadtxt
    raw = np.loadtxt(dat_file, comments = '#', skiprows = 175, 
                     usecols = (0, 1, 3, 8, 9, 10, 12))
    
    # Describe the organization of the data in raw.  
    var_names = ['temperature', 'pressure', 'wetlab_fluorescence', 'z', 
                 'salinity', 'density', 'oxygen']
    var_units = ['deg C', 'db', 'mg/m^3', 'm', 'psu', 'kg/m^3', 'mg/l']
    z_col = 3
    
    # Clean the profile to remove reversals in the depth coordinate
    data = ambient.extract_profile(raw, z_col, 50.0)
    
    # Convert the profile data to standard units in TAMOC
    profile, units = ambient.convert_units(data, var_units)
    
    # Create an empty netCDF4-classic dataset to store this CTD data
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    '../../test/output'))
    nc_file = os.path.join(__location__,'BM54.nc')
    summary = 'Dataset created by profile_from_ctd in the ./bin directory' \
              + ' of TAMOC'
    source = 'R/V Brooks McCall, station BM54'
    sea_name = 'Gulf of Mexico'
    p_lat = 28.0 + 43.945 / 60.0
    p_lon = 360 - (88.0 + 22.607 / 60.0) 
    p_time = date2num(datetime(2010, 5, 30, 18, 22, 12), 
                      units = 'seconds since 1970-01-01 00:00:00 0:00', 
                      calendar = 'julian')
    nc = ambient.create_nc_db(nc_file, summary, source, sea_name, p_lat, 
                              p_lon, p_time)
    
    # Insert the CTD data into the netCDF dataset
    comments = ['measured'] * len(var_names)
    nc = ambient.fill_nc_db(nc, profile, var_names, units, comments, z_col)
    
    # Create an ambient.Profile object for this dataset
    bm54 = ambient.Profile(nc, chem_names=['oxygen'])
    
    # Close the netCDF dataset
    bm54.nc.close()
    
    # Since the netCDF file is now fully stored on the hard drive in the 
    # correct format, we can initialize an ambient.Profile object directly
    # from the netCDF file
    bm54 = ambient.Profile(nc_file, chem_names='all')
    
    # Plot the density profile using the interpolation function
    z = np.linspace(bm54.nc.variables['z'].valid_min, 
                    bm54.nc.variables['z'].valid_max, 250)
    rho = np.zeros(z.shape)
    tsp = bm54.get_values(z, ['temperature', 'salinity', 'pressure'])
    for i in range(len(z)):
        rho[i] = seawater.density(tsp[i,0], tsp[i,1], tsp[i,2])
    
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.plot(rho, z)
    ax1.set_xlabel('Density (kg/m^3)')
    ax1.set_ylabel('Depth (m)')
    ax1.invert_yaxis()
    ax1.set_title('Computed data')
    
    # Compare to the measured profile
    z_m = bm54.nc.variables['z'][:]
    rho_m = bm54.nc.variables['density'][:]
    ax2 = plt.subplot(1,2,2)
    ax2.plot(rho_m, z_m)
    ax2.set_xlabel('Density (kg/m^3)')
    ax2.invert_yaxis()
    ax2.set_title('Measured data')
    
    plt.show()
    
    # Close the netCDF dataset
    bm54.nc.close()

