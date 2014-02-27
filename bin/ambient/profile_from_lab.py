""" 
Create a profile object from a `numpy.ndarray` of data
======================================================

Use the TAMOC ambient module to create profiles in netCDF format for use by
TAMOC from idealized laboratory data. This file demonstrates working with the
data input directly by the user as a `numpy.ndarray`.

Notes
-----
Much of the input data in this script (e.g., columns to extract, column names,
lat and lon location data, date and time, etc.) must be known from the user
(e.g., in this case mostly fictitious) and is hand-coded in the script 
text.

Returns
-------
This script generates a `ambient.Profile` object, whose netCDF file is written
to the file::

    ./Profiles/Profiles/Lab.nc

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
    
    # Create the synthetic temperature and salinity profiles from idealized
    # laboratory conditions
    z = np.array([0.0, 2.4])
    T = np.array([21.0, 20.0]) + 273.15
    S = np.array([0.0, 30.0])
    profile = np.vstack((z, T, S)).transpose()
    
    # Create an empty netCDF4-classic dataset to store this CTD data
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    '../../test/output'))
    nc_file = os.path.join(__location__,'Lab.nc')
    summary = 'Dataset created by profile_from_txt in the ./bin directory' \
              + ' of TAMOC'
    source = 'Synthetic data for idealized laboratory conditions'
    sea_name = 'None'
    p_lat = -999
    p_lon = -999 
    p_time = date2num(datetime(2013, 7, 15, 20, 24, 0), 
                      units = 'seconds since 1970-01-01 00:00:00 0:00', 
                      calendar = 'julian')
    nc = ambient.create_nc_db(nc_file, summary, source, sea_name, p_lat, 
                              p_lon, p_time)
    
    # Insert the CTD data into the netCDF dataset
    comments = ['synthetic'] * 3
    nc = ambient.fill_nc_db(nc, profile, ['z', 'temperature', 'salinity'], 
                            ['m', 'K', 'psu'], comments, 0)
    
    # Calculate and insert the pressure data
    z = nc.variables['z'][:]
    T = nc.variables['temperature'][:]
    S = nc.variables['salinity'][:]
    P = ambient.compute_pressure(z, T, S, 0)
    P_data = np.vstack((z, P)).transpose()
    nc = ambient.fill_nc_db(nc, P_data, ['z', 'pressure'], ['m', 'Pa'], 
                            ['measured', 'computed'], 0)
    
    # Create an ambient.Profile object for this dataset
    lab = ambient.Profile(nc)
    
    # Close the netCDF dataset
    lab.nc.close()
    
    # Since the netCDF file is now fully stored on the hard drive in the 
    # correct format, we can initialize an ambient.Profile object directly
    # from the netCDF file
    lab = ambient.Profile(nc_file)
    
    # Plot the density profile using the interpolation function
    z = np.linspace(lab.nc.variables['z'].valid_min, 
                    lab.nc.variables['z'].valid_max, 250)
    rho = np.zeros(z.shape)
    tsp = lab.get_values(z, ['temperature', 'salinity', 'pressure'])
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
    lab.nc.close()

