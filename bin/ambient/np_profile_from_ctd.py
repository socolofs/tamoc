"""
Create a profile from an ASCII CTD datafile
===========================================

Use the TAMOC ambient module to create profiles in netCDF format for use by 
TAMOC from data in text files downloaded from a CTD.  This file demonstrates
working with the data from the R/V Brooks McCall at Station BM 54 on May 30,
2010, stored in the file /Raw_Data/ctd_BM54.cnv.

This script demonstrates the new version of the `ambient.Profile` object, which uses `xarray`.  For the older version, which used netCDF datasets, see the script with the same file name but prepended by 'nc'.  

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

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from tamoc import ambient
from tamoc import seawater

from netCDF4 import date2num, num2date
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
    dat_file = os.path.join(__location__,'ctd_BM54.cnv')
    
    # Load in the data using numpy.loadtxt
    raw = np.loadtxt(dat_file, comments = '#', skiprows = 175, 
                     usecols = (0, 1, 3, 8, 9, 10, 12))
    
    # Remove reversals in the CTD data and get only the down-cast
    raw_data = ambient.extract_profile(raw, z_col=3, z_start=50.0)
    
    # Reorganize this data into the correct order
    data = np.zeros(raw_data.shape)
    ztsp = ['z', 'temperature', 'salinity', 'pressure']
    ztsp_units = ['m', 'deg C', 'psu', 'db']
    chem_names = ['oxygen', 'wetlab_fluorescence', 'density']
    chem_units = ['mg/l', 'mg/m^3', 'kg/m^3']
    data[:,0] = raw_data[:,3]
    data[:,1] = raw_data[:,0]
    data[:,2] = raw_data[:,4]
    data[:,3] = raw_data[:,1]
    data[:,4] = raw_data[:,6]
    data[:,5] = raw_data[:,2]
    data[:,6] = raw_data[:,5]
    
    # Create an ambient.Profile object for this dataset
    chem_names = ['oxygen', 'wetlab_fluorescence', 'density']
    bm54 = ambient.Profile(data, ztsp=ztsp, ztsp_units=ztsp_units, 
        chem_names=chem_names, chem_units=chem_units, err=0.00001)
    
    # Plot the density profile using the interpolation function
    z = np.linspace(bm54.z_min, bm54.z_max, 250)
    rho = np.zeros(z.shape)
    T = np.zeros(z.shape)
    S = np.zeros(z.shape)
    C = np.zeros(z.shape)
    O2 = np.zeros(z.shape)
    tsp = bm54.get_values(z, ['temperature', 'salinity', 'pressure'])
    for i in range(len(z)):
        rho[i] = seawater.density(tsp[i,0], tsp[i,1], tsp[i,2])
        T[i], S[i], C[i], O2[i] = bm54.get_values(z[i], ['temperature', 
            'salinity', 'wetlab_fluorescence', 'oxygen'])
    
    # Extract data for comparison
    z_m = bm54.ds.coords['z'].values
    rho_m = bm54.ds['density'].values
    
    plt.figure(1)
    plt.clf()
    plt.show()
    
    ax1 = plt.subplot(121)
    ax1.plot(rho, z)
    ax1.set_xlabel('Density (kg/m^3)')
    ax1.set_ylabel('Depth (m)')
    ax1.invert_yaxis()
    ax1.set_title('Computed data')
    
    # Compare to the measured profile
    ax2 = plt.subplot(1,2,2)
    ax2.plot(rho_m, z_m)
    ax2.set_xlabel('Density (kg/m^3)')
    ax2.invert_yaxis()
    ax2.set_title('Measured data')
    
    plt.draw()
    
    plt.figure(2)
    plt.clf()
    plt.show()
    
    ax1 = plt.subplot(131)
    ax1.plot(C*1.e6, z, '-', label='Fluorescence (g/m^3)')
    ax1.set_xlabel('CTD component values')
    ax1.set_ylabel('Depth (m)')
    ax1.set_ylim([800, 1500])
    ax1.set_xlim([0, 40])
    ax1.invert_yaxis()        
    ax1.locator_params(tight=True, nbins=6)
    ax1.legend(loc='upper right', prop={'size':10})
    ax1.grid(True)
    
    ax2 = plt.subplot(132)
    ax2.plot(T - 273.15, z, '-', label='Temperature (deg C)')
    ax2.plot(O2*1.e3, z, '--', label='Oxygen (g/m^3)')
    ax2.set_xlabel('CTD component values')
    ax2.set_ylabel('Depth (m)')
    ax2.set_ylim([800, 1500])
    ax2.set_xlim([0, 8])
    ax2.invert_yaxis()        
    ax2.locator_params(tight=True, nbins=6)
    ax2.legend(loc='upper right', prop={'size':10})
    ax2.grid(True)
    
    ax3 = plt.subplot(133)
    ax3.plot(S, z, '-', label='Salinity (psu)')
    ax3.set_xlabel('CTD component values')
    ax3.set_ylabel('Depth (m)')
    ax3.set_ylim([800, 1500])
    ax3.set_xlim([34.5, 35])
    ax3.invert_yaxis()        
    ax3.locator_params(tight=True, nbins=6)
    ax3.legend(loc='upper right', prop={'size':10})
    ax3.grid(True)
    
    plt.draw()
    
    # Close the netCDF dataset
    bm54.close_nc()

