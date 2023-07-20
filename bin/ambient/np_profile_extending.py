"""
Extend a profile's CTD data to deeper depths
============================================

Use the TAMOC ambient module to open a Profile object, compute some buoyancy
frequencies and then artificially extend the profile to deeper depths while
maintaining a fixed buoyancy frequency.

This script demonstrates the new version of the `ambient.Profile` object, which uses `xarray`.  For the older version, which used netCDF datasets, see the script with the same file name but prepended by 'nc_'.  

Notes
-----
There are any number of ways that CTD data could be artificially extended to
deeper than the measured depths.  This script demonstrates one rational 
method coded in the `ambient.Profile` class and documented in the class 
method `ambient.Profile.extend_profile_deeper()`.

Requires
--------
This script reads data from a netCDF object already in the format of a 
TAMOC `ambient.Profile` object, stored in the file::

    ./Profiles/Profiles/BM54.nc

If this file is not yet present in your directory structure, it can be 
generated by the `profile_from_ctd` script.  To execute that file, change
directory at the command promt to the `./Profiles` root directory and at the 
IPython prompt execute::

    >>> run profile_from_ctd

Returns
-------
This script generates a new `ambient.Profile` object, whose netCDF file is
written to the file::

    ./Profiles/Profiles/BM54_deeper.nc

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from tamoc import ambient #old_ambient as ambient
from tamoc import seawater

from netCDF4 import date2num, num2date
from datetime import datetime
from time import ctime

import xarray as xr

import numpy as np
import matplotlib.pyplot as plt
import os

    
def get_ctd_profile():
    """
    Load CTD Data into an 'ambient.Profile' object.
    
    This function performs the steps in ./profile_from_ctd.py to read in the
    CTD data and create a Profile object.  This is the data set that will be
    used to demonstrate how to append data to a Profiile object.
    
    """
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
    
    # Create an ambient.Profile object for this dataset
    chem_names = ['oxygen']
    bm54 = ambient.Profile(data, ztsp=ztsp, ztsp_units=ztsp_units, 
        chem_names=chem_names, chem_units=chem_units)
    
    return bm54

if __name__ == '__main__':
        
    # Get the ambient.Profile object with the original CTD data
    ctd = get_ctd_profile()
    
    # Print the buoyancy frequency at a few selected depths
    z = np.array([500., 1000., 1500.])
    N = ctd.buoyancy_frequency(z)
    print('Buoyancy frequency is: ')
    for i in range(len(z)):
        print('    N(%d m) = %g (1/s) ' % (z[i], N[i]))
    
    # Plot the potential density profile and corresponding buoyancy frequency
    z_min = ctd.z_min
    z_max = ctd.z_max
    z = np.linspace(z_min, z_max, 500)
    ts = ctd.get_values(z, ['temperature', 'salinity'])
    rho = seawater.density(ts[:,0], ts[:,1], 101325.)
    N = ctd.buoyancy_frequency(z)
    fig = plt.figure(3)
    plt.clf()
    ax1 = plt.subplot(121)
    ax1.plot(rho, z)
    ax1.set_xlabel('Potential density, (kg/m^3)')
    ax1.set_ylabel('Depth, (m)')
    ax1.set_ylim([0., 2500.])
    ax1.invert_yaxis()
    ax2 = plt.subplot(122)
    ax2.plot(N, z)
    ax2.set_xlabel('N, (1/s)')
    ax2.set_ylim([0., 2500.])
    ax2.invert_yaxis()
    plt.show()
    
    # Get the path to the output file
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    '../../tamoc/test/output'))
    
    # Extend the CTD profile to 2500 m
    dat_file = os.path.join(__location__,'BM54_deeper.nc')
    ctd.extend_profile_deeper(2500., dat_file)
    
    # Plot the new potential density and buoyancy frequency profiles
    z_min = ctd.z_min
    z_max = ctd.z_max
    z = np.linspace(z_min, z_max, 750)
    ts = ctd.get_values(z, ['temperature', 'salinity'])
    rho = seawater.density(ts[:,0], ts[:,1], 101325.)
    N = ctd.buoyancy_frequency(z)
    fig = plt.figure(4)
    plt.clf()
    ax1 = plt.subplot(121)
    ax1.plot(rho, z)
    ax1.set_xlabel('Potential density, (kg/m^3)')
    ax1.set_ylabel('Depth, (m)')
    ax1.invert_yaxis()
    ax2 = plt.subplot(122)
    ax2.plot(N, z)
    ax2.set_xlabel('N, (1/s)')
    ax2.invert_yaxis()
    plt.show()
    
    ctd.close_nc()

    