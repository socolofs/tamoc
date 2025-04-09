"""
Append data to an existing CTD Profile dataset
==============================================

Use the TAMOC ambient module to append data to a CTD Profile object that has
already been created as in the other examples in this ./bin director. This
file demonstrates working with the data from the R/V Brooks McCall at Station
BM 54 on May 30, 2010, stored in the file /Raw_Data/ctd_BM54.cnv.

In this example, we compute a typical nitrogen profile and append that data 
to the data in the CTD dataset.

This script demonstrates the new version of the `ambient.Profile` object, which uses `xarray`.  For the older version, which used netCDF datasets, see the script with the same file name but prepended by 'nc_'.  

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
This script reads data from the text file::

    ../../tamoc/data/ctd_BM54.cnv

Returns
-------
This script generates an `ambient.Profile` object, whose netCDF file is 
written to the file::

    ../../test/output/BM54.nc

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from tamoc import ambient
from tamoc import seawater
from tamoc import dbm

from netCDF4 import date2num, num2date
from datetime import datetime

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
    """
    Demonstrate how to add data to an existing Profile object
    
    """
    # Get the ambient.Profile object with the original CTD data
    profile = get_ctd_profile()
    
    # Compute a dissolved nitrogen profile...start with a model for air
    air = dbm.FluidMixture(['nitrogen', 'oxygen', 'argon', 'carbon_dioxide'])
    yk = np.array([0.78084, 0.20946, 0.009340, 0.00036])
    m = air.masses(yk)
    
    # Compute the solubility of nitrogen at the air-water interface, then 
    # correct for seawater compressibility
    z_coords = profile.interp_ds.coords['z'].values
    n2_conc = np.zeros(len(z_coords))
    for i in range(len(z_coords)):
        T, S, P = profile.get_values(z_coords[i], ['temperature', 'salinity', 
                  'pressure'])
        Cs = air.solubility(m, T, 101325., S)[0,:] * \
             seawater.density(T, S, P) / seawater.density(T, S, 101325.)
        n2_conc[i] = Cs[0]
    
    # Add this computed nitrogen profile to the Profile dataset
    data = np.vstack((z_coords, n2_conc)).transpose()
    symbols = ['z', 'nitrogen']
    units = ['m', 'kg/m^3']
    comments = ['measured', 'computed from CTD data']
    profile.append(data, symbols, units, comments, 0)
    
    # Close the dataset
    profile.close_nc()
    
    # Plot the oxygen and nitrogren profiles to show that data have been 
    # added to the Profile object
    z = np.linspace(profile.z_min, profile.z_max, 250)
    o2 = np.zeros(z.shape)
    n2 = np.zeros(z.shape)
    for i in range(len(z)):
        n2[i], o2[i] = profile.get_values(z[i], ['nitrogen', 'oxygen'])
    
    plt.figure()
    plt.clf()
    plt.show()
    
    ax1 = plt.subplot(121)
    ax1.plot(o2, z)
    ax1.set_xlabel('Oxygen (kg/m^3)')
    ax1.set_ylabel('Depth (m)')
    ax1.invert_yaxis()
    ax1.set_title('Measured data')
    
    ax2 = plt.subplot(122)
    ax2.plot(n2, z)
    ax2.set_xlabel('Nitrogen (kg/m^3)')
    ax2.set_ylabel('Depth (m)')
    ax2.invert_yaxis()
    ax2.set_title('Computed data')
    
    plt.draw()


    