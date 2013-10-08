"""
Create a Profile from ROMS model output netCDF files
====================================================

Use the TAMOC ambient module to create profiles in netCDF format for use by
TAMOC from data in ROMS output files.

Notes
-----
Most of the profile information (e.g., lat, lon, time, depths, temperature, 
etc.) is read automatically from the ROMS netCDF file by the package methods
and functions.  Hence, a minimal amount of auxiliary information must be 
provided to the script file to generate fully-annotated Profile objects.

Requires
--------
The ROMS netCDF files do not directly store the depth information, and the 
external package `octant.roms` is used to tranlsate the ROMS data.  In 
addition, this script reads ROMS data from a THREDDS server in the file::

    'http://barataria.tamu.edu:8080/thredds/dodsC/ROMS_Daily/08122012/' + \
    'ocean_his_08122012_24.nc'

Returns
-------
This script generates a `ambient.Profile` object, whose netCDF file is written
to the file::

    ./Profiles/Profiles/roms.nc

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
    
    # Get the path to the input dataset and a path to the local TAMOC dataset
    # to create
    dat_file = 'http://barataria.tamu.edu:8080/thredds/dodsC/' + \
              'ROMS_Daily/08122012/ocean_his_08122012_24.nc'
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    '../../test/output'))
    nc_file = os.path.join(__location__,'roms.nc')
    
    # Select the indices to the point in the ROMS output where we want to 
    # extract the profile
    t_idx = 0
    j_idx = 400
    i_idx = 420
    
    # Open the ROMS output file and create the TAMOC input file
    (nc, dat_file) = ambient.get_nc_db_from_roms(dat_file, nc_file, t_idx, 
        j_idx, i_idx, ['dye_01', 'dye_02'])
    dat_file.close()
    
    # Create an ambient.Profile object for this dataset
    roms = ambient.Profile(nc, chem_names=['dye_01', 'dye_02'])
    
    # Close the netCDF dataset
    roms.nc.close()
    
    # Since the netCDF file is now fully stored on the hard drive in the 
    # correct format, we can initialize an ambient.Profile object directly
    # from the netCDF file
    roms = ambient.Profile(nc_file, chem_names='all')
    
    # Plot the density profile using the interpolation function
    z = np.linspace(roms.nc.variables['z'].valid_min, 
                    roms.nc.variables['z'].valid_max, 250)
    rho = np.zeros(z.shape)
    tsp = roms.get_values(z, ['temperature', 'salinity', 'pressure'])
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
    roms.nc.close()

