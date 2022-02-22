""" 
Create a profile object from a `numpy.ndarray` of data
======================================================

Use the TAMOC ambient module to create profiles in netCDF format for use by
TAMOC from idealized laboratory data. This file demonstrates working with the
data input directly by the user as a `numpy.ndarray`.

This script demonstrates the new version of the `ambient.Profile` object, which uses `xarray`.  For the older version, which used netCDF datasets, see the script with the same file name but prepended by 'nc'.  

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

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from tamoc import ambient
from tamoc import seawater

import xarray as xr

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Create the synthetic temperature and salinity profiles from idealized
    # laboratory conditions
    z = np.array([0.0, 2.4])
    T = np.array([21.0, 20.0]) + 273.15
    S = np.array([0.0, 30.0])
    data = xr.Dataset()
    data.coords['z'] = z
    data['temperature'] = (('z'), T)
    data['salinity'] = (('z'), S)
    
    # Create an ambient.Profile object for this dataset
    lab = ambient.Profile(data)
        
    # Plot the density profile using the interpolation function
    z = np.linspace(lab.z_min, 
                    lab.z_max, 250)
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


