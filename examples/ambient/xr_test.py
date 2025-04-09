
"""
Scratch pad looking at ways to replace the ambient.py module with 
xarray.DataArray objects.  Following along at::

https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html

"""
#from tamoc import seawater

import numpy as np
import pandas as pd
import xarray as xr

# Create some random profile data


z = np.array([0., 10., 35., 37.5, 45., 100.])
T = np.array([23.5, 22.3, 20.4, 15.3, 12.7, 10.2]) + 273.15
S = np.array([32.4, 32.5, 33.4, 34.5, 34.5, 34.5])
#rho = seawater.density(T[0], S[0], 101325.)
rho = 1025.5
P = 101325. + 9.81 * rho * z

data = xr.Dataset(
    {
        'temperature' : (['z'], T),
        'salinity' : (['z'], S),
        'pressure' : (['z'], P)
    },
    coords = {
        'z' : (['z'], z),
    },
)

