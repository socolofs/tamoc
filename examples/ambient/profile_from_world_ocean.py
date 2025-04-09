"""
Create a profile from the world-ocean average data
==================================================

The `tamoc.data` directory comes with data for the world-ocean average
profiles of temperature, salinity, and oxygen, taken from Sarmiento and
Gruber (2006). This script demonstrates how to initialize a profile using
these data.

There are two ways to use the world-ocean average data.  First, you can just import that data as is, in which case, no data are provided to the `ambient.Profile` initializer.  Or, you can specify the surface conditions, in which case, a one-dimensional array of data is provided.  Here, we demonstrate the earlier option.

"""
# S. Socolofsky, February 2022, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from tamoc import ambient
from tamoc import seawater


import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Create a profile object with the world-ocean average data
    data = None
    chem_names = ['oxygen']
    profile = ambient.Profile(data, chem_names=chem_names, err=0.)
    
    # Plot some typical profiles...
    
    # Select one parameter...need to have an open figure to plot into
    plt.figure(1)
    plt.clf()
    profile.plot_parameter('temperature')
    
    # Select several parameters
    parms = ['temperature', 'salinity', 'oxygen']
    profile.plot_profiles(parms, fig=2)
    
    # Plot all the physics variables
    profile.plot_physical_profiles(fig=3)
    
    # Plot also the chemical data
    profile.plot_chem_profiles(fig=4)
    
    plt.show()