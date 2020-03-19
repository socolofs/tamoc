"""
Insoluble fluid particles
=========================

Use the ``TAMOC`` ``DBM`` to specify an oil droplet that cannot dissolve 
(e.g., a dead, heavy oil with negligible dissolution) and calculate all of its 
properties in deepwater conditions.

In particular, this script demonstrates the methods:

* `dbm.InsolubleParticle.density`
* `dbm.InsolubleParticle.mass_by_diameter`
* `dbm.InsolubleParticle.diameter`
* `dbm.InsolubleParticle.particle_shape`
* `dbm.InsolubleParticle.slip_velocity`
* `dbm.InsolubleParticle.surface_area`
* `dbm.InsolubleParticle.heat_transfer`

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from tamoc import dbm
from tamoc import seawater

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Define the type of inert fluid particle
    isfluid = True
    iscompressible = False
    gamma=29.       # deg API
    beta=0.0007     # Pa^(-1)
    co=3.7e-9   # K^(-1)
    
    # Create a DBM InsolubleParticle object for this simple oil
    oil = dbm.InsolubleParticle(isfluid, iscompressible, gamma=gamma, 
                                beta=beta, co=co)
    
    # Specify some generic deepwater ocean conditions
    T = 273.15 + 4.
    P = 150.0 * 1.0e5
    Ta = 273.15 + 4.0
    Sa = 34.5
    L = 1500;
    
    # Compute the rise velocity for several droplet sizes
    de = np.logspace(np.log10(0.00010), np.log10(0.05))
    t = np.zeros(len(de))
    for i in range(len(t)):
        m = oil.mass_by_diameter(de[i], T, P, Sa, Ta)
        us = oil.slip_velocity(m, T, P, Sa, Ta)
        t[i] = L / us
    
    plt.figure(1)
    plt.clf()
    plt.show()
    
    ax1 = plt.subplot(111)
    ax1.loglog(de, t/60/60, '.-')
    ax1.set_xlabel('Diameter (m)')
    ax1.set_ylabel('Rise Time (hrs)')
    
    plt.draw()
    
    
    
    

