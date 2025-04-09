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
    iscompressible = True
    gamma=29.       # deg API
    beta=0.0007     # Pa^(-1)
    co=3.7e-9   # K^(-1)
    
    # Create a DBM InsolubleParticle object for this simple oil
    oil = dbm.InsolubleParticle(isfluid, iscompressible, gamma=gamma, 
                                beta=beta, co=co)
    
    # Specify some generic deepwater ocean conditions
    P = 150.0 * 1.0e5
    Ta = 273.15 + 4.0
    Sa = 34.5
    
    # Echo the ambient conditions to the screen
    print('\nAmbient conditions: \n')
    print('   P = %g (Pa)' % P)
    print('   T = %g (K)' % Ta)
    print('   S = %g (psu)' % Sa)
    print('   rho_sw = %g (kg/m^3)' % (seawater.density(Ta, Sa, P)))
    
    # Get the general properties of the oil
    T = 273.15 + 60.
    print('\nBasic properties of liquid oil: \n')
    print('   T = %g (K)' % T)
    print('   rho_p = %g (kg/m^3) at %g (K) and %g (Pa)' % 
        (oil.density(T, P, Sa, Ta), T, P))
    
    # Get the masses in a 1.0 cm effective diameter droplet
    de = 0.01
    m = oil.mass_by_diameter(de, T, P, Sa, Ta)
    
    # Echo the properties of the droplet to the screen
    print('\nBasic droplet properties:  \n')
    print('   de = %g (m)' % (oil.diameter(m, T, P, Sa, Ta)))
    shape, de, rho_p, rho, mu_p, mu, sigma = oil.particle_shape(m, T, P, Sa, Ta)
    print('   shape = %g (1: Sphere, 2: Ellipsoid, 3: Spherical Cap)'
        % shape)
    print('   us = %g (m/s)' % (oil.slip_velocity(m, T, P, Sa, Ta)))
    print('   A = %g (m^2)' % (oil.surface_area(m, T, P, Sa, Ta)))
    print('   beta_T = %g (m/s)' % (oil.heat_transfer(m, T, P, Sa, Ta)))

