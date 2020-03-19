"""
Liquid fluid particles
======================

Use the ``TAMOC`` ``DBM`` to specify a droplet containing liquid hydrocarbons 
that can dissolve and calculate all of its properties in deepwater conditions.

In particular, this script demonstrates the methods:

* `dbm.FluidParticle.mass_frac`
* `dbm.FluidParticle.density`
* `dbm.FluidParticle.mass_by_diameter`
* `dbm.FluidParticle.diameter`
* `dbm.FluidParticle.particle_shape`
* `dbm.FluidParticle.slip_velocity`
* `dbm.FluidParticle.surface_area`
* `dbm.FluidParticle.mass_transfer`
* `dbm.FluidParticle.heat_transfer`
* `dbm.FluidParticle.solubility`
    
"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from tamoc import dbm
from tamoc import seawater

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Define the composition of a natural gas
    composition = ['neohexane', 'benzene', 'toluene', 'ethylbenzene']
    mol_frac = np.array([0.25, 0.28, 0.18, 0.29])
    
    # Specify that we are interested in properties for the liquid phase
    fl_type = 1
    
    # Create a DBM FluidParticle object for this simple oil assuming zeros
    # for all the binary interaction coefficients
    delta = np.zeros((4,4))
    oil = dbm.FluidParticle(composition, fl_type, delta)
    
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
    mf = oil.mass_frac(mol_frac)
    T = 273.15 + 60.
    print('\nBasic properties of liquid oil: \n')
    print('   T = %g (K)' % T)
    print('   mol_frac = [' + ', '.join('%g' % mol_frac[i] for i in 
        range(oil.nc)) + '] (--)')
    print('   mass_frac = [' + ', '.join('%g' % mf[i] for i in 
        range(oil.nc)) + '] (--)')
    print('   rho_p = %g (kg/m^3) at %g (K) and %g (Pa)' %
        (oil.density(mf, T, P), T, P))
    
    # Get the masses in a 1.0 cm effective diameter droplet
    de = 0.01
    m = oil.masses_by_diameter(de, T, P, mol_frac)
    
    # Echo the properties of the droplet to the screen
    print('\nBasic droplet properties:  \n')
    print('   de = %g (m)' % (oil.diameter(m, T, P)))
    shape, de, rho_p, rho, mu_p, mu, sigma = oil.particle_shape(m, T, P, Sa, Ta)
    print('   shape = %g (1: Sphere, 2: Ellipsoid, 3: Spherical Cap)' \
        % shape)
    print('   us = %g (m/s)' % (oil.slip_velocity(m, T, P, Sa, Ta)))
    print('   A = %g (m^2)' % (oil.surface_area(m, T, P, Sa, Ta)))
    beta = oil.mass_transfer(m, T, P, Sa, Ta)
    print('   beta = [' + ', '.join('%g' % beta[i] for i in
        range(oil.nc)) + '] (m/s)')
    print('   beta_T = %g (m/s)' % (oil.heat_transfer(m, T, P, Sa, Ta)))
    Cs = oil.solubility(m, T, P, Sa)
    print('   Cs = [' + ', '.join('%g' % Cs[i] for i in
        range(oil.nc)) + '] (kg/m^3)')


