"""
Gas Fluid Particles
===================

Use the ``TAMOC`` ``DBM`` to specify a natural gas bubble that can dissolve 
and calculate all of its properties in deepwater conditions.

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

from tamoc import dbm
from tamoc import seawater

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Define the composition of a natural gas
    composition = ['methane', 'ethane', 'propane']
    mol_frac = np.array([0.90, 0.07, 0.03])
    
    # Specify that we are interested in properties for the gas phase
    fl_type = 0
    
    # Create a DBM FluidParticle object for this natural gas assuming zeros
    # for all the binary interaction coefficients
    delta = np.zeros((3,3))
    ng = dbm.FluidParticle(composition, fl_type, delta)
    
    # Specify some generic deepwater ocean conditions
    P = 150.0 * 1.0e5
    Ta = 273.15 + 4.0
    Sa = 34.5
    
    # Echo the ambient conditions to the screen
    print '\nAmbient conditions: \n'
    print '   P = %g (Pa)' % P
    print '   T = %g (K)' % Ta
    print '   S = %g (psu)' % Sa
    print '   rho_sw = %g (kg/m^3)' % (seawater.density(Ta, Sa, P))
    
    # Get the general properties of the gas
    mf = ng.mass_frac(mol_frac)
    T = 273.15 + 60.
    print '\nBasic properties of gas: \n'
    print '   T = %g (K)' % T
    print '   mol_frac = [' + ', '.join('%g' % mol_frac[i] for i in 
        range(ng.nc)) + '] (--)'
    print '   mass_frac = [' + ', '.join('%g' % mf[i] for i in 
        range(ng.nc)) + '] (--)'
    print '   rho_p = %g (kg/m^3) at %g (K) and %g (Pa)' % \
        (ng.density(mf, T, P), T, P)
    
    # Get the masses in a 1.0 cm effective diameter bubble
    de = 0.01
    m = ng.masses_by_diameter(de, T, P, mol_frac)
    
    # Echo the properties of the bubble to the screen
    print '\nBasic bubbles properties:  \n'
    print '   de = %g (m)' % (ng.diameter(m, T, P))
    shape, de, rho_p, rho, mu, sigma = ng.particle_shape(m, T, P, Sa, Ta)
    print '   shape = %g (1: Sphere, 2: Ellipsoid, 3: Spherical Cap)' \
        % shape
    print '   us = %g (m/s)' % (ng.slip_velocity(m, T, P, Sa, Ta))
    print '   A = %g (m^2)' % (ng.surface_area(m, T, P, Sa, Ta))
    beta = ng.mass_transfer(m, T, P, Sa, Ta)
    print '   beta = [' + ', '.join('%g' % beta[i] for i in \
        range(ng.nc)) + '] (m/s)'
    print '   beta_T = %g (m/s)' % (ng.heat_transfer(m, T, P, Sa, Ta))
    Cs = ng.solubility(m, T, P, Sa)
    print '   Cs = [' + ', '.join('%g' % Cs[i] for i in \
        range(ng.nc)) + '] (kg/m^3)'

