"""
Equilibrium partitioning
========================

Use the ``TAMOC`` `dbm` to compute the equilibrium partitioning between 
gas and liquid phases of a hydrocarbon mixture.

In particular, this script demonstrates the methods:

* `dbm.FluidParticle.equilibrium`
* `dbm.equilibrium`
* `dbm.gas_liq_eq`

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from tamoc import dbm

import numpy as np

if __name__ == '__main__':
    
    # Define the thermodynamic state
    T = 273.15 + 5./9. * (160. - 32.)
    P = 1000. * 6894.76
    
    # Define the properties of the complete mixture (gas + liquid)
    composition = ['methane', 'n-butane', 'n-decane']
    oil = dbm.FluidMixture(composition)
    yk = np.array([0.5301, 0.1055, 0.3644])
    m = oil.masses(yk)
    
    # Compute the mass partitioning at equilibrium
    (mk, xi, K) = oil.equilibrium(m, T, P)
    
    # Print the results to the screen
    print('Gas/Liquid Equilibrium Calculations')
    print('=====================================================')
    print('\nMixture contains: %s' % composition)
    
    print('\nThermodynamic state is:')
    print('T = %g (K)' % T)
    print('P = %g (Pa)' % P)
    
    print('\nTotal mass in each component in kg is:')
    print(m)
    
    print('\nAt equilibrium, the masses in gas (top row) and')
    print('liquid (bottom row) are:')
    print(mk)
    
    print('\nThe associated mole fractions are:')
    print(xi)
    
    print('\nAnd the K-factors are:')
    print(K)
