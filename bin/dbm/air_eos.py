"""
Perform calculations for air
============================

Use the ``TAMOC`` ``DBM`` to evaluate the Peng-Robinson equation of state for 
air over a range of temperatures, pressures, and salinities.

In particular, this script demonstrates using:

* `dbm.FluidMixture.masses`
* `dbm.FluidMixture.density`
* `dbm.FluidMixture.solubility`

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from tamoc import dbm
from tamoc import seawater

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Define the composition of air
    composition = ['nitrogen', 'oxygen', 'argon', 'carbon_dioxide']
    mol_frac = np.array([0.78084, 0.20946, 0.009340, 0.00036])
    
    # Create a DBM FluidMixture object for air...assume the default zeros
    # matrix for the binary interaction coefficients
    air = dbm.FluidMixture(composition)
    
    # Get the mass composition for one mole of air
    m = air.masses(mol_frac)
    
    # Calculate the density at standard conditions
    rho_m = air.density(m, 273.15+15., 101325.)[0]
    print '\nStandard density of air is: %g (kg/m^3)' % rho_m
    
    # Calculate the viscosity at standard conditions
    mu = air.viscosity(m, 273.15+15., 101325.)[0]
    print '\nStandard viscosity of air is: %g (Pa s)' % mu
    
    # Calculate the density at deepwater ocean conditions
    rho_m = air.density(m, 273.15+4., 150.*1.e5)[0]
    print '\nDensity of air at 4 deg C and 150 bar is: %g (kg/m^3)' % rho_m
    
    # Compute the solubility of air into fresh and seawater at atmospheric
    # pressure
    T = 273.15 + np.linspace(4.0, 30.0, 100)
    P = 101325.
    Sa = np.array([0., 35])
    Cs = np.zeros((len(T), air.nc, 2))
    for i in range(len(T)):
        for j in range(2):
            Cs[i,:,j] = air.solubility(m, T[i], P, Sa[j])[0,:]
    
    # Plot the results
    fig = plt.figure()
    ax1 = plt.subplot(121)
    for i in range(len(composition)):
        ax1.semilogy(T, Cs[:,i,0])
    ax1.set_xlabel('T (K)')
    ax1.set_ylabel('Cs (kg/m^3)')
    ax1.set_title('Salinity = %g' % Sa[0])
    
    ax2 = plt.subplot(122)
    for i in range(len(composition)):
        ax2.semilogy(T, Cs[:,i,1])
    ax2.set_xlabel('T (K)')
    ax2.set_title('Salinity = %g' % Sa[1])
    
    plt.show()

