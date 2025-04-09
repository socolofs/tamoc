"""
Perform calculations for carbon dioxide
=======================================

Use the ``TAMOC`` ``DBM`` to evaluate the Peng-Robinson equation of state for 
co2 over a range of temperatures, pressures, and salinities.

Here, we demonstrate a phase change for CO2 from gas to liquid at around 45 
bar and demonstrate the use of the methods:

* `dbm.FluidMixture.density`

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

from tamoc import dbm
from tamoc import seawater

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Define the composition of co2 and assume 1 kg for the calculations
    composition = ['carbon_dioxide']
    m = 1.0
    
    # Create a DBM FluidMixture object for co2
    co2 = dbm.FluidMixture(composition)
    
    # Calculate the density at standard conditions
    rho_m = co2.density(m, 273.15+15., 101325.)[0]
    print('\nStandard density of co2 is: %g (kg/m^3)' % rho_m)
    
    # Calculate the density at deepwater ocean conditions
    rho_m = co2.density(m, 273.15+4., 150.*1.e5)[0]
    print('\nDensity of co2 at 4 deg C and 150 bar is: %g (kg/m^3)' % rho_m)
    
    # Density of co2 for a range of pressures, demonstrating the phase 
    # transition from gas to liquid.
    T = 273.15 + 10.
    P = np.linspace(1., 150., 300) * 1e5
    rho_m = np.zeros((P.shape[0],2))
    for i in range(len(P)):
        rho_m[i,:] = co2.density(m, T, P[i]).transpose()
    
    # Plot the results
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.plot(P*1e-5, rho_m[:,0])
    ax1.plot(P*1e-5, rho_m[:,1])
    ax1.set_xlabel('P (bar)')
    ax1.set_ylabel('rho (kg/m^3)')
    ax1.set_title('T = %g (deg C)' % (T-273.15))
    
    # Calculate the solubility for a range of salinities at 10 deg C and 
    # 1 atmosphere.
    T = 273.15 + 10.
    P = 101325.
    Sa = np.linspace(0., 35., 100)
    Cs = np.zeros(Sa.shape)
    for i in range(len(Cs)):
        Cs[i] = co2.solubility(m, T, P, Sa[i])[0,0]
    
    ax2 = plt.subplot(122)
    ax2.plot(Sa, Cs)
    ax2.set_xlabel('S (psu)')
    ax2.set_ylabel('Cs (kg/m^3)')
    ax2.set_title('T = %g (deg C), P = %g (bar)' % (T-273.15, P*1e-5))
    
    plt.show()

