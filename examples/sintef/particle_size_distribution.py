"""
SINTEF Module:  Particle size distributions
===========================================

Use the ``TAMOC`` `sintef` module to predict the volume mean diameter and
particle size distribution for a typical blowout release using the methods
described in Johansen, Brandvik, and Farooq (2013), "Droplet breakup in subsea
oil releases - Part 2: Predictions of droplet size distributions with and
without injection of chemical dispersants." Marine Pollution Bulletin, 73:
327-335. doi:10.1016/j.marpolbul.2013.04.012.

These examples use the ambient conditions and oil and gas properties 
specified in the 2014 American Petroleum Institute Model Intercomparison
Workshop.  These data are for a hypothetical light crude oil blowout in deep
water.

"""
# S. Socolofsky, February 2014, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import dbm
from tamoc import sintef

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Tell user not to use these functions anymore
    print('\n-------')
    print('WARNING:  The sintef model is now deprecated.  ')
    print('          Use the particle_size_models module or the')
    print('          functions in the psf module instead.\n')
    
    # Enter parameters of the base test case:  20,000 bbl/d, 2000 m depth,
    # GOR of 2000, 30 cm orifice.  Per the results of the modeled oil in 
    # MultiFlash, these specifications result in the following parameter
    # values:
    D = 0.30
    rho_gas = 131.8
    m_gas = 7.4
    mu_gas = 0.00002
    sigma_gas = 0.06
    rho_oil = 599.3
    m_oil = 34.5
    mu_oil = 0.0002
    sigma_oil = 0.015
    rho = 1037.1
    
    # Compute the volume mean diameter (d_50) of oil and gas using the 
    # modified Weber number model
    d50_gas, d50_oil = sintef.modified_We_model(D, rho_gas, m_gas, mu_gas,
                       sigma_gas, rho_oil, m_oil, mu_oil, sigma_oil, rho)
    
    # Return results to the screen
    print('Deepwater blowout (GOR = 2000):')
    print('   d50_gas = %f (mm)' % (d50_gas * 1000.))
    print('   d50_oil = %f (mm)' % (d50_oil * 1000.))
    
    # Compare to maximum stable bubble/droplet size
    dmax_gas = sintef.de_max(sigma_gas, rho_gas, rho)
    dmax_oil = sintef.de_max(sigma_oil, rho_oil, rho)
    print('\nMaximum stable particle sizes:')
    print('   dmax_gas = %f (mm)' % (dmax_gas * 1000.))
    print('   dmax_oil = %f (mm)' % (dmax_oil * 1000.))
    
    # If the release were pure oil, ignore the gas contribution
    d50_gas, d50_oil = sintef.modified_We_model(D, 0., 0., 0., 0., rho_oil, 
                       m_oil, mu_oil, sigma_oil, rho)
    print('\nNo free gas:')
    print('   d50_oil = %f (mm)' % (d50_oil * 1000.))
    
    # Generate a gas and oil particle size distribution from the Rosin-
    # Rammler distribution using 30 bins
    d50_gas, d50_oil = sintef.modified_We_model(D, rho_gas, m_gas, mu_gas,
                       sigma_gas, rho_oil, m_oil, mu_oil, sigma_oil, rho)
    nbins = 30
    de_gas, md_gas = sintef.rosin_rammler(nbins, d50_gas, np.sum(m_gas), 
                     sigma_gas, rho_gas, rho)
    de_oil, md_oil = sintef.rosin_rammler(nbins, d50_oil, np.sum(m_oil),
                     sigma_oil, rho_oil, rho)
    
    # Plot the resulting size distributions
    plt.figure(1)
    plt.clf()
    
    # Prepare data for plotting
    index = np.arange(nbins)
    bar_width = 0.75
    opacity = 0.4
    
    # Gas bubble distribution
    ax1 = plt.subplot(211)
    ax1.bar(index, md_gas, bar_width, alpha=opacity, color='b')
    ax1.set_xlabel('Gas bubble diameter (mm)')
    ax1.set_ylabel('Gas mass flux (kg/s)')
    ntics = 10
    dtics = int(round(nbins/np.float(ntics)))
    ticnums = []
    ticlocs = []
    for i in range(ntics):
        ticnums.append('%2.2f' % (de_gas[(i-1)*dtics]*1000.))
        ticlocs.append(index[(i-1)*dtics] + bar_width/2)
    plt.xticks(ticlocs, ticnums)
    
    # Oil droplet distribution
    ax1 = plt.subplot(212)
    ax1.bar(index, md_oil, bar_width, alpha=opacity, color='r')
    ax1.set_xlabel('Oil droplet diameter (mm)')
    ax1.set_ylabel('Oil mass flux (kg/s)')
    ntics = 10
    dtics = int(round(nbins/np.float(ntics)))
    ticnums = []
    ticlocs = []
    for i in range(ntics):
        ticnums.append('%2.2f' % (de_oil[(i)*dtics]*1000.))
        ticlocs.append(index[(i)*dtics] + bar_width/2)
    plt.xticks(ticlocs, ticnums)
    
    plt.show()

