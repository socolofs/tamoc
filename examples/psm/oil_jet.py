"""
Particle Size Models:  Pure Oil Jet
===================================

Use the ``TAMOC`` `particle_size_models` module to simulate a laboratory
scale pure oil jet into water. This script demonstrates the typical steps
involved in using the `particle_size_models.PureJet` object, which requires
specification of all of the fluid properties of the jet.

"""
# S. Socolofsky, March 2020, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import seawater, particle_size_models

import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    print('\n---------------------------------------------------------------')
    print('Demonstration using the PureJet class in the')
    print('particle_size_models module of TAMOC for the ')
    print('experiments in the paper by Brandvik et al. (2013).')
    print('\nComparisons are for the data reported in Table 3')
    print('of the paper')
    print('---------------------------------------------------------------')
    
    # Simulate an experiment from Brandvik et al. (2013).  Their data uses
    # Oseberg oil, with the following reported properties
    rho_oil = 839.3
    mu_oil = 5.e-3
    sigma = 15.5e-3
    
    # We will simulate data from Table 3 in the Brandvik et al. (2013) paper.
    # These experiments have a nozzle diameter of 1.5 mm
    d0 = 0.0015
    
    # They also used seawater (assumed salinity of 34.5 psu) and released the
    # oil from a depth of about 6 m at a temperature of 13 deg C
    T = 273.15 + 13.
    S = 34.5
    rho = seawater.density(T, S, 101325.)
    P = 101325. + rho * 9.81 * 6.
    rho = seawater.density(T, S, P)
    mu = seawater.mu(T, S, P)
    
    # With this information, we can initialize a
    # `particle_size_models.PureJet` object
    jet = particle_size_models.PureJet(rho_oil, mu_oil, sigma, rho, mu, 
                                       fp_type = 1)
    
    # Brandvik et al. (2013) report the exit velocity at the nozzle.  We
    # need to convert this to a mass flow rate.  The mass flow rate should
    # always be reported within a numpy array, which allows for different
    # mass fluxes for different pseudocomponents of the oil.
    u_oil = 11.3
    A_oil = np.pi * (d0 / 2.)**2
    q_oil = u_oil * A_oil
    md_oil = np.array([rho_oil * q_oil])
    
    # To simulate the no-dispersant case, all of the oil properties in the
    # jet object are currently correct.  Hence, we may use:
    jet.simulate(d0, md_oil)
    
    # We compare the result to the measured data as follows:
    print('\nThe median droplet size for the no-disperant experiment is:')
    print('    Measured:  %3.3d um' % 237)
    print('    Modeled :  %3.3d um\n' % (jet.get_d50() * 1.e6))
    
    # When dispersant is added in sufficient quantities, the interfacial 
    # tension reduces and the droplet size gets smaller.  At a dispersant
    # to oil ratio of 50, sigma is:
    sigma = 0.05e-3
    
    # We can run this case by updating the properties of the jet object and
    # re-running the simualtion
    jet.update_properties(rho_oil, mu_oil, sigma, rho, mu, fp_type = 1)
    jet.simulate(d0, md_oil)
    
    # We compare the result to the measured data as follows:
    print('\nThe median droplet size for an experiments with a')
    print('dispersant to oil ratio of 50 is:')
    print('    Measured:  %3.3d um' % 170)
    print('    Modeled :  %3.3d um\n' % (jet.get_d50() * 1.e6))
    
    # We can also plot the size distribution
    print('\nThe corresponding size distribution is plotted in Figure 1')
    jet.get_distributions(15)
    jet.plot_psd(1)
    
    
    