"""
Particle Size Models:  Blowout Jet
==================================

Use the ``TAMOC`` `particle_size_models` module to simulate a subsea accidental oil well blowout. This script demonstrates the typical steps
involved in using the `particle_size_models.Model` object, which uses the `dbm` module to compute fluid properties.

"""
# S. Socolofsky, March 2020, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient, dbm_utilities, particle_size_models

import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    print('\n---------------------------------------------------------------')
    print('Demonstration using the Model class in the')
    print('particle_size_models module of TAMOC for a ')
    print('synthetic subsea accidental blowout case')
    print('---------------------------------------------------------------')
    
    # The inputs for a particle_size_models.Model simulation are similar to
    # those of the bent_plume_model.  We start by defining the ambient CTD
    # data.  Here, we will use the world-ocean average data distributed with
    # TAMOC.  Since the particle_size_models do not use ambient current data,
    # we will ignore that.
    profile = ambient.Profile(None)
    
    # For the particle_size_models.Model object, we need to create a 
    # dbm.FluidMixture object.  Here, we make use of the dbm_utilities
    # to create synthetic blowout data.
    
    # Start by defining a substance
    substance={
        'composition' : ['n-hexane', '2-methylpentane', '3-methylpentane',
                         'neohexane', 'n-heptane', 'benzene', 'toluene', 
                         'ethylbenzene', 'n-decane'],
        'masses' : np.array([0.04, 0.07, 0.08, 0.09, 0.11, 0.12, 0.15, 0.18,
                             0.16])
    }
    
    # Decide which atmospheric gases to track.  The world-ocean average data 
    # only includes oxygen
    ca = ['oxygen']
    
    # Set the oil flow rate, GOR, release depth, and temperature
    q_oil = 20000.   # bbl/d
    gor = 500.       # ft^3/bbl at standard conditions
    z0 = 100.        # release depth (m)
    Tj = profile.get_values(z0, 'temperature') # release temperature (K)
    
    # Use the dbm_utilities to create the dbm.FluidMixture object and 
    # compute the mass fluxes of each pseudo-component of the oil at the
    # release
    oil, mass_flux = dbm_utilities.get_oil(substance, q_oil, gor, ca)
    
    # With this information, we can create the particle_size_models.Model
    # object
    spill = particle_size_models.Model(profile, oil, mass_flux, z0, Tj)
    
    # We compute the characteristic values of the particle size distribution
    # using the .simulate() method
    d0 = 0.15
    spill.simulate(d0, model_gas='wang_etal', model_oil='sintef')
    
    # We can report the particle sizes as follows:
    print('\nThe particle sizes for gas and oil are:')
    print('    d_50_gas = %3.3f mm' % (spill.get_d50(0) * 1.e3))
    print('    d_50_oil = %3.3f mm' % (spill.get_d50(1) * 1.e3))
    
    # We can also generate and plot particle size distributions
    de_gas, vf_gas, de_oil, vf_oil = spill.get_distributions(15,15)
    spill.plot_psd(1)
    
    