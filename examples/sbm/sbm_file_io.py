"""
Single Bubble Model:  File input/output
=======================================

Use the ``TAMOC`` `single_bubble_model` to analyze results of previous 
simulations.  This script reads in data stored on the hard drive in netCDF
format and creates `single_bubble_model.Model` objects to store the saved
data.  It then plots the stored results as a demonstration that the data
are loaded correctly and as a means to analyze the results.

This script assumes that the results files have already been created by 
running the scripts::

    ./bin/sbm/bubble.py
    ./bin/sbm/drop.py
    ./bin/sbm/particle.py

It also uses the ambient data stored in the file
`../../test/output/test_bm54.nc`,created by the `./test/test_ambient` module. 
Please make sure all tests have passed before running this script or modify 
the script to use a different source of ambient data.

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient
from tamoc import dbm
from tamoc import seawater
from tamoc import single_bubble_model

import numpy as np

if __name__ == '__main__':
    
    # Create a single_bubble_model.Model object from the bubble.py results
    # stored on the hard drive.  Note that the netCDF file is self-documenting
    # and actually points to the ambient CTD profile data.  If that file
    # does not exist, the post_processing method cannot be applied as it 
    # reads from the ambient data.
    sbm = single_bubble_model.Model(simfile='./bubble.nc')
    
    # Echo the results to the screen:
    print('The results of ./bin/bubble.py have been loaded into memory')
    print('   len(t)   : %d' % sbm.t.shape[0])
    print('   shape(y) : %d, %d' % (sbm.y.shape[0], sbm.y.shape[1]))
    print('   composition : %s, ' % sbm.particle.composition)
    
    # You can re-run the simulation with different parameters by calling 
    # the simulate method.
    print('\nRe-running simulation with de = 0.01 m:')
    z0 = sbm.y[0,2]
    yk = sbm.particle.particle.mol_frac(sbm.particle.m0)
    T0 = sbm.particle.T0
    K = sbm.particle.K
    fdis = sbm.particle.fdis
    sbm.simulate(sbm.particle.particle, z0, 0.01, yk, T0, 
        K=K, K_T=sbm.K_T0, fdis=fdis, delta_t=sbm.delta_t)
    sbm.post_process()
    
    # You an also load stored simulation data to replace the current 
    # simulation
    sbm.load_sim('./particle.nc')
    
    # Echo the results to the screen:
    print('The results of ./bin/particle.py have been loaded into memory')
    print('   len(t)   : %d' % sbm.t.shape[0])
    print('   shape(y) : %d, %d' % (sbm.y.shape[0], sbm.y.shape[1]))
    print('   composition : %s, ' % sbm.particle.composition)
    
    # Plot these other results starting at figure 5
    sbm.post_process(5)

