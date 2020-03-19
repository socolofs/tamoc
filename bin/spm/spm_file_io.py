"""
Stratified Plume Model:  File input/output
==========================================

Use the ``TAMOC`` `stratified_plume_model` to analyze results of previous 
simulations.  This script reads in data stored on the hard drive in netCDF
format and creates `stratified_plume_model.Model` objects to store the saved
data.  It then plots the stored results as a demonstration that the data
are loaded correctly and as a means to analyze the results.

This script assumes that the results files have already been created by 
running the scripts::

    ./bin/spm/lake_bub.py
    ./bin/spm/lake_part.py
    ./bin/spm/blowout.py

It also uses the ambient data stored in the file
`../../test/output/test_bm54.nc`, created by the `./test/test_ambient` module
and the file `../../test/output/lake.nc`, created by the above scripts. Please
make sure all tests have passed before running this script or modify the
script to use a different source of ambient data.

"""
# S. Socolofsky, August 2013, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient
from tamoc import dbm
from tamoc import seawater
from tamoc import stratified_plume_model

import numpy as np

if __name__ == '__main__':
    
    # Create a stratified_plume_model.Model object from the lake_bub.py 
    # results stored on the hard drive.  Note that the netCDF file is 
    # self-documenting and actually points to the ambient CTD profile data.  
    # If that file does not exist, the post_processing method cannot be 
    # applied as it reads from the ambient data.
    spm = stratified_plume_model.Model(simfile='../../test/output/spm_gas.nc')
    
    # Echo the results to the screen:
    print('The results of ./bin/spm/lake_bub.py have been loaded into memory')
    print('   len(zi)   : %d' % spm.zi.shape[0])
    print('   shape(yi) : %d, %d' % (spm.yi.shape[0], spm.yi.shape[1]))
    print('   len(zo)   : %d' % spm.zo.shape[0])
    print('   shape(yo) : %d, %d' % (spm.yo.shape[0], spm.yo.shape[1]))
    for i in range(len(spm.particles)):
        print('   composition %d: %s, ' % (i, spm.particles[i].composition))
    
    # You can plot the results of that stored simulation
    spm.plot_state_space(1)
    
    # You an also load stored simulation data to replace the current 
    # simulation
    spm.load_sim('../../test/output/spm_blowout.nc')
    
    # Echo the results to the screen:
    print('The results of ./bin/spm/blowout.py have been loaded into memory')
    print('   len(zi)   : %d' % spm.zi.shape[0])
    print('   shape(yi) : %d, %d' % (spm.yi.shape[0], spm.yi.shape[1]))
    print('   len(zo)   : %d' % spm.zo.shape[0])
    print('   shape(yo) : %d, %d' % (spm.yo.shape[0], spm.yo.shape[1]))
    for i in range(len(spm.particles)):
        print('   composition %d: %s, ' % (i, spm.particles[i].composition))
    
    # Plot the results of that stored simulation
    spm.plot_state_space(1)

