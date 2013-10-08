from tamoc import ambient
from tamoc import dbm
from tamoc import stratified_plume_model
from tamoc import params

import numpy as np

if __name__ == '__main__':

    # Get the ambient CTD profile data
    nc = '../../test/output/test_BM54.nc'
    try:
        # Open the lake dataset as a Profile object if it exists
        ctd = ambient.Profile(nc, chem_names='all')
        
    except RuntimeError:
        # Tell the user to create the dataset
        print 'CTD data not available; run test cases in ./test first.'
    
    # Create the stratified plume model object
    spm = stratified_plume_model.Model(ctd)
    
    # Set the release conditions
    T0 = 273.15 + 35.   # Release temperature in K
    R = 0.15            # Radius of leak source in m
    
    # Create the gas phase particles
    composition = ['methane', 'ethane', 'propane', 'oxygen']
    yk = np.array([0.93, 0.05, 0.02, 0.0])
    gas = dbm.FluidParticle(composition)
    z0 = 1000.
    disp_phases = []
    
    # Larger free gas bubbles
    mb0 = 8.         # total mass flux in kg/s
    de = 0.025       # bubble diameter in m
    lambda_1 = 0.85
    disp_phases.append(stratified_plume_model.particle_from_mb0(ctd, z0, gas, 
                       yk, mb0, de, lambda_1, T0))
    
    # Smaller free gas bubbles (note, it is not necessary to have more than
    # one bubble size)
    mb0 = 2.         # total mass flux in kg/s
    de = 0.0075      # bubble diameter in m
    lambda_1 = 0.9
    disp_phases.append(stratified_plume_model.particle_from_mb0(ctd, z0, gas, 
                       yk, mb0, de, lambda_1, T0))
    
    # Liquid hydrocarbon.  This could either be a dissolving phase (mixture
    # of liquid phases) or an inert phase.  We demonstrate here the simple
    # case of an inert oil phase
    oil = dbm.InsolubleParticle(True, True, rho_p=890., gamma=30., 
                                beta=0.0007, co=2.90075e-9)
    mb0 = 10.        # total mass flux in kg/s
    de = 0.004       # bubble diameter in m
    lambda_1 = 0.9
    disp_phases.append(stratified_plume_model.particle_from_mb0(ctd, z0, oil,
                       np.array([1.]), mb0, de, lambda_1, T0))
    
    # Compute the governing scales
    case = params.Scales(ctd, disp_phases)
    
    print case.get_variables(z0, 0.15)
    print 'h_T = %f (m)' % case.h_T(z0)
    print 'h_P = %f (m)' % case.h_P(z0)
    print 'h_S = %f (m)' % case.h_S(z0, 0.15)
    print 'lambda_1 = %f (--)' % case.lambda_1(z0, 0)
    print 'ua_crit = %f (m/s)' % case.u_inf_crit(z0)

