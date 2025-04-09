"""
Bent Plume Model: Bubble Plume in Crossflow
===========================================

Use the ``TAMOC`` `bent_plume_model` to simulate a typical bubble plume in a
laboratory crossflow experiment. This script demonstrates the typical steps
involved in running the bent bubble model with petroleum fluids in the ocean.

This simulation creates new ambient crossflow data to match a typical 
laboratory flume and saves the data to the local directory in the file 
name crossflow_plume.nc.

"""
# S. Socolofsky, October 2014, Texas A&M University <socolofs@tamu.edu>.
from __future__ import (absolute_import, division, print_function)

from tamoc import seawater, ambient, dbm
from tamoc import bent_plume_model as bpm
from tamoc import dispersed_phases

from netCDF4 import date2num
from datetime import datetime

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def get_profile(profile_fname, z0, D, uj, vj, wj, Tj, Sj, ua, T, F, 
                sign_dp, H):
    """
    Create and ambient profile dataset for an integral jet model simulation
    
    """
    # Use mks units
    g = 9.81
    rho_w = 1000. 
    Pa = 101325.
    
    # Get the ambient density at the discharge from F (Assume water is 
    # incompressible for these laboratory experiments)
    rho_j = seawater.density(Tj, Sj, 101325. + rho_w * g * z0)
    Vj = np.sqrt(uj**2 + vj**2 + wj**2)
    if F == 0.:
        rho_a = rho_j
    else:
        rho_a = rho_j / (1. - Vj**2 / (sign_dp * F**2 * D * g))
    
    # Get the ambient stratification at the discharge from T
    if T == 0.:
        dpdz = 0.
    else:
        dpdz = sign_dp * (rho_a - rho_j) / (T * D)
    
    # Find the salinity at the discharge assuming the discharge temperature
    # matches the ambient temperature
    Ta = Tj
    def residual(Sa, rho, H):
        """
        docstring for residual
        
        """
        return rho - seawater.density(Ta, Sa, Pa + rho_w * g * H)
    Sa = fsolve(residual, 0., args=(rho_a, z0))
    
    # Find the salinity at the top and bottom assuming linear stratification
    if dpdz == 0.:
        S0 = Sa
        SH = Sa
    else:
        rho_H = dpdz * (H - z0) + rho_a
        rho_0 = dpdz * (0.- z0) + rho_a
        # Use potential density to get the salinity
        SH = fsolve(residual, Sa, args=(rho_H, z0))
        S0 = fsolve(residual, Sa, args=(rho_0, z0))
    
    # Build the ambient data arrays    
    z = np.array([0., H])
    T = np.array([Ta, Ta])
    S = np.array([S0, SH])
    ua = np.array([ua, ua])
    
    # Build the profile
    profile = build_profile(profile_fname, z, T, S, ua)
    
    # Return the ambient data
    return profile


def build_profile(fname, z, T, S, ua):
    """
    docstring for build_profile
    
    """
    # Prepare the data for insertion in the netCDF database
    data = np.zeros((z.shape[0], 4))
    names = ['z', 'temperature', 'salinity', 'ua']
    units = ['m', 'K', 'psu', 'm/s']
    data[:,0] = z.transpose()
    data[:,1] = T.transpose()
    data[:,2] = S.transpose()
    data[:,3] = ua.transpose()
    
    # Create the netCDF file to store the data
    nc_file = fname
    summary = 'Test case for jet in crossflow'
    source = 'Laboratory data'
    sea_name = 'Laboratory'
    p_lat = 0.
    p_lon = 0.
    p_time = date2num(datetime(2014, 10, 15, 16, 0, 0), 
                      units = 'seconds since 1970-01-01 00:00:00 0:00',
                      calendar = 'julian')
    nc = ambient.create_nc_db(nc_file, summary, source, sea_name, p_lat,
                              p_lon, p_time)
    
    # Insert the data into the netCDF dataset
    comments = ['measured', 'measured', 'measured', 'measured']
    nc = ambient.fill_nc_db(nc, data, names, units, comments, 0)
    
    # Compute the pressure and insert into the netCDF dataset
    P = ambient.compute_pressure(data[:,0], data[:,1], data[:,2], 0)
    P_data = np.vstack((data[:,0], P)).transpose()
    nc = ambient.fill_nc_db(nc, P_data, ['z', 'pressure'], ['m', 'Pa'], 
                            ['measured', 'computed'], 0)
    
    # Create an ambient.Profile object from this dataset
    profile = ambient.Profile(nc, chem_names='all')
    profile.close_nc()
    
    return profile


def crossflow_plume(fig):
    """
    Define, run, and plot the simulations for a pure bubble plume in crossflow
    for validation to data in Socolofsky and Adams (2002).
    
    """
    # Jet initial conditions
    z0 = 0.64
    U0 = 0.
    phi_0 = - np.pi / 2.
    theta_0 = 0.
    D = 0.01
    Tj = 21. + 273.15
    Sj = 0.
    cj = 1.
    chem_name = 'tracer'
    
    # Ambient conditions
    ua = 0.15
    T = 0.
    F = 0.
    H = 1.0
    
    # Create the correct ambient profile data
    uj = U0 * np.cos(phi_0) * np.cos(theta_0)
    vj = U0 * np.cos(phi_0) * np.sin(theta_0)
    wj = U0 * np.sin(phi_0)
    profile_fname = './crossflow_plume.nc'
    profile = get_profile(profile_fname, z0, D, uj, vj, wj, 
              Tj, Sj, ua, T, F, 1., H)
    
    # Create a bent plume model simulation object
    jlm = bpm.Model(profile)
    
    # Define the dispersed phase input to the model
    composition = ['nitrogen', 'oxygen', 'argon', 'carbon_dioxide']
    mol_frac = np.array([0., 0., 0., 1.])
    air = dbm.FluidParticle(composition)
    particles = []
    
    # Large bubbles
    Q_N = 0.5 / 60. / 1000.
    de0 = 0.008
    T0 = Tj
    lambda_1 = 1.
    (m0, T0, nb0, P, Sa, Ta) = dispersed_phases.initial_conditions(
        profile, z0, air, mol_frac, Q_N, 1, de0, T0)
    particles.append(bpm.Particle(0., 0., z0, air, m0, T0, nb0, lambda_1,
        P, Sa, Ta, K=1., K_T=1., fdis=1.e-6))
    
    # Small bubbles
    Q_N = 0.5 / 60. / 1000.
    de0 = 0.003
    T0 = Tj
    lambda_1 = 1.
    (m0, T0, nb0, P, Sa, Ta) = dispersed_phases.initial_conditions(
        profile, z0, air, mol_frac, Q_N, 1, de0, T0)
    particles.append(bpm.Particle(0., 0., z0, air, m0, T0, nb0, lambda_1,
        P, Sa, Ta, K=1., K_T=1., fdis=1.e-6))
    
    # Run the simulation
    jlm.simulate(np.array([0., 0., z0]), D, U0, phi_0, theta_0,
        Sj, Tj, cj, chem_name, particles, track=True, dt_max=60., 
        sd_max = 100.)
    
    # Perpare variables for plotting
    xp = jlm.q[:,7] / jlm.D
    yp = jlm.q[:,9] / jlm.D
    
    plt.figure(fig)
    plt.clf()
    plt.show()
    
    ax1 = plt.subplot(111)
    ax1.plot(xp, yp, 'b-')
    ax1.set_xlabel('x / D')
    ax1.set_ylabel('z / D')
    ax1.invert_yaxis()
    ax1.grid(visible=True, which='major', color='0.65', linestyle='-')
    
    plt.draw()
    
    return jlm


if __name__ == '__main__':
    
    # Bubble plume in crossflow
    sim_01 = crossflow_plume(1)
    sim_01.plot_all_variables(10)


