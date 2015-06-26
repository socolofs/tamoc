"""
Stratified Plume Model
======================

Simulate a buoyant plume in stratification dominate or quiescent conditions

This module defines the classes, methods, and functions necessary to simulate
the buoyant plume behavior in stratification dominant conditions, where the
effects of crossflows are negligible. The ambient water properties are
provided through an `ambient.Profile` class object, which contains a
netCDF4-classic dataset of CTD data and the needed interpolation methods. The
`dbm` class objects `dbm.FluidParticle` and `dbm.InsolubleParticle` report the
properties of the dispersed phase during the simulation, and these methods 
are provided to the model through the objects defined in `dispersed_phases`.  

Notes 
----- 
This model is a double plume integral model following the approach in
Socolofsky et al. (2008), but including the capability to have an arbitrary
number of dispersed phases, each with its own particle size distribution.

Detrainment to for subsurface intrusions follows the approach in Crounse
(2000).

See Also
--------
`single_bubble_model` : Tracks the trajectory of a single bubble, drop or 
    particle through the water column.  The numerical solution used here, 
    including the various object types and their functionality, follows the
    pattern in the `single_bubble_model`.  The main difference is the more
    complex state space and governing equations.

"""
# S. Socolofsky, November 2014, Texas A&M University <socolofs@tamu.edu>.

from tamoc import model_share
from tamoc import seawater
from tamoc import ambient
from tamoc import dbm
from tamoc import dispersed_phases
from tamoc import single_bubble_model
from tamoc import smp

from netCDF4 import Dataset
from datetime import datetime

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy
from warnings import warn
import os

class Model(object):
    """
    Master class object for controlling and post-processing the simulation
    
    Parameters
    ----------
    profile : `ambient.Profile` object, default = None
        An object containing the ambient CTD data and associated methods.  
        The netCDF dataset stored in the `ambient.Profile` object may be open 
        or closed at instantiation.  If open, the initializer will close the 
        file since this model does not support changing the ambient data once 
        initialized.
    simfile: str, default = None
        File name of a netCDF file containing the results of a previous 
        simulation run.  
    
    Attributes
    ----------
    profile : `ambient.Profile`
        Ambient CTD data
    got_profile : bool
        Flag indicating whether or not the profile object was successfully
        loaded into the `Model` object memory.\
    p : `ModelParams`
        Container for the fixed model parameters
    sim_stored : bool
        Flag indicating whether or not a simulation result is stored in the 
        object memory
    particles : list 
        List of `dispersed_phases.PlumeParticle` objects describing each 
        dispersed phase in the simulation
    K_T0 : ndarray
        Array of the initial values of K_T for each of the dispersed phase
        particles.  Since this solution is iterative, heat transfer needs
        to be re-initialized at the start of each iteration, and thus, it is
        not sufficient to rely on the value of K_T stored inside the 
        `particles` list.
    R : float
        Radius of the release point (m)
    maxit : float
        Maximum number of iterations allowed in the interative solution
    toler : float
        Relative error sufficient to consider the iterative solution to have
        converged
    delta_z : float
        Maximum step size to take in the storage of the simulation solution
        (m)
    chem_names : list
        List of chemical parameters to track for the dissolution
    zi : ndarray
        Array of depths for the inner plume solution (m)
    yi : ndarray
        Array of state space values computed for the inner plume solution
    zo : ndarray
        Array of depths for the outer plume solution (m)
    yo : ndarray
        Array of state space values computed for the outer plume solution
    yi_local : `InnerPlume` object
        Object that translates the `yi` state space into the comprehensive 
        list of derived variables.
    yo_local : `OuterPlume` object
        Object that translates the `yo` state space into the comprehensive 
        list of derived variables.
   
    See Also
    --------
    simulate, save_sim, load_sim, plot_state_space, plot_all_variables
    
    """
    def __init__(self, profile=None, simfile=None):
        super(Model, self).__init__()
        
        if profile is None:
            # Create a Model object from a save file
            self.load_sim(simfile)
        else:
            # Create a new Model object
            self.profile = profile
            self.got_profile = True
            profile.close_nc()
            
            # Get the model parameters that the user cannot adjust
            self.p = ModelParams(self.profile)
            
            # Indicate that a simulation has not yet been conducted
            self.sim_stored = False
    
    def simulate(self, particles, z, R, maxit=10, toler=0.1, delta_z=1., 
                 plots=True):
        """
        Simulate the plume dynamics from given initial conditions
        
        Simulate the buoyant plume using a double plume integral model 
        approach until the plume reaches the surface or the plume momentum
        goes to zero following complete dissolution of the dispersed
        phases.  
        
        Parameters
        ----------
        particles : list of `dispersed_phases.PlumeParticle` objects
            List of `dispersed_phases.PlumeParticle` objects containing the 
            dispersed phase initial conditions
        z : float
            Depth of the release port (m)
        R : float
            Radius of the release port (m)
        maxit : float, default = 10
            Maximum number of iterations to converge between inner and outer
            plumes
        toler : float, default = 0.1
            Relative error tolerance to accept for convergence (--)
        delta_z : float, default = 1
            Maximum step size to use in the simulation (m).  The ODE solver 
            in `calculate` is set up with adaptive step size integration, so 
            in theory this value determines the largest step size in the 
            output data, but not the numerical stability of the calculation.
        plots : bool, default = True
            Flag specifying whether or not to generate the state space plots
            for each iteration of the simulation.  Turn off when running 
            multiple runs from scripts or when not using ``IPython`` with 
            the ``--pylab`` flag.
        
        """
        # Store the input parameters
        self.particles = particles
        self.K_T0 = np.array([self.particles[i].K_T for i in 
                              range(len(self.particles))])
        self.R = R
        self.maxit = maxit
        self.toler = toler
        self.delta_z = delta_z
        
        # Get the initial conditions for the simulation run
        z0, y0, self.chem_names = smp.main_ic(self.profile, self.particles,  
                                              self.p, z, self.R)
        
        # Store the initial conditions in the inner plume object
        self.yi_local = InnerPlume(z0, y0, self.profile, self.particles, 
                                   self.p, self.chem_names)
        
        self.yo_local = OuterPlume(z0, np.zeros(4 + self.yi_local.nchems), 
                                   self.profile, self.p, self.chem_names, 
                                   self.yi_local.b)
        
        # Set up the iteration and convergence error
        iter = 0
        ea = 9999.
        
        # Initialize the neighbor interpolation object to carry solution 
        # from one iteration to the next
        neighbor = interp1d(np.array([0, self.profile.z_max]), 
                   np.zeros((2, 4 + self.yo_local.nchems)).transpose())
        
        print '\n-- TEXAS A&M OIL-SPILL CALCULATOR (TAMOC) --\n'
        print '-- Stratified Plume Model                 --\n'
        while iter < self.maxit and np.abs(ea) > self.toler:
            
            # Store old solutions to check for convergence
            iter += 1
            if iter > 1:
                zi_old = copy(self.zi)
                zo_old = copy(self.zo)
                yi_old = copy(self.yi)
                yo_old = copy(self.yo)
            
            # Enter the main calculation loop
            print '\nIteration %2.2d---------------------------------------' \
                  % iter
            print '\n  Calculate inner plume...'
            sys.stdout.flush()
            
            self.zi, self.yi, neighbor = inner_main(self.yi_local, 
                self.yo_local, self.particles, self.profile, self.p, neighbor,
                self.delta_z)
            
            print '\n  Calculate outer plume...'
            self.zo, self.yo, neighbor = outer_main(self.yi_local, 
                self.yo_local, self.particles, self.profile, self.p, neighbor,
                self.delta_z)
            
            # Check the convergence criteria
            if iter > 1:
                # Compare the latest two solutions of the state space
                ea  = err_check(self.zi, self.yi, self.zo, self.yo, zi_old, 
                                yi_old, zo_old, yo_old, self.yi_local,
                                self.yo_local, self.particles, self.profile, 
                                self.p)
                print '\n  Relative error:  %g' % ea
            
            # Plot the state space to help track the convergence
            self.sim_stored = True
            if plots:
                self.plot_state_space(iter)
            
            # Restart heat transfer
            for i in range(len(self.particles)):
                self.particles[i].K_T = self.K_T0[i]
    
    def save_sim(self, fname, profile_path, profile_info):
        """
        Save the current simulation results
        
        Save the current simulation results and the model parameters so that 
        all information needed to rebuild the class object is stored in a
        file.  The output data are stored in netCDF4-classic format.
        
        Parameters
        ----------
        fname : str
            File name of the file to write
        profile_path : str
            String stating the file path to the ambient profile data relative 
            to the directory where `fname` will be saved.  
        profile_info : str
            Single line of text describing the ambient profile data.
        
        See Also
        --------
        dispersed_phases.save_particle_to_nc_file
        
        Notes
        -----
        It does not make sense to store the ambient data together with every
        simulation output file.  On the other hand, the simulation results
        may be meaningless without the context of the ambient data.  The 
        parameter `profile_path` provides a means to automatically load the
        ambient data assuming the profile data are kept in the same place 
        relative to the output file.  Since this cannot be guaranteed, the 
        `profile_info` variable provides additional descriptive information 
        so that the ambient data can be identified if they have been moved.
        
        """
        if self.sim_stored is False:
            print 'No simulation results to store...'
            print 'Saved nothing to netCDF file.\n'
            return
        
        # Create the netCDF dataset object
        title = 'Simulation results for the TAMOC Stratified Plume Model'
        nc = model_share.tamoc_nc_file(fname, title, profile_path, profile_info)
        
        # Create variables for the dimensions
        z = nc.createDimension('z', None)
        p = nc.createDimension('profile', 2)
        nsi = nc.createDimension('nsi', len(self.yi_local.y0))
        nso = nc.createDimension('nso', len(self.yo_local.y0))
        params = nc.createDimension('params', 1)
        
        # Create variables for the dispersed_phases.PlumeParticle objects
        dispersed_phases.save_particle_to_nc_file(nc, self.chem_names, 
            self.particles, self.K_T0)
        
        # Create the independent variables (depth for inner and outer plumes)
        z = nc.createVariable('z', 'f8', ('z', 'profile',))
        z.long_name = 'depth below the water surface'
        z.standard_name = 'depth'
        z.units = 'm'
        z.axis = 'Z'
        z.positive = 'down'
        z.n_inner = len(self.zi)
        z.n_outer = len(self.zo)
        z[:,0] = self.zi[:]
        z[:,1] = self.zo[:]
        
        # Create the dependent variables (inner and outer plumes)
        yi = nc.createVariable('yi', 'f8', ('z', 'nsi',))
        yi.long_name = 'inner plume state space'
        yi.standard_name = 'yi'
        yi.units = 'variable'
        yi.coordinate = 'z'
        for i in range(len(nc.dimensions['nsi'])):
            yi[0:len(self.zi),i] = self.yi[:,i]
        
        yo = nc.createVariable('yo', 'f8', ('z', 'nso',))
        yo.long_name = 'outer plume state space'
        yo.standard_name = 'yo'
        yo.units = 'variable'
        yo.coordinate = 'z'
        for i in range(len(nc.dimensions['nso'])):
            yo[0:len(self.zo),i] = self.yo[:,i]
        
        # Store the remaining variables needed to define this simulation
        R = nc.createVariable('R', 'f8', ('params',))
        R.long_name = 'radius of the release point'
        R.standard_name = 'R'
        R.units = 'm'
        R[0] = self.R
        Ta_p, Sa_p, P_p = self.profile.get_values(np.max(self.zi), 
                          ['temperature', 'salinity', 'pressure'])
        Ta = nc.createVariable('Ta', 'f8', ('params',))
        Ta.long_name = 'ambient temperature at the release point'
        Ta.standard_name = 'Ta'
        Ta.units = 'K'
        Ta[0] = Ta_p
        Sa = nc.createVariable('Sa', 'f8', ('params',))
        Sa.long_name = 'ambient salinity at the release point'
        Sa.standard_name = 'Sa'
        Sa.units = 'psu'
        Sa[0] = Sa_p
        P = nc.createVariable('P', 'f8', ('params',))
        P.long_name = 'ambient pressure at the release point'
        P.standard_name = 'P'
        P.units = 'Pa'
        P[0] = P_p
        maxit = nc.createVariable('maxit', 'f8', ('params',))
        maxit.long_name = 'maximum allowable number of iterations'
        maxit.standard_name = 'maxit'
        maxit.units = 'nondimensional'
        maxit[0] = self.maxit
        toler = nc.createVariable('toler', 'f8', ('params',))
        toler.long_name = 'relative error tolerance for convergence'
        toler.standard_name = 'toler'
        toler.units = 'nondimensional'
        toler[0] = self.toler
        delta_z = nc.createVariable('delta_z', 'f8', ('params',))
        delta_z.long_name = 'maximum step size in output'
        delta_z.standard_name = 'delta_z'
        delta_z.units = 'm'
        delta_z[0] = self.delta_z
        
        # Close the netCDF dataset
        nc.close()
    
    def save_txt(self, base_name, profile_path, profile_info):
        """
        Save the state space in ascii text format for exporting
        
        Save the state space (dependent and independent variables) in an 
        ascii text file for exporting to other programs (e.g., Matlab).  
        Because the vertical coordinate is different for the inner and 
        outer plume solutions and because we want this data to be easily
        loadable in Matlab (i.e., each column of data must have the same
        number of rows), this method writes two data files, one for the 
        inner plume and one for the outer plume.  
        
        Parameters
        ----------
        base_name : str
            Base file name for the output file.  This method appends 'inner'
            and 'outer' to the inner and outer solutions upstream of the 
            dot-extension.
        profile_path : str
            String stating the file path to the ambient profile data relative 
            to the directory where `fname` will be saved.  
        profile_info : str
            Single line of text describing the ambient profile data.
        
        See Also
        --------
        bent_plume_model.Model.save_txt, single_bubble_model.Model.save_txt
        
        Notes
        -----
        The output will be organized in columns, with each column as follows:
        
            0 : depth (m)
            1-n : state space
        
        A header will be written to a separate file with `yi_header` and 
        `yo_header` appended to the base file name given above specifying
        the exact contents of the state space.  
        
        These output files are written using the `numpy.savetxt` method.
        
        """
        if self.sim_stored is False:
            print 'No simulation results to store...'
            print 'Saved nothing to text file.\n'
            return
        
        # Create the header string that contains the column descriptions
        # for the inner plume
        p_list = ['Stratified Plume Model ASCII Output File \n']
        p_list.append('Created: ' + datetime.today().isoformat(' ') + '\n')
        p_list.append('Simulation based on CTD data in:\n')
        p_list.append(profile_path)
        p_list.append('\n')
        p_list.append(profile_info)
        p_list.append('\n\n')
        p_list.append('Column Descriptions:\n')
        p_list.append('    0:  Depth in m\n')
        p_list.append('    1:  Volume flux Q in m^3/s\n')
        p_list.append('    2:  Momentum flux M in m^4/s^2\n')
        p_list.append('    3:  Salinity flux S in psu m^3/s\n')
        p_list.append('    4:  Heat flux H in J/s\n')
        idx = 4
        for i in range(self.yi_local.np):
            for j in range(len(self.particles[i].composition)):
                idx += 1
                p_list.append(
                    '    %d:  Total mass flux of %s in particle %d in kg/s\n' %
                    (idx, self.particles[i].composition[j], i))
            idx += 1
            p_list.append('    %d:  Total heat flux of particle %d in J/s\n' %
                          (idx, i))
        for i in range(self.yi_local.nchems):
            idx += 1
            p_list.append('    %d:  Dissolved mass flux of %s in kg/s\n' % 
                          (idx, self.yi_local.chem_names[i]))
        header_inner = ''.join(p_list)
        
        # Create the header string that contains the column descriptions
        # for the outer plume
        p_list = ['Stratified Plume Model ASCII Output File \n']
        p_list.append('Created: ' + datetime.today().isoformat(' ') + '\n')
        p_list.append('Simulation based on CTD data in:\n')
        p_list.append(profile_path)
        p_list.append('\n')
        p_list.append(profile_info)
        p_list.append('\n\n')
        p_list.append('Column Descriptions:\n')
        p_list.append('    0:  Depth in m\n')
        p_list.append('    1:  Volume flux Q in m^3/s\n')
        p_list.append('    2:  Momentum flux M in m^4/s^2\n')
        p_list.append('    3:  Salinity flux S in psu m^3/s\n')
        p_list.append('    4:  Heat flux H in J/s\n')
        idx = 4
        for i in range(self.yo_local.nchems):
            idx += 1
            p_list.append('    %d:  Dissolved mass flux of %s in kg/s\n' % 
                          (idx, self.yi_local.chem_names[i]))
        header_outer = ''.join(p_list)
        
        # Assemble and write the inner plume data
        data_inner = np.hstack((np.atleast_2d(self.zi).transpose(), self.yi))
        np.savetxt(base_name + '_inner.txt', data_inner)
        with open(base_name + '_inner_header.txt', 'w') as dat_file:
            dat_file.write(header_inner)
        
        # Assemble and write the outer plume data
        data_outer = np.hstack((np.atleast_2d(self.zo).transpose(), self.yo))
        np.savetxt(base_name + '_outer.txt', data_outer)
        with open(base_name + '_outer_header.txt', 'w') as dat_file:
            dat_file.write(header_outer)
    
    def load_sim(self, fname):
        """
        Load in a saved simulation result file for post-processing
        
        Load in a saved simulation result file and rebuild the `Model`
        object attributes.  The input files are in netCDF4-classic data 
        format.
        
        Parameters
        ----------
        fname : str
            File name of the file to read
        
        See Also
        --------
        save_sim
        
        Notes
        -----
        This method will attempt to load the ambient profile data from the
        `profile_path` attribute of the `fname` netCDF file.  If the load 
        fails, a warning will be reported to the terminal, but the other 
        steps of loading the `Model` object attributes will be performed.
        
        This method performs the same function as the 
        `single_bubble_model.Model.load_sim` method, but with slightly 
        different variables and variable dimensions.  
        
        """
        # Open the netCDF dataset object containing the simulation results
        nc = Dataset(fname)
        
        # Try to get the profile data
        self.profile = model_share.profile_from_model_savefile(nc, fname)
        if self.profile is not None:
            self.p = ModelParams(self.profile)
            self.got_profile = True
        else:
            self.p = None
            self.got_profile = False
        
        # Create the dispersed_phases.PlumeParticle objects
        self.particles, self.chem_names = \
            dispersed_phases.load_particle_from_nc_file(nc, 1)
        
        # Create the remaining model attributes
        nzi = nc.variables['z'].n_inner
        nzo = nc.variables['z'].n_outer
        nsi = len(nc.dimensions['nsi'])
        nso = len(nc.dimensions['nso'])
        self.K_T0 = np.array([self.particles[i].K_T for i in 
                              range(len(self.particles))])
        self.R = nc.variables['R'][0]
        self.maxit = nc.variables['maxit'][0]
        self.toler = nc.variables['toler'][0]
        self.delta_z = nc.variables['delta_z'][0]
        if self.got_profile:
            self.yi_local = InnerPlume(nc.variables['z'][0,0], 
                nc.variables['yi'][0,:], self.profile, self.particles, 
                self.p, self.chem_names)
            self.yo_local = OuterPlume(nc.variables['z'][nzo-1,1], 
                nc.variables['yo'][nzo-1,:], self.profile, self.p, 
                self.chem_names, self.yi_local.b)
        self.zi = nc.variables['z'][0:nzi,0]
        self.zo = nc.variables['z'][0:nzo,1]
        self.yi = np.zeros((nzi, nsi))
        self.yo = np.zeros((nzo, nso))
        for i in range(nsi):
            self.yi[:,i] = nc.variables['yi'][0:nzi,i] 
        for i in range(nso):
            self.yo[:,i] = nc.variables['yo'][0:nzo,i]
        
        # Close the netCDF dataset
        nc.close()
        self.sim_stored = True
    
    def plot_state_space(self, fig):
        """
        Plot the simulation state space
        
        Plot the standard set of state space variables used to evaluate 
        model convergence and quality of the model solution
        
        Parameters
        ----------
        fig : int
            Number of the figure window in which to draw the plot
        
        See Also
        --------
        plot_all_variables
        
        """
        if self.sim_stored is False:
            print 'No simulation results available to plot...'
            print 'Plotting nothing.\n'
            return
        
        # Plot the results        
        print 'Plotting the state space...'
        plot_state_space(self.zi, self.yi, self.zo, self.yo, 
                         self.yi_local, self.yo_local, self.particles,
                         self.profile, self.p, fig)
        print 'Done.\n'
    
    def plot_all_variables(self, fig):
        """
        Plot a comprehensive suite of simulation results
        
        Generate a comprehensive suite of graphs showing the state and 
        derived variables along with ambient profile data in order to 
        view the model output for detailed analysis.
        
        Parameters
        ----------
        fig : int
            Number of the figure window in which to draw the plot
        
        See Also
        --------
        plot_state_space
        
        """
        if self.sim_stored is False:
            print 'No simulation results available to plot...'
            print 'Plotting nothing.\n'
            return
        
        # Plot the results        
        print 'Plotting the full variable suite...'
        plot_all_variables(self.zi, self.yi, self.zo, self.yo, 
                           self.yi_local, self.yo_local, self.particles,
                           self.profile, self.p, fig)
        print 'Done.\n'
    

class ModelParams(single_bubble_model.ModelParams):
    """
    Fixed model parameters for the stratified plume model
    
    This class stores the set of model parameters that should not be adjusted
    by the user and that are needed by the stratified plume model.
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    
    Attributes
    ----------
    rho_r : float
        Reference density (kg/m^3) evaluated at mid-depth of the water body.
    alpha_1 : float
        entrainment coefficient from the ambient fluid to the inner plume
    alpha_2 : float
        entrainment coefficient from the inner plume to the outer plume
    alpha_3 : float
        entrainment coefficient from the ambient fluid to the outer plume
    lambda_2 : float
        spreading rate of dissolved constituents
    epsilon : float
        continuous peeling parameter (see Crounse 2000)
    c1 : float
        coefficient for counterflow entrainment model
    c2 : float
        coefficient for counterflow entrainment model
    fe : float
        extra entrainment factor when the inner plume hits the free 
        surface
    gamma_i : float
        momentum amplification factor for inner plume (see Milgram 1983)
    gamma_o : float
        momentum amplification factor for outer plume
    Fr_0 : float
        initial plume Froude number (see Wueest et al. 1992)
    Fro_0 : float
        initial plume Froude number for outer plume segments
    nwidths : float
        average size of detraining eddies as a fraction of the plume
        half-width
    naverage : float
        number of previous outer plume solutions to average
    g : float
        acceleration of gravity (m/s^2)
    Ru : float
        universal ideal gas constant (J/(mol K))
    
    Notes
    -----
    This object inherits the `single_bubble_model.ModelParams` object, which
    defines the attribute `rho_r`.
    
    """
    def __init__(self, profile):
        super(ModelParams, self).__init__(profile)
        
        # Set the model parameters to the values recommended by Socolofsky
        # et al. (2008)
        self.alpha_1 = 0.055
        self.alpha_2 = 0.110
        self.alpha_3 = 0.110
        self.lambda_2 = 1.00
        self.epsilon = 0.015
        self.qdis_ic = 0.1
        self.c1 = 0.
        self.c2 = 1.
        self.fe = 0.1
        self.gamma_i = 1.10
        self.gamma_o = 1.10
        self.Fr_0 = 1.6
        self.Fro_0 = 0.1
        self.nwidths = 1
        self.naverage = 1


class InnerPlume(object):
    """
    Manages the inner plume state space and derived variables
    
    Manages storage and calculation of the derived variables for the inner
    plume as a function of the state space. 
    
    Parameters
    ----------
    z0 : float
        Starting depth (m) of the inner plume
    y0 : ndarray
        Initial conditions for the inner plume.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    particles : list of `dispersed_phases.PlumeParticle` objects
        List of `dispersed_phases.PlumeParticle` objects containing the 
        dispersed phase local conditions and behavior.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    chem_names : str list
        List of chemical names to track for the dissolution.
    
    Attributes
    ----------
    z0 : float
        Starting depth (m) of the inner plume
    y0 : ndarray
        Initial conditions for the inner plume
    chem_names : str lit
        List of chemical names to track for the dissolution
    len : int
        Number of variables in the state space vector y
    nchems : int
        Number of chemical names to track for the dissolution
    np : int
        Number of dispersed phase particle objects
    z : float
        Current value of the depth (m)
    y : ndarray
        Current value of the state space vector
    Q : float
        Continuous phase volume flux (m^3/s)
    J : float
        Continuous phase kinematic momentum flux (m^4/s^2)
    S : float
        Plume salinity flux (psu m^3/s)
    H : float
        Continuous phase heat flux (J/s)
    M_p : dict of ndarrays
        For integer key: the total mass fluxes (kg/s) of each component in a 
        particle.
    H_p : ndarray
        Total heat flux for each particle (J/s)
    C : ndarray
        Mass flux of dissolved components (kg/s)
    Ta : float
        Ambient temperature outside the plume (K)
    Sa : float
        Ambient salinity outside the plume (psu)
    P : float
        Local pressure (Pa)
    rho_a : float
        Ambient density outside the plume (kg/m^3)
    ca : ndarray
        Ambient concentrations outside the plume (kg/m^3)
    u : float
        Velocity of the continuous phase fluid (m/s)
    b : float
        Plume half-width (m)
    s : float
        Salinity of the plume fluid (psu)
    T : float
        Temperature of the plume fluid (K)
    c : ndarray
        Concentrations of the dissolved components in the plume fluid 
        (kg/m^3)
    rho : float
        Density of the continuous phase inside the plume (kg/m^3)
    xi : ndarray
        Void fraction of each particle object (--)
    fb : ndarray
        Buoyant force of each particle object (kg/m^3)
    Xi : float
        Total void fraction in the plume (--)
    Fb : float
        Total buoyant force of all particles in the plume (??)
    Ep : float
        Local peeling flux (m^2/s)
    
    """
    def __init__(self, z0, y0, profile, particles, p, chem_names):
        super(InnerPlume, self).__init__()
        
        # Store the input variables that will stay with the inner plume
        self.z0 = z0
        self.y0 = y0
        self.chem_names = chem_names
        self.len = y0.shape[0]
        self.nchems = len(chem_names)
        self.np = len(particles)
        
        # Extract the state variables and compute the derived quantities
        self.update(z0, y0, particles, profile, p)
    
    def update(self, z, y, particles, profile, p):
        """
        Update the `InnerPlume` object with the current local conditions
        
        Extract the state variables and compute the derived quantities given
        the current local conditions.
        
        Parameters
        ----------
        z : float
            Local depth (m)
        y : ndarray
            Local values of the inner plume state space
        particles : list of `dispersed_phases.PlumeParticle` objects
            List of `dispersed_phaes.PlumeParticle` objects containing the 
            dispersed phase local conditions and behavior.
        profile : `ambient.Profile` object
            The ambient CTD object used by the simulation.
        p : `ModelParams` object
            Object containing the fixed model parameters for the stratified 
            plume model.
        
        """
        # Save the current state space 
        self.z = z
        self.y = y
        
        # Extract the state space variables from y
        self.Q = y[0]
        self.J = y[1]
        self.S = y[2]
        self.H = y[3]
        idx = 4
        M_p = {} # Since inert particles have different components than soluble
        H_p = [] # Each particle has one value for temperature (heat)
        t_p = [] # Each particle has its own age (time since release)
        for i in range(self.np):
            M_p[i] = y[idx:idx + particles[i].particle.nc]
            idx += particles[i].particle.nc
            H_p.extend(y[idx:idx + 1])
            idx += 1
            t_p.extend(y[idx:idx + 1])
            idx += 1
        self.M_p = M_p
        self.H_p = np.array(H_p)
        self.t_p = np.array(t_p)
        if self.nchems >= 1:
            self.C = y[idx:]
        else:
            self.C = np.array([])
        
        # Get the local ambient conditions
        self.Ta, self.Sa, self.P = profile.get_values(self.z, ['temperature',
                                   'salinity', 'pressure'])
        self.rho_a = seawater.density(self.Ta, self.Sa, self.P)
        self.ca = profile.get_values(self.z, self.chem_names)
        
        if self.Q > 0.:
            # Compute the continuous phase derived quantities
            self.u = self.J / self.Q
            self.b = self.Q / np.sqrt(np.pi * self.J)
            self.s = self.S / self.Q
            self.T = self.H / (p.rho_r * seawater.cp() * self.Q)
            self.c = self.C / self.Q
            self.rho = seawater.density(self.T, self.s, self.P)
            
            # Compute the dispersed phase derived quantities
            self.xi = np.zeros(self.np)
            self.fb = np.zeros(self.np)
            for i in range(self.np):
                # Update the particle properties
                m_p = self.M_p[i] / particles[i].nb0
                T_p = self.H_p[i] / (np.sum(self.M_p[i]) * particles[i].cp)
                particles[i].update(m_p, T_p, self.P, self.s, self.T, 
                                    self.t_p[i])
                
                # Calculate the individual void fractions in the plume
                self.xi[i] = np.sum(self.M_p[i]) / (particles[i].rho_p *
                    (np.pi * particles[i].lambda_1**2 * self.b**2 *
                    (particles[i].us + 2. * self.u / (1. + 
                    particles[i].lambda_1**2))))
                
                # Calculate the buoyant force of the current particle
                self.fb[i] = particles[i].lambda_1**2 * self.xi[i] * \
                             (self.rho_a - particles[i].rho_p)
                
                # Force void fraction and bubble force to zero if bubbles 
                # have dissolved
                if self.rho == particles[i].rho_p:
                    self.xi[i] = 0.
                    self.fb[i] = 0.
            
            # Compute the net void fraction and buoyancy flux
            self.Xi = np.sum(self.xi)
            self.Fb = np.sum(self.fb)
            
            # Get the peeling flux
            self.Ep = smp.cp_model(p.epsilon, particles, self.rho_a, 
                                    self.rho, p.g, p.rho_r, self.b, self.u)
        else:
            # There is no inner plume or the momentum is reversing
            self.u = 0.
            self.b = 0.
            self.s = self.Sa
            self.T = self.Ta
            self.c = self.ca
            self.rho = seawater.density(self.Ta, self.Sa, self.P)
            
            # Compute the dispersed phase derived quantities
            self.xi = np.zeros(self.np)
            self.fb = np.zeros(self.np)
            for i in range(self.np):
                # Update the particle properties
                m_p = self.M_p[i] * 0.
                T_p = self.Ta
                particles[i].update(m_p, T_p, self.P, self.Sa, self.Ta, 
                                    self.t_p[i])
            
            # Compute the net void fraction and buoyancy flux
            self.Xi = 0.
            self.Fb = 0.
            
            # Get the peeling flux
            self.Ep = 0.

class OuterPlume(object):
    """
    Manages the outer plume state space and derived variables
    
    Manages storage and calculation of the derived variables for the outer
    plume as a function of the state space. 
    
    Parameters
    ----------
    z0 : float
        Initial depth (m) of the outer plume
    y0 : ndarray
        Initial values for the outer plume state space.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    chem_names : str list
        List of chemical names to track for the dissolution.
    bi : float
        Half-width of the inner plume (m)
    
    Attributes
    ----------
    chem_names : str list
        List is dissolved chemical species to track
    len : int
        Number of dependent state space variables
    nchems : int
        Number of chemical species to track for the dissolution
    z : float
        Current value of the depth (m)
    y : ndarray
        Current value of the state space vector
    Q : float
        Volume flux of outer plume fluid (m^3/s)
    J : float
        Momentum flux of outer plume fluid (m^3/s)
    S : float
        Salinity flux in the outer plume (psu m^3/s)
    H : float
        Heat flux in the outer plume (J/s)
    C : ndarray
    Mass flux of dissolved species in the outer plume (kg/s)
    Ta : float
        Ambient temperature outside the plume (K)
    Sa : float
        Ambient salinity outside the plume (psu)
    P : float
        Local pressure (Pa)
    rho_a : float
        Ambient density outside the plume (kg/m^3)
    ca : ndarray
        Ambient concentrations outside the plume (kg/m^3)
    u : float
        Velocity of the outer plume (m/s)
    b : float
        Half-width of the outer plume (m)
    s : float
        Salinity of the outer plume continuous phase fluid (psu)
    T : float
        Temperature of the outer plume continous phase fluid (K)
    c : ndarray
        Concentration of dissolved species in the outer plume fluid (kg/m^2)
    rho : float
        Density of the outer plume continuous phase fluid (kg/m^3)
    
    """
    
    def __init__(self, z0, y0, profile, p, chem_names, bi):
        super(OuterPlume, self).__init__()
        
        # Store the input variables that will stay with the outer plume
        self.z0 = z0
        self.y0 = y0
        self.chem_names = chem_names
        self.len = y0.shape[0]
        self.nchems = len(chem_names)
        
        # Extract the state variables and compute the derived quantities
        self.update(z0, y0, profile, p, bi)
    
    def update(self, z, y, profile, p, bi):
        """
        Update the `OuterPlume` object with the current local conditions
         
        Extract the state variables and compute the derived quantities given
        the current local conditions.
        
        Parameters
        ----------
        z : float
            Local depth (m).
        y : ndarray
            Local values of the outer plume state space.
        profile : `ambient.Profile` object
            The ambient CTD object used by the simulation.
        p : `ModelParams` object
            Object containing the fixed model parameters for the stratified 
            plume model.
        bi : float
            Inner plume half-width (m)
        
        Notes
        -----
        This method writes the `OuterPlume` object attributes `z`, `y`, `Q`, 
        `J`, `S`, `H`, `C`, `Ta`, `Sa`, `P`, `rho_a`, `ca`, `u`, `b`, `s`, 
        `T`, `c`, and `rho`.
        
        """
        # Save the current state space
        self.z = z
        self.y = y
        
        # Extract the state-space variables from y
        self.Q = y[0]
        self.J = y[1]
        self.S = y[2]
        self.H = y[3]
        self.C = y[4:]
        
        # Get the local ambient conditions
        self.Ta, self.Sa, self.P = profile.get_values(self.z, ['temperature',
                                   'salinity', 'pressure'])
        self.rho_a = seawater.density(self.Ta, self.Sa, self.P)
        self.ca = profile.get_values(self.z, self.chem_names)
        
        if self.Q < 0.:
            # Compute the continuous phase derived quantities
            self.u = self.J / self.Q
            self.b = np.sqrt(self.Q**2 / (np.pi * self.J) + bi**2)
            self.s = self.S / self.Q
            self.T = self.H / (p.rho_r * seawater.cp() * self.Q)
            self.c = self.C / self.Q
            self.rho = seawater.density(self.T, self.s, self.P)
        else:
            # There is no outer plume or the momentum is reversing
            self.u = 0.
            self.b = 0.
            self.s = self.Sa
            self.T = self.Ta
            self.c = self.ca
            self.rho = self.rho_a
    

# ----------------------------------------------------------------------------
# Integration controllers
# ----------------------------------------------------------------------------

def inner_main(yi, yo, particles, profile, p, neighbor, delta_z):
    """
    Manage the integration of the inner plume
    
    Calculates the inner plume solution, creates the appropriate `neighbor`
    interpolation object, and returns the complete solution
    
    Parameters
    ----------
    yi : `InnerPlume` object
        Object containing the inner plume state space and methods to extract
        the state space variables
    yo : `OuterPlume` object
        Object containing the outer plume state space and methods to extract
        the state space variables
    particles : list of `dispersed_phases.PlumeParticle` objects
        List of `dispersed_phases.PlumeParticle` objects containing the 
        dispersed phase local conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    neighbor : `scipy.interpolate.interp1d` object
        Container holding the latest solution for the outer plume state
        space
    delta_z : float
        Maximum step size to use in the simulation (m).  The ODE solver 
        in `calculate` is set up with adaptive step size integration, so 
        in theory this value determines the largest step size in the 
        output data, but not the numerical stability of the calculation.
    
    Returns
    -------
    z : ndarray
        Vector of elevations where the inner plume solution is obtained (m)
    y : ndarray
        Matrix of inner plume state space solutions.  Each row corresponds to
        a depth in z
    neighbor : `scipy.interpolate.interp1d` object
        An updated neighbor interpolation object with the inner plume solution
        ready to use with integration of the outer plume
    
    Notes
    -----
    With the Crounse (2000) continuous peeling term, the inner plume can 
    integrate from the initial conditions to the free surface or top of a 
    dissolving plume without stopping--there are no discontinuities in the 
    solution.  Hence, no iteration loop is required for the inner plume 
    integration.
    
    """
    # Initialize the inner and outer plume objects with the conditions at the
    # base of the plume (needed for later iterations)
    yi.update(yi.z0, yi.y0, particles, profile, p)
    yo.update(yo.z0, yo.y0, profile, p, yi.b)
    
    # Integrate up the inner plume using the double plume integral model
    z, y = smp.calculate(yi, yo, particles, profile, p, neighbor, 
                          smp.derivs_inner, yi.z, yi.y, 0., -1., delta_z)
    
    print '  Done with inner plume calculations...'
    
    # Update yi and build the neighbor matrix
    yi.update(z[-1], y[-1,:], particles, profile, p)
    neighbor = interp1d(np.flipud(z), np.flipud(y).transpose())
    
    return (z, y, neighbor)

def outer_main(yi, yo, particles, profile, p, neighbor, delta_z):
    """
    Manage the integration of the outer plume segments
    
    Calculates the outer plume solution, creates the appropriate `neighbor`
    interpolation object, and returns the complete solution
    
    Parameters
    ----------
    yi : `InnerPlume` object
        Object containing the inner plume state space and methods to extract
        the state space variables.
    yo : `OuterPlume` object
        Object containing the outer plume state space and methods to extract
        the state space variables.
    particles : list of `dispersed_phases.PlumeParticle` objects
        List of `dispersed_phases.PlumeParticle` objects containing the 
        dispersed phase local conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    neighbor : `scipy.interpolate.interp1d` object
        Container holding the latest solution for the inner plume state
        space.
    delta_z : float
        Maximum step size to use in the simulation (m).  The ODE solver 
        in `calculate` is set up with adaptive step size integration, so 
        in theory this value determines the largest step size in the 
        output data, but not the numerical stability of the calculation.
    
    Returns
    -------
    z : ndarray
        Vector of elevations where the outer plume solution is obtained (m).
    y : ndarray
        Matrix of outer plume state space solutions.  Each row corresponds to
        a depth in z.
    neighbor : `scipy.interpolate.interp1d` object
        An updated neighbor interpolation object with the outer plume solution
        ready to use with integration of the inner plume.
    
    Notes
    -----
    There can be many outer plume segments, each associated with a localized, 
    intense area of detrainment from the inner plume.  Each of these outer 
    plume segments generally stop at a level of neutral buoyancy before 
    reaching the next segment, so they will each need to have an independent
    integration.  Thus, this function contains an iteration loop that 
    terminates when the plume has been integrated from the top to the bottom
    of the inner plume.  
    
    Once an outer plume segment stops at a level of neutral buoyancy, this
    function searches for the next outer plume by collecting detrained fluid
    over a length of inner plume equal to `nwidths` times the half-width and
    attempts to start an integration.  If the negative buoyancy of that 
    fluid is inadequate to overcome the upward drag of the inner plume, then
    the outer plume is said to be "not viable," and the algorithm attemps to
    do this again with the next `nwidths` of detrained water.  Once the 
    outer plume segment becomes viable, those initial conditions are passed
    to the `smp.calculate` function, and the outer plume is integrated to
    neutral buoyancy.  This succession of steps repeats until the bottom of
    the inner plume is reached.
    
    When dissolution in the inner plume is large enough that the detained 
    fluid is heavier than ambient (e.g., enriched by CO2 such that the 
    solution is not dilute), then outer plume segments can tend to overlap.
    In this case, also, the lowest outer plume segment may descend beyond
    the starting point of the inner plume.  This function assumes that the
    bottom of the CTD cast indicates the sea floor; hence, integration 
    always stops at the sea bottom.
    
    """
    # Get the initial conditions for the top of the plume
    if yi.z <= 0.:
        # The inner plume hit the free surface
        yi.update(0., neighbor(0.), particles, profile, p)
        z0, y0 = smp.outer_surf(yi, p)
    else:
        # The inner plume stopped because it fully dissolved
        z_0 = yi.z
        z0, y0 = smp.outer_dis(yi, particles, profile, p, neighbor, z_0)
    
    # Initialize the outer plume object with these initial conditions
    yi.update(z0, neighbor(z0), particles, profile, p)
    yo.update(z0, y0, profile, p, yi.b)
    
    # Integrate down the first outer plume segment
    print '\n    Top outer plume...'       
    z, y = smp.calculate(yi, yo, particles, profile, p, neighbor,
                          smp.derivs_outer, yo.z, yo.y, profile.z_max, 1., 
                          delta_z)
    
    # Start building the complete solution for the outer plume
    z_sol = z
    y_sol = y
    
    # Record the depth of the current calculations and the bottom of the 
    # inner plume
    z_depth = z[-1]
    z_source = yi.z0
    
    # Integrate down to the bottom of the inner plume and continue until the 
    # outer plume stops or reaches the bottom of the reservoir
    k = 1
    while z_depth < z_source:
        
        # Get the initial conditions for the next outer plume segment
        z0, y0, flag = smp.outer_cpic(yi, yo, particles, profile, p, 
                                       neighbor, z_depth)
        
        # Integrate the outer plume if it is viable
        if flag is True:
            # Integrate the outer plume
            print '    - Outer plume %2.2d' % k
            k += 1
            
            # Initialize the outer plume object with these initial conditions
            yi.update(z0, neighbor(z0), particles, profile, p)
            yo.update(z0, y0, profile, p, yi.b)
            
            # Integrate down the this outer plume segment
            z, y = smp.calculate(yi, yo, particles, profile, p, neighbor,
                                  smp.derivs_outer, yo.z, yo.y, 
                                  profile.z_max, 1., delta_z)
        
        else:
            # This intermediate peel is not viable
            z = z0
            y = y0
        
        # Add the latest iteration to the outer plume solution
        z_sol = np.hstack((z_sol, z))
        y_sol = np.vstack((y_sol, y))
        
        # Update the current depth
        z_depth = z_sol[-1]
    
    print '  Done with outer plume calculations...'
    
    # Build the neighbor matrix
    neighbor = interp1d(z_sol, y_sol.transpose())
    
    return (z_sol, y_sol, neighbor)

def err_check(zi, yi, zo, yo, zi_old, yi_old, zo_old, yo_old, yi_local, 
              yo_local, particles, profile, p):
    """
    Computes the error between two solutions of the state space.
    
    Compares two state space solutions and returns the maximum relative error
    among a suite of error tests that include state space variable magnitudes
    as well as geometric scales (intrusion and peel depths, etc.).
    
    Parameters
    ----------
    zi : ndarray
        Vector of depths for the inner plume solution (m).
    yi : ndarray
        Matrix of inner plume state space variables, each row corresponding
        to a depth in `zi`.
    zo : ndarray
        Vector of depths for the outer plume solution (m).
    yo : ndarray
        Matrix of outer plume state space variables, each row corresponding
        to a depth in `zo`.
    zi_old : ndarray
        Values of `zi` for the previous iteration.
    yi_old : ndarray
        Values of `yi` for the previous iteration.
    zo_old : ndarray
        Values of `zo` for the previous iteration.
    yo_old : ndarray
        Values of `yo` for the previous iteration.
    yi_local : `InnerPlume` object
        Object containing the inner plume state space and methods to extract
        the state space variables
    yo_local : `OuterPlume` object
        Object containing the outer plume state space and methods to extract
        the state space variables
    particles : list of `dispersed_phases.PlumeParticle` objects
        List of `dispersed_phases.PlumeParticle` objects containing the 
        dispersed phase local conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    
    Returns
    -------
    ea : float
        Maximum value of the relative error for the given error tests.
    
    """
    ea = []
    # Volume flux maxima should not change much:
    Qi_ea = np.abs((np.max(yi[:,0]) - np.max(yi_old[:,0])) / np.max(yi[:,0]))
    Qo_ea = np.abs((np.max(-yo[:,0]) - np.max(-yo_old[:,0])) / np.max(-yo[:,0]))
    ea.append(Qi_ea)
    ea.append(Qo_ea)
    
    # Momentum flux maxima should not change much:
    Ji_ea = np.abs((np.max(yi[:,1]) - np.max(yi_old[:,1])) / np.max(yi[:,1]))
    Jo_ea = np.abs((np.max(yo[:,1]) - np.max(yo_old[:,1])) / np.max(yo[:,1]))
    ea.append(Ji_ea)
    ea.append(Jo_ea)
    
    # The height of the inner plume should be stable:
    Hi = zi[0] - zi[-1]
    Hi_old = zi_old[0] - zi_old[-1]
    H_ea = np.abs((Hi - Hi_old) / Hi)
    ea.append(H_ea)
    
    # The volume flux at the top of the inner plume should be stable:
    Qtop_ea = np.abs((yi[-1,0] - yi_old[-1,0]) / yi[-1,0])
    ea.append(Qtop_ea)
    
    # TODO (S. Socolofsky, August 4, 2013): Build in the capability to find
    # and count intrusion layers.  
    ea = np.array(ea)
    
    return np.max(ea)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def particle_from_Q(profile, z0, dbm_particle, yk, Q_N, de, lambda_1, T0=None, 
                    K=1., K_T=1., fdis=1.e-6, t_hyd=0.):
    """
    Create a `dispersed_phases.PlumeParticle` object from volume flux
    
    Returns a `dispered_phases.PlumeParticle` object given the particle 
    attributes and the initial total volume flux at standard conditions.  
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    z0 : float
        Depth of the release point (m)
    dbm_particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Object describing the particle properties and behavior
    yk : ndarray
        Vector of mol fractions of each component of the dispersed phase 
        particle.  If the particle is a `dbm.InsolubleParticle`, then yk 
        should be equal to one.
    Q_N : float
        Volume flux (m^3/s) at standard conditions, defined as 0 deg C and 
        1 bar
    de : float
        Initial diameter (m) of the particle
    lambda_1 : float 
        spreading rate of the dispersed phase in a plume (--)
    T0 : float, default = None
        Initial temperature of the of `dbm` particle object (K).  If None, 
        then T0 is set equal to the ambient temperature.
    K : float, default = 1.
        Mass transfer reduction factor (--).
    K_T : float, default = 1.
        Heat transfer reduction factor (--).
    fdis : float, default = 0.01
        Fraction of the initial total mass (--) remaining when the particle 
        should be considered dissolved.
    t_hyd : float, default = 0.
        Hydrate film formation time (s).  Mass transfer is computed by clean
        bubble methods for t less than t_hyd and by dirty bubble methods
        thereafter.  The default behavior is to assume the particle is dirty
        or hydrate covered from the release.
    
    Returns
    -------
    particle : dispersed_phases.PlumeParticle object
        A `dispersed_phases.PlumeParticle` object containing the initial 
        conditions for the given particle in the simulation
    
    """
    # Convert the flow rate and diameter to the state variables of a 
    # dispersed_phases.PlumeParticle object
    m0, T0, nb0, P, Sa, Ta = dispersed_phases.initial_conditions(
        profile, z0, dbm_particle, yk, Q_N, 1, de, T0)
    
    # Create the particle object
    return dispersed_phases.PlumeParticle(dbm_particle, m0, T0, nb0, lambda_1,
                                          P, Sa, Ta, K, K_T, fdis, t_hyd)

def particle_from_mb0(profile, z0, dbm_particle, yk, mb0, de, lambda_1, 
                      T0=None, K=1., K_T=1., fdis=1.e-6, t_hyd=0.):
    """
    Create a `dispersed_phases.PlumeParticle` object from the mass flux 
    
    Returns a `dispersed_phases.PlumeParticle` object given the particle 
    attributes and the initial total mass flux at the source.
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation
    z0 : float
        Depth of the release point (m)
    dbm_particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Object describing the particle properties and behavior
    yk : ndarray
        Vector of mol fractions of each component of the dispersed phase 
        particle.  If the particle is a `dbm.InsolubleParticle`, then yk 
        should be equal to one.
    mb0 : float
        Mass flux (kg/s) of all particle components together at the release 
        points
    de : float
        Initial diameter (m) of the particle
    lambda_1 : float 
        Spreading rate of the dispersed phase in a plume (--)
    T0 : float, default = None
        Initial temperature of the of `dbm` particle object (K).  If None, 
        then T0 is set equal to the ambient temperature.
    K : float, default = 1.
        Mass transfer reduction factor (--).
    K_T : float, default = 1.
        Heat transfer reduction factor (--).
    fdis : float, default = 1.e-6
        Fraction of the initial total mass (--) remaining when the particle 
        should be considered dissolved.
    t_hyd : float, default = 0.
        Hydrate film formation time (s).  Mass transfer is computed by clean
        bubble methods for t less than t_hyd and by dirty bubble methods
        thereafter.  The default behavior is to assume the particle is dirty
        or hydrate covered from the release.
    
    Returns
    -------
    particle : dispersed_phases.PlumeParticle object
        A `dispersed_phases.PlumeParticle` object containing the initial 
        conditions for the given particle in the simulation
    
    """
    # Convert the mass flux and diameter to the state variables of a 
    # dispersed_phases.PlumeParticle object
    m0, T0, nb0, P, Sa, Ta = dispersed_phases.initial_conditions(
        profile, z0, dbm_particle, yk, mb0, 2, de, T0)
    
    # Create the particle object
    return dispersed_phases.PlumeParticle(dbm_particle, m0, T0, nb0, lambda_1,
                                          P, Sa, Ta, K, K_T, fdis)

def plot_state_space(zi, yi, zo, yo, yi_local, yo_local, particles, profile, 
                     p, fig):
    """
    Plot the state space for interrogation during model convergence
    
    Plot the simple state space variables volume flux, momentum flux, total
    particle mass flux, and dissolved component flux in order to observe 
    changes during each step of model iteration.
    
    Parameters
    ----------
    zi : ndarray
        Vector of depth for the inner plume solution (m).
    yi : ndarray
        Matrix of inner plume solutions, with each row corresponding to a 
        depth in `zi`.
    zo : ndarray
        Vector of depth for the outer plume solution (m).
    yo : ndarray
        Matrix of outer plume solutions, with each row corresponding to a 
        depth in `zo`.
    yi_local : `InnerPlume` object
        Inner plume object for extracting variables from the state space.
    yo_local : `OuterPlume` object
        Outer plume object for extracting variables from the state space.
    particles : list of `dispersed_phases.PlumeParticle` objects
        List of `dispersed_phases.PlumeParticle` objects containing the 
        dispersed phase local conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    fig : int
        Figure number.  Number of the figure window in which to draw the plot.
    
    """
    # Extract the inner plume variables to plot
    Qi = yi[:,0]
    Ji = yi[:,1]
    idx = 4
    Mp = np.zeros((zi.shape[0], yi_local.np))
    for i in range(yi_local.np):
        if particles[i].particle.issoluble:
            Mp[:,i] = np.sum(yi[:, idx:idx + len(particles[i].composition)], 
                             axis=1)
            idx += len(particles[i].composition)
        else:
            Mp[:,i] = np.sum(yi[:, idx:idx+1], axis=1)
            idx += 1
        idx += 1
    if yi_local.nchems > 0:
        Ci = yi[:,idx:]
    else:
        Ci = np.zeros(zi.shape)
    
    # Extract the outer plume variables to plot
    Qo = yo[:,0]
    Jo = yo[:,1]
    if yo_local.nchems > 0:
        Co = yo[:,4:]
    else:
        Co = np.zeros(zo.shape)
    
    # Plot the figures
    plt.figure(fig)
    plt.clf()
    
    # Volume flux
    ax1 = plt.subplot(221)
    ax1.plot(Qi, zi)
    ax1.plot(Qo, zo)
    ax1.set_xlabel('Q (m^3/s)')
    ax1.set_ylabel('Depth (m)')
    ax1.invert_yaxis()
    ax1.grid(True)
    
    # Momentum flux
    ax2 = plt.subplot(222)
    ax2.plot(Ji, zi)
    ax2.plot(-Jo, zo)
    ax2.set_xlabel('J (m^4/s^2)')
    ax2.set_ylabel('Depth (m)')
    ax2.invert_yaxis()
    ax2.grid(True)
    
    # Mass fluxes
    ax3 = plt.subplot(223)
    ax3.plot(Mp, zi)
    ax3.set_xlabel('M (kg/s)')
    ax3.set_ylabel('Depth (m)')
    ax3.invert_yaxis()
    ax3.grid(True)
    
    # Dissolved component concentrations
    ax4 = plt.subplot(224)
    ax4.plot(Ci, zi)
    ax4.plot(Co, zo)
    ax4.set_xlabel('C (kg/s)')
    ax4.set_ylabel('Depth (m)')
    ax4.invert_yaxis()
    ax4.grid(True)
    
    plt.show()

def plot_all_variables(zi, yi, zo, yo, yi_local, yo_local, particles, 
                       profile, p, fig):
    """
    Plot the complete variable suite for model validation and interpretation
    
    Plot a complete suite of state and derived variables along with some 
    key model parameters and environmental data.  These plots provide both
    validation that model calculations are within expected ranges and a clear
    presentation of the results to aid in model interpretation.
    
    Parameters
    ----------
    zi : ndarray
        Vector of depth for the inner plume solution (m).
    yi : ndarray
        Matrix of inner plume solutions, with each row corresponding to a 
        depth in `zi`.
    zo : ndarray
        Vector of depth for the outer plume solution (m).
    yo : ndarray
        Matrix of outer plume solutions, with each row corresponding to a 
        depth in `zo`.
    yi_local : `InnerPlume` object
        Inner plume object for extracting variables from the state space.
    yo_local : `OuterPlume` object
        Outer plume object for extracting variables from the state space.
    particles : list of `dispersed_phases.PlumeParticle` objects
        List of `dispersed_phases.PlumeParticle` objects containing the 
        dispersed phase local conditions and behavior.
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    p : `ModelParams` object
        Object containing the fixed model parameters for the stratified 
        plume model.
    fig : int
        Figure number.  Number of the figure window in which to draw the plot.
    
    
    """
    # Calculate some counting variables to set up memory for storing the 
    # results before plotting
    n_inner = zi.shape[0]
    n_outer = zo.shape[0]
    n_part = yi_local.np
    pchems = 1
    for i in range(n_part):
        if len(particles[i].composition) > pchems:
            pchems = len(particles[i].composition)
    nchems = yi_local.nchems
    if nchems == 0:
        nchems = 1
    
    # Initialize space to store the inner plume solution
    Qi = np.zeros(n_inner)
    Ji = np.zeros(n_inner)
    Si = np.zeros(n_inner)
    Hi = np.zeros(n_inner)
    Mpf = np.zeros((n_inner, n_part, pchems))
    Hp = np.zeros((n_inner, n_part))
    Ci = np.zeros((n_inner, nchems))
    ui = np.zeros(n_inner)
    bi = np.zeros(n_inner)
    si = np.zeros(n_inner)
    Ti = np.zeros(n_inner)
    ci = np.zeros((n_inner, nchems))
    rho_i = np.zeros(n_inner)
    xi = np.zeros((n_inner, n_part))
    fb = np.zeros((n_inner, n_part))
    Xi = np.zeros(n_inner)
    Fb = np.zeros(n_inner)
    Ep = np.zeros(n_inner)
    
    # Initialize space to store the particle solution
    Mp = np.zeros((n_inner, n_part, pchems))
    Tp = np.zeros((n_inner, n_part))
    de = np.zeros((n_inner, n_part))
    us = np.zeros((n_inner, n_part))
    rho_p = np.zeros((n_inner, n_part))
    A = np.zeros((n_inner, n_part))
    Cs = np.zeros((n_inner, n_part, pchems))
    beta = np.zeros((n_inner, n_part, pchems))
    beta_T = np.zeros((n_inner, n_part))
    
    # Get the ambient data on the same grid as the inner plume
    Ta = np.zeros(n_inner)
    Sa = np.zeros(n_inner)
    P = np.zeros(n_inner)
    rho_a = np.zeros(n_inner)
    ca = np.zeros((n_inner, nchems))
    
    # Get the inner plume solution at each calculation level
    for i in range(n_inner):
        yi_local.update(zi[i], yi[i,:], particles, profile, p)
        Qi[i] = yi_local.Q
        Ji[i] = yi_local.J
        Si[i] = yi_local.S
        Hi[i] = yi_local.H
        for j in range(n_part):
            Mpf[i,j,0:len(yi_local.M_p[j])] = yi_local.M_p[j][:]
            Mp[i,j,0:len(particles[j].m)] = particles[j].m[:]
            Tp[i,j] = particles[j].T
            de[i,j] = particles[j].diameter(Mp[i,j,0:len(particles[j].m)], 
                                            Tp[i,j], yi_local.P, yi_local.Sa, 
                                            yi_local.Ta)
            us[i,j] = particles[j].us
            rho_p[i,j] = particles[j].rho_p
            A[i,j] = particles[j].A
            Cs_local = particles[j].Cs[:]
            beta_local = particles[j].beta[:]
            if len(Cs_local) > 0:
                Cs[i,j,0:len(Cs_local)] = Cs_local
                beta[i,j,0:len(beta_local)] = beta_local
            beta_T[i,j] = particles[j].beta_T
        Hp[i,:] = yi_local.H_p
        if len(yi_local.C) > 0:
            Ci[i,:] = yi_local.C[:]
        ui[i] = yi_local.u
        bi[i] = yi_local.b
        si[i] = yi_local.s
        Ti[i] = yi_local.T
        if len(yi_local.c) > 0:
            ci[i] = yi_local.c[:]
        rho_i[i] = yi_local.rho
        xi[i,:] = yi_local.xi[:]
        fb[i,:] = yi_local.fb[:]
        Xi[i] = yi_local.Xi
        Fb[i] = yi_local.Fb
        Ep[i] = yi_local.Ep
        Ta[i] = yi_local.Ta
        Sa[i] = yi_local.Sa
        P[i] = yi_local.P
        rho_a[i] = yi_local.rho_a
        if len(yi_local.ca) > 0:
            ca[i,:] = yi_local.ca
    
    # Create an interpolator for the inner plume
    neighbor = interp1d(np.flipud(zi), np.flipud(yi).transpose())
    
    # Initialize space to store the outer plume solution
    Qo = np.zeros(n_outer)
    Jo = np.zeros(n_outer)
    So = np.zeros(n_outer)
    Ho = np.zeros(n_outer)
    Co = np.zeros((n_outer, nchems))
    uo = np.zeros(n_outer)
    bo = np.zeros(n_outer)
    so = np.zeros(n_outer)
    To = np.zeros(n_outer)
    co = np.zeros((n_outer, nchems))
    rho_o = np.zeros(n_outer)
    
    # Get the inner plume solution at each calculation level
    for i in range(n_outer):
        try:
            yi_local.update(zo[i], neighbor(zo[i]), particles, profile, p)
            yo_local.update(zo[i], yo[i,:], profile, p, yi_local.b)
        except ValueError:
            # Above or below an inner plume segment; set bi = 0
            yo_local.update(zo[i], yo[i,:], profile, p, 0.)
        Qo[i] = yo_local.Q 
        Jo[i] = yo_local.J
        So[i] = yo_local.S
        Ho[i] = yo_local.H
        if len(yo_local.C) > 0:
            Co[i,:] = yo_local.C[:]
        uo[i] = yo_local.u
        bo[i] = yo_local.b
        so[i] = yo_local.s
        To[i] = yo_local.T
        if len(yo_local.c) > 0:
            co[i] = yo_local.c[:]
        rho_o[i] = yo_local.rho
    
    # Figure -----------------------------------------------------------------
    plt.figure(fig)
    plt.clf()
    
    # Volume flux
    ax1 = plt.subplot(221)
    ax1.plot(Qi, zi)
    ax1.plot(Qo, zo)
    ax1.set_xlabel('Q (m^3/s)')
    ax1.set_ylabel('Depth (m)')
    ax1.locator_params(tight=True, nbins=6)
    ax1.invert_yaxis()
    ax1.grid(True)
    
    # Momentum flux
    ax2 = plt.subplot(222)
    ax2.plot(Ji, zi)
    ax2.plot(-Jo, zo)
    ax2.set_xlabel('J (m^4/s^2)')
    ax2.locator_params(tight=True, nbins=6)
    ax2.invert_yaxis()
    ax2.grid(True)
    
    # Salinity flux
    ax3 = plt.subplot(223)
    ax3.plot(Si, zi)
    ax3.plot(So, zo)
    ax3.set_xlabel('S (psu m^3/s)')
    ax3.set_ylabel('Depth (m)')
    ax3.locator_params(tight=True, nbins=6)
    ax3.invert_yaxis()
    ax3.grid(True)
    
    # Heat flux
    ax4 = plt.subplot(224)
    ax4.plot(Hi/1.e6, zi)
    ax4.plot(Ho/1.e6, zo)
    ax4.set_xlabel('H (MJ/s)')
    ax4.locator_params(tight=True, nbins=6)
    ax4.invert_yaxis()
    ax4.grid(True)
    
    plt.show()
    fig += 1
    
    # Figure -----------------------------------------------------------------
    plt.figure(fig)
    plt.clf()
    
    # Determine how many rows and columns of plots to make
    nsp = np.floor(np.sqrt(nchems))
    if np.sqrt(nchems) > nsp:
        nsp += 1
    
    # Plot each dissolved flux separately
    for i in range(nchems):
        ax = plt.subplot(nsp, nsp, i+1)
        ax.plot(Ci[:,i], zi)
        ax.plot(Co[:,i], zo)
        try:
            ax.set_xlabel(yi_local.chem_names[i] + ' flux (kg/s)')
        except IndexError:
            ax.set_xlabel('inert' + ' flux (kg/s)')
        if i/nsp - np.floor(i/nsp) == 0:
            ax.set_ylabel('Depth (m)')
        ax.locator_params(tight=True, nbins=6)
        ax.invert_yaxis()
        ax.grid(True)
    
    plt.show()
    fig += 1
    
    # Figure -----------------------------------------------------------------
    for i in range(n_part):
        plt.figure(fig)
        plt.clf()
        
        # Particle mass flux
        ax1 = plt.subplot(121)
        ax1.plot(Mpf[:,i,:], zi)
        ax1.set_xlabel('M_pf (kg/s)')
        ax1.set_ylabel('Depth (m)')
        ax1.locator_params(tight=True, nbins=4)
        ax1.invert_yaxis()
        ax1.grid(True)
        
        # Particle heat flux
        ax2 = plt.subplot(122)
        ax2.plot(Hp[:,i]/1.e6, zi)
        ax2.set_xlabel('Hf (MJ/s)')
        ax2.locator_params(tight=True, nbins=4)
        ax2.invert_yaxis()
        ax2.grid(True)
        
        plt.show()
        fig += 1
    
    # Figure -----------------------------------------------------------------
    plt.figure(fig)
    plt.clf()
    
    # Velocity
    ax1 = plt.subplot(221)
    ax1.plot(ui, zi)
    ax1.plot(uo, zo)
    ax1.set_xlabel('u (m/s)')
    ax1.set_ylabel('Depth (m)')
    ax1.locator_params(tight=True, nbins=6)
    ax1.invert_yaxis()
    ax1.grid(True)
    
    # Half-width
    ax2 = plt.subplot(222)
    ax2.plot(bi, zi)
    ax2.plot(bo, zo)
    ax2.set_xlabel('b (m)')
    ax2.locator_params(tight=True, nbins=6)
    ax2.invert_yaxis()
    ax2.grid(True)
    b_max = 1.25 * (bo[0] + 0.1 * (zo[-1] - zo[0]))
    ax2.set_xlim([0, b_max])
    
    # Salinity
    ax3 = plt.subplot(223)
    ax3.plot(si, zi)
    ax3.plot(so, zo)
    ax3.plot(Sa, zi)
    ax3.set_xlabel('s (psu)')
    ax3.set_ylabel('Depth (m)')
    ax3.locator_params(tight=True, nbins=6)
    ax3.invert_yaxis()
    ax3.grid(True)
    
    # Temperature
    ax4 = plt.subplot(224)
    ax4.plot(Ti - 273.15, zi)
    ax4.plot(To - 273.15, zo)
    ax4.plot(Ta - 273.15, zi)
    ax4.set_xlabel('T (deg C)')
    ax4.locator_params(tight=True, nbins=6)
    ax4.invert_yaxis()
    ax4.grid(True)    
    
    plt.show()
    fig += 1
    
    # Figure -----------------------------------------------------------------
    plt.figure(fig)
    plt.clf()
    plt.ticklabel_format(useOffset=False, axis='x')
    
    ax1 = plt.subplot(111)
    ax1.plot(rho_i, zi)
    ax1.plot(rho_o, zo)
    ax1.plot(rho_a, zi)
    ax1.set_xlabel('rho (kg/m^3)')
    ax1.set_ylabel('Depth (m)')
    ax1.locator_params(tight=True, nbins=6)
    ax1.invert_yaxis()
    ax1.grid(True)
    
    plt.show()
    fig += 1
    
    # Figure -----------------------------------------------------------------
    plt.figure(fig)
    plt.clf()
    
    # Determine how many rows and columns of plots to make
    nsp = np.floor(np.sqrt(nchems))
    if np.sqrt(nchems) > nsp:
        nsp += 1
    
    # Plot each dissolved flux separately
    for i in range(nchems):
        ax = plt.subplot(nsp, nsp, i+1)
        plt.ticklabel_format(useOffset=False, axis='x')
        ax.plot(ci[:,i], zi)
        ax.plot(co[:,i], zo)
        ax.plot(ca[:,i], zi)
        try:
            ax.set_xlabel(yi_local.chem_names[i] + ' concentration (kg/m^3)')
        except IndexError:
            ax.set_xlabel('inert' + ' concentration (kg/m^3)')
        if i/nsp - np.floor(i/nsp) == 0:
            ax.set_ylabel('Depth (m)')
        ax.locator_params(tight=True, nbins=4)
        ax.invert_yaxis()
        ax.grid(True)
    
    plt.show()
    fig += 1
    
    # Figure -----------------------------------------------------------------
    plt.figure(fig)
    plt.clf()
    
    # Velocity
    ax1 = plt.subplot(221)
    ax1.plot(xi, zi)
    ax1.plot(Xi, zi, 'k-', linewidth=2)
    ax1.set_xlabel('xi (--)')
    ax1.set_ylabel('Depth (m)')
    ax1.locator_params(tight=True, nbins=6)
    ax1.invert_yaxis()
    ax1.grid(True)
    
    # Half-width
    ax2 = plt.subplot(222)
    ax2.plot(fb, zi)
    ax2.plot(Fb, zi, 'k-', linewidth=2)
    ax2.set_xlabel('fb (kg/m^3)')
    ax2.locator_params(tight=True, nbins=6)
    ax2.invert_yaxis()
    ax2.grid(True)
    
    # Peeling flux
    ax3 = plt.subplot(223)
    ax3.plot(Ep, zi)
    ax3.set_xlabel('Ep (m^2/s)')
    ax3.locator_params(tight=True, nbins=6)
    ax3.invert_yaxis()
    ax3.grid(True)
    
    plt.show()
    fig += 1
    
    # Figure -----------------------------------------------------------------
    for i in range(n_part):
        plt.figure(fig)
        plt.clf()
        
        # Diameter
        ax1 = plt.subplot(331)
        ax1.plot(de[:,i]*1000., zi)
        ax1.set_xlabel('de (mm)')
        ax1.set_ylabel('Depth (m)')
        ax1.locator_params(tight=True, nbins=6)
        ax1.invert_yaxis()
        ax1.grid(True)
        
        # Slip velocity
        ax2 = plt.subplot(332)
        ax2.plot(us[:,i]*100., zi)
        ax2.set_xlabel('us (cm)')
        ax2.locator_params(tight=True, nbins=6)
        ax2.invert_yaxis()
        ax2.grid(True)
        
        # Surface area
        ax3 = plt.subplot(333)
        ax3.plot(A[:,i]*100.**2, zi)
        ax3.set_xlabel('A (cm^2)')
        ax3.locator_params(tight=True, nbins=6)
        ax3.invert_yaxis()
        ax3.grid(True)
        
        # Temperature
        ax4 = plt.subplot(334)
        ax4.plot(Tp[:,i] - 273.15, zi)
        ax4.plot(Ti - 273.15, zi)
        ax4.set_xlabel('Tp (deg C)')
        ax4.set_ylabel('Depth (m)')
        ax4.locator_params(tight=True, nbins=6)
        ax4.invert_yaxis()
        ax4.grid(True)
        
        # Density
        ax5 = plt.subplot(335)
        ax5.plot(rho_p[:,i], zi)
        ax5.set_xlabel('rho_p (kg/m^3)')
        ax5.locator_params(tight=True, nbins=6)
        ax5.invert_yaxis()
        ax5.grid(True)
        
        # Beta_T
        ax6 = plt.subplot(336)
        ax6.plot(beta_T[:,i], zi)
        ax6.set_xlabel('beta_T (m/s)')
        ax6.locator_params(tight=True, nbins=6)
        ax6.invert_yaxis()
        ax6.grid(True)
        
        # Mp
        ax7 = plt.subplot(337)
        ax7.plot(Mp[:,i,:]/1.e6, zi)
        ax7.set_xlabel('mp (mg)')
        ax7.set_ylabel('Depth (m)')
        ax7.locator_params(tight=True, nbins=6)
        ax7.invert_yaxis()
        ax7.grid(True)
        
        # Cs
        ax8 = plt.subplot(338)
        ax8.plot(Cs[:,i,:], zi)
        ax8.set_xlabel('Cs (kg/m^3)')
        ax8.locator_params(tight=True, nbins=6)
        ax8.invert_yaxis()
        ax8.grid(True)
        
        # Beta
        ax9 = plt.subplot(339)
        ax9.plot(beta[:,i,:], zi)
        ax9.set_xlabel('beta (m/s)')
        ax9.locator_params(tight=True, nbins=6)
        ax9.invert_yaxis()
        ax9.grid(True)
        
        plt.show()
        fig += 1

