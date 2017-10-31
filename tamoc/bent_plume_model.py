"""
Bent Plume Model
================

Simulate a buoyant plume in crossflowing ambient conditions

This module defines the classes, methods, and functions necessary to simulate
the buoyant plume behavior in crossflowing ambient conditions, where the
intrusion layer is not expected to interact with the rising stage of the
plume. The ambient water properties are provided through an `ambient.Profile`
class object, which contains a netCDF4-classic dataset of CTD data and the
needed interpolation methods. The `dbm` class objects `dbm.FluidParticle` and
`dbm.InsolubleParticle` report the properties of the dispersed phase during
the simulation, and these methods are provided to the model through the 
objects defined in `dispersed_phases`.  

This module manages the setup, simulation, and post-processing for the model.
The numerical solution is contained in the `lpm` module.

Notes 
----- 
This model is a Lagrangian plume integral model following the approach in
Lee and Cheung (1990) for single-phase plumes, updated using the shear 
entrainment formulation in Jirka (2004), and adapted to multiphase plumes
following the methods of Johansen (2000, 2003) and Zheng and Yapa (1997).
Several modifications are made to make the model consistent with the approach
in Socolofsky et al. (2008) and to match the available validation data.

The model can run as a single-phase or multi-phase plume.  A single-phase 
plume simply has an empty `particles` list.

See Also
--------
`stratified_plume_model` : Predicts the plume solution for quiescent ambient
    conditions or weak crossflows, where the intrusion (outer plume) 
    interacts with the upward rising plume in a double-plume integral model
    approach.  Such a situation is handeled properly in the 
    `stratified_plume_model` and would violate the assumption of non-
    iteracting Lagrangian plume elements as required in this module.

`single_bubble_model` : Tracks the trajectory of a single bubble, drop or 
    particle through the water column.  The numerical solution used here, 
    including the various object types and their functionality, follows the
    pattern in the `single_bubble_model`.  The main difference is the more
    complex state space and governing equations.

"""
# S. Socolofsky, November 2014, Texas A&M University <socolofs@tamu.edu>.

from tamoc import model_share
from tamoc import ambient
from tamoc import seawater
from tamoc import single_bubble_model
from tamoc import dispersed_phases
from tamoc import lmp

from netCDF4 import Dataset
from datetime import datetime

import numpy as np
from numpy.linalg import inv
from scipy.optimize import fsolve
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Main Model object
# ----------------------------------------------------------------------------

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
        loaded into the `Model` object memory
    p : `ModelParams`
        Container for the fixed model parameters
    sim_stored : bool
        Flag indicating whether or not a simulation result is stored in the 
        object memory
    X : ndarray
        Release location (x, y, z) in (m)
    D : float
        Diameter for the equivalent circular cross-section of the release (m)
    Vj : float
        Scalar value of the magnitude of the discharge velocity for continuous
        phase fluid in the discharge.  This variable should be 0. or None for 
        a pure multiphase discharge.
    phi_0 : float
        Vertical angle from the horizontal for the discharge orientation 
        (rad in range +/- pi/2)
    theta_0 : float
        Horizontal angle from the x-axis for the discharge orientation.  The
        x-axis is taken in the direction of the ambient current.  (rad in 
        range 0 to 2 pi)
    Sj : float
        Salinity of the continuous phase fluid in the discharge (psu)
    Tj : float
        Temperature of the continuous phase fluid in the discharge (T)
    cj : ndarray
        Concentration of passive tracers in the discharge (user-defined)
    tracers : string list
        List of passive tracers in the discharge.  These can be chemicals 
        present in the ambient `profile` data, and if so, entrainment of these
        chemicals will change the concentrations computed for these tracers.
        However, none of these concentrations are used in the dissolution of 
        the dispersed phase.  Hence, `tracers` should not contain any 
        chemicals present in the dispersed phase particles.
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the 
        simulation
    track : bool
        Flag indicating whether or not to track the particle through 
        the water column using the `single_bubble_model`.
    dt_max : float
        Maximum step size to take in the storage of the simulation solution
        (s)
    sd_max : float
        Maximum number of orifice diameters to compute the solution along
        the plume centerline (m/m)
    K_T0 : ndarray
        Array of heat transfer reduction factors for the particle objects
        which is used to restart heat transfer after the simulation ends
        since during simulation, heat transfer is turned off.
    chem_names : string list
        List of chemical parameters to track for the dissolution.  Only the 
        parameters in this list will be used to set background concentration
        for the dissolution, and the concentrations of these parameters are 
        computed separately from those listed in `tracers` or inputed from
        the discharge through `cj`.
    q_local : `LagElement` object
        Object that translates the `Model` state space `t` and `q` into the 
        comprehensive list of derived variables.
    t : ndarray
        Array of times computed in the solution (s)
    q : ndarray
        Array of state space values computed in the solution
    
    See Also
    --------
    simulate, save_sim, save_txt, load_sim, plot_state_space, 
    plot_all_variables
    
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
            
            # Set the model parameters that the user cannot adjust
            self.p = ModelParams(self.profile)
            
            # Indicate that the simulation has not yet been conducted
            self.sim_stored = False
    
    def simulate(self, X, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers, 
                 particles=[], track=False, dt_max=60., sd_max=350.):
        """
        Simulate the plume dynamics from given initial conditions
        
        Simulate the buoyant plume using a Lagrangian plume integral model
        approach until the plume reaches the surface, the integration 
        exceeds the given s/D (`sd_max`), or the intrusion reaches a point of 
        neutral buoyancy.
        
        Parameters
        ----------
        X : ndarray
            Release location (x, y, z) in (m)
        D : float
            Diameter for the equivalent circular cross-section of the release 
            (m)
        Vj : float
            Scalar value of the magnitude of the discharge velocity for 
            continuous phase fluid in the discharge.  This variable should be 
            0. or None for a pure multiphase discharge.
        phi_0 : float
            Vertical angle from the horizontal for the discharge orientation 
            (rad in range +/- pi/2)
        theta_0 : float
            Horizontal angle from the x-axis for the discharge orientation.  
            The x-axis is taken in the direction of the ambient current.  
            (rad in range 0 to 2 pi)
        Sj : float
            Salinity of the continuous phase fluid in the discharge (psu)
        Tj : float
            Temperature of the continuous phase fluid in the discharge (T)
        cj : ndarray
            Concentration of passive tracers in the discharge (user-defined)
        tracers : string list
            List of passive tracers in the discharge.  These can be chemicals 
            present in the ambient `profile` data, and if so, entrainment of 
            these chemicals will change the concentrations computed for these 
            tracers.  However, none of these concentrations are used in the 
            dissolution of the dispersed phase.  Hence, `tracers` should not 
            contain any chemicals present in the dispersed phase particles.
        particles : list of `Particle` objects
            List of `Particle` objects describing each dispersed phase in the 
            simulation
        track : bool
            Flag indicating whether or not to track the particle through 
            the water column using the `single_bubble_model`.
        dt_max : float
            Maximum step size to take in the storage of the simulation 
            solution (s)
        sd_max : float
            Maximum number of orifice diameters to compute the solution along
            the plume centerline (m/m)
        
        """
        # Make sure the position is an array
        if not isinstance(X, np.ndarray):
            if not isinstance(X, list):
                # Assume user specified the depth only
                X = np.array([0., 0., X])
            else:
                X = np.array(X)
        
        # Make sure the tracer data are in an array
        if not isinstance(cj, np.ndarray):
            if not isinstance(cj, list):
                cj = np.array([cj])
            else:
                cj = np.array(cj)
        if not isinstance(tracers, list):
            tracers = [tracers]
        
        # Store the input parameters
        self.X = X
        self.D = D
        self.Vj = Vj
        self.phi_0 = phi_0
        self.theta_0 = theta_0
        self.Sj = Sj
        self.Tj = Tj
        self.cj = cj
        self.tracers = tracers
        self.particles = particles
        self.track = track
        self.dt_max = dt_max
        self.sd_max = sd_max
        self.K_T0 = np.array([self.particles[i].K_T for i in 
                              range(len(self.particles))])
        
        # Create the initial state space from the given input variables
        t0, q0, self.chem_names = lmp.main_ic(self.profile, 
            self.particles, self.X, self.D, self.Vj, self.phi_0, 
            self.theta_0, self.Sj, self.Tj, self.cj, self.tracers, self.p)
        
        # Store the initial conditions in a Lagrangian element object
        self.q_local = LagElement(t0, q0, D, self.profile, self.p, 
                       self.particles, self.tracers, self.chem_names)
        
        # Compute the buoyant jet trajectory
        print '\n-- TEXAS A&M OIL-SPILL CALCULATOR (TAMOC) --'
        print '-- Bent Plume Model                       --\n'
        self.t, self.q, = lmp.calculate(t0, q0, self.q_local, self.profile, 
            self.p, self.particles, lmp.derivs, self.dt_max, self.sd_max)
        
        # Track the particles
        if self.track:
            for i in range(len(self.particles)):
                if particles[i].integrate is False and particles[i].z > 0.:
                    print '\nTracking Particle %d of %d:' % \
                        (i+1, len(self.particles))
                    particles[i].run_sbm(self.profile)
        
        # Update the status of the solution
        self.sim_stored = True
        
        # Update the status of the particles
        for i in range(len(self.particles)):
            self.particles[i].sim_stored = True
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
            File name of the netCDF file to write
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
        title = 'Simulation results for the TAMOC Bent Plume Model'
        nc = model_share.tamoc_nc_file(fname, title, profile_path, 
            profile_info)
        
        # Create variables for the dimensions
        t = nc.createDimension('t', None)
        p = nc.createDimension('profile', 1)
        ns = nc.createDimension('ns', len(self.q_local.q0))
        params = nc.createDimension('params', 1)
        
        # Save the names of the chemicals in the tracers and particle objects
        nc.tracers = ' '.join(self.tracers)
        nc.chem_names = ' '.join(self.chem_names)
        
        # Create variables to store the initial conditions
        x0 = nc.createVariable('x0', 'f8', ('params',))
        x0.long_name = 'Initial value of the x-coordinate'
        x0.standard_name = 'x0'
        x0.units = 'm'
        
        y0 = nc.createVariable('y0', 'f8', ('params',))
        y0.long_name = 'Initial value of the y-coordinate'
        y0.standard_name = 'y0'
        y0.units = 'm'
        
        z0 = nc.createVariable('z0', 'f8', ('params',))
        z0.long_name = 'Initial depth below the water surface'
        z0.standard_name = 'depth'
        z0.units = 'm'
        z0.axis = 'Z'
        z0.positive = 'down'
        
        D = nc.createVariable('D', 'f8', ('params',))
        D.long_name = 'Orifice diameter'
        D.standard_name = 'diameter'
        D.units = 'm'
        
        Vj = nc.createVariable('Vj', 'f8', ('params',))
        Vj.long_name = 'Discharge velocity'
        Vj.standard_name = 'Vj'
        Vj.units = 'm'
        
        phi_0 = nc.createVariable('phi_0', 'f8', ('params',))
        phi_0.long_name = 'Discharge vertical angle to horizontal'
        phi_0.standard_name = 'phi_0'
        phi_0.units = 'rad'
        
        theta_0 = nc.createVariable('theta_0', 'f8', ('params',))
        theta_0.long_name = 'Discharge horizontal angle to x-axis'
        theta_0.standard_name = 'theta_0'
        theta_0.units = 'rad'
        
        Sj = nc.createVariable('Sj', 'f8', ('params',))
        Sj.long_name = 'Discharge salinity'
        Sj.standard_name = 'Sj'
        Sj.units = 'psu'
        
        Tj = nc.createVariable('Tj', 'f8', ('params',))
        Tj.long_name = 'Discharge temperature'
        Tj.standard_name = 'Tj'
        Tj.units = 'K'
        
        cj = nc.createVariable('cj', 'f8', ('params',))
        cj.long_name = 'Discharge tracer concentration'
        cj.standard_name = 'cj'
        cj.units = 'nondimensional'
        
        Ta = nc.createVariable('Ta', 'f8', ('params',))
        Ta.long_name = 'ambient temperature at the release point'
        Ta.standard_name = 'Ta'
        Ta.units = 'K'
        
        Sa = nc.createVariable('Sa', 'f8', ('params',))
        Sa.long_name = 'ambient salinity at the release point'
        Sa.standard_name = 'Sa'
        Sa.units = 'psu'
        
        P = nc.createVariable('P', 'f8', ('params',))
        P.long_name = 'ambient pressure at the release point'
        P.standard_name = 'P'
        P.units = 'Pa'
        
        # Create variables for the simulation setup
        track = nc.createVariable('track', 'i4', ('params',))
        track.long_name = 'SBM Status (0: false, 1: true)'
        track.standard_name = 'track'
        track.units = 'boolean'
        
        dt_max = nc.createVariable('dt_max', 'f8', ('params',))
        dt_max.long_name = 'Simulation maximum duration'
        dt_max.standard_name = 'dt_max'
        dt_max.units = 's'
        
        sd_max = nc.createVariable('sd_max', 'f8', ('params',))
        sd_max.long_name = 'Maximum distance along centerline s/D'
        sd_max.standard_name = 'sd_max'
        sd_max.units = 'nondimensional'
        
        # Create a variable for the independent variable
        t = nc.createVariable('t', 'f8', ('t', 'profile',))
        t.long_name = 'time along the plume centerline'
        t.standard_name = 'time'
        t.units = 's'
        t.axis = 'T'
        t.n_times = len(self.t)
        
        # Create a variable for the model state space
        q = nc.createVariable('q', 'f8', ('t', 'ns',))
        q.long_name = 'Lagranian plume model state space'
        q.standard_name = 'q'
        q.units = 'variable'
        
        # Store the model initial conditions
        x0[0] = self.X[0]
        y0[0] = self.X[1]
        z0[0] = self.X[2]
        D[0] = self.D
        Vj[0] = self.Vj
        phi_0[0] = self.phi_0
        theta_0[0] = self.theta_0
        Sj[0] = self.Sj
        Tj[0] = self.Tj
        cj[0] = self.cj
        Ta[0], Sa[0], P[0] = self.profile.get_values(np.max(self.X[2]), 
            ['temperature', 'salinity', 'pressure'])
        
        # Store the model setup
        if self.track:
            track[0] = 1
        else:
            track[0] = 0
        dt_max[0] = self.dt_max
        sd_max[0] = self.sd_max
        
        # Save the dispersed phase particles
        dispersed_phases.save_particle_to_nc_file(nc, self.chem_names, 
            self.particles, self.K_T0)
        
        # Save the tracked particles if they exist
        for i in range(len(self.particles)):
            if self.particles[i].farfield:
                fname_sbm = fname.split('.nc')[0] + '%3.3d.nc' % i
                self.particles[i].sbm.save_sim(fname_sbm, profile_path, 
                    profile_info)
        
        # Store the plume simulation solution
        t[:,0] = self.t[:]
        for i in range(len(nc.dimensions['ns'])):
            q[:,i] = self.q[:,i]
        
        # Store any single bubble model simulations
        
        
        # Close the netCDF dataset
        nc.close()
    
    def save_txt(self, base_name, profile_path, profile_info):
        """
        Save the state space in ascii text format for exporting
        
        Save the state space (dependent and independent variables) in an 
        ascii text file for exporting to other programs (e.g., Matlab).
        
        Parameters
        ----------
        base_name : str
            Base file name for the output file.  This method will append the
            .txt file extension to the data output and write a second file
            with the header information called base_name_header.txt.  If the
            particles that left the plume were tracked in the farfield, it
            will also save the trajectory of those particles as 
            base_name_nnn.txt (output data) and base_name_nnn_header.txt 
            (header data for far field data).  
        profile_path : str
            String stating the file path to the ambient profile data relative 
            to the directory where `fname` will be saved.  
        profile_info : str
            Single line of text describing the ambient profile data.
        
        See Also
        --------
        save_sim, load_sim, stratified_plume_model.Model.save_txt, 
        single_bubble_model.Model.save_txt
        
        Notes
        -----
        The output will be organized in columns, with each column as follows:
        
            0   : time (s)
            1-n : state space
        
        The header to the output file will give the extact organization of 
        each column of the output data.
        
        These output files are written using the `numpy.savetxt` method.
        
        Note, also, that this only saves the state space solution, and the 
        data saved by this function is inadequate to rebuild the `Model` 
        object by reloading a saved solution.  To have seamless saving and
        loading of `Model` objects, use the `save_sim` and `load_sim` 
        commands.
        
        """
        if self.sim_stored is False:
            print 'No simulation results to store...'
            print 'Saved nothing to text file.\n'
            return
        
        # Create the header string that contains the column descriptions 
        # for the Lagrangian plume state space
        p_list = ['Lagrangian Plume Model ASCII Output File \n']
        p_list.append('Created: ' + datetime.today().isoformat(' ') + '\n\n')
        p_list.append('Simulation based on CTD data in:\n')
        p_list.append(profile_path)
        p_list.append('\n\n')
        p_list.append(profile_info)
        p_list.append('\n\n')
        p_list.append('Column Descriptions:\n')
        p_list.append('    0: time (s)\n')
        p_list.append('    1: mass (kg)\n')
        p_list.append('    2: salinity (psu)\n')
        p_list.append('    3: heat (J)\n')
        p_list.append('    4: x-direction momentum (kg m/s)\n')
        p_list.append('    5: y-direction momentum (kg m/s)\n')
        p_list.append('    6: z-direction momentum (kg m/s)\n')
        p_list.append('    7: relative thickness h/V (s)\n')
        p_list.append('    8: x-coordinate (m)\n')
        p_list.append('    9: y-coordinate (m)\n')
        p_list.append('    10: z-coordinate (m)\n')
        p_list.append('    11: distance along plume centerline (m)\n')
        idx = 11
        for i in range(len(self.particles)):
            for j in range(len(self.particles[i].m0)):
                idx += 1
                p_list.append(
                    '    %d: Total mass flux of %s in particle %d (kg/s)\n' %
                    (idx, self.particles[i].composition[j], i))
            idx += 1
            p_list.append('    %d: Total heat flux of particle %d (J/s)\n' %
                          (idx, i))
            idx += 1
            p_list.append('    %d: Time since release of particle %d (s)\n' %
                          (idx, i))
            idx += 1
            p_list.append('    %d: Lambda coordinate of particle %d (m)\n' %
                          (idx, i))
            idx += 1
            p_list.append('    %d: Eta coordinate of particle %d (m)\n' %
                          (idx, i))
            idx += 1
            p_list.append('    %d: Csi coordinate of particle %d (m)\n' %
                          (idx, i))            
        for i in range(len(self.chem_names)):
            idx += 1
            p_list.append('    %d: Mass of dissolved %s (kg)\n' % 
                          (idx, self.chem_names[i]))
        for i in range(len(self.tracers)):
            idx += 1
            p_list.append('    %d: Mass of %s in (kg)\n' % 
                          (idx, self.tracers[i]))
        header = ''.join(p_list)
        
        # Assemble and write the state space data
        data = np.hstack((np.atleast_2d(self.t).transpose(), self.q))
        np.savetxt(base_name + '.txt', data)
        with open(base_name + '_header.txt', 'w') as dat_file:
            dat_file.write(header)
        
        # Save the tracked particles if they exist
        for i in range(len(self.particles)):
            if self.particles[i].farfield:
                fname_sbm = base_name + '%3.3d' % i
                self.particles[i].sbm.save_txt(fname_sbm, profile_path, 
                    profile_info)
    
    def load_sim(self, fname):
        """
        Load in a saved simulation result file for post-processing
        
        Load in a saved simulation result file and rebuild the `Model`
        object attributes.  The input file must be in netCDF4-classic data 
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
        
        # Get the release location of the plume
        self.X = np.zeros(3)
        self.X[0] = nc.variables['x0'][0]
        self.X[1] = nc.variables['y0'][0]
        self.X[2] = nc.variables['z0'][0]
        
        # Create the Particle objects
        self.particles, self.chem_names = \
            dispersed_phases.load_particle_from_nc_file(nc)
        
        # Extract the remaining model constants
        self.D = nc.variables['D'][0]
        self.Vj = nc.variables['Vj'][0]
        self.phi_0 = nc.variables['phi_0'][0]
        self.theta_0 = nc.variables['theta_0'][0]
        self.Sj = nc.variables['Sj'][0]
        self.Tj = nc.variables['Tj'][0]
        self.cj = nc.variables['cj'][0]
        if nc.variables['track'][0] == 1:
            self.track = True
        else:
            self.track = False
        self.dt_max = nc.variables['dt_max'][0]
        self.sd_max = nc.variables['sd_max'][0]
        
        # Compute the dimensions of the arrayed data
        ns = len(nc.dimensions['ns'])
        nt = nc.variables['t'].n_times
        
        # Extract the arrayed data
        self.tracers = nc.tracers.split()
        self.K_T0 = np.array([self.particles[i].K_T for i in 
                              range(len(self.particles))])
        self.t = np.zeros(nt)
        self.t[:] = nc.variables['t'][0:nt,0]
        self.q = np.zeros((nt, ns))
        for i in range(ns):
            self.q[:,i] = nc.variables['q'][0:nt,i]
        
        # Create the local Lagrangian plume element
        self.q_local = LagElement(self.t[0], self.q[0,:], self.D, 
                       self.profile, self.p, self.particles, self.tracers, 
                       self.chem_names)
        
        # Load in any farfield tracking results
        for i in range(len(self.particles)):
            if self.particles[i].farfield:
                fname_sbm = fname.split('.nc')[0] + '%3.3d.nc' % i
                self.particles[i].sbm = \
                    single_bubble_model.Model(simfile=fname_sbm)
        
        # Close the netCDF dataset
        nc.close()
        self.sim_stored = True
    
    def plot_state_space(self, fig):
        """
        Plot the simulation state space
        
        Plot the standard set of state space variables used to evaluate 
        the quality of the model solution
        
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
        plot_state_space(self.t, self.q, self.q_local, self.profile, self.p, 
            self.particles, fig)
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
        plot_all_variables(self.t, self.q, self.q_local, self.profile, 
            self.p, self.particles, self.track, fig)
        print 'Done.\n'


# ----------------------------------------------------------------------------
# Model Parameters objects
# ----------------------------------------------------------------------------

class ModelParams(single_bubble_model.ModelParams):
    """
    Fixed model parameters for the bent plume model
    
    This class stores the set of model parameters that should not be adjusted
    by the user and that are needed by the bent plume model.
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        The ambient CTD object used by the simulation.
    
    Attributes
    ----------
    alpha_j : float
        Jet shear entrainment coefficient.
    alpha_Fr : float
        Plume entrainment coefficient in Froude-number expression.
    gamma : float
        Momentum amplification factor
    Fr_0 : float
        Initial plume Froude number for the Wuest et al. (1992) multiphase
        plume initial conditions
    rho_r : float
        Reference density (kg/m^3) evaluated at mid-depth of the water body.
    g : float
        Acceleration of gravity (m/s^2)
    Ru : float
        Ideal gas constant (J/mol/K)
    
    Notes
    -----
    This object inherits all of the parameters of the 
    `single_bubble_model.ModelParams` object.
    
    """
    def __init__(self, profile):
        super(ModelParams, self).__init__(profile)
        
        # Set the model parameters to the values in Lee and Cheung (1990)
        self.alpha_j = 0.057
        self.alpha_Fr = 0.544
        self.gamma = 1.10
        
        # Set some of the multiphase plume model parameters
        self.Fr_0 = 1.6


# ----------------------------------------------------------------------------
# Particle object that handles tracking and exiting the plume
# ----------------------------------------------------------------------------

class Particle(dispersed_phases.PlumeParticle):
    """
    Special model properties for tracking inside a Lagrangian plume object
    
    This new `Particle` class is needed to allow dispersed phase particles to 
    be tracked within the Lagrangian plume element during the solution and 
    to exit the plume at the right time.  
    
    This object inherits the `dispersed_phases.PlumeParticle` object and 
    adds functionality for three-dimensional positioning and particle 
    tracking.  All behavior not associated with tracking is identical to 
    that in the `dispersed_phases.PlumeParticle` object.  Moreover, this
    object can be used in a `stratified_plume_model.Model` simulation.
    
    Parameters
    ----------
    x : float
        Initial position of the particle in the x-direction (m)
    y : float
        Initial position of the particle in the y-direction (m)
    z : float
        Initial position of the particle in the z-direction (m)
    dbm_particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Object describing the particle properties and behavior
    m0 : ndarray
        Initial masses of one particle for the components of the 
        `dbm_particle` object (kg)
    T0 : float
        Initial temperature of the of `dbm` particle object (K)
    nb0 : float
        Initial number flux of particles at the release (#/s)
    lambda_1 : float 
        spreading rate of the dispersed phase in a plume (--)
    P : float
        Local pressure (Pa)
    Sa : float
        Local salinity surrounding the particle (psu)
    Ta : float
        Local temperature surrounding the particle (K)
    K : float, default = 1.
        Mass transfer reduction factor (--).
    K_T : float, default = 1.
        Heat transfer reduction factor (--).
    fdis : float, default = 1.e-6
        Fraction (--) of the initial mass of each component of the mixture
        when that component should be considered totally dissolved.
    
    Attributes
    ----------
    particle : `dbm.FluidParticle` or `dbm.InsolubleParticle` object
        Stores the `dbm_particle` object passed to at creation.
    composition : str list
        Copy of the `composition` attribute of the `dbm_particle` object.
    m0 : ndarray
        Initial masses (kg) of one particle for the particle components
    T0 : float
        Initial temperature (K) of the particle
    cp : float
        Heat capacity at constant pressure (J/(kg K)) of the particle.
    K : float
        Mass transfer reduction factor (--)
    K_T : float
        Heat transfer reduction factor (--)
    fdis : float
        Fraction of initial mass remaining as total dissolution (--)
    diss_indices : ndarray bool
        Indices of m0 that are non-zero.
    t_hyd : float
        Formation time for a hydrate skin (s)
    nb0 : float
        Initial number flux of particles at the release (#/s)
    nbe : float
        Number of particles associated with a Lagrangian element (#).  This 
        number with the mass per particle sets the total mass of particles
        inside the Lagrangian element at any given time.  This value is set
        by `lmp.bent_plume_ic`.
    lambda_1 : float 
        Spreading rate of the dispersed phase in a plume (--)
    m : ndarray
        Masses of the particle components for a single particle (kg)
    us : float
        Slip velocity (m/s)
    rho_p : float
        Particle density (kg/m^3)
    A : float
        Particle surface area (m^2)
    Cs : ndarray
        Solubility of each dissolving component in the particle (kg/m^3)
    beta : ndarray
        Mass transfer coefficients (m/s)
    beta_T : float
        Heat transfer coefficient (m/s)
    T : float
        Temperature of the particle (K)
    integrate : bool
        Flag indicating whether or not the particle is still inside the plume,
        where its trajectory should continue to be integrated.
    t : float
        Current time since the particle was released (s)
    x : float
        Current position of the particle in the x-direction (m)
    y : float
        Current position of the particle in the y-direction (m)
    z : float
        Current position of the particle in the z-direction (m)
    p_fac : float
        Buoyant force reduction factor due to a reduced buoyancy as the
        particle moves to the edge of the plume (--)
    b_local : float
        Width of the bent plume model at the location where the particle 
        exited the plume.
    sbm : `single_bubble_model.Model` object
        Model object for tracking the particle outside the plume
    
    
    See Also
    --------
    dispersed_phases.SingleParticle, dispersed_phases.PlumeParticle
    
    """
    def __init__(self, x, y, z, dbm_particle, m0, T0, nb0, lambda_1, 
                 P, Sa, Ta, K=1., K_T=1., fdis=1.e-6, t_hyd=0.):
        super(Particle, self).__init__(dbm_particle, m0, T0, nb0, lambda_1, 
                                       P, Sa, Ta, K, K_T, fdis, t_hyd)
        
        # Particles start inside the plume and should be integrated
        self.integrate = True
        self.sim_stored = False
        self.farfield = False
        
        # Store the initial particle locations
        self.t = 0.
        self.x = x
        self.y = y
        self.z = z
        
        # Update the particle with its current properties
        self.update(m0, T0, P, Sa, Ta, self.t)
    
    def track(self, t_p, X_cl, X_p, q_local, Ainv=None):
        """
        Track the particle in the Lagragian plume model
        
        Track the location of the particle within a Lagrangian plume model
        element and stop the integration when the particle exits the plume.
        
        Parameters
        ----------
        t_p : float
            Time since the particle was released (s)
        X_cl : ndarray
            Array of Cartesian coordinates (x,y,z) for the plume centerline
            (m).
        X_p : ndarray
            Array of local plume coordinates (l,n,m) for the current 
            particle position (m)().  This method converts these coordinates, 
            which are solved by the bent plume model state space solution, to 
            Cartesian coordinates.
        q_local : `LagElement` object
            Object that translates the bent plume model state space `t` and 
            `q` into the comprehensive list of derived variables.
        Ainv : ndarray, default = None
            Coordinate transformation matrix from the local plume coordinates
            (l,n,m) to Cartesian coordinates (x,y,z).  If `Ainv` is known, it
            can be passed to this function; otherwise, this function can 
            solve for `Ainv` using q_local.
        
        Returns
        -------
        xp : ndarray
            Array of Cartesian coordinates (x,y,z) for the current particle 
            position (m).
        
        """
        if self.integrate:
            # Compute the transformation matrix from local plume coordinates 
            # (l,n,m) to Cartesian coordinates (x,y,z) if needed
            if Ainv is None:
                A = lmp.local_coords(q_local, q_local, 0.)
                Ainv = inv(A)
            
            # Update the particle age
            self.t = t_p
            tp = self.t
            
            # Get the particle position
            xp = np.dot(Ainv, X_p) + X_cl
            self.x = xp[0]
            self.y = xp[1]
            self.z = xp[2]
            
            # Compute the particle offset from the plume centerline
            lp = np.sqrt(X_p[0]**2 + X_p[1]**2 + X_p[2]**2)
            
            # Compute the buoyant force reduction factor
            self.p_fac = (q_local.b - lp)**4 / q_local.b**4
            if self.p_fac < 0.:
                self.p_fac = 0.
            
            # Check if the particle exited the plume
            if lp > q_local.b:
                self.p_fac = 0.
                self.b_local = q_local.b
        
        else:
            # Return the time and position when the particle exited the plume
            tp = self.te
            xp = np.array([self.xe, self.ye, self.ze])
            self.p_fac = 0.
        
        # Return the particle position as a matrix
        return (tp, xp)
    
    def outside(self, Ta, Sa, Pa):
        """
        Remove the effect of particles if they are outside the plume
        
        Sets all of the particle properties that generate forces or 
        dissolution to zero effect if the particle is outside the plume.
        
        Parameters
        ----------
        Ta : float
            Local temperature surrounding the particle (K)
        Sa : float
            Local salinity surrounding the particle (psu)
        Pa : float
            Local pressure (Pa)
        
        """
        self.us = 0.
        self.rho_p = seawater.density(Ta, Sa, Pa)
        self.A = 0.
        self.Cs = np.zeros(len(self.composition))
        self.beta = np.zeros(len(self.composition))
        self.beta_T = 0.
        self.T = Ta
    
    def run_sbm(self, profile):
        """
        Run the `single_bubble_model` to track particles outside the plume
        
        Continues the simulation of the particle outside the plume using 
        the `single_bubble_model`.  The object containing the simulation 
        and simulation results will be added to the attributes of this 
        Particle object.
        
        Parameters
        ----------
        profile : `ambient.Profile` object
            Ambient CTD data for the model simulation
        
        """
        # Create the simulation object
        self.sbm = single_bubble_model.Model(profile)
        
        # Create the inputs to the sbm.simulate method
        X0 = np.array([self.x, self.y, self.z])
        Ta, Sa, P = profile.get_values(X0[2], ['temperature', 'salinity', 
                    'pressure'])
        de = self.diameter(self.m, self.T, P, Sa, Ta)
        if self.particle.issoluble:
            yk = self.particle.mol_frac(self.m)
        else:
            yk = 1.
        
        # Run the simulation
        self.sbm.simulate(self.particle, X0, de, yk, self.T, self.K, 
            self.K_T, self.fdis, delta_t=100000.)
        
        # Set flag indicating that far-field solution was computed
        self.farfield = True


# ----------------------------------------------------------------------------
# Object to translate the state space into all the derived variables
# ----------------------------------------------------------------------------

class LagElement(object):
    """
    Manages the Lagragian plume element state space and derived variables
    
    Translates the state space variables for a Lagrangian plume element into
    its individual parts and derived quantitites.
    
    Parameters
    ----------
    t0 : float
        Initial time of the simulation (s)
    q0 : ndarray
        Initial values of the simulation state space, q
    D : float
        Diameter for the equivalent circular cross-section of the release (m)
    profile : `ambient.Profile`
        Ambient CTD data    
    p : `ModelParams`
        Container for the fixed model parameters
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the 
        simulation
    tracers : string list
        List of passive tracers in the discharge.  These can be chemicals 
        present in the ambient `profile` data, and if so, entrainment of these
        chemicals will change the concentrations computed for these tracers.
        However, none of these concentrations are used in the dissolution of 
        the dispersed phase.  Hence, `tracers` should not contain any 
        chemicals present in the dispersed phase particles.
    chem_names : string list
        List of chemical parameters to track for the dissolution.  Only the 
        parameters in this list will be used to set background concentration
        for the dissolution, and the concentrations of these parameters are 
        computed separately from those listed in `tracers` or inputed from
        the discharge through `cj`.
    
    Attributes
    ----------
    t0 : float
        Initial time of the simulation (s)
    q0 : ndarray
        Initial values of the simulation state space, q
    D : float
        Diameter for the equivalent circular cross-section of the release (m)
    tracers : string list
        List of passive tracers in the discharge.  These can be chemicals 
        present in the ambient `profile` data, and if so, entrainment of these
        chemicals will change the concentrations computed for these tracers.
        However, none of these concentrations are used in the dissolution of 
        the dispersed phase.  Hence, `tracers` should not contain any 
        chemicals present in the dispersed phase particles.
    chem_names : string list
        List of chemical parameters to track for the dissolution.  Only the 
        parameters in this list will be used to set background concentration
        for the dissolution, and the concentrations of these parameters are 
        computed separately from those listed in `tracers` or inputed from
        the discharge through `cj`. 
    len : int
        Number of variables in the state space q (--)
    ntracers : int
        Number of passive chemical tracers (--)
    nchems : int
        Number of chemicals tracked for dissolution of the dispersed phase
        particles (--)
    np : int
        Number of dispersed phase particles (--)
    t : float
        Independent variable for the current time (s)
    q : ndarray
        Dependent variable for the current state space 
    M : float
        Mass of the Lagrangian element (kg)
    Se : float
        Salt in the Lagrangian element (psu kg)
    He : float
        Heat of the Lagrangian element (J)
    Jx : float
        Dynamic momentum of the Lagrangian element in the x-direction 
        (kg m/s)
    Jy : float
        Dynamic momentum of the Lagrangian element in the y-direction 
        (kg m/s)
    Jz : float
        Dynamic momentum of the Lagrangian element in the z-direction 
        (kg m/s)
    H : float
        Relative thickness of the Lagrangian element h/V (s)
    x : float
        Current x-position of the Lagrangian element (m)
    y : float
        Current y-position of the Lagrangian element (m)
    z : float
        Current z-position of the Lagrangian element (m)
    s : float
        Current s-position along the centerline of the plume for the 
        Lagrangian element (m)
    M_p : dict of ndarrays
        For integer key: the total mass fluxes (kg/s) of each component in a 
        particle.
    H_p : ndarray
        Total heat flux for each particle (J/s)
    t_p : ndarray
        Time since release for each particle (s)
    X_p : ndarray
        Position of each particle in local plume coordinates (l,n,m) (m).
    cpe : ndarray
        Masses of the chemical components involved in dissolution (kg)
    cte : ndarray
        Masses of the passive tracers in the plume (concentration kg)
    Pa : float
        Ambient pressure at the current element location (Pa)
    Ta : float
        Ambient temperature at the current element location (K)
    Sa : float
        Ambient salinity at the current element location (psu)
    ua : float
        Crossflow velocity in the x-direction at the current element location 
        (m/s)
    ca_chems : ndarray
        Ambient concentration of the chemical components involved in 
        dissolution at the current element location (kg/m^3)
    ca_tracers : 
        Ambient concentration of the passive tracers in the plume at the 
        current element location (concentration)
    rho_a : float
        Ambient density at the current element location (kg/m^3)
    S : float
        Salinity of the Lagrangian element (psu)
    T : float
        Temperature of the Lagrangian element (T)
    c_chems : 
        Concentration of the chemical components involved in dissolution for
        the Lagrangian element (kg/m^3)
    c_tracers : 
        Concentration of the passive tracers in the Lagrangian element 
        (concentration)
    u : float
        Velocity in the x-direction of the Lagrangian element (m/s)
    v : float
        Velocity in the y-direction of the Lagrangian element (m/s)
    w : float
        Velocity in the z-direction of the Lagrangian element (m/s)
    hvel : float
        Velocity in the horizontal plane for the Lagrangian element (m/s)
    V : float
        Velocity in the s-direction of the Lagrangian element (m/s)
    h : float
        Current thickness of the Lagrangian element (m)
    rho : float
        Density of the entrained seawater in the Lagrangian element (kg/m^3)
    b : float
        Half-width of the Lagrangian element (m)
    sin_p : float
        The sine of the angle phi (--)
    cos_p : float
        The cosine of the angle phi (--)
    sin_t : float
        The sine of the angle theta (--)
    cos_t : float
        The cosine of the angle theta (--)
    phi : float
        The vertical angle from horizontal of the current plume trajectory
        (rad in range +/- pi/2).  Since z is positive down (depth), phi = 
        pi/2 point down and -pi/2 points up.
    theta : float
        The lateral angle in the horizontal plane from the x-axis to the 
        current plume trajectory (rad in range 0 to 2 pi)
    mp : ndarray
        Masses of each of the dispersed phase particles in the `particles`
        variable
    fb : ndarray
        Buoyant force for each of the dispersed phase particles in the 
        `particles` variable as density difference (kg/m^3)
    x_p : ndarray
    Mp : float
        Total mass of dispersed phases in the Lagrangian element (kg)
    Fb : float
        Total buoyant force as density difference of the dispersed phases in 
        the Lagrangian element (kg/m^3)
    
    """
    def __init__(self, t0, q0, D, profile, p, particles, tracers, 
                 chem_names):
        super(LagElement, self).__init__()
        
        # Store the inputs to stay with the Lagrangian element
        self.t0 = t0
        self.q0 = q0
        self.D = D
        self.tracers = tracers
        self.chem_names = chem_names
        self.len = q0.shape[0]
        self.ntracers = len(self.tracers)
        self.nchems = len(self.chem_names)
        self.np = len(particles)
        
        # Extract the state variables and compute the derived quantities
        self.update(t0, q0, profile, p, particles)
    
    def update(self, t, q, profile, p, particles=[]):
        """
        Update the `LagElement` object with the current local conditions
        
        Extract the state variables and compute the derived quantities given
        the current local conditions.
        
        Parameters
        ----------
        t : float
            Current time of the simulation (s)
        q : ndarray
            Current values of the simulation state space, q
        profile : `ambient.Profile`
            Ambient CTD data    
        p : `ModelParams`
            Container for the fixed model parameters
        particles : list of `Particle` objects
            List of `Particle` objects describing each dispersed phase in the 
            simulation
        
        """
        # Save the current state space
        self.t = t
        self.q = q
        
        # Extract the state-space variables from q
        self.M = q[0]
        self.Se = q[1]
        self.He = q[2]
        self.Jx = q[3]
        self.Jy = q[4]
        self.Jz = q[5]
        self.H = q[6]
        self.x = q[7]
        self.y = q[8]
        self.z = q[9]
        self.s = q[10]
        idx = 11
        M_p = {}
        H_p = []
        t_p = []
        X_p = []
        for i in range(self.np):
            M_p[i] = q[idx:idx + particles[i].particle.nc]
            idx += particles[i].particle.nc
            H_p.extend(q[idx:idx + 1])
            idx += 1
            t_p.extend(q[idx:idx + 1])
            idx += 1
            X_p.append(q[idx:idx + 3])
            idx += 3
        self.M_p = M_p
        self.H_p = np.array(H_p)
        self.t_p = np.array(t_p)
        self.X_p = np.array(X_p)
        self.cpe = q[idx:idx + self.nchems]
        idx += self.nchems
        if self.ntracers >= 1:
            self.cte = q[idx:]
        else:
            self.cte = np.array([])
        
        # Get the local ambient conditions
        self.Pa, self.Ta, self.Sa, self.ua, self.va, self.wa = \
            profile.get_values(self.z, ['pressure', 'temperature', 
            'salinity', 'ua', 'va', 'wa'])
        self.ca_chems = profile.get_values(self.z, self.chem_names)
        self.ca_tracers = profile.get_values(self.z, self.tracers)
        self.rho_a = seawater.density(self.Ta, self.Sa, self.Pa)
        
        # Compute the derived quantities
        self.S = self.Se / self.M
        self.T = self.He / (self.M * seawater.cp())
        self.rho = seawater.density(self.T, self.S, self.Pa)
        self.c_chems = self.cpe / (self.M / self.rho)
        self.c_tracers = self.cte / (self.M / self.rho)
        self.u = self.Jx / self.M
        self.v = self.Jy / self.M
        self.w = self.Jz / self.M
        self.hvel = np.sqrt(self.u**2 + self.v**2)
        self.V = np.sqrt(self.hvel**2 + self.w**2)
        self.h = self.H * self.V
        self.b = np.sqrt(self.M / (self.rho * np.pi * self.h))
        self.sin_p = self.w / self.V
        self.cos_p = self.hvel / self.V
        if self.hvel == 0.:
            # if hvel = 0, flow is purely along z; let theta = 0
            self.sin_t = 0.
            self.cos_t = 1.
        else:
            self.sin_t = self.v / self.hvel
            self.cos_t = self.u / self.hvel
        self.phi = np.arctan2(self.w, self.hvel)
        self.theta = np.arctan2(self.v, self.u)
        
        # Compute the transformation matrix from the local plume coordinates
        # (l,n,m) to Cartesian coordinates (x,y,z)
        Ainv = inv(lmp.local_coords(self, self, 0.))
        
        # Get the particle characteristics
        self.mp = np.zeros(self.np)
        self.fb = np.zeros(self.np)
        self.x_p = np.zeros((self.np, 3))
        
        for i in range(self.np):
            # If this is a post-processing call, update the status of the 
            # integration flag
            if particles[i].sim_stored:
                if np.isnan(self.X_p[i][0]):
                    particles[i].integrate = False
                else:
                    particles[i].integrate = True
            
            # Update the particles with their current properties
            m_p = self.M_p[i] / particles[i].nbe
            T_p = self.H_p[i] / (np.sum(self.M_p[i]) * particles[i].cp)
            particles[i].update(m_p, T_p, self.Pa, self.S, self.T, 
                                self.t_p[i])
            
            # Track the particle in the plume
            self.t_p[i], self.x_p[i,:] = particles[i].track(self.t_p[i], 
                            np.array([self.x, self.y, self.z]), 
                            self.X_p[i], self, Ainv)
            
            # Get the mass of particles following this Lagrangian element
            self.mp[i] = np.sum(m_p) * particles[i].nbe
            
            # Compute the buoyant force coming from this set of particles
            self.fb[i] = self.rho / particles[i].rho_p * self.mp[i] * \
                         (self.rho_a - particles[i].rho_p) * \
                         particles[i].p_fac
            
            # Force the particle mass and bubble force to zero if the bubble
            # has dissolved
            if self.rho == particles[i].rho_p:
                self.mp[i] = 0.
                self.fb[i] = 0.
        
        # Compute the net particle mass and buoyant force
        self.Fb = np.sum(self.fb)


# ----------------------------------------------------------------------------
# Functions to plot output from the simulations
# ----------------------------------------------------------------------------

def plot_state_space(t, q, q_local, profile, p, particles, fig):
    """
    Plot the Lagrangian model state space
    
    Plot the standard set of state space variables used to evaluate the
    quality of the model solution
    
    Parameters
    ----------
    t : ndarray
        Array of times computed in the solution (s)
    q : ndarray
        Array of state space values computed in the solution
    q_local : `LagElement` object
        Object that translates the `Model` state space `t` and `q` into the 
        comprehensive list of derived variables.
    profile : `ambient.Profile`
        Ambient CTD data
    p : `ModelParams`
        Container for the fixed model parameters
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the 
        simulation
    fig : int
        Number of the figure window in which to draw the plot
    
    Notes
    -----
    Plots the trajectory of the jet centerline, the trajectory of the 
    simulated particles, and the Lagrangian element mass.
    
    """
    # Extract the trajectory variables
    x = q[:,7]
    y = q[:,8]
    z = q[:,9]
    s = q[:,10]
    M = q[:,0]
    
    # Extract the particle positions from the q state space
    xp = np.zeros((len(t),3*len(particles)))
    for i in range(len(t)):
        q_local.update(t[i], q[i,:], profile, p, particles)
        for j in range(len(particles)):
            xp[i,j*3:j*3+3] = q_local.x_p[j,:]
    
    # Plot the figure
    plt.figure(fig)
    plt.clf()
    plt.show()
    
    # x-z plane
    ax1 = plt.subplot(221)
    ax1.plot(x, z)
    for i in range(len(particles)):
        ax1.plot(xp[:,i*3], xp[:,i*3 + 2], '.--')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('Depth (m)')
    ax1.invert_yaxis()
    ax1.grid(b=True, which='major', color='0.65', linestyle='-')
    
    # y-z plane
    ax2 = plt.subplot(222)
    ax2.plot(y, z)
    for i in range(len(particles)):
        ax2.plot(xp[:,i*3+1], xp[:,i*3 + 2], '.--')
    ax2.set_xlabel('y (m)')
    ax2.set_ylabel('Depth (m)')
    ax2.invert_yaxis()
    ax2.grid(b=True, which='major', color='0.65', linestyle='-')
    
    # x-y plane
    ax3 = plt.subplot(223)
    ax3.plot(x, y)
    for i in range(len(particles)):
        ax3.plot(xp[:,i*3], xp[:,i*3 + 1], '.--')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.grid(b=True, which='major', color='0.65', linestyle='-')
    
    # M-s plane
    ax4 = plt.subplot(224)
    ax4.plot(s, M)
    ax4.set_xlabel('s (m)')
    ax4.set_ylabel('M (kg)')
    ax4.grid(b=True, which='major', color='0.65', linestyle='-')
    
    plt.draw()


def plot_all_variables(t, q, q_local, profile, p, particles, tracked, 
    fig):
    """
    Plot a comprehensive suite of simulation results
    
    Generate a comprehensive suite of graphs showing the state and 
    derived variables along with ambient profile data in order to 
    view the model output for detailed analysis.
    
    Parameters
    ----------
    t : ndarray
        Array of times computed in the solution (s)
    q : ndarray
        Array of state space values computed in the solution
    q_local : `LagElement` object
        Object that translates the `Model` state space `t` and `q` into the 
        comprehensive list of derived variables.
    profile : `ambient.Profile`
        Ambient CTD data
    p : `ModelParams`
        Container for the fixed model parameters
    particles : list of `Particle` objects
        List of `Particle` objects describing each dispersed phase in the 
        simulation
    tracked : bool
        Flag indicating whether or not the `single_bubble_model` was run to 
        track the particles.
    fig : int
        Number of the figure window in which to draw the plot
    
    """
    # Don't offset any of the axes
    formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    
    # Create a second Lagrangian element in order to compute entrainment
    q0_local = LagElement(t[0], q[0,:], q_local.D, profile, p, particles, 
        q_local.tracers, q_local.chem_names)
    n_part = q0_local.np
    pchems = 1
    for i in range(n_part):
        if len(particles[i].composition) > pchems:
            pchems = len(particles[i].composition)
    
    # Store the derived variables
    M = np.zeros(t.shape)
    S = np.zeros(t.shape)
    T = np.zeros(t.shape)
    Mpf = np.zeros((len(t), n_part, pchems))
    Hp = np.zeros((len(t), n_part))
    Mp = np.zeros((len(t), n_part))
    Tp = np.zeros((len(t), n_part))
    xp = np.zeros((len(t), 3*n_part))
    u = np.zeros(t.shape)
    v = np.zeros(t.shape)
    w = np.zeros(t.shape)
    V = np.zeros(t.shape)
    h = np.zeros(t.shape)
    x = np.zeros(t.shape)
    y = np.zeros(t.shape)
    z = np.zeros(t.shape)
    s = np.zeros(t.shape)
    rho = np.zeros(t.shape)
    b = np.zeros(t.shape)
    cos_p = np.zeros(t.shape)
    sin_p = np.zeros(t.shape)
    cos_t = np.zeros(t.shape)
    sin_t = np.zeros(t.shape)
    rho_a = np.zeros(t.shape)
    Sa = np.zeros(t.shape)
    Ta = np.zeros(t.shape)
    ua = np.zeros(t.shape)
    E = np.zeros(t.shape)
    
    for i in range(len(t)):
        if i > 0:
            q0_local.update(t[i-1], q[i-1,:], profile, p, particles)
        q_local.update(t[i], q[i,:], profile, p, particles)
        M[i] = q_local.M
        S[i] = q_local.S
        T[i] = q_local.T
        for j in range(n_part):
            Mpf[i,j,0:len(q_local.M_p[j])] = q_local.M_p[j][:]
            Mp[i,j] = np.sum(particles[j].m[:])
            Tp[i,j] = particles[j].T
            xp[i,j*3:j*3+3] = q_local.x_p[j,:]
        Hp[i,:] = q_local.H_p
        u[i] = q_local.u
        v[i] = q_local.v
        w[i] = q_local.w
        V[i] = q_local.V
        h[i] = q_local.h
        x[i] = q_local.x
        y[i] = q_local.y
        z[i] = q_local.z
        s[i] = q_local.s
        rho[i] = q_local.rho
        b[i] = q_local.b
        cos_p[i] = q_local.cos_p
        sin_p[i] = q_local.sin_p
        cos_t[i] = q_local.cos_t
        sin_t[i] = q_local.sin_t
        rho_a[i] = q_local.rho_a
        Sa[i] = q_local.Sa
        Ta[i] = q_local.Ta
        ua[i] = q_local.ua
        E[i] = lmp.entrainment(q0_local, q_local, p)
    
    # Compute the unit vector along the plume axis
    Sz = sin_p
    Sx = cos_p * cos_t
    Sy = cos_p * sin_t
    
    # Plot cross-sections through the plume
    plt.figure(fig)
    plt.clf()
    plt.show()
    fig += 1
    
    ax1 = plt.subplot(221)
    ax1.plot(x, z, 'b-')
    [x1, z1, x2, z2] = width_projection(Sx, Sz, b)
    ax1.plot(x + x1, z + z1, 'b--')
    ax1.plot(x + x2, z + z2, 'b--')
    for i in range(len(particles)):
        ax1.plot(particles[i].x, particles[i].z, 'o')
        ax1.plot(xp[:,i*3], xp[:,i*3+2], '.--')
        if tracked:
            if particles[i].integrate is False and particles[i].z > 0.:
                ax1.plot(particles[i].sbm.y[:,0], particles[i].sbm.y[:,2], 
                         '.:')
    ax1.invert_yaxis()
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('z (m)')
    ax1.grid(b=True, which='major', color='0.5', linestyle='-')
    
    ax2 = plt.subplot(222)
    ax2.plot(y, z, 'b-')
    [y1, z1, y2, z2] = width_projection(Sy, Sz, b)
    ax2.plot(y + y1, z + z1, 'b--')
    ax2.plot(y + y2, z + z2, 'b--')
    for i in range(len(particles)):
        ax2.plot(particles[i].y, particles[i].z, 'o')
        ax2.plot(xp[:,i*3+1], xp[:,i*3+2], '.--')
        if tracked:
            if particles[i].integrate is False and particles[i].z > 0.:
                ax2.plot(particles[i].sbm.y[:,1], particles[i].sbm.y[:,2], 
                         '.:')
    ax2.invert_yaxis()
    ax2.set_xlabel('y (m)')
    ax2.set_ylabel('z (m)')
    ax2.grid(b=True, which='major', color='0.5', linestyle='-')
    
    ax3 = plt.subplot(223)
    ax3.plot(x, y, 'b-')
    [x1, y1, x2, y2] = width_projection(Sx, Sy, b)
    ax3.plot(x + x1, y + y1, 'b--')
    ax3.plot(x + x2, y + y2, 'b--')
    for i in range(len(particles)):
        ax3.plot(particles[i].x, particles[i].y, 'o')
        ax3.plot(xp[:,i*3], xp[:,i*3+1], '.--')
        if tracked:
            if particles[i].integrate is False and particles[i].z > 0.:
                ax3.plot(particles[i].sbm.y[:,0], particles[i].sbm.y[:,1], 
                         '.:')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.grid(b=True, which='major', color='0.5', linestyle='-')
    
    ax4 = plt.subplot(224)
    ax4.plot(s, np.zeros(s.shape), 'b-')
    ax4.plot(s, b, 'b--')
    ax4.plot(s, -b, 'b--')
    ax4.set_xlabel('s (m)')
    ax4.set_ylabel('r (m)')
    ax4.grid(b=True, which='major', color='0.5', linestyle='-')
    
    plt.draw()
    
    # Plot the Lagrangian element height and entrainment rate
    plt.figure(fig)
    plt.clf()
    plt.show()
    fig += 1
    
    ax1 = plt.subplot(121)
    ax1.plot(s, h, 'b-')
    ax1.set_xlabel('s (m)')
    ax1.set_ylabel('h (m/s)')
    ax1.grid(b=True, which='major', color='0.5', linestyle='-')    
    
    ax2 = plt.subplot(122)
    ax2.plot(s, E, 'b-')
    ax2.set_xlabel('s (m)')
    ax2.set_ylabel('E (kg/s)')
    ax2.grid(b=True, which='major', color='0.5', linestyle='-')    
    
    plt.draw()
    
    # Plot the velocities along the plume centerline
    plt.figure(fig)
    plt.clf()
    plt.show()
    fig += 1
    
    ax1 = plt.subplot(221)
    ax1.plot(s, u, 'b-')
    ax1.plot(s, ua, 'g--')
    ax1.set_xlabel('s (m)')
    ax1.set_ylabel('u (m/s)')
    ax1.grid(b=True, which='major', color='0.5', linestyle='-')
    
    ax2 = plt.subplot(222)
    ax2.plot(s, v, 'b-')
    ax2.set_xlabel('s (m)')
    ax2.set_ylabel('v (m/s)')
    ax2.grid(b=True, which='major', color='0.5', linestyle='-')
    
    ax3 = plt.subplot(223)
    ax3.plot(s, w, 'b-')
    ax3.set_xlabel('s (m)')
    ax3.set_ylabel('w (m/s)')
    ax3.grid(b=True, which='major', color='0.5', linestyle='-')
    
    ax4 = plt.subplot(224)
    ax4.plot(s, V, 'b-')
    ax4.set_xlabel('s (m)')
    ax4.set_ylabel('V (m/s)')
    ax4.grid(b=True, which='major', color='0.5', linestyle='-')
    
    plt.draw()
    
    # Plot the salinity, temperature, and density in the plume
    plt.figure(fig)
    plt.clf()
    plt.ticklabel_format(useOffset=False, axis='y')
    plt.show()
    fig += 1
    
    ax1 = plt.subplot(221)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.plot(s, S, 'b-')
    ax1.plot(s, Sa, 'g--')
    if np.max(S) - np.min(S) < 1.e-6:
        ax1.set_ylim([S[0] - 1, S[0] + 1])
    ax1.set_xlabel('s (m)')
    ax1.set_ylabel('Salinity (psu)')
    ax1.grid(b=True, which='major', color='0.5', linestyle='-')
    
    ax2 = plt.subplot(222)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.plot(s, T - 273.15, 'b-')
    ax2.plot(s, Ta - 273.15, 'g--')
    if np.max(T) - np.min(T) < 1.e-6:
        ax2.set_ylim([T[0] - 273.15 - 1., T[0] - 273.15 + 1.])
    ax2.set_xlabel('s (m)')
    ax2.set_ylabel('Temperature (deg C)')
    ax2.grid(b=True, which='major', color='0.5', linestyle='-')
    
    ax3 = plt.subplot(223)
    ax3.yaxis.set_major_formatter(formatter)
    ax3.plot(s, rho, 'b-')
    ax3.plot(s, rho_a, 'g--')
    if np.max(rho) - np.min(rho) < 1.e-6:
        ax3.set_ylim([rho[0] - 1, rho[0] + 1])
    ax3.set_xlabel('s (m)')
    ax3.set_ylabel('Density (kg/m^3)')
    ax3.grid(b=True, which='major', color='0.5', linestyle='-')
    
    plt.draw()
    
    # Plot the particle mass and temperature
    if n_part > 0:
        plt.figure(fig)
        plt.clf()
        plt.ticklabel_format(useOffset=False, axis='y')
        plt.show()
        fig += 1
    
        ax1 = plt.subplot(121)
        ax1.yaxis.set_major_formatter(formatter)
        ax1.plot(s, Mp / 1e-6, 'b-')
        ax1.set_xlabel('s (m)')
        ax1.set_ylabel('m (mg)')
        ax1.grid(b=True, which='major', color='0.5', linestyle='-')
    
        ax2 = plt.subplot(122)
        ax2.yaxis.set_major_formatter(formatter)
        ax2.plot(s, Tp - 273.15, 'b-')
        ax2.set_xlabel('s (m)')
        ax2.set_ylabel('Temperature (deg C)')
        ax2.grid(b=True, which='major', color='0.5', linestyle='-')
    
        plt.draw()


def width_projection(Sx, Sy, b):
    """
    Find the location of the plume width in x, y, z space
    
    Converts the width b and plume orientation phi and theta into an 
    (x, y, z) location of the plume edge.  This function provides a two-
    dimensional result given the unit vector along the plume centerline
    (Sx, Sy) along two dimensions in the (x, y, z) space
    
    Parameters
    ----------
    Sx : float
        Unit vector projection of the plume trajectory on one of the 
        coordinate axes in (x, y, z) space.
    Sy : float
        Unit vector projection of the plume trajectory on another of the 
        coordinate axes in (x, y, z) space.
    b : float
        Local plume width
    
    Returns
    -------
    x1 : float
        Plume edge for Sx projection to left of plume translation direction
    y1 : float
        Plume edge for Sy projection to left of plume translation direction
    x2 : float
        Plume edge for Sx projection to right of plume translation direction
    y1 : float
        Plume edge for Sy projection to right of plume translation direction
    
    Notes
    -----
    The values of S in the (x, y, z) sytem would be::
    
        Sz = sin ( phi )
        Sx = cos ( phi ) * cos ( theta )
        Sy = cos ( phi ) * sin ( theta )
    
    Any two of these coordinates of the unit vector can be provided to this 
    function as input.  
    
    """
    # Get the angle to the s-direction in the x-y plane
    alpha = np.arctan2(Sy, Sx)
    
    # Get the coordinates of the plume edge to the right of the s-vector 
    # moving with the plume
    x1 = b * np.cos(alpha - np.pi/2.)
    y1 = b * np.sin(alpha - np.pi/2.)
    
    # Get the coordinates of the plume edge to the left of the s-vector
    # moving with the plume
    x2 = b * np.cos(alpha + np.pi/2.)
    y2 = b * np.sin(alpha + np.pi/2.)
    
    return (x1, y1, x2, y2)

