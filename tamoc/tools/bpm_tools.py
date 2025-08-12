"""
bpm_tools.py
------------

Tools to aid in setting up and running bent plume model simulations

"""
# S. Socolofsky, Texas A&M University, August 2025, <socolofs@tamu.edu>

from tamoc.tools import eos_tools
from tamoc import dbm, dispersed_phases, bent_plume_model
from tamoc import particle_size_models

import numpy as np

class BPM_Sim(object):
    """
    Class for handling bent plume model simulations
    
    """
    def __init__(self, profile, dbm_mixture, mass_frac, inp_dict):
        
        super(BPM_Sim, self).__init__()
        
        # Store the input parameters
        self.profile = profile
        self.dbm_mixture = dbm_mixture
        self.mass_frac = mass_frac
        self.K = inp_dict['K']
        self.K_T = inp_dict['K_T']
        self.f_dis = inp_dict['fdis']
        self.t_hyd = inp_dict['t_hyd']
        self.lag_time = inp_dict['lag_time']
        self.single_phase_particles = inp_dict['single_phase_particles']
        self.X0 = inp_dict['x0']
        self.D = inp_dict['D']
        self.phi_0 = inp_dict['phi_0']
        self.theta_0 = inp_dict['theta_0']
        self.Qp = inp_dict['Qp']
        self.Sp = inp_dict['Sp']
        self.Tj = inp_dict['Tj']
        self.cj = inp_dict['cj']
        self.tracers = inp_dict['tracers']
        self.m0 = inp_dict['m0']
        self.q0 = inp_dict['q0']
        self.gor = inp_dict['gor']
        self.gas_model = inp_dict['gas_model']
        self.pdf_gas = inp_dict['pdf_gas']
        self.n_gas = inp_dict['n_gas']
        self.de_gas = inp_dict['de_gas']
        self.vf_gas = inp_dict['vf_gas']
        self.liq_model = inp_dict['liq_model']
        self.pdf_liq = inp_dict['pdf_liq']
        self.n_liq = inp_dict['n_liq']
        self.de_liq = inp_dict['de_liq']
        self.vf_liq = inp_dict['vf_liq']
        
        # Get the mass flow rate of the spill
        if isinstance(self.m0, type(None)):
            # User must have specified bbl/d and GOR...get the correct 
            # mixture composition to match the given GOR
            self.dbm_mixture, self.mass_frac = \
                eos_tools.adjust_mass_frac_for_gor(self.dbm_mixture, 
                    self.mass_frac, self.gor)
            
            # Get the component mass flow rates to match the given volume
            # flow rate
            self.m0 = eos_tools.mass_flowratefrom_volume_flowrate(
                self.dbm_mixture, self.mass_frac, self.q0
            )
        
        elif isinstance(self.m0, float):
            # The user specified the total mixture mass flow rate...
            # Convert to flow rate for each component of the mixture
            self.m0 = self.m0 * self.mass_frac
        
        # Determine the produced water discharge velocity
        self.Vj = self.Qp / (np.pi * self.D**2 / 4.)
        
        # Create the particle size model
        self.Pj = self.profile.get_values(self.X0[2], 'pressure')[0]
        self.psm = particle_size_models.Model(self.profile, self.dbm_mixture,
            self.m0, self.X0[2], self.Tj, self.Pj)
        
        # Perform flash equilibrium at the release
        Ta, Pa = self.profile.get_values(self.X0[2], ['temperature', 
            'pressure'])
        m, xi, K = self.dbm_mixture.equilibrium(self.m0, Ta, Pa)
        self.m0_gas = m[0,:]
        self.m0_liq = m[1,:]
        self.yk0_gas = self.dbm_mixture.mol_frac(self.m0_gas)
        self.yk0_liq = self.dbm_mixture.mol_frac(self.m0_liq)        
        
        # Create the plume particles list
        if not isinstance(self.gas_model, type(None)):
            # We need to use the particle size model to get the bubble and 
            # droplet size distributions
            self.psm.simulate(self.D, self.gas_model, self.pdf_gas, 
                self.liq_model, self.pdf_liq)
            self.de_gas, self.vf_gas, self.de_liq, self.vf_liq = \
                self.psm.get_distributions(self.n_gas, self.n_liq)
                    
        # Get the mass flow rate for each bubble and droplet size
        self.mf_gas = self.vf_gas * np.sum(self.m0_gas)
        self.mf_liq = self.vf_liq * np.sum(self.m0_liq)
        
        # Create gas and liquid dbm.FluidParticle objects
        user_data = self.dbm_mixture.user_data
        delta = self.dbm_mixture.delta
        delta_groups = self.dbm_mixture.delta_groups
        if self.single_phase_particles:
            fp_gas = 0
            fp_liq = 1
        else:
            fp_gas = 2
            fp_liq = 2
        sigma_correction = self.dbm_mixture.sigma_correction[0]
        isair = self.dbm_mixture.isair
        self.gas = dbm.FluidParticle(self.dbm_mixture.composition, 
            fp_type=fp_gas, delta=delta, delta_groups=delta_groups, 
            user_data=user_data, isair=isair, 
            sigma_correction=sigma_correction)
        self.liq = dbm.FluidParticle(self.dbm_mixture.composition,
            fp_type=fp_liq, delta=delta, delta_groups=delta_groups, 
            user_data=user_data, isair=isair, 
            sigma_correction=sigma_correction)
        
        # Create a single list of gas and liquid particles
        self.particles = []
        get_plume_particles(self.particles, self.profile, self.X0,
            self.gas, self.yk0_gas, self.mf_gas, self.de_gas, self.Tj)
        get_plume_particles(self.particles, self.profile, self.X0,
            self.liq, self.yk0_liq, self.mf_liq, self.de_liq, self.Tj)
        
        # Create the bent plume model 
        self.bpm = bent_plume_model.Model(self.profile)
        self.sim_stored = False

    def simulate(self, dt_max=60., track=True):
        """
        Run a bent plume model simulation
        
        Run the bent plume model using the present simulation settings
        
        Notes
        -----
        This method updates the `bpm` attribute of this class with an 
        object that contains the simulation result.
        
        """
        # Determine how far to simulate along the plume trajectory
        sd_max = 1.5 * self.X0[2] / self.D
        
        # Run the bent plume model
        self.bpm.simulate(self.X0, self.D, self.Vj, self.phi_0, self.theta_0,
            self.Sp, self.Tj, self.cj, self.tracers, particles=self.particles, 
            track=track, dt_max=dt_max, sd_max=sd_max)
        self.sim_stored = True
    
    def _report_run_bpm(self):
        """
        Print an error message requesting user to run the `simulate` method
        
        """
        print('\nERROR:  Bent plume model simulation not available.')
        print('        Run the `simulate` method before calling this')
        print('        method.\n')
        return (0,)
    
    def plot_initial_psds(self, fig=1, clear_fig=True):
        """
        Plot the initial bubble and droplet size distributions
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces one figure
            with multiple subplots.
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        fig : plt.figure
            Returns a handle to the figure that could be used for saving 
            the figure
        
        """
        if self.sim_stored:
            return self.bpm.plot_psds(fig, 0, 0, clear_fig)
        else:
            return _report_run_bpm()
    
    def plot_state_space(self, fig=2, clear_fig=True):
        """
        Plot the bent plume model state space
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces one figure
            with multiple subplots.
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        fig : plt.figure
            Returns a handle to the figure that could be used for saving 
            the figure
        
        """
        if self.sim_stored:
            return self.bpm.plot_state_space(fig, clear_fig)
        else:
            return _report_run_bpm()
        
    def plot_all_variables(self, fig=3, clear_fig=True):
        """
        Plot all variables from the bent plume model solution
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces multiple
            figures.
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        fig : plt.figure
            Returns a handle to the figure that could be used for saving 
            the figure
        
        """
        if self.sim_stored:
            return self.bpm.plot_all_variables(fig, clear_fig)
        else:
            return _report_run_bpm()
    
    def plot_fractions_dissolved(self, fig=100, clear_fig=True):
        """
        Plot the fraction dissolved for each tracked chemical component
        
        This method creates three figures.  The first figure displays the 
        fate of released chemicals in the near-field plume.  The second
        figure displays the fate of released chemicals in the far-field
        portion of the simulation only.  The third figure displays the fate
        of the released chemicals through the whole near-field and far-field
        simulation domains.        
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces multiple
            figures.
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        figs : list
            Returns a list of handles to the figures that could be used for
            saving the figures
        
        """
        if not self.sim_stored:
            return _report_run_bpm()
            
        else:
            # Plot only those compounds with a non-zero mass flow rate at the
            # release
            chems = []
            for i in range(len(self.dbm_mixture.composition)):
                if self.mass_frac[i] > 0:
                    chems.append(self.dbm_mixture.composition[i])
        
            # Create a list to hold the figures
            figs = []
        
            # Start with the near-field simulation
            f = self.bpm.plot_fractions_dissolved(fig, chems=chems, stage=0, 
                clear=clear_fig)
            figs.append(f)
        
            # Then the far-field simulation
            f = self.bpm.plot_fractions_dissolved(fig+1, chems=chems, stage=1,
                clear=clear_fig)
            figs.append(f)
        
            # And finally the whole simulation
            f = self.bpm.plot_fractions_dissolved(fig+2, chems=chems, stage=-1,
                clear=clear_fig)
            figs.append(f)
        
            return figs
    
    def plot_mass_balance(self, fig=200, t_max=-1, clear_fig=True):
        """
        Plot the time-history of the mass balance
        
        Parameters
        ----------
        fig : int, default=1
            Figure number to begin plotting.  This method produces multiple
            figures.
        t_max : float, default=-1
            The maximum time to include in the history plot (days).  If -1,
            then the maximum surfacing time in the simulation is used.        
        clear_fig : bool, default=True
            Boolean flag stating whether to clear the figure before plotting
        
        Returns
        -------
        fig : list
            Returns a handles to the figure that could be used for saving the
            figures
        
        """
        if not self.sim_stored:
            return _report_run_bpm()
            
        else:            
            # Plot only those compounds with a non-zero mass flow rate at the
            # release
            chems = []
            for i in range(len(self.dbm_mixture.composition)):
                if self.mass_frac[i] > 0:
                    chems.append(self.dbm_mixture.composition[i])
            
            # Create the plot
            return self.bpm.plot_mass_balance(fig, chems=chems, fp_type=-1, 
                t_max=t_max, clear=clear_fig)
        

def get_plume_particles(particles, profile, X0, dbm_fluid, yk, mf, de, 
    Tj, lambda_1=0.9):
    """
    Create plume particles for use in the bent and stratified plume models
    
    Create `bent_plume_model.Particle` objects from a given size distribution
    and mass flowrates.  These objects, though for the `bent_plume_model`, are
    compatible with simulations of the bent plume or stratified plume models
    
    Parameters
    ----------
    particles : list
        List to which to append new particles
    profile : ambient.Profile
        An ambient profile object for getting ambient properties
    X0 : ndarray
        Array containing the release location of the plume (x, y, z), m
    dbm_fluid : dbm.FluidParticle
        A `dbm.FluidParticle` object for the present set of particles.  This
        object differs from the `dbm.FluidMixture` in that it is expected to
        be single-phse or nearly single-phase and can report bubble and
        droplet properties.
    yk : ndarray
        Array of mole fractions for each component in the fluid mixture.  
        Theses should be the mole fractions for the single-phase fluid 
        described by the `dbm_fluid` object.
    mf : ndarray
        Array of mass flow rates (kg/s) for each bubble or droplet in the
        set of particles created by this function
    de : ndarray
        Array of corresponding equivalent spherical diameters (m) for each
        bubble or droplet in the set of particles created by this function
    Tj : float
        Temperature of the bubbles or droplets.
    lamba_1 : float, default=0.9
        Spreading ratio of dispersed phases to the entrained fluid phase.
        This value is typically between 0.9 and 1 for small particles with
        low inertia and low rise velocity.  Large bubbles or heavy sediment 
        particles may have values between 0.6 and 0.8.  The model is not very 
        sensitive to this value.
    
    Notes
    -----
    Because lists are mutable in Python, the input `particles` list will be
    modified in place by this function.  Because the input list will
    become the desired output list, we do not provide a return value so that
    the user never believes the input list would be left alone.
    
    This function creates the particles that will be released into the plume
    models. Hence, these particles must be computed at the in situ pressure of
    the ambient water at the release location. It is not possible to specify
    some other pipeline pressure when creating these bubbles or droplets
    because they will immediately be subjected to the ambient pressure in the
    plume.
    
    """
    # Make sure the input mass flow rate and diameters are lists or arrays
    if isinstance(mf, float):
        mf = np.array([mf])
    if isinstance(de, float):
        de = np.array([de])
    
    # Create each particle separately
    for i in range(len(de)):
        
        # Get the initial conditions for this particle size class
        m0, T0, nb0, Pa, Sa, Ta = dispersed_phases.initial_conditions(
            profile, X0[2], dbm_fluid, yk, mf[i], 2, de[i], Tj
        )
        
        # Use these initial conditions to create the particle object and 
        # add it to the list
        particles.append(bent_plume_model.Particle(
            X0[0], X0[1], X0[2], dbm_fluid, m0, Tj, nb0, lambda_1, Pa, Sa, Ta
        )) 
       