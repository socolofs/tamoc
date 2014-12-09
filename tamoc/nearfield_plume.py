"""
Nearfield Plume
===============

Handle decisions and perform simulations for a multiphase plume

This models takes the initial conditions describing an multiphase plume and 
makes the appropriate choices to run the applicable models in order to 
generate the nearfield solution.

This module can run simulations with each of the nearfield modeling modules
in TAMOC, including the single bubble model, stratified plume model, or the
bent plume model.  It can also run multiple iterations of a model (e.g., it
can run the bent plume model more than once to create multiple intrusions
in a crossflow environment).  It makes choices as to which model to use based
on scaling analysis contained within `params`.  

See Also
--------
`single_bubble_model` : Tracks the trajectory of a single bubble, drop or 
    particle through the water column.  The numerical solution used here, 
    including the various object types and their functionality, follows the
    pattern in the `single_bubble_model`.  The main difference is the more
    complex state space and governing equations.

`stratified_plume_model` : Predicts the plume solution for quiescent ambient
    conditions or weak crossflows, where the intrusion (outer plume) 
    interacts with the upward rising plume in a double-plume integral model
    approach.  Such a situation is handeled properly in the 
    `stratified_plume_model` and would violate the assumption of non-
    iteracting Lagrangian plume elements as required in this module.

`bent_plume_model` : Predicts the plume solution for crossflowing ambient 
    conditions, where currents cause the plume to bend in the downstream 
    direction.  This model is of the Lagrangian integral plume type and
    assumes that each Lagrangian element does not interact with any other 
    Lagrangian elements.
"""
# S. Socolofsky, November 2014, Texas A&M University <socolofs@tamu.edu>.

from tamoc import params
from tamoc import single_bubble_model
from tamoc import stratified_plume_model
from tamoc import bent_plume_model

import numpy as np

class Model(object):
    """
    docstring for Model
    
    """
    def __init__(self, profile, X, D, Vj, phi_0, theta_0, Sj, Tj, 
                 particles, profile_path, profile_info):
        super(Model, self).__init__()
        
        # Make sure the position is an array
        if not isinstance(X, np.ndarray):
            if not isinstance(X, list):
                # Assume user specified the depth only
                X = np.array([0., 0., X])
            else:
                X = np.array(X)
        
        # Make sure the particles are a list
        if not isinstance(particles, list):
            particles = [particles]
        
        # Store the object initialization data
        self.profile = profile
        self.profile_path = profile_path
        self.profile_info = profile_info
        profile.close_nc()
        self.X = X
        self.D = D
        self.Vj = Vj
        self.phi_0 = phi_0
        self.theta_0 = theta_0
        self.Sj = Sj
        self.Tj = Tj
        self.particles = particles
    
    def simulate(self):
        """
        docstring for simulate
        
        """
        # Iteratively run the bent plume model
        
        # Create a list for storing model solutions
        bpm = []
        
        # Create and run the fundamental solution
        bpm.append(bent_plume_model.Model(self.profile))
        bpm[0].simulate(self.X, self.D, self.Vj, self.phi_0, self.theta_0, 
                        self.Sj, self.Tj, 1., 'tracer', self.particles,
                        track=False, dt_max=60., sd_max=4*self.X[2]/self.D)
        
        # Sort the particles into potential new simulations
        (p_lists, X, D, T) = sort_particles(bpm[0].particles)
        
        # Run each of those simulations
        for i in range(len(p_lists)):
            bpm.append(bent_plume_model.Model(self.profile))
            bpm[-1].simulate(X[i,:], D[i], None, -np.pi / 2., 0., T[i], 1., 
                'tracer', p_lists[i], track=False, dt_max=60., 
                sd_max=4*X[i,2]/D[i])
        
        # Next, we would need to run a bpm for each particle within each 
        # of the simulations we just ran
        # This is getting pretty much impossible...move this to a new object
        # and make this object capable of deciding whether or not the plume
        # if viable and which model to run...
    
    def empirical_model(self, ua=None, report=False, base_name=None):
        """
        Run the empirical model defined by the scale equations
        
        """
        # Create a params.Scales object for the empirical model
        epm = params.Scales(self.profile, self.particles)
        
        # Get the ambient velocity
        if ua is None:
            self.ua = self.profile.get_values(self.X[2], 'ua')
        
        # Compute the predictions from the model
        self.h_T = epm.h_T(self.X[2])
        self.h_P = epm.h_P(self.X[2])
        self.h_S = epm.h_S(self.X[2], ua)
        self.u_inf_crit = epm.u_inf_crit(self.X[2])
        
        # Echo solution to the screen if requested
        if report:
            epm.simulate(self.X[2], ua)
        
        # Save the data if requested
        if base_name is not None:
            epm.save_txt(self.X[2], ua, base_name, self.profile_path, 
                         self.profile_info)
    
    def correct_lambda_1(self):
        """
        Use `params` to store the correct value of `lambda_1` in particles
        
        """
        # Create a params.Scales object for the empirical model
        epm = params.Scales(self.profile, self.particles)
        
        # Iteratively fill the particles objects with the correct value of
        # lambda_1
        for i in range(len(self.particles)):
            self.particles[i].lambda_1 = epm.lambda_1(self.X[2], i)
    

def sort_particles(particles):
    """
    docstring for sort_particles
    
    """
    # Create a look-up table for the plume to which each particle belongs
    plumes = np.arange(len(particles)) - np.arange(len(particles))
    matches = 0
    
    # Sort the particles into plumes
    for i in range(len(particles)):
        
        if plumes[i] == i:
            # This particle has not been assigned to a plume
            plumes[i] = i - matches
            
            # Check the remaining particles in the list        
            for j in range(i, len(particles)):
                
                # Get the distance between particles i and j
                L = np.sqrt((particles[i].x - particles[j].x)**2 + 
                    (particles[i].y - particles[j].y)**2+ (particles[i].z - 
                    particles[j].z)**2)
                
                if L < particles[i].b_local:
                    # These particles are in the same plume
                    plumes[j] = plumes[i]
                    matches += 1
    
    # Create separate particles lists for each new plume simulation
    p_list = []
    X = []
    D = []
    T = []
    for i in range(np.max(plumes)+1):
        
        # Insert a blank list for this plume
        p_list.append([])
        X.append([0., 0., 0.])
        D.append(0.)
        T.append(0.)
        # Select the particles that go in this list
        for j in range(len(particles)):
            if plumes[j] == i:
                
                # This particle (j) belongs in this plume (i)
                p_list[-1].append(particles[j])
                X[-1][0] += particles[j].x
                X[-1][1] += particles[j].y
                X[-1][2] += particles[j].z
                D[-1] += 2 * particles[j].b_local
                T[-1] += particles[j].T
    
    # Convert X and D or numpy arrays and compute their average values
    X = np.array(X)
    D = np.array(D)
    T = np.array(T)
    for i in range(len(p_list)):
        X[i,:] = X[i,:] / np.float(len(p_list[i]))
        D[i] = D[i] / np.float(len(p_list[i]))
        T[i] = T[i] / np.float(len(p_list[i]))
    
    # Return the particle list
    return (p_list, X, D, T)
    
    
    
    
    
    
    
    
    
        