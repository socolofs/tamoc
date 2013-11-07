"""
Params Module
=============

Calculate the governing scales and determine the appropriate simulation model

This module calculates the governing non-dimensional parameters defined in 
papers by Socolofsky and Adams (2002, 2005) and Socolofsky et al. (2011) and
uses this information to recommend the appropriate simulation model.  

"""
# S. Socolofsky, October 2013, Texas A&M University <socolofs@tamu.edu>.

from tamoc import seawater

import numpy as np
from scipy.optimize import fsolve

class Scales(object):
    """
    Compute the characteristic scales for a ``TAMOC`` simulation
    
    Compute the governing non-dimensional parameters defined in Socolofsky
    and Adams (2002, 2005) to provide guidance on model selection for the 
    ``TAMOC`` modeling suite.
    
    Parameters
    ----------
    profile : `ambient.Profile` object
        Ambient CTD data for the model simulation
    particles : list
        List of `stratified_plume_model.Particle` objects that define all of 
        the dispersed phases in the simulation
    
    Attributes
    ----------
    profile : `ambient.Profile` object
        Ambient CTD data for the model simulation
    particles : list
        List of `stratified_plume_model.Particle` objects for the simulation
    
    """
    def __init__(self, profile, particles):
        super(Scales, self).__init__()
        
        # Store the ambient profile data and close any open netCDF files
        self.profile = profile
        profile.close_nc()
        
        # Store the dispersed phase particles
        self.particles = particles
    
    def get_variables(self, z0, u_inf):
        """
        Compute the governing variables at a given depth
        
        Compute the governing variables B (kinematic buoyancy flux), N 
        (buoyancy frequency) and u_slip (dispersed-phase slip velocity) at 
        the given depth and cross-flow velocity.  These are the main 
        ingredients to each of the scales calculations.
        
        Parameters
        ----------
        z0 : float
            Depth to evaluate the governing variables (m)
        u_inf : float
            Magnitude of the local ambient cross-flow velocity (m/s)
        
        Returns
        -------
        A tuble containing the governing variables:
            B : float
                Total kinematic buoyancy flux of all dispersed phases 
                together (m^4/s^3)
            N : float
                Local value of the ambient buoyancy frequency (1/s)
            u_slip : float
                Slip velocity of the dispersed phase containing the greatest
                buoyancy flux (m/s)
            u_inf : float
                Magnitude of the local ambient cross-flow velocity (m/s)
                TODO (S. Socolofsky, October 2013): Eventually, this should
                be read from the ambient CTD data and removed as an input to
                this method.
        
        Notes
        -----
        When more than one dispersed phase particle is present, the slip 
        velocity used as the governing variables is the value for the 
        dispersed phase particle that has the greatest effect on the dynamics
        of the plume.  This particle is the one for which the buoyancy flux
        is highest.  The governing variables, B, on the other hand is the 
        total buoyancy flux of all dispersed phase particles combined.  This 
        is consistent with the way the governing parameters have been used 
        in papers by Socolofsky and Adams (e.g., Socolofsky et al. 2011).
        
        """
        # Get the ambient data from the CTD profile
        Ta, Sa, P = self.profile.get_values(z0, ['temperature', 'salinity',
                                            'pressure'])
        rho = seawater.density(Ta, Sa, P)
        
        # Compute the properties of each dispersed-phase particle
        us = np.zeros(len(self.particles))
        rho_p = np.zeros(len(self.particles))
        m_p = np.zeros(len(self.particles))
        B_p = np.zeros(len(self.particles))
        for i in range(len(self.particles)):
            m0 = self.particles[i].m0
            T0 = self.particles[i].T0
            m_p[i] = np.sum(m0) * self.particles[i].nb0
            if m_p[i] > 0.:
                us[i], rho_p[i]= self.particles[i].properties(m0, T0, P, Sa, 
                                                               Ta)[0:2]
                B_p[i] = (rho - rho_p[i]) / rho * 9.81 * (m_p[i] / rho_p[i])
        
        # Select the correct slip velocity
        u_slip = us[0]
        for i in range(len(self.particles) - 1):
            if B_p[i+1] > B_p[i]:
                u_slip = us[i+1]
        
        # Compute the total buoyancy flux
        B = np.sum(B_p)
        
        # Get the ambient buoyancy frequency
        N = self.profile.buoyancy_frequency(z0)
        
        # Return the governing parameters
        return (B, N, u_slip, u_inf)
    
    def h_T(self, z0):
        """
        Compute the trap height for the lowest intrusion
        
        Compute the intrusion layer height above the bottom for the lowest
        intrusion based on the correlations in Socolofsky and Adams (2005).
        
        Parameters
        ----------
        z0 : float
            Depth to evaluate the trap height (m)
        
        Returns
        -------
        h_T : float
            Plume trap height for the first intrusion measured in height 
            above the bottom (m)
        
        """
        # Get the governing variables
        (B, N, u_slip, u_inf) = self.get_variables(z0, 0.)
        
        # Compute U_N
        U_N = u_slip / (B * N)**(1./4.)
        
        # Compute the correlation equation
        return 2.9 * np.exp(-(U_N - 1.0)**2 / 28.09) * (B / N**3)**(1./4.)
    
    def h_P(self, z0):
        """
        Compute the peel height for the lowest intrusion
        
        Compute the height above the bottom for the peeling region that forms
        the lowest intrusion based on the correlations in Socolofsky and Adams 
        (2005).
        
        Parameters
        ----------
        z0 : float
            Depth to evaluate the trap height (m)
        
        Returns
        -------
        h_P : float
            Plume peel height for the first intrusion measured in height 
            above the bottom (m)
        
        """
        # Get the governing variables
        (B, N, u_slip, u_inf) = self.get_variables(z0, 0.)
        
        # Compute U_N
        U_N = u_slip / (B * N)**(1./4.)
        
        # Compute the correlation equation
        return 5.2 * np.exp(-(U_N - 1.8)**2 / 10.24) * (B / N**3)**(1./4.)
    
    def h_S(self, z0, u_inf):
        """
        Compute the cross-flow separation height
        
        Compute the height above the bottom where the cross-flow causes 
        separation of the entrained plume fluid from the plume based on the
        correlation in Socolofsky and Adams (2002).
        
        Parameters
        ----------
        z0 : float
            Depth to evaluate the trap height (m)
        u_inf : float
            Magnitude of the local ambient cross-flow velocity (m/s)
        
        Returns
        -------
        h_S : float
            Cross-flow separation height measured in height above the bottom 
            (m)
        
        """
        # Get the governing variables
        (B, N, u_slip, u_inf) = self.get_variables(z0, u_inf)
        
        # Compute the correlation equation
        return 5.1 * B / (u_inf * u_slip**2.4)**(0.88)
    
    def lambda_1(self, z0, n):
        """
        Compute the spreading ratio of particle n at the given depth
        
        Compute the spreading ratio lambda_1 for particle n at the given 
        depth z0 from the correlation equations in Socolofsky and Adams 
        (2005).
        
        Parameters
        ----------
        z0 : float
            Depth to evaluate the trap height (m)
        n : int
            Index to the self.Particle object for which the spreading ratio
            should be calculated
        
        Returns
        -------
        lambda_1 : float
            Dispersed-phase spreading ratio lambda_1 for the selected 
            dispersed-phase particle.
        
        """
        # Get the governing variables
        (B, N, u_slip, u_inf) = self.get_variables(z0, 0.)
        
        # Compute the slip velocity for the selected dispersed-phase particle
        Ta, Sa, P = self.profile.get_values(z0, ['temperature', 'salinity',
                                            'pressure'])
        u_slip = self.particles[n].properties(self.particles[n].m0, 
                                              self.particles[n].T0, P, 
                                              Sa, Ta)[0]
        
        # Compute the particle value of U_N
        U_N = u_slip / (B * N)**(1./4.)
        
        # Get the spreading ratio
        return 1.0 - 0.19 * U_N**(0.61)
    
    def u_inf_crit(self, z0):
        """
        Determine the critical cross-flow velocity
        
        Calculate the critical value of the cross-flow velocity for which 
        the peel height matches the cross-flos separation height.
        
        Parameters
        ----------
        z0 : float
            Depth to evaluate the trap height (m)
        
        Returns
        -------
        u_inf_crit : float
            Crossflow velocity for which h_P = h_S
        
        
        """
        # Get h_P, which is independent of the crossflow velocity
        h_P = self.h_P(z0)
        
        # Define an objective function for root finding
        def residual(us):
            """
            Residual for use in root finding to find u_inf_crit
            
            Returns the difference h_S - h_P, which should be zero at the 
            critical cross-flow velocity.
            
            """
            return self.h_S(z0, us) - h_P
        
        # Return the critical crossflow velocity
        return fsolve(residual, 0.05)
    


