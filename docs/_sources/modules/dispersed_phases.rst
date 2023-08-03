.. currentmodule:: tamoc

.. _dispersed_phases:

#######################
Dispersed Phases Module
#######################

This page is an index and launchpad to explore all of the classes, methods, and functions in the ``dispersed_phases`` module of ``tamoc``, a module for creating ``Particle`` objects used by each simulation module of the suite. 

.. autosummary::
   :toctree: ../generated/
   
   dispersed_phases

``dispersed_phases`` Class Objects
----------------------------------

.. currentmodule:: tamoc.dispersed_phases

.. autosummary::
   :toctree: ../generated/
   
   SingleParticle
   PlumeParticle

``dispersed_phases.SingleParticle`` Methods
-------------------------------------------

.. autosummary::
   :toctree: ../generated/
   
   SingleParticle.properties
   SingleParticle.diameter
   SingleParticle.biodegradation_rate

``dispersed_phases.PlumeParticle`` Methods
------------------------------------------

.. autosummary::
   :toctree: ../generated/
   
   PlumeParticle.properties
   PlumeParticle.diameter
   PlumeParticle.biodegradation_rate
   PlumeParticle.update

``dispersed_phases`` Module Helper Functions
--------------------------------------------

.. autosummary::
   :toctree: ../generated/
   
   initial_conditions
   save_particle_to_nc_file
   load_particle_from_nc_file
   shear_entrainment
   hydrate_formation_time
   zfe_volume_flux
   wuest_ic
   bf_average
   get_chem_names
   particles_state_space