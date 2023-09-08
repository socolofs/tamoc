.. currentmodule:: tamoc

.. _bpm:

#######################
Bent Plume Model Module
#######################

This page is an index and launchpad to explore all of the classes, methods, and functions in the ``bent_plume_model`` module of ``tamoc``. 

.. autosummary::
   :toctree: ../generated/
   
   bent_plume_model

``bent_plume_model`` Class Objects
----------------------------------

.. currentmodule:: tamoc.bent_plume_model

.. autosummary::
   :toctree: ../generated/
   
   Model
   ModelParams
   Particle
   LagElement
   
``bent_plume_model.Model`` Methods
----------------------------------

.. autosummary::
   :toctree: ../generated/
   
   Model.simulate
   Model.get_intrusion_initial_condition
   Model.get_intrusion_concentration
   Model.get_grid_concentrations
   Model.get_planar_concentrations
   Model.get_derived_variables
   Model.save_sim
   Model.save_txt
   Model.save_derived_variables
   Model.load_sim
   Model.report_mass_fluxes
   Model.report_surfacing_fluxes
   Model.report_watercolumn_particle_fluxes
   Model.report_psds
   Model.plot_psds
   Model.plot_state_space
   Model.plot_all_variables
   Model.plot_fractions_dissolved
   Model.plot_mass_balance

``bent_plume_model.Particle`` Methods
-------------------------------------

.. autosummary::
   :toctree: ../generated/
   
   Particle.track
   Particle.outside
   Particle.run_sbm
   Particle.point_concentration
   Particle.grid_concentrations

``bent_plume_model.LagElement`` Methods
---------------------------------------

.. autosummary::
   :toctree: ../generated/
   
   LagElement.update

``bent_plume_model`` Module Helper Functions
--------------------------------------------

.. autosummary::
   :toctree: ../generated/
   
   plot_state_space
   plot_all_variables
   width_projection
   chem_idx_list