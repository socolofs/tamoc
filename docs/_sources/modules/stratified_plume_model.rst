.. currentmodule:: tamoc

.. _spm:

#############################
Stratified Plume Model Module
#############################

This page is an index and launchpad to explore all of the classes, methods, and functions in the ``stratified_plume_model`` module of ``tamoc``. 

.. autosummary::
   :toctree: ../generated/
   
   stratified_plume_model

``stratified_plume_model`` Class Objects
----------------------------------------

.. currentmodule:: tamoc.stratified_plume_model

.. autosummary::
   :toctree: ../generated/
   
   Model
   ModelParams
   InnerPlume
   OuterPlume
   
``stratified_plume_model.Model`` Methods
----------------------------------------

.. autosummary::
   :toctree: ../generated/
   
   Model.simulate
   Model.get_derived_variables
   Model.report_psds
   Model.report_intrusion_fluxes
   Model.save_sim
   Model.save_txt
   Model.save_derived_variables
   Model.load_sim
   Model.plot_state_space
   Model.plot_all_variables

``stratified_plume_model.InnerPlume`` Methods
---------------------------------------------

.. autosummary::
   :toctree: ../generated/
   
   InnerPlume.update

``stratified_plume_model.OuterPlume`` Methods
---------------------------------------------

.. autosummary::
   :toctree: ../generated/
   
   OuterPlume.update

``stratified_plume_model`` Module Helper Functions
--------------------------------------------------

.. autosummary::
   :toctree: ../generated/
   
   inner_main
   outer_main
   err_check
   particle_from_Q
   particle_from_mb0
   plot_state_space
   plot_all_variables