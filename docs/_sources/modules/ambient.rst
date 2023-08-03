.. currentmodule:: tamoc

.. _ambient:

##############
Ambient Module
##############

This page is an index and launchpad to explore all of the classes, methods, and functions in the ``ambient`` module of ``tamoc``. 

.. autosummary::
   :toctree: ../generated/
   
   ambient

``ambient`` Class Objects
---------------------------------

.. currentmodule:: tamoc.ambient

.. autosummary::
   :toctree: ../generated/
   
   Profile
   BaseProfile

``ambient.Profile`` Methods
---------------------------

.. autosummary::
   :toctree: ../generated/
   
   Profile.get_values
   Profile.get_units
   Profile.append
   Profile.extend_profile_deeper
   Profile.add_computed_gas_concentrations
   Profile.buoyancy_frequency
   Profile.insert_density
   Profile.insert_potential_density
   Profile.insert_buoyancy_frequency
   Profile.plot_parameter
   Profile.plot_profiles
   Profile.plot_physical_profiles
   Profile.plot_chem_profiles
   Profile.close_nc

``ambient`` Module Helper Functions
-----------------------------------

.. autosummary::
   :toctree: ../generated/
   
   xr_check_units
   xr_convert_units
   xr_coarsen_dataset
   xr_stabilize_dataset
   xr_dataset_to_array
   xr_array_to_dataset
   xr_add_data_from_numpy
   create_nc_db
   fill_nc_db
   fill_nc_db_variable
   get_nc_data
   get_xarray_data
   add_data
   extract_profile
   coarsen
   stabilize
   compute_pressure
   convert_units
   get_world_ocean
   load_raw
   
   


