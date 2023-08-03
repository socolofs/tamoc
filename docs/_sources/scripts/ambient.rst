.. currentmodule:: bin.ambient

.. _ambient_scripts:

###############
Ambient Scripts
###############

The ``ambient`` module in ``tamoc`` provides tools to interact with ambient profile data, including CTD data and currents.  The following scripts demonstrate the current and past ways to create and manipulate these data.  All of these methods are still supported within ``tamoc``.  To see the actual source code, consult the ``.py`` files provided in the ``./bin`` directory of ``tamoc``.

Current ``xarray`` Approach
---------------------------

The current version of the ``ambient`` module in ``tamoc`` uses the ``xarray.Dataset`` package to contain and manipulate ambient profile data.  The following scripts demonstrate ways to create profiles with these methods and add and extend data within them.

.. autosummary::
   :toctree: ../generated/
   
   profile_from_txt
   profile_from_lab
   profile_from_ctd
   profile_from_world_ocean
   profile_append
   profile_extending

Original ``netCDF`` Approach
----------------------------

The original version of the ``ambient`` module in ``tamoc`` used ``netCDF4.Dataset`` objects to contain and manipulate the profile data.  This required a file-stream and more complicated APIs to the data than the current ``xarray`` version or the more recent ``numpy`` version.  All of the old functionality is retained, and examples can be found in the following scripts.

.. autosummary::
   :toctree: ../generated/
   
   nc_profile_from_txt
   nc_profile_from_lab
   nc_profile_from_ctd
   nc_profile_append
   nc_profile_extending

Updated ``numpy`` Approach
--------------------------

Recently, the ``ambient`` module was updated to use ``numpy`` arrays instead of the ``netCDF4`` dataset objects.  This made it much easier to create profiles without having the create netCDF files.  However, these methods are not superseded by the current ``xarray`` methods.  All of the previous functionality for ``numpy`` arrays still works, and examples can be found in the following scripts.

.. autosummary::
   :toctree: ../generated/
   
   np_profile_from_txt
   np_profile_from_lab
   np_profile_from_ctd

