.. currentmodule:: bin.bpm

.. _bpm_scripts:

########################
Bent Plume Model Scripts
########################

The ``bent_plume_model`` in ``tamoc`` is the primary tools to simulate multiphase plumes in stratified crossflow.  This is the model that would normally be used to simulate a subsea blowout.  The followings scripts demonstrate its use.  The ``blowout`` module is built on the bent plume model, and provides a shortcut for many of the standard things that need to be done to simulate a blowout (see also _blowout_scripts).  To see the actual source code of these scripts, consult the ``.py`` files provided in the ``./bin`` directory of ``tamoc``.

.. autosummary::
   :toctree: ../generated/
   
   crossflow_plume
   bpm_blowout_sim
   CO2

