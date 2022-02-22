.. currentmodule:: tamoc

.. _dbm:

########################
Discrete Particle Module
########################

This page is an index and launchpad to explore all of the classes, methods, and functions in the discrete particle module, or ``dbm`` module of ``tamoc``. 

.. autosummary::
   :toctree: ../generated/
   
   dbm

``dbm`` Class Objects
---------------------

.. currentmodule:: tamoc.dbm

.. autosummary::
   :toctree: ../generated/
   
   FluidMixture
   FluidParticle
   InsolubleParticle

``dbm.FluidMixture`` Methods
----------------------------

.. autosummary::
   :toctree: ../generated/
   
   FluidMixture.masses
   FluidMixture.mass_frac
   FluidMixture.moles
   FluidMixture.mol_frac
   FluidMixture.partial_pressures
   FluidMixture.density
   FluidMixture.fugacity
   FluidMixture.viscosity
   FluidMixture.interface_tension
   FluidMixture.equilibrium
   FluidMixture.solubility
   FluidMixture.diffusivity
   FluidMixture.hydrate_stability
   FluidMixture.biodegradation_rate

``dbm.FluidParticle`` Methods
-----------------------------

.. autosummary::
   :toctree: ../generated/
   
   FluidParticle.masses
   FluidParticle.mass_frac
   FluidParticle.moles
   FluidParticle.mol_frac
   FluidParticle.partial_pressures
   FluidParticle.density
   FluidParticle.fugacity
   FluidParticle.viscosity
   FluidParticle.interface_tension
   FluidParticle.equilibrium
   FluidParticle.solubility
   FluidParticle.diffusivity
   FluidParticle.hydrate_stability
   FluidParticle.biodegradation_rate
   FluidParticle.masses_by_diameter
   FluidParticle.diameter
   FluidParticle.particle_shape
   FluidParticle.slip_velocity
   FluidParticle.surface_area
   FluidParticle.mass_transfer
   FluidParticle.heat_transfer
   FluidParticle.return_all


``dbm.InsolubleParticle`` Methods
---------------------------------

.. autosummary::
   :toctree: ../generated/
   
   InsolubleParticle.density
   InsolubleParticle.viscosity
   InsolubleParticle.interface_tension
   InsolubleParticle.biodegradation_rate
   InsolubleParticle.mass_by_diameter
   InsolubleParticle.diameter
   InsolubleParticle.particle_shape
   InsolubleParticle.slip_velocity
   InsolubleParticle.surface_area
   InsolubleParticle.heat_transfer
   InsolubleParticle.return_all

``dbm`` Module Helper Functions
-------------------------------

.. autosummary::
   :toctree: ../generated/
   
   equil_MM
   stability_analysis
   successive_substitution
   gas_liq_eq