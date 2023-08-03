How to use this documentation
-----------------------------

The ``tamoc`` package is a suite of modules that define classes, methods, and functions that can be used to simulate a wide range of multiphase fluid dynamics in the oceans, especially for hydrocarbon fluids or atmospheric gases.  Simulations are created by joining together objects created from these classes to build simulations of specific cases.  Hence, ``tamoc`` is a scripting package.  This means both that ``tamoc`` can be configured to simulate a very large array of single particle and plume dynamics, but also that the user needs to be quite familiar with the tools in the ``tamoc`` suite and how to assemble them.

The main resources for getting to know the ``tamoc`` suite are provided in the links at the left or in the main manu of this page.  

The :ref:`start` rubric provides the license, release notes, and installation guides -- these are the preliminary things that need to be done to get ``tamoc`` up-and-running, and should be accomplished before using ``tamoc`` in the ensuing documentation examples.

The :ref:`guides` section provides iPython-like scripting examples for using each of the main modules in ``tamoc``.  They are generally organized in the order of things that need to be done to set up a simulation.  An example simulation for a subsea blowout might require:

- Assembling CTD and measured current profiles for the site of interest.
- Loading the ambient profile data into an ``ambient.Profile`` object.  An important function of the ``ambient.Profile`` object is to provide a single API to the CTD data for the ``tamoc`` suite modules, including an interpolator that can report requested profile data at any height.
- Define the contents of the released fluids and their thermodynamic state.  This is done by creating objects from the discrete particle module (``dbm``) classes.  
- Define the initial conditions for each particle that will be released.  Once the fluids are defined, particles are created using the ``dispersed_phases`` module, defining importantly the initial diameter and mass flux of each particle included in a simulation.  Because the ``tamoc`` plume models are steady-state, only one particle is required for each independent size that will be modeled.
- With a list of fluid particles and the ambient data defined, the simulation modules can be initialized.  These include the ``single_bubble_model``, the ``stratified_plume_model``, and the ``bent_plume_model``.  
- To aid with selecting the correct plume simulation module, the ``params`` module can compute several non-dimensional parameters and associated length scales, helpful for determining the relative importance of the crossflow to the stratification.
- To select the initial bubble or droplet sizes, the ``particle_size_models`` module likewise includes several empirical formulas for predicting particle sizes for immiscible jets discharged into seawater.
- Finally, because ``tamoc`` was specifically designed to simulate subsea accidental oil-well blowouts, the ``blowout`` module provides a higher-level API interface to the ``bent_plume_model``, streamlining simulation set up and post-processing.

``tamoc`` is also distributed with a set of example script files, included in the ``./bin`` directory of the main ``tamoc`` repository.  The :ref:`scripts` rubric provides an index to these scripts and their header information.

Finally, the source code is carefully documented using the ``numpydocs`` format.  All of the classes, methods, and functions that may be of interest to a ``tamoc`` user are listed in the :ref:`api` section of the ``User's Guide``.  