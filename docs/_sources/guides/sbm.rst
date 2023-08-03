.. _sbm_guide:

###################
Single Bubble Model
###################

The single bubble model simulates the trajectory of an evolving fluid or solid particle along its Lagrangian path.  The following shows how to set up, run, and explore the results of a single bubble model simulation.

Examples
========

This example illustrates the tasks necessary to setup, run, save, and 
post-process simulations using the `single_bubble_model` module.  A wide 
class of bubble, droplet, or particle compositions and initial conditions 
can be simulated by the `single_bubble_model`.  In each case, the simulation
considers the rise velocity, dissolution, and heat transfer of a single
particle assuming it rises through a quiescent fluid.  This model cannot
consider multiple particle (run multiple simulations to get the results
for each desired particle type) and, thus, includes no particle-particle 
interaction.

Initialize the `single_bubble_model.Model` Object
-------------------------------------------------

There are two ways to initialize a `single_bubble_model` object. When a new
simulation will be made, this should be done by specifying the ambient
conditions data that will be used in the simulation. Alternatively, if a
previous simulation is to be reloaded for post-processing, then the filename
of the netCDF dataset containing the results is used to initialize the object.
Here, we use the profile data. In a later section of this example, once the
simulation data have been save, the second method of using the saved data to
create the `single_bubble_model.Model` object is demonstrated::

   >>> import ambient
   >>> import single_bubble_model
   >>> nc = '.test/output/test_bm54.nc'
   >>> bm54 = ambient.Profile(nc, chem_names='all')
   >>> bm54.close()
   >>> sbm = single_bubble_model.Model(profile=bm54)

Setup and Run a Simulation
--------------------------

To run a simulation, one must pass the initial conditions to the 
`single_bubble_model.simulate` method.  Here, we specify the initial 
conditions as::

   >>> composition = ['methane', 'oxygen']
   >>> mol_frac = np.array([1.0, 0.0])
   >>> de = 0.005
   >>> z0 = 1500.
   >>> T0 = None       # T0 will be set equal to the ambient temperature

The `single_bubble_model` expects the particle information to be passed 
as a `dbm.FluidParticle` or `dbm.InsolubleParticle` object.  Create a soluble
particle for this example::

   >>> import dbm
   >>> bub = dbm.FluidParticle(composition)

The `single_bubble_model` handles the conversion from initial diameter to 
the initial masses, so we can now run the simulation::

   >>> sbm.simulate(bub, z0, de, mol_frac, T0, fdis=1e-8, delta_t=10)

After executing the above command, the model will echo its progress to the 
screen.  Following the simulation, the data will be plotted showing the 
state space variables and several other derived quantities in three different
figure windows.

Saving and Loading Simulation Results
-------------------------------------

To save the simulation results in a netCDF dataset file that can also be used
to recreate the current `single_bubble_model.Model` object, use the 
`single_bubble_model.save_sim` method::

   >>> nc_file = './sims/bubble_path.nc'
   >>> profile = './test/output/test_bm54.nc'
   >>> sim_info = 'Sample results from the documentation examples'
   >>> sbm.save_sim(nc_file, profile, sim_info)

The data can also be saved as ``ASCII`` text in a format that is readable by, 
for example, Matlab.  If `numpy` version 1.7.0 is used, a header with the
file metadata can be written; otherwise, only the data table can be written.
In either case, the function call is the same::

   >>> sbm.save_txt(nc_file, profile, sim_info)

If the netCDF dataset object is used, this can be used later to reload the
simulation into the `single_bubble_model.Model` object.  Since the netCDF
dataset is self-documenting, this can be done simply by passing the file
name of the netCDF dataset to the `single_bubble_model.Model` constructor::

   >>> sbm_old = single_bubble_model.Model('./sims/bubble_path.nc')

