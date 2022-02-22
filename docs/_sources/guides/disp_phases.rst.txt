################
Dispersed Phases
################

The simulation modules in ``tamoc`` need a structured API for interacting with the properties of the fluid or solid particles in the simulation.  This is provided by the classes defined in the ``dispersed_phases`` module in ``tamoc``.  For the single bubble model, the simulation only requires one particle.  For the stratified or bent plume models, multiple particles are allowed, and these are provided through a particles list.  The following examples demonstrate creating particle objects, adding them to a list for a simulation, and extracting information from the particle objects.

Examples
========

This example illustrates the tasks involved in using the `PlumeParticle`
object to create a set of particles that will eventually be used by the
`stratified_plume_model`. The procedure is similar for the other objects in
this module, as well as for the `bent_plume_model.Particle` object.

Before running these examples, be sure to install the ``TAMOC`` package and
run all of the tests in the ``./test`` directory. The commands below should be
executed in an IPython session. Start IPython by executing::

   ipython --pylab

at the command prompt.  The ``--pylab`` flag is needed to get the correct 
behavior of the output plots.  

The first step in any spill simulation is to define the ambient CTD data. The
``TAMOC`` module `ambient` provides the tools needed to read in CTD data from
text files and organize it into the netCDF files used by the ``TAMOC``
simulation modules. Examples of how to do this are provided in the
``./bin/ambient`` directory of the TAMOC distribution. Here, we use the CTD
data created by the ``TAMOC`` test files and stored in the ``./test/output``
directory. Open a CTD file as follows (path names in this tutorial are from
any subdirectory of the ``TAMOC`` source distribution, e.g.,
``./notebooks``)::

   >>> ctd_file = '../test/output/test_BM54.nc'
   >>> ctd = ambient.Profile(ctd_file, chem_names='all')

These various particle objects are typically supplied to the models as a list.
For this example, begin with an empty list::

   >>> particles = []

Gas Bubbles 
-----------

For a blowout, we might expect both gas and oil.  Here, we create a few 
gas bubbles to add to the simulation.  The first step is to create a `dbm` 
particle object.  In this case, we choose a dissolvable particle of natural 
gas::

   >>> composition = ['methane', 'ethane', 'propane', 'oxygen']
   >>> mol_frac = [0.93, 0.05, 0.02, 0.0]
   >>> gas = dbm.FluidParticle(composition)

Next, we have to get the mass of each component in a single bubble and the
total bubble flux. A helper function is provided in the ``TAMOC`` model called
`dispersed_phases.initial_conditions`. Here, we use the function to create six
different sized bubbles::

   >>> x0 = np.array([0., 0., 1000.])
   >>> T0 = 273.15 + 30.
   >>> # Initial bubble diameter (m)
   >>> de = np.array([0.04, 0.03, 0.02, 0.01, 0.0075, 0.005])
   >>> # Total mass flux (kg/s) of gas in each bubble size class
   >>> m0 = np.array([0.5, 1.5, 2.5, 3.5, 1.5, 0.5] )
   >>> # Associate spreading ratio (--)
   >>> lambda_1 = np.array([0.75, 0.8, 0.85, 0.9, 0.9, 0.95])
   >>> # Append to the disp_phases list
   >>> for i in range(len(de)):
           m, T, nb, P, Sa, Ta = dispersed_phases.initial_conditions(
               ctd, x0[2], gas, mol_frac, m0[i], 2, de[i], T0)
           particles.append(dispersed_phases.PlumeParticle(gas, m, T, nb, 
               lambda_1[i], P, Sa,  Ta))
   
Oil Droplets
------------

Following the same procedure as for the gas but with different equations of
state, liquid droplets can be added to the simulation. Start, by defining a
new set of equations of state. Here, we assume a non-dissolving oil phase::


   >>> composition = ['inert']
   >>> rho_o = 890.     # density in kg/m^3
   >>> gamma = 30.      # API gravity in deg API
   >>> beta = 0.0007    # thermal expansion coefficient in K^(-1)
   >>> co = 2.90075e-9  # isothermal compressibility coefficient in Pa^(-1)
   >>> oil = dbm.InsolubleParticle(True, True, rho_o, gamma, beta, co)

Then, define the droplet characteristics and append them to the
``particles`` list as we did for the gas bubbles. Note that all particles go
in the same list and could be in any order::

   >>> de = np.array([0.02, 0.01, 0.0075, 0.005, 0.003])
   >>> m0 = np.array([1., 2.5, 5., 1., 0.5])
   >>> lambda_1 = np.array([0.85, 0.90, 0.95, 0.95, 1.])
   >>> for i in range(len(de)):
           m, T, nb, P, Sa, Ta = dispersed_phases.initial_conditions(
               ctd, x0[2], oil, 1., m0[i], 2, de[i], T0)
           particles.append(dispersed_phases.PlumeParticle(oil, m, T, nb, 
               lambda_1[i], P, Sa, Ta))

