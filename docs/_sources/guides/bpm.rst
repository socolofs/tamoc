.. _bpm_guide:

################
Bent Plume Model
################

The bent plume model simulates a multiphase plume release into a cross-flowing ambient, accounting for the effects of density stratification.  Unlike the stratified plume model, ambient currents are considered and only one intrusion layer is assumed to form.  The following examples demonstrate how to set up, run, and analyze bent plume model simulations.

Examples
========

This example illustrates the tasks necessary to setup, run, save, and 
post-process simulations using the `bent_plume_model` module. Before 
running these examples, be sure to install the ``TAMOC`` package and run
all of the tests in the ``./test`` directory.  The commands below should 
be executed in an IPython session.  Start IPython by executing::

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

Once the CTD data are loaded as an `ambient.Profile` object, a
`bent_bubble_model` object can be initialized::

   >>> bpm = bent_plume_model.Model(ctd)

The initial conditions for the release include the depth and discharge 
characteristics.  A few are universal to the simulation and should be 
specified right away::

   >>> x0 = np.array([0., 0., 1000.])   # release location (m)
   >>> D = 0.30                         # diameter of release region (m)
   >>> Vj = 0.                          # 0 for multiphase plume
   >>> phi_0 = -np.pi / 2               # vertical release
   >>> theta_0 = 0.                     # angle to x-axis
   >>> Sj = 0.                          # salinity of jet (psu)
   >>> Tj = 273.15 + 30.                # temperature of jet (K)
   >>> cj = 1.                          # concentration of passive tracers
   >>> tracers = ['tracers']            # tracer names

The remaining initial conditions are the dispersed phases that make up the
plume. These should be passed to the `bent_plume_model.Model` object as a list
of `bent_plume_model.Particle` objects. First, open an empty list, then create
`bent_plume_model.Particle` objects and append them to the list.

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

   >>> # Initial bubble diameter (m)
   >>> de = np.array([0.04, 0.03, 0.02, 0.01, 0.0075, 0.005])
   >>> # Total mass flux (kg/s) of gas in each bubble size class
   >>> m0 = np.array([0.5, 1.5, 2.5, 3.5, 1.5, 0.5] )
   >>> # Associate spreading ratio (--)
   >>> lambda_1 = np.array([0.75, 0.8, 0.85, 0.9, 0.9, 0.95])
   >>> # Append to the disp_phases list
   >>> for i in range(len(de)):
           m, T, nb, P, Sa, Ta = dispersed_phases.initial_conditions(
               ctd, x0[2], gas, mol_frac, m0[i], 2, de[i], Tj)
           particles.append(bent_plume_model.Particle(x0[0], x0[1], x0[2], 
               gas, m, T, nb, lambda_1[i], P, Sa, Ta))
   
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
               ctd, x0[2], oil, 1., m0[i], 2, de[i], Tj)
           particles.append(bent_plume_model.Particle(x0[0], x0[1], x0[2], 
               oil, m, T, nb, lambda_1[i], P, Sa, Ta))

Run the Simulation
------------------

At this point, all of the initial conditions are defined, and we can run 
the simulation::

   >>> dt_max = 60.     # maximum time step to take for output (s)
   >>> sd_max = 450.    # maximum number of nozzle diameters to integrate
   >>> bpm.simulate(x0, D, Vj, phi_0, theta_0, Sj, Tj, cj, tracers, 
                    particles, False, dt_max, sd_max)

The above command will echo progress to the screen

Plotting the Model Results
--------------------------

Two different methods are provided to plot the data.  To get a quick view of
the model trajectory and mass of the Lagrangian element during the simulation,
use the plot state space method::

   >>> bpm.plot_state_space(1)

To see a more comprehensive array of model output, use the plot all variables
method::

   >>> bpm.plot_all_variables(10)

Saving Model Results
-------------------- 

The simulation results can be saved to a netCDF file, which can be used to
continue analysis within the TAMOC Python package, or an ascii text file for
importing to another analysis package, such as Excel or Matlab. To save the
data, specify a base output name with relative file path, a path to the CTD
data, and a description of the CTD data::

   >>> f_name = '../test/output/test_blowout'
   >>> ctd_description = 'CTD data from Brooks McCall in file' + \
                         './test/output/test_BM54.nc'
   >>> bpm.save_sim(f_name + '.nc', ctd_file, ctd_description)
   >>> bpm.save_txt(f_name + 'ASCII', ctd_file, ctd_description)

