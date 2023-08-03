.. _spm_guide:

###################################
Stratified Plume Model User's Guide
###################################

The stratified plume model simulates the dynamic behavior of a multiphase plume in a stratified ambient with no crossflow (or very weak crossflow).  In these cases, multiple intrusion layers may form.  The following demonstrate how to set up, run, and interpret simulations of these cases.

Notebooks
=========

An IPython Notebook for modeling a submarine blowout has been created and is
available in the ``./notebooks`` directory of the source code distribution.
To open the notebook, execute the following command at a command prompt for
within the ``./notebooks`` directory::

   ipython notebook --pylab inline

The notebook will open in a web browser.  Click on the ``Blowout_sim`` link
on the IPy Ipython Dashboard.  This will open the Submarine Accidental Oil
Spill Calculator IPy Notebook.  Edit the executable lines that are indicated
by comments that begin as::

   # Enter...

You can run the simulation by clicking ``Run All`` from the left task panes.
You may also rename the notebook and save using the toolbar at the top of 
the page.  

Examples
========

This example illustrates the tasks necessary to setup, run, save, and 
post-process simulations using the `stratified_plume_model` module. Before 
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
`single_bubble_model` object can be initialized::

   >>> spm = stratified_plume_model.Model(ctd)

The initial conditions for the release include the depth and discharge 
characteristics.  A few are universal to the simulation and should be 
specified right away::

   >>> z0 = 1000.        # release depth (m)
   >>> R = 0.15          # radius of release region (m)
   >>> T0 = 273.15 + 30. # temperature of released fluids (K)

The remaining initial conditions are the dispersed phases that make up the 
plume.  These should be passed to the `stratified_plume_model.Model`
object as a list of `stratified_plume_model.Particle` objects. First, open an
empty list, then create `stratified_plume_model.Particle` objects and append
them to the list.

   >>> disp_phases = []

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
total bubble flux.  Two helper functions are provided in the ``TAMOC`` 
model.  These are `stratified_plume_model.particle_from_Q` and 
`stratified_plume_model.particle_from_mb0`.  The former function is more 
applicable to reservoir aeration plumes; the latter for a blowout.  In any
case, neither function is necessary as long as you can initialize a 
`stratified_plume_model.Particle` object.  Here, we use the function that
is based on the mass flux and create six different sized bubbles::

   >>> # Initial bubble diameter (m)
   >>> de = np.array([0.04, 0.03, 0.02, 0.01, 0.0075, 0.005])
   >>> # Total mass flux (kg/s) of gas in each bubble size class
   >>> m0 = np.array([0.5, 1.5, 2.5, 3.5, 1.5, 0.5] )
   >>> # Associate spreading ratio (--)
   >>> lambda_1 = np.array([0.75, 0.8, 0.85, 0.9, 0.9, 0.95])
   >>> # Append to the disp_phases list
   >>> for i in range(len(de)):
           disp_phases.append(stratified_plume_model.particle_from_mb0(ctd, 
               z0, gas, mol_frac, m0[i], de[i], lambda_1[i], T0))
   
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
``disp_phases`` list as we did for the gas bubbles. Note that all particles go
in the same list and could be in any order::

   >>> de = np.array([0.02, 0.01, 0.0075, 0.005, 0.003])
   >>> m0 = np.array([1., 2.5, 5., 1., 0.5])
   >>> lambda_1 = np.array([0.85, 0.90, 0.95, 0.95, 1.])
   >>> for i in range(len(de)):
           disp_phases.append(stratified_plume_model.particle_from_mb0(ctd, 
               z0, oil, mol_frac, m0[i], de[i], lambda_1[i], T0))

Run the Simulation
------------------

At this point, all of the initial conditions are defined, and we can run 
the simulation::

   >>> max_iterations = 15
   >>> err_tolerance = 0.2
   >>> output_dz_max = 10.
   >>> spm.simulate(disp_phases, z0, R, maxit=max_iterations, 
           toler=err_tolerance, delta_z = output_dz_max)

The above command will echo progress to the screen and produce a plot of the 
state space for each successive iteration.  

Saving Model Results
-------------------- 

The simulation results can be saved to a netCDF file, which can be used to
continue analysis within the TAMOC Python package, or an ascii text file for
importing to another analysis package, such as Excel or Matlab. To save the
data, specify a base output name with relative file path, a path to the CTD
data, and a description of the CTD data::

   >>> f_name = '../test/output/ntbk_blowout'
   >>> ctd_description = 'CTD data from Brooks McCall in file' + \
                         './test/output/test_BM54.nc'
   >>> spm.save_sim(f_name + '.nc', ctd_file, ctd_description)
   >>> spm.save_txt(f_name + 'ASCII', ctd_file, ctd_description)

