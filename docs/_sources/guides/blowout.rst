##############
Blowout Module
##############

Subsea accidental oil well blowouts can be simulated using the bent plume model in ``tamoc``.  Many steps for setting up bent plume model simulations are similar for any blowout, and the ``blowout`` module in ``tamoc`` attempts to automate these steps and provide a higher-level API interface to the bent plume model module.  The following examples demonstrate how this more intuitive class object can be used to simulate an oil well blowout.

Examples
========

This example demonstrates the tasks necessary to set up and run a subsea 
accidental oil well blowout using the `blowout.Blowout` class.  While this class does provide an interface to the National Oceanic and Atmospheric Administration (NOAA) Oil Library, in this example, we will restrict our usage to chemical properties distributed with ``TAMOC``.  To use the NOAA Oil Library, install the OilLibrary package, available at::

    https://github.com/NOAA-ORR-ERD/OilLibrary

Then, in the follow example, set the `substance` variable to the Adios ID
number of the oil you want to import from the Oil Library (e.g., 'AD01554').

Before running these examples, be sure to install the ``TAMOC`` package and
run all of the tests in the ``./test`` directory. The commands below should
be executed in an IPython session. Start IPython by executing::

   ipython --pylab

at the command prompt.  The ``--pylab`` flag is needed to get the correct 
behavior of the output plots.  In iPython, we also import the `blowout` module::

    >>> from tamoc import blowout

With these preliminaries completed, we are ready to work and example.

Define the Parameters for a Simulation
--------------------------------------

In order to carefully define each of the parameters required to initialize a
`blowout.Blowout` object, we will store the information in a set of
variables. To begin, we define the water depth at the release and the
equivalent spherical diameter of the release orifice::

    >>> z0 = 500.
    >>> d0 = 0.15

There are multiple options for defining the substance that is being spilled.
Here, we will use the oil properties for components included in the ``TAMOC``
chemical properties database and define a simple light oil::

    >>> composition = ['n-hexane', '2-methylpentane', '3-methylpentane',
                       'neohexane', 'n-heptane', 'benzene', 'toluene', 
                       'ethylbenzene', 'n-decane']
    >>> masses = np.array([0.04, 0.07, 0.08, 0.09, 0.11, 0.12, 0.15, 0.18,
                           0.16])

We pass this information to the `blowout.Blowout` class initializer in a
Python dictionary with the following format::

    >>> substance = {'composition' : composition, 
                     'masses' : masses}

Next, we define the oil and gas flow rate. Typically for an oil well, this is
known in terms of the amount of dead oil and free gas produced at standard
conditions. The oil flow rate is in stock barrels per day (bbl/d) and the gas
flow rate is in standard cubic feet per stock barrel of oil (ft^3/bbl). Here,
we will consider a flow rate of 20,000 bbl/d of oil with a gas-to-oil ratio
(GOR) of 1000 ft^3/bbl::

    >>> q_oil = 20000.
    >>> gor = 1000.

We can also specify the flow rate of produced water exiting with the oil and
gas. In this case, we give the velocity of the water at the release. Here, we
will set this to zero::

    >>> u0 = 0.

The ``TAMOC`` models work on a local Cartesian coordinate system. Normally,
we set the `x`- and `y`-components of the release to (0,0). The orientation
of the release in the `bent_plume_model` is defined in spherical coordinates
with `z` positive down. Thus, a vertical release has `phi` equal to -pi/2. In
the case of a vertical release, the horizontal angle can be arbitrary, hence,
this is set here to zero::

    >>> x0 = 0.
    >>> y0 = 0.
    >>> phi_0 = -np.pi / 2.
    >>> theta_0 = 0.

The `Blowout.blowout` class objects compute their bubble and droplet size
distributions automatically using the `particle_size_models` module. Here, we
only need to decide how many size classes we want for gas bubbles and oil
droplets::

    >>> num_gas_elements = 10
    >>> num_oil_elements = 15

Finally, we have to provide the ambient CTD and current data to be used in
the simulation. The `blowout` module provides significant flexibility in
defining the profile information. Here, we will load the world-ocean average
CTD data and specify a uniform current for lack of any more specific data. We
can set these parameters as follows::

    >>> water = None # requests ``TAMOC`` to load the default data
    >>> current = np.array([0.07, 0.12, 0.0])

Initialize and Run a Blowout Simulation
---------------------------------------

With the above input parameters defined, we can initialize a
`blowout.Blowout` object as follows::

    >>> spill = blowout.Blowout(z0, d0, substance, q_oil, gor, x0, y0, u0,
                                phi_0, theta_0, num_gas_elements, 
                                num_oil_elements, water, current)

The well blowout scenario is now ready to run for the present set of 
conditions.  We run the simulation with::

    >>> spill.simulate()

To see the trajectory of the simulated result for the plume-part only, we can
plot the state space::

    >>> spill.plot_state_space(1)

We could also plot the full solution suite, which includes the trajectory of
all particles that exited the plume::

    >>> spill.plot_all_variables(10)

If we want to change something about the simulation, we should use the
various `update()` methods. For example, we could change the flow rate and
ambient currents using::

    >>> spill.update_q_oil(30000.)
    >>> spill.update_gor(875.)
    >>> spill.update_current_data(np.array([0.11, 0.06, 0.]))

To see the results of the simulation for these new conditions, we re-run the
simulation and then re-plot::

    >>> spill.simulate()
    >>> spill.plot_state_space(101)
    >>> spill.plot_all_variables(110)

Example using the OilLibrary
----------------------------

If you have the `NOAA OilLibrary` installed, then you can re-run the above simulation using Louisiana Light Sweet crude oil by updating the substance to the corresponding Adios ID (e.g., 'AD01554')::

    >>> spill.update_substance('AD01554')
    >>> spill.simulate()
    >>> spill.plot_all_variables(210)

