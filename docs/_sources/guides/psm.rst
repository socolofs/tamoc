####################
Particle Size Models
####################

All of the simulation models in ``tamoc`` require the use to specify the bubble, droplet, or particle sizes at the beginning of the simulation.  The ``particle_size_models`` module in ``tamoc`` codes several empirical equations in the literature for predicting the bubble or droplet sizes of fluids released from a jet nozzle into seawater.  The following examples show how to use the module class objects to get initial particle size distributions.

Examples
========

This example demonstrates the tasks necessary to compute particle size 
distributions either from specified fluid properties (e.g., using the 
`ModelBase` or `PureJet` classes) or from fluid properties computed by 
objects in the `dbm` module (e.g., using the `Model` class).  Before 
running these examples, be sure to install the ``TAMOC`` package and run
all of the tests in the ``./test`` directory.  The commands below should 
be executed in an IPython session.  Start IPython by executing::

   ipython --pylab

at the command prompt.  The ``--pylab`` flag is needed to get the correct 
behavior of the output plots.

Using Specified Fluid Properties
--------------------------------

Sometimes, especially when comparing breakup models to measured data, we
would like to specify all of the fluid properties to use in the breakup
calculation.  We may do this using the `ModelBase` (which can handle
releases of both oil and gas) or the `PureJet` (which handles single-phase
releases) classes.  In this example, we will use the `ModelBase` class.

Start by importing the `particle_size_models` module and then defining the
properties of the fluids at the release::

    >>> from tamoc import particle_size_models as psm
    >>> # Gas Properties ----------
    >>> rho_gas = 131.8
    >>> mu_gas = 0.00002
    >>> sigma_gas = 0.06
    >>> # Oil Properties ----------
    >>> rho_oil = 599.3
    >>> mu_oil = 0.0002
    >>> sigma_oil = 0.015
    >>> # Seawater Properties -----
    >>> rho = 1037.1
    >>> mu = 0.002

Next, we create a `ModelBase` object that contains these fluid property
data::

    >>> spill = psm.ModelBase(rho_gas, mu_gas, sigma_gas, rho_oil, mu_oil,
                              sigma_oil, rho, mu)

We can now use this object to compute size distributions for a variety of
situations as long as these fluid properties do not change. Generally, these
properties would change is the release temperature or the release depth were
to change.

As an example, let's compute the characteristic values of the particle 
size distributions for a 0.30 m diameter orifice with specified gas and oil
mass flow rates of 7.4 kg/s and 34.5 kg/s, respectively::

    >>> m_gas = 7.4
    >>> m_oil = 34.5
    >>> spill.simulate(0.30, m_gas, m_oil)

The `.simulate()` method does not return any output.  Rather, the results 
are stored in the object attributes.  We may view these attribute values
either by printing them directly or using the `.get`-methods.  For example, 
the median particle sizes are::

    >>> spill.d50_gas
    0.01134713688939418
    >>> spill.get_d50(0)  # 0 = gas, 1 = liquid
    spill.get_d50(0)
    >>> spill.d50_oil
    0.0033149657926870454
    >>> spill.get_d50(1)
    0.0033149657926870454

The `.simulate()` method also computed the characteristic width of the 
particle size distributions.  To compute the distributions, we use the 
`get_distributions()` method, which returns the sizes and volume fractions
of the gas bubble and oil droplet size distributions, as follows::

    >>> de_gas, vf_gas, de_oil, vf_oil = spill.get_distributions(10, 15)
    >>> de_gas
    array([0.0057077 , 0.00655033, 0.00751736, 0.00862716, 0.0099008 ,
           0.01136247, 0.01303992, 0.01496502, 0.01717432, 0.01970979])
    >>> vf_gas
    array([0.01545088, 0.0432876 , 0.09350044, 0.15570546, 0.19990978,
           0.19788106, 0.15101303, 0.08885147, 0.04030462, 0.01409565])
    >>> de_oil
    array([0.00035434, 0.00044692, 0.00056369, 0.00071098, 0.00089675,
           0.00113105, 0.00142658, 0.00179932, 0.00226946, 0.00286243,
           0.00361034, 0.00455367, 0.00574348, 0.00724417, 0.00913696])
    >>> vf_oil
    array([0.00522565, 0.00788413, 0.01185467, 0.01773296, 0.02631885,
           0.03859967, 0.05559785, 0.07791868, 0.10476347, 0.13228731,
           0.15193437, 0.15128424, 0.12160947, 0.0710618 , 0.02592687])

It is easy to interpret these distributions after they are plotted.  Use the
`plot_psd()` method to see a default presentation of the data::

    >>> spill.plot_psd(1)

Using Fluid Properties from the `dbm` Module
--------------------------------------------

When using the plume models in ``TAMOC`` (e.g., the `bent_plume_model` or 
the `stratified_plume_model`), it is important that the fluid properties 
used to compute the model initial conditions matches that used in the plume
simulations.  The `Model` class is designed to provide this functionality.

As an example, let's consider a natural gas pipeline leak.  As with most 
models in ``TAMOC``, we need ambient CTD data before we can start with any
of the other calculations.  The `ambient` module now provides default 
world-ocean average CTD data when no other data source is available.  To 
create an `ambient.Profile` object using this built-in data, do the 
following::

    >>> from tamoc import ambient
    >>> profile = ambient.Profile(None, current=np.array([0.05, 0.1, 0]), 
                                  current_units = 'm/s')

We can test this data by requesting the properties at 50 m water depth::

    >>> T, S, u, v = profile.get_values(50., ['temperature', 'salinity', 
                                        'ua', 'va'])
    >>> T
    288.23999999999995
    >>> S
    35.01
    >>> u
    0.05
    >>> v
    0.1

We also need to create a natural gas `dbm.FluidMixture` object and specify
a mass flux of 5.5 kg/s of gas::

    >>> from tamoc import dbm, dbm_utilities
    >>> gas_comp, gas_frac = dbm_utilities.natural_gas()
    >>> gas = dbm.FluidMixture(gas_comp)
    >>> m_gas = 5.5 * gas_frac

With the profile and `dbm` object created, we can now create the
`particle_size_model.Model` object::

    >>> leak = psm.Model(profile, gas, m_gas, 50.)

Once we create the `Model` object, it can be used similarly to the
`ModelBase` object, but without having to specify the mass flux anymore. If
we want to change the mass flux, we need to use the `update_m_mixture()`
method. For instance, we can compute the characteristic particle sizes
through a 5 cm diameter hole as follows::

    >>> leak.simulate(0.05)
    >>> leak.d50_gas
    >>> 0.005861081233586573
    >>> leak.get_distributions(15, 0)
    >>> leak.plot_psd(2, 0)

If we want to change the orifice size, then we would do the following::

    >>> leak.simulate(0.1)
    >>> leak.get_distributions(15, 0)
    >>> leak.plot_psd(3,0)

Or, if we wanted to reduce the mass flux to 2.3 kg/s, then we would do the
following:

    >>> m_gas = 2.3 * gas_frac
    >>> leak.update_m_mixture(m_gas)
    >>> leak.simulate(0.1)
    >>> leak.get_distributions(15, 0)
    >>> leak.plot_psd(4,0)

This example demonstrated a pure gas plume.  Since the `Model` class takes
a `dbm.FluidMixture` object as input, it can automatically also consider
a release of oil and gas.  As a quick example, consider the following::

    >>> oil = dbm.FluidMixture(['methane', 'decane'])
    >>> m_mix = np.array([0.3, 0.7]) * 18.5
    >>> blowout = psm.Model(profile, oil, m_mix, 1000.)
    >>> blowout.simulate(0.15)
    >>> blowout.get_distributions(15, 15)
    >>> blowout.plot_psd(5)


