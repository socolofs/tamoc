#####################
Discrete Bubble Model
#####################

The simulation tools in ``tamoc`` track the trajectory and evolution of various fluid or solid particles as the transit the ocean water column from a spill source.  In order to accurately predict the rise velocity and evolution of these particles, a complete module for the equations of state of these particles is needed.  This is provided in the ``dbm`` module.  

The user interacts with this module primarily when setting up the fluids that will be used in a simulation and sometimes in post-processing simulation results.  Here is a brief introduction to the tools available in the ``dbm`` module.

Examples
========

While there are many different applications of the `dbm` module objects and
methods, here we focus on a few of the most common tasks.

The basic philosophy of the `dbm` module objects is to store all constant
chemical properties as object attributes (e.g., molecular weight, temperature
and pressure at the critical point, etc.) and to make the thermodynamic state
(e.g., temperature, pressure, salinity, etc.) as inputs to the object methods
(e.g., when calculating the density). Because this model is designed to work
with dissolving objects, the masses of each component in the mixture are also
taken as inputs to the methods. While mole could have been the fundamental
unit for composition, mass is used in ``TAMOC``.

Mixture Equations of State
--------------------------

As in the scripts above for `air_eos` and `co2_eos`, the `dbm.FluidMixture`
object provides an interface to the Peng-Robinson equation of state for any
mixture of chemicals.

As an example, consider a natural gas containing the following compounds with
given mole fractions in the mixture::

   >>> composition = ['methane', 'ethane', 'propane', 'n-butane']
   >>> yk = np.array([0.86, 0.06, 0.04, 0.04])

If the binary interaction coefficients are going to be taken as default, then
we can initialize a `dbm.FluidMixture` object as follows::

   >>> import dbm
   >>> gas = dbm.FluidMixture(composition)

This has now loaded all of the chemical properties of this mixture.  As an 
example, the molecular weights of each compound are::

   >>> gas.M
   array([ 0.0160426,  0.0300694,  0.0440962,  0.058123 ])

In order to compute thermodynamic properties, we must further define the
thermodynamic state and composition masses. The fundamental quantity
describing the variables of the mixture are the masses. In this example,
consider one mole of gas::

   >>> m = gas.masses(yk)
   >>> T = 273.15 + 10.0
   >>> P = 101325.
   >>> S = 34.5

The salinity is only used to calculate solubilities in water.  Consider 
several common properties of interest::

   >>> gas.mass_frac(yk)
   array([ 0.70070791,  0.09163045,  0.08958287,  0.11807877])
   >>> gas.mol_frac(m)
   array([ 0.86,  0.06,  0.04,  0.04])
   >>> gas.density(m, T, P)
   array([[ 0.85082097],
          [ 0.85082097]])
   >>> gas.partial_pressures(m, P)
   array([ 87139.5,   6079.5,   4053.,   4053.])
   >>> gas.fugacity(m, T, P)
   array([[ 86918.57653329,   6027.56643121,   3997.97690862,   3977.65301771],
          [ 86918.57653329,   6027.56643121,   3997.97690862,   3977.65301771]])
   >>> gas.solubility(m, T, P, S)
   array([[ 0.0205852 ,  0.00434494,  0.00355051,  0.00366598],
          [ 0.0205852 ,  0.00434494,  0.00355051,  0.00366598]])
   >>> gas.diffusivity(T)
   array([  1.82558730e-09,   1.68688060e-09,   1.35408904e-09,
            8.76029676e-10])

For those entries above with more than one row in the output, the top row
refers to the gas phase and the bottom row refers to the liquid phase. If both
rows are identical (as in this example) there is only one phase present, and
the user generally must look at the density to determine which phase (here, we
have 0.85 kg/m^3, which is a gas).

If the same mixture is brought to deepwater conditions, there would be both 
gas and liquid present::

   >>> P = 490e5
   >>> gas.density(m, T, P)
   array([[ 372.55019405],
          [ 409.80668791]])

These methods do not take steps necessary to make the gas and liquid in
equilibrium with each other. Instead, this function reports the density of gas
or liquid if each had the mass composition specified by `m`.

To evaluate the equilibrium composition, one must use the 
`dbm.FluidMixture.equilibrium` method.  As an example for this mixture::

   >>> T = 273.15 + 4.1
   >>> P = 49.5e5
   >>> gas.equilibrium(m, T, P)
   array([[ 0.01349742,  0.00168584,  0.00129938,  0.00092275],
          [ 0.00029921,  0.00011832,  0.00046447,  0.00140217]])
   
Generally, the equilibrium calculation is only meaningful when there is a 
significant two-phase region of the thermodynamic space for this mixture.  


Bubbles or Droplets
-------------------

Bubbles and droplets in general require the same set of steps; here, we focus
on a bubble with similar composition to the mixture studied above. Both
bubbles and droplets are modeled by the `dbm.FluidParticle` object. The main
differences between the `dbm.FluidParticle` and `dbm.FluidMixture` objects is
that the `dbm.FluidParticle` can only have one phase (e.g., gas or liquid) and
also contains shape information so that concepts like rise velocity have
meaning.

If we consider the same mixture as above::

   >>> composition = ['methane', 'ethane', 'propane', 'n-butane']
   >>> yk = np.array([0.86, 0.06, 0.04, 0.04])
   >>> import dbm
   >>> bub = dbm.FluidParticle(composition, fp_type=0)

We can specify the thermodynamic state similarly to before (though, consider
a hot reservoir fluid in deep water)::

   >>> T = 273.15 + 125.0
   >>> P = 150.0e5
   
The mass vector is now conceptually the masses of each component in a 
single bubble or droplet.  Typically, we know the mole or mass fractions of
the components of the bubble or droplet and a characteristic fluid particle
size.  Hence, the method `dbm.FluidParticle.masses_by_diameter` is very 
helpful for determining the actual masses of each component in the mixture::

   >>> m = bub.masses_by_diameter(0.005, T, P, yk)
   >>> print m
   [  4.54192150e-06   5.93939802e-07   5.80667574e-07   7.65375279e-07]
   
Once the masses `m` are known, it is a simple matter to determine the 
particle physical and transport attributes::

   >>> Ta = 273.15 + 4.1
   >>> Sa = 35.4
   >>> bub.density(m, T, P)
   99.036200340444182
   >>> bub.particle_shape(m, T, P, Sa, Ta)
   (2,                          # 2 : ellipsoid
    0.0050000000000000018,      # de
    99.036200340444182,         # rho_p
    1034.959691281713,          # rho_sw
    0.0015673283914517876,      # mu_sw
    0.05298375)                 # sigma
   >>> bub.slip_velocity(m, T, P, Sa, Ta)
   0.22624243729143373
   >>> bub.surface_area(m, T, P, Sa, Ta)
   7.8539816339744881e-05
   >>> bub.mass_transfer(m, T, P, Sa, Ta)
   array([  5.32070503e-05,   5.13189189e-05,   4.38949373e-05,
            3.14623966e-05])
   >>> bub.heat_transfer(m, T, P, Sa, Ta)
   array([ 0.00113312])


Insoluble Fluid Particles
-------------------------

Sometimes either a particle is truly insoluble on the time-scale of the
simulations (e.g., sand) or the composition is too complicated for the
Peng-Robinson equation of state and it is safe to neglect solubility (e.g.,
for a dead oil over short time scales). In this case, an
`dbm.InsolubleParticle` object is a simple means to capture the critical
properties of the particle yet provide an interface to the physics methods,
such as slip velocity.

Consider a sand particle.  This particle is insoluble and incompressible.
The `dbm` module can describe this particle as follows::

   >>> import dbm
   >>> sand = dbm.InsolubleParticle(isfluid=False, iscompressible=False, 
                                    rho_p=2500.)

Again, the mass must be established before the properties of the particle 
can be interrogated::

   >>> T = 273.15 + 10.0
   >>> P = 10.0e5
   >>> Ta = 273.15 + 4.1
   >>> Sa = 35.4
   >>> m = sand.mass_by_diameter(0.005, T, P, Sa, Ta)
   >>> print m
   0.000163624617374

Then, all of the fluid properties relevant to an insoluble particle can 
be calculated::

   >>> sand.density(T, P, Sa, Ta)
   2500.0
   >>> sand.particle_shape(m, T, P, Sa, Ta)
   (4,                           # 4 : rigid sphere
    0.005000000000000002,        # de
    2500.0,                      # rho_p
    1028.5585666971483,          # rho_sw
    0.0015673283914517876,       # mu_sw
    0.07423)                     # sigma
   >>> sand.slip_velocity(m, T, P, Sa, Ta)
   0.4452547003124989
   >>> sand.surface_area(m, T, P, Sa, Ta)
   7.853981633974488e-05
   >>> sand.heat_transfer(m, T, P, Sa, Ta)
   array([ 0.00155563])


   