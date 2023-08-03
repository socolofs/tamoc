#############
Params Module
#############

It is important to know when to use the stratified plume model or the bent plume model.  Generally, when the crossflow is weak enough that the density stratification will cause an intrusion layer to form before the crossflow will cause the entrained water to leave the dispersed phases, the stratified plume model may be used.  Otherwise, the bent plume model is more appropriate.  The ``params`` module computes several non-dimensional scales for multiphase plumes in stratified cross-flow and predicts the length scales for trapping, peeling, and cross-flow separation.  The following examples demonstrate its usage.

Examples
========

This example illustrates the tasks necessary to setup the `params.Scales`
object and use its internal methods.  The basic purpose of this method is
to predict the characteristic scales of a plume model solution.  As a 
result, setting up the `params.Scales` object requires many of the same
tasks as setting up the `stratified_plume_model`.  For instance, we need 
to read in ambient CTD data and create a list of dispersed phase 
`stratified_plume_model.Particle` objects.  For more details on the data 
needed to run the plume models, see the examples in the :doc:`spm`.

Before running the examples here, be sure to install the ``TAMOC`` package and
run all of the tests in the ``./test`` directory. The commands below should be
executed in an IPython session. Start IPython by executing::

   ipython --pylab

at the command prompt.  

Import the ``TAMOC`` modules that will be needed for this example::

   >>> from tamoc import ambient
   >>> from tamoc import dbm
   >>> from tamoc import stratified_plume_model
   >>> from tamoc import params

To compute the "*in situ*" properties of the dispersed phase particles, we
need to have ambient CTD data::

   >>> profile = ambient.Profile('../test/output/test_BM54.nc', chem_names='all')

The next step is to select a release depth and temperature for the plume 
origin::

   >>> z0 = 1000.
   >>> T0 = 273.15 + 35.

For this example, we will consider a pure gas release with the following 
properties::

   >>> composition = ['methane', 'ethane', 'propane', 'oxygen']
   >>> yk = np.array([0.93, 0.05, 0.02, 0.0])
   >>> gas = dbm.FluidParticle(composition)

We use this gas to define a list of dispersed phase particles for the release.
In this example, we will consider one particle size class. the `scales` script
listed above demonstrates how to include multiple particles in the 
simulation::

   >>> disp_phases = []
   >>> mb0 = 8.0
   >>> de = 0.01
   >>> lambda_1 = 0.85
   >>> disp_phases.append(stratified_plume_model.particle_from_mb0(profile, z0, gas, yk, mb0, de, lambda_1, T0))

The inputs are now collected to create the `params.Scales` object::

   >>> model = params.Scales(profile, disp_phases)

The `params.Scales` object performs two basic functions.  The first job of 
the object is to compute the key independent variables describing the plume
simulation.  This is accomplished by the `params.get_variables` method.  This
method is used by all the other internal methods, and would not normally be
called by the user.  However, its functionality is demonstrated here::

   >>> (B, N, us, ua) = model.get_variables(z0, 0.15)
   >>> B
   0.85410643976609724
   >>> N
   0.001724100901081246
   >>> us
   0.22404202921415406
   >>> ua
   0.15

The second, and more important, role of the `params.Scales` object is to 
compute the characteristic scales of a given simulation setup.  For the 
geometric scales, this is achieved as follows::

   >>> h_T = model.h_T(z0)
   >>> h_T
   329.25679149878414
   >>> h_P = model.h_P(z0)
   >>> h_P
   566.48866125997949
   >>> h_S = model.h_S(z0, 0.15)
   >>> h_S
   544.7860341109531
   >>> lambda_1 = model.lambda_1(z0, 0)
   >>> lambda_1
   0.79378359736452342

For the velocity scale, there are three choices:  the ambient current (
chosen by the user), the slip velocity of the bubbles (computed by the 
`get_variables` method above), or the critical crossflow velocity, defined
as the velocity at which ``h_S = h_P``.  This is computed as follows::

   >>> ua_crit = model.u_inf_crit(z0)
   >>> ua_crit
   0.14348700121799565

