###############################
Equations of State for Seawater
###############################

Throughout the ``tamoc`` simulation suite, the different modules need access to properties for seawater.  The ``seawater`` module provides these properties.  These may be needed by the user when processing ambient profile data, but otherwise, these functions may simply operate behind the scenes.  

Examples
--------

The `seawater` module provides a few simple equations of state for seawater.
See the code documentations under the API Reference pages for details.

In each example, the thermodynamic state is needed as input.  Consider a 
typical deepwater condition as follows::

   >>> T = 273.15 + 4.1   # K
   >>> P = 150e5          # 150 bar expressed in Pa
   >>> S = 35.4           # psu

The various properties of seawater at this state are obtained as follows::

   >>> import seawater
   >>> seawater.density(T, S, P)
   1034.959691281713
   >>> seawater.mu(T)
   0.0015673283914517876
   >>> seawater.sigma(T)
   0.0750703665
   >>> seawater.cp()
   4185.5
   >>> seawater.k()
   1.46e-07

