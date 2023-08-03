tamoc:  The Texas A&M Oil spill / Outfall Calculator
====================================================

**tamoc** is an open-source modeling suite for simulating multiphase plumes of bubbles, droplets, or particles in sea water.  It was developed to simulate subsea accidental oil well blowouts, but also includes algorithms for single-phase outfall plumes and a complete equation of state for hydrocarbon fluids and atmospheric gases.  

The ``tamoc`` suite includes helper modules, which create data structures for handling equations of state, chemical properties, and ambient ocean profile data, and simulation modules, including

- :ref:`sbm_guide`:  A model to track the Lagrangian pathway of a single bubble, droplet, or particle as it evolves by dissolution and biodegradation.
- :ref:`spm_guide`:  A multiphase plume simulation for a density stratified ambient, allowing for multiple intrusion layers in the absence of crossflows.
- :ref:`bpm_guide`: A Lagrangian multiphase plume model for stratified crossflows, which is the primary simulation module for deep ocean oil well blowouts.

This software suite is coded in Python and includes Fortran functions to speed up the equations of state, which are based on the Peng-Robinson equation of state.  The code is mainly self-documenting, and the present documentation is compiled using the ``autodoc`` features of the Sphinx documentation package.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:
   
   Getting Started <start>
   User's Guide <guides>
   Example Scripts <scripts>
   API Reference <api>
   
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Source Code:
   
   Test Package <tests>
   GitHub repository <https://github.com/socolofs/tamoc>

Requirements
------------

The ``tamoc`` suite is built on a minimal set of numerical, scientific, and plotting packages in Python.  The minimum dependencies include the following

- `fortran`_:  A modern Fortran compiler.  The ``tamoc`` suite has been tested with GNU Fortran and the ``conda`` ``fortran-compiler`` package.
- `numpy`_:  Numerical Python, a package for handling array data.
- `scipy`_:  Scientific Python, a package for numerical methods.
- `matplotlib.pyplot`_:  The premier Python plotting package.
- `netCDF4`_:  A Python package for reading and writing netCDF files.
- `xarray`_:  A Python package for multi-dimensional labeled arrays and datasets.
- `pytest`_:  A package for writing small, readable, code tests.

.. _fortran: https://gcc.gnu.org/wiki/GFortran
.. _numpy: https://numpy.org/
.. _scipy: https://scipy.org/
.. _matplotlib.pyplot: https://matplotlib.org/
.. _netCDF4: https://unidata.github.io/netcdf4-python/
.. _xarray: https://xarray.pydata.org/
.. _pytest: https://docs.pytest.org/en/7.0.x/

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Acknowledgments
---------------

This documentation was compiled using the ``Sphinx`` Python Documentation Generator using the ``sphinx_book_theme``.  The organization of these pages follows closely the documentation for `xarray`_, and many of the configuration choices were taken from the ``conf.py`` file in the ``./doc`` subfolder of the `xarray GitHub Repository`_.  

.. _xarray GitHub Repository: https://github.com/pydata/xarray