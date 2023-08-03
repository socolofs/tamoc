############
Installation
############

:Release: |version|
:Date: |today|

``tamoc`` is a Python scripting package that includes an optional equation of state module coded in ``fortran``.  The Python scripts are also dependent on a few numerical and scientific packages that are becoming standard with Python.  The following demonstrates how to install ``tamoc`` in a new ``conda`` environment.

Create a compatible ``tamoc`` environment
=========================================

First, create a new environment using Python 3::

   >>> conda update conda
   >>> conda create -n <tamoc_env> python=3
   >>> conda activate tamoc_env

You can insert whatever environment name you like into the ``<tamoc_env>`` parameter, above.

Next, install the numerical dependencies::

   >>> conda install numpy scipy

The post-processing tools mostly rely on ``matplotlib``.  While the ``pyplot`` subpackage may be adequate to run ``tamoc``, it has only been tested with the full package::

   >>> conda install matplotlib

To handle the ambient profile data, ``tamoc`` can interface with both netCDF files and ``xarray.Dataset`` objects.  Install these dependencies::

   >>> conda install netCDF4 xarray

To run the unit tests distributed with ``tamoc``, you will also need pytest::

   >>> conda install pytest

Fortran compiler
================

The physical properties and thermodynamic state functions for computing fluid mixture properties in ``tamoc`` are available in an optional Fortran library distributed with ``tamoc``.  In theory, any Fortran compiler should work, but because this has to be accessed through the ``f2py`` module in ``numpy``, this can be a tricky part of getting ``tamoc`` to work with the Fortran library extension package on your machine.  ``tamoc`` has currently been tested on Windows and Mac, and it is expected that Linux should follow the Mac instructions.  The procedure that worked during testing was slightly different on each platform.  Please follow the instructions below for your platform if you wish to try to install the Fortran extension package.

Windows
-------

A simple way to install a Fortran compiler is using conda::

   >>> conda install -c conda-forge fortran-compiler

For this to work with ``f2py`` and ``tamoc``, you will also need to install the free Microsoft Visual C++ Build `Build Tools`_.  Make sure that this is installed, and you may need administrative privileges to install it to work on your account.

.. _Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/https://visualstudio.microsoft.com/visual-cpp-build-tools/

Mac OS / (Linux?)
-----------------

On Mac OS, the ``fortran-compiler`` available from ``conda-forge`` is outdated, and not compatible with the 64-bit arithmetic libraries used in the ``tamoc`` Fortran source files.  Fortunately, install files for the GNU Fortran package are available on Mac.  Download and install these from gfortran_.

.. _gfortran: https://gcc.gnu.org/wiki/GFortranBinaries

It is also likely necessary to install the ``xtools`` command line tools.  If you have errors indicating that ``cpp`` is not available, look into installing these `xcode tools`_.  

.. _xcode tools: https://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/

Run setup.py for ``tamoc``
==========================

Download the latest ``tamoc`` repository from Github (see `tamoc on GitHub`_).  Expand the repository in a location you want ``tamoc`` to reside, open a terminal window, and change directory (``cd``) to the root directory of the ``tamoc`` repository.  If you list the directory contents, you should see ``setup.py`` in the list.

You can install ``tamoc`` in two main ways.  Using::

   >>> python setup.py install

will install ``tamoc`` like a normal Python package, putting the package contents into your environment libraries where they cannot be edited.  

If, instead, you want to be able to make changes to the ``tamoc`` source files and see those changes in your simulations (only required for developers), then you probably want your Python environment to use the files in the directory you are currently looking at.  In that case, use the ``develop`` flag to install ``tamoc``::

   >>> python setup.py develop

Skipping the Fortran extension module
-------------------------------------

If the above installation command fail because the Fortran extension module cannot be installed, you can use the Python-only version of ``tamoc``, which will replace the Fortran library with the ``dbm_p`` module distributed with ``tamoc``.  To install without Fortran, use::

   >>> python setup.py install --python-only

or::

   >>> python setup.py develop --python-only

Check your installation
-----------------------

To check your installation, you will want to run the test scripts.  Change directory out of your ``tamoc`` repository, and then run the ``pytest`` command::

   >>> pytest -v --pyargs tamoc

Because the simulation modules use adaptive-step-size solvers for stiff ODEs and because the simulations can be hundreds of steps, some of the tests may not pass. In particular, the ``test_spm.py`` tests are notorious for this.  As long as the only errors that occur are differences between the computed values and the values in the test files (and these errors are relatively small), then you may consider all tests to have passed.  ``tamoc`` is now ready for use.

.. _tamoc on GitHub: https://github.com/socolofs/tamoc

