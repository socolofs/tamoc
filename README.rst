=====================================
TAMOC - Texas A&M Oilspill Calculator
=====================================

TAMOC is a Python package with Fortran extension modules that provides various
functionality for simulating a subsea oil spill.  This includes methods to
compute equations of state of gas and oil, dissolution, transport of single
bubbles and/or droplets, and simulations of blowout plumes, including the
development of subsurface intrusions, and estimation of initial bubble/droplet
size distributions from source flow conditions.

For typical usage, please see the ./bin directory of the source distribution
for various example scripts.

Version 4.1.0:  Adds a new `Profile3DT` object to the `ambient` module that
                allows ambient profile data to be obtained from 
                three-dimensional space plus time `netCDF` datasets through
                environment objects defined in the General NOAA Operational
                Modeling Environment.  Hence, this new capability is 
                contingent on a working installation of `gnome`.  Updated
                calls to `get_values` throughout the bent plume model, 
                dispersed phases, and single bubble model modules to allow
                for use of the new `Profile3DT` object, which requires both
                space and time to define the point for interpolation.  Updated 
                some of the standard plots for more readable legends.
Version 4.0.0:  Non-beta version of the TAMOC package compatible with Meson
                build tools.  Minor adjustments to the plotting routines, 
                a few added diagnostic print statement.  The maximum allowable
                time-step for particle tracking outside the plume in the 
                bent plume model was reverted back to the value in Version 
                3.4.2 and earlier (1000 s insteady of 86400 s).  A new `utils`
                subpackage has been added to group together the utilities 
                that help build and manage TAMOC objects.
Version 4.0.0-beta:  Updated the TAMOC package to use pyproject.toml and 
                meson.build to install.  Newer versions of Python have
                depricated the setuptools distutils packages that NumPy
                f2py was using to create the dbm_f extension module.  This
                update allows the dbm_f module to be built using the latest
                versions of Python (tested with python 3.13, numpy 2.2.5, and
                scipy 1.15.2).  This is listed as a beta release as this is
                the first release using this new build platform and the 
                documentation has not yet been updated to explain how to 
                install using meson.build and pip.  The first 4.0.0 stable
                release will have fully updated documentation.  The original
                setup.py file remains part of the package repository and will
                continue to work with older versions of Python, NumPy, and 
                setuptools that support the numpy.distutils package.  This
                new version also includes a C extension module preos_c to
                replace part of the Fortran extension module dbm_f.  The 
                speed trade-offs are now only a difference of about 40% due
                to the fact that only the slowest python functions were 
                ported from dbm_p to preos_c.  See changes.txt for more 
                details.  
Version 3.4.2:  Added a method to get and save derived output (particle
			    diameter, density, slip velocity, and mass transfer 
				coefficient) from the single bubble model.  Updated the
                reporting methods in the bent plume model so that they do
                not fail when particles are tracked in the farfield but 
                fully dissolve within the plume.  Updated the equilibrium
                calculations so that the non-zero components are handled
                within the FluidMixture object rather than within the 
                equil_MM function.
Version 3.4.0:  Updated the root-finding method in the Fortran `math_funcs` 
				module so that it always finds the correct roots. Updated the
				`dbm` module `equilibrium` method so that it returns the 
				correct fluid phase when there are non-zero components in the 
				mixture. Moved the unit conversion in the 
				`chemical_properties` module into a separate function so
				that they can be used anywhere they are needed.
Version 3.3.0:  Added a few new post-processing tools to the bent plume and 
                stratified plume models and updated the particle size model
                tools so that the user can specify a non-equilibrium pressure.
Version 3.2.0:  Added the ability to save derived variables (e.g., plume width,
                trajectory, concentration, etc.) from the bent plume model and
                stratified plume model.  Each of these modules have the new 
                methods `get_derived_variables` and `save_derived_variables`
                in their `Model` class objects.  Updated the test files to be
                compatible with the `C_pen` and `C_pen_T` variables added to 
                the `dbm` module.
Version 3.1.0:  Added the ability for the user to provide values of the
                Peneloux volume shift parameter to the `dbm` module classes.
                The input parameters are `C_pen` and `C_pen_T`, which relate to
                the temperature-dependent Peneloux shift model in equation 5.9
                in Pedersen et al. (2015). Supplying `C_pen` equal to zero uses
                the original model based on Lin and Duan (2005). The original
                model is preferred unless the pseudo-component property data
                have been strongly fitted so that Peneloux shift parameters
                are outside the expected range of -0.4 to 1.2 are required. 
                Minor fixes to prevent errors when non-standard units are used 
                as input to the `ambient` module and several updates to make the
                code compatible with recent releases of `numpy` and
                `matplotlib`.
Version 3.0.0:  Added the capability to have mixed-phase particles of gas and
                liquid fused together.  Updated the initial conditions methods
                for pure multi-phase plumes so that a void fraction between
                0 and 1 is enforced.  This may result in a different orifice 
                diameter being used than specified by the user, but ensures the
                dilute plume assumption is not violated.  This new capability
                becomes important when simulating accidental spills from 
                subsea CO2 sequestration pipelines in which very large 
                quantities of CO2 may be released through a small orifice.  
                Other model updates improve intuition of using the model.
Version 2.4.0:  Added additional post-processing methods to the bent plume
                model to extract dissolved-phase concentration maps. Also made
                it easier to use the Privat and Jaubert binary interaction
                coefficients and updated the seawater module with a density
                equation for very hot seawater and a function to compute pH.
Version 2.3.1:  Added new post-processing methods to the bent plume model
                module to track mass balances and plot particle size 
                distributions.
Version 2.3.0:  Added a new module to replace the Fortran codes that are used
                by the dbm module. Now, a Fortran compiler is not required to
                install and run TAMOC.
Version 2.2.0:  Updated the ambient module so that it is not based on and
                compatible with xarray Dataset objects, updated all tests to
                pass with the latest version of TAMOC, revised the
                documentation with a new template and organization, and
                updated some of the source docstrings to compile with Sphinx.
Version 2.1.0: Updated the readme file with instructions for modern version
                of Windows. Updated the model with various improvements,
                including some additional chemical property data, additional
                functionality in the blowout.py module. Small, additional bug
                fixes.
Version 2.0.0:  Updated the complete model system for compatibility with both
                Python 2.7 and Python 3.8+. Updated the ambient.Profile
                object so that netCDF files do not have to be used and
                including the ability to create a default profile using the
                world-ocean average data now distributed with the model.
                Created new modules for particle size distributions and for
                simulating a blowout, including a new Blowout object. Created
                a new modules containing utility functions for manipulating
                data related to the ambient and dbm modules.
Version 1.2.1:  Corrected minor errors in translating between NetCDF objects
                and Numpy arrays to avoid a masked-array error and updated
                the dbm_phys.f95 function for the mass transfer rate by Kumar
                and Hartland so that the Reynolds number is defined before it
                is used.
Version 1.2.0:  Added biodegradation to the fate processes considered in the
                discrete bubble model (DBM).
Version 1.1.1:  Updated the ambient module interpolation method to be
                compatible with newer versions of numpy, updated a few of
                the ./bin examples to only read data provided with TAMOC, and
                updated all test cases to be compatible with slight changes
                in the dbm module that were done in Version 1.1.0.
Version 1.1.0:  Updated various modules to be compatible with Anaconda
                Python, including Scipy version 0.17.0 and higher.  Fixed a
                couple bugs in the test cases where output files are not
                created correctly.  Updated the documentation with some
                missing new variables.
Version 1.0.0:  Finalized the validation cases and tested the model for
                release.  This is the first non-beta version of the model,
                and is the version for which journal publications have been
                prepared.  Most of the changes going forward are expected to
                be new capabilities and improvements in the user interface:
                the model solutions are not expected to change appreciably.

Beta versions of the model:

Version 0.1.17: Updated the modeling suite so that all of the save/load
                functions are consistent with the present model variables
                and attributes.  Modified the bent plume model so that
                ambient currents can come from any direction (three-
                dimensional).  Added a new test file for the bent plume
                model.  Changed the convergence criteria for the stratified
                plume model.
Version 0.1.16: Updated the bent plume model so that post processing is
                fully consistent with the simulation results.  Also, added
                the capability for the bent plume model to stop at the
                neutral buoyancy level in the intrusion for a stratified
                case.  Updated the equilibrium calculations in the dbm module
                so that it does not crash when the first few elements of
                the mixture disappear (go to zero) and to speed up the
                calculation when successive substitution indicates the
                mixture may be single phase, but is slowly converging:
                stability analysis is initiated early, which greatly improves
                performance for difficult cases.
Version 0.1.15: Moved the particle tracking in the bent plume model inside
                the main integration loop, which then removes tp and sp
                from the model attributes and includes then in the model
                state space instead.  Updated the bent plume model state
                space so that particle mass is the state variables instead
                of particle mass flux and so that the dissovled phase
                constituents are modeled as total mass in the Lagrangian
                element instead of concentration times mass of the element.
                Made a small update to the hydrate formation time equations.
Version 0.1.14: Updated several aspects of the calibration after comparing
                to available data in Milgram (1983), Jirka (2004), Socolofsky
                and Adams (2002, 2003, 2005), Socolofs et al. (2008), and
                Socolofsky et al. (2013).  The most significant change is an
                updated shear entrainment coefficient for the stratified
                plume model.  Also, added a buoyant force reduction as bubbles
                drift away from the centerline in a crossflow.
Version 0.1.13: Updated the temperature output for the bent plume model so
                that the correct temperature is saved when heat transfer ends.
                Added the particle time to the state space of the stratified
                plume model and added the hydrate formation model of Jun et
                al. (2015) to the particle objects in the dispersed phases
                module.  The hydrate formation time is set at the start of a
                simulation and is properly implemented for all three
                simulation modules in the ``TAMOC`` suite.  To compute the
                hydrate formation time using the equations from Jun et al.
                (2015), use the function
                `dispersed_phases.hydrate_formation_time`.
Version 0.1.12: Replaced methods for equilibrium and viscosity with better
                algorithms.  Fixed small inconsistencies in the dbm.py module
                for clean bubbles, and updated the seawater equations of
                state with better methods for heat capacity and air/water
                surface tension.  Updated values for the Setschenow constant
                in ./data/ChemData.csv, and added a mass transfer equation
                for Re < 1.
Version 0.1.11: Replaced some of the -9999 values in the ./data/ChemData.csv
                file with literature values and updated the units of the
                calcualtion of Vb in dbm.py when data are not available.
                Also, updated the parameter values for the stratified plume
                model with the values recommended in Socolofsky et al. (2008).
Version 0.1.10: Updated the values for Vb in the ./data/ChemData.csv file
                with their correct values.  Improves computation of
                diffusivity and mass transfer over Version 0.1.9, and gives
                results similar to Version 0.1.8 and older that used a
                different method to estimate diffusivity.
Version 0.1.9: Made several minor changes to the equations of state per the
                guidance of Jonas Gros.  These changes make the model much
                more robust for hydrocarbon mixtures.  The updates are minor
                in that the results do not change markedly for the test
                cases already in previous versions of the model.  However,
                the changes provide major advantages for more difficult
                cases, not demonstrated in the simple ./bin examples.
Version 0.1.8: Added print capability to the params.py module and upgraded
                the shear entrainment model in the bent_plume_model.py
                to the entrainment equations in Jirka 2004.
Version 0.1.7: Added the capability for the bent_plume_model.py to continue
                to track particles outside the plume using the
                single_bubble_model.py.  Fixed a bug where particles outside
                the plume continued to dissolve and add mass to the
                bent_plume_model.
Version 0.1.6: Added a new simulation module for plumes in crossflow:  the
                bent_plume_model.py.  Refactored some of the code for the
                original model suite to make it more general and to reuse it
                in the bent_plume_model.  Added example files and unit tests
                for the new modules, and updated the documentation to reflect
                all model changes.
Version 0.1.5: Fixed a small bug in the way the bubble force is handled
                after the particle dissolves.  Fixed a bug to retain mass
                conservation for a bubble size distribution using the
                sintef.rosin_rammler() function.
Version 0.1.4: Added script for the the sintef and params modules to the
                ./bin examples directory and the /test unit tests.  Improved
                the stability of the model by added a few new checks during
                and before calculation.  Updated the unit tests to make them
                more platform and numpy-version independent.
Version 0.1.3: Removed some of the debugging catches in the iteration so that
                solutions always fully converge and fixed a few bugs.  See
                CHANGES.txt for full details.  Added the sintef.py module for
                computing initial bubble/droplet size distributions.
Version 0.1.2: Refined the test suite for compatibility with multiple
                versions of numpy and scipy.  Corrected a few more minor bugs.
Version 0.1.1: Full modeling suite with small bug fixes and complete test
                suite..
Version 0.1.0: First full modeling suite release, including the stratified
                plume module.
Version 0.0.3: Updated to include the single bubble model and the ambient
                module for handling ambient CTD data.  Includes input and
                output using netCDF files and a complete set of tests in
                ./tamoc/test
Version 0.0.2: First model release, including the discrete bubble model
                (dmb.py)
Version 0.0.1: Initial template of files using setup.py

Requirements
============

This package requires:

* Python 2.3 or higher and is now compatible with both Python 2.7 and
  Python 3.8+.  Python 2 compatibility will no longer be ensured.  Please 
  move to Python 3 if you have not already done so.

* Numpy version 1.16 or higher

* Scipy version 1.2.0 or higher

* The Python netCDF4 package

* The Python xarray package

* To use the Fortran versions of the equations of state, a modern Fortran 
  compiler is required. Otherwise, the Python version of these codes will be
  used.

* To view plots of the model output, TAMOC uses the matplotlib package

Code development and testing for this package was conducted in the Mac OS X
environment, Version 10.13.6 through 15.4.1. The Python environments were
created using miniconda. The lastest Python 3 environment used Python 3.13; the
latest tested Python 2 environment used Python 2.7.15. All scripts are tested 
in iPython with the --pylab flag.

Fortran files are written in modern Fortran style and are fully compatible
with gfortran 4.6.2 20111019 (prerelease). They have been compiled and tested
by the author using f2py Version 2 and higher.  The latest release can be 
compiled using meson.build with NumPy version 2.2.5.

Installation Overview
=====================

This package has been updated to be compatible with the new installation
approach of Python that uses `pip` and  `pyproject.toml` and uses the 
`meson.build` backend.  This was accomplished by closely following the 
approach of the Scipy package.  If any issues or questions arise, please
also consult the Scipy documentation.

TAMOC can be installed in two different ways, depending on the version of
Python, NumPy, and setuptools that you are using.  

For older versions of Python (<=3.10.14), the original approach using setup.py
is still supported. Please see the Legacy Installation instructions below if
you are installing using these older Python versions.

For the current stable Python release (>=3.13), the numpy.distutils package in
setup.py is deprecated. These installs will use pyproject.toml with the Meson
backend. TAMOC 4.0.0-beta is the first release compatible with these tools.
The following installation instructions follow this new approach.

Initial Steps
=============

The following installation instructions assume you will be installing TMAOC
in a virtual Python environment using conda or mamba.  Also, some of the
install tools are only compatible with conda-forge versions of the Python 
packages (e.g., NumPy, Scipy, etc).  Please ensure the following before
continuing to the install instructions.

Install a Conda Environment Package
-----------------------------------

Install miniconda or mamba.  If you already have one of these installed, 
make sure conda-forge is in your channels:

   >>> conda config --show channels
   >>> conda config --add channels conda-forge
   
Otherwise, it is recommended to install miniforge.  A Windows install binary
is available here:

   https://conda-forge.org/miniforge/
   
After installation, you should be able to open a miniforge command window
(search for the Miniforge application).  With miniforge, conda-forge is the
default channel, so you can continue with the instructions below.

Install Compilers
-----------------

If you want to compile the Fortran and/or C extension modules, you have to
install compilers.

Windows
-------

On Windows, this guide follows the instructions from the Scipy installation
documentation for a developement installation of Scipy.  

1. Install MS Visual Studio Community 2019 or later.  You only need to install
   the C++ Build Tools.  This is a likely a two-step process.  First, install
   MS Visual Studio.  Then launch MS Visual Studio.  In the main screen, there
   should be several options of tools / packages to install.  Select at a 
   minimum the C++ Build Tools.  All later steps below were conducted by
   accepting the default list of install items after choosing C++ Build Tools.
   This required about 10 GB of hard disk space.

2. If you want to use only the C extension module, step 1. above should 
   suffice.  For the Fortran extension module, you need to also install a 
   Fortran compiler.  Scipy recommends using the compilers from MinGW-w64.
   Follow these steps (adapted from the Scipy install instructions):
   
   A. Install chocolatey following the instructions here:
      https://chocolatey.org/install
   
   B. Open a miniconda or miniforge command window.  Then, execute the 
      command:
      
      >>> choco install rtools -y --no-progress --force --version=4.0.0.20220206
      
   C. Add the MinGW Compilers to the PATH variable.  Open Finder and search
      for System Environment Variables for your account.  Choose to Edit the
      Path.  Click New to create a new Path.  Paste the directory to your
      rtools instalation above.  Most likely this will be:
      
      C:\rtools\mingw64\bin
      
   D. For the most reliable situation, reboot now.
   
   E. Open the miniconda or miniforge command window and check whether 
      compilers are available using these commands:
      
      >>> gcc --version
      >>> gfortran --version
      
MacOS
-----

On a Mac computer, you need to have the xtools build tools installed.  Open 
a terminal where miniconda, mamba, or miniforge is running, and execute:

   >>> sudo xcode-select --install
   
You have to run this command as administrator (thus, the sudo prefix), so you 
will be prompted to enter an administror password.  This should launch a 
window that may require you to accept the license and other things.  Follow
on-screen instructions.

Install TAMOC
-------------

After the above steps, the remaining steps are almost the same on Windows and
MacOS.  The only difference is on Windows.  For Windows, find the 
environment.yml file in the TAMOC repository, open the file, and remove the
line that reads:

   - compilers  # Currently unavailable for Windows.
  
otherwise, this will mess up the compilers configured above.

The steps below will install TAMOC in a virtual environment called tamoc4. If
you want to name the environment something else, edit the environment.yml file
in the TAMOC repository before continuing. Using a virtual environment is the
recommended approach to installing TAMOC.


Next, open a command terminal (miniforge command prompt in Windows, Terminal on
MacOS). Change directory to the location that contains the root directory of
TAMOC. This is where pyproject.toml and meson.build are located in the TAMOC
repository. Create the virtual environment with:

   >>> conda env create -f environment.yml

Accept the on-screen prompts.  

Activate the new virtual environment with:

   >>> conda activate tamoc4

Install TAMOC with:

   >>> pip install --no-build-isolation --editable .

This command will follow the default settings in meson_options.txt regarding
whether or not to compile the Fortran and / or C extension modules.  If you
have trouble with either of these compilations, either edit the boolean 
flags in meson_options.txt and try the install again or send the desired 
flags directly at the pip command using, e.g.:

   >>> pip install --no-build-isolation \
          --config-settings=setup-args=-Dwith-c=true \
          --config-settings=setup-args=-Dwith-fortran=false \
          --config-settings=setup-args=-Dpython-only=false \
          --config-settings=builddir=<mydir> --editable .

Chose true/false as needed for your own installation.  If you want to use 
Spyder, this is not installed by default from environment.yml.  Install 
using:

   >>> conda install spyder
   
Inside Spyder, make sure you set your console to exectue in the tamoc4 
virtual environment.  

To run the tests, change directory to ./tamoc/test and run the command:

   >>> pytest -v
   
A few of the tests will probably fail (these are unfortunately compiler
dependent and need to be updated---on the TODO list).  As long as most of 
the tests pass, the installation is ready to go.

Next Steps
----------

After installing TAMOC, have a look in the ./examples/ folder for examples.
The Documentation in the ./docs/ folder is also mostly up-to-date.  



Legacy Installation
===================

TAMOC <4.0.0 used setup.py only to install TAMOC. This install method is still
supported, but requires older versions of Python, NumPy, and setuptools. With
Python <= 3.10.14, NumPy <= 1.26.4, and setuptools <= 69.5.1, the following
instruction should work. It is possible that there are a few other restrictions
of version numbers. Please note, however, that setup.py has not yet been
updated to build the preos_c extension module (not needed if you are compiling
the Fortran dbm_f extension module or just using a pure Python install).

For the best and most complete information, please see the documentation web pages in the `./doc/` directory of the TAMOC repository.  A step-by-step installation guide is included in the Getting Started rubric of the documentation.  A brief summary that may still work is provided below.

* Edit setup.cfg to select the appropriate C/C++ and Fortran compilers

* For a normal install, run 'python setup.py build' followed by 'python  
  setup.py install' (with sudo if necessary). To install using a local
  install directory in develop mode, use: 'python setup.py develop'.

* To skip the Fortran extension library and install a Python-only version of 
  ``tamoc``, use the ``--python-only`` flag in the install command, e.g., 
  'python setup.py develop --python-only'.

* Test the installation by opening a Python session and executing
  `import tamoc` from the Python prompt.  Be sure that you are not in the
  same directory as the setup.py file so that Python will look for tamoc in
  the main Python package repository on your system.

* To run all the tests, execute 'pytest -v --pyargs tamoc'
  from a command prompt outside of the TAMOC package. If pytest is not
  installed, follow the instructions here:
  http://pytest.org/latest/getting-started.html. The TAMOC tests write files
  to test the storage and recovery methods of the model modules. If these
  tests fail with write permission errors, you may try 'sudo pytest -v
  --pyargs tamoc'.

Platforms
=========

Windows
-------

The following method has been tested for installation on Windows 10 using Miniconda environments.

* Create a new Conda environment for Python 3. This has been tested up to
  Python version 3.8.8. Install the required dependencies using: 
  
  conda install numpy scipy matplotlib netCDF4 pytest
  
  Also install the free GNU fortran compiler using: 
  
  conda install -c conda-forge fortran-compiler 
  
  Note that this fortran compiler requires that the following, free software
  also be installed on the Windows box: Microsoft Visual C++ 14.0 or greater.
  You can obtain this with the Microsoft C++ Built Tools at:
  https://visualstudio.microsoft.com/visual-cpp-build-tools/.

* Download the TAMOC source files. Activate your conda environment, and in
  the ./tamoc directory at a command prompt try: 
  
  python setup.py install <--python-only>
  
  Alternatively, you can install a development version with: 
  
  python setup.py develop <--python-only>
  
  where the flag '--python-only' is optional

* Change directly to a directory outside of your TAMOC source files. Check
  the TAMOC package installation by running the following command at a
  command prompt: 
  
  pytest -v --pyargs tamoc


Mac OS X / Unix
---------------

The following method has been tested for installation on Mac OS X 10.7.

* Install a complete Python distribution that includes Python, Numpy, and
  Scipy with versions compatible with the above list.  Testing has been
  completed by the author using a 32-bit and 64-bit Python installations.
  The Python distribution will have to be compatible with your C/C++ and
  Fortran compiler.

* Install the free XCode app in order to provide C/C++ compiler capability.
  Be sure to install the command-line tools.

* Download and install the gfortran binary. See,
  http://gcc.gnu.org/wiki/GFortranBinaries

* Follow the steps in the Quick Start section above to complete installation.
