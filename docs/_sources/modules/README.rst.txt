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

Version 2.0.0:  Updated the complete model system for compatibility with both
                Python 2.7 and Python 3.8+.  Updated the ambient.Profile 
                object so that netCDF files do not have to be created, 
                including the ability to create a default profile for the 
                world-ocean average profile.  Create new modules for particle
                size distributions and for simulating a blowout, including a
                new Blowout object.  
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
Version 0.1.9 : Made several minor changes to the equations of state per the
                guidance of Jonas Gros.  These changes make the model much 
                more robust for hydrocarbon mixtures.  The updates are minor
                in that the results do not change markedly for the test 
                cases already in previous versions of the model.  However, 
                the changes provide major advantages for more difficult
                cases, not demonstrated in the simple ./bin examples.
Version 0.1.8 : Added print capability to the params.py module and upgraded
                the shear entrainment model in the bent_plume_model.py 
                to the entrainment equations in Jirka 2004.
Version 0.1.7 : Added the capability for the bent_plume_model.py to continue
                to track particles outside the plume using the 
                single_bubble_model.py.  Fixed a bug where particles outside
                the plume continued to dissolve and add mass to the 
                bent_plume_model.
Version 0.1.6 : Added a new simulation module for plumes in crossflow:  the
                bent_plume_model.py.  Refactored some of the code for the 
                original model suite to make it more general and to reuse it
                in the bent_plume_model.  Added example files and unit tests
                for the new modules, and updated the documentation to reflect
                all model changes.
Version 0.1.5 : Fixed a small bug in the way the bubble force is handled 
                after the particle dissolves.  Fixed a bug to retain mass
                conservation for a bubble size distribution using the 
                sintef.rosin_rammler() function.
Version 0.1.4 : Added script for the the sintef and params modules to the 
                ./bin examples directory and the /test unit tests.  Improved
                the stability of the model by added a few new checks during
                and before calculation.  Updated the unit tests to make them
                more platform and numpy-version independent.
Version 0.1.3 : Removed some of the debugging catches in the iteration so that
                solutions always fully converge and fixed a few bugs.  See 
                CHANGES.txt for full details.  Added the sintef.py module for
                computing initial bubble/droplet size distributions.
Version 0.1.2 : Refined the test suite for compatibility with multiple 
                versions of numpy and scipy.  Corrected a few more minor bugs.
Version 0.1.1 : Full modeling suite with small bug fixes and complete test 
                suite..
Version 0.1.0 : First full modeling suite release, including the stratified
                plume module.
Version 0.0.3 : Updated to include the single bubble model and the ambient
                module for handling ambient CTD data.  Includes input and 
                output using netCDF files and a complete set of tests in 
                ./tamoc/test
Version 0.0.2 : First model release, including the discrete bubble model
                (dmb.py)
Version 0.0.1 : Initial template of files using setup.py

Requirements
============

This package requires:

* Python 2.3 or higher and is now compatible with both Python 2.7 and 
  Python 3.8+

* Numpy version 1.6.1 or higher

* Scipy version 0.17.0 or higher

* A modern Fortran compiler

* The Python netCDF4 package

For interaction with ROMS output, TAMOC also requires:
   
   * octant:  download from https://github.com/hetland/octant
   
   * mpl_toolkits.basemap:  download from
     http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/
   
   * NOTE:  While this still works, octant has been deprecated and is now
            replaced by the xarray package.  TAMOC has not yet been updated
            to use xarray.  
            
            If profile data are available from ROMS, write a script to read
            the data into a numpy array and pass this data to the 
            ambient.Profile object.  

Code development and testing for this package was conducted in the Mac OS X
environment, Version 10.13.6. The Python environments were created using
miniconda.  The Python 3 environment used Python 3.8.2; the Python 2 
environment used Python 2.7.15.  All scripts are tested in iPython with the
--pylab flag. 

Fortran files are written in modern Fortran style and are fully compatible
with gfortran 4.6.2 20111019 (prerelease). They have been compiled and tested
by the author using f2py Version 2. 

Quick Start
===========

* Edit setup.cfg to select the appropriate C/C++ and Fortran compilers

* Run 'python setup.py build' followed by 'python setup.py install' (with 
  sudo if necessary).

* Test the installation by opening a Python session and executing 
  `import tamoc` from the Python prompt.  Be sure that you are not in the 
  same directory as the setup.py file so that Python will look for tamoc in 
  the main Python package repository on your system.

* To run all the tests, cd to the ./test directory and execute 'py.test'
  from a command prompt.  If pytest is not installed, follow the instructions
  here:  http://pytest.org/latest/getting-started.html

Platforms
=========

Windows 7
---------

The following method has been tested for installation on Windows 7.

* Install a complete Python distribution that includes Python, Numpy, and
  Scipy with versions compatible with the above list.  Testing has been 
  completed by the author using a 32-bit Python installation.  The Python
  distribution will have to be compatible with your C/C++ and Fortran 
  compiler.  The free compilers available from MinGW that work with Python
  f2py are typically 32 bit.  There are work-arounds, but the instructions
  here were all tested on 32-bit installations.

* Download and install the MinGW compiler suite.  During installation, be sure
  to select a C, C++, and Fortran compiler.  See, 
  http://sourceforge.net/projects/mingw/files/

* Edit the Windows > System > Environment Variables so that the PATH can find 
  your Python and MinGW installation.

* Open a command prompt from Start > Run > Command Prompt and follow the steps 
  in the Quick Start section above to complete installation.
  
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
  