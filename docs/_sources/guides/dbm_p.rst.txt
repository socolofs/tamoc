#############################################
Discrete Bubble Model -- Properties Functions
#############################################

The class objects in the discrete bubble model make use of a library of
properties functions that help to compute rise velocity, mass transfer
coefficient, and thermodynamic state data. These helpful functions are
contained in the Fortran source files in the ./src directory of ``tamoc``. When
a Fortran compiler is not available, a Python-only version of this library is
provided in the ``dbm_p`` module. These properties functions are not intended
to be used directly by the user in a ``tamoc`` session; but rather, they should
be accessed through the class objects in the ``dbm`` module. Please see that
module for a guide for making these types of calculations.
