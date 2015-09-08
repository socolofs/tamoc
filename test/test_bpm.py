"""
Unit tests for the `bent_plume_model` module of ``TAMOC``

Provides testing of the objects, methods and functions defined in the
`bent_plume_model` module of ``TAMOC``. These tests check the behavior of the
object, the results of the simulations, and the read/write algorithms.

The ambient data used here are from the `ctd_BM54.cnv` dataset, stored as::

    ./test/output/test_BM54.nc

This netCDF file is written by the `test_ambient.test_from_ctd` function, 
which is run in the following as needed to ensure the dataset is available.

Notes
-----
All of the tests defined herein check the general behavior of each of the
programmed function--this is not a comparison against measured data. The
results of the hand calculations entered below as sample solutions have been
ground-truthed for their reasonableness. However, passing these tests only
means the programs and their interfaces are working as expected, not that they
have been validated against measurements.

"""
# S. Socolofsky, November 2014, Texas A&M University <socolofs@tamu.edu>.

from tamoc import seawater
from tamoc import ambient
from tamoc import dbm
import test_sbm
from tamoc import dispersed_phases
from tamoc import bent_plume_model
from tamoc import lmp

import numpy as np
from numpy.testing import *

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------


def test_bpm():
    """
    docstring for test_bpm
    
    """
    pass