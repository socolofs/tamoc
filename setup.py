#!/usr/bin/env python

"""
SETUP.py - Setup utility for TAMOC: Texas A&M Oilspill Calculator

This script manages the installation of the TAMOC package into a standard
Python distribution.

For more information on TAMOC, see README.txt, LICENSE.txt, and CHANGES.txt.

Notes
-----
To install, use:
    > python setup.py build
    > python setup.py install

To uninstall, use:
    > pip uninstall TAMOC

To create a source distribution, use:
    > python setup.py sdist --formats=gztar,zip

Author
------
S. Socolofsky, January 2012, Texas A&M University <socolofs@tamu.edu>.

"""
import os
import setuptools
from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize


# Describe some attributes of the software
classifiers = \
"""Development Status :: beta
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: MIT
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules"""

# Define the sample programs to include
bin_files = ['./bin/dbm/air_eos.py',
             './bin/dbm/co2_eos.py',
             './bin/dbm/dead_oil.py',
             './bin/dbm/droplet_rise.py',
             './bin/dbm/equilibrium.py',
             './bin/dbm/gas_bubbles.py',
             './bin/dbm/hydrocarbon_drops.py',
             './bin/ambient/profile_extending.py',
             './bin/ambient/profile_append.py',
             './bin/ambient/profile_from_ctd.py',
             './bin/ambient/profile_from_lab.py',
             './bin/ambient/profile_from_roms.py',
             './bin/ambient/profile_from_txt.py',
             './bin/sbm/bubble.py',
             './bin/sbm/drop_biodeg.py',
             './bin/sbm/drop.py',
             './bin/sbm/sbm_file_io.py',
             './bin/sbm/particle.py',
             './bin/sbm/seep_bubble.py',
             './bin/spm/spm_blowout_sim.py',
             './bin/spm/lake_bub.py',
             './bin/spm/lake_part.py',
             './bin/spm/spm_file_io.py',
             './bin/sintef/particle_size_distribution.py',
             './bin/psm/blowout_jet.py',
             './bin/psm/oil_jet.py',
             './bin/params/scales.py',
             './bin/bpm/bpm_blowout_sim.py',
             './bin/bpm/crossflow_plume.py',
             './bin/bpm/blowout_obj.py']

# Define the external Fortran sources
ext_dbm_f = Extension(name = 'dbm_f',
                      sources = ['tamoc/src/dbm_eos.f95',
                                 'tamoc/src/dbm_phys.f95',
                                 'tamoc/src/math_funcs.f95'])

ext_dbm_c = cythonize("tamoc/src/dbm_c.pyx")[0]

def get_version(pkg_name):
    """
    Reads the version string from the package __init__ and returns it
    """
    with open(os.path.join(pkg_name, "__init__.py")) as init_file:
        for line in init_file:
            parts = line.strip().partition("=")
            if parts[0].strip() == "__version__":
                return parts[2].strip().strip("'").strip('"')
    return None


# Provide the setup utility
if __name__ == '__main__':

    setup(name='TAMOC',
          version=get_version("tamoc"),
          description='Texas A&M Oilspill Calculator',
          long_description=open('README.rst').read(),
          license='LICENSE.txt',
          author='Scott A. Socolofsky',
          author_email='socolofs@tamu.edu',
          url="https://ceprofs.civil.tamu.edu/ssocolofsky/",
          scripts=bin_files,
          packages=['tamoc', 'tamoc.test'],
          package_data={'tamoc': ['data/*.csv', 'data/*.cnv', 'data/*.dat']},
          platforms=['any'],
          ext_package='tamoc',
          ext_modules=[ext_dbm_f, ext_dbm_c],
          classifiers=classifiers.split("\n"),
          )
