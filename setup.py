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
import os, sys
import setuptools
from numpy.distutils.core import Extension

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
             './bin/ambient/profile_from_txt.py',
             './bin/ambient/nc_profile_extending.py',
             './bin/ambient/nc_profile_append.py',
             './bin/ambient/nc_profile_from_ctd.py',
             './bin/ambient/nc_profile_from_lab.py',
             './bin/ambient/nc_profile_from_txt.py',
             './bin/ambient/np_profile_extending.py',
             './bin/ambient/np_profile_append.py',
             './bin/ambient/np_profile_from_ctd.py',
             './bin/ambient/np_profile_from_lab.py',
             './bin/ambient/np_profile_from_txt.py',
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
             './bin/bpm/blowout_obj.py',
             './bin/sfm/fracture_model.py']

# Define the external Fortran sources
ext_dbm_f = Extension(name = 'dbm_f',
                      sources = ['tamoc/src/dbm_eos.f95',
                                 'tamoc/src/dbm_phys.f95',
                                 'tamoc/src/math_funcs.f95'])

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

# parameters for the setup() call:
setup_params = {'name': 'TAMOC',
                'version': get_version("tamoc"),
                'description': 'Texas A&M Oilspill Calculator',
                'long_description': open('README.rst').read(),
                'license': 'LICENSE.txt',
                'author': 'Scott A. Socolofsky',
                'author_email': 'socolofs@tamu.edu',
                'url': "https://ceprofs.civil.tamu.edu/ssocolofsky/",
                'scripts': bin_files,
                'packages': ['tamoc', 'tamoc.test'],
                'package_data': {'tamoc': ['data/*.csv', 'data/*.cnv',
                                           'data/*.dat']},
                'platforms': ['any'],
                'classifiers': classifiers.split("\n"),
                }

# additional/changed setup parameters for the version that builds the Fortran code
fortran_params = {'ext_package': 'tamoc',
                  'ext_modules': [ext_dbm_f],
                  }

# Provide the setup utility
if __name__ == '__main__':

    from numpy.distutils.core import setup

    # Check if the user wants to try installing the Fortran library extension
    if '--python-only' in sys.argv:
        python_only = True
        sys.argv.remove('--python-only')
        print("Not attempting Fortran build -- installing only the Python version")
    else:
        python_only = False

    if python_only:
        # User wants to install only the python version of TAMOC
        pass # just in case there's something to do in the future
        # setup(name='TAMOC',
        #       version=get_version("tamoc"),
        #       description='Texas A&M Oilspill Calculator',
        #       long_description=open('README.rst').read(),
        #       license='LICENSE.txt',
        #       author='Scott A. Socolofsky',
        #       author_email='socolofs@tamu.edu',
        #       url="https://ceprofs.civil.tamu.edu/ssocolofsky/",
        #       scripts=bin_files,
        #       packages=['tamoc', 'tamoc.test'],
        #       package_data={'tamoc': ['data/*.csv', 'data/*.cnv',
        #           'data/*.dat']},
        #       platforms=['any'],
        #       classifiers=classifiers.split("\n"),
        #       )
    else:
        # User want to try installing the TAMOC version that uses the
        # fortran extension module dbm_f
        setup_params.update(fortran_params)

    setup(**setup_params)


