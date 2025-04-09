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
setup_params = {
    'name': 'TAMOC',
    'version': get_version("tamoc"),
    'description': 'Texas A&M Oilspill Calculator',
    'long_description': open('README.rst').read(),
    'license': 'LICENSE.txt',
    'author': 'Scott A. Socolofsky',
    'author_email': 'socolofs@tamu.edu',
    'url': "https://ceprofs.civil.tamu.edu/ssocolofsky/",
    'packages': ['tamoc', 'tamoc.test'],
    'package_data': {'tamoc': ['data/*.csv', 'data/*.cnv',
                               'data/*.dat']},
    'platforms': ['any'],
    'classifiers': [
        'Development Status :: beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: MIT',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
                }

# additional/changed setup parameters for the version that builds the Fortran 
# code
fortran_params = {
    'ext_package': 'tamoc',
    'ext_modules': [
        Extension(
            name = 'dbm_f',
            sources = ['tamoc/src/dbm_eos.f95',
                       'tamoc/src/dbm_phys.f95',
                       'tamoc/src/math_funcs.f95']
        )
    ]
}
                  
# Provide the setup utility
if __name__ == '__main__':

    from numpy.distutils.core import setup

    # Check if the user wants to try installing the Fortran library extension
    if '--python-only' in sys.argv:
        python_only = True
        sys.argv.remove('--python-only')
        print('Not attempting Fortran build -- installing Python-only version')
    else:
        python_only = False

    if python_only:
        # User wants to install only the python version of TAMOC
        pass # just in case there's something to do in the future
    else:
        # User want to try installing the TAMOC version that uses the
        # fortran extension module dbm_f
        setup_params.update(fortran_params)

    # Run the setup function
    setup(**setup_params)


