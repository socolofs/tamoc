"""
Chemical Properties Script
==========================

Create a dictionary of chemical properties

This script provides tools to create a dictionary of the properties of
several hydrocarbons and chemicals of environmental interest in a global
dictionary for use by other programs that need to know chemical properties.

Parameters
----------

The chemical data are stored in ``./data/ChemData.csv`` and the
biodegradation data are stored in ``./data/BioData.csv``. In these files,
header rows are denoted by ``%``, the last row of pure text is taken as the
variable names, and the last row with ``()`` is taken as the units. The
columns should include a key name (e.g., `methane`) followed by numerical
values for each parameter in the file.

For the data provided by the model, the data sources and more details are
documented in the documentation (see ./docs/index.html).

Notes
-----
To use the properties database distributed by ``TAMOC``, you must import this
module and then use the `tamoc_data()` method, which reads the databases
distributed with ``TAMOC``.

See also
--------
`dbm` : Uses these dictionaries to create chemical mixture objects.

Examples
--------
>>> from tamoc import chemical_properties
>>> chem_db, chem_units, bio_db, bio_units = chemical_properties.tamoc_data()
>>> chem_db['oxygen']['M']
0.031998800000000001
>>> chem_units['M']
'(kg/mol)'

"""
# S. Socolofsky, January 2012, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

import numpy as np
import os

def load_data(fname):
    """
    Load a chemical properties file into memory
    
    Reads in a chemical properties file, creates a dictionary of the columns
    in the file, and performs some units conversions as necessary to have the
    data in SI mks units.  
    
    Parameters
    ----------
    fname : str
        file name (with relative path as necessary) where the chemical 
        property data is stored
    
    Returns
    -------
    data : dict
        dictionary of the properties for each column in the data file
    units : dict
        corresponding dictionary of units for each property in data
    
    Notes
    -----
    This function is used by the `dbm` module to load in the default chemical
    data in ./tamoc/data/chemdata.csv.  This function can also be called by
    the user to read in a user-specified file of chemical data present in any
    storage location.
    
    """
    # Set up counters to keep track of what has been and has not been read
    readnames = -1
    data = {}
    
    # Read in and parse the data from the chemistry data file.
    with open(fname) as datfile:
        for line in datfile:
            
            entries = line.strip().split(',')
            
            # Remove blank RHS column (Excel randomly includes extra columns)
            if len(entries[len(entries)-1]) == 0:
                entries = entries[0:len(entries)-1]
            
            # Identify and store the data
            if line.find('%') >= 0:
                # This is a header line
                
                if line.find('(') >= 0:
                    # This line contains the units
                    header_units = line.strip().split(',')
                
                elif (len(entries[1]) > 0) and(readnames < 0):
                    # This line contains the variable names
                    header_keys = line.strip().split(',')
                    readnames = 1
                
            else:
                # This is a data line
                data[entries[0]] = {}
                for i in range(1, len(entries)):
                    data[entries[0]][header_keys[i]] = np.float64(entries[i])
            
    # Add the units to two different dictionaries
    read_units = {}
    for i in range(len(header_units) - 1):
        read_units[header_keys[i]] = header_units[i]
    units = {}
    for i in range(len(header_units) - 1):
        units[header_keys[i]] = header_units[i]
                      
    # Convert to SI units.  If you add a new unit to the file ChemData.csv, 
    # then you should include a check for it here.
    for chemical in data:
        for variable in read_units:
            if read_units[variable].find('g/mol') >= 0:
                # Convert to kg/mol
                data[chemical][variable] = data[chemical][variable] / 1000.
                units[variable] = '(kg/mol)'
            
            if read_units[variable].find('psia') >= 0:
                # Convert to Pa
                data[chemical][variable] = data[chemical][variable] * 6894.76
                units[variable] = '(Pa)'
            
            if read_units[variable].find('(deg F)') >= 0:
                # Convert to K
                data[chemical][variable] = (data[chemical][variable] - 32.) * \
                                           5. / 9. + 273.15
                units[variable] = '(K)'
            
            if read_units[variable].find('mol/dm^3 atm') >= 0:
                # Convert to kg/(m^3 Pa)
                data[chemical][variable] = (data[chemical][variable] * \
                                           1000. / 101325. * \
                                           data[chemical]['M'])
                units[variable] = '(kg/(m^3 Pa))'
            
            if read_units[variable].find('mm^2/s') >= 0:
                # Convert to m^2/s
                data[chemical][variable] = data[chemical][variable] / 1000.**2
                units[variable] = '(m^2/s)'
            
            if read_units[variable].find('cal/mol') >= 0:
                # Convert to J/mol
                data[chemical][variable] = data[chemical][variable] / 0.238846
                units[variable] = '(J/mol)'
            
            if read_units[variable].find('L/mol') >= 0:
                # Convert to m^3/mol
                data[chemical][variable] = data[chemical][variable] / 1000.
                units[variable] = '(m^3/mol)'
            
            if read_units[variable].find('1/d') >= 0:
                # Convert to 1/s
                data[chemical][variable] = data[chemical][variable] / 86400.
                units[variable] = '(1/s)'
            
            if read_units[variable].find('(d)') >= 0.:
                # Convert to s
                data[chemical][variable] = data[chemical][variable] * 86400.
                units[variable] = '(s)'
            
            if read_units[variable].find('(g/cm^3)') >= 0.:
                # Convert to kg/m^3
                data[chemical][variable] = data[chemical][variable] / 1000. \
                    * 100.**3
                units[variable] = '(kg/m^3)'
            
            if read_units[variable].find('(ft^3/lb-mol)') >= 0.:
                # Convert to m^3/mol
                data[chemical][variable] = data[chemical][variable] * \
                    (12. * 2.54 / 100.)**3 / 453.59237
                units[variable] = '(m^3/mol)'
            
            if read_units[variable].find('(ft^3/lb-mol/deg F)') >= 0.:
                # Convert to m^3/mol
                data[chemical][variable] = data[chemical][variable] * \
                    (12. * 2.54 / 100.)**3 / 453.59237 / (5./9.)
                units[variable] = '(m^3/mol/deg C)'

            if read_units[variable].find('(BTU/lb-mol)') >= 0.:
                # Convert to J/mol
                data[chemical][variable] = data[chemical][variable] * \
                    1055.06 / 453.59237 
                units[variable] = '(J/mol)'
            
            if read_units[variable].find('(BTU/lb-mol/deg F)') >= 0.:
                # Convert to J/mol/deg C
                data[chemical][variable] = data[chemical][variable] * \
                    1055.06 / 453.59237 / (5./9.)
                units[variable] = '(J/mol/deg C)'
           
            if read_units[variable].find('(BTU/lb-mol/deg F^2)') >= 0.:
                # Convert to J/mol/deg C^2
                data[chemical][variable] = data[chemical][variable] * \
                    1055.06 / 453.59237 / (5./9.)**2
                units[variable] = '(J/mol/deg C^2)'

            if read_units[variable].find('(BTU/lb-mol/deg F^3)') >= 0.:
                # Convert to J/mol/deg C^3
                data[chemical][variable] = data[chemical][variable] * \
                    1055.06 / 453.59237 / (5./9.)**3
                units[variable] = '(J/mol/deg C^3)'

            if read_units[variable].find('(BTU/lb-mol/deg F^4)') >= 0.:
                # Convert to J/mol/deg C^4
                data[chemical][variable] = data[chemical][variable] * \
                    1055.06 / 453.59237 / (5./9.)**4
                units[variable] = '(J/mol/deg C^4)'

    return (data, units)

def tamoc_data():
    """
    Load the supplied chemical properties file from the `TAMOC` distribution 
    
    Reads in the chemical properties file provided with `TAMOC`, creates a
    dictionary of the columns in the file, and performs some units
    conversions as necessary to have the data in SI mks units.
    
    Returns
    -------
    data : dict
        dictionary of the properties for each column in the data file
    units : dict
        corresponding dictionary of units for each property in data
    
    Notes
    -----
    This function read in the the default chemical data in
    ./tamoc/data/chemdata.csv. 
    
    """
    # Get the relative path to the ./tamoc/data directory
    __location__ = os.path.realpath(os.path.join(os.getcwd(), 
                                    os.path.dirname(__file__), 'data'))
    
    # Create the full relative path to the default data in ChemData.csv
    chem_fname = os.path.join(__location__,'ChemData.csv')
    bio_fname = os.path.join(__location__,'BioData.csv')
    PJ_fname = os.path.join(__location__,'PJData.csv')
    
    # Load in the default data and their units
    chem_data, chem_units = load_data(chem_fname)
    bio_data, bio_units = load_data(bio_fname)
    PJ_data, PJ_units = load_data(PJ_fname)
    
    # Return the results
    return (chem_data, chem_units, bio_data, bio_units, PJ_data, PJ_units)



