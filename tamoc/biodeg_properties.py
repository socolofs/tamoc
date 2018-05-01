"""
Biodegradation Properties Script
================================

Create a dictionary of biodegradation properties

This script creates a dictionary of the properties of several hydrocarbons and
chemicals of environmental interest in a global dictionary for use by other
programs that need to know biodegradation properties.

Parameters
----------
The biodegradation data are stored in ``./data/BioData.csv``. Header rows are 
denoted by `%, the last row of pure text is taken as the variable names and
the last row with `()` is taken as the units.  The columns should include
a key name (e.g., `methane`), the first-order decay constant, and the 
biodegrdation lag time.  For unknown parameter values, use -9999.

For the data provided by the model, the data sources and more details are
documented in the file ``../docs/BioData_ReadMe.txt``.

This module can read in any number of columns of chemical data, including
parameters not listed below. Units will be converted to standard SI units,
and the conversion function will operate on 1/d and d. The returned variables
`units` will contain the final set of units for the database. To use the
TAMOC suite of models, all parameters listed below in the `data` dictionary
must be provided with the variable names listed.

Returns
-------
data : dict
    a nested dictionary containing the chemical name as the first key 
    and the following list of secondary keys matched with the numerical
    value of each variable:
       
       k_bio : first-order biodegradation rate constant (1/s)
       t_bio : biodegradation lag time (s)

units : dict
    dictionary with the same keys as the variable names listed above 
    (e.g., k_bio, t_bio, etc.) linked to a string containing the units of 
    each variable.

Notes
-----
To use the properties database distributed by TAMOC, simply import this file
in Python: the results will be returned in `biodeg_properties.data`. To
import a user-defined database of properties, use the function ``load_data``
provided in the `chemical_properties` module. The ``TAMOC`` suite of models
will pull data from both the default and any user-specified database, giving
first priority to parameter keys found in the user-specified database.

See also
--------
`dbm` : Uses these dictionaries to create chemical mixture objects.

Examples
--------
>>> from tamoc import biodeg_properties as biodeg
>>> biodeg.data['methane']['k_bio']
x.xxxxx
>>> biodeg.units['k_bio']
'(1/s)'

"""
# S. Socolofsky, January 2012, Texas A&M University <socolofs@tamu.edu>.
import numpy as np
import chemical_properties as chem_data
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
            if len(entries[len(entries)-1]) is 0:
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
            
            if read_units[variable].find('F') >= 0:
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
                # Convert to m^3/s
                data[chemical][variable] = data[chemical][variable] / 1000.**2
                units[variable] = '(m^3/s)'
            
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
                data[chemica][variable] = data[chemical][variable] / 86400.
                units[variable] = '(1/s)'
            
            if read_units[variable].find('d') >= 0.:
                # Convert to s
                data[chemical][variable] = data[chemical][variable] * 86400.
                units[variable] = '(s)'
            
    return (data, units)


if __name__ == 'tamoc.chemical_properties':
    # Get the relative path to the ./tamoc/data directory
    __location__ = os.path.realpath(os.path.join(os.getcwd(), 
                                    os.path.dirname(__file__), 'data'))
    
    # Create the full relative path to the default data in ChemData.csv
    fname = os.path.join(__location__,'ChemData.csv')
    
    # Load in the default data and their units
    data, units = load_data(fname)


