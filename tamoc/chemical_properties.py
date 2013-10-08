"""
Chemical Properties Script
==========================

Create a dictionary of chemical properties

This script creates a dictionary of the properties of several hydrocarbons and
chemicals of environmental interest in a global dictionary for use by other
programs that need to know chemical properties.

Parameters
----------
The chemical data are stored in ``./data/ChemData.csv``. The columns in the 
file are expected to contain a key name, the molecular weight, critical-point
pressure, critical-point temperature, acentric factor, Henry's law constant at
298.15 K, enthalpy of solution, specific volume at inifinite dilution, and 
parameters of the diffusivity model.  The data sources and more details are
documented in the file ``../docs/ChemData_ReadMe.txt``.

Returns
-------
data : dict
    a nested dictionary containing the chemical name as the first key 
    and the following list of secondary keys matched with the numerical
    value of each variable:
       
       M : molecular weight (kg/mol)
       Pc : pressure at the critical point (Pa)
       Tc : temperatue at the critical point (K)
       omega : acentric factor (--)
       kh_0 : Henry's law constant at 298.15 K (kg/(m^3 Pa))
       dH_solR : enthalpy of solution / R (K)
       nu_bar : specific volume at infinite dilution (m^3/mol)
       B : diffusivity model coefficient (m^2/s)
       dE : diffusivity model coefficient (J/mol)

units : dict
    dictionary with the same keys as the variables names listed above 
    (e.g., M, Pc, Tc, etc.) linked to a string containing the units of 
    each variable.

See also
--------
`dbm` : Uses these dictionaries to create chemical mixture objects.

Examples
--------
>>> from tamoc import chemical_properties as chem
>>> chem.data['oxygen']['M']
0.031998800000000001
>>> chem.units['M']
'(kg/mol)'

"""
# S. Socolofsky, January 2012, Texas A&M University <socolofs@tamu.edu>.
import numpy as np
import os

# Set up counters to keep track of what has been and has not been read
readnames = -1
data = {}

# Figure out the path to the ChemData.csv file
__location__ = os.path.realpath(os.path.join(os.getcwd(), 
                                os.path.dirname(__file__), 'data'))

# Read in and parse the data from the chemistry data file.
with open(os.path.join(__location__,'ChemData.csv')) as datfile:
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
            data[entries[0]] = {header_keys[i] : np.float64(entries[i])
                    for i in range(1, len(entries))}

# Add the units to the dictionary
units = {header_keys[i] : header_units[i]
         for i in range(len(header_units) - 1)}

# Convert to SI units.  If you add a new unit to the file ChemData.csv, then 
# you should include a check for it here.
for chemical in data:
    if units['M'].find('g/mol') >= 0:
        data[chemical]['M'] = data[chemical]['M'] / 1000.
    
    if units['Pc'].find('psia') >= 0:
        data[chemical]['Pc'] = data[chemical]['Pc'] * 6894.76
    
    if units['Tc'].find('R') >= 0:
        data[chemical]['Tc'] = (data[chemical]['Tc'] - 32.) * 5. / 9. + 273.15
    
    if units['kh_0'].find('mol/dm^3 atm') >= 0:
        data[chemical]['kh_0'] = (data[chemical]['kh_0'] * 1000. / 101325. *
                                  data[chemical]['M'])
    
    if units['B'].find('cm^2/sec') >= 0:
        data[chemical]['B'] = data[chemical]['B'] * 1.e-2 / 100.**2
    
    if units['dE'].find('cal/mol') >= 0:
        data[chemical]['dE'] = data[chemical]['dE'] / 0.238846
    

# Now that all the data are converted, store the correct units
units = {header_keys[0]: '(--)', 
         header_keys[1]: '(kg/mol)', 
         header_keys[2] : '(Pa)', 
         header_keys[3] : '(K)', 
         header_keys[4] : '(--)', 
         header_keys[5] : '(kg/m^3 Pa)', 
         header_keys[6] : '(K)',
         header_keys[7] : '(m^3/mol)',
         header_keys[8] : '(m^2/s)',
         header_keys[9] : '(J/mol)'}