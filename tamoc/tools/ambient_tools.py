"""
ambient_tools.py
----------------

Tools to help create ambient profiles from different data sources.

"""
# S. Socolofsky, Texas A&M University, August 2025, <socolofs@tamu.edu>

from tamoc import ambient

import numpy as np

def profile_from_excel(profile_data_file, chem_names, current_data_file,
    vel_components, stabilize_profile, err, add_air_to_profile):
    """
    Create a Profile object from Excel input files
    
    Create an `ambient.Profile` object from Excel input files.  Two input
    files should be provided.  The `profile_data_file` contains the CTD 
    data and must at least contain depth (`z`), temperature, and salinity.
    The CTD data may also contain pressure and concentrations for any
    measured dissolved compounds.  The `current_data_file` contains the
    ambient current data.  This file must at least contain the depth (`z`) 
    and one velocity component.  The velocity components should be named
    `ua`, `va`, and `wa`; missing components will be filled by zeros.
    
    Parameters
    ----------
    profile_data_file : str
        The file name and optional relative or absolute file path to the Excel
        file containing the ambient CTD data. The file should have the first
        row contain the parameter names, the second row the parameter units
        and the remaining rows the data. The input data must contain at least
        `z`, `temperature`, and `salinity`.
    chem_names : list
        A list of strings with the names of other data in the Excel CTD
        data file that should be included in the ambient profile.  An empty
        list or `None` will not add any data.  Specifying the string 'all' 
        will include all columns of data found in the CTD file.
    current_data_file : str
        The file name and optional relative or absolute file path to the Excel
        file containing the ambient current data. The input data must contain
        both `z` and columns for one or more velocity components.
    vel_components : list
        A list of strings specifying the names of the current components that
        should be included in the profile. An empty list or `None` will not
        add any data. Specifying the string 'all' will include all columns of
        data found in the current profile file.
    stabilize_profile : bool
        A boolean specifying whether to enforce density stabilitization to 
        the profile.  'False' will use all of the CTD data as is.  'True' will 
        inspect the profile and remove any data points that would result in
        unstable density stratification (e.g., a water density greater than
        the potential density at the next deeper point in the profile).
    err : float
        The relative error to allow using linear interpolation.  If set to 
        zero, all of the data in the profile will be used.  If set to a 
        non-zero value, then data will be removed from the profile such that
        this maximum error would not be exceeded by linear interpolation 
        with the remaining data.  This is used to reduce the data in large
        datasets to speed up interpolation.
    add_air_to_profile : bool
        A boolean stating whether to add the computed concentrations of 
        nitrogen, oxygen, argon, and carbon_dioxide to the profile.  If one
        of these compounds is already listed in `chem_names`, those values
        will not be replaced by this method; this only adds missing 
        atmospheric gas concentrations.
    
    Returns
    -------
    profile : ambient.Profile
        A TAMOC `ambient.Profile` object that contains the given CTD and
        current data.  Note that all data will be converted to normal
        TAMOC units (kg, m, s).  If `add_air_to_profile` is `True`, the 
        computed concentrations of nitrogen, oxygen, argon, and carbon
        dioxide will also be added to the profile.
    
    """ 
    # Use Pandas to read Excel files
    import pandas as pd
    
    # Convert chem_names and vel_components to lists if strings
    if isinstance(chem_names, str):
        chem_names = [chem_names]
    if isinstance(vel_components, str):
        vel_components = [vel_components]
    
    # Load the CTD and current data
    df_ctd = pd.read_excel(profile_data_file)
    df_current = pd.read_excel(current_data_file)
    
    # Get the names of the data columns in each spreadsheet
    def excel_column_names(df):
        """
        Return the non-empty column header names of an Excel data frame
        
        """
        names = []
        for key in df:
            if 'Unnamed' not in key:
                names.append(key)
            
        return names
    ctd_cols = excel_column_names(df_ctd)
    current_cols = excel_column_names(df_current)    
    
    # Set the names of depth, temperature, salinity, and pressure.  Note that
    # all four of these parameters are required by TAMOC
    ztsp = ['z', 'temperature', 'salinity', 'pressure']
    
    # Set the names of the ambient currents as used in TAMOC
    uvw = ['ua', 'va', 'wa']
    
    # Get complete lists for chem_names and vel_components if 'all' was used
    def compile_list(col_names, exclude_names, include_names):
        """
        Decide which column names should be included in the profile
        
        """
        # Get itemized lists and the number of parameters
        if not isinstance(include_names, type(None)):
            # There are names to include
            if 'all' in include_names:
                # Get all of the extra columns of data
                include_names = []
                for name in col_names:
                    if name not in exclude_names:
                        include_names.append(name)
                n_names = len(incluude_names)
                
            else:
                # We were provided the correct list, just count it
                n_names = len(include_names)
                            
        else:
            # No extra column should be included
            include_names = []
            n_names = 0
        
        # Return the populated list and number of parameters
        return (include_names, n_names)
    
    # Get the correct list of names for the extra ctd data and velocity 
    # components to include
    chem_names, n_chems = compile_list(ctd_cols, ztsp, chem_names)
    current_names, n_current = compile_list(current_cols, [], vel_components)
      
    # Create arrays to hold the CTD and current data
    n_ctd_z = len(df_ctd[ctd_cols[0]]) - 1
    ctd_data = np.zeros((n_ctd_z, 4 + n_chems))
    ztsp_units = []
    chem_units = []
    n_current_z = len(df_current[current_cols[0]]) - 1
    current_data = np.zeros((n_current_z, 4))
    current_units = []
    
    # Get a parameter from a Excel data frame
    def extract_data(df, param, units, data, i):
        """
        Get the data for parameter `param` from `df` and insert into the 
        units list and data arrray at column i
        
        Notes
        -----
        This function edits the units list and data array in place; hence, 
        there is no return value.
        
        """
        # The units are in the first row of data
        units.append(df[param][0].strip('()'))
        
        # The data are in the remaining rows
        data[:,i] = df[param][1:].to_numpy()
        
    # Get the required CTD data
    i = 0
    for param in ztsp:
        if param in ctd_cols:
            extract_data(df_ctd, param, ztsp_units, ctd_data, i)
            i += 1
    
    # Get the extra chemical property data
    i = 4
    for param in chem_names:
        extract_data(df_ctd, param, chem_units, ctd_data, i)
        i += 1
    
    # Get the current data
    extract_data(df_current, 'z', current_units, current_data, 0)
    i = 1
    for component in uvw:
        if component not in current_names:
            current_data[:,i] = 0.
            current_units.append('m/s')
        else:
            extract_data(df_current, component, current_units, current_data, 
                i)
        i += 1
    
    # Compute the pressure if it was not provided
    if 'pressure' not in ctd_cols:
        
        # Convert the units of depth, temperature, and salinity to TAMOC
        # units
        for i in range(3):
            ctd_data[:,i], new_unit = ambient.convert_units(
                ctd_data[:,i], [ztsp_units[i]])
            ztsp_units[i] = new_unit[0]
        
        # Compute the pressure
        p = ambient.compute_pressure(ctd_data[:,0], ctd_data[:,1], 
            ctd_data[:,2], 0)
        
        # Insert the pressure and the units
        ztsp_units.append('Pa')
        ctd_data[:,3] = p
        
    # Create the ambient profile
    profile = ambient.Profile(ctd_data, ztsp=ztsp, chem_names=chem_names,
        err=err, ztsp_units=ztsp_units, chem_units=chem_units, 
        current=current_data, current_units=current_units, 
        stabilize_profile=stabilize_profile)
    
    # Add the computed gas concentrations if requested
    if add_air_to_profile:
        profile.add_computed_gas_concentrations()
    
    # Return the final profile object
    return profile