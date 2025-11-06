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
    
    Create an `ambient.Profile` object from Excel input files. Two input
    files may be provided. The `profile_data_file` contains the CTD data and
    must at least contain depth (`z`), temperature, and salinity. The CTD
    data may also contain pressure and concentrations for any measured
    dissolved compounds. The `current_data_file` contains the ambient current
    data. If provided, this file must at least contain the depth (`z`) and
    one velocity component. The velocity components should be named `ua`,
    `va`, and `wa`; missing components will be filled by zeros.
    
    The reason to allow the CTD and current data to be provided separately is
    that they are normally not known on the same grid of depths. If the
    current data are already known at each temperature and salinity
    measurement point, then the current data may be provided in the
    `profile_data_file` and the velocity component names (i.e., 'ua', 'va',
    and 'wa', as available) can be included in the 'chem_names' list. In that
    case, the `current_data_file` should be set to `None`.
    
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
    current_data_file : str, default=None
        The file name and optional relative or absolute file path to the Excel
        file containing the ambient current data. The input data must contain
        both `z` and columns for one or more velocity components.  If `None`,
        then the currents must be provided in the profile_data_file if 
        available.
    vel_components : list, None
        A list of strings specifying the names of the current components that
        should be included in the profile. An empty list or `None` will not
        add any data. Specifying the string 'all' will include all columns of
        data found in the current profile file.  If `None`, then current
        names must be provided in the `chem_names` list.
    stabilize_profile : bool
        A boolean specifying whether to enforce density stabilitization to
        the profile. 'False' will use all of the CTD data as is. 'True' will
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
    if not isinstance(current_data_file, type(None)):
        df_current = pd.read_excel(current_data_file)
    else:
        df_current = None
    
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
    
    # Parse the column names separately for the CTD and current data
    ctd_cols = excel_column_names(df_ctd)
    if not isinstance(current_data_file, type(None)):
        current_cols = excel_column_names(df_current)
    else:
        current_cols = None
    
    # Set the names of depth, temperature, salinity, and pressure.  Note that
    # all four of these parameters are required by TAMOC
    ztsp = ['z', 'temperature', 'salinity', 'pressure']
    
    # Set the names of the ambient currents as used in TAMOC
    uvw = ['ua', 'va', 'wa']
    
    # Get the correct list of names for the extra ctd data and velocity 
    # components to include
    chem_names, n_chems = make_variables_list(ctd_cols, ztsp, chem_names)
    if not isinstance(current_data_file, type(None)):
        current_names, n_current = \
            make_variables_list(current_cols, [], vel_components)
    else:
        current_names = []
        n_current = 0
      
    # Create arrays to hold the CTD and current data
    n_ctd_z = len(df_ctd[ctd_cols[0]]) - 1
    data = np.zeros((n_ctd_z, 4 + n_chems))
    data_units = []
    data_names = ztsp + chem_names
    if not isinstance(current_data_file, type(None)):
        n_current_z = len(df_current[current_cols[0]]) - 1
        current = np.zeros((n_current_z, 4))
        current_units = []
        current_names = ['z'] + current_names
    else:
        n_current_z = 0
        current = None
        current_units = []
        current_names = []
    
    # Get a parameter from a Excel data frame
    def extract_xlsx_data(df, param, data, units, i):
        """
        Get the data for parameter `param` from `df` and insert into the 
        units list and data arrray at column i
        
        Notes
        -----
        This function edits the units list and data array in place; hence, 
        there is no return value.
        
        """
        # The units are in the first row of data
        units[i] = df[param][0].strip('()')
        
        # The data are in the remaining rows
        data[:,i] = df[param][1:].to_numpy()
        
    # Get the required CTD data
    for param in data_names:
        data_units.append('')
        if param in ctd_cols:
            extract_xlsx_data(df_ctd, param, data, data_units, 
                data_names.index(param))
    
    # Get the current data
    if not isinstance(current_data_file, type(None)):
        for param in current_names:
            current_units.append('')
            if param in current_cols:
                extract_xlsx_data(df_current, param, current, current_units,
                    current_names.index(param))
    
    # Create the profile object from these data
    profile = profile_from_np(data, data_names, data_units, 
        current, current_units, stabilize_profile, err, add_air_to_profile, 
        compute_pressure)
    
    return profile

def profile_from_txt(profile_data_file, chem_names, current_data_file,
    vel_components, stabilize_profile, err, add_air_to_profile):
    """
    Create a Profile object from a text file
    
    Create an `ambient.Profile` object from text input files. Two input files
    may be provided. The `profile_data_file` contains the CTD data and must
    at least contain depth (`z`), temperature, and salinity. The CTD data may
    also contain pressure and concentrations for any measured dissolved
    compounds. The `current_data_file` contains the ambient current data. If
    provided, this file must at least contain the depth (`z`) and one
    velocity component. The velocity components should be named `ua`, `va`,
    and `wa`; missing components will be filled by zeros.
    
    The reason to allow the CTD and current data to be provided separately is
    that they are normally not known on the same grid of depths. If the
    current data are already known at each temperature and salinity
    measurement point, then the current data may be provided in the
    `profile_data_file` and the velocity component names (i.e., 'ua', 'va',
    and 'wa', as available) can be included in the 'chem_names' list. In that
    case, the `current_data_file` should be set to `None`.
    
    These text files should be in a format that can be read by `np.loadtxt`, 
    with the `header_flag` indicating each header row followed by a data 
    table of numbers that can be parsed using `delimiter`.  The contents of
    each column of data will be read from the header, where each column 
    should be documented in the format:
    
        Col ## = name, (units)
    
    where `##` is the column number, `name` is the variable names, and 
    `units` are the units for that variable in the data table.  
    
    Parameters
    ----------
    profile_data_file : str
        The file name and optional relative or absolute file path to the text
        file containing the ambient CTD data. The header should be at the 
        top of the file, with each header line begun with a 'header_flag'.    
        The data rows should contain numbers only in a format that can be 
        read by `np.loadtxt`.  Variable names and units will be parsed from
        the header using the format specified above.
    chem_names : list
        A list of strings with the names of other data in the Excel CTD
        data file that should be included in the ambient profile.  An empty
        list or `None` will not add any data.  Specifying the string 'all' 
        will include all columns of data found in the CTD file.
    current_data_file : str, default=None
        The file name and optional relative or absolute file path to the text
        file containing the ambient current data. The input data must contain
        both `z` and columns for one or more velocity components.  If `None`,
        then the currents must be provided in the profile_data_file if 
        available.  Variable names and units will be parsed from the header 
        using the format specified above.
    vel_components : list, None
        A list of strings specifying the names of the current components that
        should be included in the profile. An empty list or `None` will not
        add any data. Specifying the string 'all' will include all columns of
        data found in the current profile file.  If `None`, then current
        names must be provided in the `chem_names` list.
    stabilize_profile : bool
        A boolean specifying whether to enforce density stabilitization to
        the profile. 'False' will use all of the CTD data as is. 'True' will
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
    # Convert chem_names and vel_components to lists if strings
    if isinstance(chem_names, str):
        chem_names = [chem_names]
    if isinstance(vel_components, str):
        vel_components = [vel_components]
    
    # Load the CTD and current data
    ctd_data, ctd_header = read_txt_file(profile_data_file)
    if not isinstance(current_data_file, type(None)):
        current_data, current_header = read_txt_file(current_data_file)
    else:
        current_data = None
        current_header = []
    
    # Parse the column names separately for the CTD and current data
    ctd_cols, ctd_col_units = parse_txt_file_header(ctd_header)
    if not isinstance(current_data_file, type(None)):
        current_cols, current_col_units = \
            parse_txt_file_header(current_header)
    else:
        current_cols = None
        current_col_units = []

    # Set the names of depth, temperature, salinity, and pressure.  Note that
    # all four of these parameters are required by TAMOC
    ztsp = ['z', 'temperature', 'salinity', 'pressure']
    
    # Set the names of the ambient currents as used in TAMOC
    uvw = ['ua', 'va', 'wa']
    
    # Get the correct list of names for the extra ctd data and velocity 
    # components to include
    chem_names, n_chems = make_variables_list(ctd_cols, ztsp, chem_names)
    if not isinstance(current_data_file, type(None)):
        current_names, n_current = \
            make_variables_list(current_cols, [], vel_components)
    else:
        current_names = []
        n_current = 0
      
    # Create arrays to hold the CTD and current data
    n_ctd_z = len(ctd_data[:,0])
    data = np.zeros((n_ctd_z, 4 + n_chems))
    data_units = []
    data_names = ztsp + chem_names
    if not isinstance(current_data_file, type(None)):
        n_current_z = len(current_data[:,0])
        current = np.zeros((n_current_z, 4))
        current_units = []
        current_names = ['z'] + current_names
    else:
        n_current_z = 0
        current = None
        current_units = []
        current_names = []
    
    # Get a parameter from a Excel data frame
    def extract_np_data(data, names, units, param, profile_data, 
        profile_units, i):
        """
        Get the data for parameter `param` from `df` and insert into the 
        units list and data arrray at column i
        
        Parameters
        ----------
        data : ndarray
            Array of data read from a text file
        names : list
            A list of string variable names for each column in the `data` 
            array
        units : list
            A list of string unit names for each column in the `data` array
        param : list
            Parameter name to extract from the `data` array
        profile_data : ndarray
            Array of profile data where the data for this parameter should
            be stored in the final data array.  The text file name not 
            always include all required data (e.g., pressure may be missing)
            or the text file may be in an order not compatible with the 
            `ambient.Profile` object requiring the first columns of data to 
            be depth, temperature, salinity, and pressure
        profile_units : ndarray
            List of units for the data moved to `profile_data`
        i : int
            Integer index to `profile_data` and `profile_units` where the 
            final set of data should be stored.
        
        Notes
        -----
        This function edits the mutable units list and data array in place;
        hence, there is no return value.
        
        """
        # The units are in the first row of data
        profile_units[i] = units[names.index(param)]
        
        # The data are in the remaining rows
        profile_data[:,i] = data[:,names.index(param)]
        
    # Get the required CTD data
    for param in data_names:
        data_units.append('')
        if param in ctd_cols:
            extract_np_data(ctd_data, ctd_cols, ctd_col_units, param, data, 
                data_units, data_names.index(param))
    
    # Get the current data
    if not isinstance(current_data_file, type(None)):
        for param in current_names:
            current_units.append('')
            if param in current_cols:
                extract_np_data(current_data, current_cols, current_col_units,
                    param, current, current_units, 
                    current_names.index(param)) 
    
    # Create the profile object from these data
    profile = profile_from_np(data, data_names, data_units, 
        current, current_units, stabilize_profile, err, 
        add_air_to_profile, compute_pressure)
    
    return profile

def profile_from_np(data, data_names, data_units, current, 
    current_units, stabilize_profile, err, add_air_to_profile, 
    compute_pressure):
    """
    Create an `ambient.Profile` object from `ndarray` data
    
    Parameters
    ----------
    data : ndarray
        Array of CTD and chemistry data.  This array may also contain 
        ambient currents.  The first column must be depth, followed by 
        temperature, salinity, and pressure.  The remaining columns can be 
        any parameter, but it must be measured at the corresponding depth
        in the first column.  
    data_names : list
        String names of the data in `data`
    data_units : list
        String names of the units of the data in `data`
    current : ndarray
        Array of ambient current data in the order depth, easterly speed,
        northerly speed, and speed along the depth axis.  The depths in 
        this array do not have to correspond to the depths in the `data`
        array
    current_units : ndarray
        Units for each column of the `current_data` array.  Note that 
        `current_names` is not an input as the currents must be in the 
        order specified for `current_data`.  Trailing columns can be missing
        or data with zero speed may be provided
    stabilize_profile : bool
        A boolean specifying whether to enforce density stabilitization to
        the profile. 'False' will use all of the CTD data as is. 'True' will
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
    compute_pressure : bool, default=True
        A boolean stating whether to insert a computed pressure profile into
        the correct column of `data`.  Note that `data` must be passed with
        space in the array to hold the pressure no matter what.
    
    Returns
    -------
    profile : ambient.Profile
        The ambient.Profile object that corresponds to the provided data
        
    """
    # Compute the pressure if it was not provided
    if compute_pressure:
        data, data_units = compute_pressure(data, data_units)

    # Create the ambient profile
    n_currents = len(current_units)
    if n_currents > 0:
        # We have current data in a current data array
        profile = ambient.Profile(
            data, 
            ztsp=data_names[0:4], 
            chem_names=data_names[4:],
            err=err, 
            ztsp_units=data_units[0:4], 
            chem_units=data_units[4:], 
            current=current, 
            current_units=current_units, 
            stabilize_profile=stabilize_profile
        )
    else:
        # We do not have a separate current data array
        profile = ambient.Profile(
            data, 
            ztsp=data_names[0:4], 
            chem_names=data_names[4:],
            err=err, 
            ztsp_units=data_units[0:4], 
            chem_units=data_units[4:], 
            stabilize_profile=stabilize_profile
        )
    
    # Add the computed gas concentrations if requested
    if add_air_to_profile:
        profile.add_computed_gas_concentrations()
    
    # Return the final profile object
    return profile

def read_txt_file(fname, header_flag='#'):
    """
    Read the header and data from a text file
    
    Parameters
    ----------
    fname : str
        A string file name and optional relative or absolute path to the 
        text file to read
    header_flag : str
        The string character placed at the start of each header line in the
        file.
    
    Returns
    -------
    header : list
        A list of strings containing each line of the header
    data : ndarray
        A `numpy` array of data read using the `np.loadtxt` function
    
    """
    # Load the header information
    header = []
    with open(fname) as txt_data:
        for line in txt_data:
            if line[0] == header_flag:
                header.append(line)
    
    # Load the dataset
    data = np.loadtxt(fname)
    
    return (data, header)

def parse_txt_file_header(header):
    """
    Parse a text file header to extract the column names and units
    
    Parameters
    ----------
    header : list
        A list of strings containing each line of the header
    
    Returns
    -------
    var_names : list
        A list of string variable names for the names of each column.  If
        column names contain spaces in the original header file, all spaces
        are replaced by an underscore
    var_units : list
        A list of strings containing the corresponding units for each 
        variable in `var_names`
    
    """
    # Create empty lists to hold the variable names and units
    var_names = []
    var_units = []
    
    # Loop through each line of the header and extract the variable names
    # and their units
    for line in header:
        if 'Col' in line:
            
            # Pull out and format the variable name
            v0 = line.index(':') + 2
            v1 = line.index('(') - 2
            name = line[v0:v1].strip().split(' ')
            var_names.append('_'.join(name).lower())
            
            # Get the associated units
            u0 = line.index('(') + 1
            u1 = line.index(')')
            var_units.append(line[u0:u1])
    
    return (var_names, var_units)

def write_txt_file(fname, data, names, units, header_header=None, 
    header_footer=None):
    """
    Write a text file for a given array of data 
    
    Write a text file for an array of data in which each variable name in
    `names` is the corresponding column of `data` and the `units` list
    provides the units for each variable. This function will write a header
    string that defines each column in the dataset using the format:
    
        Col ## : name, (unit)
    
    Parameters
    ----------
    fname : str
        File name and optional relative or absolute path to the file that 
        should be written.  This file name should include the dot-extension
        as `.txt` is not assumed or appended.  However, this function will
        write an ASCII text file using the `np.savetxt` function.  This
        function uses the default behavior of `np.savetxt`, which is to 
        separate each column of data by a space.
    data : ndarray
        Array of data to store in a text file
    names : list
        List of variable names corresponding to each column of `data`
    units : list
        List of unit names corresponding ot each variables in `names`
    header_header : str, default=None
        String to write at the top of the file before the definition table
        of each column.  Note that this string will be written to the file
        as is, so if line breaks are needed, they should be included in the
        string
    header_footer : str, default=None
        String to write at the bottom of the header, after the table of column
        definitions and before the set of numbers. Note that this string will
        be written to the file as is, so if line breaks are needed, they
        should be included in the string
    
    """
    # Create the file header string
    header = create_txt_file_header(names, units, header_header, 
        header_footer)
    
    # Write the file
    np.savetxt(fname, data, header=header, comments='# ')

def create_txt_file_header(names, units, header_header, header_footer):
    """
    Create the header string that will be written to a text file of data
    
    Create a header string for a text file to be written by `np.savetxt`. 
    Because `numpy` will insert the header string flag '# ', this header
    string will be strictly a string of left-justified text.  The header
    string created by this file will have the format:
    
    <header_header>
    
    Each column of data below is defined as follows:
    
        Col 001 : <name>, (<unit>)
        ...
    
    <header_footer>
    
    Parameters
    ----------
    fname : str
        File name and optional relative or absolute path to the file that 
        should be written.  This file name should include the dot-extension
        as `.txt` is not assumed or appended.  However, this function will
        write an ASCII text file using the `np.savetxt` function.  This
        function uses the default behavior of `np.savetxt`, which is to 
        separate each column of data by a space.
    data : ndarray
        Array of data to store in a text file
    names : list
        List of variable names corresponding to each column of `data`
    units : list
        List of unit names corresponding ot each variables in `names`
    header_header : str, default=None
        String to write at the top of the file before the definition table
        of each column.  Note that this string will be written to the file
        as is, so if line breaks are needed, they should be included in the
        string
    header_footer : str, default=None
        String to write at the bottom of the header, after the table of column
        definitions and before the set of numbers. Note that this string will
        be written to the file as is, so if line breaks are needed, they
        should be included in the string
    
    Returns
    -------
    header : str
        A string containing the full header. 
    
    """
    # Create an empty list to hold each line of the header
    header = []
    
    # Insert the top of the header
    if not isinstance(header_header, type(None)):
        header.append(header_header + '\n')
    
    # Insert the table ot column descriptions
    header.append('\nEach column of data below is defined as follows:\n\n')
    for name in names:
        
        # Build the line to store in the header
        line = '    Col %.3d : ' % names.index(name)
        line += name + ', (' + units[names.index(name)] + ')\n'
        
        # Add this line to the header
        header.append(line)
    
    # Insert the footer at the end of the header
    if not isinstance(header_footer, type(None)):
        header.append('\n' + header_footer)
    
    # Add one blank line before the data start
    header.append('\n')
    
    # Create a single string from this list
    header = ''.join(header)
    
    # Return the final string header
    return header
            
def make_variables_list(col_names, exclude_names, include_names):
    """
    Create a complete list of column headers to include in dataset
    
    Determine which variable names in a data file should be included in the
    final `ambient.Profile` object.  For chemical property data, if the 'all'
    flag is used, a list of all available data needs to be generated.  
    Likewise, if a list of data is provided, this list should be compared
    against the data in a file, listing only required and available data. 
    This method takes a list of all available data, a list of data that
    should be explicitly excluded, and a list of data that should be 
    included and builds a comprehensive list with the correct union of 
    sets.
    
    Parameters
    ----------
    col_names : list
        List of string names for the variables available in a set of data
    exclude_names : list
        List of variable names to ignore from the `col_names` list.  This 
        will often be water depth `z` since it is an independent variable 
        or `temperature`, `salinity`, and `pressure` since these are 
        required and should not be considered additional data.
    include_names : list
        A final list of available data that has been requested by the user
        for inclusion.  This list may simply include the keyword 'all', which
        means to extract all available data.
    
    Returns
    -------
    include_names : list
        An itemized list of string variable names that exist in the dataset
        and were requested by the user
    n_names : int
        The number of names included in the list    
    
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


def compute_pressure(data, units):
    """
    Insert a computed pressure profile into a set of CTD data
    
    Insert the computed pressure into an array of ctd_data 
    
    Parameters
    ----------
    data : ndarray
        An array of depth, temperature, and salinity data that is missing 
        the computed pressure data
    units : list
        A list of string names specifying the units of the data in the 
        `data` array
    
    Returns
    -------
    data : ndarray
        An array with the pressure inserted in the column following the
        salinity (e.g., Python index 3) in the `data` array, Pa
    units : list
        An updated list of string names for the units of the data in the
        `data` array.  Note that this function must convert the data 
        to standard `tamoc` units before computing the pressure; hence, 
        potentially all of the units in the `units` list may be updated
    
    """
    # Convert the units of depth, temperature, and salinity to TAMOC
    # units
    for i in range(3):
        data[:,i], new_unit = ambient.convert_units(
            data[:,i], [units[i]])
        units[i] = new_unit[0]
    
    # Compute the pressure
    p = ambient.compute_pressure(data[:,0], data[:,1], data[:,2], 0)
    
    # Insert the pressure in the data array
    m, n = data.shape
    if n == 2:
        data = np.hstack((data, np.atleast_2d(p).transpose()))
    else:
        data[:,3] = p
        
    # Insert the pressure unit in the units list
    if len(units) == 3:
        units.append('Pa')
    else:
        units[3] = 'Pa'
    
    return (data, units)