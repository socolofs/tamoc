"""
Ambient Module
==============

Define functions, classes, and methods to handle ambient sewater data

This module defines functions to read in arbitrary format ambient data files
(e.g., CTD profiles), manipulate the data to extract profiles with monotonic
depth coordinate values, and adjust parameter units to standard units. It also
defines functions and classes to store the measured CTD data in
netCDF4-classic format following NODC guidelines and to manipulate the data
for use by simulation models in ``TAMOC``. These manipulations include
interpolation methods, the ability to add or remove data from the database,
including the addition of synthetic data when needed, and to extend a profile
to deeper depths using a rational means of maintaining the stratification
structure.

These methods are particularly useful to rapidly create ambient profile 
databases for use by ``TAMOC`` and for archiving data obtained from arbitrary 
formats in standard format netCDF4-classic files.  These methods also allow
seemless coupling of ``TAMOC`` simulation modules with general ocean 
circulation models or Lagrangian particle tracking models that store their 
seawater properties data in netCDF format.

See Also
--------
`netCDF4` : 
    Package for creating and manipulating netCDF datasets

`datetime` : 
    Package to create and manipute dates

`numpy.fromfile` :
    Read data from a simple text file in single table format. Similar to the
    Matlab `load` function. Does not support comments or skipped lines. Works
    for text or binary files.

`numpy.loadtxt` : 
    A more sophisticated method to read data from a text file. This method can
    handle multiple data types (e.g., strings, floats, and integers together),
    can perform transformation on the data (e.g., date format to a date
    object), and can unpack to a tuple of variables. One limitation is that
    all rows must have the same number of columns and there is no method to
    handle missing data.

`numpy.genfromtxt` : 
    The most advanced numpy method to read data from a file. Includes the
    capabilities in `numpy.loadtxt`, but also allows for missing data, data
    flags, and multiple methods to replace missing data or flags.


"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from tamoc import seawater

from netCDF4 import Dataset
from netCDF4 import num2date, date2num
from datetime import datetime
from time import ctime

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from string import join, capwords
from copy import copy

class Profile(object):
    """
    Class object for ambient seawater profiles
    
    This object collects the data describing the ambient seawater (e.g., 
    CTD data, currents, etc.) and provides efficient access to interpolated
    results at arbitrary depths.  All of the raw data are stored in a 
    netCDF4-classic format dataset, which resides on the hard-drive.  Profile
    objects can be initiated either from an open netCDF dataset object or 
    from a file path to the desired object.  The netCDF dataset is expected
    to have variables for 'z', 'temperature', 'salinity', and 'pressure' and 
    any other variables requested at instantiation through the chem_names
    variable.  
    
    Parameters
    ----------
    nc : netCDF dataset object, path, or str
        Provides the information necessary to access the netCDF dataset.  If
        a file path or string is provided, then the netCDF file is opened
        and the resulting dataset object is stored in self.nc.
    ztsp : str list
        String list containing the variables names for depth, temperature, 
        salinity, and pressure that are used in the netCDF dataset.
    chem_names : str list, optional
        Names of the chemicals (e.g., those constituents in addition to z, T, 
        S, P) in the netCDF dataset that should be accessible through the 
        `self.get_values` interpolation method or the `self.get_units` 
        interrogator.  If `chem_names` = 'all', then all variables in the 
        netCDF file except for 'time', 'lat', 'lon', and the strings in 
        `ztsp` will be loaded as ambient chemical data.
    err : float
        The interpolation dataset is a subset of the complete raw dataset 
        stored in the netCDF file.  err sets the acceptable level of 
        relative error using linear interpolation expected of the 
        `self.get_values` method.  This value is passed to the `coarsen` 
        function to provide an optimal interpolation dataset.
    
    Attributes
    ----------
    nc_open : bool
        Flag stating whether or not the netCDF dataset is open or closed
    nchems : int 
        Number of chemicals in `chem_names`
    z : ndarray
        Array containing the complete raw dataset of depths
    y : ndarray
        Array containing the complete raw dataset for T, S, P, and chemicals 
        in `chem_names`. 
    f_names : str list
        concatenated string list containing `ztsp` and `chem_names`
    f_units : str list
        List of units associated with the variables stored in `f_names`.
    f : object
        `scipy.interpolate.interp1d` object containing `z` and `y`.
    
    See Also 
    --------
    netCDF4, create_nc_db, fill_nc_db, coarsen, chemical_properties
    
    Examples
    --------
    >>> bm54 = Profile('./test/output/test_BM54.nc', chem_names='all')
    >>> print bm54.nc.variables.keys()
    ['time', 'lat', 'lon', 'z', 'temperature', 'salinity', 'pressure', 
    'T', 'wetlab_fluorescence', 'S', 'density', 'oxygen']
    >>> bm54.get_values(1000.0, ['temperature', 'salinity', 'pressure'])
    array([  2.78274540e+02,   3.49278396e+01,   1.01933088e+07])
    >>> bm54.get_units('oxygen')
    ['kg/m^3']
    >>> bm54.buoyancy_frequency(1500.)
    0.00081815
    >>> bm54.nc_close()
    
    """
    def __init__(self, nc, ztsp=['z', 'temperature', 'salinity', 
                 'pressure'], chem_names=None, err=0.01):
        super(Profile, self).__init__()
        
        # Get the appropriate netCDF dataset object
        if isinstance(nc, str) or isinstance(nc, unicode):
            nc = Dataset(nc, 'a')
        
        # Mark the netCDF file as open
        self.nc_open = True
        
        # Check chem_names
        if chem_names == 'all':
            keys = nc.variables.keys()
            non_chems = ['time', 'lat', 'lon'] + ztsp
            chem_names = [name for name in keys if name not in non_chems]
        elif isinstance(chem_names, str):
            chem_names == [chem_names]
        
        # Store the input variables
        self.nc = nc
        self.ztsp = ztsp
        if chem_names is None:
            self.chem_names = []
            self.nchems = 0
        else:
            self.chem_names = chem_names
            self.nchems = len(chem_names)
        self.err = err
        
        # Build an interpolation function
        self.build_interpolator()
    
    def build_interpolator(self):
        """
        Build the interpolator function from the netCDF dataset
        
        Extract the raw data from the netCDF dataset using the required
        variables T, S, and P plus any variables requested by the user through
        `self.chem_names` and store the resulting data in z and y.  Coarsen
        the raw dataset to the level specified by `self.err` using the 
        `coursen` function and put the resulting reduced dataset into the 
        interpolation function `f`.
        
        Notes
        -----
        This function is responsible for creating the object attributes `z`, 
        `y`, `f_names`, `f_units`, and `f`.
        
        This method is called by the object initializer and by the object 
        append method.  *It should not be called directly by the user*.
        
        """
        if self.nc_open is True:
            # Extract the depths
            self.z = self.nc.variables[self.ztsp[0]][:]
            
            # List the ambient profile components
            self.f_names = self.ztsp[1:] + self.chem_names
            
            # Extract the ambient profile data and units
            self.f_units = []
            self.y = np.zeros((self.z.shape[0], len(self.f_names)))
            for i in range(len(self.f_names)):
                self.y[:,i] = self.nc.variables[self.f_names[i]][:]
                self.f_units.append(self.nc.variables[self.f_names[i]].units)
            
            # Create the interpolation function 
            db = np.hstack((np.atleast_2d(self.z).transpose(), self.y))
            db = coarsen(db, self.err)
            db = stabilize(db)
            self.f = interp1d(db[:,0], db[:,1:].transpose())
            
            # Set the valid range of the interpolator
            self.z_max = np.max(self.z)
            self.z_min = np.min(self.z)
            
        else:
            raise ValueError('The netCDF dataset is already closed so ' + 
                             'the interpolator cannot be updated.')
    
    def append(self, data, var_symbols, var_units, comments, z_col=0):
        """
        Add data to the netCDF dataset and update the object attributes
        
        This method provides an interface to the `fill_nc_db` function
        and performs the necessary updates to all affected object attributes.
        This is the only way that data should be added to a netCDF file 
        contained in a Profile class object.
        
        Parameters
        ----------
        data : ndarray
            Table of data to add to the netCDF database.  If it contains more
            than one variable, the data are assumed to be arranged in columns.
        var_symbols : string list
            List of string symbol names (e.g., T, S, P, etc.) in the same 
            order as the columns in the data array.  For chemical properties,
            use the key names in the chemical_properties.py database.
        var_units : string list
            List of units associated with each variable in the `var_symbols` 
            list.
        comments : string list
            List of comments associated with each variable in the 
            `var_symbols` list.  As a minimum, this list should include the 
            indications 'measured' or 'derived' or some similar indication of 
            source of the data.
        z_col : integer, default is 0
            Column number of the column containing the depth data.  The first 
            column is numbered zero.
        
        Notes
        -----
        Once a Profile object is created, data should only be added to the 
        object's netCDF dataset through this append method.  While direct 
        calls to `ambient.fill_nc_db` will not create errors, the resulting
        netCDF dataset will no longer be compatible with the Profile object
        attributes.
        
        """
        if self.nc_open is True:
            # Make sure the constituent names are in a list
            if isinstance(var_symbols, str):
                var_symbols = [var_symbols]
            
            # Add the data to the netCDF dataset
            self.nc = fill_nc_db(self.nc, data, var_symbols, var_units, 
                                 comments, z_col)
            
            # Add the new chemicals to the chem variables
            for constituent in var_symbols:
                # Make sure the dependent variable is never listed as a 
                # chemical
                if constituent != self.ztsp[0]:
                    self.chem_names.append(constituent)
                    self.nchems += 1
            
            # Rebuild the interpolator
            self.build_interpolator()
        else:
            raise ValueError('The netCDF dataset is already closed so ' + 
                             'data cannot be added to it.')
    
    def get_values(self, z, names):
        """
        Return values for the variables listed in `names` interpolated to the
        depth given by `z`
        
        Parameters
        ----------
        z : float or ndarray
            Depth(s) at which data are desired.  If the value of `z` lies 
            outside the `valid_min` or `valid_max` values of `z` in the 
            netCDF dataset, the values at the nearest boundary are returned.
        names : string list
            List of variable names (e.g., temperature, oxygen, etc.) for 
            which the interpolated data are desired.  If the parameter name
            is not present in the database, a value of zero is returned.
        
        Returns
        -------
        yp : ndarray
            An array of values sorted in the same order as `names` and 
            interpolated at the depth(s) given by `z`.  If `z` is a row 
            vector, `yp` contains `z` in the first column and the other 
            variables in the adjacent columns.
        
        """
        # Make sure names is a list
        if isinstance(names, str):
            names = [names]
        
        # Make sure z is an array
        if not isinstance(z, np.ndarray):
            if not isinstance(z, list):
                z = np.array([z])
            else:
                z = np.array(z)

        # Catch the out of range error.  This should only occur when an ODE
        # solver gets close to the boundary; thus, it is acceptable to revert
        # to the solution at the boundary
        z[z<self.z_min] = self.z_min
        z[z>self.z_max] = self.z_max
        
        # Get the columns in the output where the interpolated data will go
        ans_cols = [names.index(name) 
                    for name in names if name in self.f_names]
        
        if ans_cols:
            # Get the names of these interpolated data
            i_names = [names[col] for col in ans_cols]

            # Get the columns in the interpolation dataset where these names 
            # are located
            i_cols = [self.f_names.index(name) for name in i_names]
            
            # Interpolate the data and insert into the output array
            if z.shape[0] == 1:
                # f(z) will be a column vector
                ans = np.zeros(len(names))
                interp_data = self.f(z)[i_cols].transpose()
                ans[ans_cols] = interp_data
            else:
                # f(z) will be a 2d matrix
                ans = np.zeros((z.shape[0], len(names)))
                interp_data = self.f(z).transpose()[:,i_cols]
                ans[:,ans_cols] = interp_data
        else:
            # Return the appropriately shaped zeros matrix
            if z.shape[0] == 1:
                ans = np.zeros(len(names))
            else:
                ans = np.zeros((z.shape[0], len(names)))
        
        # Always return a ndarray         
        return ans
    
    def get_units(self, names):
        """
        Return a list of units for the variables in `names`
        
        Parameters
        ----------
       names : string list
            List of variable names (T, S, P or entries in `chem_names`) for 
            which the units are desired.
        
        Returns
        -------
        units : string list
            A list of strings specifying the units of each variables in the 
            same order as they are listed in `names`
        
        Notes
        -----
        The names of the units are extracted from the netCDF dataset.  
        
        """
        # Make sure names is a list
        if isinstance(names, str):
            names = [names]
        
        # Return the list of units
        ans = []
        for name in names:
            ans.append(self.f_units[self.f_names.index(name)])
        
        return ans
    
    def buoyancy_frequency(self, z, h=0.01):
        """
        Calculate the local buoyancy frequency
        
        Calculate the buoyancy frequency at the depth `z`, optionally using 
        the length-scale `h` to obtain smooth results.  This calculation uses
        the in-situ pressure at the depth z as a constant so that the effect
        of compressibility is removed.  
        
        Parameters
        ----------
        z : float or ndarray
            Depth(s) (m) at which data are desired.  The value of `z` must lie 
            between the `valid_min` and `valid_max` values of `z` in the 
            netCDF dataset.
        h : float, default value is 0.01
            Fraction of the water depth (--) to use as the length-scale in a 
            finite-difference approximation to the density gradient.
        
        Returns
        -------
        N : float
            The buoyancy frequency (1/s).
        
        Raises
        ------
        ValueError : 
            The input value of `z` must be between the `valid_min` and 
            `valid_max` values of the depth variable of the netCDF dataset.  
            If it is outside this range, a `ValueError` is raised.
        ValueError : 
            The parameter `h` must be between zero and one; otherwise, the 
            length-scale of the finite difference approximation will be 
            greater than the water depth.  If `h` is outside this range, a 
            `ValueError` is raised.
        
        Notes
        -----
        Uses the interpolation function to extract data from the profile; 
        therefore, it may already be smoothed by `coarsen` through the 
        `self.err` value, and the computed values are generally not taken 
        directly from measured data, but rather at interpolation points.
        
        """
        # Check validity of the input z values
        if np.max(z) > self.z_max or np.min(z) < self.z_min:
            raise ValueError('Selected depths outside range of ' + \
                             'CTD data:  %g to %g m' % (self.z_min, 
                             self.z_max))
        if h > 1.0 or h < 0.0:
            raise ValueError('Input parameter h must be between 0.0 ' + \
                             'and 1.0.')
        
        # Prepare space to store the solution
        if not isinstance(z, np.ndarray):
            N = np.zeros(1)
            z = np.array([z])
            elements = 0
        else:
            N = np.zeros(z.shape)
            elements = range(len(N))
        
        # Compute the length-scale for the finite difference method
        dz = (self.z_max - self.z_min) * h
        
        # Fill the solution matrix
        for i in range(len(z)):
            # Get the end-points of z for the finite difference formula
            if z[i] + dz > self.z_max:
                z1 = self.z_max
                z0 = z1 - 2 * dz
            elif z[i] - dz < self.z_min:
                z0 = self.z_min
                z1 = z0 + 2 * dz
            else:
                z0 = z[i] - dz
                z1 = z[i] + dz
            
            # Get the density at z0 and z1.  Use atmospheric pressure to 
            # compute the potential density and remove the effect of 
            # compressibility
            T0, S0 = self.get_values(z0, self.ztsp[1:3])
            T1, S1 = self.get_values(z1, self.ztsp[1:3])
            Pa = 101325.
            rho_0 = seawater.density(T0, S0, Pa)
            rho_1 = seawater.density(T1, S1, Pa)
            N[i] = np.sqrt(9.81 / rho_0 * (rho_1 - rho_0) / (z1 - z0))
        
        return N[elements]
    
    def extend_profile_deeper(self, z_new, nc_name, h_N=1.0, h=0.01, N=None):
        """
        Extend the CTD profile to the depth `z_new` using a fixed buoyancy
        frequency
        
        Extends the depth of a CTD profile to the new depth `z_new` using a 
        fixed buoyancy frequency to adjust the salinity and keeping all other
        variables constant below the original depth.  This should only be 
        used when no ambient data are available and when the bottom of the 
        original profile is deep enough that it is acceptable to assume all 
        variables remain constant in the extention except for salinity, which 
        is increased to maintain the desired buoyancy frequency.
        
        The fixed buoyancy frequency used to extend the profile can be 
        specified in one of two ways:
        
        1. Specify `h_N` and `h` and let the method `buoyancy_frequency` 
           evaluate the buoyancy frequency.  The evaluation depth is taken at
           the fraction `h_N` of the original CTD depth range, with 
           `h_N` = 1.0 yielding the lowest depth in the original CTD profile.
           `h` is passed to `'buoyancy_frequency` unchanged and sets the 
           length-scale of the finite difference approximation.
        2. Specify `N` directly as a constant.  In this case, the method
           `buoyancy_frequency` is not called, and any values passed to 
           `h_N` or `h` are ignored.
        
        Parameters
        ----------
        z_new : float
            New depth for the valid_max of the CTD profile
        nc_name : string or path
            Name to use when creating the new netCDF dataset with the deeper
            data generated by this method call.
        h_N : float, default 1.0
            Fraction of the water depth (--) at which to compute the buoyancy
            frequency used to extend the profile
        h : float, default 0.01
            Passed to the `buoyancy_frequency()` method.  Fraction of the 
            water depth (--) to use as the length-scale in a finite-
            difference approximation to the density gradient.
        N : float, default is None
            Optional replacement of `h_N` and `h`, forcing the extension 
            method to use `N` as the buoyancy frequency.  
        
        Notes
        -----
        This method does not explicitely return a value; however, it does
        create a new netCDF dataset with a deeper depth, closes the original 
        netCDF dataset in the object, and rebuilds the necessary object 
        attributes to be consistent with the new netCDF dataset.  The new 
        netCDF dataset filename is provided by `nc_name`.
        
        One may extend a profile any number of ways; this method merely 
        provides easy access to one rational method.  If other methods are 
        desired, it is recommended to create a new netCDF dataset with the 
        desired deeper profile data and then use that dataset to initialize 
        a new `Profile` object.
        
        See Also
        --------
        buoyancy_frequency
        
        """
        # Get the buoyancy frequency if not already specified
        if N is None:
            z = self.z_min + h_N * (self.z_max - self.z_min)
            N = self.buoyancy_frequency(z, h)
        
        # Extract the conditions at the bottom of the initial profile and 
        # calculate the required density at the base of the new profile 
        # using the potential density.
        T, S = self.get_values(self.z_max, self.ztsp[1:3])
        Pa = 101325.
        rho_0 = seawater.density(T, S, Pa)
        rho_1 = N**2 * rho_0 / 9.81 * (z_new - self.z_max) + rho_0
        
        # Find the salinity necessary to achieve rho_1 at the new depth
        def residual(S):
            """
            Compute the optimization function for finding S at z_new
            
            Keeping the temperature constant, compute the salinity `S` needed
            to achieve the desired potential density at the bottom of the 
            new profile, rho_1
            
            Parameters
            ----------
            S : float
                Current guess for the new salinity at the base of the extended
                CTD profile
            
            T, S, Pa, and rho_1 inherited from the above calculations
            
            Returns
            -------
            delta_rho : float
                Difference between the desired density `rho_1` at the base of 
                the new profile and the current estimate of `rho` using the 
                current guess for the salinity `S`.  
            
            Notes
            -----
            Because compressibility effects should be ignored in estimating 
            the buoyancy frequency, the pressure `Pa` is used to yield the 
            potential density.
            
            """
            rho = seawater.density(T, S, Pa)
            return (rho_1 - rho)
        S = fsolve(residual, S)
        
        # Create an array of data to append at the bottom of the original
        # profild data
        z_0 = self.z_max
        S_0 = self.y[-1,1]
        z_1 = z_new
        S_1 = S
        dz = (z_1 - z_0) / 50.
        z_new = np.arange(z_0, z_1+dz, dz)
        z_new[-1] = z_1
        y_new = np.zeros((z_new.shape[0], self.y.shape[1]))
        y_new[:,0] = self.y[-1,0]
        y_new[:,1] = (S_1 - S_0) / (z_1 - z_0) * (z_new - z_0) + S_0
        y_new[:,2:] = self.y[-1,2:]
        # Get the right pressure
        for i in range(len(z_new)-1):
            y_new[i+1,2] = y_new[i,2] + seawater.density(y_new[i,0], 
                           y_new[i,1], y_new[i,2]) * 9.81 * (z_new[i+1] - 
                           z_new[i])
        
        if self.nc_open is True:
            # Get the netCDF attributes not already stored in self.z and 
            # self.y
            summary = self.nc.summary
            source = self.nc.source
            sea_name = self.nc.sea_name
            p_lat = self.nc.variables['lat'][:]
            p_lon = self.nc.variables['lon'][:]
            p_time = self.nc.variables['time'][:]
            self.nc.close()
            
            # Create the new netCDF file.
            extention_text = ': extended from %g to %g' % (self.z_max, z_1)
            source = source + extention_text + ' on date ' + ctime()
            self.nc = create_nc_db(nc_name, summary, source, sea_name, p_lat, 
                                   p_lon, p_time)
            
            # Fill the netCDF file with the extended profile.
            self.y = np.vstack((self.y, y_new[1:,:]))
            self.z = np.hstack((self.z, z_new[1:]))
            data = np.hstack((np.atleast_2d(self.z).transpose(), self.y))
            var_symbols = [self.ztsp[0]] + self.f_names
            var_units = ['m'] + self.get_units(self.f_names)
            comments = ['extended'] * len(var_symbols)
            self.nc = fill_nc_db(self.nc, data, var_symbols, var_units, 
                                 comments, z_col=0)
            
            # Update the interpolator.
            self.build_interpolator()
            
        else:
            raise RuntimeError('The netCDF dataset is already closed; ' + 
                               'aborting extension.')
    
    def close_nc(self):
        """
        Close the netCDF dataset
        
        Because the raw profile data are stored in a netCDF file, a pipe is 
        open to the file source whenever this attribute of the object is in 
        use.  Once the interpolation function is built with the complete 
        dataset needed for a TAMOC simulation (e.g., once calls to `append`
        are not longer needed), the netCDF dataset does not need to remain 
        available.  This method closes the netCDF dataset file and tells
        the `Profile` object that the file is closed.  All methods that do not
        directly read or write data to the netCDF file will continue to work.
        If the object is no longer used, the Python garbage collector will
        eventually find it and remove it from memory.        
        
        Notes
        -----
        Especially during code development, scrips with open netCDF files can
        crash, and this method will not be called.  In that case, it is
        likely necessary to close Python and restart a new session so that 
        all file pointers can be flushed.  
        
        """
        # Close the netCDF file
        if self.nc_open is True:
            self.nc.close()
        
        # Remove the object attributes from memory and delete the object
        self.nc_open = False
    

def create_nc_db(nc_file, summary, source, sea_name, p_lat, p_lon, p_time):
    """
    Create an empty netCDF4-classic dataset to store ambient data
    
    Creates an netCDF dataset file in netCDF4-classic format for use in 
    storing ambient profile data.  This function creates the minimum data
    structure necessary to run the TAMOC simulation models.  This includes 
    temperature, salinity, and pressure as a function of depth, as well as
    the auxiliary metadata, such as latitude, longitude, and time, etc.
    
    Parameters
    ----------
    nc_file : file path or string
        File name (supplied as a path or string) to use to create the netCDF
        database
    summary : string
        String containing the text for the dataset global variable `summary`.
        This should describe the TAMOC simulation project for which the 
        dataset was created.
    source : string
        String containing the text for the dataset global variable `source`.
        Use this variable to document the original source of the data (e.g., 
        station BM54 recorded by R/V Brooks McCall).
    sea_name : string
        String containing the NODC Sea Name.  If unsure, consult the list 
        at http://www.nodc.noaa.gov/General/NODC-Archive/seanamelist.txt
    p_lat : float
        Latitude of the CTD profile location in decimal degrees from North
    p_lon : float
        Longitude of the CTD profile location in decimal degrees from East
    p_time : 
        Time the profile was taken in seconds since 1970-01-01 00:00:00 0:00
        following a julian calendar.  See Examples section for how to create
        this variable using Python's `datetime` module and the netCDF4 modules
        `netCDF4.date2num` and `netCDF4.num2date`.
    
    Returns
    -------
    nc : `netCDF4.Dataset` object
        An object that contains the empty database ready to input profile data
    
    Notes
    -----
    This function creates an empty netCDF dataset that conforms to the 
    standard format for an Orthogonal Multidimensional Array Representation 
    of Profiles as defined by the National Oceanographic Data Center (NODC).  
    See http://www.nodc.noaa.gov/data/formats/netcdf for more details.
    
    NODC recommends using a netCDF4-classic data format, which does not 
    support groups.  This works well for this dataset, so we implement this
    data format here.  The documentation for the Python `netCDF4` package
    here https://code.google.com/p/netcdf4-python/ uses groups throughout 
    the tutorial.  The classic format is equivalent to having the single 
    group `root_grp`.  
    
    See Also
    --------
    netCDF4 
    
    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> from netCDF4 import num2date, date2num
    >>> nc_file = './test/output/test.nc'
    >>> summary = 'Test file'
    >>> source = 'None'
    >>> sea_name = 'No Sea Name'
    >>> p_lat = 28.5
    >>> p_lon = 270.7
    >>> p_time = date2num(datetime(2010, 5, 30), 
            units = 'seconds since 1970-01-01 00:00:00 0:00', 
            calendar = 'julian')
    >>> nc = create_nc_db(nc_file, summary, source, sea_name, p_lat, 
            p_lon, p_time)
    >>> print nc.variables
    OrderedDict([('time', <netCDF4.Variable object at 0x76f2978>), 
                 ('lat', <netCDF4.Variable object at 0x76f2a98>), 
                 ('lon', <netCDF4.Variable object at 0x76f2ae0>), 
                 ('z', <netCDF4.Variable object at 0x76f2b28>), 
                 ('temperature', <netCDF4.Variable object at 0x76f2b70>), 
                 ('salinity', <netCDF4.Variable object at 0x76f2bb8>), 
                 ('pressure', <netCDF4.Variable object at 0x76f2c00>)])
    >>> print num2date(nc.variables['time'][0], 
            units = 'seconds since 1970-01-01 00:00:00 0:00', 
            calendar = 'julian')
    2010-05-30 00:00:00
    >>> z_data = np.linspace(0., 1500., 30.)
    >>> nc.variables['z'][:] = z_data
    >>> nc.variables['z'].valid_max = np.max(z_data)
    >>> print nc.variables['z']
    float64 z(u'z',)
        long_name: depth below the water surface
        standard_name: depth
        units: m
        axis: Z
        positive: down
        valid_min: 0.0
        valid_max: 1500.0
    unlimited dimensions = (u'z',)
    current size = (30,)
    >>> nc.close()
    
    """
    from tamoc import model_share
    # Create the netCDF dataset object
    title = 'Profile created by TAMOC.ambient for use in the TAMOC ' + \
            'modeling suite'
    nc = model_share.tamoc_nc_file(nc_file, title, summary, source)
    nc.sea_name = sea_name
    
    # Create variables for the dimensions
    z = nc.createDimension('z', None)
    p = nc.createDimension('profile', 1)
    
    # Create the time variable
    time = nc.createVariable('time', 'f8', ('profile',))
    time.long_name = 'Time profile was collected'
    time.standard_name = 'time'
    time.units = 'seconds since 1970-01-01 00:00:00 0:00'
    time.calendar = 'julian'
    time.axis = 'T'
    time[0] = p_time
    
    # Create variables for latitude and longitude
    lat = nc.createVariable('lat', 'f8', ('profile',))
    lat.long_name = 'Latitude of the profile location'
    lat.standard_name = 'latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'
    lat[0] = p_lat
    
    lon = nc.createVariable('lon', 'f8', ('profile',))
    lon.long_name = 'Longitude of the profile location'
    lon.standard_name = 'longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'
    lon[0] = p_lon
    
    # Create the depth variable
    z = nc.createVariable('z', 'f8', ('z',))
    z.long_name = 'depth below the water surface'
    z.standard_name = 'depth'
    z.units = 'm'
    z.axis = 'Z'
    z.positive = 'down'
    z.valid_min = 0.0
    z.valid_max = 0.0
    
    # Create variables for temperature, salinity, and pressure
    T = nc.createVariable('temperature', 'f8', ('z',))
    T.long_name = 'Absolute temperature'
    T.standard_name = 'temperature'
    T.units = 'K'
    T.coordinates = 'time lat lon z'
    
    S = nc.createVariable('salinity', 'f8', ('z',))
    S.long_name = 'Practical salinity'
    S.standard_name = 'salinity'
    S.units = 'psu'
    S.coordinates = 'time lat lon z'
    
    P = nc.createVariable('pressure', 'f8', ('z',))
    P.long_name = 'pressure'
    P.standard_name = 'pressure'
    P.units = 'Pa'
    P.coordinates = 'time lat lon z'
    
    return nc


def fill_nc_db(nc, data, var_symbols, var_units, comments, z_col=0):
    """
    Add data to a netCDF4-classic ambient profile dataset
    
    This function adds data to a netCDF4-classic dataset for a single CTD
    profile.  It is expected that this function could be called multiple 
    times to completely fill in a profile database.  As data are added, this 
    method interpolates the new data to match the current z-coordinates if 
    they are already present in the dataset.  
    
    Parameters
    ----------
    nc : netCDF4 dataset object
        This is the existing netCDF dataset object that will receive the data
        stored in `data`.
    data : ndarray
        Table of data to add to the netCDF database.  If it contains more
        than one variable, the data are assumed to be arranged in columns.
    var_symbols : string list
        List of string symbol names (e.g., T, S, P, etc.) in the same order as 
        the columns in the data array.  For chemical properties, use the 
        key name in the `chemical_properties` database.
    var_units : string list
        List of units associated with each variable in the `var_symbols` list.
    comments : string list
        List of comments associated with each variable in the `var_symbols`
        list.  As a minimum, this list should include the indications 
        'measured' or 'derived' or some similar indication of the source of
        the data.
    z_col : integer, default is 0
        Column number of the column containing the depth data.  The first 
        column is numbered zero.
    
    Returns
    -------
    nc : `netCDF4.Dataset` object
        Returns the updated netCDF4 dataset with the data and metadata 
        included.
    
    Raises
    ------
    ValueError : 
        The input data array must always include a column of depths.  If the
        input array contains a single column and the netCDF database already
        has a depth array, a `ValueError` is raised since it would appear the 
        user is trying to replace the existing depth data in the netCDF 
        database.  If such an action is required, build a new netCDF database 
        from scratch with the correct depth data.
    
    ValueError : 
        This function checks whether the units supplied by the user in the 
        list `var_units` match those expected by the database.  If not, a 
        `ValueError` is raised with a message indicating which units are 
        incompatible.
    
    Notes
    -----
    Symbol names in the ``TAMOC`` modeling suite are `z` (depth, positive down
    from the sea surface), `temperature`, `salinity`, `pressure`, and chemical
    names from the `chemical_properties` database (see 
    ``./data/ChemData.csv``).  Other names will be treated exactly like the 
    chemical names, but will likely either be unused by the ``TAMOC`` modeling
    suite or generate errors when a different symbol is expected.  Hence, it 
    is the responsibility of the user to ensure that all symbol names are 
    correct when this function is called.
    
    See Also
    --------
    create_nc_db
    
    Examples
    --------
    >>> ds = Dataset('./test/output/test_ds.nc', 'a')
    >>> z = ds.variables['z']
    >>> zp = np.array([z.valid_min, z.valid_max])  # Depth range
    >>> yp = np.array([9.15, 5.20]) / 1000         # Synthetic data
    >>> data = np.vstack((zp, yp)).transpose()
    >>> ds = fill_nc_db(ds, data, ['z', 'oxygen'], ['m', 'kg/m^3'], 
                        ['synthetic', 'synthetic'], z_col=0)
    >>> print ds.variables.keys()
    ['time', 'lat', 'lon', 'z', 'temperature', 'salinity', 'pressure', 'S', 
    'T', 'oxygen']
    >>> ds.variables['oxygen'][:].shape            # Note interpolation
    (34,)
    
    """
    # Ensure the correct data types were provided
    if isinstance(var_symbols, str):
        var_symbols = [var_symbols]
    if isinstance(var_units, str):
        var_units = [var_units]
    if isinstance(comments, str):
        comments = [comments]
    if isinstance(data, list):
        data = np.array(data)
    
    # Count the number of dependent variables in the data array
    ny = len(var_symbols) - 1
    
    # Handle the independent variable z
    z = nc.variables[var_symbols[z_col]]
    if z[:].shape[0] == 0:
        # Fill netCDF dataset with z values in data
        
        # Make sure the z data are in a column
        if ny == 0:
            data = np.atleast_2d(data)
            if data.shape[1] > 1:
                data = data.transpose()
        
        # Insert the data into the dataset
        nc = fill_nc_db_variable(nc, data[:,z_col], var_symbols[z_col], 
                                 var_units[z_col], comment=comments[z_col])
        z.valid_min = np.min(z[:])
        z.valid_max = np.max(z[:])
    
    else:
        # Use the existing z values in the netCDF dataset
        
        if ny == 0:    
            # User is trying to replace existing data; this is not allowed
            raise ValueError('Cannot replace existing depth data in ' + \
                'netCDF dataset: \n' + \
                'Provide multicolumn array of new data to interpolate ' + \
                'profile values \nonto the existing depths grid.')
        
        else:
            # Interpolate the input data to the existing z values
            
            # Extend the input data array to span the full range of z values
            # in the existing netCDF database.  Assume the data at z_min and
            # z_max of the input data array can be copied to the valid_min
            # and valid_max depths in the netCDF dataset if the input data
            # array extends over a subset of the existing netCDF z range.
            if min(data[:,z_col]) > z.valid_min:
                ctd = np.zeros((data.shape[0]+1, data.shape[1]))
                ctd[0,:] = data[0,:]
                ctd[0,z_col] = z.valid_min
                ctd[1:,:] = data
                data = np.copy(ctd)
            if max(data[:,z_col]) < z.valid_max:
                ctd = np.zeros((data.shape[0]+1, data.shape[1]))
                ctd[-1,:] = data[-1,:]
                ctd[-1,z_col] = z.valid_max
                ctd[0:-1,:] = data
                data = np.copy(ctd)
            
            # Create the interpolation function
            y_cols = np.array(range(ny+1)) != z_col
            f = interp1d(data[:,z_col], data[:,y_cols].transpose())
            
            # Replace the input data with the interpolated values that match
            # the netCDF z array
            interp_data = np.zeros((z.shape[0],ny+1))
            interp_data[:,z_col] = z[:]
            interp_data[:,y_cols] = f(z[:]).transpose()
            data = np.copy(interp_data)
    
    # Handle the dependent variables
    for i in range(ny+1):
        
        if i == z_col:
            # Skip the depth data (already handeled above)
            pass
        else:
            # Processes the dependent data depending on the variable type
            std_name = join(var_symbols[i].split('_'))
            long_name = capwords(std_name)
            nc = fill_nc_db_variable(nc, data[:,i], var_symbols[i], 
                                     var_units[i], comment=comments[i], 
                                     long_name=long_name, 
                                     std_name=std_name)
    return nc


def fill_nc_db_variable(nc, values, var_name, units, comment=None, 
                        long_name=None, std_name=None):
    """
    Copy data to a netCDF4-classic dataset variable.
    
    DO NOT CALL THIS FUNCTION DIRECTLY.  Instead, use `fill_nc_db`.
    
    This function is intended to only be called from fill_nc_db after it 
    determines where to insert an array of data into a netCDF4 dataset.  This
    function could be called directly by the user, but no checking will be 
    performed to ensure the inserted dataset is compatible with the other 
    data in the netCDF dataset.  Moreover, dimension mismatch, rugged arrays,
    or other errors could occur without the preprocessing in fill_nc_db.
    
    Parameters
    ----------
    nc : netCDF4 data object
        This is the existing netCDF dataset object that will receive the new
        variable values.
    values : ndarray, shape(z,)
        Array of variable values to add to the netCDF database.  Must have the
        same dimension as the current z values in the netCDF dataset.
    var_name : string
        Name to use in the netCDF dataset.  If it already exists, the 
        current variable space will be used; otherwise, a new variable will
        be created.
    units : string list
        Units of the input values
    comment : string, optional
        String containing comments for the comment attribute of the variable.
        For instance, 'Measured', 'Derived quantity', etc.
    long_name : string, optional
        String for the long_name attribute of the variable; only used if the
        variable does not already exist.
    std_name : string, optional
        String for the standard_name attribute of the variable; only used if
        the variable does not already exist.
    
    Returns
    -------
    nc : netCDF4 data object
        Returns the updated netCDF4 database with the data and metadata 
        included
    
    Raises
    ------
    ValueError : 
        If the variable already exists, then the input units are checked
        against the units already in the database.  If they do not match, 
        a ValueError and appriate message is raised.
    
    """
    if var_name in nc.variables:
        # Add data to an existing variable
        y = nc.variables[var_name]
        try:
            assert y.units == units
            y[:] = values
        except:
            raise ValueError('Error: %s units must be in %s not %s' % 
                             (var_name, y.units, units))
    
    else:
        # Create a new netCDF dataset variable
        z_dim = nc.dimensions.keys()[0]
        y = nc.createVariable(var_name, 'f8', (z_dim))
        y[:] = values
        y.long_name = long_name
        y.standard_name = std_name
        y.units = units
        y.coordinates = 'time lat lon z'
    
    # Insert comments
    if comment is not None:
        y.comment = comment
    
    return nc


def get_nc_db_from_roms(nc_roms, nc_file, t_idx, j_idx, i_idx, chem_names):
    """
    Create a netCDF4-classic dataset ready to use with ``TAMOC`` simulation 
    models from ROMS netCDF output.
    
    Some of the model variable names used in ROMS are different from those
    recommended for CTD data by NODC.  Moreover, ROMS output usually includes
    results at many horizonal grid locations; whereas, the ``TAMOC`` models
    expect to have profile data for a single (lat, lon) location.  This 
    function translates the ROMS output at a specified point to a netCDF 
    file in the format required by ``TAMOC``.
    
    Parameters
    ----------
    nc_roms : netCDF4 dataset object, file path or string
        File name (supplied as a path or string) to the ROMS output that will
        be used to create a ``TAMOC`` ambient profile.
    nc_file : file path or string
        File name (supplied as a path or string) to use to create the netCDF
        database.
    t_idx : integer
        time-index to the desired profile information (time steps)
    j_idx : integer
        j-index to the grid location where the profile should be extracted.
    i_idx : integer
        i-index to the grid location where the profile should be extracted.
    chem_names : string list
        string list of variables to extract in addition to depth, salinity, 
        temperature, and pressure.
    
    Returns
    -------
    nc : `netCDF4.Dataset` object
        An object that contains the new netCDF dataset object/
    
    Notes
    -----
    The author of this module (S. Socolofsky) is not a ROMS expert.  Variable
    names assumed present in ROMS were those in the netCDF file extracted from
    http://barataria.tamu.edu:8080/.  
    
    TODO (S. Socolofsky 7/12/2013): Determine whether other ROMS simulations
    will have different variable names.  If so, provide inputs to this 
    function containing an appropriate mapping of names.
    
    TODO (S. Socolofsky 7/15/2013): The depth values returned here are not 
    monotonically increasing.  Make sure you are using the updated octant
    modules.  
    
    See Also
    --------
    netCDF4, octant, octant.roms
    
    Examples
    --------
    >>> nc_roms = 'http://barataria.tamu.edu:8080/thredds/dodsC/' + \\
                  'ROMS_Daily/08122012/ocean_his_08122012_24.nc'
    >>> nc_file = './test/output/test_roms.nc'
    >>> t_idx = 0
    >>> j_idx = 400
    >>> i_idx = 420
    >>> chem_names = ['dye_01', 'dye_02']
    
    """
    import octant.roms
    
    # Open the ROMS netCDF dataset and get global attributes
    if isinstance(nc_roms, str) or isinstance(nc_roms, unicode):
        source = nc_roms
        nc_roms = Dataset(nc_roms)
    else:
        source = nc_roms.title
    summary = 'ROMS Simulation Data'
    sea_name = 'ROMS'
    
    # Get the depths grid for concentration data
    zr = octant.roms.nc_depths(nc_roms, grid='rho')
    z_unit = zr.zeta.units
    
    # Extract the depth, temperature and salinity profiles
    z = zr[0][:, j_idx, i_idx]
    z, z_unit = convert_units(z, z_unit)
    
    T = nc_roms.variables['temp'][t_idx, :, j_idx, i_idx]
    T_unit = nc_roms.variables['temp'].units
    T, T_unit = convert_units(T, T_unit)
    
    S = nc_roms.variables['salt'][t_idx, :, j_idx, i_idx]
    S_unit = nc_roms.variables['salt'].long_name
    S, S_unit = convert_units(S, S_unit)
    
    # Compute the pressure by integrating the density
    P = compute_pressure(z, T, S, -1)
    P_unit = ['Pa']
    
    # Extract any other desired datasets
    c = np.zeros((len(chem_names), z.shape[0]))
    c_units = []
    for i in range(len(chem_names)):
        c[i,:] = nc_roms.variables[chem_names[i]][t_idx, :, j_idx, i_idx]
        c_units.append(nc_roms.variables[chem_names[i]].units)
    c, c_units = convert_units(c.transpose(), c_units)
    c = c.transpose()
    
    # Get the time and location
    date = nc_roms.variables['ocean_time'][t_idx]
    date_units = nc_roms.variables['ocean_time'].units
    date_cal = nc_roms.variables['ocean_time'].calendar
    date = num2date(date, units = date_units, calendar = date_cal)
    p_time = date2num(date, units = 'seconds since 1970-01-01 00:00:00 0:00', 
        calendar = 'julian')
    
    p_lat = nc_roms.variables['lat_rho'][j_idx, i_idx]
    if nc_roms.variables['lat_rho'].units != 'degree_north':
        p_lat = 180.0 - p_lat
    p_lon = nc_roms.variables['lon_rho'][j_idx, i_idx]
    if nc_roms.variables['lon_rho'].units != 'degree_east':
        p_lon = 360.0 - p_lon
    
    # Initialize an empty nc database 
    nc_tamoc = create_nc_db(nc_file, summary, source, sea_name, p_lat, p_lon, 
                            p_time)
    
    # Stack the data and sort so that depth increases, positive down from the
    # free surface
    data = np.vstack((z, T, S, P, c)).transpose()
    data[:,0] = -data[:,0]
    data = np.flipud(data)
    
    # Extend the dataset to the free surface at z = 0, P = 101325 Pa
    data = np.vstack((data[0,:], data))
    data[0,0] = 0.0
    data[0,3] = 101325.
    
    # Fill the database with the ROMS data
    var_symbols = ['z', 'temperature', 'salinity', 'pressure'] + chem_names
    var_units = ['m', 'K', 'psu', 'Pa'] + c_units
    comments = ['ROMS Data'] * len(var_symbols)
    nc_tamoc = fill_nc_db(nc_tamoc, data, var_symbols, var_units, comments, 
                          z_col=0)
    
    # Return the completed netCDF dataset
    return (nc_tamoc, nc_roms)


def extract_profile(data, z_col=0, z_start=50, p_col=None, P_atm=101325.):
    """
    Function to extract a CTD profile with monotonically increasing depth
    
    This function scans a complete CTD profile data array looking for 
    direction reversals at the top and bottom of the profile.  It then removes
    all reversals, yielding a single profile with monotonically increasing 
    depth.  This is particularly useful for CTD datasets that include both the
    up and down cast or that have not been preprocessed to remove the surface 
    entry and swaying at the top and bottom of the profile.
    
    Parameters
    ----------
    data : ndarray
        Contains the complete CTD dataset in `numpy.array` format.  All 
        columns will be preserved; only the depth column will be used to make 
        decisions.
    z_col : integer, default is 0
        Column number of the column containing the depth data.  The first 
        column is numbered zero.
    z_start : float, default is 50
        Depth over which reversals are considered to be at the top of the 
        profile.  If a depth reversal is found below this value, the profile
        will be assumed to end there.  The top of the profile will be either
        the first row of data or the lowest row of data containing a reversal
        for which `z` < `z_start`.
    p_col : integer, default is None
        Column number of the column containing the pressure data.  If the
        profile is artificially extended to the free surface, the pressure
        must approach atmospheric pressure.
    P_amt : float, default is 101325
        Value for atmospheric pressure.  This function does not do any unit
        conversion, so if the pressure units passed to this function are not
        Pa or the pressure is different than standard atmospheric pressure, 
        then the correct value should be specified.
    
    Notes
    -----
    If the start of the profile is found to occur below `z` = 0, then a row 
    will be added to the top of the profile with depth `z` = 0 and all other 
    values equal to their value on the next row of the profile.  This is 
    generally needed by interpolation methods in the ``TAMOC`` simulation 
    suite that require data throughout the water column.
    
    This function is for use in creating a CTD data array before it is 
    added to a netCDF dataset.  Once the depths have been added to a netCDF 
    dataset, the methods defined in this module do not allow the depth to be
    further changed.
    
    """
    # Initialize counters for the start and end of the profile
    start = 0
    end = data.shape[0] - 1
    
    # Search for the start of the profile over the range z < z_start
    i = 1
    while data[i,z_col] < z_start and i <= end:
        if data[i,z_col] < data[i-1,z_col]:
            # Profile is reversing
            start = i
        i += 1
    
    # Search for the end of the profile
    while i < end:
        if data[i,z_col] < data[i-1,z_col]:
            # Profile is reversing
            end = i
        i += 1
    
    # Extend the profile to the free surface if necessary
    if data[start,z_col] > 0.:
        ctd = np.zeros((end-start+2, data.shape[1]))
        ctd[0,:] = data[start,:]
        ctd[0,z_col] = 0.0
        if p_col is not None:
            ctd[0,p_col] = P_atm
        ctd[1:,:] = data[start:end+1]
    else:
        ctd = np.zeros((end-start+1, data.shape[1]))
        ctd = data[start:end+1,:]
    
    # Return the single CTD profile
    return ctd


def coarsen(raw, err = 0.01):
    """
    Reduce the size of a raw database for interpolation
    
    Removes rows from the raw input database so that linear interpolation 
    between rows in the new dataset recovers the original data within a 
    relative error given by `err`.  
    
    Parameters
    ----------
    raw : ndarray, shape(:,:)
        An array of data with the independent variable (usually depth) 
        in the first column and the dependent variable(s) in the remaining
        columns.  Note that the first column is always ignored.  
    err : float
        The acceptable level of relative error for linear interpolation 
        between rows in the output database
    
    Returns
    -------
    data : ndarray, shape(:,:)
        An array of data in the same organization as the raw input array, but
        generally with rows removed so that the interpolation error between
        the output data set and the raw input data are within a relative 
        error specified by err.
    
    Examples
    --------
    >>> raw = np.zeros((100,3))
    >>> raw[:,0] = np.arange(100)     # Vector of dependent variables
    >>> raw[:,1] = np.sqrt(raw[:,0])  # Fictitious dependent variable 1
    >>> raw[:,2] = raw[:,0]**2        # Fictitious dependent variable 2
    >>> data = coarsen(raw, 0.5)      # Allow up to 50% error
    >>> data.shape
    (13, 3)
    >>> data[:,0]
    array([  0.,   1.,   2.,   3.,   5.,   8.,  12.,  17.,  25.,  36.,  51.,
            73.,  99.])               # Note: data are resolved in areas
                                      # with the greatest curvature (small z)
    
    """
    # Set up a blank database equal in size to the input data
    data = np.zeros(raw.shape)
    
    # Record the first data point
    data[0,:] = raw[0,:]
    
    # Loop through the remaining data and only record a data point if err will 
    # otherwise be exceeded for any column of dependent-variable data.
    j = 0
    i_0 = 0
    for i in range(1, raw.shape[0]-1):
        # Loop through each row of data
        rec = False
        for k in range(1, raw.shape[1]):
            # Check each column for an exceedance of the err criteria
            if raw[i,k] != 0.:
                ea = np.abs((raw[i,k] - raw[i_0,k]) / raw[i,k])
            else:
                ea = 0.
            if ea > err:
                # Error exceeded for this column; record this row of data
                rec = True
        if rec:
            # Need to record this row
            j += 1
            data[j,:] = raw[i,:]
            # Reset the baseline row for the error calculation
            i_0 = i
    
    # Record the last data point
    j += 1
    data[j,:] = raw[-1,:]
    
    # Remove all blank rows from database
    data = data[0:j+1,:]
    
    # Return the reduced dataset
    return data


def stabilize(raw):
    """
    Force the density profile to be stable or neutral
    
    Remove all reversals in the density profile so that it is monotonically
    increasing from the surface.  This function is based on the potential
    density so that we obtain an absolutely stable or neutral profile.
    
    Parameters
    ----------
    raw : ndarray
        An array of data organized the depth, temperature, salinity, and 
        pressure in the first four columns of the matrix.
    
    Returns
    -------
    data : ndarray, shape(:,:)
        A new array of data with the rows removed that would have produced an 
        unstable profile.
    
    See Also
    --------
    Profile.build_interpolator
    
    Notes
    -----
    This function would not normally be called directly, but rather is used
    by the `Profile.build_interpolator` method to ensure that the dataset
    used in the interpolator is neutrally stable.
    
    """
    # Start by assuming all rows with depth equal to or greater than zero 
    # should be returned.
    rows = raw[:,0] >= 0.
    
    # Potential density is computed at atmospheric pressure here
    Pa = 101325.
    
    # Check the density gradient to each adjacent row and remove the unstable
    # rows
    rho_old = seawater.density(raw[0,1], raw[0,2], Pa)
    for i in range(1, raw.shape[0]-1):
        rho = seawater.density(raw[i,1], raw[i,2], Pa)
        if rho < rho_old:
            rows[i] = False
        else:
            rho_old = copy(rho)
    
    # Build an interpolator that contains the stable rows
    f = interp1d(raw[rows,0], raw[rows,1:].transpose())
    
    # Fill the T, S, and P variables of raw with the stabilized data while 
    # keeping the variability of all the other data on the original grid
    for i in range(len(raw[:,0])):
        raw[i,1:3] = f(raw[i,0])[0:2]
    
    # Return the acceptable rows...change this to return all rows if you 
    # do not like how much data is removed.
    return raw[rows,:]


def compute_pressure(z, T, S, fs_loc):
    """
    Compute the pressure profile by integrating the density
    
    Compute the pressure as a function of depth by integrating the density, 
    given by the temperature and salinity profiles and the seawater equation 
    of state in `seawater.density`.  The depth coordinate can be either 
    positive or negative, and the free surface can be located either in the 
    first index to the `z`-array (`fs_loc` = 0) or the last row of the 
    `z`-array (`fs_loc` = -1).  The data are returned in the same order as the 
    input.
    
    Parameters
    ----------
    z : ndarray
        Array of depths in meters.
    T : ndarray
        Array of temperatures (K) at the corresponding depths in `z`.
    S : ndarray
        Array of salinities (psu) at the corresponding depth in `z`.
    fs_loc : integer (0 or -1)
        Index to the location of the free-surface in the `z`-array.  0 
        corresponds to the first element of `z`, -1 corresponds to the last 
        element.
    
    Returns
    -------
    P : ndarray
        Array of pressures (Pa) at the corresponding depth in `z`.
    
    Notes
    -----
    TAMOC requires the pressure as an input since seawater is compressible 
    over modest ocean depths.  In order to avoid having to integrate the 
    pressure over the depth anytime the density is needed, ``TAMOC`` expects 
    the pressures to have been computed *a priori*.  
    
    Examples
    --------
    >>> z = np.arange(-1500, 0, 10)
    >>> T = np.linspace(4.1, 25.0, len(z)+1)
    >>> S = np.linspace(36.5, 34.5, len(z)+1)
    >>> fs_loc = -1
    >>> P = compute_pressure(z, T, S, fs_loc)
    >>> z[-1]
    -10
    >>> P[-1]
    1558721.446785233
    >>> z[0]
    -1500
    >>> P[0]
    150155213.18007597
    
    """
    # Get the sign of the z-data for the midpoint of the dataset
    z_sign = int(np.sign(z[len(z) / 2]))
    
    # Initialize an array for storing the pressures
    P0 = 101325.0 
    g = 9.81
    P = np.zeros(z.shape)
    
    # Find the free surface in the z-data
    if fs_loc == -1:
        depth_idxs = range(len(z)-2, -1, -1)
        idx_0 = len(z) - 1
    else:
        depth_idxs = range(1, len(z))
        idx_0 = 0
    
    # Compute the pressure at the free surface
    P[idx_0] = P0 + seawater.density(T[0], S[0], P0) * g * z_sign * z[idx_0]
    
    # Compute the pressure at the remaining depths
    for i in depth_idxs:
        P[i] = P[i-z_sign] + seawater.density(T[i-z_sign], S[i-z_sign], 
               P[i-z_sign]) * g * (z[i] - z[i-z_sign]) * z_sign
    
    return P


def convert_units(data, units):
    """
    Convert the values in data to standard units
    
    This function accepts a data array with variables arranged in columns, 
    each column given by the sequential unit in units, and converts the 
    values in the array to standard units.  The function returns both the 
    transformed data and the new, standardized units.
    
    Parameters
    ----------
    data : int, float, list, or ndarray
        Data in which the values in each column are of uniform 
        units.
    units : string list
        A list of strings stating the units of the input data array.  The 
        squence of strings in the list must match the units of each column
        of data
    
    Returns
    -------
    A tuple containing:
    
    data : ndarray
        Array of data in the same order as the input array, but converted to
        standard units.
    units : string list
        A list of strings stating the new, standardized units of the values in 
        data.
    
    Notes
    -----
    This function assumes that for a one-dimensional array, each element has 
    a unique unit.  That is, one-dimensional arrays are assumed to be rows
    with the number of columns equal to the number of values.  If a single 
    column of data must be converted, you may pass
    `numpy.atleast_2d(data).transpose()`.  
    
    Examples
    --------
    >>> data = np.array([[10, 25.4, 9.5, 34], [100, 10.7, 8.4, 34.5]])
    >>> units = ['m', 'deg C', 'mg/l', 'psu']
    >>> data, units = convert_units(data, units)
    >>> data
    array([[  1.00000000e+01,   2.98550000e+02,   9.50000000e-03,
              3.40000000e+01],
           [  1.00000000e+02,   2.83850000e+02,   8.40000000e-03,
              3.45000000e+01]])
    >>> units
    ['m', 'K', 'kg/m^3', 'psu']
    
    >>> data = 10
    >>> units = 'deg C'
    >>> data, units = convert_units(data, units)
    >>> data
    array([283])
    >>> units
    ['K']
        
    """
    # Build the dictionary of units conversions
    convert = {'m' : [1.0, 0., 'm'], 
               'meter' : [1.0, 0., 'm'], 
               'deg C' : [1.0, 273.15, 'K'], 
               'Celsius' : [1.0, 273.15, 'K'], 
               'K' : [1.0, 0., 'K'],
               'db' : [1.e4, 101325., 'Pa'], 
               'Pa' : [1.0, 0., 'Pa'],
               'mg/m^3': [1.e-6, 0., 'kg/m^3'], 
               'S/m': [1.0, 0., 'S/m'], 
               'psu': [1.0, 0., 'psu'], 
               'salinity': [1.0, 0., 'psu'], 
               'kg/m^3': [1.0, 0., 'kg/m^3'], 
               'kilogram meter-3': [1.0, 0., 'kg/m^3'], 
               'm/s': [1.0, 0., 'm/s'], 
               'mg/l': [1.e-3, 0., 'kg/m^3']}
    
    # Make sure the data are a numpy array and the units are a list
    if isinstance(data, float) or isinstance(data,int):
        data = np.array([data])
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(units, str):
        units = [units]
    if isinstance(units, unicode):
        units = [units]
    
    # Make sure you can slice through the columns:  must be two-dimensional
    sh = data.shape
    data = np.atleast_2d(data)
    
    # Allow conversion of a row of data if all of the same unit
    if len(units) == 1 and data.shape[1] > 1:
        data = data.transpose()
    
    # Convert the units
    for i in range(len(units)):
        try:
            data[:,i] = data[:,i] * convert[units[i]][0] + \
                        convert[units[i]][1]
            units[i] = convert[units[i]][2]
        except KeyError:
            print 'Do not know how to convert %s to mks units' % units[i]
            print 'Continuing without converting these units...'
    
    # Return the converted data in the original shape
    data = np.reshape(data, sh, 'C')
    return (data, units)


def load_raw(fname):
    """
    Read data from a text file.
    
    Read all of the data in a text file `fname` with `*` or `#` as comment
    characters.  Data are assumed to be sparated by spaces.  
    
    Parameters
    ----------
    fname : string
        Name of file containing the data.
    
    Returns
    -------
    ctd : ndarray
        Array of data contained in `fname`.
    
    See Also
    --------
    numpy.fromfile : Read data from a simple text file in single table format.
        Similar to the Matlab `load` function.  Does not support comments or 
        skipped lines.  Works for text or binary files.
    
    numpy.loadtxt : A more sophisticated method to read data from a text file.
        This method can handle multiple data types (e.g., strings, floats, and
        integers together), can perform transformation on the data (e.g., date
        format to a date object), and can unpack to a tuple of variables.  One
        limitation is that all rows must have the same number of columns and 
        there is no method to handle missing data.
    
    numpy.genfromtext : The most advanced numpy method to read data from a 
        file. Includes the capabilities in `np.loadtxt`, but also allows for 
        missing data, data flags, and multiple methods to replace missing data 
        or flags.
    
    """
    # Read all the data from the file
    ctd = []
    with open(fname) as ctdfile:
        
        for line in ctdfile:
            
            if (line.find('*') < 0) and (line.find('#') < 0):
                
                # This line contains data; parse the line
                entries = line.strip().split()
                # Convert data to float64
                entries = [np.float64(entries[i]) 
                           for i in range(len(entries))]
                # Append to list
                ctd.append(entries)
    
    # Return the raw data as an numpy array
    return np.array(ctd)

