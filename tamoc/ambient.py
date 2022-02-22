"""
Ambient Module
==============

Define functions, classes, and methods to handle ambient seawater data

This module provides tools to read in ambient data files (e.g., CTD profiles)
in several different formats and to contain the data in a way that makes it
usable by the other modules in ``tamoc``. Tools are available to manipulate
data to extract profiles with monotonic depth coordinate values, remove
density reversals, thin the data while preserving a given error level in
interpolation, and exchange the data through different formats.

The main purpose of the ``ambient`` module is to provide an interpolator that
can be queried for an arbitrary parameter in the profile database at a given
depth and return the corresponding values.

The primary tool in this module is the ``Profile`` class, which creates the
user API for interacting with the profile data. The interpolator is accessed
through the ``get_values()`` method of a ``Profile`` class object. Other
useful methods are the ``append()`` method, which can add a parameter to the
profile database, the ``extend_profile_deeper()`` method, which can create a
synthetic, deeper profile by preserving the stratification, and the
``buoyancy_frequency()`` method, which reports the local buoyancy frequency
for a given depth.

In its original design, the ``ambient`` module required all
``ambient.Profile`` objects to be stored as netCDF datasets. This was later
updated to allow ``numpy`` arrays as data storage containers. In its present
version, ``Profile`` objects can be created from ``netCDF`` datasets,
``numpy`` arrays, or ``xarray`` datasets. Irrespective of what data are used
to create the ``ambient.Profile`` object, data are converted to
``xarray.Dataset`` objects, which allow access to the power database tools in
``xarray``.

All of the original functionality is preserved, and the present version of
``ambient.Profile`` should be compatible with older versions for creating
Profile objects. Some of the attributes of the older version of the
``ambient.Profile`` class are no longer relevant, and have been removed;
hence, any scripts that directly accessed object attributes may need to be
updated. However, the API to the class and its methods has remained unchanged.

Notes
-----

The latest version of the ``ambient`` module preserves the capability to
instantiate from `'netCDF4'` Dataset objects and netCDF files. It also uses
the `'scipy'` interpolation functions for interpolation. The profile database
is organized in an ``xarray.Dataset``; hence, the current code has the
following minimum dependencies:

* ``netCDF4``
* ``datetime``
* ``numpy``
* ``scipy``
* ``xarray``

For the plotting methods built into the `ambient.Profile` object, you will
also need

* ``matplotlib.pyplot``

Examples
--------

Example scripts that create ``ambient.Profile`` objects from different data
sources are provided in the ./bin directory of ``tamoc``. Please refer to
those scripts for relevant examples.

"""
# S. Socolofsky, February 2022, Texas A&M University <socolofs@tamu.edu>

from __future__ import (absolute_import, division, print_function)
unicode = type(u' ')

from tamoc import seawater

from copy import copy
import os

from netCDF4 import Dataset
from time import ctime

import xarray as xr
import numpy as np 
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt


class BaseProfile(object):
    """
    Base functionality for ``ambient.Profile`` objects
    
    This class includes the base functionality needed by all ambient.Profile
    objects. To the extent possible, this base class uses ``xarray.Dataset``
    objects to handle all of the profile data. Users should use the
    ``ambient.Profile`` class to create objects rather than using this class
    directly. See the documentation for the ``ambient.Profile`` class for
    details.
    
    """
    def __init__(self, data, ztsp=['z', 'temperature', 'salinity',
                 'pressure'], ztsp_units=['m', 'K', 'psu', 'Pa'],
                 chem_names=None, chem_units=None, err=0.01):
        
        super(BaseProfile, self).__init__()
        
        # Convert the input data to appropriate types
        if isinstance(chem_names, type(None)):
            chem_names = []
        if isinstance(chem_units, type(None)):
            chem_units = []
        
        # Save the raw input data
        self.data = data
        self.ztsp = ztsp.copy()
        self.ztsp_units = ztsp_units.copy()
        self.chem_names = chem_names.copy()
        self.chem_units = chem_units.copy()
        self.nchems = len(self.chem_names)
        self.err = err
        
        # Build the BaseProfile from the provided xarray Dataset
        self._create_profile_from_xarray()
    
    def _create_profile_from_xarray(self, time=None):
        """
        Initialize a `BaseProfile` object from an ``xarray Dataset``
        
        Notes
        -----
       `` xarray.Dataset`` objects are mutable; hence, functions and methods
        that act on them usually change the object in place.
        
        """
        # Select one time from the larger dataset
        if not isinstance(time, type(None)):
            
            # User specified the time to use
            if 'time' in self.data.dims:
                
                # Check if time exists in dataset
                if time <= self.data.dims['time']:
                    # Store selected time in self.ds
                    self.ds = self.data.isel(time=time)
                
                else:
                    # Time selection is out of range
                    error_message = '\nWarning!!!  Time is outside of range'
                    error_message += ' of xarray Dataset!\nUsing latest'
                    error_message += ' time in Dataset instead.'
                    print(error_message)
                    self.ds = self.data.isel(time=-1)
            
            else:
                # Unfortunately, dataset does not have times to select
                error_message = '\nWarning!!!  Time is not a coordinate'
                error_message += ' of the Dataset!  \nUsing the whole'
                error_message += ' Dataset instead.\n'
                print(error_message)
                self.ds = self.data
            
        else:
            # Use the first time-step in the dataset
            if 'time' in self.data.dims:
                self.ds = self.data.isel(time=0)
            else:
                self.ds = self.data
        
        # Insert the pressure data if missing by integrating the density
        if self.ztsp[-1] not in self.ds:
            # Extract the depth and temperature data
            zs = self.ds.coords[self.ztsp[0]].values
            Ts = self.ds[self.ztsp[1]].values
            Ss = self.ds[self.ztsp[2]].values
            fs_loc = np.min(np.where(zs == np.min(zs)))
            if fs_loc > 0:
                fs_loc = -1
            # Compute the pressure for this density profile
            Ps = compute_pressure(zs, Ts, Ss, fs_loc)
            # Insert the computed pressure into the dataset
            self.ds[self.ztsp[-1]] = ((self.ztsp[0]), Ps)
            self.ds[self.ztsp[-1]].attrs['units'] = 'Pa'
            self.ztsp_units[-1] = 'Pa'
        
        # Remove any data not requested by user
        keep_names = self.ztsp + self.chem_names
        for name in self.ds.data_vars:
            if name not in keep_names:
                self.ds = self.ds.drop_vars([name])
        
        # Add the unit labels passed to the initializer if they are not
        # already in the dataset
        xr_check_units(self.ds, self.ztsp, self.ztsp_units)
        xr_check_units(self.ds, self.chem_names, self.chem_units)
        
        # Update all the units in the dataset to match standard units
        # used in TAMOC
        xr_convert_units(self.ds, self.ztsp[0])
        
        # Coarsen the data for interpolation
        if self.err > 0.:
            self.interp_ds = xr_coarsen_dataset(self.ds, self.ztsp[0], 
                self.err)
        else:
            self.interp_ds = self.ds
        
        # Stablize the interpolation profile
        if 'pressure' in self.interp_ds:
            self.interp_ds = xr_stabilize_dataset(self.interp_ds,
                self.ztsp[0], self.ztsp)
        
        # Store the boundaries of the z-coordinate
        zs = self.interp_ds[self.ztsp[0]].values
        self.z_min = np.min(zs)
        self.z_max = np.max(zs)
        
        # Make sure the Profile.ds attribute points to the interp_ds
        self.ds = self.interp_ds
        
        # Build a hard-wired interpolator for speed
        self._build_interpolator()
    
    def _build_interpolator(self):
        """
        Create a linear interpolation function to report profile data
        
        Although the ``xarray.Dataset`` package wraps the ``scipy``
        interpolation functions, they do not operate very fast. This method
        extracts all of the data in the ``xarray.Dataset`` into ``numpy``
        arrays and then creates a ``scipy.interpolation.interp1d`` function
        as an attribute of the ``ambient.Profile`` class. This appears to be
        much faster than using the interpolation tools wrapped by ``xarray``.
        
        """
        # Extract the data from the interpolation dataset
        self.interp_data, names, units = xr_dataset_to_array(self.interp_ds, 
            self.ztsp[0])
        
        # Record the variables and their units
        self.f_names = names[1:]
        self.f_units = units[1:]
        
        # Create the interpolator
        self.f = interp1d(self.interp_data[:,0], 
            self.interp_data[:,1:].transpose())
    
    def append(self, data, data_names, data_units, comments=None, z_col=0):
        """
        Add data to the ``xarray.Dataset`` and ``Profile`` object
        
        This method operates only on the `xarray` data variables in the
        ``Profile`` object. Since the use may also need to append data to a
        ``netCDF4`` dataset, the ``Profile`` class extends this method. See
        the ``Profile.append()`` method for details.
        
        """
        # Make sure the strings are in lists
        if isinstance(data_names, str) or isinstance(data_names, unicode):
            data_names = [data_names]
        if isinstance(data_units, str) or isinstance(data_units, unicode):
            data_units = [data_units]
        if isinstance(comments, str) or isinstance(comments, unicode):
            comments = [comments]
        
        # Add each column of data in the appropriate location
        for i in range(len(data_names)):
            
            # Make sure we never replace the independent variable
            if data_names[i] != self.ztsp[0]:
                
                # Insert the data into the Dataset
                new_data = np.vstack((data[:,z_col], data[:,i])).transpose()
                
                xr_add_data_from_numpy(self.interp_ds, self.ztsp[0],
                    new_data, data_names[i], data_units[i])
                
                # Insert the comments
                if comments != None:
                    self.interp_ds[data_names[i]].attrs['comment'] = \
                        comments[i]
                
                # Update the chemical counter
                if data_names[i] not in self.ztsp and \
                    data_names[i] not in self.chem_names:
                    self.nchems += 1
                    self.chem_names += [data_names[i]]
                    self.chem_units += \
                        self.interp_ds[data_names[i]].attrs['units']
                
        # Update the units
        xr_convert_units(self.interp_ds, self.ztsp[0])
        
        # Update the variables available in the dataset
        self.ds = self.interp_ds
        
        # Re-build the interpolate
        self._build_interpolator()
    
    def extend_profile_deeper(self, z_new, h_N=1.0, h=0.01, N=None):
        """
        Extend the ``Profile`` data using a fixed buoyancy frequency
        
        This method operates only on the `xarray` data variables in the
        ``Profile`` object. Since the user may also need to append data to a
        ``netCDF4`` dataset, the ``Profile`` class extends this method. See
        the ``Profile.extend_profile_deeper()`` method for details.
        
        """
        # Get the buoyancy frequency if not already specified
        if N == None:
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
            
            Keeping the temperature constant, compute the salinity S needed
            to achieve the desired potential density at the bottom of the 
            new profile, rho_1
            
            Parameters
            ----------
            S : float
                Current guess for the new salinity at the base of the extended
                CTD profile
            
            T, S, Pa, and rho_1 passed in globally from the above calculations
            
            Returns
            -------
            delta_rho : float
                Difference between the desired density rho_1 at the base of 
                the new profile and the current estimate of `rho` using the 
                current guess for the salinity S.  
            
            Notes
            -----
            Because compressibility effects should be ignored in estimating
            the buoyancy frequency, a constant pressure is used to yield
            the potential density.
            
            """
            rho = seawater.density(T, S, Pa)
            return (rho_1 - rho)
        
        # Find the optimum values of salinity
        from scipy.optimize import fsolve
        S = fsolve(residual, S)
        
        # Create the extended profile
        names = list(self.interp_ds.keys())
        z_0 = self.z_max
        z_1 = z_new
        y_0 = self.get_values(z_0, names)
        y_1 = self.get_values(z_0, names)
        y_1[names.index(self.ztsp[2])] = S
        
        # Create several points along the profile so that the pressure
        # integration will be accurate
        z_new = np.linspace(z_0, z_1, num=50)
        y_new = np.zeros((z_new.shape[0], len(y_0)))
        y_new[:,:] = y_0
        
        # Fill in the salinity by linear interpolation
        S_0 = y_0[names.index(self.ztsp[2])]
        S_1 = S
        y_new[:,names.index(self.ztsp[2])] = \
            (S_1 - S_0) / (z_1 - z_0) * (z_new - z_0) + S_0
        
        # Get the right pressure
        Te = y_new[:,names.index(self.ztsp[1])]
        Se = y_new[:,names.index(self.ztsp[2])]
        Pa = compute_pressure(z_new, Te, Se, 0)
        y_new[:,names.index(self.ztsp[3])] = Pa
        
        # Create a new xarray Dataset with this extended data
        data, names, units = xr_dataset_to_array(self.interp_ds, 
            self.ztsp[0])
        new_data = np.hstack((np.atleast_2d(z_new).transpose(), y_new))
        new_data = np.vstack((data[:-1,:], new_data))
        self.interp_ds = xr_array_to_dataset(new_data, names, units)
        self.z_max = z_1
        self.ds = self.interp_ds
        
        # Rebuild the interpolator
        self._build_interpolator()
    
    def add_computed_gas_concentrations(self):
        """
        Estimate dissolved concentrations of atmospheric gases
        
        Concentrations of atmospheric gases in seawater can be estimated by
        assuming equilibrium with the atmosphere at formation and then
        adjusting for pressure. This method encapsulates this approach and
        automatically estimates concentrations for nitrogen, oxygen, argon,
        and carbon dioxide. If measured values for one of these gases already
        exist in the ``Profile`` dataset, the text ``computed_`` is
        pre-pended to the parameter name of the computed variable from this
        function (e.g., computed_oxygen).  
        
        This method has a description of standard air built in, and all 
        calculations are conducted automatically.
        
        """
        # Extract the z-coordinate and T, S, P profile
        zs = self.interp_ds.coords[self.ztsp[0]].values
        Ts = self.interp_ds[self.ztsp[1]].values
        Ss = self.interp_ds[self.ztsp[2]].values
        Ps = self.interp_ds[self.ztsp[3]].values
        
        # Create an air object
        air_names = ['nitrogen', 'oxygen', 'argon', 'carbon_dioxide']
        yk = np.array([0.78084, 0.20946, 0.009340, 0.00036])
        from tamoc import dbm
        air = dbm.FluidMixture(air_names)
        m = air.masses(yk)
        
        # Compute the concentrations adjusted for depth
        Cs = np.zeros((len(zs), len(air_names)))
        for i in range(len(zs)):
            Cs[i,:] = air.solubility(m, Ts[i], 101325., Ss[i])[0,:] * \
                seawater.density(Ts[i], Ss[i], Ps[i]) / \
                seawater.density(Ts[i], Ss[i], 101325.)
        
        # Make sure none of these gases are already in the measured profile
        for name in air_names:
            if name in self.interp_ds:
                air_names[air_names.index(name)] = 'computed_' + name
        
        # Add these data to the Profile object
        data = np.hstack((np.atleast_2d(zs).transpose(), Cs))
        names = [self.ztsp[0]] + air_names 
        units = [self.ztsp_units[0]] + 4*['kg/m^3']
        self.append(data, names, units)
        
        # Rebuild the interpolator
        self._build_interpolator()
    
    def get_values(self, z, names):
        """
        Return the values for the requested variables at the given depths
        
        This method queries the ``Profile`` database for the parameters
        listed in ``names`` and interpolates their values to the depth ``z``.
        The results are returned as a ``numpy`` array, with the values
        reported in the same order as the parameters are listed in ``names``.
        If a requested parameter is not present in the database, a value of
        zero is returned. If a depth range outside of the profile range is
        requested, the nearest value of the profile is returned. It is
        expected that the out-of-range case only occurs near the ocean free
        surface when an ODE solver of the other parts of the TAMOC suite
        overshoot the surface.
        
        Parameters
        ----------
        z : float or np.ndarray
            Depth(s) for which the parameter values are desired.
        names : list of str
            A list of string names corresponding to parameters in the
            ``Profile`` database. These names are case sensitive, and must
            exactly match a parameter in the database; otherwise, a zero
            value is returned
        
        Returns
        -------
        y : np.ndarray
            An array of numerical values obtained by linear interpolation
            from the stored ``Profile`` database, reported in the same order
            as the parameter names in the ``names`` input parameter.
        
        """
        # Make sure names is a list
        if isinstance(names, str) or isinstance(names, unicode):
            names = [names]
        
        # Make sure z is an array
        if not isinstance(z, np.ndarray):
            if not isinstance(z, list):
                z = np.array([z])
            else:
                z = np.array(z)
        
        # Catch the out-of-range error.  This should only occur when an ODE
        # solver gets close to the boundary; thus, it is acceptable to revert
        # to the solution at the boundary
        z[np.where(z < self.z_min)] = self.z_min
        z[np.where(z > self.z_max)] = self.z_max
        
        # Determine which parameters are in the database and where they go
        # in the output vector
        ans_cols = [names.index(name) 
                    for name in names if name in self.f_names]
        
        # Interpolate the data
        if ans_cols:
            # Get the names and locations of these interpolated data
            i_names = [names[col] for col in ans_cols]
            i_cols = [self.f_names.index(name) for name in i_names]
            
            if z.shape[0] == 1:
                ans = np.zeros(len(names))
                interp_data = self.f(z)[i_cols].transpose()
                ans[ans_cols] = interp_data
            else:
                ans = np.zeros((z.shape[0], len(names)))
                interp_data = self.f(z).transpose()[:,i_cols]
                ans[:,ans_cols] = interp_data
        else:
            if z.shape[0] == 1:
                ans = np.zeros(len(names))
            else:
                ans = np.zeros((z.shape[0], len(names)))
        
        # Return the results
        return ans
    
    def get_units(self, names):
        """
        Return a list of units for the requested variables
        
        Parameters
        ----------
        names : list
            List of string variable names for which the units are desired
        
        Returns
        -------
        units : list
            List of string unit abbreviations specifying the units of each
            variable in the same order as they are listed in ``names``
        
        Notes
        -----
        The names of the units are extracted from the xarray.Dataset
        metadata, which belong in the attribute ``.attrs['units']``.
        
        """
        # Make sure names is a list
        if isinstance(names, str) or isinstance(names, unicode):
            names = [names]
        
        # Return the list of units
        ans = []
        for name in names:
            if name in self.interp_ds:
                ans.append(self.interp_ds[name].attrs['units'])
            else:
                ans.append('Not Available in Dataset')
        
        return ans
    
    def buoyancy_frequency(self, z, h=0.01):
        """
        Calculate the local buoyancy frequency at the given depth
        
        Calculate the buoyancy frequency at the depth ``z``, optionally using
        the length-scale ``h`` to obtain smooth results. This calculation
        uses the in-situ pressure at the depth z as a constant so that the
        effect of compressibility is removed, yielding a buoyancy frequency
        for the potential density.
        
        Parameters
        ----------
        z : float or ndarray
            Depth(s) (m) at which data are desired.  The value of z must 
            lie between the ``z_min`` and ``z_max`` values of the depth 
            coordinate in the ``Profile`` database.
        h : float, default value is 0.01
            Fraction of the water depth (--) to use as the length-scale in a 
            finite-difference approximation to the density gradient.
        
        Returns
        -------
        N : float
            The buoyancy frequency (1/s).
        
        Notes
        -----
        The ``Profile`` object reduces the total amount of stored data when
        it creates the built-in interpolator, and the level of acceptable
        error is specified by the attribute ``Profile.err``. The buoyancy
        frequency calculation here uses this interpolation function, when the
        buoyancy frequency of the smoothed dataset will be returned.
        
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
    
    
    def insert_density(self, P0=None):
        """
        Compute the density and add it to the profile database
        
        This method computes the seawater density at each depth in the
        profile database and adds the computed value to the database with the
        parameter name 'density' and units of kg/m^3.
        
        This method has the optional parameter ``P0``. If this parameter is
        not used, the local density with compressibility effects is computed.
        If this parameter is specified, then the density is computed at all
        depth using this single pressure, hence, returning a potential
        density based on this pressure.
        
        Parameters
        ----------
        P0 : float, default=None
            If a value is given, all densities are computed with this fixed
            values of the pressure (Pa)
        
        """
        # Extract the z-coordinate
        zs = self.interp_ds.coords[self.ztsp[0]].values
        
        # Commpute the density at each depth
        density = np.zeros(len(zs))
        Ts, Ss, Ps = self.get_values(zs, ['temperature', 'salinity', 
            'pressure']).transpose()
        for i in range(len(zs)):
            if P0:
                density[i] = seawater.density(Ts[i], Ss[i], P0)
            else:
                density[i] = seawater.density(Ts[i], Ss[i], Ps[i])
        
        # Insert these data into the dataset
        if not P0:
            self.interp_ds['density'] = ((self.ztsp[0]), density)
            self.interp_ds['density'].attrs['units'] = 'kg/m^3'
        else:
            return (density)
        
        # Rebuild the interpolator
        self._build_interpolator()
    
    def insert_potential_density(self):
        """
        Compute the potential density and add it to the profile database
        
        This method computes the seawater potential density using atmospheric
        pressure to the baseline. The computed values is stored in the
        profile database with the parameter name 'theta' in units
        of kg/m^3.
        
        """
        # Compute the potential density
        density = self.insert_density(P0=101325.)
        
        # Insert these data into the dataset
        self.interp_ds['theta'] = ((self.ztsp[0]), density)
        self.interp_ds['theta'].attrs['units'] = 'kg/m^3'
        self.interp_ds['theta'].attrs['long_name'] = 'potential density'
        
        # Rebuild the interpolator
        self._build_interpolator()
    
    def insert_buoyancy_frequency(self):
        """
        Compute the local buoyancy frequency and add it to the profile 
        
        This method uses the ``Profile.buoyancy_frequency()`` method to
        compute the buoyancy frequency at each depth in the profile database
        and then adds these values to the profile with the parameter name
        'N' in units of 1/s.
        
        """
        # Extract the z-coordinate
        zs = self.interp_ds.coords[self.ztsp[0]].values
        
        # Compute the buoyancy frequency
        n_vals = self.buoyancy_frequency(zs)
        
        # Insert these data into the dataset
        self.interp_ds['N'] = ((self.ztsp[0]), n_vals)
        self.interp_ds['N'].attrs['units'] = '1/s'
        self.interp_ds['N'].attrs['long_name'] = 'buoyancy frequency'
        
        # Rebuild the interpolator
        self._build_interpolator()
    
    def plot_parameter(self, parm):
        """
        Plot a depth profile for the given parameter
        
        This method uses the ``xarray`` built-in plotting methods to plot a
        depth profile of the selected parameter in the currently active
        figure axes.  These methods rely on ``matplotlib.pyplot``.
        
        Parameters
        ----------
        parm : str
            String name of the parameter to add to the plot.
        
        """
        # If user wants to plot density, make sure it exists
        if parm == 'density' and 'density' not in self.ds.data_vars:
            self.insert_density()
        
        if parm == 'theta' and 'theta' not in self.ds.data_vars:
            self.insert_potential_density()
        
        if parm == 'N' and 'N' not in self.ds.data_vars:
            self.insert_buoyancy_frequency()
        
        # Use xarray to plot this parameter
        self.ds[parm].plot(y=self.ztsp[0])
        if plt.ylim()[0] <= 0:
            plt.gca().invert_yaxis()
        plt.tight_layout()
    
    def plot_profiles(self, parameters, fig=1):
        """
        Plot depth profiles for the given list of parameters
        
        This method uses the ``xarray`` built-in plotting methods to plot a
        depth profile of each the selected parameters. These data are plotted
        in separate subplots, with up to three plots per row.These methods
        rely on ``matplotlib.pyplot``.
        
        Parameters
        ----------
        parm : list of str
            List of string names of the parameter to plot, each parameter
            added to its own subplot.
        fig : int, default=1
            Figure number in which to create the plot
        
        """
        # Make sure the parameters are a list
        if isinstance(parameters, str) or isinstance(parameters, unicode):
            parameters = [parameters]
        
        # Decide how many subplots to create
        nrows = int(len(parameters) / 3)
        if len(parameters) % 3 > 0:
            nrows += 1
        
        # Initialize a figure for plotting
        width = 8
        height = 4 * nrows
        plt.figure(fig, figsize=(width, height))
        plt.clf()
        
        # Plot each parameter
        for parm in parameters:
            ax = plt.subplot(nrows, 3, parameters.index(parm)+1)
            self.plot_parameter(parm)
    
    def plot_physical_profiles(self, fig=2):
        """
        Create a plot of temperature, salinity, potential density, and
        buoyancy frequency.
        
        Parameters
        ----------
        fig : int, default=2
            Figure number in which to create the plot
        
        """
        # Select the physical parameters
        phys_parms = ['temperature', 'salinity', 'theta', 'N']
        
        # Plot these parameters
        self.plot_profiles(phys_parms, fig)
    
    def plot_chem_profiles(self, fig=3):
        """
        Create a plot of all of the parameters in the chem_names list.
        
        Parameters
        ----------
        fig : int, default=3
            Figure number in which to create the plot
        
        """
        # Select the physical parameters
        chem_parms = self.chem_names
        
        # Plot these parameters
        self.plot_profiles(chem_parms, fig)
        
    def update_xr_time(self, time):
        """
        Set the ``Profile`` database to the given time index
        
        With the profile data stored in ``xarray.Dataset`` objects, it
        becomes easy to hold profiles for multiple time steps in the
        ``Profile`` object. This method allows the user to select a time
        index from this dataset. A profile is extracted at that time
        interval, and the interpolator is built. The ``Profile`` object will
        continue to report data from that time interval until a new time is
        selected through this method.
        
        """
        # Extract the new profile
        self._create_profile_from_xarray(time)


class Profile(BaseProfile):
    """
    Main class object for ambient seawater profiles
    
    This class object is the main container in ``tamoc`` for handling ambient
    seawater data. This particular class handles all of the backward
    compatibility with the original versions of the ``ambient.Profile``
    class, which was initially based on the ``netCDF4`` package. This was
    recently updated to ``numpy`` data and now is based on ``xarray.Dataset``
    objects. This class contains all of the capabilities to convert various
    types of inputs data (text, lists, numpy arrays, etc.) to the xarray
    Datasets used by the current ``ambient.BaseProfile`` class, and maintains
    backward compatibility with previous versions of the ``ambient.Profile``
    class.
    
    This class inherits the capabilities of the ``ambient.BaseProfile``
    class, which does all of the numerical work using ``xarray``.
    
    A profile is defined as a set of dependent variables (parameters) mapped
    to a single independent variable (depth). The minimum set of parameters
    to define a profile in ``tamoc`` includes temperature and salinity. Any
    other parameters can also be included, either when the object is
    initialized, or appended later.
    
    The ``Profile`` class indexes parameters in the dataset by unique string
    names. The user provides these names as input and requests data using the
    same names; hence, the ``Profile`` class itself does not hard-wire any
    parameter names into the database. Where possible, it is recommended to
    use names from the CF Convensions used by netCDF.
    
    Parameters
    ----------
    data : various
        An array of data that include the depth coordinate and several other
        dependent variables as additional parameters. The parameter ztsp
        contains the string names of the first four parameters in the
        dataset. Any other parameters passed to the class ``__init__()``
        method should be listed in the ``chem_names`` parameter.
    ztsp : list of str
        A list of string names that will be used to specify the depth
        (``z``), temperature, salinity, and pressure.
    chem_names : list of str
        A list of additional parameters passed in through data.  These
        will be the string names used to access ``Profile`` data through the
        ``get_values()`` method.
    err : float, default=0.01
        A parameter specifying the allowable relative error to use when
        thinning the profile data to reduce the amount of data that must be
        searched when the profile data are queried.
    ztsp_units : list of str
        A list of units corresponding to each parameter listed in ztsp
    chem_units : list of str
        A list of units corresponding to each parameter listed in
        chem_names
    current : np.ndarray
        Array of data corresponding to the ambient currents. This array is
        separate from the data parameter; hence, it must start with an array
        of depth coordinates, followed by values for the x-, y-, and
        z-components of velocity. This parameter is retained for backward
        compatibility, but it is recommended to instead use the ``append()``
        method of the ``Profile`` class to add current data after first
        initializing the ``Profile`` with standard CTD-type data.
    current_units : list of str
        A list of units corresponding to each column of the current array.
    
    Notes
    -----
    The ``data`` parameter can contain a wide variety of data and be in a
    variety of data types.  The following is a brief summary.
    
    data : xarray.Dataset
        The most appropriate way to initialize a ``Profile`` object in the
        present version of the ``ambient`` module is by storing the data in
        an ``xarray.Dataset`` object and passing this object as input. The
        other parameters used by ``__init__()`` (e.g., ztsp, chem_names,
        etc.) are used to select the data that should be read from the
        dataset and used to create the interpolation function. If data are
        provided in any of the other formats listed below, these data are
        converted to an ``xarray.Dataset`` object before they are passed on
        to the ``BaseProfile`` class to complete instantiation.
    data : str
        If data contains only a string, it is assumed this is the relative 
        path to a netCDF file.  The initializer will open the file using 
        ``xarray.open_dataset()`` and then build the Profile from those data
    data : netCDF4.Dataset
        If the user has already opened a netCDF file using the ``netCDF4``
        package in Python and passes this dataset as input, the original
        methods of the older version of the ``ambient.Profile`` process the
        data to create the object.
    data : np.ndarray
        The profile data can be passed directly as a ``numpy`` array, with
        the first column of data being the depth, the next three columns
        containing the temperature, salinity, and (optionally) pressure, and
        the remaining columns containing data in the order specified in
        ``chem_names``. If only depth, temperature, and salinity are given, a
        hydrostatic pressure profile will be calculated and added to the
        profile database.
    data : 1D np.ndarray or list
        It is also possible for the user to initialize a profile by only
        specifying the surface ocean properties. These may be passed as a 1D
        ``numpy`` array or as a list of numbers. These values are taken as
        the surface properties, and the remaining values at depth are
        supplied from the world-ocean average data in Sarmiento and Gruber
        (2006) (see next option).
    data : None
        If no data are given, the ``ambient`` module will construct a profile
        using the world-ocean average data for temperature, salinity, and
        oxygen concentration reported in Sarmiento and Gruber (2006) and
        distributed in the ``./tamoc/tamoc/data`` directory with ``tamoc``.
    
    Examples
    --------
    See the scripts in the ``./tamoc/bin/ambient`` directory for several
    examples of creating ``Profile`` objects from different types of data.
    
    """
    def __init__(self, data, ztsp=['z', 'temperature', 'salinity',
                 'pressure'], chem_names=None, err=0.01,
                 ztsp_units=['m', 'K', 'psu', 'Pa'],
                 chem_units=None, current=None, current_units=None):
        
        # Make sure we use the correct chem_names
        if isinstance(chem_names, str) or isinstance(chem_names, unicode):
            chem_names = [chem_names]
        if isinstance(chem_units, str) or isinstance(chem_units, unicode):
            chem_units = [chem_units]
        if chem_names == None:
            chem_names = []
            chem_units = []
        
        # Set some global class attributes
        self.nc = None
        self.nc_open = False
        
        # Determine what kind of data were sent to the constructor and call
        # the correct set of pre-processing algorithms
        if isinstance(data, Dataset):
            # The user passed a netCDF4 Dataset object connected to an open
            # netCDF file
            data, ztsp, ztsp_units, chem_names, chem_units = \
                self._from_netCDF_Dataset(data, ztsp, chem_names)
        
        elif isinstance(data, str) or isinstance(data, unicode):
            # The user passed the name of a netCDF file
            data, ztsp, ztsp_units, chem_names, chem_units = \
                self._from_netCDF_file(data, ztsp, chem_names)
        
        elif isinstance(data, np.ndarray) and \
            np.atleast_2d(data).shape[0] > 1:
            # The user passed a numpy array of data
            data, ztsp, ztsp_units, chem_names, chem_units = \
                self._from_numpy(data, ztsp, ztsp_units, chem_names, 
                chem_units)
        
        elif isinstance(data, type(xr.Dataset())):
            # The user passed an xarray Dataset object...no pre-processing
            # is required
            pass
        
        else:
            # The user wants to use data from the world-ocean average
            data = self._from_tamoc(data, ztsp, ztsp_units, chem_names, 
                chem_units)
        
        # Process the xarray.Dataset object through the BaseProfile
        # constructor
        super(Profile, self).__init__(data, ztsp, ztsp_units, chem_names,
                                      chem_units, err)
        
        # Add in the current data
        current_names = ['z', 'ua', 'va', 'wa']
        if not isinstance(current, type(None)):
            self._append_current(current, current_names, current_units)
    
    def _from_netCDF_Dataset(self, nc, ztsp, chem_names):
        """
        Create a profile object from an open netCDF dataset
        
        """
        # Store the dataset
        self.nc = nc
        self.nc_open = True
        
        # Get the variable names in the Dataset
        keys = []
        for key in self.nc.variables.keys():
            keys += [key]
        
        # Get the correct set of variables chosen by the user
        if 'all' in chem_names:
            # We need to take all variables in the dataset
            non_chems = ['time', 'lat', 'lon'] + ztsp
            chem_names = [name for name in keys if name not in non_chems]
            
        # Make sure the current data are included as the user probably
        # considers them much like the temperature and pressure
        current_vars = ['ua', 'va', 'wa']
        for var in current_vars:
            if var not in chem_names and var in keys:
                chem_names += [var]
        
        # Load the data from the netCDF dataset variables
        data, ztsp_units, chem_units = get_nc_data(self.nc, 
            ztsp, chem_names)
        
        # Create an xarray Dataset with these data
        names = ztsp + chem_names
        units = ztsp_units + chem_units
        data = xr_array_to_dataset(data, names, units)
        
        # Return the final set of data
        return (data, ztsp, ztsp_units, chem_names, chem_units)
    
    def _from_netCDF_file(self, nc_file, ztsp, chem_names):
        """
        Create a profile object from the data stored in a netCDF file
        
        """
        # Load the dataset into memory and close the file
        data = xr.open_dataset(nc_file)
        data = xr.Dataset.load(data)
        data.close()
        
        # Get the variable names available in the dataset
        keys = list(data.data_vars)
        
        # Set the list of chem_names if the 'all' flag is used
        if 'all' in chem_names:
            non_chems = ['time', 'lat', 'lon'] + ztsp
            chem_names = [name for name in keys if name not in non_chems]
        
        # Make sure the current data are included
        current_vars = ['ua', 'va', 'wa']
        for var in current_vars:
            if var not in chem_names and var in keys:
                chem_names += [var]
        
        # Pull the units for the required variables
        ztsp_units = []
        for var in ztsp:
            ztsp_units += [data[var].attrs['units']]
        chem_units = []
        for var in chem_names:
            chem_units += [data[var].attrs['units']]
        
        # Return the final set of data
        return (data, ztsp, ztsp_units, chem_names, chem_units)
    
    def _from_numpy(self, data, ztsp, ztsp_units, chem_names, chem_units):
        """
        Create a profile from a numpy array of data
        
        """
        # Join all the variable names together...note that the depth 
        # coordinate must be listed first
        if data.shape[1] == 3:
            # Pressure data are not included yet...
            names = ztsp[:data.shape[1]]
            units = ztsp_units[:data.shape[1]]
        else:
            names = ztsp + chem_names
            units = ztsp_units + chem_units
        
        # Create an xarray dataset from these variables
        ds = xr_array_to_dataset(data, names, units)
        
        # Return the data
        return (ds, ztsp, ztsp_units, chem_names, chem_units)
        
    
    def _from_tamoc(self, data, ztsp, ztps_units, chem_names, chem_units):
        """        
        Create a profile from data for the world-ocean average temperature,
        salinity, and oxygen concentration, reported in Sarmiento and Gruber
        (2006)
        
        """
        # Get the appropriate version of the data
        if isinstance(data, np.ndarray):
            
            # Extract the temperature and salinity
            Ts = data[1]
            Ss = data[2]
            
            # Use these values to get the world-ocean average profile
            data, ztsp, ztsp_units, chem_names, chem_units = \
                get_world_ocean(Ts, Ss)
        
        else:
            # Get the default world-ocean dataset
            data, ztsp, ztsp_units, chem_names, chem_units = \
                get_world_ocean()
        
        # Create an xarray dataset of these numpy data
        ds, ztsp, ztsp_units, chem_names, chem_units = \
            self._from_numpy(data, ztsp, ztsp_units, chem_names, chem_units)
        
        # Return the profile
        return ds
    
    def _append_current(self, current, current_names, current_units):
        """
        Append current data passed to the ``__init__()`` method through the
        array ``current`` and list ``current_units``.
        
        """
        # Make sure the units list is complete
        if isinstance(current_units, str) or \
            isinstance(current_units, unicode):
            # Assume all components have the same units
            current_units = [current_units] * 3
        elif len(current_units) == 2:
            # Add units for a vertical velocity component
            current_units += [current_units[0]]
        
        # Create an input vector that includes a complete set of vector
        # components
        if isinstance(current, float):
            # User specified only the x-coordinate velocity
            current = np.array([current, 0., 0.])
        if isinstance(current, list):
            # Convert the input list to a np.ndarray
            current = np.array(current)
        if np.atleast_2d(current).shape[0] == 1:
            if len(current) == 2:
                # Add zero vertical velocity to uniform current
                current = np.append(current, 0.)
        else:
            if current.shape[1] == 3:
                # Add a zero vertical velocity to current data
                wa = np.zeros((current.shape[0], 1))
                current = np.hstack(current, np.atleast_2d(wa))
        
        # Insert the currents
        if np.atleast_2d(current).shape[0] == 1:
            # Currents are uniform over the depth
            current_data = np.zeros((2,4))
            current_data[:,0] = np.array([self.z_min, self.z_max])
            current_data[0,1:] = current
            current_data[1,1:] = current
        else:
            # The user provided a profile of currents data
            current_data = current
        
        # Make sure we have the write units for the currents
        if len(current_units) < 4:
            current_units = ['m'] + current_units
        
        # Append these data to the dataset
        self.append(current_data, current_names, current_units)
    
    def append(self, data, var_symbols, var_units, comments=None, z_col=0):
        """
        Add data to the ``xarray.Dataset`` and ``Profile`` object
        
        This method adds new data to a ``Profile`` object. This method
        performs all tasks necessary to make the ``Profile`` object aware of
        the new data and make the data accessible through the
        ``get_values()`` and ``get_units()`` methods.
        
        Parameters
        ----------
        data : ndarray
            Table of data to add to the profile database.  If it contains more
            than one variable, the data are assumed to be arranged in columns.
        var_symbols : string list
            List of string symbol names (e.g., T, S, P, etc.) in the same 
            order as the columns in the data array.
        var_units : string list
            List of units associated with each variable in the var_symbols
            list.
        comments : string list
            List of comments associated with each variable in the 
            var_symbols list.  As a minimum, this list should include the 
            indications 'measured' or 'derived' or some similar indication of 
            source of the data.  With the xarray.Dataset, these comments are 
            not currently used.
        z_col : integer, default is 0
            Column number of the column containing the depth data.  The first 
            column is numbered zero.
        
        Notes
        -----
        
        This method has the identical API to the original ``append`` method
        in the ``ambient`` module. It is also possible to append data
        directly to the xarray Dataset by modifying the ``Profile.ds``
        attribute variable. However, that would not add data to the
        interpolator or make it available to the ``Profile`` object. Hence,
        always use this ``append()`` method to add data to an existing
        ``Profile`` object.
        
        """
        if self.nc_open:
            # Make sure the parameter names are in a list
            if isinstance(var_symbols, str) or \
                isinstance(var_symbols, unicode):
                    var_symbols = [var_symbols]
            # Add the data to the netCDF dataset
            self.nc = fill_nc_db(self.nc, data, 
                var_symbols, var_units, comments, z_col)
        
        # Add the data to the ambient Profile object
        BaseProfile.append(self, data, var_symbols, var_units, 
            comments, z_col)
    
    def extend_profile_deeper(self, z_new, nc_name=None, h_N=1.0, h=0.01,
        N=None): 
        """ 
        Extend the depth of a `Profile` object 
        
        Extends the depth of a CTD profile to the new depth ``z_new`` using a
        fixed buoyancy frequency to adjust the salinity and keeping all other
        variables constant below the original depth. This should only be used
        when no ambient data are available and when the bottom of the
        original profile is deep enough that it is acceptable to assume all
        variables remain constant in the extension except for salinity, which
        is increased to maintain the desired buoyancy frequency. Pressure is
        also adjusted assuming a hydrostatic profile.
        
        The fixed buoyancy frequency used to extend the profile can be 
        specified in one of two ways:
        
        1. Specify ``h_N`` and ``h`` and let the method
           ``buoyancy_frequency`` evaluate the buoyancy frequency. The
           evaluation depth is taken at the fraction ``h_N`` of the original
           CTD depth range, with ``h_N`` = 1.0 yielding the lowest depth in
           the original CTD profile. ``h`` is passed to
           ``buoyancy_frequency`` unchanged and sets the length-scale of the
           finite difference approximation.
        
        2. Specify ``N`` directly as a constant.  In this case, the method
           ``buoyancy_frequency`` is not called, and any values passed to 
           ``h_N`` or ``h`` are ignored.
        
        Parameters
        ----------
        z_new : float
            New depth for the valid_max of the CTD profile
        h_N : float, default 1.0
            Fraction of the water depth (--) at which to compute the buoyancy
            frequency used to extend the profile
        h : float, default 0.01
            Passed to the ``buoyancy_frequency()`` method.  Fraction of the 
            water depth (--) to use as the length-scale in a finite-
            difference approximation to the density gradient.
        N : float, default is None
            Optional replacement of h_N and h, forcing the extension 
            method to use N as the buoyancy frequency.  
        
        Notes
        -----
        One may extend a profile any number of ways; this method merely
        provides easy access to one rational method. If other methods are
        desired, it is recommended to create a new dataset with the desired
        deeper profile data and then use that dataset to initialize a new
        ``BaseProfile`` object.
        
        See Also
        --------
        buoyancy_frequency
        
        """
        # Record the present depth of the profile
        z0 = self.z_max
        
        # Extend the profile to the desired depth
        BaseProfile.extend_profile_deeper(self, z_new, h_N=h_N, h=h, N=N)
        
        # Update the netCDF dataset if needed
        if self.nc_open == True:
            
            # Extract the data from the current xarray Dataset
            data, names, units = xr_dataset_to_array(self.interp_ds,
                self.ztsp[0])
            
            # Get the netCDF attributes for the present dataset
            summary = self.nc.summary
            source = self.nc.source
            sea_name = self.nc.sea_name
            p_lat = self.nc.variables['lat'][:]
            p_lon = self.nc.variables['lon'][:]
            p_time = self.nc.variables['time'][:]
            self.nc.close()
            
            # Create the new netCDF file.
            extention_text = ': extended from %g to %g' % (z0, self.z_max)
            source = source + extention_text + ' on date ' + ctime()
            self.nc = create_nc_db(nc_name, summary, source, sea_name, p_lat, 
                                   p_lon, p_time)
            
            # Fill the netCDF file with the extended profile.
            comments = ['extended'] * len(names)
            
            self.nc = fill_nc_db(self.nc, data, names, units, comments,
                z_col=0)
    
    def close_nc(self):
        """
        Close an open `netCDF4.Dataset` object
        
        If the raw profile data are stored in a netCDF file, a pipe is open
        to the file source whenever this attribute of the object is in use.
        Once the xarray Dataset is built with the complete data needed for a
        TAMOC simulation (e.g., once calls to `append` or
        `extend_profile_deeper` are no longer needed), the netCDF dataset
        does not need to remain available. This method closes the netCDF
        dataset file and tells the `Profile` object that the file is closed.
        All methods that do not directly read or write data to the netCDF
        file will continue to work. If the object is no longer used, the
        Python garbage collector will eventually find it and remove it from
        memory.
        
        Notes
        -----
        Especially during code development, scrips with open netCDF files can
        crash, and this method will not be called.  In that case, it is
        likely necessary to close Python and restart a new session so that 
        all file pointers can be flushed.  
        
        """
        # Close the netCDF file
        if self.nc_open == True:
            self.nc.close()
        
        # Remove the object attributes from memory and delete the object
        self.nc_open = False
        self.nc = None


# - Functions for manipulating xarray.Dataset objects ------------------------

def xr_check_units(ds, params, units):
    """
    Check that units are specified for a dataset
    
    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset object
    params : list
        List of variable names stored in the xarray Dataset
    units : list
        List of units corresponding to the variables in the params list
    
    Returns
    -------
    ds : xarray.Dataset
        The original Dataset with meta data for units included if they 
        were absent
    
    Notes
    -----
    This function simply inserts the given units into the Dataset if the
    Dataset did not already have units specified in its metadata.  It is 
    assumed that the units listed in the input are the units of the variables
    in the Dataset.  This function does not do any unit conversions.
    
    """
    # Check each parameter and its units separately
    for param in params:
        
        # Update the units with the specified set
        try:
            param_unit = ds[param].attrs['units']
        
        except KeyError:
            
            # Try to update the units
            try:
                ds[param].attrs['units'] = units[params.index(param)]
            
            except KeyError:
                # This variable does not exist in the dataset
                error_message = '\nWarning!!! The parameter %s ' % \
                    (param)
                error_message += 'is not contained in the xarray Dataset '
                error_message += '\nbeing processed!\n'
                print(error_message)

def xr_convert_units(ds, z_coord):
    """
    Convert variables in an xarray Dataset to standard TAMOC units
    
    Parameters
    ----------
    ds : xarray.Dataset
        The Dataset for which we want to check and update units
    z_coord : str
        String name for the dependent variable in the dataset
    
    Returns
    -------
    ds : xarray.Dataset
        The updated Dataset with correct TAMOC units
    
    Notes
    -----
    For a list of recognized units and standard ``tamoc`` units, please see
    ``ambient.convert_units()``.
    
    See Also
    --------
    convert_units
    
    """
    # Convert units for the z-coordinate
    vals = ds.coords[z_coord]
    units = ds.coords[z_coord].attrs['units']
    vals, units = convert_units(vals, units)
    ds.coords[z_coord] = vals
    ds.coords[z_coord].attrs['units'] = units[0]
    
    # Go through all the dependent variables in the dataset
    params = list(ds.keys())
    for param in params:
        # Convert units for each variable
        vals = ds[param].values
        units = ds[param].attrs['units']
        vals, units = convert_units(vals, units)
        ds[param] = ((z_coord), vals)
        ds[param].attrs['units'] = units[0]

def xr_coarsen_dataset(ds, z_coord, err):
    """
    Reduce the size of a raw database for interpolation
    
    Some CTD data, especially data from at-sea casts, can contain a very
    large number of depth measurements. The ``Profile`` objects only need to
    retain enough data so that linear interpolation will give results within
    an acceptable level of error. This function removes unnecessary data
    points, preserving the accuracy within a relative error given by the
    ``err`` parameter.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the complete database of profile data
    z_coord : str
        String name for the dependent variable in the dataset
    err : float
        Relative error that is to be retained by linear interpolation of the
        thinned dataset
    
    Returns
    -------
    ds_new : xarray.Dataset
        A new dataset contains the reduced set of data
    
    See Also
    --------
    coarsen
    
    """
    # Extract the data for coarsening
    data, names, units = xr_dataset_to_array(ds, z_coord)
    
    # Use the coarsen function
    data = coarsen(data, err)
    
    # Rebuild a dataset with the reduced amount of data
    ds_new = xr_array_to_dataset(data, names, units)
    
    return ds_new

def xr_stabilize_dataset(ds, z_coord, ztsp_names):
    """
    Stabilize the profile so that there are no density reversals
    
    When collecting CTD data, it is common that reversals in the density
    profile are captured (heavier water at lower depths than lighter water).
    This unstable density profile can cause some of the ``tamoc`` simulation
    modules to get stuck or throw errors because unstable stratification is
    not expected. This function scans the profile data, identifies regions of
    instability, and removes unstable data points until the whole profile is
    neutral or stable.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the complete database of profile data
    z_coord : str
        String name for the dependent variable in the dataset
    ztsp_names : list of str
        A list of string names that will be used to specify the depth
        (``z``), temperature, salinity, and pressure.
    
    Returns
    -------
    ds_new : xarray.Dataset
        A new dataset contains the reduced set of data
    
    See Also
    --------
    stabilize
    
    """
    # Extract the data for stabilizing
    data, names, units = xr_dataset_to_array(ds, z_coord)
    
    # Put the data into the right order of columns
    raw_data = np.zeros(data.shape)
    raw_names = ['temporary'] * raw_data.shape[1]
    j = 4
    for name in names:
        if name in ztsp_names:
            raw_data[:,ztsp_names.index(name)] = data[:,names.index(name)]
            raw_names[ztsp_names.index(name)] = name
        else:
            raw_data[:,j] = data[:,names.index(name)]
            raw_names[j] = name
            j += 1
    
    # Stabilize the data
    data = stabilize(raw_data)
    
    # Rebuild the dataset with the stabilized data
    ds_new = xr_array_to_dataset(data, raw_names, units)
    
    return ds_new

def xr_dataset_to_array(ds, z_coord):
    """
    Extract the data in an xarray.Dataset and store as a numpy array
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the complete database of profile data
    z_coord : str
        String name for the dependent variable in the dataset
    
    Returns
    -------
    data : np.ndarray
        A numpy array of profile data with the organized in columns by the
        depth coordinate, temperature, salinity, pressure, and the chem_names
        properties, which could include the ambient velocity data.
    names : list of str
        A list of string names for each column in the data array
    units : list of str
        A list of units for each parameter in the names list
    
    """
    # Determine the size of the dataset
    nvars = len(ds.keys())
    nvals = len(ds.coords[z_coord].values)
    
    # Create an empty array to store the data
    data = np.zeros((nvals, nvars+1))
    units = []
    
    # Insert the depth coordinate
    data[:,0] = ds.coords[z_coord].values
    units.append(ds.coords[z_coord].attrs['units'])
    
    # Insert the rest of the data
    variables = list(ds.keys())
    for i in range(len(variables)):
        data[:,i+1] = ds[variables[i]].values
        units.append(ds[variables[i]].attrs['units'])
    
    # Create a list of variables names
    names = [z_coord] + variables
    
    # Return the data
    return (data, names, units)

def xr_array_to_dataset(data, names, units):
    """
    Replace data in an xarray Dataset with a new array of data
    
    Parameters
    ----------
    data : np.ndarray
        A numpy array of profile data with the organized in columns by the
        depth coordinate, temperature, salinity, pressure, and the chem_names
        properties, which could include the ambient velocity data.
    names : list of str
        A list of string names for each column in the data array
    units : list of str
        A list of units for each parameter in the names list
    
    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the complete database of profile data
    z_coord : str
        String name for the dependent variable in the dataset
    
    """
    # Create an empty dataset
    ds_new = xr.Dataset()
    
    # Insert the coordinates
    ds_new.coords[names[0]] = data[:,0]
    ds_new.coords[names[0]].attrs['units'] = units[0]
    
    # Insert the rest of the data
    for i in range(1, len(names)):
        ds_new[names[i]] = ((names[0]), data[:,i])
        ds_new[names[i]].attrs['units'] = units[i]
    
    return ds_new

def xr_add_data_from_numpy(ds, z_coord, data, params, units):
    """
    Add data from a numpy array to an xarray Dataset
    
    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset object
    z_coord : str
        Name of the z-coordinate of the dataset
    data : ndarray
        A two-dimensional array of data to add to an xarray Dataset.  The 
        rows should be for different depths and the columns should be the
        variables specified in the params list.  The first column should
        contain the depth coordinate (m).
    params : list
        List of variable names corresponding to the data in the numpy data
        array.  This list names columns 1:-1 (i.e., the depth coordinate
        is not included in this list)
    units : list
        List of units corresponding to the variables in the params list.
    
    Returns
    -------
    ds : xarray.Dataset
        The updated xarray dataset with the new data included
    
    """
    # Make sure the params list is a list
    if isinstance(params, str) or isinstance(params, unicode):
        params = [params]
    if isinstance(units, str) or isinstance(units, unicode):
        units = [units]
    
    # Get the z-coordinate used by the xarray Dataset
    zs = ds.coords[z_coord].values
    
    # Create an interpolator to map the new data to the depth coordinates
    # in the xarray dataset
    z = data[:,0]
    y = data[:,1:]
    f = interp1d(z, y, axis=0, bounds_error=False, 
        fill_value=(y[0,:], y[-1,:]))
    
    # Interpolate the new data onto the coordinates of the xarray Dataset
    new_data = np.zeros((len(zs), len(params)))
    for i in range(len(zs)):
        new_data[i,:] = f(zs[i])
    
    # Insert these data into the xarray Dataset
    for param in params:
        ds[param] = ((z_coord), new_data[:, params.index(param)])
        ds[param].attrs['units'] = units[params.index(param)]


# - Functions for manipulating netCDF4.Dataset objects ------------------------

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
    z.valid_max = 12000.0
    
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
        List of string symbol names (e.g., T, S, P, etc.) in the same order 
        as the columns in the data array.  For chemical properties, use the 
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
    if isinstance(var_symbols, str) or isinstance(var_symbols, unicode):
        var_symbols = [var_symbols]
    if isinstance(var_units, str) or isinstance(var_units, unicode):
        var_units = [var_units]
    if isinstance(comments, str) or isinstance(comments, unicode):
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
            interp_data[:,z_col] = np.array(z[:])
            interp_data[:,y_cols] = f(np.array(z[:])).transpose()
            data = np.copy(interp_data)
    
    # Handle the dependent variables
    for i in range(ny+1):
        
        if i == z_col:
            # Skip the depth data (already handeled above)
            pass
        else:
            # Processes the dependent data depending on the variable type
            std_name = ' '.join(var_symbols[i].split('_'))
            long_name = std_name.capitalize()
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
        z_dim = 'z'
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

def get_nc_data(nc, ztsp, chem_names):
    """
    Extract named data from a netCDF file
    
    Parameters
    ----------
    nc : netCDF4 data object
        An existing netCDF dataset object that contains the data that are 
        to be extracted
    ztsp : list of str
        A list of string names that will be used to specify the depth
        (``z``), temperature, salinity, and pressure.
    chem_names : list of str
        A list of additional parameters passed in through data.  These
        will be the string names used to access ``Profile`` data through the
        ``get_values()`` method.
    
    Returns
    -------
    data : np.ndarray
        A numpy array of profile data with the organized in columns by the
        depth coordinate, temperature, salinity, pressure, and the chem_names
        properties, which could include the ambient velocity data.
    ztsp_units : list of str
        A list of units for the depth, temperature, salinity, and pressure 
        data
    chem_units : list of str
        A list of units for the data in the chem_names list
    
    """
    z = nc.variables[ztsp[0]][:]
    z_units = [nc.variables[ztsp[0]].units]
    
    y_names = ztsp[1:] + chem_names
    y = np.zeros((z.shape[0], len(y_names)))
    y_units = []
    for i in range(len(y_names)):
        y[:,i] = nc.variables[y_names[i]][:]
        y_units.append(nc.variables[y_names[i]].units)
    
    data = np.hstack((np.atleast_2d(z).transpose(), y))
    ztsp_units = z_units + y_units[0:len(ztsp)]
    chem_units = y_units[len(ztsp):]
    
    return (data, ztsp_units, chem_units)


# - Functions for manipulating numpy arrays -----------------------------------

def add_data(data, col, var, new_data, var_symbols, var_units, comments, 
             z_col):
    """
    Add data to a numpy array
    
    Adds data to a profile data base contained in a numpy array.  This 
    function is similar to fill_nc_db(), but operating on numpy arrays
    instead of a netCDF dataset.
    
    Parameters
    ----------
    data : ndarray
        `Numpy` array of profile data organized by column.  The first column
        should be the independent variable (depth), followed sequentially
        by the dependent variables (temperature, salinity, pressure and any
        included dissolved chemicals).
    col : int
        Number of the column in `data` to be added.
    var : str
        String containing the variable name of the new data to add to the
        profile dataset.
    new_data : ndarray
        `Numpy` array containing the new data to be added to the current 
        dataset.
    var_symbols : list of str
        List of strings containing the variable names of the variables 
        in the new_data dataset.
    var_units : list of str
        List of strings containing the units of the variables in the 
        new_data dataset.
    comments : str
        String containing comments for the comment attribute of the variable.
        For instance, 'Measured', 'Derived quantity', etc.
    z_col : int, default is 0
        Column number of the column containing the depth data.  The first 
        column is numbered zero.
    
    Returns
    -------
    data : ndarray
        Numpy array of profile data updated with the new variable in 
        new_data.
    
    """
    # Make sure the new data extend to the top and bottom of the existing data
    if np.min(new_data[:,z_col]) > data[0,0]:
        ctd = np.zeros((new_data.shape[0]+1, new_data.shape[1]))
        ctd[0,:] = new_data[0,:]
        ctd[0,z_col] = data[0,0]
        ctd[1:,:] = new_data
        new_data = np.copy(ctd)
    if np.max(new_data[:,z_col]) < data[-1,0]:
        ctd = np.zeros((new_data.shape[0]+1, new_data.shape[1]))
        ctd[-1,:] = new_data[-1,:]
        ctd[-1,z_col] = data[-1,0]
        ctd[0:-1,:] = new_data
        new_data = np.copy(ctd)
    
    # Create an interpolation function to map the new data to the exiting
    # depths
    y = new_data[:,var_symbols.index(var)]
    z = new_data[:,z_col]
    f = interp1d(z, y)
    
    # Insert the data where they belong
    n_cols = data.shape[1]
    if col < n_cols:
        # Replace existing data
        data[:,col] = f(data[:,0]).transpose()
    else:
        # Add new data to the dataset
        ctd = np.zeros((data.shape[0], data.shape[1]+1))
        ctd[:,0:-1] = data
        ctd[:,-1] = f(data[:,0]).transpose()
        data = np.copy(ctd)
    
    return data


# - Functions for manipulating profile data -----------------------------------

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
    z_sign = int(np.sign(z[len(z) // 2]))
    
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
               'mS/m' : [1.e-3, 0., 'S/m'],
               'psu': [1.0, 0., 'psu'], 
               'salinity': [1.0, 0., 'psu'], 
               'kg/m^3': [1.0, 0., 'kg/m^3'], 
               'kilogram meter-3': [1.0, 0., 'kg/m^3'], 
               'm/s': [1.0, 0., 'm/s'], 
               'mg/l': [1.e-3, 0., 'kg/m^3'],
               'meter second-1' : [1.0, 0., 'm/s'],
               'm.s-1' : [1.0, 0., 'm/s'],
               'pH units' : [1.0, 0., 'pH units']
           } 
    
    # Make sure the data are a numpy array and the units are a list
    if isinstance(data, float) or isinstance(data, int):
        data = np.array([data])
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(units, str) or isinstance(units, unicode):
        units = [units]
    if units == None:
        units = ['']
    
    # Make sure you can slice through the columns:  must be two-dimensional
    sh = data.shape
    data = np.atleast_2d(data)
    
    # Allow conversion of a row of data if all of the same unit
    if len(units) == 1 and data.shape[1] > 1:
        data = data.transpose()
    
    # Create an emtpy array to hold the output
    out_data = np.zeros(data.shape)
    out_units = []
    
    # Convert the units
    for i in range(len(units)):
        try:
            out_data[:,i] = data[:,i] * convert[units[i]][0] + \
                        convert[units[i]][1]
            out_units += [convert[units[i]][2]]
        except KeyError:
            print('Do not know how to convert %s to mks units' % units[i])
            print('Continuing without converting these units...')
            out_data[:,i] = data[:,i]
            out_units += units[i]
    
    # Return the converted data in the original shape
    out_data = np.reshape(out_data, sh, 'C')
    return (out_data, out_units)


# - Functions to access the world-ocean average data --------------------------

def get_world_ocean(Ts=290.41, Ss=34.89):
    """
    Load the world ocean average CTD data
    
    Load the world ocean average temperature, salinity, and oxygen profile
    data from Levitus et al. (1998) as reported on Page 226 of Sarmiento and
    Gruber (2006), "Ocean Biogeochemical Dynamics." If surface ocean
    properties of temperature and salinity are known, then scale the profile
    to match these properties.
    
    Parameters
    ----------
    Ts : float, default=290.41
        Temperature of the ocean surface (K)
    Ss : float, default=34.89
        Salinity of the ocean surface (psu)
    
    Returns
    -------
    data : ndarray
        Array containing the profile data organized with depth in the 
        first column, temperature, salinity and pressure in the next three
        columns and any chemical concentration data in the remaining 
        columns.
    ztsp : str list
        String list containing the variables names for depth, temperature, 
        salinity, and pressure that are to be used in the Profile data.
    ztsp_units : str list
        String list containing the units for depth, temperature, salinity, 
        and pressure.
    chem_names : str list
        Names of the chemicals (e.g., those constituents in addition to z, T,
        S, P) in the dataset that should be accessible through the
        `self.get_values` interpolation method or the `self.get_units`
        interrogator of the `Profile` object.
    chem_units : str list
        Names of the units for each constituent in the `chem_names` variable
    
    """
    # Find the path to the world_ocean_ave_ctd.txt file distributed with
    # `tamoc`
    __location__ = os.path.realpath(os.path.join(os.getcwd(), 
                                    os.path.dirname(__file__), 'data'))
    ctd_fname = os.path.join(__location__,'world_ocean_ave_ctd.dat')
    
    # Load the CTD data
    raw_data = np.loadtxt(ctd_fname, comments='%')
    
    # Convert to `tamoc` standard units
    raw_data[:,1] = raw_data[:,1] + 273.15
    raw_data[:,3:] = raw_data[:,3:] * 31.9988 / 1.e6
    
    # Adjust the temperature and salinity
    Ts += 273.15
    S_fac = Ss / raw_data[0,2]
    for i in range(raw_data.shape[0]):
        if raw_data[i,1] > Ts:
            raw_data[i,1] = Ts
        raw_data[i,2] *= S_fac
    
    # Compute the pressure
    z = raw_data[:,0]
    Tz = raw_data[:,1]
    Sz = raw_data[:,2]
    Pz = compute_pressure(z, Tz, Sz, 0)
    
    # Assemble the data in the expected order
    data = np.zeros((z.shape[0], raw_data.shape[1]+1))
    data[:,:3] = raw_data[:,:3]
    data[:,3] = Pz
    data[:,4:] = raw_data[:,3:]
    
    # Create the remaining outputs
    ztsp = ['z', 'temperature', 'salinity', 'pressure']
    ztsp_units=['m', 'K', 'psu', 'Pa']
    chem_names = ['oxygen', 'oxygen_sat']
    chem_units = ['kg/m^3', 'kg/m^3']
    
    return (data, ztsp, ztsp_units, chem_names, chem_units)


# - Functions required to complete backward compatibility -------------------- 

def load_raw(fname):
    """
    Use numpy.loadtxt instead when possible
    
    This function is retained to allow complete backward compatibility with
    any scripts that may have used it. Please use ``numpy.loadtxt`` instead
    when possible.
    
    Read all of the data in file fname with `*` or `#` as comment characters.
    Data are assumed to be separated by spaces.
    
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

