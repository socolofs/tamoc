###########################################
Tools for Manipulating Ambient Profile Data
###########################################

Before any simulation can be conducted with ``tamoc``, you must provide
ambient profile data for temperature, salinity, and any other properties that
the simulation may need.

A lot of the work to generate ambient CTD data and to put it into the
appropriate format for use by ``TAMOC`` is a 'hands on' process that is
unique to each project. Many times, this work can be easily completed in an
IPython interactive session using the tools supplied in this package.

Examples
========

In each of these examples, the general process follows a similar sequence of
steps. Here we demonstrate working with CTD data following some of the steps
in the `profile_from_ctd` script given above.

Reading in Ambient Data Files
-----------------------------

Read in some (or all) of the data. The first step will be to prepare a
`numpy.ndarray` of data that includes the depths coordinate. For this
example, we read in selected columns from ``./data/ctd_BM54.cnv``. We selected
these columns by reading the ``.cnv`` file by hand. After changing
directory to the ``./data/`` directory, we start an IPython session::

   >>> cols = (0, 1, 3, 8, 9, 10, 12)
   >>> raw = np.loadtxt('ctd_BM54.cnv', skiprows = 175, usecols = cols)
   >>> symbols = ['temperature', 'pressure', 'wetlab_fluorescence', 'z', 
                  'salinity', 'density', 'oxygen']
   >>> units = ['deg C', 'db', 'mg/m^3', 'm', 'psu', 'kg/m^3', 'mg/l']
   >>> z_col = 3

Many times, the raw CTD profile will contain information at the top or the 
bottom of the profile that must be discarded, typically indicated by 
reversals in the depth profile.  It is particularly important to remove
these reversals so that the interpolation methods will be able to find 
unique profiles values for any input depth::

   >>> import ambient
   >>> data = ambient.extract_profile(raw, z_col, z_start = 50.0)

Before the data should be stored in the netCDF dataset used by TAMOC, the 
units should be converted to the standard mks system::

   >>> profile, units = ambient.convert_units(data, units)

Preparing the netCDF Dataset
----------------------------
   
An empty netCDF dataset must be created with the global metadata describing
this ambient profile before the data can be imported into the dataset::

   >>> summary = 'Description of the TAMOC project using this data'
   >>> source = 'Documentation of the data source'

This next set of information is read manually by the user from the header 
file of the CTD text file and entered as follows::

   >>> sea_name = 'Gulf of Mexico'
   >>> lat = 28.0 + 43.945 / 60.0
   >>> lon = 360 - (88.0 + 22.607 / 60.0)
   
Finally, we must set the time that the CTD data were collected.  This is
done using several data manipulation methods:: 

   >>> from datetime import datetime
   >>> from netCDF4 import num2date, date2num
   >>> date = datetime(2010, 5, 30, 18, 22, 12)
   >>> t_units = 'seconds since 1970-01-01 00:00:00 0:00'
   >>> calendar = 'julian'
   >>> time = date2num(date, units=t_units, calendar=calendar)
   
Create the empty dataset::

   >>> nc = ambient.create_nc_db('../Profiles/BM54.nc', summary, source, \
                                 sea_name, lat, lon, time)

Adding Data to the netCDF Dataset
---------------------------------

Insert the CTD data and the associated comments into the netCDF dataset::

   >>> comments = ['measured'] * len(symbols)
   >>> nc = ambient.fill_nc_db(nc, profile, symbols, units, comments, z_col)

At this point the CTD data are now in a netCDF dataset with the correct
units and including all data needed by TAMOC.  If the data had originated
in netCDF format, the process could have started here.  To demonstrate
methods to work with netCDF data, we close this file and then continue our
session using the stored netCDF profile data.

   >>> nc.close()   # This is the end of the preprocessing stage

Using the `ambient.Profile` Object
----------------------------------

A profile object can be initialized either by passing the file-name of the
netCDF dataset or by passing the `netCDF4.Dataset` object itself.  If the 
variable names in the dataset match those used by ``TAMOC``, the 
`ambient.Profile` class instantiation can extract all the information itself::

   >>> ctd_auto = ambient.Profile('../test/output/BM54.nc', chem_names='all')

If you want to specify the variable names for z, T, S, and P, and also
for the chemicals to load, that may also be done::

   >>> ztsp = ['z', 'temperature', 'salinity', 'pressure']
   >>> chem_names = ['oxygen']   # This selects a subset of available data
   >>> ctd_manual = ambient.Profile('../test/output/BM54.nc', ztsp, chem_names)

If you prefer to open the netCDF file and pass the `netCDF4.Dataset` object, 
that works identically::

   >>> from netCDF4 import Dataset
   >>> nc = Dataset('../Profiles/BM54.nc')
   >>> ctd_from_nc = ambient.Profile(nc)  # This will not load any chemicals

Occasionally, it is necessary simulate a problem nearby, where the depth is 
somewhat deeper than that in the measured profile, or in another region, 
where data are not available.  The `ambient.Profile` object provides a method
to extend the profile to a deeper total depth while maintaining the 
stratification profile::

   >>> ctd_auto.extend_profile_deeper(2500., '../test/output/BM54_deeper.nc')

Pipes to the netCDF datasets should be closed before ending an interactive
or script session::

   >>> ctd_auto.close_nc()
   >>> ctd_manual.close_nc()
   >>> ctd_from_nc.close_nc()





