"""
Unit tests for the `ambient` module of ``TAMOC``

Provides testing of all of the functions, classes and methods in the `ambient`
module. These tests rely on data stored in the ``./data`` folder and will
write data to and read data from the ``./test/output`` folder.

"""
# S. Socolofsky, July 2013, Texas A&M University <socolofs@tamu.edu>.

from tamoc import ambient

import os
import numpy as np
from numpy.testing import *
from scipy.interpolate import interp1d
from datetime import datetime
from netCDF4 import Dataset, date2num, num2date

# ----------------------------------------------------------------------------
# Functions used by unit tests
# ----------------------------------------------------------------------------

def get_units(data, units, nr, nc, mks_units, ans=None):
    """
    Run the ambient.convert_units function and test that the data are 
    correctly converted per the inputs given above.
    
    """
    # Apply the units conversion function
    data, units = ambient.convert_units(data, units)
    
    # Check shape of output compared to input
    assert np.atleast_2d(data).shape[0] == nr
    assert np.atleast_2d(data).shape[1] == nc
    
    # Check units converted as expected
    for i in range(len(units)):
        assert units[i] == mks_units[i]
    
    # Check numerical result is correct if known
    if ans is not None:
        assert_array_almost_equal(data, ans, decimal = 6)
    
    # Send back the converted data
    return (data, units)


def get_profile(data, z_col, z_start, p_col, P, z_min, z_max, nr, nc):
    """
    Run the ambient.extract_profile function and test that the data are 
    correctly parsed per the inputs given above.
    
    """
    # Apply the profile extraction function
    prof_data = ambient.extract_profile(data, z_col=z_col, z_start=z_start,
                                        p_col=p_col, P_atm=P)
    
    # Check that the returned profile extends to the free surface
    assert prof_data[0,z_col] == 0.0
    
    # Check that the profile is clipped at the expected depths
    assert_approx_equal(prof_data[1,z_col], z_min, significant = 6)
    assert_approx_equal(prof_data[-1,z_col], z_max, significant = 6)
    
    # Check that the returned profile is the right shape and data type
    assert prof_data.shape[0] == nr
    if nc is not None:
        assert prof_data.shape[1] == nc
    assert isinstance(prof_data, np.ndarray)
    
    # Check the that returned profile is in ascending order
    for i in range(1, prof_data.shape[0]):
        assert prof_data[i,z_col] > prof_data[i-1,z_col]
    
    # Send back the extracted profile
    return prof_data


def check_nc_db(nc_file, summary, source, sea_name, p_lat, 
                p_lon, p_time):
    """
    Use the ambient.create_nc_db() function to create a netCDF4-classic 
    dataset from the given inputs and then check whether the dataset is 
    created properly.
    
    """
    # Create the dataset
    nc = ambient.create_nc_db(nc_file, summary, source, sea_name, p_lat, 
                              p_lon, p_time)
    
    # Access the variables in the dataset
    time = nc.variables['time']
    lat = nc.variables['lat']
    lon = nc.variables['lon']
    z = nc.variables['z']
    T = nc.variables['temperature']
    S = nc.variables['salinity']
    P = nc.variables['pressure']
    
    # Check that the global attributes are set correctly
    assert nc.summary == summary
    assert nc.source == source
    assert nc.sea_name == sea_name
    
    # Check that the imutable data are written properly
    assert lat[0] == p_lat
    assert lon[0] == p_lon
    assert time[0] == p_time
    assert z.shape == (0,)
    
    # Check the units are correct on the following variables
    assert z.units == 'm'
    assert T.units == 'K'
    assert S.units == 'psu'
    assert P.units == 'Pa'
    
    # Send back the template database
    return nc


def get_filled_nc_db(nc, data, symbols, units, comments, z_col, 
                     long_names, std_names):
    """
    Check that data written to a netCDF dataset has been stored correctly.
    
    """
    # Store the data in the netCDF dataset
    z_len = nc.variables['z'][:].shape
    nc = ambient.fill_nc_db(nc, data, symbols, units, comments, z_col)
    
    # Check that data and metadata were stored properly
    if len(symbols) == 1:
        data = np.atleast_2d(data).transpose()
    for i in range(len(symbols)):
        assert_array_almost_equal(nc.variables[symbols[i]][:], 
                                  data[:,i], decimal = 6)
        assert nc.variables[symbols[i]].long_name == long_names[i]
        assert nc.variables[symbols[i]].standard_name == std_names[i]
        assert nc.variables[symbols[i]].units == units[i]
        assert nc.variables[symbols[i]].comment == comments[i]
    
    # Send back the correctly-filled dataset
    return nc


def get_profile_obj(nc, chem_names, chem_units):
    """
    Check that an ambient.Profile object is created correctly and that the
    methods operate as expected.
    
    """
    if isinstance(chem_names, str):
        chem_names = [chem_names]
    if isinstance(chem_units, str):
        chem_units = [chem_units]
    
    # Create the profile object
    prf = ambient.Profile(nc, chem_names=chem_names)
    
    # Check the chemical names and units are correct
    for i in range(len(chem_names)):
        assert prf.chem_names[i] == chem_names[i]
    assert prf.nchems == len(chem_names)
    
    # Check the error criteria on the interpolator
    assert prf.err == 0.01
    
    # Check the get_units method
    name_list = ['temperature', 'salinity', 'pressure'] + chem_names
    unit_list = ['K', 'psu', 'Pa'] + chem_units
    for i in range(len(name_list)):
        assert prf.get_units(name_list[i])[0] == unit_list[i]
    units = prf.get_units(name_list)
    for i in range(len(name_list)):
        assert units[i] == unit_list[i]
    
    # Check the interpolator function ...
    # Pick a point in the middle of the raw dataset and read off the depth
    # and the values of all the variables
    nz = prf.nc.variables['z'].shape[0] / 2
    z = prf.z[nz]
    y = prf.y[nz,:]
    # Get an interpolated set of values at this same elevation
    yp = prf.f(z)
    # Check if the results are within the level of error expected by err
    for i in range(len(name_list)):
        assert np.abs((yp[i] - y[i]) / yp[i]) <= prf.err
    
    # Next, check that the variables returned by the get_values function are
    # the variables we expect
    Tp, Sp, Pp = prf.get_values(z, ['temperature', 'salinity', 'pressure'])
    T = prf.nc.variables['temperature'][nz]
    S = prf.nc.variables['salinity'][nz]
    P = prf.nc.variables['pressure'][nz]
    assert np.abs((Tp - T) / T) <= prf.err
    assert np.abs((Sp - S) / S) <= prf.err
    assert np.abs((Pp - P) / P) <= prf.err
    if prf.nchems > 0:
        c = np.zeros(prf.nchems)
        cp = np.zeros(prf.nchems)
        for i in range(prf.nchems):
            c[i] = prf.nc.variables[chem_names[i]][nz]
            cp[i] = prf.get_values(z, chem_names[i])
            assert np.abs((cp[i] - c[i]) / c[i]) <= prf.err
    
    # Test the append() method by inserting the temperature data as a new 
    # profile, this time in degrees celsius using the variable name temp
    n0 = prf.nchems
    z = prf.nc.variables['z'][:]
    T = prf.nc.variables['temperature'][:]
    T_degC = T - 273.15
    assert_array_almost_equal(T_degC + 273.15, T, decimal = 6)
    data = np.vstack((z, T_degC)).transpose()
    symbols = ['z', 'temp']
    units = ['m', 'deg C']
    comments = ['measured', 'identical to temperature, but in deg C']
    prf.append(data, symbols, units, comments, 0)
    
    # Check that the data were inserted correctly
    Tnc = prf.nc.variables['temp'][:]
    assert_array_almost_equal(Tnc, T_degC, decimal = 6)
    assert prf.nc.variables['temp'].units == 'deg C'
    
    # Check that get_values works correctly with vector inputs for depth
    depths = np.linspace(prf.nc.variables['z'].valid_min, 
                         prf.nc.variables['z'].valid_max, 100)
    Temps = prf.get_values(depths, ['temperature', 'temp'])
    for i in range(len(depths)):
        assert_approx_equal(Temps[i,0], Temps[i,1] + 273.15, significant = 6)
    
    # Make sure the units are returned correctly
    assert prf.get_units('temp')[0] == 'deg C'
    assert prf.nc.variables['temp'].units == 'deg C'
    
    # Check that temp is now listed as a chemical
    assert prf.nchems == n0 + 1
    assert prf.chem_names[-1] == 'temp'
    
    # Test the API for calculating the buoyancy frequency (note that we do 
    # not check the result, just that the function call does not raise an 
    # error)
    N = prf.buoyancy_frequency(depths)
    N = prf.buoyancy_frequency(depths[50], h=0.1)
    
    # Send back the Profile object
    return prf


# ----------------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------------

def test_conv_units():
    """
    Test the units conversion methods to make sure they produce the expected
    results.
    
    """
    # Test conversion of 2d array data
    data = np.array([[10, 25.4, 9.5, 34], [100, 10.7, 8.4, 34.5]])
    units = ['m', 'deg C', 'mg/l', 'psu']
    mks_units = ['m', 'K', 'kg/m^3', 'psu']
    ans = np.array([[1.00000000e+01, 2.98550000e+02, 9.50000000e-03,
                     3.40000000e+01],
                    [1.00000000e+02, 2.83850000e+02, 8.40000000e-03,
                     3.45000000e+01]])
    data, units = get_units(data, units, 2, 4, mks_units, ans)
    
    # Test conversion of scalar data
    data = 10.
    data, units = get_units(data, 'deg C', 1, 1, ['K'], 
                            np.array([273.15+10.]))
    
    # Test conversion of a row of data
    data = [10, 25.4, 9.5, 34]
    units = ['m', 'deg C', 'mg/l', 'psu']
    mks_units = ['m', 'K', 'kg/m^3', 'psu']
    ans = np.array([1.00000000e+01, 2.98550000e+02, 9.50000000e-03,
                     3.40000000e+01])
    data, units = get_units(data, units, 1, 4, mks_units, ans)
    
    # Test conversion of a column of data
    data = np.array([[10., 20., 30., 40]]).transpose()
    unit = 'deg C'
    ans = np.array([[ 283.15], [293.15], [303.15], [313.15]])
    data, units = get_units(data, unit, 4, 1, ['K'], ans)


def test_from_ctd():
    """
    Test the ambient data methods on a Sea-Bird SBE 19plus Data File.  
    
    This unit test reads in the CTD data from ./data/ctd.BM54.cnv using
    `numpy.loadtxt` and then uses this data to test the data manipulation and
    storage methods in ambient.py.
    
    """
    # Get a platform-independent path to the datafile
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    '../tamoc/data'))
    dfile = os.path.join(__location__,'ctd_BM54.cnv')
    
    # Load in the raw data using np.loadtxt
    raw = np.loadtxt(dfile, comments = '#', skiprows = 175, 
                     usecols = (0, 1, 3, 8, 9, 10, 12))
    
    # State the units of the input data (read by hand from the file)
    units = ['deg C', 'db', 'mg/m^3', 'm', 'psu', 'kg/m^3', 'mg/l']
    
    # State the equivalent mks units (translated here by hand)
    mks_units = ['K', 'Pa', 'kg/m^3', 'm', 'psu', 'kg/m^3', 'kg/m^3']
    
    # Clean the profile to remove depth reversals
    z_col = 3
    p_col = 1
    profile = get_profile(raw, z_col, 50, p_col, 0., 2.124, 1529.789, 11074, 
                          7)
    
    # Convert the profile to standard units
    profile, units = get_units(profile, units, 11074, 7, mks_units)
    
    # Create an empty netCDF4-classic dataset to store the CTD information
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    'output'))
    nc_file = os.path.join(__location__,'test_BM54.nc')
    summary = 'Py.Test test file'
    source = 'R/V Brooks McCall, station BM54'
    sea_name = 'Gulf of Mexico'
    p_lat = 28.0 + 43.945 / 60.0
    p_lon = 360 - (88.0 + 22.607 / 60.0) 
    p_time = date2num(datetime(2010, 5, 30, 18, 22, 12), 
                      units = 'seconds since 1970-01-01 00:00:00 0:00', 
                      calendar = 'julian')
    nc = check_nc_db(nc_file, summary, source, sea_name, p_lat, 
                     p_lon, p_time)
    
    # Fill the netCDF4-classic dataset with the data in profile
    symbols = ['temperature', 'pressure', 'wetlab_fluorescence', 'z', 
               'salinity', 'density', 'oxygen']
    comments = ['measured', 'measured', 'measured', 'measured', 'measured',
                'measured', 'measured']
    long_names = ['Absolute temperature', 'pressure', 'Wetlab Fluorescence', 
                  'depth below the water surface', 'Practical salinity', 
                  'Density', 'Oxygen']
    std_names = ['temperature', 'pressure', 'wetlab fluorescence', 'depth', 
                 'salinity', 'density', 'oxygen']
    nc = get_filled_nc_db(nc, profile, symbols, units, comments, z_col, 
                          long_names, std_names)
    
    # Create a Profile object from this netCDF dataset and test the Profile
    # methods
    bm54 = get_profile_obj(nc, ['oxygen'], ['kg/m^3'])
    
    # Close down the pipes to the netCDF dataset files
    bm54.nc.close()


def test_from_txt():
    """
    Test the ambient data methods on simple text files.  
    
    This unit test reads in the text files ./data/C.dat and 
    ./data/T.dat using `numpy.loadtxt` and then uses this data to test 
    the data manipulation and storage methods in ambient.py.
    
    """
    # Get a platform-independent path to the datafile
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    '../tamoc/data'))
    cdat_file = os.path.join(__location__,'C.dat')
    tdat_file = os.path.join(__location__,'T.dat')
    
    # Load in the raw data using np.loadtxt
    C_raw = np.loadtxt(cdat_file, comments = '%')
    T_raw = np.loadtxt(tdat_file, comments = '%')
    
    # Clean the profile to remove depth reversals
    C_data = get_profile(C_raw, 1, 25, None, 0., 1.0256410e+01, 8.0000000e+02, 
        34, 2)
    T_data = get_profile(T_raw, 1, 25, None, 0., 1.0831721e+01, 7.9922631e+02, 
        34, 2)
    
    # Convert the data to standard units
    C_data, C_units = get_units(C_data, ['psu', 'm'], 34, 2, ['psu', 'm'])
    T_data, T_units = get_units(T_data, ['deg C', 'm'], 34, 2, ['K', 'm'])
    
    # Create an empty netCDF4-classic dataset to store the CTD information
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    'output'))
    nc_file = os.path.join(__location__,'test_DS.nc')
    summary = 'Py.Test test file'
    source = 'Profiles from the SINTEF DeepSpill Report'
    sea_name = 'Norwegian Sea'
    p_lat = 64.99066
    p_lon = 4.84725 
    p_time = date2num(datetime(2000, 6, 27, 12, 0, 0), 
                      units = 'seconds since 1970-01-01 00:00:00 0:00', 
                      calendar = 'julian')
    nc = check_nc_db(nc_file, summary, source, sea_name, p_lat, 
                     p_lon, p_time)
    
    # Fill the netCDF4-classic dataset with the data in the salinity profile
    symbols = ['salinity', 'z']
    comments = ['measured', 'measured']
    long_names = ['Practical salinity', 'depth below the water surface']
    std_names = ['salinity', 'depth']
    nc = get_filled_nc_db(nc, C_data, symbols, C_units, comments, 1, 
                            long_names, std_names)
    
    # Because the temperature data will be interpolated to the vertical 
    # coordinates in the salinity profile, insert the data and test that 
    # insertion worked correctly by hand    
    symbols = ['temperature', 'z']
    comments = ['measured', 'measured']
    long_names = ['Absolute temperature', 'depth below the water surface']
    std_names = ['temperature', 'depth']
    nc = ambient.fill_nc_db(nc, T_data, symbols, T_units, comments, 1)
    assert_array_almost_equal(nc.variables['z'][:], 
        C_data[:,1], decimal = 6)
    z = nc.variables['z'][:]
    T = nc.variables['temperature'][:]
    f = interp1d(z, T)
    for i in range(T_data.shape[0]):
        assert_approx_equal(T_data[i,0], f(T_data[i,1]), significant = 5)
    assert nc.variables['temperature'].comment == comments[0]
    
    # Calculate and insert the pressure data
    z = nc.variables['z'][:]
    T = nc.variables['temperature'][:]
    S = nc.variables['salinity'][:]
    P = ambient.compute_pressure(z, T, S, 0)
    P_data = np.vstack((z, P)).transpose()
    nc = ambient.fill_nc_db(nc, P_data, ['z', 'pressure'], ['m', 'Pa'], 
                            ['measured', 'computed'], 0)
    
    # Test the Profile object 
    ds = get_profile_obj(nc, [], [])
    
    # Close down the pipes to the netCDF dataset files
    ds.nc.close()


def test_from_calcs():
    """
    Test the ambient data methods on synthetic profiles.
    
    This unit test creates synthetic data (e.g., profiles matching laboratory
    idealized conditions) and then uses this data to test the data 
    manipulation and storage methods in ambient.py.
    
    """
    # Create the synthetic temperature and salinity profiles
    z = np.array([0.0, 2.4])
    T = np.array([21.0, 20.0])
    S = np.array([0.0, 30.0])
    
    # Create an empty netCDF4-classic dataset to store the CTD information
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    'output'))
    nc_file = os.path.join(__location__,'test_Lab.nc')
    summary = 'Py.Test test file'
    source = 'Synthetic profiles for idealized laboratory conditions'
    sea_name = 'None'
    p_lat = -999
    p_lon = -999 
    p_time = date2num(datetime(2013, 7, 12, 11, 54, 0), 
                      units = 'seconds since 1970-01-01 00:00:00 0:00', 
                      calendar = 'julian')
    nc = check_nc_db(nc_file, summary, source, sea_name, p_lat, 
                     p_lon, p_time)
    
    # Convert the temperature units
    T, T_units = get_units(T, ['deg C'], 1, 2, ['K'])
    
    # Fill the netCDF4-classic dataset with the data in these variables
    nc = get_filled_nc_db(nc, z, ['z'], ['m'], ['synthetic'], 0, 
                          ['depth below the water surface'], ['depth'])
    
    # Check that we cannot overwrite this existing z-data
    try:
        nc = ambient.fill_nc_db(nc, z, 'z', 'm', 'synthetic', 0)
    except ValueError:
        assert True is True
    else:
        assert True is False
    
    # Fill in the remaining data
    data = np.zeros((2, 3))
    data[:,0] = z
    data[:,1] = T
    data[:,2] = S
    nc = get_filled_nc_db(nc, data, ['z', 'temperature', 'salinity'], 
                          ['m', 'K', 'psu'], 
                          ['synthetic', 'synthetic', 'synthetic'], 0, 
                          ['depth below the water surface', 
                           'Absolute temperature', 'Practical salinity'], 
                          ['depth', 'temperature', 'salinity'])
    
    # Calculate and insert the pressure data
    P = ambient.compute_pressure(data[:,0], data[:,1], data[:,2], 0)
    P_data = np.vstack((data[:,0], P)).transpose()
    nc = ambient.fill_nc_db(nc, P_data, ['z', 'pressure'], ['m', 'Pa'], 
                            ['measured', 'computed'], 0)
    
    # Create and test a Profile object for this dataset.
    lab = get_profile_obj(nc, [], [])
    
    # Close down the pipes to the netCDF dataset files
    lab.nc.close()


def check_from_roms():
    """
    Test the ambient data methods on data read from ROMS.
    
    this unit test reads in a ROMS netCDF output file, extracts the profile
    information, and creates a new netCDF dataset and Profile class object
    for use by the TAMOC modeling suite.  
    
    TODO (S. Socolofsky 7/15/2013):  After fixing the octant.roms module to
    have monotonically increasing depth, try to reinstate this test by 
    changing the function name from check_from_roms() to test_from_roms().  
    I was also having problems with being allowed to use the THREDDS netCDF
    file with py.test.  I could run the test under ipython, but not under
    py.test.
    
    """
    # Get a path to a ROMS dataset on a THREDDS server
    nc_roms = 'http://barataria.tamu.edu:8080/thredds/dodsC/' + \
              'ROMS_Daily/08122012/ocean_his_08122012_24.nc'
    
    # Prepare the remaining inputs to the get_nc_db_from_roms() function
    # call
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    'output'))
    nc_file = os.path.join(__location__,'test_roms.nc')
    
    t_idx = 0
    j_idx = 400
    i_idx = 420
    chem_names = ['dye_01', 'dye_02']
    
    (nc, nc_roms) = ambient.get_nc_db_from_roms(nc_roms, nc_file, t_idx, 
        j_idx, i_idx, chem_names)
    
    # Check the data are inserted correctly from ROMS into the new netCDF
    # dataset
    assert nc.summary == 'ROMS Simulation Data'
    assert nc.sea_name == 'ROMS'
    assert nc.variables['z'][:].shape[0] == 51
    assert nc.variables['z'][0] == nc.variables['z'].valid_min
    assert nc.variables['z'][-1] == nc.variables['z'].valid_max
    assert_approx_equal(nc.variables['temperature'][0], 303.24728393554688, 
                        significant = 6)
    assert_approx_equal(nc.variables['salinity'][0], 36.157352447509766, 
                        significant = 6)
    assert_approx_equal(nc.variables['pressure'][0], 101325.0, 
                        significant = 6)
    assert_approx_equal(nc.variables['dye_01'][0], 3.4363944759034656e-22, 
                        significant = 6)
    assert_approx_equal(nc.variables['dye_02'][0], 8.8296093939330156e-21, 
                        significant = 6)
    assert_approx_equal(nc.variables['temperature'][-1], 290.7149658203125, 
                        significant = 6)
    assert_approx_equal(nc.variables['salinity'][-1], 35.829414367675781, 
                        significant = 6)
    assert_approx_equal(nc.variables['pressure'][-1], 3217586.2927573984, 
                        significant = 6)
    assert_approx_equal(nc.variables['dye_01'][-1], 8.7777050221856635e-22, 
                        significant = 6)
    assert_approx_equal(nc.variables['dye_02'][-1], 4.0334050451121613e-20, 
                        significant = 6)
    
    # Create a Profile object from this netCDF dataset and test the Profile
    # methods
    roms = get_profile_obj(nc, chem_names, ['kg/m^3', 'kg/m^3'])
    
    # Close the pipe to the netCDF dataset
    roms.nc.close()
    nc_roms.close()


def test_profile_deeper():
    """
    Test the methods to compute buoyancy_frequency and to extend a CTD profile
    to greater depths.  We just test the data from ctd_bm54.cnv since these
    methods are independent of the source of data.
    
    """
    # Make sure the netCDF file for the ctd_BM54.cnv is already created by 
    # running the test file that creates it.  
    test_from_ctd()
    
    # Get a Profile object from this dataset
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__), 
                                    'output'))
    nc_file = os.path.join(__location__,'test_BM54.nc')
    ctd = ambient.Profile(nc_file, chem_names=['oxygen'])
    
    # Compute the buoyancy frequency at 1500 m and verify that the result is 
    # correct
    N = ctd.buoyancy_frequency(1529.789, h=0.01)
    assert_approx_equal(N, 0.00061463758327116565, significant=6)
    
    # Record a few values to check after running the extension method
    T0, S0, P0, o20 = ctd.get_values(1000., ['temperature', 'salinity', 
                                     'pressure', 'oxygen'])
    z0 = ctd.nc.variables['z'][:]
    
    # Extend the profile to 2500 m
    nc_file = os.path.join(__location__,'test_BM54_deeper.nc')
    ctd.extend_profile_deeper(2500., nc_file)
    
    # Check if the original data is preserved
    T1, S1, P1, o21 = ctd.get_values(1000., ['temperature', 'salinity', 
                                     'pressure', 'oxygen'])
    z1 = ctd.nc.variables['z'][:]
    
    # Make sure the results are still right
    assert_approx_equal(T1, T0, significant=6)
    assert_approx_equal(S1, S0, significant=6)
    assert_approx_equal(P1, P0, significant=6)
    assert_approx_equal(o21, o20, significant=6)
    assert z1.shape[0] == z0.shape[0] + 50
    assert_array_almost_equal(z1[:-50], z0)
    assert z1[-1] == 2500.
    # Note that the buoyancy frequency shifts very slightly because density
    # is not linearly proportional to salinity.  Nonetheless, the results are
    # close to what we want, so this method of extending the profile works
    # adequately.
    N = ctd.buoyancy_frequency(1500.)
    assert_approx_equal(N, 0.0006377576016247663, significant=6)
    N = ctd.buoyancy_frequency(2500.)
    assert_approx_equal(N, 0.0006146292892002274, significant=6)

