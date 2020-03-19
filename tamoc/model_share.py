"""
Model Share
===========

Provides functions that are common to all models in the TAMOC suite

This module defines functions that perform standard tasks for the modeling
modules in the TAMOC suite.  These include standard authoring of output 
files and reading ambient data from output files.  

Notes 
----- 
These class objects and helper functions may be used throughout the TAMOC 
modeling suite.

"""
# S. Socolofsky, November 2014, Texas A&M University <socolofs@tamu.edu>.

from __future__ import (absolute_import, division, print_function)

from tamoc import ambient

from netCDF4 import Dataset
from datetime import datetime
import os
from warnings import warn


# ----------------------------------------------------------------------------
# Functions to work with netCDF datasets of model output
# ----------------------------------------------------------------------------


def tamoc_nc_file(fname, title, summary, source):
    """
    Write the header meta data to an netCDF file for a TAMOC output
    
    The TAMOC suite stores its output by detaul in a netCDF dataset file.  
    This function writes the standard TAMOC metadata to the header of the 
    netCDF file.  
    
    Parameters
    ----------
    fname : str
        File name of the file to write
    title:  str
        String stating the TAMOC module where the data originated and the 
        type of data contained.  
    summary : str
        String summarizing what is contained in the dataset or information
        needed to interpret the dataset
    source : str
        String describing the source of the data in the dataset or of related
        datasets
    
    Returns
    -------
    nc : `netCDF4.Dataset` object
        The `netCDF4.Dataset` object containing the open netCDF4 file where
        the data should be stored.
    
    """
    
    # Create the netCDF dataset object
    nc = Dataset(fname, 'w', format='NETCDF4_CLASSIC')
    
    # Write the netCDF header data for a TAMOC suite output
    nc.Conventions = 'TAMOC Modeling Suite Output File'
    nc.Metadata_Conventions = 'TAMOC Python Model'
    nc.featureType = 'profile'
    nc.cdm_data_type = 'Profile'
    nc.nodc_template_version = \
        'NODC_NetCDF_Profile_Orthogonal_Template_v1.0'
    nc.title = title
    nc.summary = summary
    nc.source = source
    nc.creator_url = 'http://github.com/socolofs/tamoc'
    nc.date_created = datetime.today().isoformat(' ')
    nc.date_modified = datetime.today().isoformat(' ')
    nc.history = 'Creation'
    
    # Return the netCDF dataset
    return nc


def profile_from_model_savefile(nc, fname, ctdname=None):
    """
    Load the `ambient.Profile` data pointed to by the netCDF file
    
    Each netCDF model save file for the TAMOC suite models stores the file
    name and path to the ambient CTD data used in the simulation.  This 
    prevents the save files from having to encapsulate all of the ambient 
    CTD data with results for every simulation run.  This block of code
    looks for the ambient data and loads it into memory if found.
    
    Parameters
    ----------
    nc : `netCDF4.Dataset` object
        The netCDF dataset object containing the saved model simulation
    
    fname : str
        File name of the netCDF file.  This is used to get the complete
        path to the current directory so that the Profile data can be found.
    
    Returns
    -------
    profile : `ambient.Profile` object
        The profile data as an `ambient.Profile` object.
    
    See Also
    --------
    single_bubble_model.Model.save_sim, stratified_plume_model.Mode.save_sim,
    bent_plume_model.Model.save_sim
    
    Notes
    -----
    If the profile data are not found, a message is echoed to the command
    line.  No other warnings or errors are thrown.
    
    """
    try:
        # Try to locate and load in the profile data
        nc_path = os.path.normpath(os.path.join(os.getcwd(), \
                                   os.path.dirname(fname)))
        if ctdname is not None:
            prf_path = os.path.normpath(os.path.join(nc_path, ctdname))
        else:
            prf_path = os.path.normpath(os.path.join(nc_path, nc.summary))
        amb_data = Dataset(prf_path)
        profile = ambient.Profile(amb_data, chem_names='all')
        profile.close_nc()
        
    except RuntimeError:
        # Tell user that profile data read failed
        message = ['File not found: %s' % prf_path]
        message.append(' ... Continuing without profile data')
        warn(''.join(message))
        profile = None
    
    # Send back the final profile object
    return profile

    