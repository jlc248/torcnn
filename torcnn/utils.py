import matplotlib.pyplot as plt
from subprocess import Popen,PIPE,call
from datetime import datetime,timedelta
import time
import os,sys,re
import smtplib
import getpass
from datetime import datetime
from netCDF4 import Dataset
import errno
import numpy as np
from numba import jit
import pickle
#from pyhdf.SD import SD, SDC
#----------------------------------------------------------------------------------------------------
@jit(nopython=True)
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

#----------------------------------------------------------------------------------------------------
def bytescale(data_arr, vmin, vmax, min_byte_val=0, max_byte_val=255):
    """
    Scales a data array to a specified byte range.

    Args:
        data_arr (np.ndarray): The input data array to be scaled.
        vmin (float): The minimum value of the original data range.
        vmax (float): The maximum value of the original data range.
        min_byte_val (int, optional): The minimum byte value for the scaled output. Defaults to 0.
        max_byte_val (int, optional): The maximum byte value for the scaled output. Defaults to 255.

    Returns:
        np.ndarray: The bytescaled array as np.uint8.
    """
    assert(vmin < vmax)
    assert(min_byte_val < max_byte_val)

    # Calculate the range of the target byte values
    byte_range = max_byte_val - min_byte_val

    # Scale the data to the new byte range
    # First, normalize the data to 0-1 range based on vmin/vmax
    # Then, scale to the desired byte_range and shift by min_byte_val
    DataImage = np.round(((data_arr - vmin) / (vmax - vmin)) * byte_range + min_byte_val)

    # Clip values to ensure they stay within the specified byte range
    DataImage[DataImage < min_byte_val] = min_byte_val
    DataImage[DataImage > max_byte_val] = max_byte_val

    return DataImage.astype(np.uint8)

#----------------------------------------------------------------------------------------------------
def unbytescale(scaled_arr, vmin, vmax, min_byte_val=0, max_byte_val=255):
    """
    Unscales a byte-scaled array back to its original data range.

    Args:
        scaled_arr (np.ndarray): The byte-scaled input array.
        vmin (float): The minimum value of the original data range.
        vmax (float): The maximum value of the original data range.
        min_byte_val (int, optional): The minimum byte value used during scaling. Defaults to 0.
        max_byte_val (int, optional): The maximum byte value used during scaling. Defaults to 255.

    Returns:
        np.ndarray: The unscaled array as np.float32.
    """
    assert(vmin < vmax)
    assert(min_byte_val < max_byte_val)

    scaled_arr = scaled_arr.astype(np.float32)

    # Calculate the range of the byte values
    byte_range = max_byte_val - min_byte_val

    # Unscale the array
    # First, shift back by min_byte_val and normalize to 0-1 based on byte_range
    # Then, scale back to the original (vmax - vmin) range and shift by vmin
    unscaled_arr = ((scaled_arr - min_byte_val) / byte_range) * (vmax - vmin) + vmin
    
    return unscaled_arr

#----------------------------------------------------------------------------------------------------
def get_bsinfo():
    """
    Returns dictionary of info for bytescaling variables
    """

    bsinfo = {}

    # L2 NEXRAD
    bsinfo['Reflectivity'] = {'vmin':5, 'vmax':70}        # dBZ
    bsinfo['Velocity'] = {'vmin':-100, 'vmax':100}        # m/s
    bsinfo['AliasedVelocity'] = {'vmin':-100, 'vmax':100} # m/s
    bsinfo['AzShear'] = {'vmin':-0.01, 'vmax':0.04}       # /s
    bsinfo['DivShear'] = {'vmin':-0.01, 'vmax':0.04}      # /s
    bsinfo['PhiDP'] = {'vmin':0, 'vmax':360}             # degrees
    bsinfo['RhoHV'] = {'vmin':0, 'vmax':1}                # dimless
    bsinfo['SpectrumWidth'] = {'vmin':0, 'vmax':70}       # m/s
    bsinfo['Zdr'] = {'vmin':-4, 'vmax':6}                 # dB

    # Other vars
    bsinfo['azimuth'] = {'vmin':0, 'vmax':360}            # degrees
    bsinfo['range'] = {'vmin':0, 'vmax': 460}             # km 

    return bsinfo
