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
import pygrib as pg
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
def bytescale(data_arr,vmin,vmax):
    assert(vmin < vmax)
    DataImage = np.round((data_arr - vmin) / (vmax - vmin) * 255.9999)
    DataImage[DataImage < 0] = 0
    DataImage[DataImage > 255] = 255
    return DataImage.astype(np.uint8)

#----------------------------------------------------------------------------------------------------
def unbytescale(scaled_arr,vmin,vmax):
    assert(vmin < vmax)
    scaled_arr = scaled_arr.astype(np.float32)
    unscaled_arr = scaled_arr / 255.9999 * (vmax - vmin) + vmin
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
    bsinfo['PhiDP'] = {'vmin':0, 'vmax':3600}             # degrees
    bsinfo['RhoHV'] = {'vmin':0, 'vmax':1}                # dimless
    bsinfo['SpectrumWidth'] = {'vmin':0, 'vmax':70}       # m/s
    bsinfo['Zdr'] = {'vmin':-4, 'vmax':6}                 # dB

    # Other vars
    #bsinfo['azimuth'] = {'vmin':0, 'vmax':360}            # degrees
    bsinfo['range'] = {'vmin':0, 'vmax': 460}             # km 

    return bsinfo
