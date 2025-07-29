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
    bsinfo['PhiDP'] = {'vmin':0, 'vmax':360}              # degrees
    bsinfo['RhoHV'] = {'vmin':0, 'vmax':1}                # dimless
    bsinfo['SpectrumWidth'] = {'vmin':0, 'vmax':70}       # m/s
    bsinfo['Zdr'] = {'vmin':-4, 'vmax':6}                 # dB

    # Other vars
    bsinfo['azimuth'] = {'vmin':0, 'vmax':360}            # degrees
    bsinfo['rangeKm'] = {'vmin':0, 'vmax': 460}           # km 

    return bsinfo

#---------------------------------------------------------------------------------------------------
def training_history_figs(history, outdir=None):

    """
    Make some figures of metrics during training.

    Inputs:
    - history: pickle file (str) or dictionary that contains the scores
    - outdir (optional): string for the output directory. Default is to use
      os.path.dirname(history)
    """

    if isinstance(history, str): # Assume it's a pickle file
        if outdir is None: outdir = os.path.dirname(history)
        tmphist = pickle.load(open(history, 'rb'))
        history = tmphist

    nepochs = len(history['loss'])
    xvals = np.arange(nepochs)+1

    t1, = plt.plot(xvals, history['loss'], color='blue', linestyle='-', label='Train loss')
    v1, = plt.plot(xvals, history['val_loss'], color='orange', linestyle='-', label='Val loss')
    plt.title('Training and validation loss')
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(outdir,'loss_history.png'))
    plt.close()


    # Find the number of indexes or outputs there are
    all_idxs = []
    for key in history.keys():
        if '_index' in key:
            idx = key.split('_index')[1]
            if idx not in all_idxs:
                all_idxs.append(idx)

    # For each output, find best CSI and best threshold at each epoch
    for idx in all_idxs:
        # Training images

        best_csi_by_epoch = []
        far_by_epoch = [] # as best csi
        pod_by_epoch = [] # at best csi
        best_prob_by_epoch = []
        for ii in range(nepochs):
            best = 0; prob = 0
            for key in history.keys():
                if key.startswith('csi') and key.endswith(f'index{idx}'):
                    if history[key][ii] > best:
                        best = history[key][ii]
                        far = history[f'far{key[3:5]}_index{idx}'][ii]
                        pod = history[f'pod{key[3:5]}_index{idx}'][ii]
                        prob = round(int(key[3:5])/100, 2)
            best_csi_by_epoch.append(best)
            best_prob_by_epoch.append(prob)
            far_by_epoch.append(far)
            pod_by_epoch.append(pod)

        t1, = plt.plot(xvals, pod_by_epoch, color='blue', linestyle='--', label='POD')
        t2, = plt.plot(xvals, far_by_epoch, color='blue', linestyle=':', label='FAR')
        t3, = plt.plot(xvals, best_csi_by_epoch, color='blue', linestyle='-', label='CSI')
        t4, = plt.plot(xvals, history[f'auprc_index{idx}'], color='blue', linestyle='-', linewidth=3, label='AUPRC')
        t5, = plt.plot(xvals, history[f'brier_score_index{idx}'], color='blue', linestyle='-.', label='Brier Score')
        t6, = plt.plot(xvals, best_prob_by_epoch, color='cyan', linestyle='-', label='best prob.')
        plt.title(f'Training scores index{idx}')
        plt.ylabel('Score')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(outdir,f'training_history_index{idx}.png'))
        plt.close()

        # Validation images

        best_csi_by_epoch = []
        far_by_epoch = [] # as best csi
        pod_by_epoch = [] # at best csi
        best_prob_by_epoch = []
        for ii in range(nepochs):
            best = 0; prob = 0
            for key in history.keys():
                if key.startswith('val_csi') and key.endswith(f'index{idx}'):
                    if history[key][ii] > best:
                        best = history[key][ii]
                        far = history[f'val_far{key[7:9]}_index{idx}'][ii]
                        pod = history[f'val_pod{key[7:9]}_index{idx}'][ii]
                        prob = round(int(key[7:9])/100, 2)
            best_csi_by_epoch.append(best)
            best_prob_by_epoch.append(prob)
            far_by_epoch.append(far)
            pod_by_epoch.append(pod)

        t1, = plt.plot(xvals, pod_by_epoch, color='red', linestyle='--', label='POD')
        t2, = plt.plot(xvals, far_by_epoch, color='red', linestyle=':', label='FAR')
        t3, = plt.plot(xvals, best_csi_by_epoch, color='red', linestyle='-', label='CSI')
        t4, = plt.plot(xvals, history[f'val_auprc_index{idx}'], color='red', linestyle='-', linewidth=3, label='AUPRC')
        t5, = plt.plot(xvals, history[f'val_brier_score_index{idx}'], color='red', linestyle='-.', label='Brier Score')
        t6, = plt.plot(xvals, best_prob_by_epoch, color='orange', linestyle='-', label='best prob.')
        plt.title(f'Validation scores index{idx}')
        plt.ylabel('Score')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(outdir,f'val_history_index{idx}.png'))
        plt.close()

#----------------------------------------------------------------------------------------------------------

def add_scores_to_file(outdir):
    """
    Will append to {outdir}/verification_scores.txt or create it if it does not exist.
    If scores['score']['best_score'] = 0, then "higher_is_better" = True.
    If scores['score']['best_score'] = 1, then "higher_is_better" = False (i.e., lower is better).
    """

    of = open(f"{outdir}/log.csv", "r")
    lines = of.readlines()
    of.close()

    scores = {}

    for ii, line in enumerate(lines):
        parts = line.split(",")
        if ii == 0:  # first line
            for score in parts:
                if score.startswith('val_'):
                    scores[score] = {'idx': parts.index(score)}
                    scores[score]["best_epoch"] = 0
                    if 'csi' in score or 'pod' in score or 'aupr' in score or 'accuracy' in score:
                        scores[score]["higher_is_better"] = True
                        scores[score]["best_score"] = 0
                    elif 'far' in score or 'brier_score' in score or 'loss' in score:
                        scores[score]["higher_is_better"] = False
                        scores[score]["best_score"] = 1
        else:  # All other lines
            for score in scores:
                idx = scores[score]["idx"]
                best_val_score = scores[score]["best_score"]
                val_score = float(parts[idx])  # the validation metric for this epoch
                if scores[score]["higher_is_better"]:
                    if val_score > best_val_score:
                        scores[score]["best_score"] = val_score
                        scores[score]["best_epoch"] = int(parts[0]) + 1
                else:  # lower is better
                    if val_score < best_val_score:
                        scores[score]["best_score"] = val_score
                        scores[score]["best_epoch"] = int(parts[0]) + 1

    of = open(f"{outdir}/verification_scores.txt", "a")
    for score in scores:
        best_score = scores[score]["best_score"]
        best_epoch = scores[score]["best_epoch"]
        of.write(f"Best {score}: {np.round(best_score,5)}; at epoch {best_epoch}\n")
    of.close()

#------------------------------------------------------------------------------------------------------
def read_data(file, var_name, atts=None, global_atts=None,nogzip=False,startX=None,endX=None,startY=None,endY=None):

  gzip=0
  if(file[-3:] == '.gz'):
    if(not(nogzip)): gzip=1
    unzipped_file = file[0:len(file)-3]
    the_file = unzipped_file
    s = call(['gunzip','-f',file])
  else:
    gzip=0
    the_file = file

  try:
    openFile = Dataset(the_file,'r')
  except (OSError,IOError) as err:
    logging.error(str(err))
    return -1

  if(startX is None):
    data = openFile.variables[var_name][:]
  else:
    data = openFile.variables[var_name][startY:endY,startX:endX]

  #get variable attributes
  if(isinstance(atts,dict)):
    for att in atts:
      if(att in openFile.variables[var_name].ncattrs()):
        atts[att] = getattr(openFile.variables[var_name],att) #atts is modified and returned
  #get global attributes
  if(isinstance(global_atts,dict)):
    for gatt in global_atts:
      if(gatt in openFile.ncattrs()):
        global_atts[gatt] = getattr(openFile,gatt) #global_atts is modified and returned
  openFile.close()

  # gzip file after closing?
  if gzip:
    s = Popen(['gzip','-f',unzipped_file]) #==will run in background since it doesn't need output from command.

  return data

#--------------------------------------------------------------------------------------------------------
def read_netcdf(file, datasets, global_atts=None,startY=None,endY=None,startX=None,endX=None):
  #datasets is a dict, formatted thusly:
  #datasets = {'varname1':{'atts':['attname1','attname2']}, 'varname2':{'atts':[]}}
  #atts are variable attributes. If 'atts':['ALL'], then it will grab all variable attributes.
  #global_atts are file global attributes.
  #Each key/var in datasets is returned with a 'data' key...which contains a numpy array
  #startY,etc. are the start and end points to grab the data from from each varname.
  #Default is to get full data from each varname.

  if(file[-3:] == '.gz'):
    gzip=1
    unzipped_file = file[0:len(file)-3]
    the_file = unzipped_file
    s = call(['gunzip','-f',file])
  else:
    gzip=0
    the_file = file

  try:
    openFile = Dataset(the_file,'r')
  except (OSError,IOError) as err:
    raise

  for key in datasets:
    if(startY is None):
      datasets[key]['data'] = openFile.variables[key][:]
    else:
      datasets[key]['data'] = openFile.variables[key][startY:endY,startX:endX]
    #get each variable attribute
    if(datasets[key]['atts'] == ['ALL']):
      datasets[key]['atts'] = {}
      for att in openFile.variables[key].ncattrs():
        datasets[key]['atts'][att] = getattr(openFile.variables[key],att)
    #get select variable attributes
    else:
      tmpdict = {}
      for att in datasets[key]['atts']:
        if(att in openFile.variables[key].ncattrs()):
          tmpdict[att] = getattr(openFile.variables[key],att)
      datasets[key]['atts'] = tmpdict
  #get global attributes
  if(isinstance(global_atts,dict)):
    for gatt in global_atts:
      if(gatt in openFile.ncattrs()):
        global_atts[gatt] = getattr(openFile,gatt) #global_atts is modified and returned
  openFile.close()

  # gzip file after closing?
  if gzip:
      #s = Popen(['gzip','-f',unzipped_file]) #==will run in background since it doesn't need output from command.
      s = call(['gzip','-f',unzipped_file])

#-----------------------------------------------------------------------------------------------
def read_grib2(infile,gpd_info,constants,startX=None,endX=None,startY=None,endY=None):
  #logging.info('Process started. Received {infile}')

  #now read grib2
  grb = pg.open(infile)
  msg = grb[1]  #should only be 1 'message' in grib2 file
  data = msg.values
  #make file lats/lons. Could pull from grib, e.g.: lats = grb[1]['latitudes'] ; but making them is faster than pulling from grib.
  grb.close()
  orig_nlat,orig_nlon = np.shape(data)
  origNWlat = constants['MRMS']['origNWlat'] #usu. 55.0
  origNWlon = constants['MRMS']['origNWlon'] #usu. -130.0

  #gpd_info = read_gpd_file(gpd)
  NW_lat = gpd_info['NW_lat']
  NW_lon = gpd_info['NW_lon']
  nlon = gpd_info['nlon']
  nlat = gpd_info['nlat']
  dy = gpd_info['dy']
  dx = gpd_info['dx']

  #This is mainly for the azshears; everything is relative to 0.01 res, whereas the azshears are 0.005 degree res.
  if data.shape == (7000,14000):
    data = data[::2,::2]
    #data = max_pooling_2x2(data)

  #just cut out the part we want
  offset_from_origNWlat = int((origNWlat - NW_lat)/dy)
  offset_from_origNWlon = int((NW_lon - origNWlon)/dx)
  new_data = (data[offset_from_origNWlat:(offset_from_origNWlat + nlat),offset_from_origNWlon:(offset_from_origNWlon + nlon)]).astype(np.float32)
  if(startX is not None): #crop
    new_data = new_data[startY:endY,startX:endX]

  #logging.info('Got new data. Process ended.')
  return new_data
#####################################################################################################
def write_netcdf(output_file,datasets,dims,atts={},gzip=False,wait=False,**kwargs):
  logging.info('Process started')

  os.makedirs(os.path.dirname(output_file), exist_ok=True)
  ncfile = Dataset(output_file,'w')#,format='NETCDF3_CLASSIC')

  #dimensions
  for dim in dims:
    ncfile.createDimension(dim,dims[dim])
  #variables

  for varname in datasets:
    if(isinstance(datasets[varname]['data'],np.ndarray)):
      dtype = str((datasets[varname]['data']).dtype)
    elif(isinstance(datasets[varname]['data'],int) or isinstance(datasets[varname]['data'],np.int32) or isinstance(datasets[varname]['data'],np.int16)):
      dtype = 'i'
    elif(isinstance(datasets[varname]['data'],float) or isinstance(datasets[varname]['data'],np.float32) or isinstance(datasets[varname]['data'],np.float16)):
      dtype = 'f'
    elif isinstance(datasets[varname]['data'],np.uint8):
      dtype = 'u1'
    elif isinstance(datasets[varname]['data'],np.int8):
      dtype = 'i1'

    if('_FillValue' in datasets[varname]['atts']):
      dat = ncfile.createVariable(varname,dtype,datasets[varname]['dims'],fill_value=datasets[varname]['atts']['_FillValue'])
    else:
      dat = ncfile.createVariable(varname,dtype,datasets[varname]['dims'])
    dat[:] = datasets[varname]['data']
    #variable attributes
    if('atts' in datasets[varname]):
      for att in datasets[varname]['atts']:
        if(att != '_FillValue'): dat.__setattr__(att,datasets[varname]['atts'][att]) #_FillValue is made in 'createVariable'
  #global attributes
  for key in atts:
    ncfile.__setattr__(key,atts[key])
  ncfile.close()
  logging.info(f'Wrote out {output_file}')

  if(gzip):
    if(wait):
      s = call(['gzip','-f',output_file]) #==wait to finish
    else:
      s = Popen(['gzip','-f',output_file]) #==will run in background

  logging.info('Process ended')
  return True
