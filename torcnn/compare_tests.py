import pandas as pd
import numpy as np
import pickle
import rad_utils
from datetime import datetime, timedelta
import sys,os
import matplotlib.pyplot as plt

evaldir='/raid/jcintineo/torcnn/eval/nospout2024/'

m1='test23'
m2='torp'

# Load labels and preds
m1_preds = np.load(f"{evaldir}/{m1}/predictions.npy")
m1_labs = np.load(f"{evaldir}{m1}/labels.npy")

m2_preds = np.load(f"{evaldir}/{m2}/predictions.npy")
m2_labs = np.load(f"{evaldir}{m2}/labels.npy")

# Load torp pickle
df = pd.read_pickle('/raid/jcintineo/torcnn/eval/nospout2024/torp_nospout2024.pkl')
df['m1_preds'] = m1_preds
df['m2_preds'] = m2_preds
df['m1-m2'] = np.round((m1_preds - m2_preds)*100).astype(int)
df['m2-m1'] = np.round((m2_preds - m1_preds)*100).astype(int)
df['labels'] = m1_labs

window_size = 64, 128 # these are half-sizes
varname = ['Reflectivity', 'Velocity'] #, 'RhoHV', 'AzShear'] #'RhoHV', 'Zdr', 'PhiDP', 'Velocity', 'SpectrumWidth', 'AzShear', 'DivShear']
dataroot = '/myrorss2/work/thea.sandmael/radar/' #20230405/KJL/netcdf/Velocity/00.50/%Y%m%d-%H%M%S.netcdf' 

# Best improved hits
df_tmp = df[df['labels'] == 1]
df_tmp = df_tmp.sort_values(by='m1-m2', ascending=False)
subtype='better_hits'
os.makedirs(f'{evaldir}/{subtype}/', exist_ok=True)
for idx in range(100):

    row = df_tmp.iloc[idx]
    prob_diff = str(row['m1-m2']).zfill(2)

    file_path = f'{dataroot}/{row.radarTimestamp[0:8]}/{row.radar}/netcdf/Velocity/00.50/{row.radarTimestamp}.netcdf'

    fig, radar = rad_utils.plot_ppi(file_path,
                                    varname=varname,
                                    Xlat=row.latitudeExtractCenter,
                                    Xlon=row.longitudeExtractCenter,
                                    window_size=window_size,
                                    rangemax=300,
                                    plot_segment=True,
    )
   
    # annotate
    textstr = f"{m1}: {int(row['m1_preds']*100)}%\n{m2}: {int(row['m2_preds']*100)}%\nTornado: {int(row.labels)}"
    
    # Define properties for the text box background
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  
    ax=plt.gca()  
    ax.text(0.8, 1.15, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left', bbox=props)
 
    if isinstance(varname, str):
        figname = f'{evaldir}/{subtype}/{prob_diff}_{radar}_{varname}_{os.path.basename(file_path).split(".")[0]}.png'
    else:
        figname = f'{evaldir}/{subtype}/{prob_diff}_{radar}_{len(varname)}panel_{os.path.basename(file_path).split(".")[0]}.png'
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    print(f'Saved {figname}')
    
    plt.close()

print('')

# Best improved FAs
df_tmp = df[df['labels'] == 0]
df_tmp = df_tmp.sort_values(by='m2-m1', ascending=False)
subtype='better_FAs'
os.makedirs(f'{evaldir}/{subtype}/', exist_ok=True)
for idx in range(100):

    row = df_tmp.iloc[idx]
    prob_diff = str(row['m2-m1']).zfill(2)

    file_path = f'{dataroot}/{row.radarTimestamp[0:8]}/{row.radar}/netcdf/Velocity/00.50/{row.radarTimestamp}.netcdf'

    fig, radar = rad_utils.plot_ppi(file_path,
                                    varname=varname,
                                    Xlat=row.latitude,
                                    Xlon=row.longitude,
                                    window_size=window_size,
                                    rangemax=300,
                                    plot_segment=True,
    )

    # annotate
    textstr = f"{m1}: {int(row['m1_preds']*100)}%\n{m2}: {int(row['m2_preds']*100)}%\nTornado: {int(row.labels)}"

    # Define properties for the text box background
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax=plt.gca()
    ax.text(0.8, 1.15, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left', bbox=props)

    if isinstance(varname, str):
        figname = f'{evaldir}/{subtype}/{prob_diff}_{radar}_{varname}_{os.path.basename(file_path).split(".")[0]}.png'
    else:
        figname = f'{evaldir}/{subtype}/{prob_diff}_{radar}_{len(varname)}panel_{os.path.basename(file_path).split(".")[0]}.png'
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    print(f'Saved {figname}')

    plt.close()

print('')

# Worst hits
df_tmp = df[df['labels'] == 1]
df_tmp = df_tmp.sort_values(by='m2-m1', ascending=False)
subtype='worse_hits'
os.makedirs(f'{evaldir}/{subtype}/', exist_ok=True)
for idx in range(100):

    row = df_tmp.iloc[idx]
    prob_diff = str(row['m2-m1']).zfill(2)

    file_path = f'{dataroot}/{row.radarTimestamp[0:8]}/{row.radar}/netcdf/Velocity/00.50/{row.radarTimestamp}.netcdf'

    fig, radar = rad_utils.plot_ppi(file_path,
                                    varname=varname,
                                    Xlat=row.latitudeExtractCenter,
                                    Xlon=row.longitudeExtractCenter,
                                    window_size=window_size,
                                    rangemax=300,
                                    plot_segment=True,
    )

    # annotate
    textstr = f"{m1}: {int(row['m1_preds']*100)}%\n{m2}: {int(row['m2_preds']*100)}%\nTornado: {int(row.labels)}"

    # Define properties for the text box background
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax=plt.gca()
    ax.text(0.8, 1.15, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left', bbox=props)

    if isinstance(varname, str):
        figname = f'{evaldir}/{subtype}/{prob_diff}_{radar}_{varname}_{os.path.basename(file_path).split(".")[0]}.png'
    else:
        figname = f'{evaldir}/{subtype}/{prob_diff}_{radar}_{len(varname)}panel_{os.path.basename(file_path).split(".")[0]}.png'
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    print(f'Saved {figname}')

    plt.close()

print('')

# Worst FAs
ind = np.where((m1_labs == 0) & (m1_preds - m2_preds >= 0.3))
df_tmp = df[df['labels'] == 0]
df_tmp = df_tmp.sort_values(by='m1-m2', ascending=False)
subtype='worse_FAs'
os.makedirs(f'{evaldir}/{subtype}/', exist_ok=True)
for idx in range(100):

    row = df_tmp.iloc[idx]
    prob_diff = str(row['m1-m2']).zfill(2)

    file_path = f'{dataroot}/{row.radarTimestamp[0:8]}/{row.radar}/netcdf/Velocity/00.50/{row.radarTimestamp}.netcdf'

    fig, radar = rad_utils.plot_ppi(file_path,
                                    varname=varname,
                                    Xlat=row.latitude,
                                    Xlon=row.longitude,
                                    window_size=window_size,
                                    rangemax=300,
                                    plot_segment=True,
    )

    # annotate
    textstr = f"{m1}: {int(row['m1_preds']*100)}%\n{m2}: {int(row['m2_preds']*100)}%\nTornado: {int(row.labels)}"

    # Define properties for the text box background
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax=plt.gca()
    ax.text(0.8, 1.15, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left', bbox=props)

    if isinstance(varname, str):
        figname = f'{evaldir}/{subtype}/{prob_diff}_{radar}_{varname}_{os.path.basename(file_path).split(".")[0]}.png'
    else:
        figname = f'{evaldir}/{subtype}/{prob_diff}_{radar}_{len(varname)}panel_{os.path.basename(file_path).split(".")[0]}.png'
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    print(f'Saved {figname}')

    plt.close()
