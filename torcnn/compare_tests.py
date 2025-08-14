import pandas as pd
import numpy as np
import pickle
import rad_utils
from datetime import datetime, timedelta
import sys,os
import matplotlib.pyplot as plt

evaldir='/raid/jcintineo/torcnn/eval/nospout_2023/'

m1='test15'
m2='torp'

# Load labels and preds
m1_preds = np.load(f"{evaldir}/{m1}/predictions.npy")
m1_labs = np.load(f"{evaldir}{m1}/labels.npy")

m2_preds = np.load(f"{evaldir}/{m2}/predictions.npy")
m2_labs = np.load(f"{evaldir}{m2}/labels.npy")

# Load torp pickle
df = pd.read_pickle('/raid/jcintineo/torcnn/eval/nospout_2023/torp_2023_nospout_cleaned.pkl')

window_size = 64, 128 # these are half-sizes
varname = ['Reflectivity', 'Velocity'] #, 'RhoHV', 'AzShear'] #'RhoHV', 'Zdr', 'PhiDP', 'Velocity', 'SpectrumWidth', 'AzShear', 'DivShear']
dataroot = '/work/thea.sandmael/radar/' #20230405/KJL/netcdf/Velocity/00.50/%Y%m%d-%H%M%S.netcdf' 

# Improved hits
ind = np.where((m1_labs == 1) & (m1_preds - m2_preds >= 0.5))
os.makedirs(f'{evaldir}/hits/', exist_ok=True)
for idx in ind[0]:

    row = df.iloc[idx]

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
    textstr = f"{m1}: {int(m1_preds[idx]*100)}%\n{m2}: {int(m2_preds[idx]*100)}%\nTornado: {int(m1_labs[idx])}"
    
    # Define properties for the text box background
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  
    ax=plt.gca()  
    ax.text(0.8, 1.15, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left', bbox=props)
 
    if isinstance(varname, str):
        figname = f'{evaldir}/hits/{radar}_{varname}_{os.path.basename(file_path).split(".")[0]}.png'
    else:
        figname = f'{evaldir}/hits/{radar}_{len(varname)}panel_{os.path.basename(file_path).split(".")[0]}.png'
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    print(f'Saved {figname}')
    
    plt.close()

print('')

# Improved FAs
ind = np.where((m1_labs == 0) & (m2_preds - m1_preds >= 0.8))
os.makedirs(f'{evaldir}/FAs/', exist_ok=True)
for idx in ind[0]:

    row = df.iloc[idx]

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
    textstr = f"{m1}: {int(m1_preds[idx]*100)}%\n{m2}: {int(m2_preds[idx]*100)}%\nTornado: {int(m1_labs[idx])}"

    # Define properties for the text box background
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax=plt.gca()
    ax.text(0.8, 1.15, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left', bbox=props)

    if isinstance(varname, str):
        figname = f'{evaldir}/FAs/{radar}_{varname}_{os.path.basename(file_path).split(".")[0]}.png'
    else:
        figname = f'{evaldir}/FAs/{radar}_{len(varname)}panel_{os.path.basename(file_path).split(".")[0]}.png'
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    print(f'Saved {figname}')

    plt.close()
