from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys,os
sys.path.append('/mnt/home/phi/localdata/PHI_Processing/generalServices/getCNN/Tor/')
import addCnnTorToDB as add2db
import glob


datadir = '/sas8tb/jcintineo/torcnn_output/products/'

radar_globs = glob.glob(f"{datadir}/K???")
radars = [os.path.basename(rg) for rg in radar_globs]
# == or ==
#radars = ['KVNX', 'KICT', 'KTLX', 'KLBB', 'KAMA', 'KFDR']

startdt = datetime(2026,3,16,12,0)
enddt = datetime(2026,3,17,12,0)


for radar in radars:
    print(radar)
    # super_list is a list of dicts.
    # Each dict is a row/record from a TORP object.
    super_list = []

    dt = startdt
    while startdt <= dt <= enddt:
        files = glob.glob(dt.strftime(f"{datadir}/{radar}/%Y/%Y%m%d/%Y%m%d-%H%M??_{radar}_0050_tordetections_torcnn.csv"))
        #print(dt.strftime(f"{datadir}/{radar}/%Y/%Y%m%d/%Y%m%d-%H%M??_{radar}_0050_tordetections_torcnn.csv"))
        if len(files):
            for fff in files:
                df = pd.read_csv(fff)
                if len(df) > 0:
                    super_list += df.to_dict(orient='records')

        dt += timedelta(minutes=1)
    
    print(len(super_list))
    if len(super_list):
        add2db.main(super_list)
