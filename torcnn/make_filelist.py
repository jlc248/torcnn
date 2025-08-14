import pandas as pd
import glob
import sys, os
import numpy as np
import pickle

# bad times in 2023
#2023-04-29 00:00:13 is too far from 2023-04-29 01:26:47. KEWX - Reflectivity
#2023-05-10 00:00:09 is too far from 2023-05-10 00:23:48. KDDC - Reflectivity

"""
This script will create a colated list of tfrecs and corresponding 
DataFrame of samples that is read for TORP predictions (see torp/predict.py).
"""


df = pd.read_csv('/raid/jcintineo/torcnn/torp_datasets/2023_Storm_Reports_Expanded_tilt0050_radar_r2500_nodup.csv')
df = df[df.spout == 0]

outdir = '/raid/jcintineo/torcnn/eval/nospout_2023/'
outpickle_name = f'{outdir}/torp_2023_nospout_cleaned.pkl'


# Dropping some bad rows
#mask_to_remove = ((df['radarTimestamp'] == '20230429-000013') & (df['radar'] == 'KEWX')) | ((df['radarTimestamp'] == '20230510-000009') & (df['radar'] == 'KDDC'))
#    # the bad indices in 2023 were: [11570, 14466]

#indices_to_remove = df.index[mask_to_remove].tolist()
#print(indices_to_remove)
#sys.exit()

    
tfrec_list = []
indices_to_drop = []

print('len(df):',len(df))

for row in df.itertuples():
    if row.tornado:
        ttype = 'tor'
    else:
        ttype = 'nontor'
    #neglecting spouts for now

    expected_file = f'/raid/jcintineo/torcnn/tfrecs/{row.year}/{row.radarTimestamp[0:8]}/{ttype}/{row.radar}_{np.round(row.latitude,2)}_{np.round(row.longitude,2)}_{row.radarTimestamp}.tfrec'

    if os.path.isfile(expected_file):
        tfrec_list.append(expected_file)
    else:
       indices_to_drop.append(row.Index)
       print(f'{expected_file} does not exist.')

print(len(indices_to_drop))
print(len(tfrec_list))
print(len(indices_to_drop)/len(tfrec_list))
#sys.exit()
df_new = df.drop(indices_to_drop)
df_new.to_pickle(outpickle_name)

pickle.dump(tfrec_list, open(f'{outdir}/colated_filelist.pkl', 'wb'))
print(f'Saved {outdir}/colated_filelist.pkl')
