import pandas as pd
import glob
import sys, os
import numpy as np
import pickle

# bad times in 2023
#2023-04-29 00:00:13 is too far from 2023-04-29 01:26:47. KEWX - Reflectivity
#2023-05-10 00:00:09 is too far from 2023-05-10 00:23:48. KDDC - Reflectivity

"""
This script will create a colated DataFrame including tfrecs and corresponding 
samples that is read for TORP predictions (see torp/predict.py).
"""


df1 = pd.read_csv('/raid/jcintineo/torcnn/torp_datasets/2023_Storm_Reports_Expanded_tilt0050_radar_r2500_nodup.csv')
#tmp = df1[(df1.radar == 'KMRX') & (df1.radarTimestamp == '20230807-174942')]
#print(tmp.tornadoesWithin100kmAnd2hr)
#sys.exit()
# Removing spouts
df1 = df1[df1.spout == 0]

ndf1 = len(df1)
df1['preTornado'] = np.zeros(ndf1)
df1['minPreTornado'] = np.full(ndf1, -1)

df2 = pd.read_csv('/raid/jcintineo/torcnn/torp_datasets/2018_pretornadic_expanded_tilt0050_radar_r2500_nodup.csv')
# Rename some columns to match df1
df2.rename(columns={'longitudeAzShearMax': 'longitudeExtractCenter',
                   'latitudeAzShearMax': 'latitudeExtractCenter',
                   'rng_int': 'RangeInterval'}, inplace=True)

common_cols = set(list(df1)) & set(list(df2))
common_cols = list(common_cols)
common_cols.sort()
#for cc in common_cols:
#    print(cc)
#sys.exit()

# Drop the columns in df1 and df2 that we don't want
columns_to_drop_df1 = df1.columns.difference(common_cols)
df1.drop(columns=columns_to_drop_df1, inplace=True)
columns_to_drop_df2 = df2.columns.difference(common_cols)
df2.drop(columns=columns_to_drop_df2, inplace=True)

# Combine dfs
df = pd.concat([df1, df2], ignore_index=True)

outdir = '/raid/jcintineo/torcnn/eval/nospout2023_pretor2018/'
outpickle_name = f'{outdir}/torp_nospout2023_pretor2018.pkl'
os.makedirs(outdir, exist_ok=True)

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
        indices_to_drop.append(row.Index)
        continue
    else:
        # NB the tfrecs have tornado=1 for pretor cases, whereas the TORP csvs have tornado=0
        if row.preTornado:
            ceil_minPreTor = int(np.ceil(row.minPreTornado / 15) * 15)
            if ceil_minPreTor == 0:
                ceil_minPreTor = 15
            elif ceil_minPreTor > 60:
                ceil_minPreTor = 120
            ttype = f'pretor_{ceil_minPreTor}'
        else:
            ttype = 'nontor'
    # neglecting spouts for now

    expected_file = f'/raid/jcintineo/torcnn/tfrecs_100km1hr/{row.year}/{row.radarTimestamp[0:8]}/{ttype}/{row.radar}_{np.round(row.latitude,2)}_{np.round(row.longitude,2)}_{row.radarTimestamp}.tfrec'

    if os.path.isfile(expected_file):
        tfrec_list.append(expected_file)
    else:
       indices_to_drop.append(row.Index)
       print(f'{expected_file} does not exist.')

print('number of indices_to_drop:', len(indices_to_drop))
print('number of tfrecs:', len(tfrec_list))
print('dropped ratio:',len(indices_to_drop)/len(df))
#sys.exit()
df_new = df.drop(indices_to_drop)

# Add new column 'tfrec'
df_new['tfrec'] = tfrec_list
df_new.to_pickle(outpickle_name)
print(f'Saved {outpickle_name}')
df_new.to_csv(f"{outpickle_name.split('.pkl')[0]}.csv")
print(f'Saved {outpickle_name.split(".pkl")[0]}.csv')

splice_leadtimes=True
if splice_leadtimes:
    print('saving spliced leadtime pkls and csvs')

    subtype='15min'
    os.makedirs(f'{outdir}/{subtype}', exist_ok=True)
    df15 = df_new[((df_new.preTornado == 1) & (df_new.minPreTornado <= 15)) | (df_new.preTornado == 0)]
    df15.to_pickle(f'{outdir}/{subtype}/{os.path.basename(outpickle_name).split(".pkl")[0]}-{subtype}.pkl')
    df15.to_csv(f'{outdir}/{subtype}/{os.path.basename(outpickle_name).split(".pkl")[0]}-{subtype}.csv')
    
    subtype='30min'
    os.makedirs(f'{outdir}/{subtype}', exist_ok=True)
    df30 = df_new[((df_new.preTornado == 1) & (df_new.minPreTornado > 15) & (df_new.minPreTornado <= 30)) | (df_new.preTornado == 0)]
    df30.to_pickle(f'{outdir}/{subtype}/{os.path.basename(outpickle_name).split(".pkl")[0]}-{subtype}.pkl')
    df30.to_csv(f'{outdir}/{subtype}/{os.path.basename(outpickle_name).split(".pkl")[0]}-{subtype}.csv')

    subtype='45min'
    os.makedirs(f'{outdir}/{subtype}', exist_ok=True)
    df45 = df_new[((df_new.preTornado == 1) & (df_new.minPreTornado > 30) & (df_new.minPreTornado <= 45)) | (df_new.preTornado == 0)]
    df45.to_pickle(f'{outdir}/{subtype}/{os.path.basename(outpickle_name).split(".pkl")[0]}-{subtype}.pkl')
    df45.to_csv(f'{outdir}/{subtype}/{os.path.basename(outpickle_name).split(".pkl")[0]}-{subtype}.csv')

    subtype='60min'
    os.makedirs(f'{outdir}/{subtype}', exist_ok=True)
    df60 = df_new[((df_new.preTornado == 1) & (df_new.minPreTornado > 45) & (df_new.minPreTornado <= 60)) | (df_new.preTornado == 0)]
    df60.to_pickle(f'{outdir}/{subtype}/{os.path.basename(outpickle_name).split(".pkl")[0]}-{subtype}.pkl')
    df60.to_csv(f'{outdir}/{subtype}/{os.path.basename(outpickle_name).split(".pkl")[0]}-{subtype}.csv')
