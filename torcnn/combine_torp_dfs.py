import pandas as pd
import sys
import glob
import numpy as np

torp_folder = "/work2/jcintineo/TORP/"
rep_folder = "/home/john.cintineo/StormReportsExpanded_20251107/"
pretor_folder = "/home/john.cintineo/pretor_manual/"
outdir = "/work2/jcintineo/TORP/combined_torp_rep_pretor/"

for year in ['2020', '2021', '2022', '2023', '2024']:
    files_to_combine1 = sorted(glob.glob(f"{torp_folder}/{year}/{year}*Info.csv"))
    rep_file = glob.glob(f"{rep_folder}/{year}*Storm*nodup.csv")
    assert(len(rep_file) == 1)
    df2 = pd.read_csv(rep_file[0])
    pretor_file = glob.glob(f"{pretor_folder}/{year}*pretornadic*nodup.csv")
    try:
        assert(len(pretor_file) == 1)
    except AssertionError:
        print("WARNING: pretor file doesn't exist")
        df3 = None
    except:
        df3 = pd.read_csv(pretor_file[0])

    # Combine all of the TORP detect files
    df_list = [pd.read_csv(f) for f in files_to_combine1]
    df1 = pd.concat(df_list, ignore_index=True)

    # Add fill or overwrite df1 and df2 pretor with 0s.
    # We will only use the manual pretor rows.
    df1['pretor'] = 0
    df1['pretorMinutes'] = -1
    df2['pretor'] = 0
    df2['pretorMinutes'] = -1

    # Add some fill columns in the pretor data
    if df3:
        df3['tornado'] = 1
        df3['hail'] = 0
        df3['wind'] = 0
        df3['spout'] = 0
        df3['magtornado'] = -1
        df3['maghail'] = -1
        df3['magwind'] = -1
 
        # Rename columns to conform to TORP folder 
        df3 = df3.rename(columns={
            'radarTimestamp': 'Time',
            'radar': 'Radar',
            'latitudeExtractCenter': 'Lat',
            'longitudeExtractCenter': 'Lon',
            'rangeKm': 'RangeKm',
            'elevationExtractCenter': 'Elev',
            'Velocity_MedianFiltered_max': 'Velocity_MedianFiltered_absmax',
            'Velocity_MedianFiltered_min': 'Velocity_MedianFiltered_absmin',
            'preTornado': 'pretor',
            'minPreTornado': 'pretorMinutes',
        })

    # Rename columns to conform to TORP folder
    df2 = df2.rename(columns={
        'radarTimestamp': 'Time',
        'radar': 'Radar',
        'latitudeExtractCenter': 'Lat',
        'longitudeExtractCenter': 'Lon',
        'rangeKm': 'RangeKm',
        'elevationExtractCenter': 'Elev',
        'Velocity_MedianFiltered_max': 'Velocity_MedianFiltered_absmax',
        'Velocity_MedianFiltered_min': 'Velocity_MedianFiltered_absmin',
    })
    
    # Create magntidue columns
    df2['magtornado'] = np.where(df2['tornado'] == 1, df2['magnitude'], 0)
    # Make everything float except "U" tornadoes. Keep those as is.
    converted = pd.to_numeric(df2['magtornado'], errors='coerce')
    df2['magtornado'] = np.where(pd.isna(converted), df2['magnitude'], converted) 
    df2['maghail'] = np.where(df2['hail'] == 1, df2['magnitude'], 0).astype(float)
    df2['magwind'] = np.where(df2['wind'] == 1, df2['magnitude'], 0).astype(float)
    
    ## Warning types (double checking with "warned")
    #df2['severeWarned'] = (df2['warningType'].str.contains('SV', na=False) & (df2['warned'] == 1)).astype(int)
    #df2['marineWarned'] = (df2['warningType'].str.contains('MA', na=False) & (df2['warned'] == 1)).astype(int)
    #df2['tornadoWarned'] = (df2['warningType'].str.contains('TO', na=False) & (df2['warned'] == 1)).astype(int)

    # Get common columns
    #cols1=set(df1.columns)
    #cols2=set(df2.columns)
    #cols3=set(df3.columns)

    #common_cols = sorted(list(cols1.intersection(cols2, cols3)))
    #print(common_cols)

    if df3 is None:
        df_combined = pd.concat([df1, df2], join="inner", ignore_index=True)
        df_combined.to_csv(f"{outdir}/{year}_combined_no-pretor.csv")
    else:
        df_combined = pd.concat([df1, df2, df3], join="inner", ignore_index=True)
        df_combined.to_csv(f"{outdir}/{year}_combined.csv")


#unique_to_df1 = sorted(list(cols1 - cols2))
#unique_to_df2 = sorted(list(cols2 - cols1))
#unique_to_df3 = sorted(list(cols3-cols2))
#print(f"Common: {common_cols}")
#print('')
#print(f"Unique to df1: {unique_to_df1}")
#print('')
#print(f"Unique to df2: {unique_to_df2}")
#print('')
#print(f"Unique to df3: {unique_to_df3}")

