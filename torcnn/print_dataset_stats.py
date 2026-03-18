from torp.torp_dataset import TORPDataset2
import pandas as pd
import numpy as np
import pickle
import sys

def stats(dd):
    total = dd['tor'] + dd['hail'] + dd['wind'] + dd['nonsev'] + dd['pretor_15'] + dd['pretor_30'] + dd['pretor_45'] + dd['pretor_60']
    p_hail = np.round(dd['hail']/total,3)
    p_wind = np.round(dd['wind']/total,3)
    p_tor = np.round(dd['tor']/total,3)
    p_nonsev = np.round(dd['nonsev']/total,3)
    p_pretor15 = np.round(dd['pretor_15']/total,3)
    p_pretor30 = np.round(dd['pretor_30']/total,3)
    p_pretor45 = np.round(dd['pretor_45']/total,3)
    p_pretor60 = np.round(dd['pretor_60']/total,3)
    return total, p_hail, p_wind, p_tor, p_nonsev, p_pretor15, p_pretor30, p_pretor45, p_pretor60

f=pickle.load(open('sample_counts.pickle','rb'))
years=np.sort(np.array(list(f.keys())))
total_totals = total_tor = total_hail = total_wind = total_nonsev = total_pretor = 0
for y in years:
    total, hail, wind, tor, nonsev, pretor15, pretor30, pretor45, pretor60 = stats(f[y])
    total_totals += total
    total_tor += tor * total
    total_wind += wind * total
    total_hail += hail * total
    total_nonsev += nonsev * total
    total_pretor += (pretor15 + pretor30 + pretor45 + pretor60)*total
    print(y)
    print('total samples:', total)
    print('%tor:', tor*100)
    print('%pretor:',(pretor15 + pretor30 + pretor45 + pretor60)*100)
    print('%hail:',hail*100)
    print('%wind:',wind*100)
    print('%nonsev:',nonsev*100)
    print('')
print('TOTAL')
print('tor:',int(total_tor), np.round(100*int(total_tor)/total_totals,1),'%')
print('pretor:', int(total_pretor), np.round(100*int(total_pretor)/total_totals,1),'%')
print('hail:', int(total_hail), np.round(100*int(total_hail)/total_totals,1),'%')
print('wind:', int(total_wind), np.round(100*int(total_wind)/total_totals,1),'%')
print('nonsev:', int(total_nonsev), np.round(100*int(total_nonsev)/total_totals,1),'%')
    
sys.exit(1)

years = np.arange(2011,2021,dtype=int)
dataset_type='WarningReportPreTornadoInfo'

#years = np.arange(2021,2025, dtype=int)
#dataset_type = 'WarningReportInfo'

for y in years:
    dataset = TORPDataset2(dirpath='/work2/jcintineo/TORP/',
                          years=[y],
                          dataset_type=dataset_type
    )
    ds = dataset.load_dataframe()

    n_samples = len(ds)
    n_tor = np.sum(ds.tornado > 0)
    n_severe = np.sum((ds.tornado > 0) | (ds.wind > 0) | (ds.hail > 0))
    try:
        n_pretor = np.sum(ds.pretor > 0)
    except AttributeError:
        n_pretor = 0

    print(f"{y}: n_samples: {n_samples}; tor%: {np.round(n_tor/n_samples, 3)}; sev%: {np.round(n_severe/n_samples, 3)}; pretor%: {np.round(n_pretor/n_samples, 3)}")
