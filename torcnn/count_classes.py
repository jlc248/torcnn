import glob
import os
import numpy as np
import pickle
import collections


def count_and_save_classes():
    rootdir = '/work2/jcintineo/torcnn/tfrecs_combined'

    classes = ['tornado','hail','wind','nonsev','spout',
               'pretor_15','pretor_30','pretor_45','pretor_60','pretor_120']
    
    counts = collections.OrderedDict()

    year_months = glob.glob(f'{rootdir}/2*')
    years = np.sort(list(set([os.path.basename(ym)[0:4] for ym in year_months])))

    for year in years:
        counts[year] = {}

        for cls in classes:
            counts[year][cls] = 0
            files = glob.glob(f'{rootdir}/{year}??/{cls}/*tfrec')
            for ff in files:
                # Count for each shard is in the filename: pretor_30_201206_000__n158.tfrec
                counts[year][cls] += int(os.path.basename(ff).split('.tfrec')[0].split('__n')[1])
        print(year)
    
    pickle.dump(counts, open('sample_counts_combined.pickle','wb'))


def count_classes():
    f = pickle.load(open('sample_counts_combined.pickle','rb'))

    total_tor_sum = total_all_sum = 0

    for year,val in f.items():
        tor_sum = all_sum = 0
        for cls,count in val.items():
            if cls == 'tornado' or cls == 'pretor_15' or cls == 'pretor_30':
                tor_sum += count
                all_sum += count
            elif cls in ['hail', 'wind', 'nonsev']:
                all_sum += count
        total_tor_sum += tor_sum
        total_all_sum += all_sum
        print(year, tor_sum, all_sum, np.round(tor_sum/all_sum, 3))

    print('ALL', total_tor_sum, total_all_sum, np.round(total_tor_sum/total_all_sum, 3))

if __name__ == "__main__":
    count_and_save_classes()
    count_classes()
