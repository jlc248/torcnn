import glob
import os
import numpy as np
import pickle





def count_and_save_classes():
    rootdir = '/work2/jcintineo/torcnn/tfrecs/'
    
    counts = {}
    for year in glob.glob('/work2/jcintineo/torcnn/tfrecs/2*'):
        yyyy = os.path.basename(year)
        counts[yyyy] = {}
        subdirs = ['tor','hail','wind','nonsev','spout',
                   'pretor_15','pretor_30','pretor_45','pretor_60','pretor_120']
        for subdir in subdirs:
            counts[yyyy][subdir] = len(glob.glob(f"{year}/2*/{subdir}/*tfrec"))
        print(year)
    
    pickle.dump(counts, open('sample_counts.pickle','wb'))


def count_classes():
    f = pickle.load(open('sample_counts.pickle','rb'))

    total_tor_sum = total_all_sum = 0

    for year,val in f.items():
        tor_sum = all_sum = 0
        for subdir,count in val.items():
            if subdir == 'tor' or subdir == 'pretor_15' or subdir == 'pretor_30':
                tor_sum += count
                all_sum += count
            else:
                all_sum += count
        total_tor_sum += tor_sum
        total_all_sum += all_sum
        print(year, tor_sum, all_sum, np.round(tor_sum/all_sum, 3))

    print('ALL', total_tor_sum, total_all_sum, np.round(total_tor_sum/total_all_sum, 3))

if __name__ == "__main__":
    count_classes()
