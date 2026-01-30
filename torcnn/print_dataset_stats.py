from torp.torp_dataset import TORPDataset2
import pandas as pd
import numpy as np

#years = np.arange(2011,2021,dtype=int)
#dataset_type='WarningReportPreTornadoInfo'

years = np.arange(2021,2025, dtype=int)
dataset_type = 'WarningReportInfo'

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
