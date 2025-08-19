import matplotlib.pyplot as plt
import numpy as np
import performance_diagrams
import attributes_diagrams
import pickle
import os,sys
import sklearn
import pandas as pd

# Use old-sklearn environment for TORPv1 scoring
#indir='/raid/jcintineo/torcnn/eval/nospout_2023/torp/'
indir='/raid/jcintineo/torcnn/eval/nontor2024_pretor2018/60min/torp/'
preds = np.load(f'{indir}/predictions.npy')
labels = np.load(f'{indir}/labels.npy')

outdir=indir


# Performance Diagram
scores, axes = performance_diagrams.plot_performance_diagram(labels, preds, return_axes=True)

AUPRC = performance_diagrams.get_area_under_perf_diagram(scores['sr'], scores['pod'])

axes.text(0.67, 0.93, f'AUPRC: {np.round(AUPRC,3)}', fontdict={'fontsize':8},
          bbox={'linewidth':1, 'alpha':0.5, 'facecolor':'white', 'edgecolor':'black', 'boxstyle':'round,pad=0.3'})

plt.savefig(f'{outdir}/perf_diagram.png', dpi=300, bbox_inches='tight')
pickle.dump(scores, open(f'{outdir}/scores.pkl', 'wb'))
print(f'Saved {outdir}/perf_diagram.png')
plt.close()

# Attributes Diagram

brier_score = sklearn.metrics.mean_squared_error(labels, preds)

_, _, _, axes, fig = attributes_diagrams.plot_attributes_diagram(labels, preds, return_main_axes=True, plot_hist=True)

axes.text(0.6, 0.12, f'Brier Score: {np.round(brier_score,4)}', fontdict={'fontsize':8},
          bbox={'linewidth':1, 'alpha':0.5, 'facecolor':'white', 'edgecolor':'black', 'boxstyle':'round,pad=0.3'})
plt.savefig(f'{outdir}/att_diagram.png', dpi=300, bbox_inches='tight')
print(f'Saved {outdir}/att_diagram.png')
