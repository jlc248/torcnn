import matplotlib.pyplot as plt
import numpy as np
import performance_diagrams
import attributes_diagrams
import pickle
import os,sys
import sklearn
import pandas as pd

indir='/raid/jcintineo/torcnn/eval/nontor2024_pretor2013/'
outdir = f"{indir}/all"
model='test23'

colors = ['#990000', '#ff0000', '#ff6666', '#ffcccc']
all_pd_lines = []
leadtimes = ['15min','30min','45min','60min']
for ii, lt in enumerate(leadtimes):

    tmp_in = f"{indir}/{lt}"
    preds = np.load(f'{tmp_in}/{model}/predictions.npy')
    labels = np.load(f'{tmp_in}/{model}/labels.npy')

    if ii == 0:
        _, axes_obj = performance_diagrams.plot_performance_diagram(
                                           labels,
                                           preds,
                                           line_colour=colors[ii],
                                           return_axes=True
        )
        all_pd_lines.append(axes_obj.get_lines()[0]) # assuming the first one is the one we want
    else:
        scores_dict = performance_diagrams._get_points_in_perf_diagram(
            observed_labels=labels,
            forecast_probabilities=preds,
        )

        this_line, = axes_obj.plot(scores_dict['sr'], scores_dict['pod'], color=colors[ii], linewidth=2)
        all_pd_lines.append(this_line)

        # Plot points
        xs = (
            [scores_dict['sr'][5]]
            + list(scores_dict['sr'][10:100:10])
            + [scores_dict['sr'][95]]
        )
        ys = (
            [scores_dict['pod'][5]]
            + list(scores_dict['pod'][10:100:10])
            + [scores_dict['pod'][95]]
        )
        labs = ["5", "10", "20", "30", "40", "50", "60", "70", "80", "90", "95"]
        axes_obj.plot(
            xs, ys, linestyle="None", color=colors[ii], marker="o", markersize=4
        )

        #for i in range(len(xs)):
        #    axes_obj.annotate(labs[i], xy=(xs[i]+0.02,ys[i]), color='black', fontsize=8)

axes_obj.legend(all_pd_lines,
                leadtimes,
                loc='upper right',
                fontsize=8
)
    
plt.savefig(f'{outdir}/perf_diagram_all_LTs.png', dpi=300, bbox_inches='tight')
print(f'Saved {outdir}/perf_diagram_all_LTs.png')
plt.close()
