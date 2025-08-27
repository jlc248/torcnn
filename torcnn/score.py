import matplotlib.pyplot as plt
import numpy as np
import performance_diagrams
import attributes_diagrams
import pickle
import os,sys
import sklearn
import pandas as pd

#inroot = '/raid/jcintineo/torcnn/eval/nontor2024_pretor2013/all'
inroot = '/raid/jcintineo/torcnn/eval/nospout2024'

indir=f'{inroot}/test23'
preds = np.load(f'{indir}/predictions.npy')
labels = np.load(f'{indir}/labels.npy')
model1 = os.path.basename(indir)

outdir=indir

two_models = False
three_models = True
if two_models or three_models:
    indir2=f'{inroot}/new_rf01'
    model2 = os.path.basename(indir2)
    preds2 = np.load(f'{indir2}/predictions.npy')
    labels2 = np.load(f'{indir2}/labels.npy')

    # parent dir outdir
    outdir = inroot

    if three_models:
        indir3=f'{inroot}/torp'
        model3 = os.path.basename(indir3)
        preds3 = np.load(f'{indir3}/predictions.npy')
        labels3 = np.load(f'{indir3}/labels.npy')

    # Performance Diagram

    scores1_dict = performance_diagrams._get_points_in_perf_diagram(
        observed_labels=labels,
        forecast_probabilities=preds,
    )
    # Get 5%, 10%, 20%,..., 90%, 95%.
    # Assuming we have every 1% 
    pod = list(scores1_dict['pod'][10:-1:10])
    pod.insert(0, scores1_dict['pod'][5]) #5%
    pod.append(scores1_dict['pod'][95]) #95%
    sr = scores1_dict['sr']
    far = list(1-sr[10:-1:10])
    far.insert(0, 1-sr[5]) #5%
    far.append(1-sr[95]) #95%
    
    scores2_dict = performance_diagrams._get_points_in_perf_diagram(
        observed_labels=labels2,
        forecast_probabilities=preds2,
    )
    pod2 = list(scores2_dict['pod'][10:-1:10])
    pod2.insert(0, scores2_dict['pod'][5]) #5%
    pod2.append(scores2_dict['pod'][95]) #95%
    sr2 = scores2_dict['sr']
    far2 = list(1-sr2[10:-1:10])
    far2.insert(0, 1-sr2[5]) #5%
    far2.append(1-sr2[95]) #95%

    if three_models:
        scores3_dict = performance_diagrams._get_points_in_perf_diagram(
            observed_labels=labels3,
            forecast_probabilities=preds3,
        )
        pod3 = list(scores3_dict['pod'][10:-1:10])
        pod3.insert(0, scores3_dict['pod'][5]) #5%
        pod3.append(scores3_dict['pod'][95]) #95%
        sr3 = scores3_dict['sr']
        far3 = list(1-sr3[10:-1:10])
        far3.insert(0, 1-sr3[5]) #5%
        far3.append(1-sr3[95]) #95%
     
        perf_outfilename = f"{outdir}/{model1}_{model2}_{model3}_perfdiagram.png"
        legend_labs = ['cnn','new_rf','torp']
    else:
        perf_outfilename = f"{outdir}/{model1}_{model2}_perfdiagram.png"
        pod3 = far3 = labs3 = None
        legend_labs = ['cnn','torp']

    labs = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    performance_diagrams.perf_diagram_pod_far(pod,
                                              far,
                                              perf_outfilename,
                                              labs=labs,
                                              pod2=pod2,
                                              far2=far2,
                                              labs2=labs,
                                              pod3=pod3,
                                              far3=far3,
                                              labs3=labs,
                                              legend_labs=legend_labs) 

    plt.savefig(perf_outfilename, dpi=300, bbox_inches='tight')
    print(f'Saved {perf_outfilename}')


    # Attributes diagram
    num_bins = 20
    (   mean_forecast_probs1,
        mean_event_frequencies1,
        num_examples_by_bin1,
    ) = attributes_diagrams._get_points_in_relia_curve(
                            observed_labels=labels,
                            forecast_probabilities=preds,
                            num_bins=num_bins,
    )
    obs_cts1 = mean_event_frequencies1 * num_examples_by_bin1


    (   mean_forecast_probs2,
        mean_event_frequencies2,
        num_examples_by_bin2,
    ) = attributes_diagrams._get_points_in_relia_curve(
                            observed_labels=labels2,
                            forecast_probabilities=preds2,
                            num_bins=num_bins,
    ) 
    obs_cts2 = mean_event_frequencies2 * num_examples_by_bin2

    if three_models:
        (   mean_forecast_probs3,
            mean_event_frequencies3,
            num_examples_by_bin3,
        ) = attributes_diagrams._get_points_in_relia_curve(
                                observed_labels=labels3,
                                forecast_probabilities=preds3,
                                num_bins=num_bins,
        )
        obs_cts3 = mean_event_frequencies3 * num_examples_by_bin3
        atts_outfilename = f"{outdir}/{model1}_{model2}_{model3}_attsdiagram.png"
    else:
        obs_cts3 = num_examples_by_bin3 = None
        atts_outfilename = f"{outdir}/{model1}_{model2}_attsdiagram.png"

    bins = np.linspace(0,1,num_bins)
   
    attributes_diagrams.rel_with_obs_and_fcst_cts(obs_cts1,
                                                  num_examples_by_bin1,
                                                  bins,
                                                  atts_outfilename,
                                                  obs_cts2=obs_cts2,
                                                  fcst_cts2=num_examples_by_bin2,
                                                  obs_cts3=obs_cts3,
                                                  fcst_cts3=num_examples_by_bin3,
                                                  labels=legend_labs,
    ) 


else:
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
