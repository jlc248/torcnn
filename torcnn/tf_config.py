import sys, os
import numpy as np
import collections
import pickle
import utils

def tf_config():
    binarize = -1
  
    byte_scaling_vals = utils.get_bsinfo()
  
    cnn = 'cnn'
  
    inputs = []

    # The number of records/samples, NOT the number of files.
    # Find this apriori for sharded datasets using count_records.py
    ## output of count_records.py
    cts = pickle.load(open('sample_counts_combined.pickle','rb'))
    train_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2019] # 2019 has no pretor counts
    val_years = [2018]
    pos_classes = ['tornado', 'pretor_15', 'pretor_30']
    neg_classes = ['hail', 'wind', 'nonsev']
    n_tsamples = 0
    for yy in train_years:
        for cl in pos_classes + neg_classes:
            n_tsamples += cts[str(yy)][cl]
    n_vsamples = 0
    for yy in val_years:
        for cl in pos_classes + neg_classes:
            n_vsamples += cts[str(yy)][cl]
    
    # For conventional CNNs
    if cnn == 'cnn':
        ngpu = 1
        batchsize = max([ngpu,1]) * 256
        targets = ['tornado']
  
        tfrec_dir = "/work2/jcintineo/torcnn/tfrecs_combined" 
        # N.B. Can't have any "//" in the train_list or val_list!!!
        # subdirs: hail, wind, nonsev, pretor_15, pretor_30, pretor_45, pretor_60, pretor_120, tor, spout
        train_list = []
        for yy in train_years:
            for cl in pos_classes + neg_classes:
                train_list.append(f"{tfrec_dir}/{yy}??/{cl}/{cl}_{yy}??_*tfrec")
        val_list = []
        for yy in val_years:
            for cl in pos_classes + neg_classes:
                val_list.append(f"{tfrec_dir}/{yy}??/{cl}/{cl}_{yy}??_*tfrec") 

        outprefix = '/work2/jcintineo/torcnn/tests/2011-19/'
        outdir = f'{outprefix}/test01'
  
        # Inputs
        # 'Reflectivity', 'Velocity', 'SpectrumWidth', 'AzShear', 'DivShear', 'RhoHV', 'PhiDP', 'Zdr', 'range_folded_mask', 'out_of_range_mask', 'range', 'range_inv'
        inputs.append(['Reflectivity', 'Velocity', 'RhoHV', 'range_folded_mask', 'out_of_range_mask','range'])
        #inputs.append(['range','range_inv']) # we need coords for coordconv
        scalar_vars = []
  
        ps = (128,256)
        input_tuples = [(ps[0], ps[1], len(inputs[0]))]
        if len(inputs) == 2:
            input_tuples.append( (ps[0], ps[1], len(inputs[1])) )
     
        # Use coordinate convolution?
        coord_conv = False
 
        label_smoothing = 0.1
 
        loss_fcn = 'binary_crossentropy' # binary_focal_crossentropy, binary_crossentropy, csi, iou
        learning_rate = 0.01
        lr_schedule = None #{'type':'cosine', # dict or None
                       #'warmup_epochs':3,
                       #'alpha':0.01
        #}
        sample_weights = {} #{'dbz_thresh':30, 'clear_wt':0.25, 'precip_wt':1, 'pos_class_wt':10} #leave empty if you don't want sample weights
        class_weights = False
        #augmentations -- options: 'random_rotation:1', 'random_noise':0.1
        img_aug = {} #{'random_noise':0.1}
  
        #architecture
        num_conv_filters = 128
        bias_init = None #np.array([-2.52378297]) #this np.log([pos/neg]) ; see https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        dropout_rate = [0.2, 0.0] # for dense layers
  
        num_encoding_blocks = 5
        num_conv_per_block = 2
        nfmaps_by_block = [int(num_conv_filters), int(num_conv_filters), int(num_conv_filters),
                           int(num_conv_filters), int(num_conv_filters)]
        assert(len(nfmaps_by_block) == num_encoding_blocks)
        spatial_dropout = [0, 0, 0, 0, 0]
        assert(len(spatial_dropout) == num_encoding_blocks)
        num_decoding_blocks = 0
  
        dense_layers = [256, 16] # Number of nuerons per dense layer
        regs = [0, 0]
        assert(len(dropout_rate) == len(dense_layers) == len(regs)) 
  
  
    channels = []
    for inp in inputs:
      channels += inp
  
    if scalar_vars:
        #for sv in scalar_vars:
        #    std_scaling_vals[ch] = std_scaling_vals[ch]
        input_tuples.append((len(scalar_vars),))
  
    return {
           'coord_conv':coord_conv,
           'ps':ps,
           'img_aug':img_aug,
           'sample_weights':sample_weights,
           'class_weights':class_weights,
           'batch_norm':True,
           'bias_init':bias_init,
           'scalar_vars':scalar_vars,
           'channels':channels,
           'l2_reg':0.0,
           'dropout_rate':dropout_rate,
           'spatial_dropout':spatial_dropout,
           'batchsize':batchsize,
           'num_encoding_blocks':num_encoding_blocks,
           'num_decoding_blocks':num_decoding_blocks,
           'num_conv_per_block':num_conv_per_block,
           'nfmaps_by_block':nfmaps_by_block,
           'dense_layers':dense_layers,
           'regs':regs,
           'filter_width':3,
           'conv_activation':'leaky_relu',
           'padding':'same',
           'learning_rate':learning_rate,
           'lr_schedule':lr_schedule,
           'nepoch':100,
           'es_patience':6,
           'monitor':'val_auprc_index0',
           'rlr_factor':0.1, #0.2
           'rlr_patience':2, #2
           'rlr_min_delta':0.001, #0.001
           'rlr_cooldown':1, #1
           'cnn':cnn,
           'input_tuples':input_tuples,
           'inputs':inputs,
           'train_list':train_list,
           'val_list':val_list,
           'targets':targets,
           #'std_scaling_vals':std_scaling_vals,
           'byte_scaling_vals':byte_scaling_vals,
           'binarize':binarize,
           'ngpu':ngpu,
           'loss_fcn':loss_fcn,
           'optimizer':'AdamW', # AdamW, Adam, SGD, adabelief
           'outdir':outdir,
           'pool':'max',
           'label_smoothing': label_smoothing,
           'steps_per_epoch': n_tsamples // batchsize,
           'n_tsamples':n_tsamples,
           'n_vsamples':n_vsamples,
           'train_years':train_years,
           'val_years':val_years,
           'pos_classes':pos_classes,
           'neg_classes':neg_classes,
    }
  
if __name__ == "__main__":
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    print(tf_config())
    #sys.exit()
    pickle.dump(tf_config(),open(os.path.join(outdir,'model_config.pkl'),'wb'))
