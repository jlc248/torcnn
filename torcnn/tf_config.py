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
    cts = pickle.load(open('sample_counts.pickle','rb'))
    train_years = np.arange(2011,2019,1, dtype=int)
    val_years = [2019]
    pos_classes = ['tor', 'pretor_15', 'pretor_30']
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
  
        tfrec_dir = "/work2/jcintineo/torcnn/tfrecs_shard"
        # N.B. Can't have any "//" in the train_list or val_list!!!
        # subdirs: hail, wind, nonsev, pretor_15, pretor_30, pretor_45, pretor_60, pretor_120, tor, spout
        train_list = []
        for yy in train_years:
            for cl in pos_classes + neg_classes:
                train_list.append(f"{tfrec_dir}/{yy}/{yy}0610_{cl}*tfrec")
        val_list = []
        for yy in val_years:
            for cl in pos_classes + neg_classes:
                val_list.append(f"{tfrec_dir}/{yy}/{yy}0610_{cl}*tfrec") 

        outprefix = '/work2/jcintineo/torcnn/tests/2011-19/'
        outdir = f'{outprefix}/test01'
  
        # Inputs
        # 'Reflectivity', 'Velocity', 'SpectrumWidth', 'AzShear', 'DivShear', 'RhoHV', 'PhiDP', 'Zdr', 'range_folded_mask', 'out_of_range_mask', 'range', 'range_inv'
        inputs.append(['Reflectivity', 'Velocity', 'range_folded_mask', 'out_of_range_mask', 'range'])
        #inputs.append(['range','range_inv']) # we need coords for coordconv
        scalar_vars = []
  
        ps = (128,256)
        input_tuples = [(ps[0], ps[1], len(inputs[0]))]
        if len(inputs) == 2:
            input_tuples.append( (ps[0], ps[1], len(inputs[1])) )
     
        # Use coordinate convolution?
        coord_conv = False 
 
        label_smoothing = 0.1
 
        loss_fcn = 'binary_crossentropy' #tversky_coeff binary_crossentropy csi iou
        learning_rate = 0.01
        sample_weights = {} #{'dbz_thresh':30, 'clear_wt':0.25, 'precip_wt':1, 'pos_class_wt':10} #leave empty if you don't want sample weights
        class_weights = False
        #augmentations -- options: 'random_rotation:1', 'random_noise':0.1
        img_aug = {} #{'random_noise':0.1}
  
        #architecture
        num_conv_filters = 128
        bias_init = None #np.array([-2.52378297]) #this np.log([pos/neg]) ; see https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        dropout_rate = 0.2
  
        num_encoding_blocks = 5
        num_conv_per_block = 2
        nfmaps_by_block = [num_conv_filters, int(num_conv_filters), int(num_conv_filters),
                           int(num_conv_filters), int(num_conv_filters)]
        assert(len(nfmaps_by_block) == num_encoding_blocks)
        num_decoding_blocks = 0
  
        dense_layers = [256, 16] # Number of nuerons per dense layer
  
  
  
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
           'batchsize':batchsize,
           'num_encoding_blocks':num_encoding_blocks,
           'num_decoding_blocks':num_decoding_blocks,
           'num_conv_per_block':num_conv_per_block,
           'nfmaps_by_block':nfmaps_by_block,
           'dense_layers':dense_layers,
           'filter_width':3,
           'conv_activation':'leaky_relu',
           'padding':'same',
           'learning_rate':learning_rate,
           'nepoch':100,
           'es_patience':3,
           'rlr_factor':0.1,
           'rlr_patience':1,
           'rlr_cooldown':2,
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
