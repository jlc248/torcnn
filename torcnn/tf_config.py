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
  
    # For conventional CNNs
  
    if cnn == 'cnn':
        ngpu = 1
        batchsize = max([ngpu,1]) * 256
        targets = ['tornado']
  
        tfrec_dir = "/raid/jcintineo/torcnn/tfrecs2"
        # N.B. Can't have any "//" in the train_list or val_list!
        train_list = [f"{tfrec_dir}/201[1-9]/2*/*tor*/*tfrec", f"{tfrec_dir}/202[0-2]/2*/*tor*/*tfrec"]
        val_list = [f"{tfrec_dir}/2023/2*/*tor*/*.tfrec"]

        outprefix = '/raid/jcintineo/torcnn/tests/2011-23/'
        outdir = f'{outprefix}/test05'
  
        # Inputs
        #inputs.append(['Reflectivity','Velocity','SpectrumWidth','AzShear','DivShear','RhoHV','PhiDP','Zdr'])
        inputs.append(['Reflectivity', 'Velocity', 'range_folded_mask','range'?]
        scalar_vars = []
  
        ps = (128,256)
        input_tuples = [(ps[0], ps[0], len(inputs[0]))]
      
  
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
  
        dense_layers = [256, 32] # Number of nuerons per dense layer
  
  
  
    channels = []
    for inp in inputs:
      channels += inp
  
    if scalar_vars:
        #for sv in scalar_vars:
        #    std_scaling_vals[ch] = std_scaling_vals[ch]
        input_tuples.append((len(scalar_vars),))
  
    return {
           'ps':ps,
           'img_aug':img_aug,
           'sample_weights':sample_weights,
           'class_weights':class_weights,
           'batch_norm':True,
           'bias_init':bias_init,
           'scalar_vars':scalar_vars,
           'channels':channels,
           'l2_param':0.0,
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
           'pool':'max'
           }
  
if __name__ == "__main__":
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    print(tf_config())
    #sys.exit()
    pickle.dump(tf_config(),open(os.path.join(outdir,'model_config.pkl'),'wb'))
