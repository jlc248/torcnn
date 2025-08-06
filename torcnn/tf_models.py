import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers
from tensorflow.keras.regularizers import l2
import sys
#import losses
import tf_metrics
import numpy as np
try:
    import tensorflow_addons as tfa
except ModuleNotFoundError:
    print("Warning: no tfa")
    pass
##############################################################################################
def get_metrics(num_targets=1):

  metrics = []

  levs = np.arange(0.05,0.51,0.05)
  for ii in range(num_targets):

      metrics.append(tf_metrics.AUC(name=f'auprc_index{ii}', curve='PR', index=ii))
      metrics.append(tf_metrics.BrierScore(name=f'brier_score_index{ii}', index=ii))

      for lev in levs:
          metrics.append(tf_metrics.csi(use_soft_discretization=False,
                                        hard_discretization_threshold=lev,
                                        name=f'csi{str(int(lev*100)).zfill(2)}_index{ii}',
                                        index=ii
                         )
          )
          metrics.append(tf_metrics.pod(use_soft_discretization=False,
                                        hard_discretization_threshold=lev,
                                        name=f'pod{str(int(lev*100)).zfill(2)}_index{ii}',
                                        index=ii
                         )
          )
          metrics.append(tf_metrics.far(use_soft_discretization=False,
                                        hard_discretization_threshold=lev,
                                        name=f'far{str(int(lev*100)).zfill(2)}_index{ii}',
                                        index=ii
                         )
          )

  return metrics
#---------------------------------------------------------------------------------------------------------------------------------
def cnn(config):

    input_tuples = config['input_tuples']
    filter_width = config['filter_width']
    conv_activation = config['conv_activation']
    learning_rate = config['learning_rate']
    batch_norm = config['batch_norm']
    dor = config['dropout_rate']
    scalar_vars = config['scalar_vars']
    padding = config['padding']
    num_encoding_blocks = config['num_encoding_blocks']
    num_conv_per_block = config['num_conv_per_block']
    nfmaps_by_block = config['nfmaps_by_block']

    input_0 = layers.Input(shape=input_tuples[0], name='input_0')
    inputs = [input_0]
    input_0  = layers.Rescaling(1./255)(input_0)

    # encoding
    for ii in range(num_encoding_blocks):
        nfmaps = nfmaps_by_block[ii]

        for jj in range(num_conv_per_block):
            first_block_input = input_0 if(ii == 0 and jj == 0) else conv
            conv = layers.Conv2D(nfmaps, filter_width, padding=padding, )(first_block_input)
            if batch_norm:
                conv = layers.BatchNormalization()(conv)
            if conv_activation == 'leaky_relu':
                conv = layers.LeakyReLU()(conv)
            else:
                layers.Activation(conv_activation)(conv)
            if jj == num_conv_per_block - 1: #if last conv in the block
                conv = layers.MaxPooling2D(pool_size=(2, 2))(conv)

    # Flatten
    conv = layers.Flatten()(conv)

    #concatenate scalars
    if scalar_vars:
        scalar_input = layers.Input(shape=input_tuples[-1], name='input_1')
        inputs.append(scalar_input)
        scalar_input  = layers.Rescaling(1./255)(scalar_input)
        conv = layers.concatenate([conv,scalar_input])

    # Dense layers
    for ii,nneurons in enumerate(config['dense_layers']):
        conv = layers.Dense(nneurons)(conv)
        if batch_norm:
            conv = layers.BatchNormalization()(conv)
        if conv_activation == 'leaky_relu':
            conv = layers.LeakyReLU()(conv)
        else:
            layers.Activation(conv_activation)(conv)
        if dor > 0:
            conv = layers.Dropout(dor)(conv)


    # Dense output layer, equivalent to a logistic regression on the last layer
    conv = layers.Dense(1)(conv)
    conv = layers.Activation("sigmoid")(conv)
    conv_model = Model(inputs=inputs,outputs=conv)

    loss_fcn = config['loss_fcn']
    if config['loss_fcn'] == 'csi': #see page 21 of https://arxiv.org/pdf/2106.09757.pdf
        loss_fcn = losses.csi(use_as_loss_function=True, use_soft_discretization=False, hard_discretization_threshold=None)
    elif config['loss_fcn'] == 'tversky_coeff':                                               #which_class=None means binary classification
        loss_fcn = losses.tversky_coeff(use_as_loss_function=True, use_soft_discretization=False, which_class=None,
                        hard_discretization_threshold=None,false_positive_weight=0.5,false_negative_weight=0.5)
    elif config['loss_fcn'] == 'iou':
        loss_fcn = losses.iou(use_as_loss_function=True, use_soft_discretization=False, hard_discretization_threshold=None)

    METRICS = get_metrics()

    optimizer = config['optimizer']
    if optimizer.lower() == 'adabelief':
        opt = tfa.optimizers.AdaBelief
    else:
        opt = getattr(tensorflow.keras.optimizers,config['optimizer'])
    conv_model.compile(optimizer = opt(learning_rate = learning_rate), loss = loss_fcn, metrics = METRICS)
    print(conv_model.summary())

    return conv_model

#---------------------------------------------------------------------------------------------------------------------------------
