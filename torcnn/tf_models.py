import numpy as np
import tensorflow as tf
import keras
import tf_metrics
from keras import ops
try:
    import tensorflow_addons as tfa
except ModuleNotFoundError:
    print("Warning: no tfa")
    pass
    
##############################################################################################
def get_metrics(num_targets=1):

  metrics = []

  levs = np.arange(0.05,0.91,0.05)
  for ii in range(num_targets):

      metrics.append(tf_metrics.AUC(name=f'auprc_index{ii}', curve='PR', index=ii))
      #metrics.append(tf.keras.metrics.AUC(name='default_aupr', curve='PR'))
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
# From TorNet
@keras.saving.register_keras_serializable()
class CoordConv2D(keras.layers.Layer):
    """
    Adopted from the CoordConv2d layers as described in 

    Liu, Rosanne, et al. "An intriguing failing of convolutional neural networks and 
    the coordconv solution." Advances in neural information processing systems 31 (2018).
    
    """
    def __init__(self,filters,
                      kernel_size,
                      kernel_regularizer,
                      activation,
                      batch_norm=False,
                      padding='same',
                      strides=(1,1),
                      conv2d_kwargs = {},
                      **kwargs):

        super(CoordConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer
        self.activation = activation
        self.batch_norm = batch_norm
        self.padding = padding
        self.strides = strides
        self.conv2d_kwargs = conv2d_kwargs
        self.strd = strides[0]  # assume equal strides

        self.conv = keras.layers.Conv2D(
            self.filters,
            self.kernel_size,
            kernel_regularizer=self.kernel_regularizer,
            padding=self.padding,
            strides=self.strides,
            **conv2d_kwargs
        )

        self.activ_layer = keras.layers.Activation(self.activation)

        if batch_norm:
            self.bn_layer = keras.layers.BatchNormalization()

    def build(self, input_shape):
        x_shape, coord_shape = input_shape
        concat_shape = list(x_shape)
        concat_shape[-1] += coord_shape[-1]
        self.conv.build(concat_shape)

    def call(self,inputs):
        """
        inputs is a tuple 
           [N, L, W, C] data tensor,
           [N, L, W, nd] tensor of coordiantes
        """
        x, coords = inputs

        # Stack x with coordinates
        x = ops.concatenate( (x,coords), axis=-1)

        # Run convolution
        conv=self.conv(x)

        # Batch normalization
        if self.batch_norm:
            conv=self.bn_layer(conv)

        # Activation
        conv = self.activ_layer(conv)

        # The returned coordinates should have same shape as conv
        # prep the coordiantes by slicing them to the same shape
        # as conv
        if self.padding=='same' and self.strd>1:
            coords = coords[:,::self.strd,::self.strd]
        elif self.padding=='valid':
            # If valid padding,  need to start slightly off the corner
            i0 = self.kernel_size[0]//2
            if i0>0:
                coords = coords[:,i0:-i0:self.strd,i0:-i0:self.strd]
            else:
                coords = coords[:,::self.strd,::self.strd]

        return conv,coords

    def get_config(self):
        """Get model configuration, used for saving model."""
        config = super().get_config()
        config.update(
            {   "filters": self.filters,
                "kernel_size": self.kernel_size,
                "kernel_regularizer": self.kernel_regularizer,
                "activation":self.activation,
                "padding": self.padding,
                "strides": self.strides,
                "conv2d_kwargs": self.conv2d_kwargs
            }
        )
        return config

#---------------------------------------------------------------------------------------------------------------------------------
def conv_coord_block(x, c, filters=64, ksize=3, n_convs=2, l2_reg=1e-6, drop_rate=0.0, batch_norm=False, activation='relu', padding="same"):

    for _ in range(n_convs):
        x,c = CoordConv2D(filters=filters,
                          kernel_size=ksize,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          padding=padding,
                          batch_norm=batch_norm,
                          activation=activation)([x,c])
    x = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    c = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(c)
    if drop_rate>0:
        x = keras.layers.Dropout(rate=drop_rate)(x)
    return x,c
#---------------------------------------------------------------------------------------------------------------------------------
def conv_block(x, filters=64, ksize=3, n_convs=2, l2_reg=1e-6, drop_rate=0.0, batch_norm=False, activation='relu', padding="same",
               spatial_dropout=0, use_attention=False,
):

    for _ in range(n_convs):
        x = keras.layers.Conv2D(filters=filters,
                          kernel_size=ksize,
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          padding=padding)(x)
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation)(x)

    if use_attention:
        attn = channel_attention(x)
        attn = spatial_attention(attn)
        # Apply the gate
        attn = LayerScale(init_value=0.0)(attn)
        # Add back to the original identity
        x = keras.layers.Add()([x, attn])

    if spatial_dropout > 0:
        x = keras.layers.SpatialDropout2D(spatial_dropout)(x) # Forces cross-channel learning

    x = keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
        
    return x
#---------------------------------------------------------------------------------------------------------------------------------
@keras.saving.register_keras_serializable()
class LayerScale(keras.layers.Layer):
    """A learnable scalar multiplier initialized at 0."""
    def __init__(self, init_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1,),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        """Get model configuration, used for saving model."""
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config
#---------------------------------------------------------------------------------------------------------------------------------
def channel_attention(input_feature, ratio=4):
    channel = input_feature.shape[-1]
    
    # Squeeze: Global Average and Max Pooling
    avg_pool = keras.layers.GlobalAveragePooling2D()(input_feature)
    max_pool = keras.layers.GlobalMaxPooling2D()(input_feature)
    
    # Excitation: Shared MLP
    shared_layer_one = keras.layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_layer_two = keras.layers.Dense(channel, kernel_initializer='he_normal', use_bias=True)
    
    avg_out = shared_layer_two(shared_layer_one(avg_pool))
    max_out = shared_layer_two(shared_layer_one(max_pool))
    
    mask = keras.layers.Add()([avg_out, max_out])
    mask = keras.layers.Activation('sigmoid')(mask)
    mask = keras.layers.Reshape((1, 1, channel))(mask)
    
    return keras.layers.Multiply()([input_feature, mask])
#---------------------------------------------------------------------------------------------------------------------------------
def spatial_attention(input_feature):
    # Channel-wise reduction: Avg and Max pool across the channel axis
    avg_pool = ops.mean(input_feature, axis=-1, keepdims=True)
    max_pool = ops.max(input_feature, axis=-1, keepdims=True)
    
    concat = keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Generate spatial mask via 7x7 conv (standard for CBAM)
    mask = keras.layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    
    return keras.layers.Multiply()([input_feature, mask])
#---------------------------------------------------------------------------------------------------------------------------------
def cnn(config):

    input_tuples = config['input_tuples']
    filter_width = config['filter_width']
    conv_activation = config['conv_activation']
    learning_rate = config['learning_rate']
    batch_norm = config['batch_norm']
    dor = config['dropout_rate'] # a list
    l2_reg = config['l2_reg']
    scalar_vars = config['scalar_vars']
    padding = config['padding']
    num_encoding_blocks = config['num_encoding_blocks']
    num_conv_per_block = config['num_conv_per_block']
    nfmaps_by_block = config['nfmaps_by_block']
    coord_conv = config['coord_conv']
    label_smoothing = config['label_smoothing']
    regs = config['regs']
    use_attention = config['use_attention']
    spatial_dropout = config['spatial_dropout']

    radar = conv = keras.Input(shape=input_tuples[0], name='radar')
    inputs = [radar]

    if coord_conv:
        # Coordinate info
        # Assumes second input_tuple will be the coords.
        # Should be (ntheta, nrange, 2)
        ntheta, nrange, nradar = input_tuples[0]
        assert(input_tuples[1] == (ntheta, nrange, 2))
        coords = keras.Input(shape=input_tuples[1], name='coords')
        inputs.append(coords)

    # encoding
    for ii in range(num_encoding_blocks):
        nfmaps = nfmaps_by_block[ii]

        if coord_conv:
            conv, coords = conv_coord_block(conv,
                                       coords,
                                       filters=nfmaps,
                                       ksize=filter_width,
                                       l2_reg=l2_reg,
                                       n_convs=num_conv_per_block,
                                     #  drop_rate=dor, # do we want drop out in the Conv2Ds?
                                       batch_norm=batch_norm,
                                       activation=conv_activation,
                                       padding=padding,
        )

        else:
            conv = conv_block(conv,
                              filters=nfmaps,
                              ksize=filter_width,
                              l2_reg=l2_reg,
                              n_convs=num_conv_per_block,
                              batch_norm=batch_norm,
                              activation=conv_activation,
                              padding=padding,
                              spatial_dropout=spatial_dropout[ii],
                              use_attention=use_attention,
            )  


    # Flatten
    conv = keras.layers.Flatten()(conv)
    #conv = keras.layers.GlobalAveragePooling2D()(conv)

    #concatenate scalars
    if scalar_vars:
        scalar_input = keras.layers.Input(shape=input_tuples[-1], name='scalars')
        inputs.append(scalar_input)
        conv = keras.layers.concatenate([conv,scalar_input])

    # Dense layers
    for ii,nneurons in enumerate(config['dense_layers']):
        if regs[ii] > 0:
            reg = keras.regularizers.l2(regs[ii])
        else:
            reg = None
        conv = keras.layers.Dense(nneurons, kernel_regularizer=reg)(conv)
        if batch_norm:
            conv = keras.layers.BatchNormalization()(conv)
        else:
            keras.layers.Activation(conv_activation)(conv)
        if dor[ii] > 0:
            conv = keras.layers.Dropout(dor[ii])(conv)


    # Dense output layer, equivalent to a logistic regression on the last layer
    conv = keras.layers.Dense(1)(conv)
    conv = keras.layers.Activation("sigmoid")(conv)
    conv_model = keras.models.Model(inputs=inputs,outputs=conv)

    loss_fcn = config['loss_fcn']
    if config['loss_fcn'] == 'csi': #see page 21 of https://arxiv.org/pdf/2106.09757.pdf
        loss_fcn = losses.csi(use_as_loss_function=True, use_soft_discretization=False, hard_discretization_threshold=None)
    elif config['loss_fcn'] == 'tversky_coeff':                                               #which_class=None means binary classification
        loss_fcn = losses.tversky_coeff(use_as_loss_function=True, use_soft_discretization=False, which_class=None,
                        hard_discretization_threshold=None,false_positive_weight=0.5,false_negative_weight=0.5)
    elif config['loss_fcn'] == 'iou':
        loss_fcn = losses.iou(use_as_loss_function=True, use_soft_discretization=False, hard_discretization_threshold=None)
    elif config['loss_fcn'] == 'binary_crossentropy':
        loss_fcn = keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    elif config['loss_fcn'] == 'binary_focal_crossentropy':
        # For gamma=2.0, if the model predicts 0.9, the loss for that sample is reduced by a factor of 100 (0.1)^2
        # If the model is unsure (p=0.5), the discount is much smaller
        # Result: the model is forced to fucs its gradient updates on hard negatives and hard positives
        # Alpha is the "class balance" weight
        # As gamma increases, the optimal alpha usually decreases. 
        loss_fcn = keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25, label_smoothing=label_smoothing)

    METRICS = get_metrics()

    optimizer = config['optimizer']
    if optimizer.lower() == 'adabelief':
        opt = tfa.optimizers.AdaBelief
    else:
        opt = getattr(keras.optimizers,config['optimizer'])

    if config['lr_schedule'] != None:
        if config['lr_schedule']['type'] == 'cosine':
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-6, 
                decay_steps=config['steps_per_epoch'] * config['nepoch'],
                warmup_target=learning_rate,
                warmup_steps=config['steps_per_epoch'] * config['lr_schedule']['warmup_epochs'],
                alpha=config['lr_schedule']['alpha'] # Minimum learning rate value for decay as a fraction of initial_learning_rate 
            )
        opt = opt(learning_rate=lr_schedule)  #, weight_decay=1e-4)
    else:
        # usually paired with ReduceLROnPlateau callback
        opt = opt(learning_rate=learning_rate) #, weight_decay=1e-4)

    conv_model.compile(optimizer = opt, loss = loss_fcn, metrics = METRICS)
    print(conv_model.summary())

    return conv_model

#---------------------------------------------------------------------------------------------------------------------------------
