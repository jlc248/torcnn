import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def residual_block(x, filters):
    """
    A standard Skip Connection block.
    It adds the input of the block back to the output of the convolutions.
    """
    shortcut = x
    
    # First convolution
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False, 
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Second convolution
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False, 
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Add the shortcut (The Skip Connection)
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x

def build_residual_radar_yolo(input_shape=(512, 512, 3)):
    inputs = layers.Input(shape=input_shape)

    # --- Initial Layer: Expand channels to match the first block ---
    # Note that use_bias=False in Conv2D should always happen if using BatchNorm
    x = layers.Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # --- Block 1: 512x512 -> 256x256 ---
    x = layers.MaxPooling2D(2)(x)
    x = residual_block(x, 32)

    # --- Block 2: 256x256 -> 128x128 ---
    x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = residual_block(x, 64)

    # --- Block 3: 128x128 -> 64x64 ---
    x = layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = residual_block(x, 128)

    # --- Block 4: 64x64 -> 64x64 ---
    x = layers.Conv2D(256, 3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = residual_block(x, 256)

    # Final localized feature check
    x = layers.Conv2D(128, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # --- YOLO Head ---
    # Final 1x1 conv to get the 7-channel prediction vector
    # obj, x, y, w, h, class1, class2

    # Calculate a bias value that results in ~0.01 probability after sigmoid
    # log(p / (1-p)) -> log(0.01 / 0.99)
    prior_prob = 0.01
    bias_init = np.log(prior_prob / (1 - prior_prob)) # Approx -4.59

    prediction_head = layers.Conv2D(7, (1, 1), activation='linear', 
                                    kernel_initializer='he_normal', 
                                    # Set the bias initializer to a constant negative value for the objectness channel
                                    bias_initializer=tf.keras.initializers.Constant(value=[bias_init, 0, 0, 0, 0, 0, 0]),
                                    name='yolo_output')(x)

    return Model(inputs, prediction_head)

