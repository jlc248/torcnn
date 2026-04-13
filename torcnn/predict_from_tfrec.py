import os
import pickle
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from NWSColorMaps import NWSColorMaps
import utils
import sys
BSINFO=utils.get_bsinfo()

# --- Initialization ---
NWScmap = NWSColorMaps()

def normalize_channel(tensor, channel_name):
    info = BSINFO[channel_name]
    c_min = info['vmin']
    c_max = info['vmax']

    # Cast to float
    #tensor = tf.cast(tensor, tf.float32)

    # First, assume the uint8 (0-255) maps to the min/max range
    # Scaling uint8 to the physical units first:
    physical = (tensor / 255.0) * (c_max - c_min) + c_min

    if channel_name == 'Velocity':
        # Scale physical -80 to 80 -> -1.0 to 1.0
        # Formula: (val - center) / half_range
        normalized = physical / max(abs(c_min), abs(c_max))
    elif channel_name == 'AzShear' or channel_name == 'DivShear':
        # For shear, we do a Power Transform (x^0.5)
        # First, let's shift it to be purely positive for the transform
        shifted = physical + abs(c_min) # Now 0.0 to 0.05
        # Apply square root to "stretch" the low values
        stretched = tf.sqrt(shifted)
        # Final min-max scale to 0-1
        normalized = stretched / tf.sqrt(c_max - c_min)
    else:
        # Scale physical min to max -> 0.0 to 1.0
        # Formula: (val - min) / (max - min)
        normalized = (physical - c_min) / (c_max - c_min)

    return tf.clip_by_value(normalized, -1.0 if channel_name == 'Velocity' else 0.0, 1.0)


def predict_from_tfrec(tfrec_path, model_path):
    # 1. Load Model and Config
    print(f"Loading model: {model_path}")
    conv_model = keras.models.load_model(model_path, compile=False)
    
    config_path = os.path.join(os.path.dirname(model_path), 'model_config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    INPUTS = config['inputs'] # e.g., [['Reflectivity', 'Velocity', ...], ['range', ...]]
    PS = config['ps']             # e.g., [128, 256]
    bsinfo = config['byte_scaling_vals']
    TARGETS = config['targets']
    
    # Flatten variable list for the TFRecord parser
    all_vars = [var for sublist in INPUTS for var in sublist]
    
    # 2. Schema Inference & Parsing
    feature_description = {}
    for target in TARGETS:
        feature_description[target] = tf.io.FixedLenFeature([], tf.int64)
    #for scalar_var in SCALAR_VARS:
    #    feature_description[scalar_var] = tf.io.FixedLenFeature([], tf.float32)
    for inp in INPUTS:
        for chan in inp:
            if chan != 'range_inv':
                feature_description[chan] = tf.io.FixedLenFeature([], tf.string)


    def parse_and_prepare(example_proto):
    
        # Parse the record
        features = tf.io.parse_single_example(example_proto, feature_description)
    
        # Process Targets
        # This creates a single tensor of shape [num_targets]
        targetInt = tf.stack([tf.cast(features[t], tf.float32) for t in TARGETS])
    
        # Process Inputs
        processed_inputs = {}
    
        parsed_tensors = {}
        for inp in INPUTS:
            for chan in inp:
                if chan == 'range':
                    parsed_tensors[chan] = tf.cast(tf.reshape(tf.io.parse_tensor(features[chan], tf.uint8), [PS[1], 1]), tf.float32)
                elif chan == 'range_inv':
                    # Just a placeholder, logic handled below
                    continue
                else:
                    parsed_tensors[chan] = tf.cast(tf.reshape(tf.io.parse_tensor(features[chan], tf.uint8), [PS[0], PS[1], 1]), tf.float32)
                if chan != 'range_folded_mask' and chan != 'out_of_range_mask':
                    parsed_tensors[chan] = normalize_channel(parsed_tensors[chan], chan)

        # Assuming INPUTS[0] is radar and INPUTS[1] is coords
        for ii, inp_group in enumerate(INPUTS):
            group_tensors = []
            for varname in inp_group:
                if varname == 'range':
                    data = parsed_tensors['range']
                    data = tf.repeat(tf.expand_dims(data, axis=0), repeats=PS[0], axis=0)
                elif varname == 'range_inv':
                    # Use the 'range' data for the inversion
                    r = parsed_tensors['range']
                    data = tf.repeat(tf.expand_dims(1.0 / (r + 1e-6), axis=0), repeats=PS[0], axis=0)
                else:
                    data = parsed_tensors[varname]
                group_tensors.append(data)
    
            # Concat along channel axis (axis=2)
            key = 'radar' if ii == 0 else 'coords'
            processed_inputs[key] = tf.concat(group_tensors, axis=-1)
    
        return processed_inputs, targetInt, features




    # 3. Get the first sample
    ds = tf.data.TFRecordDataset(tfrec_path).map(parse_and_prepare)
    model_ready_dict, targets, raw_features = next(iter(ds.take(1)))

    # 4. Prepare Model Inputs (Normalization)
    # Replicate the logic from create_tensor but for parsed TFRecord data
    
    model_input_dict = {k: np.expand_dims(v.numpy(), axis=0) for k, v in model_ready_dict.items()}
    
    # 5. Predict
    prediction = conv_model.predict(model_input_dict)
    prob = np.squeeze(prediction)
    print(f"Prediction Probability: {prob:.4f}")

    # 6. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pick two interesting channels to plot (Reflectivity and Velocity)
    plot_vars = ['Reflectivity', 'Velocity']
    for ax, var in zip(axes, plot_vars):
        if var in sample:
            # We must parse the raw bytes from raw_features to plot
            raw_bytes = tf.io.parse_tensor(raw_features[var], tf.uint8).numpy()
            data = raw_bytes.reshape(PS)
      
            # Unscale to physical units for visualization using provided bsinfo
            phys_data = utils.unbytescale(data, vmin=bsinfo[var]['vmin'], vmax=bsinfo[var]['vmax'])
            
            # Mask missing (0) or range-folded (1)
            phys_data[data <= 1] = np.nan
            
            cmap = NWScmap if var == 'Reflectivity' else 'PiYG'
            vmin = bsinfo[var]['vmin'] if var == 'Reflectivity' else -50
            vmax = bsinfo[var]['vmax'] if var == 'Reflectivity' else 50
            
            im = ax.imshow(phys_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(var)
        ax.axis('off')

    plt.suptitle(f"Prediction: {prob:.2%} Probability of Tornado", fontsize=16, color='red' if prob > 0.5 else 'black')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #TFREC_FILE = '/work2/jcintineo/torcnn/tfrecs_combined/201705/tornado/tornado_201705_002__n650.tfrec'
    TFREC_FILE='/home/john.cintineo/temp_KDVN/tfrecs/202604/tornado/tornado_202604_000__n1.tfrec'
    MODEL_FILE = 'static/model/fit_conv_model.keras'
    
    predict_from_tfrec(TFREC_FILE, MODEL_FILE)
