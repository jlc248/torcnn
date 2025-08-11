import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
bsinfo = utils.get_bsinfo()
from NWSColorMaps import NWSColorMaps
NWScmap = NWSColorMaps()
# Plot info
plot_info = {
    'Reflectivity': {'vmin':-10, 'vmax':75, 'cmap':NWScmap},
    'AzShear': {'vmin':-0.02, 'vmax':0.02, 'cmap':'bwr'},
    'DivShear': {'vmin':-0.02, 'vmax':0.02, 'cmap': 'PiYG'},
    'Velocity': {'vmin':-50, 'vmax':50, 'cmap':'PiYG'},
    'AliasedVelocity': {'vmin':-50, 'vmax':50, 'cmap':'PiYG'},
    'SpectrumWidth': {'vmin':0, 'vmax':10, 'cmap':'cubehelix'},
    'Zdr': {'vmin':-5, 'vmax':5, 'cmap':'Spectral_r'},
    'RhoHV': {'vmin':0.45, 'vmax':1, 'cmap': 'rainbow'}, 
    'PhiDP': {'vmin':0, 'vmax':180, 'cmap': 'gnuplot2'},
}

def decode_and_visualize_tfrecord(
    tfrecord_path,
    image_keys=['Reflectivity', 'Velocity', 'RhoHV', 'AzShear'],
    all_2d_keys=['Reflectivity', 'Velocity', 'RhoHV', 'AzShear', 'DivShear',
                 'Zdr', 'PhiDP', 'SpectrumWidth'],
    image_shape=(192, 192, 1),  # You'll need to know the original shape of your 2D arrays
    image_dtype=tf.float32,  # Or tf.uint8, tf.int32, etc., depending on what you stored
    num_records_to_process=1 # How many records to decode and visualize from the TFRecord
):
    """
    Decodes TFRecord(s), prints feature names, and attempts to create a 2x2 image grid.

    Args:
        tfrecord_path (str or list of str): Path(s) to the TFRecord file(s).
        image_keys (list): A list of 4 keys corresponding to the features you want
                           to display as a 2x2 image grid.
        image_shape (tuple): The original shape (height, width) of the 2D arrays
                             stored as bytes. Essential for reshaping.
        image_dtype (tf.DType): The original TensorFlow data type of the 2D arrays.
                                Essential for tf.io.parse_tensor.
        num_records_to_process (int): The number of TFRecords to process and potentially
                                      display images from. Set to 1 to just see the first.
    """

    # 1. Define the Feature Description
    # This is crucial. You need to tell TensorFlow the expected type of each feature.
    # For serialized tensors (your 2D arrays), use tf.io.FixedLenFeature with tf.string.
    # For other types, use their corresponding FixedLenFeature.
    feature_description = {}
    
    # We need to peek into one record to get all feature names first
    # This is a bit of a workaround because FixedLenFeature requires knowing all keys upfront.
    # In a real scenario, you'd usually know your schema.
    try:
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        for raw_record in raw_dataset.take(1): # Take just one record to inspect
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            #print(f"--- Features found in TFRecord ({tfrecord_path}): ---")
            for key in example.features.feature.keys():
                #print(f"- {key}")
                # Dynamically build feature_description based on the first record's features
                # This is a heuristic and assumes consistency across records.
                feature_obj = example.features.feature[key]
                if feature_obj.HasField('bytes_list'):
                    feature_description[key] = tf.io.FixedLenFeature([], tf.string)
                elif feature_obj.HasField('float_list'):
                    feature_description[key] = tf.io.FixedLenFeature([], tf.float32) # Assume float32 for scalars
                elif feature_obj.HasField('int64_list'):
                    feature_description[key] = tf.io.FixedLenFeature([], tf.int64) # Assume int64 for scalars
                else:
                    print(f"Warning: Unknown feature type for key '{key}'. Skipping for parsing.")
            #print("-------------------------------------------------")
            break # Only need the first record for schema
    except Exception as e:
        print(f"Error reading TFRecord for schema inference: {e}")
        print("Please provide a more explicit feature_description if schema inference fails.")
        return

    if not feature_description:
        print("No features found or schema inference failed. Cannot proceed with parsing.")
        return

    # 2. Define the parsing function
    def _parse_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)

        decoded_features = {}
        for key, value in parsed_features.items():
            if feature_description[key].dtype == tf.string:
                if key in all_2d_keys:
                    # This assumes these keys hold serialized tensors (your 2D arrays)
                    # You MUST provide the correct dtype and shape for parse_tensor
                    decoded_tensor = tf.io.parse_tensor(value, out_type=image_dtype)
                    decoded_features[key] = tf.reshape(decoded_tensor, image_shape)
                else:
                    # If it's a string feature (like 'timestamp' or 'metadata'), keep it as tf.string
                    decoded_features[key] = value
            else:
                # For float, int64 scalars, they are already tensors of the right type
                decoded_features[key] = value
        return decoded_features

    # 3. Create a TFRecordDataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)

    # 4. Iterate and Visualize
    print(f"\nAttempting to visualize {min(num_records_to_process, len(list(dataset)))} records:")
    for i, features in enumerate(dataset.take(num_records_to_process)):
        print(f"\n--- Processing Record {i+1} ---")
        
        for key, value in features.items():
                # Handle TensorFlow strings for printing
                if isinstance(value, tf.Tensor) and value.dtype == tf.string:
                    try:
                        print(f"{key}: {value.numpy().decode('utf-8')}")
                    except Exception:
                        print(f"{key}: {value.numpy()}") # Fallback if not utf-8
                else:
                    # For non-string scalars (int, float), print their value
                    # For other numerical tensors, we'll check their shape next
                    if isinstance(value, tf.Tensor):
                        if len(value.shape) == 0: # It's a scalar tensor
                            print(f"{key}: {value.numpy()}")
                        elif len(value.shape) == 2: # It's a 2D tensor
                            print(f"{key} (2D Array Shape): {value.shape.as_list()}")
                        elif len(value.shape) > 0: # It's N-D but not 2D
                            print(f"{key} (Array Shape): {value.shape.as_list()}")
                    else: # Fallback for non-tensor types, though all parsed features should be tensors
                        print(f"{key}: {value}")

        # Prepare images for 2x2 plot
        images_to_plot = []
        labels_to_plot = []
        missing_keys = []

        for key in image_keys:
            if key in features and features[key].shape == image_shape and features[key].dtype == image_dtype:
                images_to_plot.append(features[key].numpy())
                labels_to_plot.append(key)
            else:
                missing_keys.append(key)
                images_to_plot.append(np.zeros(image_shape)) # Placeholder for missing image
                labels_to_plot.append(f"{key} (Missing/Wrong Type/Shape)")

        if len(images_to_plot) == 4:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            for j in range(4):
                ax = axes[j]
                img = images_to_plot[j]
                varname = labels_to_plot[j]

                missing = (img == 0)
                range_folded = (img == 1)

                img = utils.unbytescale(img, vmin=bsinfo[varname]['vmin'], vmax=bsinfo[varname]['vmax'])
                img = np.where(missing, np.nan, img)
                img = np.where(range_folded, np.nan, img)

                # Normalize float images for better display, if needed
                #if img.dtype == np.float32 or img.dtype == np.float64:
                #    if np.max(img) != np.min(img):
                #        img = (img - np.min(img)) / (np.max(img) - np.min(img))
                #    else:
                #        img = np.zeros_like(img) # Handle case where all values are the same
                
                ax.imshow(img, cmap=plot_info[varname]['cmap'], vmin=plot_info[varname]['vmin'], vmax=plot_info[varname]['vmax'])
                ax.set_title(labels_to_plot[j])
                ax.axis('off')
            
            #plt.suptitle(f"Image Grid for Record {i+1}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
            plt.show()

            if missing_keys:
                print(f"Warning: The following image keys were missing or had incorrect type/shape for visualization: {missing_keys}")
        else:
            print("Could not create 2x2 image grid. Not enough valid image keys found or correct shape/dtype was not supplied.")

    print("\nTFRecord decoding and visualization complete.")


#tfrec='/raid/jcintineo/torcnn/tfrecs/2013/20130519//tor/KTLX_35.32_-97.12_20130519-232322.tfrec'
#tfrec='/raid/jcintineo/torcnn/tfrecs/2017/20170429//tor/KFWS_32.55_-95.93_20170429-230946.tfrec'
#tfrec='/raid/jcintineo/torcnn/tfrecs2/2020/20200521/tor/KGLD_37.92_-102.31_20200521-215207.tfrec'
#tfrec='/raid/jcintineo/torcnn/tfrecs2/2020/20200522/tor/KGSP_35.1_-81.5_20200522-192612.tfrec'
#tfrec='/raid/jcintineo/torcnn/tfrecs2/2020/20200523/tor/KDVN_41.78_-91.4_20200523-173539.tfrec' # "upside down" cell?
tfrec='/raid/jcintineo/torcnn/tfrecs2/2020/20200607/tor/KABR_44.91_-99.45_20200607-231744.tfrec' # some space-time displacement
decode_and_visualize_tfrecord(tfrec, image_dtype=np.uint8)
