import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
from NWSColorMaps import NWSColorMaps

# Initialize radar-specific info
bsinfo = utils.get_bsinfo()
NWScmap = NWSColorMaps()

# Plot configurations for specific radar products
plot_info = {
    'Reflectivity': {'vmin': -10, 'vmax': 75, 'cmap': NWScmap},
    'AzShear': {'vmin': -0.02, 'vmax': 0.02, 'cmap': 'bwr'},
    'DivShear': {'vmin': -0.02, 'vmax': 0.02, 'cmap': 'PiYG'},
    'Velocity': {'vmin': -50, 'vmax': 50, 'cmap': 'PiYG'},
    'AliasedVelocity': {'vmin': -50, 'vmax': 50, 'cmap': 'PiYG'},
    'SpectrumWidth': {'vmin': 0, 'vmax': 10, 'cmap': 'cubehelix'},
    'Zdr': {'vmin': -5, 'vmax': 5, 'cmap': 'Spectral_r'},
    'RhoHV': {'vmin': 0.45, 'vmax': 1, 'cmap': 'rainbow'},
    'PhiDP': {'vmin': 0, 'vmax': 180, 'cmap': 'gnuplot2'},
}

def decode_and_visualize_tfrecord(
    tfrecord_path,
    image_keys=['Reflectivity', 'Velocity', 'RhoHV', 'AzShear'],
    all_2d_keys=['Reflectivity', 'Velocity', 'RhoHV', 'AzShear', 'DivShear',
                 'Zdr', 'PhiDP', 'SpectrumWidth'],
    image_shape=(128, 256),  
    image_dtype=tf.uint8,  
    num_records_to_process=1
):
    """
    Decodes a TFRecord shard and visualizes radar samples.
    """
    
    # 1. Dynamic Schema Inference
    feature_description = {}
    try:
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        for raw_record in raw_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            for key in example.features.feature.keys():
                feature_obj = example.features.feature[key]
                if feature_obj.HasField('bytes_list'):
                    feature_description[key] = tf.io.FixedLenFeature([], tf.string)
                elif feature_obj.HasField('float_list'):
                    feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
                elif feature_obj.HasField('int64_list'):
                    feature_description[key] = tf.io.FixedLenFeature([], tf.int64)
    except Exception as e:
        print(f"Error inferring schema: {e}")
        return

    # 2. Parsing function
    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        decoded = {}
        for key, value in parsed.items():
            if key in all_2d_keys:
                # Parse the tensor and reshape to 2D
                t = tf.io.parse_tensor(value, out_type=image_dtype)
                decoded[key] = tf.reshape(t, image_shape)
            else:
                decoded[key] = value
        return decoded

    # 3. Load Dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path).map(_parse_function)

    # 4. Iterate and Plot
    for i, features in enumerate(dataset.take(num_records_to_process)):
        print(f"\n--- Displaying Record {i+1} from shard ---")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for j, key in enumerate(image_keys):
            ax = axes[j]
            if key in features:
                # Convert to numpy for processing
                img_data = features[key].numpy().astype(np.float32)
                
                # Flag missing/folded data (Standard for these TFRecs: 0=missing, 1=folded)
                mask = (img_data <= 1)
                
                # Unbytescale using utils
                if key in bsinfo:
                    img_data = utils.unbytescale(img_data, vmin=bsinfo[key]['vmin'], vmax=bsinfo[key]['vmax'])
                
                img_data[mask] = np.nan # Set missing values to NaN for cleaner plotting

                # Use custom colormaps and bounds if available
                info = plot_info.get(key, {'vmin': None, 'vmax': None, 'cmap': 'viridis'})
                
                im = ax.imshow(img_data) #, cmap=info['cmap'], vmin=info['vmin'], vmax=info['vmax'], origin='lower')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(key)
            else:
                ax.text(0.5, 0.5, f"Key '{key}'\nnot found", ha='center')
            
            ax.axis('off')

        plt.suptitle(f"File: {tfrecord_path.split('/')[-1]}\nSample Index: {i}", fontsize=12)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage with your provided shard path
    shard_path = '/work2/jcintineo/torcnn/tfrecs_combined/201705/tornado/tornado_201705_002__n650.tfrec'
    # This is a single-sample tfrec
    shard_path = '/home/john.cintineo/temp_KDVN/tfrecs/202604/tornado/tornado_202604_000__n1.tfrec'
    
    # Increase num_records_to_process to see more samples from the shard
    decode_and_visualize_tfrecord(shard_path, num_records_to_process=1)
