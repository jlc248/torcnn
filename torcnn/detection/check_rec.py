import os,sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, '../')
sys.path.append('../vda')
import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BSINFO = utils.get_bsinfo()

def unbytescale(tensor, channel_name):
    info = BSINFO[channel_name]
    c_min = info['vmin']
    c_max = info['vmax']

    # Cast to float
    tensor = tf.cast(tensor, tf.float32)

    # First, assume the uint8 (0-255) maps to the min/max range
    # Scaling uint8 to the physical units first:
    physical = (tensor / 255.0) * (c_max - c_min) + c_min

    return physical


def parse_and_verify(example_proto):
    feature_description = {'labels': tf.io.VarLenFeature(tf.float32)}

    # Define the n-D predictor features
    feature_description['Reflectivity'] =  tf.io.FixedLenFeature([], tf.string)
    feature_description['Velocity'] =  tf.io.FixedLenFeature([], tf.string)
    feature_description['Zdr'] =  tf.io.FixedLenFeature([], tf.string)
    feature_description['RhoHV'] =  tf.io.FixedLenFeature([], tf.string)
    feature_description['SpectrumWidth'] =  tf.io.FixedLenFeature([], tf.string)

    # Parse the single example
    features = tf.io.parse_single_example(example_proto, feature_description)

    numpy_features = {}
    for chan in ['Reflectivity', 'Velocity']:
        # Use decode_raw b/c I used .tobytes()
        raw_data = tf.io.decode_raw(features[chan], out_type=tf.uint8)

        chan_tensor = tf.reshape(raw_data, [512, 512])

        # unbytescale
        numpy_features[chan] = unbytescale(chan_tensor, chan).numpy()

    
    # Decode labels (flattened: [c,x,y,w,h, c,x,y,w,h, ...])
    labels_flat = tf.sparse.to_dense(features['labels']).numpy()
    labels = labels_flat.reshape(-1, 5)
    
    return numpy_features, labels

# Load one record
#record_path = '/raid/jcintineo/torcnn/detection/tfrecs_100km60min/2024/20240515/KAMA_20240515-203059.tfrec'
#record_path = '/raid/jcintineo/torcnn/detection/tfrecs_100km60min/2024/20240429/KLSX_20240429-061156.tfrec'
record_path = '/raid/jcintineo/torcnn/detection/tfrecs_100km60min/2024/20240520/KICT_20240520-011047.tfrec'
record_path = '/raid/jcintineo/torcnn/detection/tfrecs_100km60min/2024/20240610/KJGX_20240610-201655.tfrec'

raw_dataset = tf.data.TFRecordDataset(record_path)

for raw_record in raw_dataset.take(1):
    numpy_features, labels = parse_and_verify(raw_record)
    
    # Create the target grid using YOUR function
    # target = process_labels_to_grid(labels) 
    
    # Let's visualize the VELOCITY channel (Channel 1)
    velocity = numpy_features['Velocity'].astype(np.float32)
    reflectivity = numpy_features['Reflectivity'].astype(np.float32)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    img1 = ax[0].imshow(velocity, cmap='RdBu_r', origin='upper', vmin=-30, vmax=30) # 'upper' is standard for [row, col]
    fig.colorbar(img1, ax=ax[0], label="Velocity (m/s)", shrink=0.7)
    img2 = ax[1].imshow(reflectivity, cmap='viridis', origin='upper', vmin=0, vmax=60)
    fig.colorbar(img2, ax=ax[1], label="Reflectivity (dBZ)", shrink=0.7)

    # Overlay the labels
    for axis in ax:
        for lbl in labels:
            class_id, x_norm, y_norm, w_norm, h_norm = lbl
            
            # Convert normalized 0-1 back to pixel coordinates (512x512)
            pixel_x = x_norm * 512
            pixel_y = y_norm * 512
            
            axis.scatter(pixel_x, pixel_y, color='red', s=100, marker='x', label='Label Center')
            
            # Draw the box
            rect = plt.Rectangle((pixel_x - (w_norm*512)/2, pixel_y - (h_norm*512)/2), 
                                 w_norm*512, h_norm*512, 
                                 fill=False, color='yellow', linewidth=1)
            axis.add_patch(rect)
    
    plt.show()
