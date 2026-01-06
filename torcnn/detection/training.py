import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from yolo_model import build_residual_radar_yolo
import matplotlib.pyplot as plt
import metrics
import config
import shutil
import glob
import numpy as np
import sys,os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=All, 1=No Info, 2=No Warnings, 3=No Errors

# Use only for debugging
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

c = config.config()

if os.path.isdir(c['outdir']):
    print(f"{c['outdir']} already exists. Exiting.")
    sys.exit()
else:
    os.makedirs(c['outdir'], exist_ok=True)

AUTOTUNE=tf.data.AUTOTUNE
#BUFFER_SIZE = 16 * 1024**2 # 16MB
BSINFO = c['bsinfo']
#------------------------------------------------------------------------------------------
class RadarMonitor(tf.keras.callbacks.Callback):
    def __init__(self, thresholds):
        super().__init__()
        self.thresholds = thresholds

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\n\n{'='*40}")
        print(f" EPOCH {epoch + 1} RADAR PERFORMANCE REPORT")
        print(f"{'='*40}")
        
        # 1. Overall Loss
        print(f" TOTAL LOSS: {logs.get('val_loss', 0):.4f}")
        print("-" * 40)
        
        # 2. Objectness (Finding the rotation)
        print(f" [OBJECT DETECTION]")
        for tt in self.thresholds:
            pod = round(logs.get(f'val_obj_pod_{tt}', 0), 4)
            far = round(logs.get(f'val_obj_far_{tt}', 0), 4)
            csi = round(logs.get(f'val_obj_csi_{tt}', 0), 4)
            print(f"  {int(tt*100)}%:")
            print(f"  - POD: {pod}")
            print(f"  - FAR: {far}")
            print(f"  - CSI: {csi}")
            
        # 3. Classification (Is it a TVS?)
        print(f"\n [TVS / TORNADIC]")
        for tt in self.thresholds:
            pod = round(logs.get(f'val_tvs_pod_{tt}', 0), 4)
            far = round(logs.get(f'val_tvs_far_{tt}', 0), 4)
            csi = round(logs.get(f'val_tvs_csi_{tt}', 0), 4)
            print(f"  {int(tt*100)}%:")
            print(f"  - POD: {pod}")
            print(f"  - FAR: {far}")
            print(f"  - CSI: {csi}")
        print(f"  - Avg Confidence: {logs.get('val_tvs_avg_prob', 0):.4f}") # The 'warm-up' metric
        
        # 4. Classification (Severe)
        print(f"\n [SEVERE / NON-TOR]")
        for tt in self.thresholds:
            pod = round(logs.get(f'val_sev_pod_{tt}', 0), 4)
            far = round(logs.get(f'val_sev_far_{tt}', 0), 4)
            csi = round(logs.get(f'val_sev_csi_{tt}', 0), 4)
            print(f"  {int(tt*100)}%:")
            print(f"  - POD: {pod}")
            print(f"  - FAR: {far}")
            print(f"  - CSI: {csi}")
        print(f"  - Avg Confidence: {logs.get('val_sev_avg_prob', 0):.4f}")
        
        print(f"{'='*40}\n")
#------------------------------------------------------------------------------------------
def get_callbacks(c):
    csvlogger = CSVLogger(f"{c['outdir']}/log.csv", append=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=c['es_patience'], min_delta=0.0001)
    mcp_save = ModelCheckpoint(os.path.join(c['outdir'],'model-{epoch:02d}-{val_loss:03f}.keras'), save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                       cooldown=c['rlr_cooldown'],
                                       verbose=1, # min_delta=0.00001,
                                       factor=c['rlr_factor'],
                                       patience=c['rlr_patience'],
                                       mode='min'
    )
    callbacks = [early_stopping, mcp_save, reduce_lr_loss, csvlogger, RadarMonitor(thresholds=c['monitoring_thresholds'])]

    return callbacks
#------------------------------------------------------------------------------------------
class RadarYOLOLoss(tf.keras.losses.Loss):
    def __init__(self, grid_size=64, obj_weight=5.0, noobj_weight=0.5, importance=15.0):
        super().__init__()
        self.grid_size = grid_size
        # We weigh "Object" cells higher because they are rare
        self.obj_weight = obj_weight 
        self.noobj_weight = noobj_weight
        # importance: Multiplier for Class 0 (TVS) errors
        self.importance = importance

    def call(self, y_true, y_pred):
        """
        y_true: (Batch, 64, 64, 7) -> Prepared using build_target_tensor
        y_pred: (Batch, 64, 64, 7) -> Raw output from model

        in y_pred: 7 = objectness, x, y, width, height, tvs_prob, sev_prob
        """
        # Objectness 
        true_obj = y_true[..., 0]
        pred_obj_logits = y_pred[..., 0] # These are raw linear values
        ## We need the probability for the Focal Factor calculation
        pred_obj_probs = tf.sigmoid(pred_obj_logits)

        ## Objectness Loss (BCE)
        obj_bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_obj,
            logits=pred_obj_logits
        )

        ## Corrected Focal Factor Logic
        gamma = 2.0
        ## If true is 1, we want (1-p)^gamma. If true is 0, we want (p)^gamma.
        focal_factor = tf.where(tf.equal(true_obj, 1),
                                tf.pow(1.0 - pred_obj_probs, gamma),
                                tf.pow(pred_obj_probs, gamma))

        ## Combine Focal + BCE + Alpha Weights
        obj_mask = tf.cast(true_obj > 0, tf.float32)
        noobj_mask = tf.cast(true_obj == 0, tf.float32)
        
        # Apply weights and focal factor together
        weighted_obj_loss = (self.obj_weight * obj_mask * focal_factor * obj_bce) + \
                            (self.noobj_weight * noobj_mask * focal_factor * obj_bce)  


        # Localization Loss (MSE) - true_box is already 0-1
        true_box = y_true[..., 1:5]
        ## y_true is already between 0 and 1, but we use tf.sigmoid here
        ## because y_pred (model output) has *linear* activation
        pred_box = tf.sigmoid(y_pred[..., 1:5]) # x, y, w, h
        loc_loss = obj_mask * tf.reduce_sum(tf.square(true_box - pred_box), axis=-1)

      
        # Class Loss (Stabel Softmax)
        true_classes = y_true[..., 5:7]
        pred_classes = y_pred[..., 5:7] 

        class_ce = tf.nn.softmax_cross_entropy_with_logits(
            labels=true_classes,
            logits=pred_classes
        )

        # Apply the TVS importance weight properly
        # Create a mask that is 'importance' for TVS and 1.0 for Severe
        # Weight the TVS samples (Class 0) much more heavily
        class_weights = (true_classes[..., 0] * self.importance) + (true_classes[..., 1] * 1.0)
        weighted_class_loss = obj_mask * (class_ce * class_weights)

        # Total Loss: Sum everything, then average over the batch
        total_loss = tf.reduce_sum(weighted_obj_loss + loc_loss + weighted_class_loss)

        # Return average per-batch-item
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        return total_loss / batch_size

#------------------------------------------------------------------------------------------
def normalize_channel(tensor, channel_name):
    info = BSINFO[channel_name]
    c_min = info['min']
    c_max = info['max']
    
    # Cast to float
    tensor = tf.cast(tensor, tf.float32)
    
    # First, assume the uint8 (0-255) maps to the min/max range
    # Scaling uint8 to the physical units first:
    physical = (tensor / 255.0) * (c_max - c_min) + c_min
    
    if channel_name == 'Velocity':
        # Scale physical -80 to 80 -> -1.0 to 1.0
        # Formula: (val - center) / half_range
        normalized = physical / max(abs(c_min), abs(c_max))
    else:
        # Scale physical min to max -> 0.0 to 1.0
        # Formula: (val - min) / (max - min)
        normalized = (physical - c_min) / (c_max - c_min)
        
    return tf.clip_by_value(normalized, -1.0 if channel_name == 'Velocity' else 0.0, 1.0)


#------------------------------------------------------------------------------------------

def process_labels_to_grid(labels_tensor, grid_size=64):
    """
    Runs on CPU. Standard Python debugging (print, breakpoints) works here.
    """
    # Force grid_size to be a standard Python integer
    if isinstance(grid_size, tf.Tensor):
        gs = int(grid_size.numpy())
    else:
        gs = int(grid_size)


    # Convert Tensor to NumPy immediately for easy logic
    labels = labels_tensor.numpy() 
    
    # Create the empty grid (64, 64, 7)
    grid = np.zeros((gs, gs, 7), dtype=np.float32)

    # DEBUG TIP: Uncomment the line below to see labels during training
    # print(f"Processing {len(labels)} objects into grid...")

    for label in labels:
        cls_id = int(label[0])
        x_norm, y_norm, w_norm, h_norm = label[1], label[2], label[3], label[4]

        # Calculate grid cell indices (0 to 63)
        gx = int(np.floor(x_norm * gs))
        gy = int(np.floor(y_norm * gs))
        
        # Bounds check
        gx = np.clip(gx, 0, gs - 1)
        gy = np.clip(gy, 0, gs - 1)

        # Calculate relative offsets within the 5km cell (0.0 to 1.0)
        x_offset = (x_norm * gs) - gx
        y_offset = (y_norm * gs) - gy

        # Fill the 7 channels for this specific grid cell
        grid[gy, gx, 0] = 1.0                # Objectness
        grid[gy, gx, 1] = x_offset           # X Offset
        grid[gy, gx, 2] = y_offset           # Y Offset
        grid[gy, gx, 3] = w_norm             # Width (normalized)
        grid[gy, gx, 4] = h_norm             # Height (normalized)
        
        # One-hot encoding for 2 classes (Indices 5 and 6)
        if cls_id == 0:
            grid[gy, gx, 5] = 1.0 # TVS
            grid[gy, gx, 6] = 0.0
        else: #cls_id = 1
            if grid[gy, gx, 5] == 0.0: # Only label Non-Tor if we haven't yet labeled (gy, gx) as Tor
                grid[gy, gx, 6] = 1.0 # Non-Tor
    
    return grid


#------------------------------------------------------------------------------------------
def _parse_function(proto):

    feature_description = {'labels': tf.io.VarLenFeature(tf.float32)}

    # Define the n-D predictor features
    for chan in c['channels']:
        feature_description[chan] =  tf.io.FixedLenFeature([], tf.string)

    # Parse the single example
    features = tf.io.parse_single_example(proto, feature_description)

    channel_list = []
    for chan in c['channels']:
        # Use decode_raw b/c I used .tobytes()
        raw_data = tf.io.decode_raw(features[chan], out_type=tf.uint8)
        
        chan_tensor = tf.reshape(raw_data, [c['PS'][0], c['PS'][1], 1])
 
        # Normalize
        chan_tensor = normalize_channel(chan_tensor, chan)

        channel_list.append(chan_tensor)
    
    # Combine all at once along the last axis
    pred_tensor = tf.concat(channel_list, axis=-1)


    # Decode raw labels
    raw_labels = tf.sparse.to_dense(features['labels'])
    raw_labels = tf.reshape(raw_labels, (-1, 5))

    # --- The Choice: Pure TF or py_function? ---
    # Since process_labels_to_grid uses a loop and tensor_scatter_nd_update, 
    # it is safer and easier to wrap just that part in a py_function.
    
    target_grid = tf.py_function(
        func=process_labels_to_grid,
        inp=[raw_labels, 64],
        Tout=tf.float32
    )

    # Manually set the shapes so the model knows what's coming
    pred_tensor.set_shape([c['PS'][0], c['PS'][1], len(c['channels'])])
    target_grid.set_shape([64, 64, 7])

    return pred_tensor, target_grid

#------------------------------------------------------------------------------------------
def get_dataset(tfrecord_path, batch_size=16, grid_size=64, shuffle=128):
    dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=AUTOTUNE)
    
    # Parse the TFRecord and normalize predictor data
    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
   
    dataset = dataset.shuffle(shuffle).batch(batch_size).prefetch(AUTOTUNE)
    return dataset

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

# Initialize your Residual Radar YOLO
model = build_residual_radar_yolo(input_shape=(c['PS'][0], c['PS'][1], len(c['channels'])))

# Initialize the custom Radar Loss
yolo_loss = RadarYOLOLoss(grid_size=64,
                          obj_weight=c['obj_weight'],
                          noobj_weight=c['noobj_weight'],
                          importance=c['tvs_importance']
)

# Metrics
metrics_list = [metrics.AvgPredictedClassProb()]
for tt in c['monitoring_thresholds']:
    metrics_list.append(metrics.ProbMetrics(threshold=tt))


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=c['lr']),
    # custom YOLO loss function
    loss=yolo_loss, 
    metrics=metrics_list,
)
print(model.summary())


# Make training and validation datasets

train_filenames = []
for pattern in c['train_list']:
    train_filenames.extend(glob.glob(pattern))

# Uncomment to find corrupted tfrecords
#find_corrupted_tfrecords(train_filenames)
#sys.exit()

n_tsamples = len(train_filenames)
print('n_tsamples',n_tsamples)
steps_per_epoch = n_tsamples // c['batchsize']
print('steps_per_epoch:', steps_per_epoch)

train_ds = get_dataset(train_filenames, batch_size=c['batchsize'])

val_filenames = []
for pattern in c['val_list']:
    val_filenames.extend(glob.glob(pattern))
n_vsamples = len(val_filenames)
print('n_vsamples',n_vsamples)

val_ds = get_dataset(val_filenames, batch_size=c['batchsize'])

# Print out config options
ofile = f"{c['outdir']}/model_config.txt"
of = open(ofile,'w')
for key, value in c.items():
    of.write(str(key) + ': ' + str(value) + '\n')
of.write(f"Number of training samples: {n_tsamples}\n")
of.write(f"Number of validation samples: {n_vsamples}\n")
of.close()
pickle.dump(c, open(f"{c['outdir']}/model_config.pkl", "wb"))

# Callbacks
callbacks = get_callbacks(c)


# Fit
model.fit(train_ds,
          validation_data=val_ds,
          epochs=100,
          verbose=1,
          steps_per_epoch=steps_per_epoch,
          callbacks=callbacks
)


# Saving data
print('Saving model and training history...')
#copy the model with the best val_loss
best_model = np.sort(glob.glob(f"{c['outdir']}/model-*.keras"))[-1]
shutil.copy(best_model, f"{os.path.dirname(best_model)}/fit_conv_model.keras")

