import tensorflow as tf
from yolo_model import build_residual_radar_yolo
import metrics

#------------------------------------------------------------------------------------------
class RadarYOLOLoss(tf.keras.losses.Loss):
    def __init__(self, grid_size=64, obj_weight=5.0, noobj_weight=0.5, importance=3.0):
        super().__init__()
        self.grid_size = grid_size
        # We weigh "Object" cells higher because they are rare
        self.obj_weight = obj_weight 
        self.noobj_weight = noobj_weight
        # importance: Multiplier for Class 0 (TVS) errors
        self.importance = importance

    def call(self, y_true, y_pred):
        """
        y_true: (Batch, 64, 64, 6) -> Prepared using build_target_tensor
        y_pred: (Batch, 64, 64, 6) -> Raw output from model
        """
        # Extract components
        # (Using sigmoid on pred objectness/offsets to keep them 0-1)
        true_obj = y_true[..., 0]
        pred_obj = tf.sigmoid(y_pred[..., 0])
        
        true_box = y_true[..., 1:5]
        pred_box = tf.concat([
            tf.sigmoid(y_pred[..., 1:3]), # x, y
            tf.exp(y_pred[..., 3:5])      # w, h
        ], axis=-1)

        # Classes (Indices 5 and 6)
        true_classes = y_true[..., 5:7]
        pred_classes = tf.sigmoid(y_pred[..., 5:7]) # Independent probabilities

        # Masks
        obj_mask = tf.expand_dims(tf.cast(true_obj > 0, tf.float32), -1)
        noobj_mask = tf.cast(true_obj == 0, tf.float32) 

        # Objectness Loss (Binary Cross Entropy)
        # We penalize both false positives and false negatives
        obj_bce = tf.keras.losses.binary_crossentropy(
            tf.expand_dims(true_obj, -1), tf.expand_dims(pred_obj, -1)
        )
        
        # Weighted sum
        total_obj_loss = (self.obj_weight * tf.squeeze(obj_mask) * obj_bce) + \
                         (self.noobj_weight * noobj_mask * obj_bce)       
 
        # Localization Loss (MSE) - only where cells exist
        loc_loss = tf.reduce_sum(obj_mask * tf.square(true_box - pred_box), axis=-1)

        # Class Loss with Importance Weighting
        # We calculate BCE for each class
        class_bce = tf.keras.losses.binary_crossentropy(true_classes, pred_classes)

        # We want to weight Class 0 (TVS/pre-tor) errors more heavily
        # Create a weight vector: [importance, 1.0]
        weights = tf.constant([self.importance, 1.0])
        weighted_class_bce = class_bce * tf.reduce_sum(true_classes * weights, axis=-1)

        # Only apply class loss where an object actually exists
        total_class_loss = tf.reduce_sum(obj_mask * tf.expand_dims(weighted_class_bce, -1), axis=-1)

        # Final Combined Loss
        return tf.reduce_mean(total_obj_loss + loc_loss + total_class_loss)

#------------------------------------------------------------------------------------------
def process_labels_to_grid(labels, grid_size=64):
    """
    labels: (N, 5) tensor of [class, x, y, w, h]
    returns: (grid_size, grid_size, 7) tensor
    """
    # Create an empty grid
    grid = tf.zeros((grid_size, grid_size, 7), dtype=tf.float32)
    
    # If there are no labels (background image), return the empty grid
    if tf.shape(labels)[0] == 0:
        return grid

    for i in range(tf.shape(labels)[0]):
        label = labels[i]

        # Extract class and normalized coordinates
        cls_id = tf.cast(label[0], tf.int32)
        x_norm, y_norm = label[1], label[2]
        w_norm, h_norm = label[3], label[4]

        # Calculate grid cell indices
        gx = tf.cast(tf.floor(x_norm * grid_size), tf.int32)
        gy = tf.cast(tf.floor(y_norm * grid_size), tf.int32)
        
        # Ensure indices stay within 0-63 bounds
        gx = tf.clip_by_value(gx, 0, grid_size - 1)
        gy = tf.clip_by_value(gy, 0, grid_size - 1)

        # Calculate relative offsets within the 5km cell
        x_offset = (x_norm * tf.cast(grid_size, tf.float32)) - tf.cast(gx, tf.float32)
        y_offset = (y_norm * tf.cast(grid_size, tf.float32)) - tf.cast(gy, tf.float32)

        # Create one-hot encoding for the two classes
        # If cls_id is 0 -> [1, 0]; if 1 -> [0, 1]
        one_hot_class = tf.one_hot(cls_id, depth=2)

        # Prepare the 7-item update vector
        # [Objectness, X, Y, W, H, Prob_Class0, Prob_Class1]
        update_values = tf.concat([
            [1.0],                       # Objectness
            [x_offset, y_offset],        # Box Centroid
            [w_norm, h_norm],            # Box Size
            one_hot_class                # Class Probabilities
        ], axis=0)

        indices = tf.cast([[gy, gx]], tf.int32)
        updates = tf.expand_dims(update_values, 0)
        
        grid = tf.tensor_scatter_nd_update(grid, indices, updates)

    return grid

#------------------------------------------------------------------------------------------
def preprocess_radar_batch(image, labels):
    """
    3-Channel Input: 0:Z (Reflectivity), 1:V (Velocity), 2:Rho (CC)
    """
    z = image[..., 0:1] / 70.
    v = image[..., 1:2] / 100
    rho = image[..., 2:3]

    # Reflectivity: 0 to 75 dBZ -> [0, 1]
    z = tf.clip_by_value(z, 0.0, 75.0)
    z = z / 75.0 
    
    # Velocity: -75 to 75 m/s -> [-1, 1]
    v = tf.clip_by_value(v, -10.0, 100.0)
    v = v / 100.0
    
    # Debris is often < 0.8.
    # but clipping helps remove noise.
    rho = tf.clip_by_value(rho, 0.4, 1.0)

    normalized_image = tf.concat([z, v, rho], axis=-1)
    return normalized_image, labels

#------------------------------------------------------------------------------------------
def _parse_function(proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.float32),
    }
    parsed = tf.io.parse_single_example(proto, feature_description)

    # Decode image
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, (512, 512, 4))

    # Decode labels (reshape to N objects, 5 columns)
    labels = tf.sparse.to_dense(parsed['labels'])
    labels = tf.reshape(labels, (-1, 5))

    return image, labels

#------------------------------------------------------------------------------------------
def get_dataset(tfrecord_path, batch_size=16, grid_size=64):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # 1. Parse the TFRecord
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 2. Normalize Z, V, and Rho (3 channels)
    # 3. Transform [N, 5] labels into [64, 64, 7] grid
    def finalize_data(image, labels):
        # Apply the 3-channel normalization we wrote earlier
        norm_image, _ = preprocess_radar_batch(image, labels)
        
        # Transform labels to grid
        # We use py_function if the scatter logic is tricky, 
        # but here we'll assume the process_labels_to_grid logic above.
        target_grid = process_labels_to_grid(labels, grid_size=grid_size)
        
        return norm_image, target_grid

    dataset = dataset.map(finalize_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.shuffle(128).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

# Initialize your Residual Radar YOLO
model = build_residual_radar_yolo(input_shape=(512, 512, 3))

# Initialize the custom Radar Loss
yolo_loss = RadarYOLOLoss(grid_size=64, obj_weight=5.0, noobj_weight=0.5)

# Metrics
tvs_10 = metrics.RadarMetrics(threshold=0.1)
tvs_30 = metrics.RadarMetrics(threshold=0.3)
tvs_50 = metrics.RadarMetrics(threshold=0.5)
tvs_70 = metrics.RadarMetrics(threshold=0.7)

metrics = [tvs_10, tvs_30, tvs_50, tvs_70]
metrics.append(tf.keras.metrics.BinaryAccuracy(name="obj_accuracy"))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    # Your custom YOLO loss function
    loss=yolo_loss, 
    metrics=metrics,
)

# Train
train_ds = get_dataset("radar_train.tfrecord", batch_size=16)
val_ds = get_dataset("radar_val.tfrecord", batch_size=16)

model.fit(train_ds, validation_data=val_ds, epochs=50)
