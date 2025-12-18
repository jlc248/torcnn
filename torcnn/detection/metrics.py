import tensorflow as tf

class RadarMetrics(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='radar_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Trackers for TVS (Class 0)
        self.tvs_tp = self.add_weight(name='tvs_tp', initializer='zeros')
        self.tvs_fn = self.add_weight(name='tvs_fn', initializer='zeros')
        self.tvs_fp = self.add_weight(name='tvs_fp', initializer='zeros')
        # Trackers for non-tor circulation class
        #self.circ_tp = self.add_weight(name='circ_tp', initializer='zeros')
        #self.circ_fn = self.add_weight(name='circ_fn', initializer='zeros')
        #self.circ_fp = self.add_weight(name='circ_fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Predictions
        pred_obj = tf.sigmoid(y_pred[..., 0])
        pred_classes = tf.sigmoid(y_pred[..., 5:7]) # [TVS, Non-Tor]
        
        # Truth
        true_obj = y_true[..., 0]
        true_class_idx = tf.argmax(y_true[..., 5:7], axis=-1) # 0 for TVS, 1 for Non-Tor. These dims are one-hot encoded.
        
        # If object and 
        is_actual_tvs = tf.cast(true_obj >= 0.5, tf.float32) * \
                        tf.cast(true_class_idx == 0, tf.float32) # 0 is TVS, 1 is Non-Tor
        
        # Use the threshold consistently
        is_detection = tf.cast(pred_obj >= self.threshold, tf.float32)
        predicted_class_idx = tf.argmax(pred_classes, axis=-1) # get index of class with higher prob
        
        # Predicted TVS = Object detected AND Winner is index 0 AND Winner > threshold
        is_pred_tvs = is_detection * \
                      tf.cast(predicted_class_idx == 0, tf.float32) * \
                      tf.cast(pred_classes[..., 0] >= self.threshold, tf.float32)

        # Metrics
        tp = tf.reduce_sum(is_pred_tvs * is_actual_tvs)
        fp = tf.reduce_sum(is_pred_tvs * (1.0 - is_actual_tvs))
        fn = tf.reduce_sum(is_actual_tvs * (1.0 - is_pred_tvs))
        
        self.tvs_tp.assign_add(tp)
        self.tvs_fp.assign_add(fp)
        self.tvs_fn.assign_add(fn)

    def result(self):
        #recall = self.tvs_tp / (self.tvs_tp + self.tvs_fn + 1e-7)
        #precision = self.tvs_tp / (self.tvs_tp + self.tvs_fp + 1e-7)
        pod = self.tvs_tp / (self.tvs_tp + self.tvs_fn + 1e-7)
        far = self.tvs_fp / (self.tvs_fp + self.tvs_tp + 1e-7)
        csi = self.tvs_fp / (self.tvs_tp + self.tvs_fp + self.tvs_fn + 1e-7)
        return {f"tvs_pod_{self.threshold}": pod, f"tvs_far_{self.threshold}": far, f"tvs_csi_{self.threshold}": csi}



def average_distance_error_km(y_true, y_pred):
    # Mask to only where an actual object exists
    obj_mask = tf.cast(y_true[..., 0] >= 0.5, tf.float32)
    
    # Calculate global pixel coordinates
    # (Simplified: grid_index + offset) * pixels_per_cell * km_per_pixel
    true_x = (tf.range(64, dtype=tf.float32) + y_true[..., 1]) * 5.0
    pred_x = (tf.range(64, dtype=tf.float32) + tf.sigmoid(y_pred[..., 1])) * 5.0
    
    # Euclidean distance
    dist = tf.sqrt(tf.square(true_x - pred_x)) # Add Y logic similarly
    
    # Return mean distance only for cells with objects
    return tf.reduce_sum(dist * obj_mask) / (tf.reduce_sum(obj_mask) + 1e-7)
