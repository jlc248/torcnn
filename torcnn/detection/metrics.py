import tensorflow as tf

class ProbMetrics(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='radar_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Trackers for TVS (Class 0)
        self.tvs_tp = self.add_weight(name='tvs_tp', initializer='zeros')
        self.tvs_fn = self.add_weight(name='tvs_fn', initializer='zeros')
        self.tvs_fp = self.add_weight(name='tvs_fp', initializer='zeros')
        # Trackers for non-tor circulation class (Class 1)
        self.sev_tp = self.add_weight(name='sev_tp', initializer='zeros')
        self.sev_fn = self.add_weight(name='sev_fn', initializer='zeros')
        self.sev_fp = self.add_weight(name='sev_fp', initializer='zeros')
        # Trackers for objectness
        self.obj_tp = self.add_weight(name='obj_tp', initializer='zeros')
        self.obj_fn = self.add_weight(name='obj_fn', initializer='zeros')
        self.obj_fp = self.add_weight(name='obj_fp', initializer='zeros')

    def reset_state(self):
        # REQUIRED: Clears counts at the start of every epoch
        for v in self.variables:
            v.assign(0.0)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Cast inputs to float32 to ensure math stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Truth
        ## Object
        true_obj = y_true[..., 0]

        # Predictions
        ## Object
        pred_obj = tf.sigmoid(y_pred[..., 0])
        ## Classes
        pred_probs = tf.nn.softmax(y_pred[..., 5:7], axis=-1) # [TVS, Non-Tor]
        
        # Threshold Masks
        is_actual = tf.cast(true_obj > 0.5, tf.float32)
        is_detected = tf.cast(pred_obj >= self.threshold, tf.float32)

        # Class-Specific Logical Masks (Truth)
        # y_true is one-hot encoded: [1, 0] is TVS, [0, 1] is Severe
        actual_is_tvs = is_actual * y_true[..., 5]
        actual_is_sev = is_actual * y_true[..., 6]

        # Class-Specific Logical Masks (Predictions)
        # A TVS is "predicted" if objectness is high AND TVS probability is >= threshold
        pred_is_tvs = is_detected * tf.cast(pred_probs[..., 0] >= self.threshold, tf.float32)
        pred_is_sev = is_detected * tf.cast(pred_probs[..., 1] >= self.threshold, tf.float32)

        
        # Metrics
        # TVS (Class 0)
        t_tp = tf.reduce_sum(pred_is_tvs * actual_is_tvs)
        t_fp = tf.reduce_sum(pred_is_tvs * (1.0 - actual_is_tvs))
        t_fn = tf.reduce_sum(actual_is_tvs * (1.0 - pred_is_tvs))
       
        self.tvs_tp.assign_add(t_tp)
        self.tvs_fp.assign_add(t_fp)
        self.tvs_fn.assign_add(t_fn)
 
        # Severe (Class 1)
        s_tp = tf.reduce_sum(pred_is_sev * actual_is_sev)
        s_fp = tf.reduce_sum(pred_is_sev * (1.0 - actual_is_sev))
        s_fn = tf.reduce_sum(actual_is_sev * (1.0 - pred_is_sev))
        
        self.sev_tp.assign_add(s_tp)
        self.sev_fp.assign_add(s_fp)
        self.sev_fn.assign_add(s_fn)

        # Objectness Stats
        self.obj_tp.assign_add(tf.reduce_sum(is_detected * is_actual))
        self.obj_fp.assign_add(tf.reduce_sum(is_detected * (1.0 - is_actual)))
        self.obj_fn.assign_add(tf.reduce_sum(is_actual * (1.0 - is_detected)))

    def result(self):
        #recall = self.tvs_tp / (self.tvs_tp + self.tvs_fn + 1e-7)
        #precision = self.tvs_tp / (self.tvs_tp + self.tvs_fp + 1e-7)
        pod_t = self.tvs_tp / (self.tvs_tp + self.tvs_fn + 1e-7)
        far_t = self.tvs_fp / (self.tvs_fp + self.tvs_tp + 1e-7)
        csi_t = self.tvs_tp / (self.tvs_tp + self.tvs_fp + self.tvs_fn + 1e-7)
        pod_s = self.sev_tp / (self.sev_tp + self.sev_fn + 1e-7)
        far_s = self.sev_fp / (self.sev_fp + self.sev_tp + 1e-7)
        csi_s = self.sev_tp / (self.sev_tp + self.sev_fp + self.sev_fn + 1e-7)
        # obj
        pod_o = self.obj_tp / (self.obj_tp + self.obj_fn + 1e-7)
        far_o = self.obj_fp / (self.obj_fp + self.obj_tp + 1e-7)
        csi_o = self.obj_tp / (self.obj_tp + self.obj_fp + self.obj_fn + 1e-7)
        return {f"tvs_pod_{self.threshold}": pod_t, 
                f"tvs_far_{self.threshold}": far_t,
                f"tvs_csi_{self.threshold}": csi_t,
                f"sev_pod_{self.threshold}": pod_s,
                f"sev_far_{self.threshold}": far_s,
                f"sev_csi_{self.threshold}": csi_s,
                f"obj_pod_{self.threshold}": pod_o,
                f"obj_far_{self.threshold}": far_o,
                f"obj_csi_{self.threshold}": csi_o,
        }



class AvgPredictedClassProb(tf.keras.metrics.Metric):
    def __init__(self, name='avg_predicted_class_prob', **kwargs):
        super().__init__(name=name, **kwargs)
        # Trackers for TVS (Class 0)
        self.tvs_avg_prob = self.add_weight(name='tvs_avg_prob', initializer='zeros')
        self.tvs_count = self.add_weight(name='tvs_count', initializer='zeros')
        # Trackers for non-tor circulation class
        self.sev_avg_prob = self.add_weight(name='sev_avg_prob', initializer='zeros')
        self.sev_count = self.add_weight(name='sev_count', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):

        # True objects
        true_obj = y_true[..., 0]
    
        # Predictions
        ## Classes
        pred_probs = tf.nn.softmax(y_pred[..., 5:7], axis=-1) # [TVS, Non-Tor]

        # Mask
        is_actual = tf.cast(true_obj > 0.5, tf.float32)

        # Class-Specific Logical Masks (Truth)
        # y_true is one-hot encoded: [1, 0] is TVS, [0, 1] is Severe
        actual_is_tvs = is_actual * y_true[..., 5]
        actual_is_sev = is_actual * y_true[..., 6]

        # Calculate average probability for cells that ARE actually TVS
        tvs_probs_at_truth = pred_probs[..., 0] * actual_is_tvs
        self.tvs_avg_prob.assign_add(tf.reduce_sum(tvs_probs_at_truth))
        self.tvs_count.assign_add(tf.reduce_sum(actual_is_tvs))

        # Calculate average probability for cells that ARE actually SEV
        sev_probs_at_truth = pred_probs[..., 1] * actual_is_sev
        self.sev_avg_prob.assign_add(tf.reduce_sum(sev_probs_at_truth))
        self.sev_count.assign_add(tf.reduce_sum(actual_is_sev))

    def result(self):
        return {"tvs_avg_prob": self.tvs_avg_prob / (self.tvs_count + 1e-7),
                "sev_avg_prob": self.sev_avg_prob / (self.sev_count + 1e-7)
        }

    def reset_state(self):
        # Reset all weights at the start of every epoch
        self.tvs_avg_prob.assign(0.0)
        self.tvs_count.assign(0.0)
        self.sev_avg_prob.assign(0.0)
        self.sev_count.assign(0.0)

def average_distance_error_km(y_true, y_pred):
    # Mask to only where an actual object exists
    obj_mask = tf.cast(y_true[..., 0] >= 0.5, tf.float32)

    # 1. Create a coordinate grid for the cells
    # grid_indices will have shape (64, 64)
    grid_x, grid_y = tf.meshgrid(tf.range(64, dtype=tf.float32), tf.range(64, dtype=tf.float32))

    # 2. Calculate global pixel coordinates (grid_index + offset)
    # y_true[..., 1] is x_offset, y_true[..., 2] is y_offset
    true_x_global = (grid_x + y_true[..., 1]) 
    true_y_global = (grid_y + y_true[..., 2])
    
    # We must use sigmoid on y_pred because it is linear
    pred_x_global = (grid_x + tf.sigmoid(y_pred[..., 1]))
    pred_y_global = (grid_y + tf.sigmoid(y_pred[..., 2]))

    # 3. Calculate Euclidean distance in "grid units"
    dist_grid = tf.sqrt(tf.square(true_x_global - pred_x_global) + 
                        tf.square(true_y_global - pred_y_global))
    
    # 4. Convert to KM (assuming each grid cell is 5km)
    dist_km = dist_grid * 5.0

    # Return mean distance only for cells with objects
    return tf.reduce_sum(dist_km * obj_mask) / (tf.reduce_sum(obj_mask) + 1e-7) 
