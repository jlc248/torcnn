import tensorflow as tf
import keras

#------------------------------------------------------------------------------------------------
@keras.saving.register_keras_serializable()
class csi(tf.keras.metrics.Metric):
    """
    A custom Keras Metric to calculate the Critical Success Index (CSI)
    for a specific channel of multi-channel predictions.
    """
    def __init__(self,
                 use_soft_discretization=False,
                 hard_discretization_threshold=None,
                 index=0,
                 name='csi',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.use_soft_discretization = use_soft_discretization
        self.hard_discretization_threshold = hard_discretization_threshold
        self.index = index
        
        # State variables to accumulate true positives, false positives, and false negatives
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros', dtype=tf.float32)
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros', dtype=tf.float32)
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Determine the rank of the tensors.
        rank = y_pred.shape.rank
       
        tf.debugging.assert_equal(tf.shape(y_true), tf.shape(y_pred), message="Shape mismatch between y_true and y_pred.") 
        
        if rank > 1:
            y_true = y_true[..., self.index]
            y_pred = y_pred[..., self.index]
            
            channel_sample_weight = None
            if sample_weight is not None:
                channel_sample_weight = sample_weight[..., self.index]
                # Ensure channel_sample_weight is float32 for multiplication
                channel_sample_weight = tf.cast(channel_sample_weight, dtype=tf.float32)
        else:
            channel_sample_weight = sample_weight

        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Apply discretization
        if self.hard_discretization_threshold is not None:
            y_pred = tf.cast(y_pred >= self.hard_discretization_threshold, tf.float32)
        elif self.use_soft_discretization:
            y_pred = tf.sigmoid(y_pred)


        # Calculate true positives and false positives for the current batch
        true_positives_batch = y_true * y_pred
        false_positives_batch = (1 - y_true) * y_pred
        false_negatives_batch = y_true * (1 - y_pred)

        # Apply sample weights consistently, regardless of rank
        if sample_weight is not None:
            true_positives_batch = true_positives_batch * channel_sample_weight
            false_positives_batch = false_positives_batch * channel_sample_weight
            false_negatives_batch = false_negatives_batch * channel_sample_weight

        # Update the state variables by summing the batch values
        self.true_positives.assign_add(tf.reduce_sum(true_positives_batch))
        self.false_positives.assign_add(tf.reduce_sum(false_positives_batch))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives_batch))

    def result(self):
        # Compute the final CSI value from the aggregated state
        denominator = self.true_positives + self.false_positives + self.false_negatives
        # Use tf.math.divide_no_nan to prevent division by zero
        csi_value = tf.math.divide_no_nan(self.true_positives, denominator)
        return csi_value

    def reset_state(self):
        # Reset the state variables at the start of each new epoch
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_soft_discretization": self.use_soft_discretization,
            "hard_discretization_threshold": self.hard_discretization_threshold,
            "index": self.index,
        })
        return config

#------------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class pod(tf.keras.metrics.Metric):
    """
    A custom Keras Metric to calculate the Probability of Detection (POD)
    for a specific channel of multi-channel predictions.
    """
    def __init__(self,
                 use_soft_discretization=False,
                 hard_discretization_threshold=None,
                 index=0,
                 name='pod',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.use_soft_discretization = use_soft_discretization
        self.hard_discretization_threshold = hard_discretization_threshold
        self.index = index
        
        # State variables to accumulate true positives and false negatives
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros', dtype=tf.float32)
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Determine the rank of the tensors.
        rank = y_pred.shape.rank

        tf.debugging.assert_equal(tf.shape(y_true), tf.shape(y_pred), message="Shape mismatch between y_true and y_pred.")

        if rank > 1:
            y_true = y_true[..., self.index]
            y_pred = y_pred[..., self.index]

            channel_sample_weight = None
            if sample_weight is not None:
                channel_sample_weight = sample_weight[..., self.index]
                # Ensure channel_sample_weight is float32 for multiplication
                channel_sample_weight = tf.cast(channel_sample_weight, dtype=tf.float32)
        else:
            channel_sample_weight = sample_weight

        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Apply discretization
        if self.hard_discretization_threshold is not None:
            y_pred = tf.cast(y_pred >= self.hard_discretization_threshold, tf.float32)
        elif self.use_soft_discretization:
            y_pred = tf.sigmoid(y_pred)


        # Calculate true positives and false positives for the current batch
        true_positives_batch = y_true * y_pred
        false_negatives_batch = (y_true) * (1 - y_pred)

        # Apply sample weights consistently, regardless of rank
        if sample_weight is not None:
            true_positives_batch = true_positives_batch * channel_sample_weight
            false_negatives_batch = false_negatives_batch * channel_sample_weight

        # Update the state variables by summing the batch values
        self.true_positives.assign_add(tf.reduce_sum(true_positives_batch))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives_batch))

    def result(self):
        # Compute the final FAR value from the aggregated state
        denominator = self.true_positives + self.false_negatives
        # Use tf.math.divide_no_nan to prevent division by zero
        pod_value = tf.math.divide_no_nan(self.true_positives, denominator)
        return pod_value

    def reset_state(self):
        # Reset the state variables at the start of each new epoch
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_soft_discretization": self.use_soft_discretization,
            "hard_discretization_threshold": self.hard_discretization_threshold,
            "index": self.index,
        })
        return config


#------------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class far(tf.keras.metrics.Metric):
    """
    A custom Keras Metric to calculate the False Alarm Rate (FAR)
    for a specific channel of multi-channel predictions.
    """
    def __init__(self,
                 use_soft_discretization=False,
                 hard_discretization_threshold=None,
                 index=0,
                 name='far',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.use_soft_discretization = use_soft_discretization
        self.hard_discretization_threshold = hard_discretization_threshold
        self.index = index
        
        # State variables to accumulate true positives and false positives
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros', dtype=tf.float32)
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Determine the rank of the tensors.
        rank = y_pred.shape.rank

        tf.debugging.assert_equal(tf.shape(y_true), tf.shape(y_pred), message="Shape mismatch between y_true and y_pred.") 

        if rank > 1:
            y_true = y_true[..., self.index]
            y_pred = y_pred[..., self.index]

            channel_sample_weight = None
            if sample_weight is not None:
                channel_sample_weight = sample_weight[..., self.index]
                # Ensure channel_sample_weight is float32 for multiplication
                channel_sample_weight = tf.cast(channel_sample_weight, dtype=tf.float32)
        else:
            channel_sample_weight = sample_weight

        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Apply discretization
        if self.hard_discretization_threshold is not None:
            y_pred = tf.cast(y_pred >= self.hard_discretization_threshold, tf.float32)
        elif self.use_soft_discretization:
            y_pred = tf.sigmoid(y_pred)


        # Calculate true positives and false positives for the current batch
        true_positives_batch = y_true * y_pred
        false_positives_batch = (1 - y_true) * y_pred

        # Apply sample weights consistently, regardless of rank
        if sample_weight is not None:
            true_positives_batch = true_positives_batch * channel_sample_weight
            false_positives_batch = false_positives_batch * channel_sample_weight

        # Update the state variables by summing the batch values
        self.true_positives.assign_add(tf.reduce_sum(true_positives_batch))
        self.false_positives.assign_add(tf.reduce_sum(false_positives_batch))

    def result(self):
        # Compute the final FAR value from the aggregated state
        denominator = self.true_positives + self.false_positives
        # Use tf.math.divide_no_nan to prevent division by zero
        far_value = tf.math.divide_no_nan(self.false_positives, denominator)
        return far_value

    def reset_state(self):
        # Reset the state variables at the start of each new epoch
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_soft_discretization": self.use_soft_discretization,
            "hard_discretization_threshold": self.hard_discretization_threshold,
            "index": self.index,
        })
        return config


#------------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AUC(tf.keras.metrics.Metric):
    def __init__(self, name='aupr', curve='PR', index=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.index = index
        self.curve = curve
        self.auc_metric = tf.keras.metrics.AUC(name=f'_auc_internal_channel{self.index}', curve=self.curve)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Determine the rank of the tensors.
        rank = y_pred.shape.rank

        if rank > 1:
            y_true = y_true[..., self.index]
            y_pred = y_pred[..., self.index]
            
            channel_sample_weight = None
            if sample_weight is not None:
                channel_sample_weight = sample_weight[..., self.index]
                # Ensure channel_sample_weight is float32 for multiplication
                channel_sample_weight = tf.cast(channel_sample_weight, dtype=tf.float32)
        else: 
            channel_sample_weight = sample_weight

        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Pass the correctly shaped tensors to the internal AUC metric.
        self.auc_metric.update_state(y_true, y_pred, sample_weight=channel_sample_weight)

    def result(self):
        return self.auc_metric.result()

    def reset_state(self):
        self.auc_metric.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({
            "curve": self.curve,
            "index": self.index,
        })
        return config
#------------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BrierScore(tf.keras.metrics.Metric):
    def __init__(self, name='brier_score', index=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.index = index
        # Use Mean for weighted average of the squared errors
        self.total_squared_error = self.add_weight(name='total_squared_error', initializer='zeros', dtype=tf.float32)
        self.total_weight = self.add_weight(name='total_weight', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Determine the rank of the tensors.
        rank = y_pred.shape.rank

        if rank > 1:
            y_true = y_true[..., self.index]
            y_pred = y_pred[..., self.index]

            channel_sample_weight = None
            if sample_weight is not None:
                channel_sample_weight = sample_weight[..., self.index]
                # Ensure channel_sample_weight is float32 for multiplication
                channel_sample_weight = tf.cast(channel_sample_weight, dtype=tf.float32)
        else:
            channel_sample_weight = sample_weight

        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Calculate squared errors for this channel
        squared_errors = tf.math.square(y_true - y_pred) # Shape: (batch_size, H, W)

        if sample_weight is not None:
            # Apply the sample weight directly to the squared errors
            weighted_squared_errors = tf.multiply(squared_errors, channel_sample_weight)

            # The total weight for the average
            # This also needs to be compatible with the shape of the data being summed
            current_batch_total_weight = tf.reduce_sum(channel_sample_weight)
        else:
            weighted_squared_errors = squared_errors
            current_batch_total_weight = tf.cast(tf.size(squared_errors), dtype=tf.float32) # Count total elements


        # Update the state variables
        self.total_squared_error.assign_add(tf.reduce_sum(weighted_squared_errors))
        self.total_weight.assign_add(current_batch_total_weight)

    def result(self):
        # Use tf.math.divide_no_nan to perform division and handle the zero case
        # This is the TensorFlow-native way to handle the logic from your `if` statement
        return tf.math.divide_no_nan(self.total_squared_error, self.total_weight)

    def reset_state(self):
        self.total_squared_error.assign(0.0)
        self.total_weight.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "index": self.index,
        })
        return config

#------------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ObsCt(tf.keras.metrics.Metric):
    """
    A custom metric that counts the number of hits.

    A hit is defined as a case where the prediction value is within a specified
    range [threshold1, threshold2) and the target value is 1.

    Args:
        threshold1: The lower bound (inclusive) for the prediction value.
        threshold2: The upper bound (exclusive) for the prediction value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """
    def __init__(self, threshold1, threshold2, name='obs_ct', index=0, dtype=tf.float32):
        super(ObsCt, self).__init__(name=name, dtype=dtype)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.index = index
        self.total_hits = self.add_weight(name='total_hits', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric with a new mini-batch.
        """

        # Determine the rank of the tensors.
        rank = y_pred.shape.rank

        if rank > 1:
            y_true = y_true[..., self.index]
            y_pred = y_pred[..., self.index]

        # Cast inputs to the correct dtype
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        # Apply the logic from the original function
        condition = tf.logical_and(y_pred >= self.threshold1, y_pred < self.threshold2)
        prediction_is_in_range = tf.where(condition, 1.0, 0.0)

        # Calculate hits for the current batch
        batch_hits = tf.reduce_sum(y_true * prediction_is_in_range)

        # Accumulate the total number of hits
        self.total_hits.assign_add(batch_hits)

    def result(self):
        """
        Computes and returns the final metric value.
        """
        # The result is simply the total accumulated hits.
        return self.total_hits

    def reset_state(self):
        """
        Resets all metric state variables.
        """
        self.total_hits.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold1":self.threshold1,
            "threshold2":self.threshold2,
            "index": self.index,
        })
        return config

#------------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FcstCt(tf.keras.metrics.Metric):
    """
    A custom metric that counts predictions within a specified range.

    This metric counts the number of predictions that are greater than or
    equal to threshold1 and less than threshold2.

    Args:
        threshold1: The lower bound (inclusive) for the prediction value.
        threshold2: The upper bound (exclusive) for the prediction value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """
    def __init__(self, threshold1, threshold2, name='fcst_ct', index=0, dtype=tf.float32):
        super(FcstCt, self).__init__(name=name, dtype=dtype)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.index = index
        self.total_valid_preds = self.add_weight(name='total_valid_preds', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric with a new mini-batch.
        
        Note: The `y_true` argument is required by the Keras API but is not
        used in this metric's calculation, as it only evaluates predictions.
        """

        # Determine the rank of the tensors.
        rank = y_pred.shape.rank
        
        if rank > 1:
            y_pred = y_pred[..., self.index]

        # Cast input to the correct dtype
        y_pred = tf.cast(y_pred, self.dtype)

        # Apply the logic from the original function
        condition = tf.logical_and(y_pred >= self.threshold1, y_pred < self.threshold2)
        prediction_is_in_range = tf.where(condition, 1.0, 0.0)

        # Calculate the number of valid predictions for the current batch
        batch_valid_preds = tf.reduce_sum(prediction_is_in_range)

        # Accumulate the total number of valid predictions
        self.total_valid_preds.assign_add(batch_valid_preds)

    def result(self):
        """
        Computes and returns the final metric value.
        """
        # The result is simply the total accumulated count of valid predictions.
        return self.total_valid_preds

    def reset_state(self):
        """
        Resets all metric state variables.
        """
        self.total_valid_preds.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold1":self.threshold1,
            "threshold2":self.threshold2,
            "index": self.index,
        })
        return config

#------------------------------------------------------------------------------------------------
