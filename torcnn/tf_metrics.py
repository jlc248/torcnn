import tensorflow as tf
import tensorflow.keras.backend as K

#------------------------------------------------------------------------------------------------

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
        rank = tf.rank(y_pred).ndim

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

#------------------------------------------------------------------------------------------------

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
        rank = tf.rank(y_pred).ndim

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


#------------------------------------------------------------------------------------------------

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
        rank = tf.rank(y_pred).ndim

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


#------------------------------------------------------------------------------------------------

class AUC(tf.keras.metrics.Metric):
    def __init__(self, name='aupr', curve='PR', index=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.index = index
        self.auc_metric = tf.keras.metrics.AUC(name=f'_auc_internal_channel{self.index}', curve=curve)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Determine the rank of the tensors.
        rank = tf.rank(y_pred).ndim

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
#------------------------------------------------------------------------------------------------

class BrierScore(tf.keras.metrics.Metric):
    def __init__(self, name='brier_score', index=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.index = index
        # Use Mean for weighted average of the squared errors
        self.total_squared_error = self.add_weight(name='total_squared_error', initializer='zeros', dtype=tf.float32)
        self.total_weight = self.add_weight(name='total_weight', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Determine the rank of the tensors.
        rank = tf.rank(y_pred).ndim

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

#------------------------------------------------------------------------------------------------

#Observed counts. The numerator for reliability.
def obs_ct(threshold1,threshold2,name='reliability'):
  def obs_ct(target_tensor, prediction_tensor):

    condition = tf.logical_and(prediction_tensor >= threshold1, prediction_tensor < threshold2)

    prediction_tensor = tf.where(condition, 1., 0.)
    target_tensor = tf.cast(target_tensor, tf.float32)

    num_hits = K.sum(target_tensor * prediction_tensor)

    return num_hits

  obs_ct.__name__ = name

  return obs_ct
#------------------------------------------------------------------------------------------------
#Forecast counts. The denomenator for reliability.
def fcst_ct(threshold1,threshold2,name='reliability'):
  def fcst_ct(target_tensor, prediction_tensor):

    condition = tf.logical_and(prediction_tensor >= threshold1, prediction_tensor < threshold2)

    prediction_tensor = tf.where(condition, 1., 0.)

    #target_tensor = tf.cast(target_tensor, tf.float32)

    num_valid_preds = K.sum(prediction_tensor)
    #num_hits = K.sum(target_tensor * prediction_tensor)

    return num_valid_preds

  fcst_ct.__name__ = name

  return fcst_ct
#------------------------------------------------------------------------------------------------
