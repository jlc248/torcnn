import tensorflow as tf
import tensorflow.keras.backend as K

# We also have CSI in losses, but I think we need it defined as such to use
# it as a sample-weighted (or masked) metric
def csi(
    use_soft_discretization,
    hard_discretization_threshold=None,
    name="csi",
    index=0,
):
    def csi(target_tensor, prediction_tensor, sample_weight=None):

        # here we only care about 'index'.
        target_tensor = target_tensor[..., index]
        prediction_tensor = prediction_tensor[..., index]

        if hard_discretization_threshold is not None:
            prediction_tensor = tf.where(
                prediction_tensor >= hard_discretization_threshold, 1.0, 0.0
            )
        elif use_soft_discretization:
            prediction_tensor = K.sigmoid(prediction_tensor)

        target_tensor = tf.cast(target_tensor, tf.float32)

        true_positives = target_tensor * prediction_tensor
        false_positives = (1 - target_tensor) * prediction_tensor
        false_negatives = target_tensor * (1 - prediction_tensor)

        # Apply sample weights to each term
        if sample_weight is not None:
            channel_sample_weight = sample_weight[..., index]
            true_positives = true_positives * channel_sample_weight
            false_positives = false_positives * channel_sample_weight
            false_negatives = false_negatives * channel_sample_weight

        # Sum across all dimensions to get the total counts
        num_true_positives = K.sum(true_positives)
        num_false_positives = K.sum(false_positives)
        num_false_negatives = K.sum(false_negatives)

        denominator = (
            num_true_positives + num_false_positives + num_false_negatives + K.epsilon()
        )
        csi_value = num_true_positives / denominator

        return csi_value

    csi.__name__ = name

    return csi

#------------------------------------------------------------------------------------------------

def pod(use_soft_discretization,
        hard_discretization_threshold=None,
        name='pod',
        index=0,
):
    def pod(target_tensor, prediction_tensor, sample_weight=None):

        # here we only care about 'index'.
        target_tensor = target_tensor[..., index]
        prediction_tensor = prediction_tensor[..., index]

        if hard_discretization_threshold is not None:
            prediction_tensor = tf.where(prediction_tensor >= hard_discretization_threshold, 1., 0.)
        elif use_soft_discretization:
            prediction_tensor = K.sigmoid(prediction_tensor)

        target_tensor = tf.cast(target_tensor, tf.float32)

        true_positives = target_tensor * prediction_tensor
        false_negatives = target_tensor * (1 - prediction_tensor)

        # Apply sample weights to each term
        if sample_weight is not None:
            channel_sample_weight = sample_weight[..., index]
            true_positives = true_positives * channel_sample_weight
            false_negatives = false_negatives * channel_sample_weight

        # Sum across all dimensions to get the total counts
        num_true_positives = K.sum(true_positives)
        num_false_negatives = K.sum(false_negatives)

        denominator = (num_true_positives + num_false_negatives + K.epsilon())
        pod_value = num_true_positives / denominator

        return pod_value

    pod.__name__ = name

    return pod

#------------------------------------------------------------------------------------------------

def far(use_soft_discretization,
        hard_discretization_threshold=None,
        name='far',
        index=0,
):
    def far(target_tensor, prediction_tensor, sample_weight=None):

        # here we only care about 'index'.
        target_tensor = target_tensor[..., index]
        prediction_tensor = prediction_tensor[..., index]

        if hard_discretization_threshold is not None:
            prediction_tensor = tf.where(prediction_tensor >= hard_discretization_threshold, 1., 0.)
        elif use_soft_discretization:
            prediction_tensor = K.sigmoid(prediction_tensor)

        target_tensor = tf.cast(target_tensor, tf.float32)

        true_positives = target_tensor * prediction_tensor
        false_positives = (1 - target_tensor) * prediction_tensor

        # Apply sample weights to each term
        if sample_weight is not None:
            channel_sample_weight = sample_weight[..., index]
            true_positives = true_positives * channel_sample_weight
            false_positives = false_positives * channel_sample_weight

        # Sum across all dimensions to get the total counts
        num_true_positives = K.sum(true_positives)
        num_false_positives = K.sum(false_positives)

        denominator = (num_true_positives + num_false_positives + K.epsilon())
        far_value = num_false_positives / denominator

        return far_value

    far.__name__ = name

    return far

#------------------------------------------------------------------------------------------------

class AUC(tf.keras.metrics.Metric):
    def __init__(self, name='aupr', curve='PR', index=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.index = index
        self.auc_metric = tf.keras.metrics.AUC(name=f'_auc_internal_channel{self.index}', curve=curve)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Select the channel from y_true and y_pred
        y_true_channel = y_true[..., self.index]
        y_pred_channel = y_pred[..., self.index]

        channel_sample_weight = None
        if sample_weight is not None:
            channel_sample_weight = sample_weight[..., self.index]

        self.auc_metric.update_state(y_true_channel, y_pred_channel, channel_sample_weight)

    def result(self):
        return self.auc_metric.result()

    def reset_state(self):
        self.auc_metric.reset_state()
#------------------------------------------------------------------------------------------------

class BrierScore(tf.keras.metrics.Metric):
    def __init__(self, name='brier_score', index=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.index = index
        self.bs_metric = tf.keras.metrics.MeanSquaredError(name=f'_brier_score_internal_channel{self.index}')
        # Use Mean for weighted average of the squared errors
        self.total_squared_error = self.add_weight(name='total_squared_error', initializer='zeros', dtype=tf.float32)
        self.total_weight = self.add_weight(name='total_weight', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Select the channel from y_true and y_pred
        y_true_channel = tf.cast(y_true[..., self.index], dtype=tf.float32) # Shape: (batch_size, H, W)
        y_pred_channel = tf.cast(y_pred[..., self.index], dtype=tf.float32) # Shape: (batch_size, H, W)

        # Calculate squared errors for this channel
        squared_errors = tf.math.square(y_true_channel - y_pred_channel) # Shape: (batch_size, H, W)

        channel_sample_weight = None
        if sample_weight is not None:
            channel_sample_weight = sample_weight[..., self.index] # Shape: (batch_size, H, W)

            # Ensure channel_sample_weight is float32 for multiplication
            channel_sample_weight = tf.cast(channel_sample_weight, dtype=tf.float32)

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
        # Prevent division by zero
        if self.total_weight == 0:
            return 0.0
        return self.total_squared_error / self.total_weight

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
