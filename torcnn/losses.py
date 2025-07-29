# from https://github.com/karolzak/keras-unet/blob/master/keras_unet/losses.py
# and CIRA loss paper: https://arxiv.org/pdf/2106.09757.pdf

import tensorflow as tf
import tensorflow.keras.backend as K

def iou(use_as_loss_function, use_soft_discretization, which_class=-1,
        hard_discretization_threshold=None):
  def loss(target_tensor, prediction_tensor):
    if hard_discretization_threshold is not None:
      prediction_tensor = tf.where(prediction_tensor >= hard_discretization_threshold, 1., 0.)
    elif use_soft_discretization:
      prediction_tensor = K.sigmoid(prediction_tensor)

    #print('here',target_tensor.shape, prediction_tensor.shape)
    #print('here',target_tensor.dtype, prediction_tensor.dtype)
    target_tensor = tf.cast(target_tensor, tf.float32)
    if(which_class >= 0): #multiclass classification
      intersection_tensor = K.sum(target_tensor[..., which_class] * prediction_tensor[..., which_class],axis=(1, 2) )
    else: #binary classificaiton
      intersection_tensor = K.sum(target_tensor * prediction_tensor, axis=(1, 2)) #first axis (0th) is samples; axes 1 and 2 are X and Y
    union_tensor = (K.sum(target_tensor, axis=(1, 2)) + K.sum(prediction_tensor, axis=(1, 2)) - intersection_tensor)

    iou_value = K.mean(intersection_tensor / (union_tensor + K.epsilon()))

    if use_as_loss_function:
      return 1. - iou_value
    else:
      return iou_value

  return loss

 # How to use the loss function (with no discretization, for class 1)
 # when compiling a model:
 #loss_function = iou(
 #use_as_loss_function=True, use_soft_discretization=False, which_class=1,
 #   hard_discretization_threshold=None)

class MultiTargetBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, name="pixel_wise_multi_target_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        # Use NONE reduction for the inner BCE to get per-element losses.
        # We will then apply sample_weight and sum manually before Keras's final reduction.
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred, sample_weight=None):

        # Calculate the base BCE loss for each element
        loss = self.bce(y_true, y_pred) # Shape: (batch_size, H, W, C)

        # Apply sample weight if provided
        if sample_weight is not None:
            if sample_weight.shape != loss.shape:
                raise ValueError(
                    f"sample_weight shape {sample_weight.shape} "
                    f"does not match loss shape {loss.shape}. "
                    f"Expected {loss.shape} for channel-wise weighting."
                )
            loss = loss * sample_weight

        return tf.reduce_mean(loss)


def csi(use_as_loss_function, use_soft_discretization,hard_discretization_threshold=None,name='csi'):
  def loss(target_tensor, prediction_tensor, sample_weight=None):
    if hard_discretization_threshold is not None:
      prediction_tensor = tf.where(prediction_tensor >= hard_discretization_threshold, 1., 0.)
    elif use_soft_discretization:
      prediction_tensor = K.sigmoid(prediction_tensor)

    # Apply sample weights if provided
    if sample_weight is not None:
        target_tensor = target_tensor * sample_weight
        prediction_tensor = prediction_tensor * sample_weight

    target_tensor = tf.cast(target_tensor, tf.float32)
    num_true_positives = K.sum(target_tensor * prediction_tensor)
    num_false_positives = K.sum((1 - target_tensor) * prediction_tensor)
    num_false_negatives = K.sum(target_tensor * (1 - prediction_tensor))

    denominator = (num_true_positives + num_false_positives + num_false_negatives + K.epsilon())
    csi_value = num_true_positives / denominator

    if use_as_loss_function:
      return 1. - csi_value # -tf.math.log(csi_value) #1. - csi_value
    else:
      return csi_value

  loss.__name__ = name

  return loss

ALPHA=0.5
BETA=0.5

def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):

        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)

        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

        return 1 - Tversky

#from https://arxiv.org/pdf/2106.09757.pdf
def tversky_coeff(use_as_loss_function, use_soft_discretization, which_class,
                  false_positive_weight, false_negative_weight, hard_discretization_threshold=None):
  def loss(target_tensor, prediction_tensor):
    if(hard_discretization_threshold is not None):
      prediction_tensor = tf.where(prediction_tensor >= hard_discretization_threshold, 1., 0.)
    elif(use_soft_discretization):
      prediction_tensor = K.sigmoid(prediction_tensor)
    if(which_class is None):
      intersection_tensor = K.sum(target_tensor * prediction_tensor)
      false_positive_tensor = K.sum((1 - target_tensor) * prediction_tensor)
      false_negative_tensor = K.sum(target_tensor * (1 - prediction_tensor))
    else: #for multiclassification
      intersection_tensor = K.sum(target_tensor[..., which_class] * prediction_tensor[..., which_class],axis=(1, 2))
      false_positive_tensor = K.sum((1 - target_tensor[..., which_class]) * prediction_tensor[..., which_class],axis=(1, 2))
      false_negative_tensor = K.sum(target_tensor[..., which_class] * (1 - prediction_tensor[..., which_class]),axis=(1, 2))
    denominator_tensor = (intersection_tensor + false_positive_tensor + false_negative_tensor +K.epsilon())
    tversky_value = K.mean(intersection_tensor / denominator_tensor)

    if use_as_loss_function:
      return 1. - tversky_value
    else:
      return tversky_value
  return loss

def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# weighted binary cross-entropy
def wbce(y_true, y_pred, weight1=6.73785, weight0=0.5401): #weight1=15.501, weight0=0.5167) :
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return K.mean( logloss, axis=-1)
