"""A collection of evaluation metrics to assess the performance in 3D image segmentation tasks

   Linus Meienberg
   July 2020
"""
#%%
import numpy as np
from tensorflow.keras import backend as K

def intersection_over_union(true_mask, predicted_mask, num_classes=2, smooth=1):
    """Calculate the intersection over union metric for two sparse segmentation masks.

    Parameters
    ----------
    true_mask : tensor
        true segmentation mask in format (b,x,y,z,1) or (b,x,y,z)
    predicted_mask : tensor
        predicted segmentation mask in format (b,x,y,z,1) or (b,x,y,z)
    num_classes : int, optional
        the number of classes present in the segmentation mask, by default 2
    smooth : float, optional
        constant added to intersection and union. Smooths iou score for very sparse classes.

    Returns
    -------
    list
        the iou for each predicted class
    """
    if true_mask.shape[-1] == 1:
        true_mask = true_mask[...,0]
    if predicted_mask.shape[-1] == 1:
        predicted_mask = predicted_mask[...,0]

    assert true_mask.shape == predicted_mask.shape, 'segmentation masks do not match: shapes {} and {}'.format(true_mask.shape,predicted_mask.shape)

    iou = []
    # calculate iou for every class
    for c in range(num_classes):
        y_true = true_mask == c
        y_pred = predicted_mask == c
        intersection = np.sum(y_true & y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        iou.append((intersection+smooth)/(union+smooth))
    
    return iou

def keras_IoU(num_classes = 2, smooth=1):
    """Return a callable / metric that returns the mean IoU for a semantic segmentation task.

    Parameters
    ----------
    num_classes : int, optional
        the number of classes, by default 2
    smooth : int, optional
        a constant used to smooth IoU values in very rare cases, by default 1
    """
    def IoU(y_true, y_pred):
        """Returns the mean IoU for a semantic segmentation task.

        Parameters
        ----------
        y_true : tf.Tensor
            a sparse categorical ground thruth segmentation mask (x,y,z,)
        y_pred : tf.Tensor
            multichannel logit predictions for each category 
        """
        # convert the model output to a sparse segmentation mask (use argmax on channel axis since argmax on logits and pseudoprobabilities is the same)
        mask_pred = K.argmax(y_pred, axis=-1)
        iou = []
        # calculate iou for every class
        for c in range(num_classes):
            target = y_pred == c # pixels where true segmentation mask indicates class c
            prediction = mask_pred == c # pixels where predicted mask indicates class c
            intersection = np.sum(target & prediction)
            union = np.sum(target) + np.sum(prediction) - intersection
            iou.append((intersection+smooth)/(union+smooth))
        return K.mean(iou)

    return IoU
# %%