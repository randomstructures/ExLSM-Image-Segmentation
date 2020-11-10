"""This module contains functions for post processing of segmentation masks or probability maps
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed

def clean_watershed(probability_map: np.ndarray, high_confidence_threshold = 0.98, low_confidence_threshold = 0.2):
    """This method generates a 'cleaned' segmentation mask from a probability map. It is assumed that a single object is present in the image together with some disconnected fragments that should be ignored.
    In a first step, high probability regions are taken as a seed point from which the foreground region is expanded by scikit-image's watershed algorithm.
    The region is expanded only where the foreground probability exceeds the low confidence threshold.

    Parameters
    ----------
    probability_map : np.ndarray
        Segmentation output holding foreground probabilities
    high_confidence_threshold : float
        seed regions must exceed this probability
    low_confidence_threshold : float
        the region is not expanded to parts of the map that lie below this probability
    """ 
    # Generate binary masks for high and low confidence areas
    low_confidence = probability_map > low_confidence_threshold
    high_confidence = probability_map > high_confidence_threshold
    # Set up an array of markers for skimage's watershed function
    #markers = np.zeros_like(probability_map, dtype=np.uint8)
    #markers[high_confidence] = 1
    # conversion to an integer array has the same effect !
    cleaned = watershed(-probability_map, high_confidence.astype(np.int) , mask=low_confidence)
    return cleaned > 0 # Return a boolean tensor