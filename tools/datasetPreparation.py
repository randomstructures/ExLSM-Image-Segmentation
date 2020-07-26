"""Provides tools to create a training dataset from a 3D microsopy dataset.
"""

import numpy as np
import random
import h5py, z5py

import matplotlib.pyplot as plt

import mosaic


def sampleMeanSignalStrength(tiler, n_samples):
    assert n_samples<len(tiler), 'Cannot sample more than {} samples from this tiling'.format(len(tiler))
    sample_index = np.random.choice( np.arange(len(tiler)), size=n_samples, replace=False) # choose n_samples random chunks from the volume
    sample_mean_signal = [ np.mean(tiler._cropAndPadAABB(tiler.image,tiler._getOutputTile(i))) for i in sample_index ]

    return sample_index, sample_mean_signal

def sampleMaskProportion(tiler, n_samples):
    assert n_samples<len(tiler), 'Cannot sample more than {} samples from this tiling'.format(len(tiler))
    sample_index = np.random.choice( np.arange(len(tiler)), size=n_samples, replace=False) # choose n_samples random chunks from the volume
    sample_mask_proportion = [ np.mean(tiler._cropAndPadAABB(tiler.mask,tiler._getOutputTile(i))) for i in sample_index ]

    return sample_index, sample_mask_proportion


def thresholdedSampling(sample_index, sample_mean_signal, threshold, n_samples, object_ratio=0.5):
    """Choose samples from a volume based on their average signal strength.
    Samples are split in two groups by a mean signal threshold. 
    Random samples are drawn from each of the two groups so that the one exceeding the threshold maskes of the specified object ratio.

    Parameters
    ----------
    sample_index : list
        list of sample indices
    sample_mean_signal : list
        list of mean signal strengths
    threshold : floatr
        signal strength threshold
    n_samples : int 
        the number of samples to draw
    object_ratio : float, optional
        the ratio of sampes exceeding the threshold in the returned list, by default 0.5
    """
    # clip value to prevent nummerical errors
    object_ratio = np.clip(object_ratio,1e-4,1-1e-4)
    # split the indeces
    is_high = [int(signal > threshold) for signal in sample_mean_signal]
    # weight each sample by it's class probability
    proba = np.multiply(is_high, object_ratio/np.sum(is_high)) + np.multiply(np.subtract(1,is_high), (1-object_ratio)/(len(is_high)-np.sum(is_high)))
    #print(proba)
    samples = np.random.choice(sample_index, size=n_samples, replace=False, p=proba)
    return samples

def createDataset():
    