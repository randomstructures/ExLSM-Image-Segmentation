"""Provides tools to create a training dataset from a 3D microsopy dataset.
"""

import numpy as np
import random
import h5py
from tqdm import tqdm

# import tilingStrategy

def getRandomIndices(tiler, n_samples):
    assert n_samples<len(tiler), 'Cannot sample more than {} samples from this tiling'.format(len(tiler))
    sample_indices = np.random.choice( np.arange(len(tiler)), size=n_samples, replace=False) # choose n_samples random chunks from the volume
    return sample_indices

def getMeanSignalStrengths(tiler, indices):
    mean_signal_strengths = [ np.mean(tiler._cropAndPadAABB(tiler.image,tiler._getOutputTile(i))) for i in tqdm(indices) ]
    return mean_signal_strengths

def sampleMaskProportion(tiler, indices):
    sample_volume = np.prod(tiler.output_shape) # number of pixels in the sample volume
    sample_mask_proportion = [ 
        np.count_nonzero(tiler._cropAndPadAABB(tiler.mask,tiler._getOutputTile(i))) / sample_volume
         for i in tqdm(indices) ]

    return sample_mask_proportion


def thresholdedSampling(indices, sample_mean_signal, threshold, n_samples, object_ratio=0.5):
    """Choose samples from a volume based on their average signal strength.
    Samples are split in two groups by a mean signal threshold. 
    Random samples are drawn from each of the two groups so that the one exceeding the threshold maskes of the specified object ratio.

    Parameters
    ----------
    indices : list
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
    samples = np.random.choice(indices, size=n_samples, replace=False, p=proba)
    return samples

class Dataset():
    """Utility class that maintains a dataset in h5 format

    Creates or appends to a dataset in h5 file format.
    Corresponding unet input tiles and mask output tiles are stored in a group:
    dataset
        -position a
            -image a
            -mask a
        -position b
            -image b 
            -mask b
    

    """

    def __init__(self, dataset_path, append=True):
        super().__init__()
        if append:
            self.dataset_h5 = h5py.File(dataset_path)
        else:
            self.dataset_h5 = h5py.File(dataset_path, mode='x') # create new, fail if exists
        # keep track of existing groups
        if len(self.keys())>0:
            print('Opened dataset with {} preexisting items. Overwriting items with the same name.'.format(len(self.keys())))

    def keys(self):
        return self.dataset_h5.keys()

    def add_tiles(self, tiler, indices, preprocessingFunction= None, binarizeMask = False):
        for index in tqdm(indices, desc='Tiles added'):
            
            # fetch the data
            image, mask = tiler.getSlice(index), tiler.getMaskSlice(index)

            # preprocess if necessary
            if not preprocessingFunction is None:
                image = preprocessingFunction(image)
            if binarizeMask:
                mask = np.clip(mask,0,1)

            # check if the item allready exists
            if str(index) in self.keys():
                #print('Overwriting item {}'.format(index))
                del self.dataset_h5[str(index)] # delete the group

            # create the group and write image and masks datasets
            self.dataset_h5.create_group(str(index))
            self.dataset_h5[str(index)].create_dataset('image', data=image)
            self.dataset_h5[str(index)].create_dataset('mask', data=mask)

    def delete(self,key):
        if type(key) is int:
            key = str(key)
        assert key in self.keys(), 'Key not contained in dataset'
        del self.dataset_h5[key]

    def get(self, key):
        if type(key) is int:
            key = str(key)
        assert key in self.keys(), 'Key not contained in dataset'
        image = self.dataset_h5[key]['image']
        mask =  self.dataset_h5[key]['mask']
        return (image, mask)

    def close(self):
        self.dataset_h5.close()



