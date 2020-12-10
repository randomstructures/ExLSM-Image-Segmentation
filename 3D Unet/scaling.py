#%% Imports
import sys
import numpy as np
#import z5py
import h5py


import concurrent.futures

module_path = '../tools'
#'/nrs/dickson/lillvis/temp/linus/GPU_Cluster/modules/'
sys.path.append(module_path)

import tilingStrategy, preProcessing

#%% Script variables

dataset_path = 'D:/Janelia/UnetTraining/RegionCrops/Q1/Q1.h5'
#'/nrs/dickson/lillvis/temp/linus/Unet_Evaluation/RegionCrops/Q1.h5'
input_key = 't0/channel1'
#'/mnt/d/Janelia/UnetTraining/RegionCrops/Q1/Q1.n5'
# use z5py for n5 format
#dataset = z5py.File(dataset_path)
# use h5py for h5 format

side_length = 132 * 3
chunk_shape = (side_length,side_length,side_length) # Size of the subvolumes delegated to each worker. Ideally this is a multiple of the unet output size

binary = False
n_tiles = 10 # number of tiles to randomly sample for calculation of the scaling factor

#%% Open input dataset and define tiling

dataset = h5py.File(dataset_path, mode='r')
image = dataset[input_key]
# remember the shape of the image
image_shape = image.shape

tiling = tilingStrategy.RectangularTiling(image_shape, chunk_shape=chunk_shape)

# For very large images: Calculate a common scaling factor by randomly subsampling the input image
def getTile(image, tile):
    return image[tile[0]:tile[1], tile[2]:tile[3], tile[4]:tile[5]]
# Choose a random subset of chunks
indices = np.arange(len(tiling))
subset = np.random.choice(indices, replace=False, size=n_tiles)

def calculate_sf(index):
    sf = preProcessing.calculateScalingFactor(getTile(image, tiling.getTile(index)), output_directory=None)
    return sf

#%%
# calculate the scaling factor for each subvolume
# use parallel subprocesses to speed up computation
scaling_factors = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    for index, sf in zip(subset, executor.map(calculate_sf, subset)):
        print(sf)
        scaling_factors.append(sf)

# Release Resources
dataset.close()

print("mean sf {}".format(np.mean(sf)))
print("median sf {}".format(np.median(sf)))
# %%
