"""
Segment a very large image using multiple workers on Janelia's cluster
11/16/2020
Linus Meienberg
"""

#%% Imports
import sys
import subprocess
import numpy as np
import z5py, h5py

sys.path.append('../tools/')

import tilingStrategy

#%% Script variables

dataset_path = '/mnt/d/Janelia/UnetTraining/RegionCrops/Q1/Q1.n5'

dataset = z5py.File(dataset_path)

ds = dataset['setup0/timepoint0/s0']

tiling = tilingStrategy.RectangularTiling(ds.shape, chunk_shape=(200,200,200))

dataset.close()
# %% Set output location
outds = h5py.File("/mnt/d/Janelia/UnetTraining/test.h5", mode='a')
del outds['t0/train1_epoch50']
outds.create_dataset('t0/train1_epoch50', shape=(512,1024,1024), dtype=np.float32)


# %% Dispatch jobs on subvolumes
jobs = []
for i in range(2):
    tile = tiling.getTile(i)
    tile = str(tile).replace('(','').replace(')','')
    jobs.append(subprocess.Popen(['python','volumeSegmentation.py','-l',tile]))
# %%
