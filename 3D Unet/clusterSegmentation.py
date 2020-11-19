"""
Segment a very large image using multiple workers on Janelia's cluster
11/16/2020
Linus Meienberg
"""

#%% Imports
import sys
import subprocess
import numpy as np
#import z5py
import h5py

module_path = '/nrs/dickson/lillvis/temp/linus/GPU_Cluster/modules/'
sys.path.append(module_path)

import tilingStrategy

#%% Script variables

dataset_path = '/nrs/dickson/lillvis/temp/linus/Unet_Evaluation/RegionCrops/Q1.h5'
#'/mnt/d/Janelia/UnetTraining/RegionCrops/Q1/Q1.n5'
# use z5py for n5 format
#dataset = z5py.File(dataset_path)
# use h5py for h5 format
dataset = h5py.File(dataset_path, mode='r')
image = dataset['t0/channel1']
# remember the shape of the image
image_shape = image.shape
#dataset['setup0/timepoint0/s0']
# Release Resources
dataset.close()

output_path = "/nrs/dickson/lillvis/temp/linus/GPU_Cluster/20201118_MultiWorkerSegmentation/Q1_mws.h5"
#"/mnt/d/Janelia/UnetTraining/test.h5"
output_key = "t0/test1"

tiling = tilingStrategy.RectangularTiling(image_shape, chunk_shape=(500,500,500))
print('Jobs are created based on a tiling of ' + str(tiling.shape))

binary = True



# %% Allocate a hdf5 dataset for the segmentation output
output_file = h5py.File(output_path, mode='a')

if(output_key in output_file):
    print('overwritting existing dataset')
    del output_file[output_key]

if(binary):
    output_file.create_dataset(name=output_key, shape=image_shape, dtype=np.uint8)
else:
    output_file.create_dataset(name=output_key, shape=image_shape, dtype=np.float32)

output_file.close()


# %% Dispatch jobs on subvolumes

job_prefix = 'l_mws'

jobs = []
#for i in range(len(tiling)):
for i in range(len(tiling)):
    tile = tiling.getTile(i)
    tile = str(tile).replace('(','').replace(')','')
    #jobs.append(subprocess.Popen(['python','volumeSegmentation.py','-l',tile]))
    # Command line argument to invoke volume segmentation:
    # bsub -J jobname -n 5 -gpu "num=1" -q gpu_rtx -o jobname.log python volumeSegmentation.py -l loc,at,i,o,n,n
    # bsub -J lsegtot -n 5 -gpu "num=1" -q gpu_rtx -o segtot.log python volumeSegmentation.py
    jobname = job_prefix + str(i)
    logfile = jobname + '.log'
    jobs.append(
        subprocess.Popen(
            ['bsub','-J',jobname,'-n','5','-gpu', '\"num=1\"', '-q', 'gpu_rtx', '-o', logfile, 'python', 'volumeSegmentation.py','-l', tile]
        )
    )



