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

module_path = '../tools'
#'/nrs/dickson/lillvis/temp/linus/GPU_Cluster/modules/'
sys.path.append(module_path)

import tilingStrategy, preProcessing

#%% Script variables

dataset_path = '/mnt/d/Janelia/UnetTraining/RegionCrops/Q1/Q1.h5'
#'/nrs/dickson/lillvis/temp/linus/Unet_Evaluation/RegionCrops/Q1.h5'
input_key = 't0/channel1'
#'/mnt/d/Janelia/UnetTraining/RegionCrops/Q1/Q1.n5'
# use z5py for n5 format
#dataset = z5py.File(dataset_path)
# use h5py for h5 format

output_directory = "/mnt/d/Janelia/UnetTraining/test/" # directory for report files
output_path = "/mnt/d/Janelia/UnetTraining/test/test.h5"
#"/nrs/dickson/lillvis/temp/linus/GPU_Cluster/20201118_MultiWorkerSegmentation/Q1_mws.h5"
#"/mnt/d/Janelia/UnetTraining/test.h5"
output_key = "t0/test3"

side_length = 132 * 3
chunk_shape = (side_length,side_length,side_length) # Size of the subvolumes delegated to each worker. Ideally this is a multiple of the unet output size

binary = False

precalculateScalingFactor = True
n_tiles = 10 # number of tiles to randomly sample for calculation of the scaling factor

#%% Open input dataset and define tiling

dataset = h5py.File(dataset_path, mode='r')
image = dataset[input_key]
# remember the shape of the image
image_shape = image.shape

tiling = tilingStrategy.RectangularTiling(image_shape, chunk_shape=chunk_shape)
print('Jobs are created based on a tiling of ' + str(tiling.shape) + ', ' + str(len(tiling)) + ' tiles in total.')

if(precalculateScalingFactor):
    # For very large images: Calculate a common scaling factor by randomly subsampling the input image
    def getTile(image, tile):
        return image[tile[0]:tile[1], tile[2]:tile[3], tile[4]:tile[5]]

    indices = np.arange(len(tiling))
    subset = np.random.choice(indices, replace=False, size=n_tiles)
    sf = [preProcessing.calculateScalingFactor(getTile(image, tiling.getTile(index)), output_directory=output_directory, filename='fit'+str(index)) for index in subset]
    mean_sf = np.mean(sf)
    print('tile-wise scaling factor' + str(sf))
    print('Precalculated a scaling factor of {} based on {}/{} tiles'.format(mean_sf, n_tiles, len(tiling)))

# Release Resources
dataset.close()





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
for i in range(7,11):
    tile = tiling.getTile(i)
    tile = str(tile).replace('(','').replace(')','')
    imshape = str(image_shape).replace('(','').replace(')','')
    # Command line argument to invoke volume segmentation:
    # bsub -J jobname -n 5 -gpu "num=1" -q gpu_rtx -o jobname.log python volumeSegmentation.py -l loc,at,i,o,n,n
    # bsub -J lsegtot -n 5 -gpu "num=1" -q gpu_rtx -o segtot.log python volumeSegmentation.py
    jobname = job_prefix + str(i)
    logfile = jobname + '.log'

    # debug on home desktop
    arglist = ['python','volumeSegmentation.py','-l',tile,'--image_shape',imshape,'--scaling',str(mean_sf)]
    print(str(arglist))
    jobs.append(subprocess.Popen(arglist))

    # Construct command line argument for janelia's cluster job submission system
    #arglist = ['bsub','-J',jobname,'-n','5','-gpu', '\"num=1\"', '-q', 'gpu_rtx', '-o', logfile, 'python', 'volumeSegmentation.py']
    #if(precalculateScalingFactor):
    #    arglist.extend(['--scaling', str(mean_sf)])
    #arglist.extend(['-l', tile])
    
    #jobs.append(
    #    subprocess.Popen(arglist)
    #)


# %%
