""" This script applies a pretrained model file to a large image volume saved in hdf5 format
"""

#%% Script variables

# The path where custom modules are located
module_path = '../tools'

##IMAGE I/O
# Specify the path to the image volume stored as h5 file
image_path = "C:\\Users\\Linus Meienberg\\Documents\\ML Datasets\\FruSingleNeuron_20190707\\large_image_0724.h5"
# Specify the group name of the image channel
image_channel_key = 't0/channel0'
# Specify the file name and the group name under which the segmentation output should be saved (this can also be the input file to which a new dataset is added)
output_path = "C:\\Users\\Linus Meienberg\\Documents\\ML Datasets\\FruSingleNeuron_20190707\\seg_output.h5"
output_channel_key = 't0/unet'
# Specify wheter to output a binary segmentation mask or an object probability map
binary = True

## Model File 
# Specify the path to the pretrained model file
model_path = 'C:\\Users\\Linus Meienberg\\Google Drive\\Janelia\\ImageSegmentation\\3D Unet\\retrain0727\\3d_unet_0728.h5'

# Implement the models preprocessing function
# Use empiricaly collected values for mean and std or calculate them for the entire input image
def preprocessImage(data, mean, std):
    data = np.clip(data, 0, 2000).astype(np.float32)
    data = np.divide( np.subtract(data, mean), std )
    return data

# Use constant value here or set None if they should be updated to global mean / std of entire input image
preprocessing_mean = None
preprocessing_std = None

#%% Imports 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf

from importlib import reload
import utilities, model

import os, sys, time
sys.path.append('../tools/')
import tilingStrategy, metrics

import h5py
from tqdm import tqdm
#%% Setup

# Fix for tensorflow-gpu issues that I found online... (don't ask me what it does)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#%% Data Input

print('Opening hdf5 file')
image_h5 = h5py.File(image_path, mode='r+') # Open h5 file with read / write access
print(image_h5.keys()) # Show Groups (Folders) in root Group of the h5 archive
image = image_h5[image_channel_key] # Open the image dataset

if(output_path == image_path):
    output_h5 = image_h5
else:
    output_h5 = h5py.File(output_path, mode='a') # Open h5 file, create if it does not exist yet
print('Segmentation Output is written to ' + output_path + '/' + output_channel_key + ' overwriting previous data if it exists')
mask = output_h5.require_dataset(output_channel_key, shape=image.shape , dtype=np.uint8)

# Check if image mean and std need to be calculated
if (preprocessing_mean is None):
    preprocessing_mean = np.mean(image)
if (preprocessing_std is None):
    preprocessing_std = np.std(image)

#%% Load Model File
# Restore the trained model. Specify where keras can find custom objects that were used to build the unet
unet = tf.keras.models.load_model(model_path, compile=False,
                                  custom_objects={"InputBlock" : model.InputBlock,
                                                    "DownsampleBlock" : model.DownsampleBlock,
                                                    "BottleneckBlock" : model.BottleneckBlock,
                                                    "UpsampleBlock" : model.UpsampleBlock,
                                                    "OutputBlock" : model.OutputBlock})

print('The unet works with\ninput shape {}\noutput shape {}'.format(unet.input.shape,unet.output.shape))

# Set up a unet tiler for the input image
tiler = tilingStrategy.UnetTiler3D(image, mask=mask, output_shape=(132,132,132), input_shape=(220,220,220))

#%% Perform segmentation
start = time.time()
for i in tqdm(range(len(tiler)), desc='tiles processed'): # gives feedback on progress of for loop
    # Read input slice from volume
    input_slice = preprocessImage(tiler.getSlice(i), preprocessing_mean, preprocessing_std)
    # Add batch and channel dimension and feed to unet
    output_slice = unet.predict(input_slice[np.newaxis,:,:,:,np.newaxis])
    # Convert logits to binary segmentation mask or object probability map
    if(binary):
        output_mask = np.argmax(output_slice, axis=-1)[0,...] # use argmax on channels and remove batch dimension
    else:
        output_mask = tf.nn.softmax(output_slice, axis=-1)[0,...,1] # use softmax on channels, take object cannel and remove batch dimension
    # Write slice to canvas
    tiler.writeSlice(i, output_mask)
end = time.time()
print('\ntook {:.1f} s for {} iterations'.format(end-start,len(tiler)))

# Close dataset
image_h5.close()
output_h5.close()



# %%
