""" This script applies a pretrained model file to a large image volume saved in hdf5 format
"""

#%% Script variables

# The path where custom modules are located
module_path = '/nrs/dickson/lillvis/temp/linus/GPU_Cluster/modules/'

##IMAGE I/O
# Specify the path to the image volume stored as h5 file
image_path = '/nrs/dickson/lillvis/temp/linus/Unet_Evaluation/RegionCrops/Q1.h5'
# Specify the group name of the image channel
image_channel_key = 't0/channel1'
# Specify the file name and the group name under which the segmentation output should be saved (this can also be the input file to which a new dataset is added)
output_directory = "/nrs/dickson/lillvis/temp/linus/GPU_Cluster/20201105_DeepenUnet/train1/eval50/"
output_path = "/nrs/dickson/lillvis/temp/linus/GPU_Cluster/20201105_DeepenUnet/train1/eval50/Q1_seg.h5"
output_channel_key = 't0/train1_epoch50'
# Specify wheter to output a binary segmentation mask or an object probability map
binary = False

## Model File 
# Specify the path to the pretrained model file
model_path = '/nrs/dickson/lillvis/temp/linus/GPU_Cluster/20201105_DeepenUnet/train1/deep50.h5'


#%% Imports 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.linear_model import HuberRegressor

import os, sys, time
sys.path.append(module_path)

import tilingStrategy, metrics
import utilities, model

import h5py
from tqdm import tqdm

# Implement the models preprocessing function
# Use empiricaly collected values for mean and std or calculate them for the entire input image
def preprocessImageV2(x, region):
    """This Preprocessing function scales the pixel intensities, such that the majority of pixel values lie in the intervall [0,1]

    The intensity distribution of the image is assumed to be of the form 
        P(I) = P_0 * exp(-b*I) 
            or in log form
        log(P(I)) = log(P_0) - b*I
        I : pixel intensity
        P(I) : probability / relative counts in histogram

    For a given image b is estimated using Huber regression (outlier robust linear regression, implemented in scikit learn)

    The decay rate b is adjusted to be comparable between samples by scaling the intensity values I
        scaling_factor = b_measured/b_target

    Empirically, a target decay rate of b_target = ln(10)/0.5 was chosen. This corresponds to a reduction of intensity counts by a factor of 10, every 0.5 Intensity units in the histogram.


    Parameters
    ----------
    x : [type]
        [description]
    """
    x = x.astype(np.float32)
    # calculate intensity distribution
    counts, bins = np.histogram(x, bins=1000, range=[0,4000]) # EMPIRICAL the majority of intensity values should be within this range for ALL imaged regions!
    # Calculate mean bin value and log counts
    mean_bins = (bins[:-1] + bins[1:])/2
    log_counts = np.log(counts)
    # Drop all bins with zero count (gives runnaway when taking log counts / not informative)
    mean_bins = mean_bins[np.isfinite(log_counts)]
    log_counts = log_counts[np.isfinite(log_counts)]
    # Instantiate and fit the Huber Regressor
    huber = HuberRegressor() # Use sklearns default values -> fits intercept, epsilon = 1.35, alpha = 1e-4
    huber.fit(mean_bins.reshape(-1,1), log_counts) # sklearn X,y synthax where X is a matrix (samples x observation) and y a vector (samples,) of target values

    # Show exponential Fit
    plt.figure()
    plt.scatter(mean_bins, log_counts) # scatter plot histogram data
    plt.plot(mean_bins,huber.predict(mean_bins.reshape(-1,1)), color = 'green') # line plot huber regressor fit
    plt.ylim([-1,25])
    plt.ylabel('log(Counts)')
    plt.xlabel('Pixel Intensity')
    plt.title('Approximation of Intensity Counts by Exponential Distribution\n' + region + ' log(P(I)) = ' + str(huber.coef_[0]) + ' *I+ ' + str(huber.intercept_))
    plt.savefig(output_directory + 'region_'+region+'_expFit.png')

    # Calculate scaling factor
    b_target = -np.log(10)/0.5 # EMPIRICAL Probability should reduce to 1/10th after 0.5 intensity units to get an intensity distribution within [0,1]
    scaling_factor = huber.coef_[0]/b_target

    # Scale the image
    x *= np.array(scaling_factor)
    print('scaling region ' + region + ' by ' + str(scaling_factor))

    # Show scaled intensity distribution
    plt.figure()
    plt.hist(x.reshape(-1,1), bins = 500, range=[0,2], log=True)
    plt.xlabel('log(Counts)')
    plt.ylabel('Pixel Intensity')
    plt.title('Adjusted intensity distribution for region ' + region)
    plt.savefig(output_directory + 'region_'+region+'_scaled.png')

    # return scaled image
    return x


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
if(binary):
    mask = output_h5.require_dataset(output_channel_key, shape=image.shape , dtype=np.uint8) # Use integer tensor to save memory
else:
    mask = output_h5.require_dataset(output_channel_key, shape=image.shape, dtype = np.float32)

# Check if image mean and std need to be calculated
#if (preprocessing_mean is None):
#    preprocessing_mean = np.mean(image)
#if (preprocessing_std is None):
#    preprocessing_std = np.std(image)

# Apply preprocessing globaly !
image = preprocessImageV2(image[...], region='Q1')

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
tiler = tilingStrategy.UnetTiler3D(image, mask=mask, output_shape=(36,36,36 ), input_shape=(220,220,220))

#%% Perform segmentation
start = time.time()
for i in tqdm(range(len(tiler)), desc='tiles processed'): # gives feedback on progress of for loop
    # Read input slice from volume
    # processing was applied globaly
    #input_slice = preprocessImage(tiler.getSlice(i), preprocessing_mean, preprocessing_std)
    input_slice = tiler.getSlice(i)
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
