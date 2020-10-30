    """This script prepares a Dataset of 3D tensors as described in the Dataset3D.py module.
    It can be used to train and evaluate machine learning models.
    """

#%% Imports

# Location of the custom model files
module_path = '../tools/'

# File name of the new or preexisting dataset
dataset_path= 'D:\\Janelia\\UnetTraining\\GapFilledMaskNetwork\\gapFilled_1012.h5'

import numpy as np 
import os, sys
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

import h5py
sys.path.append(module_path)
import tilingStrategy, Dataset3D, visualization


#%% Script Variables

# Path to the output directory of the script
output_directory = './test/'
os.makedirs(output_directory, exist_ok=True) 

########################################################################################
#                                      DATA IMPORT
########################################################################################
# A 'specimen' is an entire image dataset that was generated in a microscopy session
# 'Regions' are large crops that are manually extracted from specimens and show anatomical structures of interest
# Training examples are small crops automatically extracted from regions by this script.

# Indicate the storage location of the regions (Here a common base directory is assumed)
# Here we assume that regions are stored as hdf5 datasets where channels are in different groups

region_directory = "D:\\Janelia\\UnetTraining\\RegionCrops\\" # base directory of the region library
region_paths = ["A1\\A1.h5","A2\\A2.h5","B1\\B1.h5"] # Relative paths to the rehions from the base directory
regions = ["A1","A2","B1"] # A list of names that should be used as identifiers for the regions

# Retrieve Region data files
image_paths = [region_directory + region for region in region_paths] # prepend base dir to all regions in the list
regions_h5 = [h5py.File(ip, mode='r+') for ip in image_paths] # open the image regions

print('Accessing region files')
for i, h5 in enumerate(regions_h5):
    print(regions[i])
    print(h5.filename)
    print(h5['t0'].keys())
    print('')

# For every region the image and the mask channel have to be specified.
# References to the (large) image and mask arrays are stored in two dictionaries
image = {} # image[region] eg. image['A1'] should point to the image channel of the respective hdf5 file
mask = {}  # mask[region] should point to the ground truth channel of the hdf5 file
# sample entry : image['region'] = regions_h5[0]['t0/channel0']
image['A1'] = regions_h5[0]['t0/channel0']
mask['A1'] = regions_h5[0]['t0/channel2']
image['A2'] = regions_h5[1]['t0/channel0']
mask['A2'] = regions_h5[1]['t0/channel2']
image['B1'] = regions_h5[2]['t0/channel0']
mask['B1'] = regions_h5[2]['t0/channel2']

############################################################################
#                              DATA PREPROCESSING
############################################################################
# The following function will be applied globaly PER REGION to normalize the image data
# Neural Networks rely on normalized image data (eg pixel values roughly in [-1,1] and comparable between specimens)
# WARNING Once a network has been trained on input that has been preprocessed by a certain preprocessing function, all the input to the network has to be preprocess in the same way !!!
def preprocessImage(x, mean, std):
    # clip, and z normalize image
    x = x.astype(np.float32)
    x = np.clip(x,0, 1400)
    x = np.subtract(x, mean)
    x = np.divide(x, std)
    return x

# The following functions will be applied globaly PER REGION to preprocess the mask
def preprocessMask(x):
    # binarize mask and one hot encode
    x = np.clip(x,0,1)
    x = x.astype(np.int32)
    return x


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
    plt.scatter(mean_bins, log_counts) # scatter plot histogram data
    plt.plot(mean_bins,huber.predict(mean_bins.reshape(-1,1)), color = 'green') # line plot huber regressor fit
    plt.ylim([-1,25])
    plt.ylabel('log(Counts)')
    plt.xlabel('Pixel Intensity')
    plt.title('Approximation of Intensity Counts in region by Exponential Distribution Law\n' + region + ' log(P(I)) = ' + str(huber.coef_[0]) + ' *I+ ' + str(huber.intercept_))
    plt.savefig(output_directory + 'region_'+region+'_expFit.png')

    # Calculate scaling factor
    b_target = -np.log(10)/0.5 # EMPIRICAL Probability should reduce to 1/10th after 0.5 intensity units to get an intensity distribution within [0,1]
    scaling_factor = huber.coef_[0]/b_target

    # Scale the image
    x *= scaling_factor

    # Show scaled intensity distribution
    plt.hist(im.reshape(-1,1), bins = 1e3, range=[0,2], log=True)
    plt.xlabel('log(Counts)')
    plt.ylabel('Pixel Intensity')
    plt.title('Adjusted intensity distribution for region ' + region)
    plt.savefig(output_directory + 'region_'+region+'_scaled.png')

    # return scaled image
    return x

##############################################################################
#                           Training Example Mining
##############################################################################
# The mining strategy used is highly empirical
# Swap for other strategies or adjust values by trial and error

# Here, mask thresholded sampling is used.
# Examples are included with a high probability if their masks contain more objects voxels than a threshold 
above_threshold_ratio = 0.9 # What fraction of training examples should show above threshold object proportion
samples_per_region = 50 # How many samples to include per region in the list

# The size of the training examples is defined in the code



#%% Region Preprocessing

# Create or open the dataset
dataset = Dataset3D.Dataset(dataset_path)
print('preexisting keys : {}'.format(list(dataset.keys())))

# Handle Regions one by one
for region in regions:
    print('Processing Region '+region)
    im = image[region][...] # load image channel numpy array into working memory
    msk = mask[region][...] # load mask channel numpy array into working memory
    mean = np.mean(im)
    std = np.std(im)
    count, bins = np.histogram(im,bins=100,range=[0,2000])
    # Save and print region statistics
    np.savetxt(output_directory + 'region_{}_histogramm_counts'.format(region), count, delimiter=',', ) # save histogramm data to csv
    np.savetxt(output_directory + 'region_{}_histogramm_bins'.format(region), bins, delimiter=',') # save histogramm data to csv
    print(region)
    print('mean: {} std: {}'.format(mean,std))

    # Visualize the image channel histogram
    plt.figure()
    plt.hist(bins[:-1], bins, weights=count, log=True)
    plt.title('Image Channel Histogram for region ' + region)
    plt.ylabel('log(counts)')
    plt.ylabel('Signal intensity')
    plt.savefig(output_directory + 'region_'+region+'_hist.png')

    # Apply preprocessing to the image and mask arrays (copy in working memory)
    #im = preprocessImage( im, mean, std )
    im = preprocessImageV2(im, region)
    msk  = preprocessMask( msk )

    #########################################################
    # Mining training examples from the current region
    # Since most of the 3D Volume is empty space, a sampling strategy has to be used to specificaly target examples of interest.  

    # Use a Unet Tiler to divide the region into a grid of training examples. They are enummerated by the tiler class and can be referenced by their index.
    # 
    tiler = tilingStrategy.UnetTiler3D(image=im, mask=msk, output_shape=(132,132,132), input_shape=(220,220,220))
    # get a list of random indices from the region
    indices = Dataset3D.getRandomIndices(tiler, n_samples=200) # sample random tiles from the image volume

    # Calculate the fraction of foreground / object voxels in the training volumes 
    mask_proportions = Dataset3D.sampleMaskProportion(tiler, indices)
    # Plot the distribution of mask proportion
    #plt.figure()
    #plt.title('Mean mask proportion in sample masks')
    #_ = plt.hist(mask_proportions[region], bins=30)
    #plt.xlabel('mean mask proportion')
    #plt.ylabel('count')
    #plt.show()

    # Prepare a detaset with mask thresholded sampling
    # thresholded sampling samples indices above the threshold with a higher probability
    # n_samples
    mask_thresholded = Dataset3D.thresholdedSampling(indices, mask_proportions, threshold=0.001, above_threshold_ratio=above_threshold_ratio, n_samples=samples_per_region)

    
    # Add the selected training examples to the dataset
    dataset.add_tiles(tiler, mask_thresholded,
        key_prefix=region,
        cropMask=False, # DO NOT CROP MASK TO OUPUT SIZE (if data augumentation is used, this prevents artifacts from affine and elastic deformations)
        metadata={'region': region})

    ##########################################################

#%% Clean up

print('Dataset creation complete. Containes {} examples'.format(len(dataset)))

# Close dataset
dataset.close()

# %%
