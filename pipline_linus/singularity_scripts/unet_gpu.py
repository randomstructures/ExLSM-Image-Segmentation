#%% Imports
import gc
import getopt
import os
import sys
import time

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from postprocess_cpu import hdf5_read, hdf5_write

import tensorflow as tf

#TODO copy the mosaic and the model py files into the singularity script folder
import tilingStrategy
import model


#%% method definitions
def masked_binary_crossentropy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true,2), K.floatx())
    score = K.mean(K.binary_crossentropy(y_pred*mask, y_true*mask), axis=-1)
    return score


def masked_accuracy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true,2), K.floatx())
    score = K.mean(K.equal(y_true*mask, K.round(y_pred*mask)), axis=-1)
    return score


def masked_error_pos(y_true, y_pred):
    mask = K.cast(K.equal(y_true,1), K.floatx())
    error = (1-y_pred) * mask
    score = K.sum(error) / K.maximum(K.sum(mask),1)
    return score


def masked_error_neg(y_true, y_pred):
    mask = K.cast(K.equal(y_true,0), K.floatx())
    error = y_pred * mask
    score = K.sum(error) / K.maximum(K.sum(mask),1)
    return score

# %% custom code for large volume segmentation
def preprocessImage(data):
    """This function preprocesses image data for the use with a pretrained unet model.
    The details of image preprocessing might vary over time. This method has to be updated manually to enshure that the unet performs as expected.
    Normally this method performs cliping and z-score normalization of pixel values with empirically determined values.

    Parameters
    ----------
    data : image tensor
        the raw image data

    Returns
    -------
    image tensor
        preprocessed image data.
    """
    # clip the values at 1400 so that everything above is saturated, convert to float to prevent issues with unsigned int format
    data = np.clip(data, 0, 1400).astype(np.float32)
    # shift by the empirical mean and divide by the empirical std dev to get data with a approx 0 centered distribution and unit variance
    # empirical values for mean and variance were obtained using image J on large_image_0724.h5
    data = np.divide( np.subtract(data, 140), 40 )
    return data

def tf_init():
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


def apply_unet(img):
    """This function creates the segmentation mask of the input image and returns it.
    Internally, a pretrained unet is loaded, the input data is preprocessed and split into tiles which are fed to the unet.
    The resulting mask tiles are reassembled and the mask is returned.

    Parameters
    ----------
    img : image tensor
        the raw input image to create a segmentation mask from.
    """
    # Load the pretrained unet Model file 
    #TODO locate the pretrained unet model file 
    model_path = 'C:\\Users\\Linus Meienberg\\Google Drive\\Janelia\\ImageSegmentation\\3D Unet\\retrain0727\\3d_unet_0728.h5'
    # Restore the trained model. Specify where keras can find custom objects that were used to build the unet
    unet = tf.keras.models.load_model(model_path, compile=False,
                                  custom_objects={"InputBlock" : model.InputBlock,
                                                    "DownsampleBlock" : model.DownsampleBlock,
                                                    "BottleneckBlock" : model.BottleneckBlock,
                                                    "UpsampleBlock" : model.UpsampleBlock,
                                                    "OutputBlock" : model.OutputBlock})

    # Instantiate a unet tiler and allocate a new mask tensor where the predictions are assembled
    #TODO Update the shapes to the ones used by the unet if necessary
    tiler = tilingStrategy.UnetTiler3D(img, mask=None, output_shape=(132,132,132), input_shape=(220,220,220) )

    # Sequentially process all tiles of the image
    start = time.time()
    for i in range(len(tiler)):
        # Read input slice from volume
        input_slice = preprocessImage(tiler.getSlice(i))
        # Add batch and channel dimension and feed to unet
        output_slice = unet.predict(input_slice[np.newaxis,:,:,:,np.newaxis])
        output_mask = np.argmax(output_slice, axis=-1)[0,...] # use argmax on channels and remove batch dimension
        #print('unet output mask shape : '.format(input_slice.shape))
        tiler.writeSlice(i, output_mask)
    end = time.time()
    # write a message on the console
    print('UNET finished\ntook {:.1f} s for {} tiles'.format(end-start,len(tiler)))

    K.clear_session() # close keras session
    gc.collect() # delete unused objects in memory

    return tiler.mask # return the assembled mask

#%% main code for executable
def main(argv):
    """
    Main function
    """
    hdf5_file = None
    location = []
    try:
        options, remainder = getopt.getopt(argv, "i:l:", ["input_file=","location="])
    except:
        print("ERROR:", sys.exc_info()[0]) 
        print("Usage: unet_gpu.py -i <input_hdf5_file> -l <location>")
        sys.exit(1)
    
    # Get input arguments
    for opt, arg in options:
        if opt in ('-i', '--input_file'):
            hdf5_file = arg
        elif opt in ('-l', '--location'):
            location.append(arg.split(","))
            location = tuple(map(int, location[0]))
        
    # Read part of the hdf5 image file based upon location
    if len(location):
        tf_init() # initialize tensorflow
        img = hdf5_read(hdf5_file, location)
        img_path = os.path.dirname(hdf5_file)
    else:
        print("ERROR: location need to be provided!")
        sys.exit(1)

    start = time.time()
    print('#############################')
    #img = unet_test(img=img)
    img = apply_unet(img=img) # Use a custom function that returns a mask tensor of the same shape as the input image
    hdf5_write(img, hdf5_file, location)
    end = time.time()
    print("DONE! 3D U-Net running time is {} seconds".format(end-start))
    

if __name__ == "__main__":
    main(sys.argv[1:])


# %%
