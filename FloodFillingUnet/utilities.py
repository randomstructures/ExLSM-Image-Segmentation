"""Collection of utilities to run 3D Unet

CNN arithmetics
Data Input Pipeline

"""

#%% 

import os, pathlib
import random

import tensorflow as tf
from tensorflow import keras

import numpy as np
import scipy

#%% Import modules providing tools for image manipulation
import sys
sys.path.append('../tools/')
import deformation, affine 


def check_size(size, n_blocks):
    """Checks if a valid unet architecture with n blocks can be constructed from an image input 

    Parameters
    ----------
    size : int    
        side length of input volume (cube)
    n_blocks : int
        the number of blocks in the downsampling and upsampling path

    Returns
    -------
    boolean, int
        validity, size of output image (0 if false)
    """
    x = size
    outputs = []

    # Input Block
    x -= 4 # two convolution operations
    outputs.append(x)
    # downsampling
    if not x%2==0: # check if 2x2 max pooling tiles nicely
            #print('Input block: max pool input not divisible by 2')
            return False, 0
    x /= 2

    # downsampling
    for n in range(n_blocks):
        x -= 4 # two conv layers 3x3
        outputs.append(x) # store output dimension
        if not x%2==0: # check if 2x2 max pooling tiles nicely
            #print('Down {} max pool input {} not divisible by 2'.format(n+1,x))
            return False, 0
        x /= 2

    
    # bottleneck block
    x -= 4
    x *= 2

    # upsampling
    for n in range(n_blocks):
        skip = outputs.pop()
        if not (skip-x)%2==0:
            print('Up {} crop from {} to {} not centered'.format(n,skip,x))
            return False, 0
        x -= 4 # two conv layers 3x3
        x *= 2 # upsampling
    
    # output block
    skip = outputs.pop()
    if not (skip-x)%2==0:
        print('Output block: crop from {} to {} not centered'.format(skip,x))
        return False, 0
    x -=4

    #print('image size valid')
    if x>0:
        return True, x
    else:
        return False, 0

#%% Mask conversion tools
def applySoftmax(prediction):
    """Converts an output of logits to pseudo class probabilities using the softmax function

    Parameters
    ----------
    prediction : image tensor
        tensor where the last axis corresponds to different classes. Values are raw class logits.

    Returns
    -------
    image tensor
        tensor where the last axis corresponds to different classes. Values are pseudo probabilities.
    """
    return scipy.special.softmax(prediction, axis= -1)

def segmentationMask(prediction, restoreChannelDim=True):
    """Convert an image tensor with per class logit / probabilities to a segmentation mask using argmax.
    Each pixel holds the integer of the class number with the highest score.    

    Parameters
    ----------
    prediction : image tensor
        tensor where the last axis corresponds to different classes.

    Returns
    -------
    segmentation mask
        tensor with rank reduced by 1, each pixel holds the number of the class with the highest probability
    """
    seg =  np.argmax(prediction, axis = -1)
    if restoreChannelDim:
        seg = np.expand_dims(seg, axis= -1)
    return seg

def tf_elastic(image: tf.Tensor, mask: tf.Tensor):
    image_shape = image.shape
    mask_shape = mask.shape
    image, mask = tf.numpy_function(elasticDeformation, inp=[image,mask], Tout=(tf.float32,tf.int32))
    image.set_shape(image_shape)
    mask.set_shape(mask_shape)
    return image, mask

def tf_affine(image: tf.Tensor, mask: tf.Tensor):
    image_shape = image.shape
    mask_shape = mask.shape
    image, mask = tf.numpy_function(affineTransformation, inp=[image,mask], Tout=(tf.float32,tf.int32))
    image.set_shape(image_shape)
    mask.set_shape(mask_shape)
    return image, mask

def elasticDeformation(image, mask):
    """Apply the same elastic deformation to an image and it's associated mask

    Parameters
    ----------
    image : tensor
    mask : tensor

    Returns
    -------
    image, mask
        elasticaly deformed tensors
    """
    # We know that the mask region is equal or smaller than the image region
    # Generate a displacement Field for the image region
    displacementField = deformation.displacementGridField3D(image.shape)
    #displacementField = deformation.smoothedRandomField(image.shape, alpha=300, sigma=8)
    # Calculate the crop to extract the mask region from the image region
    #crop = tuple([(image.shape[i]-mask.shape[i])//2 for i in range(len(image.shape))])
    # Extract the part of the displacement field that applies to the mask
    #mask_displacementField = tuple(
    #            [dd[crop[0]:-crop[0] or None,crop[1]:-crop[1] or None, crop[2]:-crop[2] or None] 
    #            for dd in displacementField])
    # apply displacement fields
    image = deformation.applyDisplacementField3D(image, *displacementField, interpolation_order=1)
    #mask = deformation.applyDisplacementField3D(mask, *mask_displacementField, interpolation_order=0)
    mask = deformation.applyDisplacementField3D(mask, *displacementField, interpolation_order=0)
    return image, mask

def affineTransformation(image, mask):
    tm = affine.getRandomAffine()
    image = affine.applyAffineTransformation(image, tm, interpolation_order = 1)
    mask = affine.applyAffineTransformation(mask, tm, interpolation_order=0)
    return image, mask

def getTestImage(image_size = (220,220,220), mask_size= (132,132,132)):
    image = np.zeros(image_size, dtype=np.float32)
    # generate a stripe pattern
    for z in range(0,image_size[0],50):
        image[z:z+10,:,:] = np.ones((10,image_size[1],image_size[2]))
    
    # paint axes
    #image[:,:50,:50] = 1
    #image[:50,:,:50] = 1
    #image[:50,:50,:] = 1

    mask = image
    if not image_size == mask_size:
                crop = tuple([(image_size[i]-mask_size[i])//2 for i in range(len(image_size))])
                mask = mask[crop[0]:-crop[0] or None,crop[1]:-crop[1] or None, crop[2]:-crop[2] or None]
    return image[...,np.newaxis], mask[...,np.newaxis]