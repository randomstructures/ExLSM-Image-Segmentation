"""Collection of utilities to run 3D Unet
"""

#%% 

import os, pathlib
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from mayavi import mlab


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


#%% Load an example image 3d image

def _getImage(path, color_mode = 'grayscale'):
    """Load an image as a numpy array using the keras API

    Parameters
    ----------
    path : String
        path to the image
    color_mode : str, optional
        the color mode used to load the image, by default 'grayscale' which loads a single channel

    Returns
    -------
    tensor
        image tensor of format (x,y,c)
    """
    return keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path, color_mode))
#%%

def load_volume(directory):
    """Load a sclice of the 3D image dataset contained in a directory with the following structure:
    directory
        - image
            - image000
            - image001
            ...
        - mask
            - mask000
            - mask001
            ...
    where the images in each folder are ordered slices of the same 3D volume. 
    Images are assumed to contain a single chanel.

    Parameters
    ----------
    directory : string 
        path to the directory containing the slice of the 3D image

    Returns
    -------
    dict
        'shape' : The shape of the 3D volume
        'image' : Input Image tensor
        'mask'  : Target Image tensor
    """
    # Prepend common base directory
    input_dir = os.path.join(directory, 'image')
    target_dir = os.path.join(directory, 'mask')

    # The following is a multiline python generator expression !
    input_tensor = np.stack(
        [_getImage(os.path.join(input_dir, fname)) for fname in os.listdir(input_dir)]
    )
    target_tensor = np.stack(
        [_getImage(os.path.join(target_dir, fname)) for fname in os.listdir(target_dir)]
    )
    
    assert input_tensor.shape == target_tensor.shape, 'Image and mask need to have the same shape'
    
    output = {}
    output['shape'] = input_tensor.shape
    output['image'] = input_tensor
    output['mask']  = target_tensor

    return output

# %%
def show3DImage(image_tensor, channel=0, mode = 'image'):
    """Visualize a single channel 3D image using mayavi's isosurface plot

    Parameters
    ----------
    image_tensor : tensor
        tensor of rank 4 in format (x,y,z,c)
    channel : int, optional
        the channel to visualize, by default 0
    mode : str, optional
        the visualization mode. Set to {'image','mask'}
    """
    if mode is 'image':
        n_contours = 4
        transparent = True
    elif mode is 'mask':
        n_contours = 10
        transparent = False
    else:
        raise ValueError('Visualization mode undefined')

    mlab.figure(size=(500,500))
    plot = mlab.contour3d(image_tensor[...,channel], contours = n_contours, transparent=transparent)
    return mlab.gcf()

# %%
