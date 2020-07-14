"""Collection of utilities to run 3D Unet
"""

#%% 

import os, pathlib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import matplotlib.pyplot as plt

from mayavi import mlab

# Import modules providing tools for image manipulation
import sys
sys.path.append('../tools/')
import mosaic, deformation, affine 


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
    return keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path, color_mode=color_mode))
#%% Tools for dataset preparation

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

def generateVariants(images, masks, variants,  elasticDeformation=True, affineTransform=True):
    """Generate a given number of augumentes images from a list of 3D image tensors. Random transformations are reused on each input image befor new ones are drawn.

    Parameters
    ----------
    images : list
        list of 3D image tensors in format (x,y,z,c)
    masks : list 
        list of 3D segmentation masks in format (x,y,z,1)
    variants : int
        the number of augumented image, mask pairs to create
    elasticDeformation : bool, optional
        wheter to apply elastic deformation, by default True
    affineTransform : bool, optional
        wheter to apply random rotation and scaling, by default True

    Returns
    -------
    tuple   
        tuple of lists holding corresponding augumented 3D images and segmentation masks
    """
    # we want to create #variant images from #images originals minimizing the number of 
    image_variants = []
    mask_variants = []
    # Check if new variants are still needed
    while len(image_variants) < variants:
        # Draw new random operations
        if elasticDeformation:
            image_shape = images[0].shape[:-1] # All images have the same dimensions, exclude channel axis
            displacementField = deformation.displacementGridField3D(image_shape=image_shape)
        if affineTransform:
            transformationMatrix = affine.getRandomAffine()

        # Apply them to all originals
        for im, mask in zip(images,masks):
            # Process a new image and a mask only as long as we need aditional variants
            if len(image_variants) < variants:
                if elasticDeformation:
                    im = deformation.applyDisplacementField3D(im, *displacementField, interpolation_order = 1)
                    mask = deformation.applyDisplacementField3D(mask, *displacementField, interpolation_order = 0)
                if affineTransform:
                    im = affine.applyAffineTransformation(im, transformationMatrix, interpolation_order = 1)
                    mask = affine.applyAffineTransformation(mask, transformationMatrix, interpolation_order=0)

                image_variants.append(im) # Also updates the length of the list
                mask_variants.append(mask) 

    return image_variants, mask_variants


class Dataset3D(keras.utils.Sequence):

    def  __init__(self, batch_size, batches, mask_crop, images, masks, augument=False, elastic=False, affine=False):
        """Custom keras Sequence to simplify training. Performs on the fly data augumentation if specified.
        The size of the segmentation masks is reduced to fit the output of the unet by a central crop.

        Parameters
        ----------
        batch_size : int
            number of image mask pairs in a batch
        batches : int
            number of batches in the dataset
        mask_crop : int
            number of pixels to crop from each border of the segmentation mask
        images : list
            list of 3D image tensors in format (x,y,z,c)
        masks : list 
            list of 3D segmentation masks in format (x,y,z,1)
        augument : bool, optional
            wheter to augument the images if false the original data is used, by default False
        elastic : bool, optional
            wheter to use elastic deformation, by default False
        affine : bool, optional
            wheter to use random rotation and scaling, by default False
        """
        super().__init__()
        self.batch_size = batch_size
        self.batches = batches
        self.images = images
        self.masks = masks
        self.augument = augument
        self.elastic = elastic
        self.affine = affine
        if not augument:
            assert batch_size<=len(images), 'Allow data augumentation to create variants'
        if augument:
            assert elastic or affine, 'Allow at least one augumentation mechanism to generate variants'
        self.mask_crop = mask_crop
        self.cropper = keras.layers.Cropping3D(cropping=mask_crop) # Symmetric removal of mask crop pixels before and after x,y,z


    def __len__(self):
        return self.batches

    def __getitem__(self, idx):
        # images and masks are allready loaded into memory
        batch_images = []
        batch_masks = []

        # augument images 
        if self.augument:
            batch_images, batch_masks = generateVariants(self.images, self.masks, self.batch_size, self.elastic, self.affine)
        else:
            batch_images = random.choices(self.images, k= self.batch_size)
            batch_masks = random.choices(self.masks, k = self.batch_size)
        
        # stack tensor lists to batch tensors

        batch_images = np.stack(batch_images)
        batch_masks = np.stack(batch_masks)

        #BUG There are values > 1 in the binary mask ...?
        batch_masks = tf.clip_by_value(batch_masks, 0, 1)

        # crop masks to output region of Unet
        batch_masks = self.cropper(batch_masks).numpy()

        return batch_images, batch_masks





# %% 3D Visualization tools
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

def show3DDisplacementField(dx, dy, dz, mask_points=2000, scale_factor=50.):
    """Visualize a 3D vector field using mayavi.

    Parameters
    ----------
    dx, dy, dz : tensor
        rank three tensors holding the x,y,z components of the 3D vector field.
    mask_points : int, optional
        How many vectors to mask out for each vector displayed, by default 2000
    scale_factor : float, optional
        factor by which arrows are scaled for displaying the vector field, by default 50

    Returns
    -------
    mlab.figure
        Mayavi plot according to the running backend.
    """
    mlab.figure(size=(500,500))
    plot = mlab.pipeline.vector_field(dx, dy, dz)
    mlab.pipeline.vectors(plot, mask_points=mask_points, scale_factor=scale_factor)
    return mlab.gcf()
# %%
