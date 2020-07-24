"""Collection of utilities to run 3D Unet
"""

#%% 

import os, pathlib
import random

import tensorflow as tf
from tensorflow import keras

import numpy as np
import scipy

import matplotlib.pyplot as plt

from mayavi import mlab

#%% Import modules providing tools for image manipulation
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
            batch_images, batch_masks = generateVariants(self.images, self.masks, self.batch_size,
                                                         self.elastic, self.affine)
            
            # shuffle the images and mask pairs in the same order
            new_order = np.arange(self.batch_size) 
            random.shuffle(new_order) # a shuffled list of old indices is created IN PLACE
            batch_images = [batch_images[i] for i in new_order]
            batch_masks = [batch_masks[i] for i in new_order]
        else:
            indices = np.arange(len(self.images))
            random.shuffle(indices)
            indices = indices[:self.batch_size] # take the first batch_size indices (batch_size<=len(images) is guaranteed)
            batch_images = [self.images[i] for i in indices]
            batch_masks = [self.masks[i] for i in indices]
        
        # stack tensor lists to batch tensors
        batch_images = np.stack(batch_images)
        batch_masks = np.stack(batch_masks)

        #NOTE There are values > 1 in the binary mask which is an artifact of the mask creation process
        # clip the mask at 1 to binarize it again
        batch_masks = tf.clip_by_value(batch_masks, 0, 1)

        # crop masks to output region of Unet
        batch_masks = self.cropper(batch_masks).numpy()
        # cast to integer values
        batch_masks = batch_masks.astype(int)
        
        return batch_images, batch_masks





# %% 3D Visualization tools
def show3DImage(image_tensor, channel=0, mode = 'image', newFigure = True, **kwargs):
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
    if newFigure:
        mlab.figure(size=(500,500))
    plot = mlab.contour3d(image_tensor[...,channel], contours = n_contours, transparent=transparent, **kwargs)
    return mlab.gcf()

def show3DDisplacementField(dx, dy, dz, mask_points=2000, scale_factor=50., newFigure=True, **kwargs):
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
    if newFigure:
        mlab.figure(size=(500,500))
    plot = mlab.pipeline.vector_field(dx, dy, dz)
    mlab.pipeline.vectors(plot, mask_points=mask_points, scale_factor=scale_factor, **kwargs)
    return mlab.gcf()
# %%

def showCutplanes(image_tensor, channel=0, hint = True, newFigure = True, **kwargs):
    if newFigure:
        mlab.figure(size=(700,700))
    s = image_tensor[...,channel]
    src = mlab.pipeline.scalar_field(s)
    mid_x = s.shape[0] // 2
    mid_y = s.shape[1] // 2
    # numpy : np.ptp() = max()-min()
    if hint:
        mlab.pipeline.iso_surface(src, contours = [np.min(s)+0.8*s.ptp(), ], opacity = 0.3, **kwargs)
    mlab.pipeline.image_plane_widget(src,
                                    plane_orientation = 'x_axes',
                                    slice_index = mid_x, **kwargs)
    mlab.pipeline.image_plane_widget(src,
                                    plane_orientation = 'y_axes',
                                    slice_index = mid_y, **kwargs)
    mlab.outline()
    return mlab.gcf()


def showLogitDistribution(prediction):
    
    plt.figure()
    
    data_range = np.linspace(np.min(prediction), np.max(prediction), 10)

    for c in range(prediction.shape[-1]):
        hist, bins = np.histogram(prediction[...,c], bins=data_range)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width, label='channel ' + str(c))

    plt.title('Distribution of logits per channel in the prediction')
    plt.xlabel('Predicted logit class probabilities')
    plt.ylabel('count')
    plt.legend()
    plt.show()

def showZSlices(volume, channel=0, n_slices = 4, title=None, mode='gray', plot_size=4):
    # volume is expected to be in format (x,y,z,c)
    z_extent = volume.shape[2]
    if mode is 'h5':
        z_extent = volume.shape[0]
    slice_z = np.linspace(0,z_extent,n_slices+2).astype(int)[1:-1] # n_slices+2 evently spaced planes, leave first and last one out

    fig, axs = plt.subplots(1, n_slices, figsize=(plot_size*n_slices+2,plot_size+0.25))
    fig.suptitle(title, fontsize=15)

    for i, ax in enumerate(axs):
        z = slice_z[i]
        ax.set_title('slice @ z='+str(z))
        if mode is 'rgb':
            ax.imshow(volume[:,:,z,:])
        elif mode is 'gray':
            ax.imshow(volume[:,:,z,channel], cmap='Greys')
        elif mode is 'h5':
            ax.imshow(volume[z,:,:], cmap='Greys')
        else:
            raise ValueError('Mode not implemented')

    plt.show()

def testFigure():
    plt.figure()
    plt.plot([1,2,1])
    plt.legend('hello world')
    plt.show()

def makeRGBComposite(r,g,b,gain=(1,1,1)):
    if type(gain) is tuple:
        assert len(gain) is 3, 'specify gain for 3 channels (g_r,g_g,g_b)'
    else:
        gain = (gain,gain,gain)

    assert not r is None, 'specify at one input channel in r'
    if g is None:
        g = np.zeros_like(r)
    if b is None:
        b = np.zeros_like(g)

    composite = r*[1,0,0] + g*[0,1,0] + b*[0,0,1]
    composite *= gain
    return composite
    