#%%
import tensorflow_datasets as tfds 
import pathlib, os
import random

import IPython.display as dsp
import PIL
from PIL import ImageOps


import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
from matplotlib import pyplot as plt

#%%
#dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

# %% Code provided by https://keras.io/examples/vision/oxford_pets_image_segmentation/

def get_path_lists(base_dir):
    # Locate the dataset files
    # X:\lillvis\temp\linus\OxfordPetDataset
    # base_dir = 'X:/lillvis/temp/linus/OxfordPetDataset/'
    input_dir = "images/"
    target_dir = "annotations/trimaps/"
    # Prepend comman base directory
    input_dir = os.path.join(base_dir, input_dir)
    target_dir = os.path.join(base_dir, target_dir)

    # The following is a multiline python generator expression !
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    print("Number of samples:", len(input_img_paths))

    return input_img_paths, target_img_paths

#input_img_paths, target_img_paths = get_path_lists()


#for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
#    print(input_path, "|", target_path)

#NOTE this method sorts all filenames in both directories by lexicographic order. This results in the grouping of images showing the same animal and some artifacts as 1 < 10 < 100 <...<2
# At this point the corresponding elements are at the same position in both lists

#%%
def shuffle_path_lists(input_img_paths, target_img_paths):
    order = list(range(len(input_img_paths))) # List of all indexes
    random.shuffle(order) # Assign new location to each element denoted by its index
    input_img_paths_shuffled = [input_img_paths[i] for i in order]
    target_img_paths_shuffled = [target_img_paths[i] for i in order]
    return input_img_paths_shuffled, target_img_paths_shuffled

"""
input_img_paths_shuffled, target_img_paths_shuffled = shuffle_path_lists(input_img_paths, target_img_paths)
# Validate that correct pairs are still together
for input_path, target_path in zip(input_img_paths_shuffled[:10], target_img_paths_shuffled[:10]):
    print(input_path, "|", target_path)
"""

# %% Visualize image and segmentation mask
def show_image_mask_pair(index, input_img_paths, target_img_paths):
    #dsp.display(dsp.Image(filename=input_img_paths[index]))
    # Display auto-constrast version of corresponding target (per-pixel categories)
    #img = PIL.ImageOps.autocontrast(load_img(target_img_paths[index]))
    #dsp.display(img)
    image = load_img(input_img_paths[index])
    image = keras.preprocessing.image.img_to_array(image)
    mask = load_img(target_img_paths[index])
    mask = keras.preprocessing.image.img_to_array(mask)

    display([image, mask])

#show_image_mask_pair(1)

# %% Feed to keras model by constructing sequence models


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, mask_crop, input_img_paths, target_img_paths):
        self.batch_size = batch_size # Number of pictures per batch
        self.img_size = img_size # tuple, size of the images
        self.mask_crop = mask_crop # int, number of pixels to crop from each border
        self.input_img_paths = input_img_paths # exhaustive list of all image locations
        self.target_img_paths = target_img_paths # list of all mask locations in the same order

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        # Precompute the size of the cropped mask tensor as a tuple
        mask_size = (self.img_size[0]-2*self.mask_crop, self.img_size[1]-2*self.mask_crop)
        # Allocate a numpy tensor
        y = np.zeros((self.batch_size,) + mask_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img = keras.preprocessing.image.img_to_array(img)
            # Clip mask_crop pixels from each side
            img = tf.image.crop_to_bounding_box(img,
                                                offset_height=self.mask_crop,
                                                offset_width=self.mask_crop,
                                                target_height=mask_size[0] ,
                                                target_width=mask_size[1])
            # Shift class labels from [1,2,3] to [0,1,2] for keras compatibility
            img = img - 1 # subtract one from each element
            y[j] = img
        return x, y

def create_mask(pred_mask):
    """Reduce the unet output (logit channel per class) to a mask (class number of maximum probability class)


    Parameters
    ----------
    pred_mask : tf.Tensor
        Image tensor with one channel per class

    Returns
    -------
    tf.Tensor
        Image tensor with one channel. Highest probability class number per pixel
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    #pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def show_predictions(model, images, masks, num=1):
    """Plot predictions generated by a FCCN

    Parameters
    ----------
    model : keras model
        FCCN used to perform semantic segmentation
    dataset : tf.Dataset
        Dataset of images and segmentation masks
    num : int, optional
        the number of images to evaluate the model on, by default 1
    """
    for image, mask in zip(images[:num], masks[:num]):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])

def display(display_list):
    """Display an image, the true segmentation mask and the model output

    Parameters
    ----------
    display_list : list of image tensors
        [inputImage, trueMask, predictedMask]
    """
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('on')
    plt.show()

def check_image_size(image_size, n_blocks):
    """Checks if a valid unet architecture with n blocks can be constructed from an image input 

    Parameters
    ----------
    image_size : int    
        dimension of input image
    n_blocks : int
        the number of blocks in the downsampling and upsampling path

    Returns
    -------
    boolean, int
        validity, size of output image (0 if false)
    """
    x = image_size
    outputs = []
    # downsampling
    for n in range(n_blocks):
        x -= 4 # two conv layers 3x3
        outputs.append(x) # store output dimension
        if not x%2==0: # check if 2x2 max pooling tiles nicely
            #print('Down {} max pool input {} not divisible by 2'.format(n+1,x))
            return False, 0
        x /= 2
    # bridge
    x -= 4
    # upsampling
    for n in range(n_blocks):
        x *= 2
        skip = outputs.pop()
        if not (skip-x)%2==0:
            print('Up {} crop from {} to {} not centered'.format(n,skip,x))
            return False, 0
        x -= 4 # two conv layers 3x3
    #print('image size valid')
    if x>0:
        return True, x
    else:
        return False, 0