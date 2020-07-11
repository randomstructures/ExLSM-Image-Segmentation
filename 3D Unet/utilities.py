"""Collection of utilities to run 3D Unet
"""

#%% 

import os, pathlib
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


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

def _getImage(path):
    return keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path, color_mode='grayscale'))

#C:\Users\Linus Meienberg\Documents\ML Datasets\FruSingleNeuron_20190707\SampleCrops
#'C:/Users/Linus Meienberg/Documents/ML Datasets/FruSingleNeuron_20190707/SampleCrops'
base_dir = 'C:\\Users\\Linus Meienberg\\Documents\\ML Datasets\\FruSingleNeuron_20190707\\SampleCrops'
samples = os.listdir(base_dir)

def get_sample_image():
    path = os.path.join(base_dir,samples[0])

    # import image
    input_image_sequence = os.path.join(path,'image') # Navigate to image subfolder
    filenames = os.listdir(input_image_sequence)
    images = [_getImage(os.path.join(input_image_sequence, filename)) for filename in filenames]
    input_image = np.stack(images) # assembled image tensor
    return input_image

def get_sample_mask():
    path = os.path.join(base_dir,samples[0])

    # import image
    input_image_sequence = os.path.join(path,'mask') # Navigate mo mask subfolder
    filenames = os.listdir(input_image_sequence)
    images = [_getImage(os.path.join(input_image_sequence, filename)) for filename in filenames]
    input_image = np.stack(images) # assembled image tensor
    return input_image
#%%

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

# %%
