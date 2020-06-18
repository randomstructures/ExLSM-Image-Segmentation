#%%
import tensorflow_datasets as tfds 
import pathlib, os
import random

from IPython.display import Image, display
import PIL
from PIL import ImageOps


from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

import numpy as np

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
    display(Image(filename=input_img_paths[index]))
    # Display auto-constrast version of corresponding target (per-pixel categories)
    img = PIL.ImageOps.autocontrast(load_img(target_img_paths[index]))
    display(img)

#show_image_mask_pair(1)

# %% Feed to keras model by constructing sequence models


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size # Number of pictures per batch
        self.img_size = img_size # tuple, size of the images
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
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        return x, y
