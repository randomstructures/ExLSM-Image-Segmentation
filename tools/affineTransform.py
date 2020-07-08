# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Elastic deformation of 2D images
# 
# [Code by gangadhar provided on Kaggle](https://www.kaggle.com/gangadhar/nuclei-segmentation-in-microscope-cell-images)
# 
# This notebook presents an image augmentation method that uses both local distortion and random affine transformation.
# 
# These transformations uses anti-aliasing for high-quality output.

# %%
# Import stuff
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

import utility
from tensorflow import keras as k


# %%
# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    # Step 1 define three points (list of three (x,y) coords)
    # Step 2 specify where they land in the transformed image
    # Step 3 generate a matrix that represents the corresponding affine transformation
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    
    image = cv2.warpAffine(image, M, (shape_size[1],shape_size[0]) , borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


# %%
# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

# Load images
base_dir = 'C:/Users/Linus Meienberg/Documents/ML Datasets/Oxford_Pets'

images, masks = utility.get_path_lists(base_dir) # get corresponding lists af all paths

# Use matplotlib instead of cv2 i/o functions
#im = cv2.imread(images[0],1) # second argument = flag that specifies image format 1 => color image
#im_mask = cv2.imread(masks[0])
im = k.preprocessing.image.img_to_array(k.preprocessing.image.load_img(images[0]))
im_mask = k.preprocessing.image.img_to_array(k.preprocessing.image.load_img(masks[0]))
#cv2.imshow('the image',im)
#cv2.imshow('the mask',im_mask)
utility.display([im,im_mask])


# %%
# Draw grid lines
draw_grid(im, 50)
draw_grid(im_mask, 50)

# Merge images into separete channels (shape will be (cols, rols, 2))
im_merge = np.concatenate((im[...,None], im_mask[...,None]), axis=2)


# %%
# First sample...

get_ipython().run_line_magic('matplotlib', 'inline')

# Apply transformation on image
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

# Split image and mask
#im_t = im_merge_t[...,0]
#im_mask_t = im_merge_t[...,1]

# Display result
plt.figure(figsize = (16,14))
#plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')
utility.display([im_merge_t])


# %%
# Second sample (heavyer transform)...

get_ipython().run_line_magic('matplotlib', 'inline')

# Apply transformation on image
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.09)

# Split image and mask
im_t = im_merge_t[...,0]
im_mask_t = im_merge_t[...,1]

# Display result
plt.figure(figsize = (16,14))
plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')



# %%
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
from tensorflow import keras as k
  
  
img = k.preprocessing.image.img_to_array(k.preprocessing.image.load_img(images[0])).astype(np.int32)

rows, cols, ch = img.shape 
  
pts1 = np.float32([[50, 50], 
                   [200, 50],  
                   [50, 200]]) 
  
pts2 = np.float32([[10, 100], 
                   [200, 50],  
                   [100, 250]]) 
  
M = cv2.getAffineTransform(pts1, pts2) 
dst = cv2.warpAffine(img, M, (cols, rows)) 
  
plt.subplot(121) 
plt.imshow(img) 
plt.title('Input') 
  
plt.subplot(122) 
plt.imshow(dst) 
plt.title('Output') 
  
plt.show() 
  
# Displaying the image 
while(1): 
      
    cv2.imshow('image', img) 
    if cv2.waitKey(20) & 0xFF == 27: 
        break
          
cv2.destroyAllWindows() 

# %%
#NOTE check difference between image matrices of keras load image and cv imread