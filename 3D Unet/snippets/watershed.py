""" Gallery example of scikit-image watershed transform.
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py
10.11.2020
"""
#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

#%%
# Generate an initial image with two overlapping circles
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
r1, r2 = 16, 20
mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
image = np.logical_or(mask_circle1, mask_circle2)

plt.imshow(image)
#%%
# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(image) # Creates a distance transformed image (true is replaced by distance to closest border)
plt.imshow(distance)

#%%

local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)
plt.imshow(local_maxi)
#%%
# Create two integer markers (enummeration starts at 1) from each pixel that is a local maximum
markers = ndi.label(local_maxi)[0]

#%%
# Perform watershed segmentation where lowest values (negative distance =^= centrality) are prioritized
labels = watershed(-distance, markers, mask=image)
# The use of the image (Only object pixels are true) as 'mask' restricts the watershed segmentation to the object areas in the picture

#%%
fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
# %%
# Prepare a second example that reflects our challenge more closely

import skimage.io as skio

sample_mask = skio.imread('D:/Janelia/UnetTraining/20201030_Preprocessing2/z_slice_100.tif')
gt_mask = skio.imread('D:/Janelia/UnetTraining/20201030_Preprocessing2/gt_mask_z_100.tif')

# %%

gt_mask = gt_mask > 0
lower_confidence = sample_mask > 0.2
highest_confidence = sample_mask > 0.98
# %%

def compare(msk1, msk2):
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(msk1,)
    ax[0].set_title('Mask 1')
    ax[1].imshow(msk2,)
    ax[1].set_title('Mask 2')
    ax[2].imshow(msk1 != msk2)
    ax[2].set_title('Logic Difference')

    fig.tight_layout()
    plt.show()

#%% Use highest confidence points as seeds for a single 'segmentation region'
markers = np.zeros_like(sample_mask)
markers[highest_confidence] = 1
# %%
cleaned = watershed(-sample_mask, markers, mask=lower_confidence)
# %%
compare(lower_confidence,highest_confidence)
# %%
compare(cleaned,lower_confidence)
# %%
compare(lower_confidence,gt_mask)
# %%
compare(cleaned,gt_mask)
# %%
compare(sample_mask > 0.5, gt_mask)
# %%


# Try to work natively with the appropriate 3D tensors (we can load them into numpy ndarrays using the h5py library :D)

import numpy as np

segmentation_file = h5py.File('D:/Janelia/UnetTraining/20201030_Preprocessing2/Q1_seg.h5')
print(segmentation_file['t0'].keys())

gt_file = h5py.File('D:/Janelia/UnetTraining/RegionCrops/Q1/Q1.h5')
print(gt_file['t0'].keys())

#%%
segmentation = segmentation_file['t0/train1_epoch50']
segmentation = segmentation[:].astype(np.float)
gt = gt_file['t0/channel2'][:] > 0
# %%

lower_confidence = segmentation > 0.2
highest_confidence = segmentation > 0.98
# %%
markers = np.zeros_like(segmentation)
markers[highest_confidence] = 1
# %%
cleaned = watershed(-segmentation, markers, mask=lower_confidence)

# %%
del segmentation_file['t0/t1_e50_cleaned']
segmentation_file['t0'].create_dataset(name= 't1_e50_cleaned' ,data= cleaned, dtype=np.uint8)
# %%
segmentation_file.close()
gt_file.close()
# %%

import metrics

res = metrics.precisionRecall(gt, cleaned > 0)
# %%
iou = metrics.intersection_over_union(gt, cleaned > 0)
# %%
