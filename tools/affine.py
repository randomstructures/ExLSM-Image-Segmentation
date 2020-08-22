""" Affine Transformations of 3D image tensors

This module implements tools for rotation and scaling of 3D imaging data.

Linus Meienberg
June 2020
"""
#%% Imports 

import math
import numpy as np 
from scipy import ndimage


# %%


def s(x):
    return math.sin(x)
def c(x):
    return math.cos(x)

# r_z : rotation around z axis by angle x in [0,2Pi]
def R_z(x):
    return np.array(
            [[c(x), -s(x), 0],
            [s(x), c(x),  0],
            [   0,    0,   1]])
# r_x : rotation around x axis by angle x in [0, Pi]
def R_x(x):
    return np.array(
            [[   1,    0,  0],
            [   0, c(x),-s(x)],
            [   0, s(x), c(x)]])


def constructRotationMatrix(alpha, beta, gamma):
    """Construct a three dimensional rotation matrix described by three euler angles in active xzx convention

    Parameters
    ----------
    alpha, beta, gamma : float
        euler angles specifiying 3D rotation
    
    Returns
    -------
    numpy array
        Matrix with shape (3,3) describing the spatial rotation.
    """
    # Construct a general rotation matrix by composition of three elementary rotations
    # see Wikipedia (https://de.wikipedia.org/wiki/Eulersche_Winkel) Abbildungsmatrix aktive Drehung

    assert (alpha>=0) & (beta>=0) & (gamma>=0), 'specify nonnegative euler angles' 

    rzgamma = R_z(gamma)
    rxbeta = R_x(beta)
    rzalpha = R_z(alpha)
    r = np.matmul(rzalpha, np.matmul(rxbeta, rzgamma))
    return r

# %%

def getRandomRotation():
    """Constructs the transformation matrix of a random spatial rotation.

    Returns
    -------
    numpy array
        Matrix with shape (3,3) describing the spatial rotation. 
    """
    alpha = np.random.uniform() * 2 * math.pi
    beta = np.random.uniform() * 1 * math.pi
    gamma = np.random.uniform() * 2 * math.pi
    #print((alpha,beta,gamma))
    return constructRotationMatrix(alpha, beta, gamma)

# %%
def constructScalingMatrix(scaling_factor):
    """Constructs the transformation matrix of a uniform scaling operation.

    Parameters
    ----------
    scaling_factor : float
        the scaling factor

    Returns
    -------
    numpy array
        Matrix with shape (3,3) describing the uniform scaling.
    """
    return np.eye(3)*scaling_factor

def getRandomScaling(lb=0.9, ub=1.1):
    scaling = np.random.uniform(lb,ub)
    return constructScalingMatrix(scaling)

# %%
def getRandomAffine():
    """Construct a random affine transformation by composition of a random rotation and random scaling

    Returns
    -------
    numpy array 
        Matrix with shape (3,3) describing the affine transformation.
    """
    rotation = getRandomRotation()
    scaling = getRandomScaling()
    return np.matmul(rotation,scaling)

def applyAffineTransformation(image, transformation_matrix, interpolation_order = 1):
    """Apply an affine transformation to a multichannel 3D image tensor.
    The coordinate system of the image tensor is shifted to it's center before the transformation matrix is applied.
    Output coordinates that are mapped outside the input image are filled by reflecting the input image.

    Parameters
    ----------
    image : tensor
        3D image tensor with shape (x,y,z,c) where c are the channels
    transformation_matrix : matrix
        Matrix with shape (3,3) describing a three dimensional affine transformation.
    interpolation_order : int
        Fractionated pixel coordinates are interpolated by splines of this order. 
        If order 0 is specified, nearest neighbour interpolation is used. Use this setting when transforming masks.

    Returns
    -------
    tensor
        3D image tensor of the same shape as the input image.
    """
    # get inverse transformation. 
    inverse = np.linalg.inv(transformation_matrix)

    # shift the center of the coordinate system to the middle of the volume
    center = [dim//2 for dim in image.shape[:-1]] # calculate the image center exclude last dim (channel)
    center_mapping = np.dot(inverse, center) # Calculate where the center of the input region is mapped to 
    center_shift = center-center_mapping # Calculate the shift of the center point => Add this to make the center points of the input and output region congruent
    # apply affine transform to each channel of the image
    out = np.zeros_like(image)

    for channel in range(image.shape[-1]):
         out[...,channel] = ndimage.affine_transform(image[...,channel],
         inverse, offset=center_shift,
         mode='reflect',
         order=interpolation_order)
    
    return out



# %%
"""
import tensorflow as tf

im = tf.keras.preprocessing.image.load_img('cato.png')
im = tf.keras.preprocessing.image.img_to_array(im)

def _rot2D(a):
    return [[math.cos(a), -math.sin(a)],
            [math.sin(a), math.cos(a)]]

rot = _rot2D(1)
im_rot = np.zeros_like(im)

center = [dim//2 for dim in im.shape[:-1]]
center_mapping = np.dot(rot, center)
center_shift = center-center_mapping

for c in range(im.shape[-1]):
    im_rot[...,c] = ndimage.affine_transform(im[...,c], rot, offset=center_shift)
"""
# %%
