"""This Module provides methods to perform elastic deformations of images and associated segmentation masks
"""

#%% Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# OpenCV for python
import cv2
# Scipy interpolation and image manipulation 
import scipy.interpolate
import scipy.ndimage

#%% Helper functions to visualize Distortions

# Edge grid lines into an image tensor
def edge_grid(im, grid_size):
    """Returns an image tensor with a black grid overlay.
       Usefull to visualize the distortions introduced in image processing



    Parameters
    ----------
    im : image tensor
        the image that is edged
    grid_size : int     
        the spacing of the grid lines in both directions

    Returns
    -------
    image tensor    
        A copy of the inital image tensor with a grid overlay
    """
    # Copy tensor
    tensor = im.copy() # deep copy of the image tensor
    # Draw grid lines
    for i in range(0, tensor.shape[0], grid_size):
        tensor[i,:,:] = 0
    for j in range(0, tensor.shape[1], grid_size):
        tensor[:,j,:] = 0
    
    return tensor

#%% Elastic Deformation I 

def getCoordinateMappingFromDisplacementField(dx, dy):
    """Generate a coordinate mapping from output to input coordinates given a displacement field

    Parameters
    ----------
    dx, dy
       tensors of the same size as the image tensor in x,y dimensions.
       Holds the x and y component of the displacement vector applied at any position in the image

    Returns
    -------
    callable
        A coordinate mapping (x,y,c) -> (x,y,c) from output to input coordinates
    """
    # Define a callable that maps output coordinates to origin coordinates
    def getOriginCords(coords):
        # coords are assumed to be a coordinate tuple (x,y,c)
        x = coords[0] + dx[coords[0], coords[1]]
        y = coords[1] + dy[coords[0], coords[1]]
        return (x,y,coords[2])

    # return the callable
    return getOriginCords

def displacementGridField(image_shape, n_lines = 5, loc = 0, scale = 10):
    """Generate a displacement field that results in an elastic deformation of the input image.
       
       This method implements the approach described in (#TODO cite unet paper)[]
       A coarse grid with the same extent as the image is created.
       For each node in the grid a random displacement vector is sampled from a gaussian distribution
       The displacement at any given point in the output image is interpolated to get a smooth displacement vector field.


    Parameters
    ----------
    image_shape : tuple
        shape of the image tensor
    n_lines : int, optional
        the number of lines in the displacment grid in each direction, by default 5
    loc, scale : float
        center and standard deviation of the normal distribution that is sampled to populate the displacement grid

    dx, dy
       tensors of the same size as the image tensor in x,y dimensions.
       Holds the x and y component of the displacement vector applied at any position in the image
    """    
    ## define grid
    input_shape = image_shape # (x,y,c) shape of input image
    # n_lines = 5 # first and last line coincide with the image border !
    assert n_lines >=4, 'Bicubic interpolation needs at least 4 displacement vectors in each direction.'

    # Set up the coordinates op the displacement grid 
    grid_x, grid_y = np.linspace(0,input_shape[0],n_lines, dtype=np.integer), np.linspace(0,input_shape[1],n_lines, dtype=np.integer)
    #grid_cx, grid_cy = np.meshgrid(grid_x,grid_y) # point n in the mesh has position (grid_cx[n],grid_cy[n])
    mesh_size = (len(grid_x),len(grid_y))

    ## draw displacement vectors on grid
    # draw (dx,dy) ~ N(loc,scale) for every entry in the mesh
    grid_dx = np.random.normal(loc = loc, scale = scale, size = mesh_size) 
    grid_dy = np.random.normal(loc = loc, scale = scale, size = mesh_size) 

    ## calculate pixel wise displacement by bicubic interpolation
    """ 
    RectBivariateSpline(x, y, z)
    Bivariate spline approximation over a rectangular mesh.
    Can be used for both smoothing and interpolating data.

    x,y array_like 1-D arrays of coordinates in strictly ascending order.

    z array_like 2-D array of data with shape (x.size,y.size).
    """
    interpolator_dx = scipy.interpolate.RectBivariateSpline(grid_x, grid_y, grid_dx)
    interpolator_dy = scipy.interpolate.RectBivariateSpline(grid_x, grid_y, grid_dy)

    xx, yy = np.meshgrid(np.arange(input_shape[0]), np.arange(input_shape[1]), indexing='ij')
    dx = interpolator_dx.ev(xx,yy)
    dy = interpolator_dy.ev(xx,yy)

    return dx, dy
    

def smoothedRandomField(image_shape, alpha=300, sigma=8):
    """Generate a displacement field that results in an elastic deformation of the input image.

    Samples an uniform random distribution over the extent of the input image and smooths the values by applying a gaussian filter.
    the resulting displacement field is added to determine the origin coordinates for each position in the output image
    
    Parameters
    ----------
    image_shape : tuple
        shape of the input image
    alpha : float
        amplitude of the displacement field
    sigma : float
        standard deviation of the gaussian kernel

    Returns
    -------
    dx, dy
       tensors of the same size as the image tensor in x,y dimensions.
       Holds the x and y component of the displacement vector applied at any position in the image
    """
    random_state = np.random.RandomState(None)
    # Local distortion 
    # *tuple unpacks the content of the tuple and passes them as arguments to the function -> collected by *args
    # (random_state.rand(*shape) * 2 - 1) gives a tensor specified by shape filled with univariate random numbers shifted to [-1,1)
    # gaussian filter smooths this array where the std in all direction is specified by sigma (large sigma gives smooth displacement arrays)
    dx = scipy.ndimage.gaussian_filter((random_state.rand(*image_shape[:2]) * 2 - 1), sigma) * alpha # apply a gaussian filter (smoothing) on a list of displacement values
    dy = scipy.ndimage.gaussian_filter((random_state.rand(*image_shape[:2]) * 2 - 1), sigma) * alpha
    # sigma => smoothing of displacement vector field
    # alpha => amplitude of displacement vector field

    return dx, dy

#%% Method to transform images given a mapping

def mapImage(image, mapping, interpolation_order=1):
    """Perform a gemetric transformation of the input image as given by the mapping.
    
    Parameters
    ----------
    image : image tensor
        input image
    mapping : callable
        mapping: output (x,y,c) -> input (x,y,c)
    interpolation_order : int 
        Order of the spline polynomial used in interpolation of fractionated input coordinates.
        Set order 0 to use nearest neighbour interpolation which preserves integer class labels

    Returns
    -------
    image tensor
        transformed image
    """
    mapped = scipy.ndimage.geometric_transform(image,
                                               mapping, # Provide a callable that maps input to output cords
                                               mode='reflect', # Reflect image at borders to get values outside image domain
                                               order=interpolation_order) # Interpolate fractionated pixel values using biquadratic interpolation
    return mapped

#%% Methods to efficiently transform a collection of images

def applyDisplacementField(image, dx, dy, interpolation_order = 1):
    """Transform an image by applying a displacement field of the same dimensions.

    Parameters
    ----------
    image : image tensor
        the input image
    dx, dy : matrix
        matrix that holds the x/y component of the displacement vector for each pixel position
    interpolation_order : int, optional
        the order of the spline used to interpolate fractionated pixel values, by default 1
        set 0 to use nearest neighbour interpolation (e.g. in segmentation masks)

    Returns
    -------
    [type]
        [description]
    """
    shape = image.shape
    xx, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij') # coordinates (x,y) of all pixels in two lists (x1,x2,...)(y1,y2,...)
    # var = a,b,c assigns a tuple
    # np.reshape(y+dy, (-1, 1)) recasts the output (coord mesh y + y displacement) y coordinates to a onedimensional array (col vector, all lined up in x direction)
    input_coordinates = np.reshape(xx+dx, (-1, 1)), np.reshape(yy+dy, (-1, 1))
    #print(indices[0].shape)

    # Use the x,y mapping relation to map all channels
    output = np.zeros_like(image)
    # after interpolating all values in single file reshape to input dimensions
    for c in range(shape[2]):
        output[:,:,c] =scipy.ndimage.map_coordinates(image[:,:,c], input_coordinates, order=interpolation_order, mode='reflect').reshape(shape[:2])
    
    return output

