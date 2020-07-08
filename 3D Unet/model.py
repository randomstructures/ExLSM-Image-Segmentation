"""Implementation of a Unet architecture for 3D microscopy data

The original Unet architecure: "U-Net: Convolutional Networks for Biomedical
Image Segmentation" by Ronneberger et al.
3D Implementation as demonstrated in "3D U-Net: Learning Dense Volumetric
Segmentation from Sparse Annotation" by Ozgün et al. 
Implementation details inspired by the model code a the NVIDIA Deep Learning Example [UNet_Medical](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical)

"""

import tensorflow as tf 

""" When tf.keras.Model is subclassed:

Use the constructor method __init__ to instantiate the layers as variables of the model instance
layer parameters are specified but the input shape is passed in the call function


define the method call() providing the input arguments for evaluation of the model
pass them stepwise through the model layers
return the output of the block / model

"""

def _crop_and_concat(inputs, residual_input):
    """Perform a central crop of the skip connection so that it can be oncatenated to the inputs

    Parameters
    ----------
    inputs : tf.Tensor
        tensor with inputs (normally from the lower layer)
    residual_input : tf.Tensor
        tensor holding fine grained information for reconstruction. Was diverted from a corresponding downsampling block

    """

    crop_ratio = inputs.shape[1] / residual_input.shape[1] # The ratio of the residual inputs that is retained
    cropped = tf.image.central_crop(residual_input, crop_ratio)
    concatenated = tf.concat([inputs, cropped], axis=-1) # Join the overlapping 3D regions by concatenating the channel (last) axis
    return concatenated

class InputBlock(tf.keras.Model):
    #TODO a class that implements a keras model (can be used as building block) and bundles the input operations of the Unet 
    def __init__(self, initial_filters):
        """Unet Input Block

        Performs:
        Convolution of input image with #initial_filters
        Convolution doubling the #filters
        Divert output for skip connection
        Downsample by Max Pooling

        Parameters
        ----------
        initial_filters : int
            the number of initial convolution filters. Grows exponentially with model depth.
        """
        super().__init__()
        # Instantiate Block 
        with tf.name_scope('input_block'):
            self.conv1 = tf.keras.layers.Conv3D(filters = initial_filters,
                                                kernel_size=(3,3,3),
                                                activation = tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv3D(filters= initial_filters*2,
                                                kernel_size= (3,3,3),
                                                activation=tf.nn.relu)
            self.maxpool = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), strides= 2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        out = self.maxpool(x)
        return out, x # Provide full res intermediate x for skip connection

class DownsampleBlock(tf.keras.Model):
    #TODO a class that bundels the steps of one downsampling BlockingIOError
    
class UpsampleBlock(tf.keras.Model):
    #TODO a class that bundles the steps of one upsampling block

class BottleneckBlock(tf.keras.Model)
    #TODO a class that bundles the steps in the bottleneck of the Unet

