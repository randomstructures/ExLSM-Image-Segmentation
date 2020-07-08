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
    Use the functional API to configure them and provide inputs as arguments at construction time
    return the model output

    define the method call() providing the input arguments for evaluation of the model
    pass them stepwise through the model layers
    return the output of the block / model

    """

def _crop_and_concat(inputs, residual_input):
    #TODO a method that performs a central crop of the residual input and concatenates it to the inputs 

class InputBlock(tf.keras.Model):
    #TODO a class that implements a keras model (can be used as building block) and bundles the input operations of the Unet 

class DownsampleBlock(tf.keras.Model):
    #TODO a class that bundels the steps of one downsampling BlockingIOError
    
class UpsampleBlock(tf.keras.Model):
    #TODO a class that bundles the steps of one upsampling block

class BottleneckBlock(tf.keras.Model)
    #TODO a class that bundles the steps in the bottleneck of the Unet

