"""Implementation of a Unet architecture for 3D microscopy data

The original Unet architecure: "U-Net: Convolutional Networks for Biomedical
Image Segmentation" by Ronneberger et al.

3D Implementation as demonstrated in "3D U-Net: Learning Dense Volumetric
Segmentation from Sparse Annotation" by Ozgün et al. (Lift operations to 3D, different scheme for expansion and contraction of the number of feature channels)

Implementation details inspired by the model code a the NVIDIA Deep Learning Example [UNet_Medical](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical)

"""

#NOTE If layers receive multiple input tensors when called, pass them as a list in the inputs argument

#%%
import tensorflow as tf 
import tensorflow.keras.backend as K
import numpy as np

#%% CONSTRUCTION OF UNET by subclassing model

class Unet(tf.keras.Model):

    def __init__(self, name= 'Unet', n_blocks= 2, initial_filters= 32, **kwargs):
        super(Unet, self).__init__(name=name, **kwargs)

        # instantiate unet blocks
        filters = initial_filters
        self.input_block = InputBlock(initial_filters= filters) 
        filters *= 2 # filters are doubled in second conv operation
        

        self.down_blocks = []
        for index in range(n_blocks):
            self.down_blocks.append(DownsampleBlock(filters, index+1))
            filters *= 2  # filters are doubled in second conv operation


        self.bottleneck_block = BottleneckBlock(filters)
        filters *= 2  # filters are doubled in second conv operation

        self.up_blocks = []
        for index in range(n_blocks)[::-1]:
            filters = filters//2  # half the number of filters in first convolution operation
            self.up_blocks.append(UpsampleBlock(filters, index+1))

        filters = filters//2 # half the number of filters in first convolution operation
        self.output_block = OutputBlock(filters, n_classes=2)

    def call(self, inputs, training=True):
        skip = []

        out, residual = self.input_block(inputs)
        skip.append(residual)

        for down_block in self.down_blocks:
            out, residual = down_block(out)
            skip.append(residual)

        out = self.bottleneck_block(out, training)

        for up_block in self.up_blocks:
            out = up_block([out, skip.pop()])

        out = self.output_block([out, skip.pop()])

        return out

#%% construct model via sequential API

def build_unet(input_shape, n_blocks= 2, initial_filters= 32, **kwargs):
    # Create a placeholder for the data that will be fed to the model
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    skips = []

    # instantiate unet blocks
    filters = initial_filters

    # Thread through input block
    x, residual = InputBlock(initial_filters=filters)(x)
    skips.append(residual)
    filters *= 2 # filters are doubled in second conv operation
    
    for index in range(n_blocks):
        x, residual = DownsampleBlock(filters=filters, index=index+1)(x)
        skips.append(residual)
        filters *= 2  # filters are doubled in second conv operation

    x = BottleneckBlock(filters)(x)
    filters *= 2  # filters are doubled in second conv operation

    for index in range(n_blocks)[::-1]:
        filters = filters//2  # half the number of filters in first convolution operation
        x = UpsampleBlock(filters, index+1)([x, skips.pop()])

    filters = filters//2 # half the number of filters in first convolution operation
    x = OutputBlock(filters, n_classes=2)([x,skips.pop()])
    
    unet = tf.keras.Model(inputs=inputs, outputs=x)
    return unet


#%%IMPLEMENTATION OF UNET BLOCKS

""" When tf.keras.Model is subclassed:

Use the constructor method __init__ to instantiate the layers as variables of the model instance
layer parameters are specified but the input shape is passed in the call function


define the method call() providing the input arguments for evaluation of the model
pass them stepwise through the model layers
return the output of the block / model

"""

def _crop_concat(input, residual_input):
    """Concatenate two 3D images after cropping residual input to the size of input.
    The last (channel) dimension of the tensors is joined. The difference of the input sizes must be even to allow for a central crop.

    Parameters
    ----------
    input : tf.Tensor   
        3d image tensor in the format (batch, x, y, z, channels)
    residual_input : tf.Tensor
        3d image tensor in the format (batch, x, y, z, channels)

    Returns
    -------
    tf.Tensor
        Cropped and conatenated 3d image tensor of the same shape as input
    """
    crop = [(residual_input.shape[d]-input.shape[d])//2 for d in range(1,4)]
    #print('crop = {}'.format(crop))
    x = tf.keras.layers.Cropping3D(cropping=crop)(residual_input)
    x = tf.keras.layers.Concatenate(axis=-1)([input, x])
    return x 


class InputBlock(tf.keras.layers.Layer):
    #TODO a class that implements a keras model (can be used as building block) and bundles the input operations of the Unet 
    def __init__(self, initial_filters=8, **kwargs):
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
        super(InputBlock, self).__init__(**kwargs)
        self.initial_filters = initial_filters
        # Instantiate Block 
        with tf.name_scope('input_block'):
            self.conv1 = tf.keras.layers.Conv3D(filters = initial_filters,
                                                kernel_size=(3,3,3),
                                                activation = tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv3D(filters= initial_filters*2,
                                                kernel_size= (3,3,3),
                                                activation=tf.nn.relu)
            self.maxpool = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), strides= (2,2,2))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        out = self.maxpool(x)
        return out, x # Provide full res intermediate x for skip connection

    def get_config(self):
        config = super(InputBlock, self).get_config()
        config.update({"initial_filters" : self.initial_filters})
        return config

class DownsampleBlock(tf.keras.layers.Layer):
    """Unet Downsample Block

    Perform two convolutions with a specified number of filters.
    Double the amount of filters in the second convolution.
    Divert output for skip connection.
    Downsample by max pooling for lower level input.
    """
    def __init__(self, filters, index, **kwargs):
        """Unet Downsample Block

        Parameters
        ----------
        filters : int
            Number of filters in the first convolution
        index : int
            index / depth of the block
        """
        super(DownsampleBlock,self).__init__(**kwargs)
        with tf.name_scope('downsample_block_{}'.format(index)):
            self.index = index
            self.filters = filters
            self.conv1 = tf.keras.layers.Conv3D(filters=filters,
                                        kernel_size = (3,3,3),
                                        activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv3D(filters=filters*2,
                                        kernel_size = (3,3,3),
                                        activation=tf.nn.relu)
            self.maxpool = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), strides = (2,2,2))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        out = self.maxpool(x)
        return out, x

    def get_config(self):
        config = super(DownsampleBlock, self).get_config()
        config.update({"filters" : self.filters, "index" : self.index})
        return config

class BottleneckBlock(tf.keras.layers.Layer):
    """Central / Bottleneck Block of Unet Architecture
    
    Perform two unpadded convolutions before upsampling to begin the reconstructing pathway
    Include a Dropout layer for training the network
    """
    def __init__(self, filters, **kwargs):
        """Unet Bottleneck Block

        Parameters
        ----------
        filters : int
            number of filters in the first convolution operation.
        """
        super(BottleneckBlock,self).__init__(**kwargs)
        with tf.name_scope('bottleneck_block'):
            self.filters = filters
            self.conv1 = tf.keras.layers.Conv3D(filters=filters,
                                        kernel_size = (3,3,3),
                                        activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv3D(filters=filters*2,
                                        kernel_size = (3,3,3),
                                        activation=tf.nn.relu)
            self.dropout = tf.keras.layers.Dropout(rate=0.2)
            self.upsample = tf.keras.layers.Conv3DTranspose(filters=filters*2,
                                                            kernel_size = (2,2,2),
                                                            strides= (2,2,2))
    
    def call(self, inputs, training):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.dropout(x, training=training) # Don't use dropout for predictions outside training
        x = self.upsample(x)
        return x

    def get_config(self):
        config = super(BottleneckBlock, self).get_config()
        config.update({"filters" : self.filters})
        return config
    
class UpsampleBlock(tf.keras.layers.Layer):
    """Unet Upsample Block

    Crop and concatenate skip input of corresponding depth to input from layer below.
    Perform two convolutions with a specified number of filters and upsample.
    The first convolution operation reduces the number of feature channels accoring to the current depth in the network

    """

    def __init__(self, filters, index, **kwargs):
        """Upsample Block if Unet Architecture

        Parameters
        ----------
        filters : int 
            Number of feature channels in the convolution operation.
        index : int
            index / depth of the block
        """
        super(UpsampleBlock, self).__init__(**kwargs)
        with tf.name_scope('upsample_block_{}'.format(index)):
            self.index = index
            self.filters = filters
            self.conv1 = tf.keras.layers.Conv3D(filters=filters,
                                        kernel_size = (3,3,3),
                                        activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv3D(filters=filters,
                                        kernel_size = (3,3,3),
                                        activation=tf.nn.relu)
            self.upsample = tf.keras.layers.Conv3DTranspose(filters=filters,
                                                            kernel_size=(2,2,2),
                                                            strides=(2,2,2))

    def call(self, inputs):
        x = _crop_concat(input=inputs[0], residual_input=inputs[1])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

    def get_config(self):
        config = super(UpsampleBlock, self).get_config()
        config.update({"filters" : self.filters, "index" : self.index})
        return config

class OutputBlock(tf.keras.layers.Layer):
    """Unet Ouput Block

    Perform three unpadded convolutions.
    The last convolution operation reduces the output volume to the desired number of output channels for classification.
    The model returns raw logits.

    """
    def __init__(self, filters,  n_classes, **kwargs):
        super(OutputBlock, self).__init__(**kwargs)
        with tf.name_scope('output_block'):
            self.filters = filters
            self.n_classes = n_classes
            self.conv1 = tf.keras.layers.Conv3D(filters=filters,
                                        kernel_size = (3,3,3),
                                        activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv3D(filters=filters,
                                        kernel_size = (3,3,3),
                                        activation=tf.nn.relu)
            self.conv3 = tf.keras.layers.Conv3D(filters=n_classes,
                                                kernel_size=(1,1,1),
                                                activation = None)


    def call(self, inputs):
        x = _crop_concat(input=inputs[0], residual_input=inputs[1])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def get_config(self):
        config = super(OutputBlock, self).get_config()
        config.update({"filters" : self.filters, "n_classes" : self.n_classes})
        return config

# %%
def weighted_sparse_categorical_crossentropy(class_weights):
    # As seen on GitHub https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d by wassname
    weights = tf.keras.backend.variable(class_weights)
    num_classes = len(class_weights)

    def loss(y_true, y_pred):
        # Keras defines the following shapes for the inputs to the loss function:
        # y_true => (batch_size, d0, ..., dN-1) for sparse loss functions
        # y_pred => (batch_size, d0, ..., dN-1, c) for predicted values
        # where c denotes the number of classes
        
        # in our case there is a channel dimension of 1 in the mask tensors
        # (b, x, y, z, 1) which can be easily eliminated
        # y true is an integer in [0,n_classes)
        y_true = K.cast(y_true[...,0], dtype='int32')
        # expand to one hot encoded tensor (b, x, y, z, c)
        y_true_expanded = K.one_hot(y_true, num_classes=num_classes)

        # our predictions are tensors of shape (b,x,y,z,c)
        # the predicted values are raw logits. Apply softmax normalization along the channel axis to convert each output to a class probability
        y_pred_softmax = K.softmax(y_pred, axis=-1)
        # clip to prevent NaN's and Inf's
        y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1 - K.epsilon()) 
        
        # crossentropy is the negative sum of y_true*log(y_pred)
        # tensorflow uses operator overriding tf.tensor * tf.tensor invokes tf.math.multiply
        # tf.math.multiply uses array broadcasting as explained here (https://numpy.org/doc/stable/user/basics.broadcasting.html) if tensors have differing shapes
        l = y_true_expanded * K.log(y_pred_softmax) # element wise multiplication that preserves tensor shape (b,x,y,z,c)
        l_weighted = l * weights # (b,x,y,z,c) * (c,)  uses broadcasting of (c,) to the shape of (b,x,y,z,c,) so that effectively each channel is multiplied by the appropriate weight
                
        # we now have computed a tensor of weighted, pixel wise contributions to the loss
        # sum up the tensor along all spatial coordinates to get an array of loss per sample in the batch
        sum = K.sum(l_weighted, axis=-1) # sum over channels (b,x,y,z,)
        sum = K.sum(sum, axis=-1) # sum z (b,x,y,)
        sum = K.sum(sum, axis=-1) # sum y (b,x,)
        sum = K.sum(sum, axis=-1) # sum x => array of length (b,)
        # divide the loss by the number of pixels in each sample for better comparability
        sum /= tf.reduce_sum(tf.ones_like(y_pred[0,...]))
        return -sum # return the negative sum
    
    return loss


# %%

def weighted_sparse_categorical_crossentropy_v2(class_weights):
    # As seen on GitHub https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d by wassname
    weights = tf.keras.backend.variable(class_weights)
    num_classes = len(class_weights)

    def loss(y_true, y_pred):
        # Keras defines the following shapes for the inputs to the loss function:
        # y_true => (batch_size, d0, ..., dN-1) for sparse loss functions
        # y_pred => (batch_size, d0, ..., dN-1, c) for predicted values
        # where c denotes the number of classes
        
        # in our case there is a channel dimension of 1 in the mask tensors
        # (b, x, y, z, 1) which can be easily eliminated
        # y true is an integer in [0,n_classes)
        y_true = K.cast(y_true[...,0], dtype='int32')
        labels_one_hot = K.one_hot(y_true, num_classes=num_classes)
        
        # deduce weights for batch samples based on their true label (tensor of the same shape as onehot_labels with the corresponding class weight as value)
        voxel_weights = tf.reduce_sum(weights * labels_one_hot, axis=-1)

        # compute a tensor with the unweighted cross entropy loss
        unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot,logits=y_pred) #(x,y,z,c)
        weighted_loss = unweighted_loss * voxel_weights # (x,y,z,c) * (c,) broadcasts the second array such that each channel is multiplied by it's weight

        return tf.reduce_mean(weighted_loss, axis=[1,2,3])
    
    return loss


def soft_dice_loss(y_true, y_pred, num_classes):
    """
    Soft dice loss is derived from the dice score. 
    It measures the overlap between the predicted and true mask regions for each channel.
    Dice loss penalizes low confidence predictions in the ground truth region and high confidence predictions outside of it.

    Wrapper for Jeremy Jordany implementation.

    Parameters
    ----------
    y_true : tensor with shape (...,1)
        ground truth integer segmentation mask
    y_pred : tensor with shape (...,c)
        raw logit predictions of the model

    Returns
    -------
    callable   
        the averaged soft dice loss
    """
    def loss(y_true,y_pred):
        # convert the segmentation mask to one hot encoding
        ohe_true = K.one_hot(y_true, num_classes)
        # apply softmax to logits
        softmax_pred = K.softmax(y_pred, axis=-1)
        return soft_dice(ohe_true,softmax_pred)

# %%
def soft_dice(y_true, y_pred, epsilon=1e-6):
    ''' 
    This code adapted from [Jeremy Jordan](https://www.jeremyjordan.me/semantic-segmentation/) 

    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch