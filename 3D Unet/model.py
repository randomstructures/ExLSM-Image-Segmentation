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
    crop = [(residual_input.shape[d]-input.shape[d])//2 for d in range(1,4)]
    print('crop = {}'.format(crop))
    x = tf.keras.layers.Cropping3D(cropping=crop)(residual_input)
    x = tf.keras.layers.Concatenate(axis=-1)([input, x])
    return x 

"""
def _crop_concat(input, residual_input):
    with tf.name_scope("crop_and_concat"):
        input_shape = tf.shape(input)
        residual_input_shape = tf.shape(residual_input)
        # offsets for the top left corner of the crop
        offsets = [0, (residual_input_shape[1] - input_shape[1]) // 2, (residual_input_shape[2] - input_shape[2]) // 2, (residual_input_shape[3] - input_shape[3]) // 2 , 0]
        size = [-1, input_shape[1], input_shape[2], input_shape[3], input_shape[4]+ residual_input_shape[4]]
        residual_input_crop = tf.slice(residual_input, offsets, size)
        output = tf.concat([input, residual_input_crop], axis = 4)
        return output
"""

#def _crop_concat(input, residual_input):
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
"""
# Inputs are tensors in format (b,l,l,l,c) b: batch size l 
# Crop Region from skip connection so that it matches the input from the lower layer

# Helper method to return the `idx`-th dimension of `tensor`, along with
# a boolean signifying if the dimension is dynamic.
def _get_dim(tensor, idx):
    static_shape = tensor.get_shape().as_list()[idx]
    if static_shape is not None:
    return static_shape #, False
    return tf.shape(tensor)[idx]# , True # Use dynamic input shape which is only defined once computational graph is built

#target_shape = input.shape.as_list() # Shape of the output generated by the upsampling layer
#given_shape = residual_input.shape.as_list() # Shape of the output used in skip connection
target_shape = [_get_dim(input,i) for i in range(1,4)]
given_shape = [_get_dim(residual_input,i) for i in range(1,4)]
print('input shape {} residual input shape {}'.format(target_shape, given_shape))

# Difference of shapes divided by two for symmetric crop

# Ignore the first dimension (batch dimension)
# Calculate the difference in each dimension (should be the same in Unet usecase)
# Crop away half of the difference at each side

crop = tuple([((given_shape[d]-target_shape[d])//2, (given_shape[d]-target_shape[d])//2 ) for d in range(3)])

#print(target_shape)
#print(given_shape)
print(crop) 

# Concatenate with cropped outputs in reverse order
x = tf.keras.layers.Cropping3D(cropping=crop, # a tuple (a,b,c) is interpreted as specifying symmetric croppings ((a,a),(b,b),(c,c)) for each dimension
                                name='crop')(residual_input)
x = tf.keras.layers.Concatenate(name='concat')([input, x])
return x

"""

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
            self.dropout = tf.keras.layers.Dropout(rate=0.5)
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
