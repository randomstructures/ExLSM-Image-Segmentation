"""This script trains a Unet architecture and saves the trained model to a folder
   The training process is geared towards multi-gpu environments.
"""

#%% Input Ouput loctions
#TODO add custom modules to path
module_path = '..\\'
training_dataset_path = 'D:\\Janelia\\UnetTraining\\GapFilledMaskNetwork\\gapFilled_0923.h5'
save_dir = 'D:\\Janelia\\UnetTraining\\GapFilledMaskNetwork\\gpu_test\\'
model_file_name = 'gpuTest'
log_file_name = 'gpuTest'

#%% Architecture Parameters
#bottleneck_dropout_rate = 0.3
initial_filters = 4 # the number of filter maps in the first convolution operation

# ATTENTION these parameters are not freely changable -> CNN arithmetics
n_blocks = 2 # the number of Unet downsample/upsample blocks

#%% Training Parameters
test_fraction = 0.2 # fraction of training examples that are set aside in the validation set
affineTransform = True 
elasticDeformation = False
n_epochs = 2 # number of epochs to train the model
object_class_weight = 5 # factor by which pixels showing the neuron are multiplied in the loss function
dice_weight = 0.3 # contribution of dice loss (rest is cce)
batch_size = 1

#%% Setup

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(os.getcwd())
os.makedirs(save_dir, exist_ok=True) # Enshure that output folder for diagnostics is created


sys.path.append(module_path)

sys.path.append(module_path+'tools\\')
import metrics
import model
import utilities
import Dataset3D


# Perform dark GPU MAGIK

"""
# TODO maybe this fix does not apply for gpu cluster ? -> try out
# Fix for tensorflow-gpu issues that I found online... (don't ask me what it does)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
"""

# List the available devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Define a mirrored training strategy (maintain synchronized model instances and train on different batches)
mirrored_strategy = tf.distribute.MirroredStrategy()

#%% Build data input pipeline
print('loading training dataset')
#load the training dataset
dataset = Dataset3D.Dataset(training_dataset_path) # The Dataset3D class handles all file level i/o operations
# get a list of all records in the database and shuffle entries
entries = list(dataset.keys())
np.random.shuffle(entries)

# Make a train test split and retrieve a callable -> that produces a generator -> that yields the recods specified by the key list in random order
n_val = np.ceil(test_fraction*len(entries)).astype(np.int)
training = dataset.getGenerator(entries[:-n_val])
test = dataset.getGenerator(entries[-n_val:])

# Instantiate tf Datasets from the generator producing callables, specify the datatype and shape of the generator output
trainingset_raw = tf.data.Dataset.from_generator(training, 
    output_types=(tf.float32, tf.int32),
    output_shapes=(tf.TensorShape([220,220,220]),tf.TensorShape([220,220,220])))
testset_raw = tf.data.Dataset.from_generator(test, 
    output_types=(tf.float32, tf.int32),
    output_shapes=(tf.TensorShape([220,220,220]),tf.TensorShape([220,220,220])))

# the dataset is expected to be preprocessed (image normalized, mask binarized)
def preprocess(x,y):
    x = tf.expand_dims(x, axis=-1) # The unet expects the input data to have an additional channel axis.
    y = tf.one_hot(y, depth=2, dtype=tf.int32) # one hot encode to int tensor
    return x, y

def crop_mask(x, y, mask_size=(132,132,132)):
    # apply crop after batch dimension is added x and y have (b,x,y,z,c) format while mask size has (x,y,z) format => add offset of 1
    crop = [(y.shape[d+1]-mask_size[d])//2 for d in range(3)]
    #keras implicitly assumes channels last format
    y = tf.keras.layers.Cropping3D(cropping=crop)(y)
    return x, y

# chain dataset transformations to construct the input pipeline for training
trainingset = trainingset_raw.map(preprocess)
if affineTransform:
    trainingset = trainingset.map(utilities.tf_affine)
if elasticDeformation:
    trainingset = trainingset.map(utilities.tf_elastic)

trainingset = trainingset.batch(batch_size).map(crop_mask).prefetch(1)
testset = testset_raw.map(preprocess).batch(batch_size).map(crop_mask).prefetch(1)

#%% Construct model
with mirrored_strategy.scope():
    unet = model.build_unet(input_shape=(220,220,220,1), n_blocks=n_blocks, initial_filters=initial_filters)
#%% Setup Training
with mirrored_strategy.scope():
    unet.compile(
        optimizer = tf.keras.optimizers.Adam(),
        #loss = model.weighted_categorical_crossentropy(class_weights=[1,40]),
        loss = model.weighted_cce_dice_loss(class_weights=[1,object_class_weight], dice_weight=dice_weight),
        metrics = ['acc', metrics.MeanIoU(num_classes=2, name='meanIoU')]
                )
#%% Train
history = unet.fit(trainingset, epochs=n_epochs,
                   validation_data= testset,
                   verbose=2,
                   callbacks=[tf.keras.callbacks.ModelCheckpoint(save_dir+model_file_name+'{epoch}.h5', # Name of checkpoint file
                                                                 #save_best_only=True, # Wheter to save each epoch or only the best model according to a metric
                                                                 #monitor='val_meanIoU', # Which quantity should be used for model selection
                                                                 #mode='max' # We want this metric to be as large as possible
                                                                 ),
                              tf.keras.callbacks.CSVLogger(filename=save_dir+log_file_name+'.log')
                             ],
                   )

#%% Evaluate

## Generate some Plots from training history 
# Plot the evolution of the training loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Evolution of training loss')
plt.xlabel('epochs')
plt.ylabel('Spare Categorial Crossentropy')
plt.savefig(save_dir +'loss.png')

#Plot the evolution of pixel wise prediction accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Evolution of Accuracy')
plt.xlabel('epoch')
plt.ylabel('categorial accuracy')
plt.legend(['training', 'validation'])
plt.savefig(save_dir+'accuracy.png')

#Plot evolution of mean IoU Metric
plt.figure()
plt.plot(history.history['meanIoU'])
plt.plot(history.history['val_meanIoU'])
plt.title('Evolution of Mean IoU')
plt.xlabel('epoch')
plt.ylabel('mean intersection over union')
plt.legend(['training', 'validation'])
plt.savefig(save_dir+'iou.png')

#%% Tidy up
