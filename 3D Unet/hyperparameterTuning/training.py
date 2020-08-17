#%% Setup
import sys, getopt
import numpy as np 
import matplotlib.pyplot as plt 
import os, time
import h5py
from tqdm import tqdm
from importlib import reload
import itertools

import tensorflow as tf
import kerastuner as kt

# Import modules providing tools for image manipulation
sys.path.append('../../tools')
import tilingStrategy, Dataset3D, visualization
sys.path.append('..')
import utilities, model
reload(utilities)
reload(Dataset3D) 

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

#%% parse command line arguments (https://stackabuse.com/command-line-arguments-in-python/)

# Get full command-line arguments
full_cmd_arguments = sys.argv
# Keep all but the first (which is the script file path)
argument_list = full_cmd_arguments[1:]

# define options to parse with getopt
short_options = 'i:o:'
long_options = ['dataset=','output=']

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    print (str(err))
    sys.exit(2)

assert '-i' in arguments or '--dataset' in arguments, 'Specify location of the training dataset'
assert '-o' in arguments or '--output' in arguments, 'Specify location of output folder for the trained model'

#%% Data input pipeline

# Load images and masks from a previously created dataset
#base_dir = 'C:\\Users\\lillvisj\\linus\\'
base_dir = 'C:\\Users\\Linus Meienberg\\Documents\\ML Datasets\\FruSingleNeuron_20190707\\'
dataset_path = base_dir+'test_dataset.h5'
dataset = Dataset3D.Dataset(dataset_path) # The Dataset3D class handles all file level i/o operations

print('Dataset {} contains {} records'.format(dataset, len(dataset)))

# get a list of all records in the database
entries = list(dataset.keys())

# Make a train test split and retrieve a callable -> that produces a generator -> that yields the recods specified by the key list in random order
training = dataset.getGenerator(entries[:30])
test = dataset.getGenerator(entries[:30])

# gen = training() # calling the callable training produces a generator
# next(gen) -> (image, mask) yields the records identified by the key list

# Instantiate tf Datasets from the generator producing callables, specify the datatype and shape of the generator output
trainingset_raw = tf.data.Dataset.from_generator(training, 
    output_types=(tf.float32, tf.int32),
    output_shapes=(tf.TensorShape([220,220,220]),tf.TensorShape([132,132,132])))
testset_raw = tf.data.Dataset.from_generator(test, 
    output_types=(tf.float32, tf.int32),
    output_shapes=(tf.TensorShape([220,220,220]),tf.TensorShape([132,132,132])))

# The unet expects the input data to have an additional channel axis. We can add this during preprocessing
def addChannel(x,y):
    return tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)

# each entry is preprocessed by passing it through this function
def preprocess(x,y):
    # clip, and z normalize image
    x = tf.clip_by_value(x, 0, 1400)
    x = tf.subtract(x, 140)
    x = tf.divide(x, 40)
    # binarize mask
    y = tf.clip_by_value(y,0,1)
    return x, y
    
# chain dataset transformations to construct the input pipeline for training
# 1. preprocess the raw data by clipping and normalizing the image. The mask is binarized
# 2. A new axis is added to the tensors -> (x,y,z,c) format
# 3. elastic transformations are applied to both tensors. This mapping operation is parallelized to increase throughput as this is the most time consuming step in preprocessing. The number of parallel calls is automatically tuned for performance. 
# 4. random affine transformations are applied
# 5. entries are batched -> (b,x,y,z,c) format
# 6. Some training examples are prefetched which decouples preprocessing from model execution. The number of prefetched samples is tuned automatically for performance. 
trainingset = trainingset_raw.map(preprocess).map(addChannel).map(utilities.tf_elastic, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(utilities.tf_affine).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

# chain dataset transformations to construct the input pipeline for testing
# 1. preprocess the raw data by clipping and normalizing the image. The mask is binarized
# 2. A new axis is added to the tensors -> (x,y,z,c) format
# 3. random affine transformations are applied
# 5. entries are batched -> (b,x,y,z,c) format
# 6. Some training examples are prefetched which decouples preprocessing from model execution. The number of prefetched samples is tuned automatically for performance. 
testset = testset_raw.map(preprocess).map(addChannel).map(utilities.tf_affine).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

#%% Instantiate a Hyperparameter Tuner and let build and evaluate models
tuner = kt.Hyperband(
    model.build_unet_with_hp,
    objective = kt.Objective('val_IoU'),
    max_epochs = 5,
    directory = 'test\\',
    project_name = 'test_0815'
)
