import cv2
import numpy as np
import pandas as pd

import glob 
from skimage.io import imread
import params
import os

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from preprocess import *

# SET GPU USAGE TO MINIMUM
def get_session():
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(allow_growth=True)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
# ###############################

input_size = params.input_size
epochs = params.max_epochs
batch_size = params.batch_size

model = params.model_factory(input_shape=(input_size,input_size,3))

folder_mask = 'input/masks/'

folder_in = 'input/train/'
ids_train_split = glob.glob(folder_in+"*.*")

folder_val = 'input/valid/'
ids_valid_split = glob.glob(folder_val+"*.*")

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))
print('Input net : {}x{}'.format(input_size,input_size))


### THIS NEED CHANGE A LOT
def generator(is_train_generator=True):
    if is_train_generator:
        augmentation_functions = [
            randomHueSaturationValue,
            randomShiftScaleRotate,
            randomHorizontalFlip,
            randomTranspose
        ]
        ids_split = ids_train_split
        folder = folder_in
    else:
        augmentation_functions = []
        ids_split = ids_valid_split
        folder = folder_val
        
    while True:
        for start in range(0, len(ids_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_split))
            ids_batch = ids_split[start:end]
            for id in ids_batch:
                img = imread(id)
                img = cv2.resize(img, (input_size, input_size))
                file_name = id[len(folder):]
                mask = imread('input/masks/{}.gif'.format(file_name))
                mask = 255-mask
                mask = cv2.resize(mask, (input_size, input_size))
                
                for func in augmentation_functions:
                    img, mask = func(img, mask)
    
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=10,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               min_delta=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True)
            ]

model.fit_generator(generator=generator(is_train_generator=True),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=generator(is_train_generator=False),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
