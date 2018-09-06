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

def load_labels(labels_dir=params.labels_dir):
    """
    return example:
    {
        'video1_0001.jpg': {
            'class': 0,
            'frame_num': 1
        },
        'video1_0002.jpg': {
            'class': 1,
            'frame_num': 11
        }
    }
    """
    labels = {}
    data = pd.read_csv(labels_dir)
    for index, row in data.iterrows():
        labels[row['filename']] = {
            'class': row['class'],
            'frame_num': row['frame_num']
        }
    return labels


model = params.model_factory(input_shape=params.input_shape)
labels = load_labels()

ids_train_split = glob.glob(params.folder_train+"*.*")
ids_valid_split = glob.glob(params.folder_val+"*.*")


print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))
print('Input net : {}'.format(params.image_size))


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
        folder = params.folder_train
    else:
        augmentation_functions = []
        ids_split = ids_valid_split
        folder = params.folder_valid
        
    while True:
        for start in range(0, len(ids_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_split))
            ids_batch = ids_split[start:end]
            for fname in ids_batch:
                img = imread(fname)
                img = cv2.resize(img, params.image_size)
                img_class = labels[fname]['class']
                
                for func in augmentation_functions:
                    img = func(img)
    
                x_batch.append(img)
                y_batch.append(img_class)
            x_batch = np.array(x_batch, np.float32) / 255
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
                             filepath='checkpoint/model.h5',
                             save_best_only=True)
            ]

model.fit_generator(generator=generator(is_train_generator=True),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=generator(is_train_generator=False),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
