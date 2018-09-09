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
from keras import optimizers, losses, metrics
from model import custom_losses
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
            'frame_number': 1
        },
        'video1_0002.jpg': {
            'class': 1,
            'frame_number': 11
        }
    }
    """
    labels = {}
    data = pd.read_csv(labels_dir)
    for index, row in data.iterrows():
        labels[row['filename']] = {
            'class': row['class'],
            'frame_number': row['frame_number']
        }
    return labels


#from stackoverflow
def get_model_memory_usage(model, batch_size=params.batch_size):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


model = params.model_factory(input_shape=params.input_shape)
model.compile(
    optimizer=optimizers.RMSprop(lr=0.01), 
    loss='binary_crossentropy', 
    #loss=custom_losses.floss,
    metrics=['accuracy', custom_losses.fmeasure, custom_losses.recall, custom_losses.precision]
)

labels = load_labels()

ids_train_split = glob.glob(params.folder_train+"*.*")
ids_valid_split = glob.glob(params.folder_valid+"*.*")


print('Memory needed estimation: {}GB'.format(get_model_memory_usage(model)))
print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))
print('Input net : {}'.format(params.image_size))

def generator(is_train_generator=True):
    if is_train_generator:
        augmentation_functions = [
            randomHueSaturationValue,
            randomShiftScaleRotate,
            randomHorizontalFlip
        ]
        ids_split = ids_train_split
    else:
        augmentation_functions = []
        ids_split = ids_valid_split
        
    while True:
        for start in range(0, len(ids_split), params.batch_size):
            x_batch = []
            y_batch = []
            end = min(start + params.batch_size, len(ids_split))
            ids_batch = ids_split[start:end]
            for fname in ids_batch:
                img = cv2.imread(fname)
                img = cv2.resize(img, params.image_size[::-1])

                fname = fname.split('/')[-1]
                img_class = [0,1] if labels[fname]['class']==1 else [1,0]
                
                for func in augmentation_functions:
                    img = func(img)
    
                x_batch.append(img)
                y_batch.append(img_class)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32)
            # print(str(x_batch.shape) + "===" + str(y_batch))
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
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(params.batch_size)),
                    epochs=params.max_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator(is_train_generator=False),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(params.batch_size)))
