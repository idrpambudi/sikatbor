"""Inception V3 model for Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras import backend
from keras import engine
from keras import layers
from keras import models
from keras import utils as keras_utils


from keras.applications.imagenet_utils import _obtain_input_shape, decode_predictions
from model.custom_losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff, false_negative, false_negative_pixel
from keras.optimizers import RMSprop


WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              trainable=True):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
        trainable=trainable)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name, trainable=trainable)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def conv2d_transpose_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              trainable=True,
              use_bias=True):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2DTranspose(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=conv_name,
        trainable=trainable)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name, trainable=trainable)(x)
    x = layers.Activation('relu', name=name)(x)
    return x

def InceptionV3(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(512,512,3),
                pooling=None,
                classes=1):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(
    #     input_shape,
    #     default_size=299,
    #     min_size=139,
    #     data_format=backend.image_data_format(),
    #     require_flatten=False,
    #     weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='same')
    
    x = conv2d_bn(x, 32, 3, 3, padding='same')
    x = conv2d_bn(x, 64, 3, 3)
    skip1 = x
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = conv2d_bn(x, 80, 1, 1, padding='same')
    x = conv2d_bn(x, 192, 3, 3, padding='same')
    skip2 = x
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')
    
    skip3 = x

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='same')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='same')
        # branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')
    
    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    skip4 = x
    
    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='same')
                          # strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='same')
        # branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='inception_v3')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = keras_utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
        
    return [x, skip4, skip3, skip2, skip1]


def inc3_mirror(include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(512,512,3),
            pooling=None,
            classes=1):
    
    channel_axis = 3
    img_input = layers.Input(shape=input_shape)
    inception3_end_layers = InceptionV3(weights=weights, input_tensor=img_input, input_shape=input_shape)
    
    x = inception3_end_layers[0]
    
    # decoder_mixed 9: 16 x 16 x 2048
    for i in range(2):
        branch1x1 = conv2d_transpose_bn(x, 320, 1, 1)

        branch3x3 = conv2d_transpose_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_transpose_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_transpose_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='decoder_mixed9_' + str(i))

        branch3x3dbl = conv2d_transpose_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_transpose_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_transpose_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_transpose_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_transpose_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='decoder_mixed' + str(9 + i))
    
    # decoder_mixed 8: 32 x 32 x 1280
    branch3x3 = conv2d_transpose_bn(x, 192, 1, 1)
    branch3x3 = conv2d_transpose_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='same')
                          # strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_transpose_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_transpose_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_transpose_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_transpose_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='same')
        # branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.UpSampling2D()(x)
    # branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='decoder_mixed8')
    
    # Create skip layer
    x = layers.concatenate([x, inception3_end_layers[1]], axis=-1)

    # mixed 7: 32 x 32 x 768
    branch1x1 = conv2d_transpose_bn(x, 192, 1, 1)

    branch7x7 = conv2d_transpose_bn(x, 192, 1, 1)
    branch7x7 = conv2d_transpose_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_transpose_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_transpose_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_transpose_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='decoder_mixed7')
   
    # mixed 5, 6: 32 x 32 x 768
    for i in range(2):
        branch1x1 = conv2d_transpose_bn(x, 192, 1, 1)

        branch7x7 = conv2d_transpose_bn(x, 160, 1, 1)
        branch7x7 = conv2d_transpose_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_transpose_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_transpose_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_transpose_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='decoder_mixed' + str(5 + i)) 

        
    # mixed 4: 32 x 32 x 768
    branch1x1 = conv2d_transpose_bn(x, 192, 1, 1)

    branch7x7 = conv2d_transpose_bn(x, 128, 1, 1)
    branch7x7 = conv2d_transpose_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_transpose_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_transpose_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_transpose_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_transpose_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='decoder_mixed4')

    
    # mixed 3: 64 x 64 x 768
    branch3x3 = conv2d_transpose_bn(x, 384, 3, 3, strides=(2, 2), padding='same')

    branch3x3dbl = conv2d_transpose_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_transpose_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_transpose_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='same')
        # branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')
    
    branch_pool = layers.UpSampling2D()(x)
    
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='decoder_mixed3')
    
    # Create skip layer
    x = layers.concatenate([x, inception3_end_layers[2]], axis=-1)

    # mixed 2: 128 x 128 x 256
    branch1x1 = conv2d_transpose_bn(x, 64, 1, 1)

    branch5x5 = conv2d_transpose_bn(x, 48, 1, 1)
    branch5x5 = conv2d_transpose_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_transpose_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_transpose_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_transpose_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_transpose_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='decoder_mixed2')
    
    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_transpose_bn(x, 64, 1, 1)

    branch5x5 = conv2d_transpose_bn(x, 48, 1, 1)
    branch5x5 = conv2d_transpose_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_transpose_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_transpose_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_transpose_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_transpose_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='decoder_mixed1')    
    
    # mixed 0, 1, 2: 128 x 128 x 256
    branch1x1 = conv2d_transpose_bn(x, 64, 1, 1)

    branch5x5 = conv2d_transpose_bn(x, 48, 1, 1)
    branch5x5 = conv2d_transpose_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_transpose_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_transpose_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_transpose_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_transpose_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='decoder_mixed0')
    
    x = layers.UpSampling2D()(x)
    
    # Create skip layer
    x = layers.concatenate([x, inception3_end_layers[3]], axis=-1)
    
    x = conv2d_transpose_bn(x, 192, 3, 3, padding='same')
    x = conv2d_transpose_bn(x, 80, 1, 1, padding='same')
    
    x = layers.UpSampling2D()(x)
    
    # Create skip layer
    x = layers.concatenate([x, inception3_end_layers[4]], axis=-1)
    
    x = conv2d_transpose_bn(x, 64, 3, 3)
    x = conv2d_transpose_bn(x, 32, 3, 3, padding='same')
    
    x = conv2d_transpose_bn(x, 32, 3, 3, strides=(2, 2), padding='same')
    
    # ngide
    res = conv2d_transpose_bn(img_input, 32, 3, 3)
    res = conv2d_transpose_bn(res, 32, 3, 3)
    x = layers.concatenate([x, res], axis=-1)
    x = conv2d_transpose_bn(x, 32, 3, 3)
    x = conv2d_transpose_bn(x, 32, 3, 3, padding='same')

    x = layers.Conv2D(classes,(1,1),activation="sigmoid", name="out_inc3_mirror")(x)

    model = models.Model(img_input, x, name='inc3_mirror')
    model.compile(optimizer=RMSprop(lr=0.0001), loss=dice_loss, metrics=[dice_coeff, false_negative, false_negative_pixel])
        
    print(model.summary())
    return model


def inc3_u(include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(512,512,3),
            pooling=None,
            classes=1):
    
    img_input = layers.Input(shape=input_shape)
    inception3_end_layers = InceptionV3(input_tensor=img_input, input_shape=input_shape)
    x = inception3_end_layers[0]
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([x, inception3_end_layers[1]], axis=3)

    x = conv2d_transpose_bn(x, 512, 3, 3)
    x_res = x
    x = conv2d_transpose_bn(x, 512, 3, 3)
    x = conv2d_transpose_bn(x, 512, 3, 3)
    x = layers.add([x_res, x])
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([x, inception3_end_layers[2]], axis=3)

    x = conv2d_transpose_bn(x, 256, 3, 3)
    x_res = x
    x = conv2d_transpose_bn(x, 256, 3, 3)
    x = conv2d_transpose_bn(x, 256, 3, 3)
    x = layers.add([x_res, x])
    x = layers.Activation('relu')(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([x, inception3_end_layers[3]], axis=3)
    # 56
    
    x = conv2d_transpose_bn(x, 128, 3, 3)
    x_res = x
    x = conv2d_transpose_bn(x, 128, 3, 3)
    x = conv2d_transpose_bn(x, 128, 3, 3)
    x = layers.add([x_res, x])
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([x, inception3_end_layers[4]], axis=3)

    x = conv2d_transpose_bn(x, 64, 3, 3)
    x_res = x    
    x = conv2d_transpose_bn(x, 64, 3, 3)
    x = conv2d_transpose_bn(x, 64, 3, 3)
    x = layers.add([x_res, x])
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D((2, 2))(x)

    res = conv2d_bn(img_input, 32, 3, 3)
    x_res = res
    res = conv2d_bn(res, 32, 3, 3)
    res = conv2d_bn(res, 32, 3, 3)
    res = layers.add([x_res, res])
    x = layers.Activation('relu')(x)
    
    x = layers.concatenate([x, res], axis=3)

    x = conv2d_transpose_bn(x, 32, 3, 3)
    x_res = x
    x = conv2d_transpose_bn(x, 32, 3, 3)
    x = conv2d_transpose_bn(x, 32, 3, 3)
    x = layers.add([x_res, x])
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(classes,(1,1),activation="sigmoid", name="out_inc3_u_net")(x)

    model = models.Model(img_input, x, name='inc3_u_net')
    model.compile(optimizer=RMSprop(lr=0.0001), loss=dice_loss, metrics=[dice_coeff, false_negative, false_negative_pixel])
        
    print(model.summary())
    return model
