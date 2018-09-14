from keras.applications import VGG16
from keras.activations import relu
from keras.optimizers import RMSprop
from keras.layers import Dense, Input, MaxPooling2D, \
                         Conv2D, Activation, Conv2DTranspose, \
                         UpSampling2D, concatenate, ZeroPadding2D, \
                         BatchNormalization
from keras.models import Sequential
from keras.utils import get_file
from keras.optimizers import RMSprop
from keras import Model

from model.custom_losses import bce_dice_loss, dice_loss, dice_coeff, \
                   false_negative, false_negative_pixel



WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

def get_layer(vgg16_model, input, index, is_freeze=False):
    layer = vgg16_model.get_layer(index=index)(input)
    
    if is_freeze:
        layer.trainable = False

    layer = BatchNormalization()(layer)

    return layer

def conv_bn_relu(filters, layer):
    layer = Conv2D(filters, (3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    return layer

def unet16(input_shape=(512, 512, 3), num_classes=1):
    """
    :param input_shape:
    :param num_classes:
    """

    vgg16_model = VGG16(include_top=False)
    inputs = Input(shape=input_shape)

    down1 = get_layer(vgg16_model, inputs, 1)
    down1 = get_layer(vgg16_model, down1, 2)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = get_layer(vgg16_model, down1_pool, 4)
    down2 = get_layer(vgg16_model, down2, 5)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 128

    down3 = get_layer(vgg16_model, down2_pool, 7)
    down3 = get_layer(vgg16_model, down3, 8)
    down3 = get_layer(vgg16_model, down3, 9)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 256

    down4 = get_layer(vgg16_model, down3_pool, 11)
    down4 = get_layer(vgg16_model, down4, 12)
    down4 = get_layer(vgg16_model, down4, 13)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 512

    down5 = get_layer(vgg16_model, down4_pool, 15)
    down5 = get_layer(vgg16_model, down5, 16)
    down5 = get_layer(vgg16_model, down5, 17)
    down5_pool = MaxPooling2D((2, 2), strides=(2, 2))(down5)
    # 512

    center = Conv2D(1024, (3, 3), padding='same')(down5_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up5 = UpSampling2D((2, 2))(center)
    up5 = concatenate([down5, up5], axis=3)
    up5 = conv_bn_relu(512, up5)
    up5 = conv_bn_relu(512, up5)
    up5 = conv_bn_relu(512, up5)

    up4 = UpSampling2D((2, 2))(up5)
    up4 = concatenate([down4, up4], axis=3)
    up4 = conv_bn_relu(512, up4)
    up4 = conv_bn_relu(512, up4)
    up4 = conv_bn_relu(512, up4)

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = conv_bn_relu(256, up3)
    up3 = conv_bn_relu(256, up3)
    up3 = conv_bn_relu(256, up3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = conv_bn_relu(128, up2)
    up2 = conv_bn_relu(128, up2)
    up2 = conv_bn_relu(128, up2)

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = conv_bn_relu(64, up1)
    up1 = conv_bn_relu(64, up1)
    up1 = conv_bn_relu(64, up1)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(lr=0.0001), 
                  loss=bce_dice_loss, 
                  metrics=[dice_coeff, false_negative, false_negative_pixel])

    return model


