from keras.models import Sequential
from keras.layers import Reshape
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Concatenate, ReLU, DepthwiseConv2D, add
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.optimizers import Adam , SGD
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.regularizers import l2

from FSM import fsm


def conv2d_bn_relu(input, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def depth_conv_bn_relu(input, filters, kernel_size, strides=(1, 1), padding='same', dilation_rate=(1, 1),
                   initializer='he_normal', regularizer=l2(1e-5)):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding,
                        depthwise_initializer=initializer, use_bias=False, depthwise_regularizer=regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding=padding,
               kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def x_block(x, channels):
    res = conv2d_bn_relu(x, filters=channels, kernel_size=(1, 1))
    x = depth_conv_bn_relu(x, filters=channels, kernel_size=(3, 3))
    x = depth_conv_bn_relu(x, filters=channels, kernel_size=(3, 3))
    x = depth_conv_bn_relu(x, filters=channels, kernel_size=(3, 3))
    x = add([x, res])
    return x


def create_xception_unet_n(input_shape=(224, 192, 1), pretrained_weights_file=None):
    input = Input(input_shape)

    # stage_1
    x = x_block(input, channels=64)
    stage_1 = x

    # stage_2
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = x_block(x, channels=128)
    stage_2 = x

    # stage_3
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = x_block(x, channels=256)
    stage_3 = x

    # stage_4
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = x_block(x, channels=512)
    stage_4 = x

    # stage_5
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = x_block(x, channels=1024)
    x = fsm(x)

    # stage_4
    x = UpSampling2D(size=(2,2))(x)
    x = conv2d_bn_relu(x, filters=512, kernel_size=3)
    x = Concatenate()([stage_4, x])
    x = x_block(x, channels=512)

    # stage_3
    x = UpSampling2D(size=(2,2))(x)
    x = conv2d_bn_relu(x, filters=256, kernel_size=3)
    x = Concatenate()([stage_3, x])
    x = x_block(x, channels=256)

    # stage_2
    x = UpSampling2D(size=(2, 2))(x)
    x = conv2d_bn_relu(x, filters=128, kernel_size=3)
    x = Concatenate()([stage_2, x])
    x = x_block(x, channels=128)

    # stage_1
    x = UpSampling2D(size=(2, 2))(x)
    x = conv2d_bn_relu(x, filters=64, kernel_size=3)
    x = Concatenate()([stage_1, x])
    x = x_block(x, channels=64)

    # output
    x = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(x)

    # create model
    model = Model(input=input, output=x)
    model.summary()
    print('Create X-Net with N block, input shape = {}, output shape = {}'.format(input_shape, model.output.shape))

    # load weights
    if pretrained_weights_file is not None:
        model.load_weights(pretrained_weights_file, by_name=True, skip_mismatch=True)

    return model


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = create_xception_unet_n()
    model.summary()
