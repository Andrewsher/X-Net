from keras.layers import Activation, Reshape, Lambda, dot, add, Input, BatchNormalization, ReLU, DepthwiseConv2D, Concatenate
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2


def conv2d_bn_relu(input, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def fsm(x):
    channel_num = x.shape[-1]

    res = x

    x = conv2d_bn_relu(x, filters=int(channel_num // 8), kernel_size=(3, 3))

    # x = non_local_block(x, compression=2, mode='dot')

    ip = x
    ip_shape = K.int_shape(ip)
    batchsize, dim1, dim2, channels = ip_shape
    intermediate_dim = channels // 2
    rank = 4
    if intermediate_dim < 1:
        intermediate_dim = 1

    # theta path
    theta = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    theta = Reshape((-1, intermediate_dim))(theta)

    # phi path
    phi = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    phi = Reshape((-1, intermediate_dim))(phi)

    # dot
    f = dot([theta, phi], axes=2)
    size = K.int_shape(f)
    # scale the values to make it size invariant
    f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    # g path
    g = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    g = Reshape((-1, intermediate_dim))(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])
    y = Reshape((dim1, dim2, intermediate_dim))(y)
    y = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-5))(y)
    y = add([ip, y])

    x = y
    x = conv2d_bn_relu(x, filters=int(channel_num), kernel_size=(3, 3))
    print(x)

    x = add([x, res])
    return x


def create_fsm_model(input_shape=(224, 192, 1)):
    input_img = Input(shape=input_shape)
    x = fsm(input_img)
    model = Model(input_img, x)
    model.summary()
    return model


if __name__ == '__main__':
    create_fsm_model(input_shape=(14, 12, 1024))
