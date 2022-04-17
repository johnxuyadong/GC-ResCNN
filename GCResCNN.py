from keras.layers import Conv1D, Conv2DTranspose, Input, MaxPooling1D, UpSampling1D, BatchNormalization, add, GlobalAveragePooling1D
from keras.layers import Concatenate, Activation, Add, Multiply, multiply, Dense, Reshape
from keras.models import Model
import keras as K
import numpy as np
import math
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import Softmax
from keras.layers import Permute
from keras.layers import add, dot

from keras import backend as K
import keras

from group_norm1 import GroupNormalization


def global_context_block(ip, reduction_ratio=16, transform_activation='linear'):
    """
    Adds a Global Context attention block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).

    # Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        reduction_ratio: Reduces the input filters by this factor for the
            bottleneck block of the transform submodule. Node: the reduction
            ratio must be set such that it divides the input number of channels,
        transform_activation: activation function to apply to the output
            of the transform block. Can be any string activation function availahle
            to Keras.

    # Returns:
        a tensor of same shape as input
    """
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    if rank > 3:
        flat_spatial_dim = -1 if K.image_data_format() == 'channels_first' else 1
    else:
        flat_spatial_dim = 1

    """ Context Modelling Block """
    # [B, ***, C] or [B, C, ***]
    input_flat = _spatial_flattenND(ip, rank)
    # [B, ..., C] or [B, C, ...]
    context = _convND(ip, rank, channels=1, kernel=1)
    # [B, ..., 1] or [B, 1, ...]
    context = _spatial_flattenND(context, rank)
    # [B, ***, 1] or [B, 1, ***]
    context = Softmax(axis=flat_spatial_dim)(context)

    # Compute context block outputs
    context = dot([input_flat, context], axes=flat_spatial_dim)
    # [B, C, 1]
    context = _spatial_expandND(context, rank)
    # [B, C, 1...] or [B, 1..., C]

    """ Transform block """
    # Transform bottleneck
    # [B, C // R, 1...] or [B, 1..., C // R]
    transform = _convND(context, rank, channels // reduction_ratio, kernel=1)
    # Group normalization acts as Layer Normalization when groups = 1
    transform = GroupNormalization(groups=1, axis=channel_dim)(transform)
    transform = Activation('relu')(transform)

    # Transform output block
    # [B, C, 1...] or [B, 1..., C]
    transform = _convND(transform, rank, channels, kernel=1)
    transform = Activation(transform_activation)(transform)

    # apply context transform
    out = add([ip, transform])

    return out


def _convND(ip, rank, channels, kernel=1):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, kernel, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (kernel, kernel), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (kernel, kernel, kernel), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)

    return x


def _spatial_flattenND(ip, rank):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    ip_shape = K.int_shape(ip)
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    if rank == 3:
        x = ip  # identity op for rank 3

    elif rank == 4:
        if channel_dim == 1:
            # [C, D1, D2] -> [C, D1 * D2]
            shape = [ip_shape[1], ip_shape[2] * ip_shape[3]]
        else:
            # [D1, D2, C] -> [D1 * D2, C]
            shape = [ip_shape[1] * ip_shape[2], ip_shape[3]]

        x = Reshape(shape)(ip)

    else:
        if channel_dim == 1:
            # [C, D1, D2, D3] -> [C, D1 * D2 * D3]
            shape = [ip_shape[1], ip_shape[2] * ip_shape[3] * ip_shape[4]]
        else:
            # [D1, D2, D3, C] -> [D1 * D2 * D3, C]
            shape = [ip_shape[1] * ip_shape[2] * ip_shape[3], ip_shape[4]]

        x = Reshape(shape)(ip)

    return x


def _spatial_expandND(ip, rank):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    if rank == 3:
        x = Permute((2, 1))(ip)  # identity op for rank 3

    elif rank == 4:
        if channel_dim == 1:
            # [C, D1, D2] -> [C, D1 * D2]
            shape = [-1, 1, 1]
        else:
            # [D1, D2, C] -> [D1 * D2, C]
            shape = [1, 1, -1]

        x = Reshape(shape)(ip)

    else:
        if channel_dim == 1:
            # [C, D1, D2, D3] -> [C, D1 * D2 * D3]
            shape = [-1, 1, 1, 1]
        else:
            # [D1, D2, D3, C] -> [D1 * D2 * D3, C]
            shape = [1, 1, 1, -1]

        x = Reshape(shape)(ip)

    return x


def activation(x, func='relu'):
    return Activation(func)(x)


def conv3x3(x, filter_num, stride=1):
    return Conv1D(filter_num, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)

def conv5x5(x, filter_num, stride=1):
    return Conv1D(filter_num, kernel_size=5, strides=stride, padding='same', use_bias=False)(x)


def conv7x7(x, filter_num, stride=1):
    return Conv1D(filter_num, kernel_size=7, strides=stride, padding='same', use_bias=False)(x)

def Attention(x):
    reduction = 4                                                                # reduction =2    #  reduction =4
    x = Conv1D(filters=112, kernel_size=1, use_bias=False, padding='same')(x)     #16               # 32
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=448, kernel_size=1, use_bias=False, padding='same')(x)    # 32                # 128
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    return x
#
#
def TAM(x):
    out = Attention(x)
    x = keras.layers.add([x, keras.layers.multiply([x, out])])
    x = Conv1D(filters=128, kernel_size=3, use_bias=False, padding='same')(x)        # 32
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def BasicBlock3x3(x, filter_num, stride=2):
    filter_num3 =64
    # residual = x

    out1 = conv3x3(x, filter_num, stride)
    out1 = BatchNormalization()(out1)
    out1 = Activation('relu')(out1)

    out2 = conv3x3(out1, filter_num, stride=1)
    out2 = BatchNormalization()(out2)
    out2 = Activation('relu')(out2)


    if stride != 1 or filter_num != filter_num3:
        residual = Conv1D(filter_num, kernel_size=1, strides=2, padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

    out = add([residual, out2])
    out = Activation('relu')(out)

    return out1, out2, out

def side_branch(x, factor):
    x = global_context_block(x, reduction_ratio=8)
    x = Conv1D(64, 3, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = global_context_block(x, reduction_ratio=8) # 放在后面效果得到认证
    return x


def RDBlocks(x ,name , count = 3 , g=32):
    ## 6 layers of RDB block
    ## this thing need to be in a damn loop for more customisability
    li = [x]
    pas = Conv1D(filters=g, kernel_size=3, strides=1, padding='same' , activation='relu' , name = name +'_conv1')(x)

    for i in range(2, count+1):
        li.append(pas)
        out = Concatenate(axis=-1)(li)# conctenated out put
        pas = Conv1D(filters=g, kernel_size=3, strides=1, padding='same', activation='relu', name=name+'_conv'+str(i))(out)

    # feature extractor from the dense net
    li.append(pas)
    out = Concatenate(axis=-1)(li)
    feat = Conv1D(filters=21, kernel_size=1, strides=1, padding='same', activation='relu', name=name + '_Local_Conv')(out)

    feat = Add()([feat, x])
    return feat


def vgg_rcf(input_shape=None):
    # Input
    inputs = Input(shape=input_shape)  # 320, 480, 3

    # Block 1
    [x1_conv1, x1_conv2, k1] = BasicBlock3x3(inputs, 64, 2)
    x1_conv1_out = Conv1D(21, 3, activation='relu', padding='same', name='block1_conv_o1')(x1_conv1)
    x1_conv2_out = Conv1D(21, 3, activation='relu', padding='same', name='block1_conv_o2')(x1_conv2)
    x1_add = Add()([x1_conv1_out,x1_conv2_out])
    b1 = side_branch(x1_add, 1) # 480 480 1
    x1 = k1

    # Block 2
    [x2_conv1, x2_conv2, k2] = BasicBlock3x3(x1, 128, 2)
    x2_conv1_out = Conv1D(21, 3, activation='relu', padding='same', name='block2_conv_o1')(x2_conv1)
    x2_conv2_out = Conv1D(21, 3, activation='relu', padding='same', name='block2_conv_o2')(x2_conv2)
    x2_add = Add()([x2_conv1_out,x2_conv2_out])
    b2= side_branch(x2_add, 2) # 480 480 1
    x2 = k2

    # Block 3
    [x3_conv1, x3_conv2, k3] = BasicBlock3x3(x2, 256, 2)
    x3_conv1_out = Conv1D(21, 3, activation='relu', padding='same', name='block5_conv_o1')(x3_conv1)
    x3_conv2_out = Conv1D(21, 3, activation='relu', padding='same', name='block5_conv_o2')(x3_conv2)
    x3_add = Add()([x3_conv1_out, x3_conv2_out])
    b3= side_branch(x3_add, 16) # 480 480 1



    # fuse
    b1 = MaxPooling1D(4, strides=4, padding='same')(b1)# 240 240 64
    b2 = MaxPooling1D(2, strides=2, padding='same')(b2)# 240 240 64
    fuse = Concatenate(axis=-1)([b1, b2, b3, k3])
    fuse = TAM(fuse)

    model = Model(inputs=[inputs], outputs=fuse)

    return model
