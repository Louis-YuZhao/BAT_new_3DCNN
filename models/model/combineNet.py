import os
from models.GPU_config import gpuConfig
os.environ["CUDA_VISIBLE_DEVICES"]= gpuConfig['GPU_using']

import numpy as np
from keras.engine import Input, Model
from keras.layers import Conv3D, Activation, BatchNormalization, LeakyReLU, PReLU, Flatten, Dense, MaxPooling3D, concatenate
from keras.optimizers import Adam
from models.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
from keras_contrib.layers.normalization import InstanceNormalization, GroupNormalization
from keras import backend as K
K.set_image_data_format("channels_first")


def combineNet_3d(input_shape, n_labels=1, depth=5, n_base_filters=32,normMethod = 'batch_norm', activation_name = LeakyReLU,
                  initial_learning_rate=0.00001, include_label_wise_dice_coefficients=False, metrics=dice_coefficient):
    """
    Builds the 3D UNet Keras model.f
    
    Input:
    input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    n_labels: Number of binary labels that the model is learning.
    depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling    
    n_base_filters: The number of filters that the first layer in the convolution network will have. Following
                    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
                    to train the model.
    initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
                                        coefficient for each label as metric.
    metrics: List metrics to be calculated during model training (default is dice coefficient).    
    
    return: Untrained 3D UNet Model
    """
    div = depth // 2 
    mod = depth % 2    
    
    inputs = Input(input_shape)
    current_layer = inputs
    # add levels with max pooling
    for layer_depth in range(div):
        layer = create_convolution_block(input_layer = current_layer, n_filters = n_base_filters*(2**layer_depth),
                                          normMethod = normMethod)
        current_layer = layer
    if mod != 0:
        layer = create_convolution_block(input_layer = current_layer, n_filters = n_base_filters*(2**(div+mod-1)),
                                          normMethod = normMethod)
    for layer_depth in range(div-1, -1, -1):
        layer = create_convolution_block(input_layer = current_layer, n_filters = n_base_filters*(2**layer_depth),
                                          normMethod = normMethod)
        current_layer = layer
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    if n_labels > 1:
        act = Activation("softmax", name = 'goutput')(final_convolution)
    else:
        act = Activation("sigmoid", name = 'goutput')(final_convolution)
    
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    return model


def create_convolution_block(input_layer, n_filters, kernel=(3, 3, 3), activation=LeakyReLU,
                             padding='same', strides=(1, 1, 1), normMethod = 'batch_norm'):
    """
    strides:
    input_layer:
    n_filters:
    batch_normalization:
    kernel:
    activation: Keras activation layer to use. (default is 'relu')
    padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if normMethod == 'batch_norm':
        layer = BatchNormalization(axis=1)(layer)
    elif normMethod == 'instance_norm':
        layer = InstanceNormalization(axis=1)(layer)
    elif normMethod == 'group_norm':
        layer = GroupNormalization(groups=4, axis=1, epsilon=0.1)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)
    
def create_discriminator(input_shape, depth=4, n_base_filters=1, normMethod = 'batch_norm', learningRate = 0.001):
    """
    Build 3d GAN discriminator
    """
    inputs = Input(input_shape) #Segmentation input
    current_layer = inputs
    # 3d_CNN
    for layer_depth in range(depth):
        layer = create_convolution_block(input_layer = current_layer, n_filters = n_base_filters,
                                          normMethod = normMethod)
        layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)
        current_layer = layer
    # FCN
    current_layer = Flatten()(current_layer)
    for i in range(3):
        layer = Dense(2**(8-2*i), activation='relu')(current_layer)
        current_layer = layer
    current_layer = Dense(1, activation='sigmoid')(current_layer)

    discriminator = Model(inputs = inputs, outputs = current_layer)
    discriminator.compile(optimizer=Adam(lr=learningRate), loss='binary_crossentropy')
        
    return discriminator

def get_GAN(generator, discriminator, input_shape, learningRate = 0.01):
        
    inputs = Input(input_shape)
    G_out = generator(inputs)
    generator.name = 'gnet'
    D_out = discriminator(G_out)
    discriminator.name = 'dnet'
        
    GAN = Model(inputs=inputs, outputs=[G_out, D_out])
    GAN.compile(optimizer=Adam(lr=learningRate), loss={'gnet':dice_coefficient_loss, 'dnet': 'binary_crossentropy'},\
                  loss_weights={'gnet': 1., 'dnet': 0.3})
    
    return GAN