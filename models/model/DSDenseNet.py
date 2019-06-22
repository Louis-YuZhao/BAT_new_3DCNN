import os
from models.GPU_config import gpuConfig
os.environ["CUDA_VISIBLE_DEVICES"]= gpuConfig['GPU_using']

from functools import partial

from keras.layers import Input, Conv3D, Activation, LeakyReLU, PReLU, Add, SpatialDropout3D, MaxPooling3D, UpSampling3D, Deconvolution3D
from keras.engine import Model
from keras.optimizers import Adam
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras import backend as K  
K.set_image_data_format('channels_first') 
from ..metrics import weighted_dice_coefficient_loss

def DSDenseNet_model(input_shape=(4, 128, 128, 128),
                     n_base_filters=16, 
                     depth=5, 
                     dropout_rate=0.3,
                     n_segmentation_levels=3, 
                     n_labels=4, 
                     optimizer=Adam, 
                     initial_learning_rate=5e-4,
                     loss_function=weighted_dice_coefficient_loss, 
                     activation_name="sigmoid"):
    
    # this model is channel first
    """
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    denseBlock_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)        

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        summation_layer, n_dense_filter = down_dense_block(in_conv, 3, n_level_filters, growth_rate_k=16, kernel=(3,3,3), 
                                                bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                                                activation=LeakyReLU, batch_normalization=False, instance_normalization=True,
                                                grow_nb_filters=True, return_concat_list=False)        
        denseBlock_filters.append(n_dense_filter)
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, denseBlock_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)

        localization_output, n_dense_filter = up_dense_block(concatenation_layer, 3, level_filters[level_number], growth_rate_k=16, 
                                                    kernel=(3,3,3), bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                                                    activation=LeakyReLU, batch_normalization=False, instance_normalization=True, 
                                                    grow_nb_filters=True, return_concat_list=False)
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block(current_layer, n_filters=n_labels, kernel=(1, 1, 1)))
        # n_filters = n_labels: reason why using sigmoid instead of softmax

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model

def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2

def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution

def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

def create_convolution_block(input_layer, n_filters, kernel=(3, 3, 3), activation=LeakyReLU,
                             padding='same', strides=(1, 1, 1), batch_normalization=False, instance_normalization=True):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def conv_block(input_layer, nb_filter, kernel=(3,3,3), bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                activation=LeakyReLU, batch_normalization=False, instance_normalization=True):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        input_layer: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    layer = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_layer)
    layer = Activation('relu')(layer)

    if bottleneck:
        n_filters = nb_filter * 4  

        layer = Conv3D(n_filters, kernel=(1,1,1), kernel_initializer='he_normal', padding='same', use_bias=False,
                        kernel_regularizer=l2(weight_decay))(layer)
        if batch_normalization:
            layer = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(layer)
        elif instance_normalization:
            layer = InstanceNormalization(axis=concat_axis, epsilon=1.1e-5)(layer)
        if activation is None:
            return Activation('relu')(layer)
        else:
            return activation()(layer)

    layer = Conv3D(nb_filter, kernel=kernel, kernel_initializer='he_normal', padding='same', use_bias=False)(layer)
    
    if batch_normalization:
            layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

    if dropout_rate:
        layer = SpatialDropout3D(rate=dropout_rate, data_format=K.image_data_format())(layer)        
    return layer

def down_dense_block(layer, nb_layers, n_filters_k0, growth_rate_k, kernel=(3,3,3), bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  activation=LeakyReLU, batch_normalization=False, instance_normalization=True, grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        layer: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        n_filters_k0: number of input filters
        growth_rate_k: growth rate
        kernel: kernel of the 3D conv
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [layer]
    for i in range(nb_layers):
        convBlock = conv_block(layer, growth_rate_k, kernel, bottleneck, dropout_rate, weight_decay, activation, batch_normalization, instance_normalization)
        x_list.append(convBlock)

        layer = concatenate([layer, convBlock], axis=concat_axis)

        if grow_nb_filters:
            n_filters_k0 += growth_rate_k

    if return_concat_list:
        return layer, n_filters_k0, x_list
    else:
        return layer, n_filters_k0

def up_dense_block(layer, nb_layers, n_filters_k0, growth_rate_k, kernel=(3,3,3), bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  activation=LeakyReLU, batch_normalization=False, instance_normalization=True, grow_nb_filters=True, return_concat_list=False):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x_list = [layer]
    for i in range(nb_layers-1):
        convBlock = conv_block(layer, growth_rate_k, kernel, bottleneck, dropout_rate, weight_decay, activation, batch_normalization, instance_normalization)
        x_list.append(convBlock)

        layer = concatenate([layer, convBlock], axis=concat_axis)

        if grow_nb_filters:
            n_filters_k0 += growth_rate_k
    #--------------------------------------#
    # # in the last layer, the kernel should be (1,1,1)
    # convBlock = conv_block(layer, output_filter_num, (1,1,1), bottleneck, dropout_rate, weight_decay, activation, batch_normalization, instance_normalization)
    # x_list.append(convBlock)

    # layer = concatenate([layer, convBlock], axis=concat_axis)

    # if grow_nb_filters:
    #     n_filters_k0 += growth_rate_k    
    # ------------------------------------- #
    if return_concat_list:
        return layer, n_filters_k0, x_list
    else:
        return layer, n_filters_k0