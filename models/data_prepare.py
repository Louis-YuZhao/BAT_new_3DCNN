# -*- coding: utf-8 -*-
'''
Generate one Epoch dataset without augumentation. The aim of this step is for saving time during training phase
'''
import os
import copy
import itertools
from random import shuffle

import numpy as np

from models.utils import pickle_dump, pickle_load
from models.utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from models.augment import augment_data, random_permutation_x_y

def get_training_and_validation_data(data_file, outputfolder, n_labels, labels, training_list, validation_list, 
                                    patch_shape = None, training_patch_overlap = 0, validation_patch_overlap = 0, 
                                    training_patch_start_offset = None, skip_blank = True):
    """
    Creates the training and validation generators that can be used when training the model.

      
    data_file: hdf5 file to load the data from.
    outputpath: output path to save the data.
    n_labels: Number of binary labels.
    labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
        should be equal to the n_labels value.
        Example: (10, 25, 50)
        The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    training_list: the list of training images.
    validation_list: the list of the validation images.
   
    patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
            (default is None)
    validation_patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
            patch_shape to not be None)    

    training_patch_start_offset: Tuple of length 3 containing integer values. Training data will randomly be
            offset by a number of pixels between (0, 0, 0) and the given tuple. (default is None)
    skip_blank: If True, any blank (all-zero) label images/patches will be skipped by the data generator.
          
    
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    # Set the number of training and testing samples per epoch correctly
    train_data_x, train_data_y, train_index_restore, trainDataCount = data_prepare(data_file, training_list,
                                        n_labels=n_labels,
                                        # Number of binary labels.
                                        labels=labels,
                                        # List or tuple containing the ordered label values in the image files.
                                        # The length of the list or tuple should be equal to the n_labels value. Example: (10, 25, 50)
                                        patch_shape=patch_shape,
                                        # Shape of the data to return with the generator. If None, the whole image will be returned.
                                        patch_overlap=training_patch_overlap,
                                        patch_start_offset=training_patch_start_offset,
                                        skip_blank=skip_blank)

    val_data_x, val_data_y, val_index_restore, valDataCount = data_prepare(data_file, validation_list,
                                          n_labels = n_labels,
                                          labels = labels,
                                          patch_shape = patch_shape,
                                          patch_overlap = validation_patch_overlap,
                                          skip_blank = skip_blank)
    training_prepare = (train_data_x, train_data_y, train_index_restore, trainDataCount)
    validation_prepare = (val_data_x, val_data_y, val_index_restore, valDataCount)
    np.save(training_prepare, os.path.join(outputfolder,'training_patch_data.npy'))
    np.save(validation_prepare, os.path.join(outputfolder,'test_patch_data.npy'))
    return trainDataCount, valDataCount

def data_prepare(data_file, index_list, batch_size=1, n_labels=1, labels=None, patch_shape=None, 
                    patch_overlap=0, patch_start_offset=None, shuffle_index_list=True, skip_blank=True):
    
    orig_index_list = index_list
    
    while True:
        x_list = list()
        y_list = list()
        index_list_restore = list()
        if patch_shape:
            index_list = create_patch_index_list(orig_index_list, data_file.root.data.shape[-3:], patch_shape,
                                                 patch_overlap, patch_start_offset)
        else:
            index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            # The method pop() removes and returns last object or obj from the list.
            index = index_list.pop()
            add_data_index(x_list, y_list, index_list_restore, data_file, index, patch_shape=patch_shape, skip_blank=skip_blank)
            if len(index_list) == 0 and len(x_list) > 0:
                data_x, data_y = convert_data(x_list, y_list, n_labels=n_labels, labels=labels)
    DataCount = len(x_list)
    index_restore = np.asarray(index_list_restore)
    return data_x, data_y, index_restore, DataCount 

def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
    patch_index = list()
    for index in index_list:
        if patch_start_offset is not None:
            random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
        else:
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
            # pathces: a dense multi-dimensional “meshgrid”.
            # 2d Array Example: array([[1,1,1],
            #                         [1,1,2],
            #                         [1,1,3]])
        patch_index.extend(itertools.product([index], patches))
        # itertools.product： cartesian product, equivalent to a nested for-loop
        # Example：[(1,2d-array),(2,2d-array),...]
    return patch_index


def add_data_index(x_list, y_list, index_list_restore, data_file, index, patch_shape=False, skip_blank=True):
    """
    Adds data from the data file to the given lists of feature and target data
    
    Input:
    x_list: list of data to which data from the data_file will be appended.
    y_list: list of data to which the target data from the data_file will be appended.
    index_list: list to restore the index of the patch
    data_file: hdf5 data file.
    index: index of the data file from which to extract the data.
    skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
   
    :return:
    """
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
    truth = truth[np.newaxis]

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)
        index_list_restore.append(index)

def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        # data: (channel, weight, height, depth)
        # truth: (weight, height, depth)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        x, y = data_file.root.data[index], data_file.root.truth[index, 0]
    return x, y

def convert_data(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    return x, y


def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    data: numpy array containing the label map with shape: (n_samples, 1, ...).
    n_labels: number of labels.
    labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y

def data_augmentation(data, truth, ori_data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
             patch_shape=False, skip_blank=True, permute=False):
    """
    Adds data from the data file to the given lists of feature and target data
    
    Input:
    data: list of data to which data from the ori_data_file will be appended.
    truth: list of data to which the target data from the ori_data_file will be appended.
    ori_data_file: hdf5 data file.
    index: index of the data file from which to extract the data.
    augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
        augment_distortion_factor)
    augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    augment_distortion_factor: if augment is True, this determines the standard deviation from the original
        that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
        augmentation from distorting the data in this way.
    skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    permute: will randomly permute the data (data must be 3D cube)
    
    :return:
    """
    truth=truth[0]
    if augment:
        if patch_shape is not None:
            affine = ori_data_file.root.affine[index[0]]
        else:
            affine = ori_data_file.root.affine[index]
        # augment_flip and augment_distortion
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])
    else:
        truth = truth[np.newaxis]
    return data, truth

def get_training_and_validation_generators(outputfolder, ori_data_file, batch_size, validation_batch_size=None, patch_shape=None,
                                        augment=False, augment_flip=True, augment_distortion_factor=0.25,  permute=False):
    """
    Creates the training and validation generators that can be used when training the model.         
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    if not validation_batch_size:
        validation_batch_size = batch_size
    training_data = np.load(os.path.join(outputfolder,'training_patch_data.npy'))
    validation_data = np.load(os.path.join(outputfolder,'test_patch_data.npy'))
    print('finish loading ......')
    training_generator = data_generator(training_data, ori_data_file,
                                        batch_size=batch_size,
                                        # Size of the batches that the training generator will provide.                                        
                                        augment=augment,
                                        # If True, training data will be distorted on the fly so as to avoid over-fitting.
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        patch_shape=patch_shape,
                                        # Shape of the data to return with the generator. If None, the whole image will be returned.
                                        permute=permute)

    validation_generator = data_generator(validation_data, ori_data_file,
                                          batch_size = validation_batch_size,
                                          patch_shape = patch_shape)

    # Set the number of training and testing samples per epoch correctly
    num_training_steps = get_number_of_steps(training_data[3],batch_size)
    print("Number of training steps: "+str(num_training_steps))

    num_validation_steps = get_number_of_steps(validation_data[3],validation_batch_size)
    print("Number of validation steps: "+str(num_validation_steps))

    return training_generator, validation_generator, num_training_steps, num_validation_steps

def data_generator(data_file, ori_data_file, batch_size=1, patch_shape=None, augment=False, augment_flip=True,
                   augment_distortion_factor=0.25, shuffle_index_list=True, skip_blank=True, permute=False):
    
    data_x, data_y, index_restore, DataCount = data_file
    
    index_list = list(range(DataCount))
    while True:
        x_list = list()
        y_list = list()
        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            # The method pop() removes and returns last object or obj from the list.
            index = index_list.pop()
            data, truth = data_augmentation(data_x[index], data_y[index], ori_data_file, index_restore[index], augment, augment_flip, 
            augment_distortion_factor, patch_shape, skip_blank, permute)
            x_list.append(data)
            y_list.append(truth)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                x = np.asarray(x_list)
                y = np.asarray(y_list)
                yield (x, y)
                x_list = list()
                y_list = list()
                
def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1