import os
import collections

import numpy as np
from nilearn.image import new_img_like

from models.utils.utils import resize
from .utils import crop_img, crop_img_to, read_image


def reslice_image_set(set_of_files, image_shape, out_files=None, label_indices=None):
    '''
    1. crop
    2. resize

    input: 
    set_of_files, a tuple of image needed to be cropped and resized
    image_shape: proposed output image shape
    
    output: list of the image objects

    '''
    
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]

    crop_slices = get_cropping_parameters([set_of_files])
    print('reslice_image_set')
    images = list()
    for index, in_file in enumerate(set_of_files):
        interpolation = "continuous"
        if index in label_indices:
            interpolation = "nearest"
        images.append(read_image(in_file, image_shape=image_shape, crop=crop_slices, interpolation=interpolation))
        # read , crop, and resize image
    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images

def find_downsized_info(list_of_data_files, input_shape):
    foreground = get_complete_foreground(list_of_data_files)
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_cropping_parameters(list_of_data_files):
    # generate the whole foreground of the whole training dataset
    foreground = get_complete_foreground(list_of_data_files)
    # Will crop img, removing as many zero entries as possible
    # without touching non-zero entries.
    # return_slices: If True, the slices that define the cropped image will be returned.
    return crop_img(foreground, return_slices=True, copy=True)

def get_complete_foreground(list_of_data_files):
    '''
    generalize a common mask for the whole training dataset
    '''
    for i, set_of_files in enumerate(list_of_data_files):
        subject_foreground = get_foreground_from_set_of_files(set_of_files)
        # initialize
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1
    print('Runing get_complete_foreground')
    # nilearn.image.new_img_like(ref_niimg, data, affine=None, copy_header=False)
    # list_of_data_files[0][-1]) reference image
    return new_img_like(read_image(list_of_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001):
    '''
    From a set of images to canculate the foreground
    '''
    print('get_foreground_from_set_of_files')
    for i, image_file in enumerate(set_of_files):
        image = read_image(image_file)
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        # initialize 
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1

    return foreground


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage