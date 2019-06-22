import os

import numpy as np
import tables

from .normalize import normalize_data_storage, reslice_image_set

def write_data_to_file(list_of_data_files, output_file, image_shape, truth_dtype=np.uint8):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    
    Input: 
    list_of_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image. 
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    output_file: Where the hdf5 file will be written to.
    image_shape: Shape of the images that will be saved to the hdf5 file.
    truth_dtype: Default is 8-bit unsigned integer. 
    
    Output: Location of the hdf5 file with the image data written to it. 
    """
    n_samples = len(list_of_data_files)
    n_channels = len(list_of_data_files[0]) - 1 # data/image channel number
    
    # create hdf5 file for storing
    try:
        hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(output_file,
                                                                                  n_samples=n_samples,
                                                                                  n_channels=n_channels, 
                                                                                  image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(output_file)
        raise e

    write_image_data_to_file(list_of_data_files, data_storage, truth_storage,affine_storage,image_shape,
                             truth_dtype = truth_dtype, n_channels = n_channels)
    normalize_data_storage(data_storage)
    hdf5_file.close()
    return output_file 

def create_data_file(output_file, n_samples, n_channels, image_shape):
    '''
    create hdf5 file

    Input:
    output_file: output file name
    n_samples: sample number of the data
    image_shape: the tuple of the image shape
    n_channels: the channel number of the data

    Output:
    hdf5 files to store the related data
    '''

    hdf5_file = tables.open_file(output_file, mode='w')
    # complevel: compression lever (0-9), complib: compression library
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage

def create_patch_data_file(output_file, n_samples, n_channels, image_shape):
    '''
    create hdf5 file

    Input:
    output_file: output file name
    n_samples: sample number of the data
    image_shape: the tuple of the image shape
    n_channels: the channel number of the data

    Output:
    hdf5 files to store the related data
    '''

    hdf5_file = tables.open_file(output_file, mode='w')
    # complevel: compression lever (0-9), complib: compression library
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage


def write_image_data_to_file(list_of_data_files, data_storage, truth_storage, affine_storage, image_shape, n_channels,
                             truth_dtype = np.uint8):
    '''
    write images to hdf5 files

    Input:
    list_of_data_files: List of tuples containing the training data files. The modalities should be listed in
                the same order in each tuple. The last item in each tuple must be the labeled image. 
                Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
                        ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    data_storage : hdf5 file to restore the data
    truth_storage : hdf5 file to restore the ground truth label
    truth_storage : hdf5 file to restore the affine information of the image
    image_shape: the tuple of the image shape
    n_channels: the channel number of the data
    truth_dtype: the data type of the ground truth label
    '''
    for set_of_files in list_of_data_files:
        # crop and resize set of files of one sample 
        # return list of the image objects
        images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1)
        subject_data = [image.get_data() for image in images]
        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, images[0].affine, n_channels,
                            truth_dtype)
    return data_storage, truth_storage


def write_patch_data_to_file(data, truth, affine, data_storage, truth_storage, affine_storage, truth_dtype = np.uint8):
    '''
    write images to hdf5 files

    Input:
    list_of_data_files: List of tuples containing the training data files. The modalities should be listed in
                the same order in each tuple. The last item in each tuple must be the labeled image. 
                Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
                        ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    data_storage : hdf5 file to restore the data
    truth_storage : hdf5 file to restore the ground truth label
    truth_storage : hdf5 file to restore the affine information of the image
    truth_dtype: the data type of the ground truth label
    '''

    data_storage.append(data[np.newaxis])
    # data_storage 1 * n_channels * L*W*H
    truth_storage.append(np.asarray(truth, dtype=truth_dtype)[np.newaxis])
    # truth_storage 1 * L*W*H
    affine_storage.append(affine[np.newaxis])
    # affine 1 * 4 * 4
    return data_storage, truth_storage

def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype):
    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    # data_storage 1 * n_channels * L*W*H
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
    # truth_storage 1 * 1 * L*W*H
    affine_storage.append(np.asarray(affine)[np.newaxis])
    # affine 1 * 4 * 4

def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)

def get_data_from_file(data_file, key):
    f = open_data_file(data_file, 'r')
    if key == 'data':
        data = np.array(f.root.data)
    elif key == 'affine':
        data = np.array(f.root.affine)
    elif key == 'truth':
        data = np.array(f.root.truth)
    else:
        raise ValueError("No such data key...")
    f.close()
    return data

def get_discriminator_input_and_label(combineNet_output, combineNet_label):
    D_label_idx = np.argsort(np.random.choice(len(combineNet_label), len(combineNet_label)//2, replace=False))
    D_input = combineNet_output
    D_input[D_label_idx] = combineNet_label[:,np.newaxis,:,:,:][D_label_idx]
    D_label = np.zeros(len(combineNet_output))
    D_label[D_label_idx] = 1
    return D_input, D_label