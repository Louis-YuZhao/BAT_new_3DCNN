# -*- coding: utf-8 -*-
import os
import copy
import string
from random import shuffle
import itertools

import numpy as np

from .utils import pickle_dump, pickle_load
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from .augment import augment_data, random_permutation_x_y

#%%
def set_train_validation_test(trainList, testList, t_v_split, training_file, validation_file, test_file, overwrite):
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        training_list, validation_list = split_list(trainList, split=t_v_split)
        # training: shuffled training orders
        # testing: shuffled testing orders
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        pickle_dump(testList, test_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)

def set_train_validation(training_list, validation_list, training_file, validation_file, overwrite):

    if overwrite or not os.path.exists(training_file):
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)

def get_cross_validation_split(data_file, training_file, validation_file, crossValconfig, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param crossValconfig:
    :param overwrite:
    :return:
    """

    NumFold = crossValconfig['NumFold']
    currentFold = crossValconfig['currentFold']
    defaultOrder = crossValconfig['defaultOrder']
    groupOneList = crossValconfig['groupOne']
    groupTwoList = crossValconfig['groupTwo']

    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        
        if defaultOrder:
            sample_list = list(range(nb_samples))
            splitedList = list_split(sample_list, NumFold)
            training_list, validation_list = choose_train_test_list(splitedList, currentFold)
            pickle_dump(training_list, training_file)
            pickle_dump(validation_list, validation_file)
            return training_list, validation_list
        else:
            splitedListOne = list_split(groupOneList, NumFold)
            training_list_one, validation_list_one = choose_train_test_list(splitedListOne, currentFold)
            splitedListTwo = list_split(groupTwoList, NumFold)
            training_list_two, validation_list_two = choose_train_test_list(splitedListTwo, currentFold)

            training_list = training_list_one + training_list_two
            validation_list = validation_list_one + validation_list_two
            shuffle(training_list)
            shuffle(validation_list)
            pickle_dump(training_list, training_file)
            pickle_dump(validation_list, validation_file)
            return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)

def list_split(lst, num):
    k, m = divmod(len(lst), num)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num)]

def choose_train_test_list(splitedList, currentFold):
        if (currentFold<0) or (currentFold>len(splitedList)):
            raise ValueError('currentFold is not right')
    
        tempList = copy.copy(splitedList)
        validation_list = tempList.pop(currentFold)
        training_list = []
        for listI in tempList:
            training_list = training_list + listI       
        return training_list, validation_list

def get_validation_split(data_file, training_file, validation_file, data_split=0.8, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        # training: shuffled training orders
        # testing: shuffled testing orders
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)

def split_list(input_list, split=0.8, shuffle_list=True):
    
    """
    split the list

    return:
    training: shuffled training orders
    testing: shuffled testing orders
    """
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing

class GetListFromFiles(object):
    """ class for Brown Adipose Tissue preprocessing """
    def __init__(self):
        self.trainDataDict = None
        self.testDataDict = None
        self.trainNum = None
        self.testNum = None
    
    def readImage(self, train_path, test_path):
        outputDict = dict()
        self.trainDataDict, self.trainNum = self.__inputImage(train_path)
        self.testDataDict, self.testNum = self.__inputImage(test_path)
        for key in self.testDataDict.keys():
            outputDict[key] = self.trainDataDict[key] + self.testDataDict[key]
        return outputDict, self.trainNum, self.testNum

    def __inputImage(self, data_path_dict):
        
        DataDict = dict()
        keys = list(data_path_dict.keys())
        i = 0
        for key in keys:
            DataDict[key] = self.__readTxtIntoList(data_path_dict[key])
            if i == 0:
                DataNum = len(DataDict[key])
            else:
                if DataNum != len(DataDict[key]):
                    raise ValueError('the length of the files should be same: ' + key)
            i=i+1

        for i in range(DataNum):
            checkItem = '_'.join((os.path.basename(DataDict[keys[0]][i])).split("_")[1:2])
            for j in range(1,len(keys)):
                if not (checkItem in (os.path.basename(DataDict[keys[j]][i]))):
                    raise ValueError(str(i)+' th do not match each other')
        return DataDict, DataNum
   
    def __readTxtIntoList(self, filename):
        flist = []
        with open(filename) as f:
            flist = f.read().splitlines()
        return flist

    def __WriteListtoFile(self, filelist, filename):
        with open(filename, 'w') as f:
            for i in filelist:
                f.write(i+'\n')
        return 1