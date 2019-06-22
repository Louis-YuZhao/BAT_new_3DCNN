#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 01:06:35 2019

@author: user
"""
import pickle
# import SimpleITK as sitk
# import glob
# import os

# rootdir = '/media/data/yuzhao/project/BAT_UNET_ThreeTwoD/test_predict'

# filelist = glob.glob(rootdir+ '/*/*/*.nrrd')
# for item in filelist:
#     root, ext = os.path.splitext(item)
#     image = sitk.ReadImage(item)
#     outputFile = os.path.join(root+'.nii.gz')
#     sitk.WriteImage(image,outputFile)
#     print(item)
#     os.remove(item)

# Load the required packages
# import time
# import psutil
# import numpy as np
# import pandas as pd
# import multiprocessing as mp

# # Check the number of cores and memory usage
# num_cores = mp.cpu_count()
# print("This kernel has ",num_cores,"cores and you can find the information regarding the memory usage:",psutil.virtual_memory())


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)
validation_indices = pickle_load('/media/data/yuzhao/project/BAT_3D_CNN/resultData/BAT_Combined_test_ids.pkl')