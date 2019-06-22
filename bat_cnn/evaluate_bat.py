# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
#import matplotlib
##matplotlib.use('agg')
import matplotlib.pyplot as plt
from train_isensee2017 import config

root_dir = os.path.abspath("/media/data/yuzhao/project/PSMA/Cross_Validation/resultData_5")
acc_area = 25

def get_bone_lesion(data):
    output = data == 1
    output.dtype = np.uint8
    return output
def get_lymphNode_lesion(data):
    output = data == 2
    output.dtype = np.uint8
    return output
def get_prostate_lesion(data):
    output = data == 3
    output.dtype = np.uint8
    return output

def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

prediction_dir = os.path.join(root_dir,"prediction_isensee2017")
def main():
    # header_choose = ("Bone Lesion",)
    # masking_functions_choose = (get_bone_lesion,)
    
    # header_choose = ("Lymph Node Lesion",)
    # masking_functions_choose = (get_lymphNode_lesion,)
    
    header_choose = ("Bone Lesion","Lymph Node Lesion","Local Lesion")
    masking_functions_choose = (get_bone_lesion, get_lymphNode_lesion, get_prostate_lesion)
    headerlist = []
    masking_functions_list = []
    for i in range(len(header_choose)):
        if (i+1) in config["labels"]:
            headerlist.append(header_choose[i])
            masking_functions_list.append(masking_functions_choose[i])
    header = tuple(headerlist)
    masking_functions = tuple(masking_functions_list)
    rows = list()
    case_folder_list = glob.glob(os.path.join(prediction_dir,"validation_case*"))
    case_folder_list.sort()
    for case_folder in case_folder_list:
        print(str(case_folder))
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = sitk.ReadImage(truth_file)
        truth = sitk.GetArrayFromImage(truth_image)
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = sitk.ReadImage(prediction_file)
        prediction = sitk.GetArrayFromImage(prediction_image)        
        # rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])
        dice_list = list()
        for func in masking_functions:
            if np.sum(func(truth)) > acc_area:
                dice_list.append(dice_coefficient(func(truth), func(prediction)))
            else:
                dice_list.append(np.int(0))
            print('truth:'+ str(np.sum(func(truth))))
            print('dice:' + str(dice_coefficient(func(truth), func(prediction))))
        rows.append(dice_list)
    df = pd.DataFrame.from_records(rows, columns=header)
    df.to_csv(os.path.join(prediction_dir,"brats_scores.csv"))

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        values = values[np.isnan(values) == False]
        scores[score] = values[values != np.int(0)]

    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig(os.path.join(root_dir, "prediction_isensee2017","validation_scores_boxplot.png"))
    plt.show()
    plt.close()

    training_df = pd.read_csv("./training_isensee2017.log").set_index('epoch')

    plt.plot(training_df['loss'].values, label='training loss')
    plt.plot(training_df['val_loss'].values, label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, len(training_df.index)))
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(root_dir, "prediction_isensee2017", 'loss_graph.png'))
    plt.show()
    plt.close()
    
if __name__ == "__main__":
    main()
