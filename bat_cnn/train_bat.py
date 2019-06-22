import os
import sys
import numpy as np

sys.path.append('../')
from models.data import write_data_to_file, open_data_file, get_data_from_file
from models.data import get_discriminator_input_and_label
from models.data_prepare_v1 import get_training_and_validation_data
from models.data_split import GetListFromFiles, set_train_validation_test
from models.data_prepare_v1 import get_training_and_validation_generators
from models.model import combineNet_3d, create_discriminator, get_GAN
from models.training import load_old_model, train_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model

from models.GPU_config import gpuConfig
os.environ["CUDA_VISIBLE_DEVICES"]= gpuConfig['GPU_using']

config = dict()
config["image_shape"] = (288, 288, 96)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = (64, 64, 64)  # switch to None to train on the whole image
config["labels"] = (1,)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["FF", "T2S", "F", "W", "pred_z", "pred_x", "pred_y"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]

config["n_base_filters"] = 32


config["n_epochs"] = 500  # cutoff the training after this many epochs
config["initial_learning_rate"] = 1e-3
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["learning_rate_patience"] = 15  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 40  # training will be stopped after this many epochs without the validation loss improving

config["batch_size"] = 6
config["validation_batch_size"] = 12
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 6  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("../resultData"+"/BAT_data.h5")
config["model_file"] = os.path.abspath("../resultData" +"/BAT_Combined_model.h5")
config["training_file"] = os.path.abspath("../resultData" +  "/BAT_Combined_training_ids.pkl")
config["validation_file"] = os.path.abspath("../resultData" +"/BAT_Combined_validation_ids.pkl")
config["test_file"] = os.path.abspath("../resultData" +"/BAT_Combined_test_ids.pkl")
config["trainingLog"] = os.path.abspath("../resultData"+"/BAT_Combined.log")
config["overwrite"] = True  # If True, will previous files. If False, will use previously written files.
config["GAN_model"] =  os.path.abspath("../resultData"+"/BAT_Combined_GAN_model.h5")

def fetch_training_data_files(folderNameDict):
    '''return list
    [(modality1, modality2, modality3, modality4, label)
    (modality1, modality2, modality3, modality4, label)]
    '''
    training_data_files = list()
    keys = list(folderNameDict.keys())    
    NN = 0
    for i in range(len(keys)):
        if i == 0:
            NN = len(folderNameDict[keys[i]])
        else:
            if NN != len(folderNameDict[keys[i]]):
                raise ValueError('check the len of the list with key: ' + keys[i])  
    for i in range(NN):
        subject_files = list()
        for modality in config["training_modalities"] + ["Label"]:
            subject_files.append(folderNameDict[modality][i])
        training_data_files.append(tuple(subject_files))
    return training_data_files

def main(input_data_root, overwrite=False):

    #-------------------------------------------------------#
    # convert input images into an hdf5 file
    outputfolder = os.path.dirname(config["data_file"])
    if overwrite or not os.path.exists(config["data_file"]):        
        train_path = {}
        train_path['FF'] = os.path.join(input_data_root, 'TrainingData/FF/Filelist.txt')
        train_path['T2S'] = os.path.join(input_data_root, 'TrainingData/T2S/Filelist.txt')
        train_path['F'] = os.path.join(input_data_root, 'TrainingData/F/Filelist.txt')
        train_path['W'] = os.path.join(input_data_root, 'TrainingData/W/Filelist.txt')
        train_path['pred_z'] = os.path.join(input_data_root, 'TrainingData/pred_z/Filelist.txt')
        train_path['pred_x'] = os.path.join(input_data_root, 'TrainingData/pred_x/Filelist.txt')
        train_path['pred_y'] = os.path.join(input_data_root, 'TrainingData/pred_y/Filelist.txt')
        train_path['Label'] = os.path.join(input_data_root, 'TrainingData/Label/Filelist.txt')
        
        test_path = {}
        test_path['FF'] = os.path.join(input_data_root, 'TestData/FF/Filelist.txt')
        test_path['T2S'] = os.path.join(input_data_root, 'TestData/T2S/Filelist.txt')
        test_path['F'] = os.path.join(input_data_root, 'TestData/F/Filelist.txt')
        test_path['W'] = os.path.join(input_data_root, 'TestData/W/Filelist.txt')
        test_path['pred_z'] = os.path.join(input_data_root, 'TestData/pred_z/Filelist.txt')
        test_path['pred_x'] = os.path.join(input_data_root, 'TestData/pred_x/Filelist.txt')
        test_path['pred_y'] = os.path.join(input_data_root, 'TestData/pred_y/Filelist.txt')
        test_path['Label'] = os.path.join(input_data_root, 'TestData/Label/Filelist.txt')       

        dataSplit = GetListFromFiles()
        folderName, trainNum, testNum = dataSplit.readImage(train_path, test_path)           
        
        training_files = fetch_training_data_files(folderName)
        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"])
        data_file_opened = open_data_file(config["data_file"])
        #-------------------------------------------------------#

        #-------------------------------------------------------#
        # split the training data to training, validation, and test
        trainList = list(range(trainNum))
        testList = list(range(trainNum, trainNum+testNum))
        training_list, validation_list = set_train_validation_test(trainList, testList, t_v_split = config["validation_split"], 
                                training_file = config["training_file"], validation_file = config["validation_file"], 
                                test_file = config["test_file"], overwrite = overwrite)
        
        train_patch_data_dir, val_patch_data_dir = get_training_and_validation_data(data_file_opened, 
                                                                        outputfolder, 
                                                                        n_labels=config["n_labels"], 
                                                                        labels=config["labels"], 
                                                                        training_list = training_list,  
                                                                        validation_list = validation_list, 
                                                                        patch_shape = config["patch_shape"],
                                                                        training_patch_overlap = 0, 
                                                                        validation_patch_overlap = config["validation_patch_overlap"], 
                                                                        training_patch_start_offset = config["training_patch_start_offset"], 
                                                                        skip_blank = config["skip_blank"])                        
        data_file_opened.close()
    
    train_patch_data_dir = os.path.join(outputfolder,'train_patch_data_save.h5')
    val_patch_data_dir = os.path.join(outputfolder,'val_patch_data_save.h5')
    train_patch_data_file = open_data_file(train_patch_data_dir)
    val_patch_data_file = open_data_file(val_patch_data_dir)
    
    train_patch_data = get_data_from_file(train_patch_data_dir, 'data')
    val_patch_data = get_data_from_file(val_patch_data_dir, 'data')
    train_patch_label = get_data_from_file(train_patch_data_dir, 'truth')
    val_patch_label = get_data_from_file(val_patch_data_dir, 'truth')  
    combineNet_input = np.concatenate((train_patch_data, val_patch_data), axis=0)
    combineNet_label = np.concatenate((train_patch_label, val_patch_label), axis=0)
    #-------------------------------------------------------#
    # set the model
    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = combineNet_3d(input_shape=config["input_shape"], 
                                n_labels=config["n_labels"], 
                                depth=5, 
                                n_base_filters=config["n_base_filters"],
                                normMethod = 'batch_norm',
                                initial_learning_rate = config["initial_learning_rate"])                                                                                                     
    
    #-------------------------------------------------------#
    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps \
    = get_training_and_validation_generators(train_patch_data_file, 
                                            val_patch_data_file, 
                                            batch_size = config["batch_size"],
                                            n_labels=config["n_labels"], 
                                            labels=config["labels"], 
                                            validation_batch_size = config["validation_batch_size"], 
                                            patch_shape = config["patch_shape"],
                                            augment = config["augment"], 
                                            augment_flip=config["flip"], 
                                            augment_distortion_factor=config["distort"],  
                                            permute=config["permute"])    
    #-------------------------------------------------------#
    # run training combineNet
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                n_epochs=config["n_epochs"],
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["learning_rate_patience"],
                early_stopping_patience=config["early_stop"],
                workers = 0, # when workers is not 0, it will cause the problem.
                use_multiprocessing=False, 
                logging_file = config["trainingLog"])    
    #-------------------------------------------------------#
    # predict final output and prepare all models
    combineNet_output = model.predict(combineNet_input, verbose=1)
    D_input, D_label = get_discriminator_input_and_label(combineNet_output, combineNet_label)
    discriminator = create_discriminator(tuple([1] + list(config["patch_shape"])), learningRate = 1e-3)
    weightDir_d = os.path.join(outputfolder, 'BAT_discriminator.h5')
    if (not config["overwrite"]) and (os.path.exists(weightDir_d)):
        discriminator.load_weights(weightDir_d)
    GAN = get_GAN(model, discriminator, config["input_shape"], learningRate=1e-3)
    weightDir_G = os.path.join(outputfolder, 'BAT_GAN.h5')
    if (not config["overwrite"]) and (os.path.exists(weightDir_G)):
        GAN.load_weights(weightDir_G)
    GAN_label = np.ones(len(D_label))
    train_patch_data_file.close()
    val_patch_data_file.close()

    #-------------------------------------------------------#
    # training GAN part
    for i in range(2):
        #-------------------------------------------------------#
        # train discriminator    
        model_checkpoint = ModelCheckpoint(weightDir_d, monitor='val_loss', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
        discriminator.fit(D_input, D_label, batch_size=config['batch_size'], epochs=config["n_epochs"], verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint, early_stop])
        #-------------------------------------------------------#
        # train GAN    
        model_checkpoint = ModelCheckpoint(weightDir_G, monitor='val_loss', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
        GAN.fit(combineNet_input, {'gnet':combineNet_label, 'dnet': GAN_label}, batch_size=5, epochs=100, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint, early_stop])
        #GAN_combineNet = Model(inputs = GAN.get_layer('gnet').get_input_at(0), outputs=GAN.get_layer('gnet').get_layer('goutput').output)
        GAN_output = model.predict(combineNet_input, verbose=1)
        D_input, D_label = get_discriminator_input_and_label(GAN_output, combineNet_label)
        
    model.save('../resultData/BAT_Combined_GAN_model.h5')                
    
if __name__ == "__main__":
    input_data_root = '/home/louis/project/BAT_new_3DCNN/'
    if not os.path.exists(os.path.abspath("../resultData")):
        os.mkdir(os.path.abspath("../resultData"))    
    main(input_data_root, overwrite=config["overwrite"])
