import math
from functools import partial

import os
from models.GPU_config import gpuConfig
os.environ["CUDA_VISIBLE_DEVICES"]= gpuConfig['GPU_using']

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
import numpy as np

from models.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient)

K.set_image_dim_ordering('th')


# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks


def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss,
                      'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef,
                      'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    return load_model(model_file, custom_objects=custom_objects)


def train_model(model, model_file, training_generator, validation_generator, 
                steps_per_epoch, 
                validation_steps,
                n_epochs=500,
                initial_learning_rate=0.001, 
                learning_rate_drop=0.5, 
                learning_rate_epochs=None,                
                learning_rate_patience=20, 
                early_stopping_patience=None,
                workers = 0, # when workers is not 0, it will cause the problem.
                use_multiprocessing=False, 
                logging_file="training.log"):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """
    # call validation generator by itsself
    # cnt = 0
    # v_data_x = None
    # v_data_y = None
    
    # for i in range(validation_steps):
    #     data = validation_generator.next()
    #     v_data_x = data[0] if cnt == 0 else np.concatenate([v_data_x, data[0]], axis=0)
    #     v_data_y = data[1] if cnt == 0 else np.concatenate([v_data_y, data[1]], axis=0)
    #     cnt += 1

    # v_data = (v_data_x, v_data_y)
    

    model.fit_generator(generator = training_generator,
                        steps_per_epoch = steps_per_epoch,
                        epochs = n_epochs,
                        validation_data = validation_generator,
                        validation_steps = validation_steps,
                        workers = 0, # when workers is not 0, it will cause the problem.
                        use_multiprocessing=False,
                        callbacks = get_callbacks(model_file,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience,
                                                logging_file = logging_file)
                        )
                        # A callback is a set of functions to be applied at given stages of the training procedure. 
                        # You can use callbacks to get a view on internal states and statistics of the model during training. 
                        # You can pass a list of callbacks (as the keyword argument callbacks) to the .fit() 
                        # method of the Sequential or Model classes. The relevant methods of the callbacks will 
                        # then be called at each stage of the training.
