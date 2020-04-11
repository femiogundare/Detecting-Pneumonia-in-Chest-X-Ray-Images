# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:32:31 2020

@author: femiogundare
"""

import os

#define the paths to the training, validation and test images directory
TRAIN_PATHS = '../datasets/chest_xray/chest_xray/train'
VAL_PATHS = '../datasets/chest_xray/chest_xray/val'
TEST_PATHS = '../datasets/chest_xray/chest_xray/test'

#define the paths to the training, validation and test data HDF5 files
TRAIN_HDF5 = '../datasets/hdf5/TRAIN.HDF5'
VAL_HDF5 = '../datasets/hdf5/VAL.HDF5'
TEST_HDF5 = '../datasets/hdf5/TEST.HDF5'

#define the output models path
LOGREGRESS_MODEL_PATH = 'output/logistic_regression/model.cpickle'

#define the path to the output plots
LOGREGRESS_PLOT_PATH = 'output/logistic_regression/model_plot.png'

#define the path to the results json
LOGREGRESS_RESULTS_PATH = 'output/logistic_regression/results.json'
