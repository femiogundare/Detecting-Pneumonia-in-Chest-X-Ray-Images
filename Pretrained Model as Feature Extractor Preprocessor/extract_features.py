# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:01:43 2020

@author: femiogundare
"""

#import the required packages
import os
import cv2
import random
import progressbar
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import InceptionV3
from utilities.preprocessing.aspectratioresize import AspectAwarePreprocessor
from featureXtractor import FeatureExtractor
from config import config_paths as config


print('Loading the images...')
trainPaths = list(paths.list_images(config.TRAIN_PATHS))
valPaths = list(paths.list_images(config.VAL_PATHS))
testPaths = list(paths.list_images(config.TEST_PATHS))

#shuffle the training images paths to allow for random distribution of the classes accross the paths
random.shuffle(trainPaths) 

#extract the labels from the training, validation and test images paths
print('Getting the labels out of the images paths...')
trainLabels = [c.split(os.path.sep)[-2] for c in trainPaths]
valLabels = [k.split(os.path.sep)[-2] for k in valPaths]
testLabels = [x.split(os.path.sep)[-2] for x in testPaths]

#initialize the label encoder
print('Encoding the labels...')
le = LabelEncoder()
#encode the training labels
trainLabels = le.fit_transform(trainLabels)
#encode the validation and test data labels
valLabels = le.transform(valLabels)
testLabels = le.transform(testLabels)

#initialize the preprocessor(s)
aap = AspectAwarePreprocessor(width=150, height=150)

#load the Inception V3 network
print('Loading Inception V3...')
model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
print(model.summary())


#extract features from the training, validation and testing sets
print('Extracting features from the training images...')
train_features = FeatureExtractor(
        model=model, dType='train', paths=trainPaths, labels=trainLabels, 
        outputPath=config.TRAIN_HDF5, batch_size=32, class_names=le.classes_, preprocessors=aap
        ).extract_features()

print('Extracting features from the validation images...')
val_features = FeatureExtractor(
        model=model, dType='val', paths=valPaths, labels=valLabels, 
        outputPath=config.VAL_HDF5, batch_size=4, class_names=le.classes_, preprocessors=aap
        ).extract_features()

print('Extracting features from the test images...')
test_features = FeatureExtractor(
        model=model, dType='test', paths=testPaths, labels=testLabels, 
        outputPath=config.TEST_HDF5, batch_size=32, class_names=le.classes_, preprocessors=aap
        ).extract_features()

print('done extracting features...!')