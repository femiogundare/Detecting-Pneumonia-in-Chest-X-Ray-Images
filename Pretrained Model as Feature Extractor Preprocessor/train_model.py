# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:07:53 2020

@author: femiogundare
"""

import pickle
import json
import numpy as np
import h5py

import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from config import config_paths as config


#open the HDF5 databases for the training, validation and test sets
print('Grabbing the extracted features from the HDF5 files...')
db_train = h5py.File(name=config.TRAIN_HDF5, mode='r')
db_val = h5py.File(name=config.VAL_HDF5, mode='r')
db_test = h5py.File(config.TEST_HDF5, 'r')

#initialize the features and labels for the training, validation and test sets
train_features, train_labels = db_train['features'], db_train['labels']
val_features, val_labels = db_val['features'], db_val['labels']
test_features, test_labels = db_test['features'], db_test['labels']

#check to see if a GPU is available or not
print('GPU is', 'Available' if tf.test.is_gpu_available() else 'Not Available')

#initialize a the data scaler
print('Scaling the features...')
scaler = StandardScaler()
train_features = scaler.fit_transform(np.array(train_features))
test_features = scaler.transform(np.array(test_features))


#----------------------------------LOGISTIC REGRESSION-------------------------------------
print('Logistic Regression...')
#hyperparameter space
params = {
    'C': [0.1, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0],
}

#instantiate the model class
model = LogisticRegression(class_weight='balanced', random_state=42)

#search for the best hyperparameters combination
print('Searching for the best hyperparameter...')
model = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=1, verbose=1)
model.fit(train_features, train_labels)

best_hyperparameters = model.best_params_
best_score = model.best_score_
print('Best hyperparameters: {}'.format(best_hyperparameters))
print('Best score: {}'.format(best_score))

print('Evaluating the model with the best hyperparameter on the test set...')
predictions = model.best_estimator_.predict(test_features)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)
class_report = classification_report(test_labels, predictions, target_names=np.array(db_train['label_names']))

print('Precision: {:.2f}%'.format(precision*100))
print('\nRecall: {:.2f}%'.format(recall*100))
print('\nF1: {:.2f}%'.format(f1*100))
print('\nConfusion matrix')
print(conf_matrix)
print('\nClassification report')
print(class_report)

#serialize the precision_score, recall_score, true negatives, false positives false negatives and true positives values to a json file
tn, fp, fn, tp = conf_matrix.ravel()
print('Serializing the results to json...')
dic = {
       'Precision score' : precision, 
       'Recall score' : recall, 
       'TP' : float(tp), 
       'FP' : float(fp), 
       'TN' : float(tn),
       'FN' : float(fn),
       'Best hyperparameter(s)' : best_hyperparameters,
       'Best score' : best_score
       }
f = open(config.LOGREGRESS_RESULTS_PATH, 'w')
f.write(json.dumps(dic))
f.close()

print('Saving the model...')
f = open(config.LOGREGRESS_MODEL_PATH, 'wb')
f.write(pickle.dumps(model.best_estimator_))
f.close()