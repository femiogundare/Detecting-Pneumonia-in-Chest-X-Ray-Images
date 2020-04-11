# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:59:08 2020

@author: femiogundare
"""

#import the required packages
import cv2
import h5py
import numpy as np
import progressbar
from hdf5datasetwriter import HDF5DatasetWriter


class FeatureExtractor:
    def __init__(self, model, dType, paths, labels, outputPath, class_names, buffSize=500, batch_size=32, preprocessors=None):
        
        self.model = model
        self.dType = dType
        self.paths = paths
        self.labels = labels
        self.outputPath = outputPath
        self.buffSize = buffSize
        self.batch_size = batch_size
        self.class_names = class_names
        self.preprocessors = None
        
    def extract_features(self):
        #initialize the hdf5 dataset writer
        datasetWriter = HDF5DatasetWriter(
                    dims=(len(self.paths), 3*3*2048), outputPath=self.outputPath,
                    dataKey='features', buffSize=self.buffSize
                    )
        datasetWriter.storeClassLabels(self.class_names)
            
        #initialize the progressbar
        widgets = [
                "Extracting Features: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", 
                progressbar.ETA()
                ]
        pbar = progressbar.ProgressBar(maxval=len(self.paths),widgets=widgets).start()
            
        #loop over the training images in batches, preprocess them and store them in hdf5
        for i in np.arange(0, len(self.paths), self.batch_size):
            batchPaths = self.paths[i:i + self.batch_size]
            batchLabels = self.labels[i:i + self.batch_size]
            batchImages = []
                
            if self.preprocessors is None:
                #loop over each of the images in the current batch and convert to size 150x150
                for img in batchPaths:
                    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)  #reads the image as grayscale
                    img = cv2.resize(img, (150, 150))
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  #converts the image to RGB
                    img = img.astype('float32') / 255.0          #normalizes the image to scale [0,1]
                    batchImages.append(img)
                    
            elif self.preprocessors is not None:
                #loop over each of the images in the current batch
                for img in batchPaths:
                    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                        
                    for p in self.preprocessors:
                        img = p.preprocess(img)
                    
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img = img.astype('float32') / 255.0
                        
                    batchImages.append(img)
                    
            #allow the images to assume shape of (batch_size, 150, 150, 3)
            batchImages = np.array(batchImages)
            #make the model perform prediction
            features = self.model.predict(batchImages, batch_size=self.batch_size) #shape is (batch_size, 3, 3, 2048)     
            #flatten the image vectors
            features = features.reshape((features.shape[0], 3*3*2048)) #shape becomes (batch_size, 3*3*2048)

            #add the features and labels to the hdf5 dataset
            datasetWriter.add(features, batchLabels)
            pbar.update(i) 
                    
        datasetWriter.close()
        pbar.finish()