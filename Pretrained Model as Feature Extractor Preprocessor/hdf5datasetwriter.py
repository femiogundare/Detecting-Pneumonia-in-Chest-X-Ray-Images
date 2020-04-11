# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:24:13 2020

@author: femiogundare
"""

#import the required packages
import os
import h5py


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey='images', buffSize=500):
        
        if os.path.exists(outputPath):
            raise ValueError(
                    'The output path supplied already exists and cannot be overwritten;\
                    manually delete the file so that the program will run', outputPath
                    )
        
        self.db = h5py.File(outputPath, 'w')
        self.data = self.db.create_dataset(dataKey, shape=dims, dtype='float')
        self.labels = self.db.create_dataset('labels', shape=(dims[0], ), dtype=int)
        
        self.buffSize = buffSize
        self.buffer = {'data' : [], 'labels' : []}
        self.idx = 0
        
    def add(self, rows, labels):
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)
        
        if len(self.buffer['data']) >= self.buffSize:
            self.flush()
            
    def flush(self):
        #write the buffers to disk, and then reset
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i
        self.buffer = {'data' : [], 'labels' : []}
        
    def storeClassLabels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset(
                'label_names', shape=(len(classLabels), ), dtype=dt
                )

    def close(self):
        if len(self.buffer['data']) > 0:
            self.flush()
        
        #close the dataset
        self.db.close()        