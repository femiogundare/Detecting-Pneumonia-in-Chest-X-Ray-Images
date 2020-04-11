# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:33:10 2020

@author: femiogundare
"""

import cv2


#build the class
class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        (h, w) = image.shape[:2]
        dW, dH = 0, 0
        
        if w < h:
            image = imutils.resize(image, width=self.width)
            dH = int((image.shape[0] - self.height) / 2.0)
            
        else:
            image = imutils.resize(image, height=self.height)
            dW = int((image.shape[1] - self.width) / 2.0)
            
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]
        
        return cv2.resize(image, (self.height, self.width), interpolation=self.inter)