# -*- coding: utf-8 -*-
"""
Created on sun May 24 09:26:26 2020

@author: sys
"""

from keras.preprocessing.image import ImageDataGenerator
train_data=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_data=ImageDataGenerator(rescale=1./255)
x_train=train_data.flow_from_directory(r'E:\datasets\trainset',target_size=(64,64),batch_size=32,class_mode='categorical')
x_test=test_data.flow_from_directory(r'E:\datasets\testset',target_size=(64,64),batch_size=32,class_mode='categorical')