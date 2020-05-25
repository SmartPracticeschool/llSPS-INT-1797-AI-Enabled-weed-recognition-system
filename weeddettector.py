# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:19:26 2020

@author: sys
"""
from keras.preprocessing.image import ImageDataGenerator
train_data=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_data=ImageDataGenerator(rescale=1./255)
x_train=train_data.flow_from_directory(r'E:\datasets\trainset',target_size=(64,64),batch_size=32,class_mode='categorical')
x_test=test_data.flow_from_directory(r'E:\datasets\testset',target_size=(64,64),batch_size=32,class_mode='categorical')
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

weed=Sequential()
weed.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
weed.add(MaxPooling2D(pool_size=(2,2)))
weed.add(Flatten())
weed.add(Dense(output_dim=128,activation='relu',init='random_uniform'))
weed.add(Dense(output_dim=120,activation='relu',init='random_uniform'))
weed.add(Dense(output_dim=128,activation='relu',init='random_uniform'))
weed.add(Dense(output_dim=6,activation='sigmoid',init='random_uniform'))
weed.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(x_train.class_indices)
weed.fit_generator(x_train,samples_per_epoch = 15326,epochs=10,validation_data=x_test,nb_val_samples=337)
weed.save('model1.h5')

from keras.models import load_model
import numpy as np
import cv2
weed=load_model('model1.h5')
weed.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from skimage.transform import resize
def detect(frame):
    try:
        im=resize(frame,(64,64))
        im=np.expand_dims(im,axis=0)
        if(np.max(im)>1):
            im=im/255.0
        pred=weed.predict(im)
        print(pred)
        pred_cl=weed.predict_classes(im)
        print(pred_cl)
    except AttributeError:
        print('shape not found')
        
        
frame=cv2.imread('1147.tif')
data=detect(frame)