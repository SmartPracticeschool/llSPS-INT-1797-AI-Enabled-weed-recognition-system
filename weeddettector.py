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
weed.save('weedmodel.h5')


