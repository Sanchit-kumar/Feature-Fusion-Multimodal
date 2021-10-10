import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate,MaxPool2D

# SIZE_X = 224 #352
# SIZE_Y = 224 # 352
SIZE_X = 352
SIZE_Y = 352
input_shape=(SIZE_X, SIZE_Y, 3)

class Model_t:
    def __init__(self):
        print("T.py called--Model created")
        self.Input_t=Input(shape=input_shape)
    def build_model(self, Input_t=None):
        # Input_r=Input(shape=input_shape)
        # Input_t=Input(shape=input_shape)
        if Input_t==None:
            Input_t=self.Input_t
        self.conv1_1=Conv2D(filters=64,kernel_size=(3,3),padding="same", name="conv1_1_t", activation="relu",use_bias=True)(Input_t)
        self.conv1_2=Conv2D(filters=64,kernel_size=(3,3),padding="same", name="conv1_2_t", activation="relu",use_bias=True)(self.conv1_1)
        self.pool1=MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool_1_t")(self.conv1_2)
        self.conv2_1=Conv2D(filters=128, kernel_size=(3,3), padding="same",name="conv2_1_t", activation="relu",use_bias=True)(self.pool1)
        self.conv2_2=Conv2D(filters=128, kernel_size=(3,3), padding="same",name="conv2_2_t", activation="relu",use_bias=True)(self.conv2_1)
        self.pool2=MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool_2_t")(self.conv2_2)
        self.conv3_1=Conv2D(filters=256, kernel_size=(3,3), padding="same",name="conv3_1_t", activation="relu",use_bias=True)(self.pool2)
        self.conv3_2=Conv2D(filters=256, kernel_size=(3,3), padding="same",name="conv3_2_t", activation="relu",use_bias=True)(self.conv3_1)
        self.conv3_3=Conv2D(filters=256, kernel_size=(3,3), padding="same",name="conv3_3_t", activation="relu",use_bias=True)(self.conv3_2)
        self.pool3=MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool_3_t")(self.conv3_3)
        self.conv4_1=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv4_1_t", activation="relu",use_bias=True)(self.pool3)
        self.conv4_2=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv4_2_t", activation="relu",use_bias=True)(self.conv4_1)
        self.conv4_3=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv4_3_t", activation="relu",use_bias=True)(self.conv4_2)
        self.pool4=MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool_4_t")(self.conv4_3)
        self.conv5_1=Conv2D(filters=512, kernel_size=(3,3), padding="same", name="conv5_1_t", activation="relu",use_bias=True)(self.pool4)
        self.conv5_2=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv5_2_t", activation="relu",use_bias=True)(self.conv5_1)
        self.conv5_3=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv5_3_t", activation="relu",use_bias=True)(self.conv5_2)
        self.pool5=MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool_5_t")(self.conv5_3)

        # tconv2_3=Conv2D(128,kernel_size=(3,3), dilation_rate=2, padding="same")(conv2_2)
        # tpool2_4=MaxPool2D(pool_size=(2,2),strides=(2,2))(tconv2_3)

        tconv2_3=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=1, name="tconv2_3", padding="same", activation="relu",use_bias=True)(self.conv2_2)

        tpool2_4=MaxPool2D(pool_size=(3,3),strides=(1,1),padding="same",name="tpool_2_4")(tconv2_3)
        tconv2_4=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=3,name="tconv2_4", padding="same", activation="relu",use_bias=True)(tpool2_4)

        tpool2_5=MaxPool2D(pool_size=(5,5),strides=(1,1),padding="same",name="tpool_2_5")(tconv2_4)
        tconv2_5=Conv2D(filters=64, kernel_size=(3,3), dilation_rate=5,name="tconv2_5", padding="same", activation="relu",use_bias=True)(tpool2_5)

        tpool2_6=MaxPool2D(pool_size=(7,7),strides=(1,1),padding="same",name="tpool_2_6")(tconv2_5)
        tconv2_6=Conv2D(filters=64, kernel_size=(3,3), dilation_rate=7, name="tconv2_6", padding="same", activation="relu",use_bias=True)(tpool2_6)

        t_concat1=concatenate([tconv2_3,tconv2_4,tconv2_5, tconv2_6], axis=3)


        ###########
        self.tconv2_5_=Conv2D(filters=128,kernel_size=(3,3), padding="same",name="tconv2_5_", activation="relu",use_bias=True)(t_concat1)

        tconv3_4=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=1, padding="same",name="tconv3_4", activation="relu",use_bias=True)(self.conv3_3)

        tpool3_5=MaxPool2D(pool_size=(3,3),strides=(1,1),padding="same",name="tpool_3_5")(tconv3_4)
        tconv3_5=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=3, padding="same",name="tconv3_5", activation="relu",use_bias=True)(tpool3_5)

        tpool3_6=MaxPool2D(pool_size=(5,5),strides=(1,1),padding="same",name="tpool_3_6")(tconv3_5)
        tconv3_6=Conv2D(filters=64, kernel_size=(3,3), dilation_rate=5, padding="same",name="tconv3_6", activation="relu",use_bias=True)(tpool3_6)

        tpool3_7=MaxPool2D(pool_size=(7,7),strides=(1,1),padding="same",name="tpool_3_7")(tconv3_6)
        tconv3_7=Conv2D(filters=64, kernel_size=(3,3), dilation_rate=7, padding="same",name="tconv3_7", activation="relu",use_bias=True)(tpool3_7)

        t_concat2=concatenate([tconv3_4, tconv3_5, tconv3_6, tconv3_7])

        ###########
        self.tconv3_6_=Conv2D(filters=128,kernel_size=(3,3), padding="same", name="tconv3_6_", activation="relu",use_bias=False)(t_concat2)

        # tconv2_3=Conv2D(128,kernel_size=(3,3), dilation_rate=2, padding="same")(conv2_2)
        # tpool2_4=MaxPool2D(pool_size=(2,2),strides=(2,2))(tconv2_3)

        # self.conv5_input_shape=(3,3,512,256)

        #####

        self.conv5_3_t=Conv2D(filters=128, kernel_size=(3,3), padding="same", name="conv5_3_t_t", activation="relu",use_bias=False)(self.pool5)


        conv6_2=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=2, padding="same", name="conv6_2_t", activation="relu",use_bias=False)(self.conv5_3_t)

        pool6_3=MaxPool2D(pool_size=(5,5),strides=(1,1),padding="same",name="pool_6_3_t")(conv6_2)
        conv6_3=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=5, padding="same", name="conv6_3_t", activation="relu",use_bias=False)(pool6_3)
        conv6_3_=Conv2D(filters=128, kernel_size=(3,3), padding="same", name="conv6_3__t", activation="relu",use_bias=False)(concatenate([self.conv5_3_t, conv6_3],axis=3))



        pool6_4=MaxPool2D(pool_size=(9,9),strides=(1,1),padding="same",name="pool_6_4_t")(conv6_3_)
        conv6_4=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=9, padding="same", name="conv6_4_t", activation="relu",use_bias=False)(pool6_4)

        pool6_5=MaxPool2D(pool_size=(15,15),strides=(1,1),padding="same",name="pool_6_5_t")(conv6_4)
        conv6_5=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=15, padding="same", name="conv6_5_t", activation="relu",use_bias=False)(pool6_5)

        t_concat3=concatenate([self.conv5_3_t, conv6_2, conv6_3, conv6_4, conv6_5], axis=3)

        ###########
        self.conv6_6=Conv2D(filters=128,kernel_size=(3,3), padding="same", name="conv6_6_t", activation="relu",use_bias=False)(t_concat3)

