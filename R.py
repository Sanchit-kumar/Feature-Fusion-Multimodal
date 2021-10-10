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

# SIZE_X = 224#352
# SIZE_Y = 224 #352
SIZE_X = 352
SIZE_Y = 352
input_shape=(SIZE_X, SIZE_Y, 3)

class Model_r:
    def __init__(self):
        print("R.py called--Model created")
        self.Input_r=Input(shape=input_shape)
    def build_model(self, Input_r=None):
        # Input_r=Input(shape=input_shape)
        # Input_r=Input(shape=input_shape)
        if Input_r==None:
            Input_r=self.Input_r
        self.conv1_1=Conv2D(filters=64,kernel_size=(3,3),padding="same", name="conv1_1_r", activation="relu",use_bias=True)(Input_r)
        self.conv1_2=Conv2D(filters=64,kernel_size=(3,3),padding="same",name="conv1_2_r", activation="relu",use_bias=True)(self.conv1_1)
        self.pool1=MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool_1_r")(self.conv1_2)
        self.conv2_1=Conv2D(filters=128, kernel_size=(3,3), padding="same", name="conv2_1_r", activation="relu",use_bias=True)(self.pool1)
        self.conv2_2=Conv2D(filters=128, kernel_size=(3,3), padding="same",name="conv2_2_r", activation="relu",use_bias=True)(self.conv2_1)
        self.pool2=MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool_2_r")(self.conv2_2)
        self.conv3_1=Conv2D(filters=256, kernel_size=(3,3), padding="same",name="conv3_1_r", activation="relu",use_bias=True)(self.pool2)
        self.conv3_2=Conv2D(filters=256, kernel_size=(3,3), padding="same", name="conv3_2_r",activation="relu",use_bias=True)(self.conv3_1)
        self.conv3_3=Conv2D(filters=256, kernel_size=(3,3), padding="same",name="conv3_3_r", activation="relu",use_bias=True)(self.conv3_2)
        self.pool3=MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool_3_r")(self.conv3_3)
        self.conv4_1=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv4_1_r", activation="relu",use_bias=True)(self.pool3)
        self.conv4_2=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv4_2_r", activation="relu",use_bias=True)(self.conv4_1)
        self.conv4_3=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv4_3_r", activation="relu",use_bias=True)(self.conv4_2)
        self.pool4=MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool_4_r")(self.conv4_3)
        self.conv5_1=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv5_1_r", activation="relu",use_bias=True)(self.pool4)
        self.conv5_2=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv5_2_r", activation="relu",use_bias=True)(self.conv5_1)
        self.conv5_3=Conv2D(filters=512, kernel_size=(3,3), padding="same",name="conv5_3_r", activation="relu",use_bias=True)(self.conv5_2)
        self.pool5=MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool_5_r")(self.conv5_3)

        # rconv2_3=Conv2D(128,kernel_size=(3,3), dilation_rate=2, padding="same")(conv2_2)
        # rpool2_4=MaxPool2D(pool_size=(2,2),strides=(2,2))(rconv2_3)

        rconv2_3=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=1, padding="same",name="rconv2_3", activation="relu",use_bias=False)(self.conv2_2)

        rpool2_4=MaxPool2D(pool_size=(3,3),strides=(1,1),padding="same",name="rpool_2_4")(rconv2_3)
        rconv2_4=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=3, padding="same",name="rconv2_4", activation="relu",use_bias=False)(rpool2_4)

        rpool2_5=MaxPool2D(pool_size=(5,5),strides=(1,1),padding="same",name="rpool_2_5")(rconv2_4)
        rconv2_5=Conv2D(filters=64, kernel_size=(3,3), dilation_rate=5, padding="same",name="rconv2_5", activation="relu",use_bias=False)(rpool2_5)

        rpool2_6=MaxPool2D(pool_size=(7,7),strides=(1,1),padding="same",name="rpool_2_6")(rconv2_5)
        rconv2_6=Conv2D(filters=64, kernel_size=(3,3), dilation_rate=7, padding="same",name="rconv2_6", activation="relu",use_bias=False)(rpool2_6)

        r_concat1=concatenate([rconv2_3,rconv2_4,rconv2_5, rconv2_6], axis=3)


        ###########
        self.rconv2_5_=Conv2D(filters=128,kernel_size=(3,3), padding="same",name="rconv2_5_", activation="relu",use_bias=False)(r_concat1)

        rconv3_4=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=1, padding="same",name="rconv3_4", activation="relu",use_bias=False)(self.conv3_3)

        rpool3_5=MaxPool2D(pool_size=(3,3),strides=(1,1),padding="same",name="rpool_3_5")(rconv3_4)
        rconv3_5=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=3, padding="same",name="rconv3_5", activation="relu",use_bias=False)(rpool3_5)

        rpool3_6=MaxPool2D(pool_size=(5,5),strides=(1,1),padding="same",name="rpool_3_6")(rconv3_5)
        rconv3_6=Conv2D(filters=64, kernel_size=(3,3), dilation_rate=5, padding="same",name="rconv3_6", activation="relu",use_bias=False)(rpool3_6)

        rpool3_7=MaxPool2D(pool_size=(7,7),strides=(1,1),padding="same",name="rpool_3_7")(rconv3_6)
        rconv3_7=Conv2D(filters=64, kernel_size=(3,3), dilation_rate=7, padding="same",name="rconv3_7", activation="relu",use_bias=False)(rpool3_7)

        r_concat2=concatenate([rconv3_4, rconv3_5, rconv3_6, rconv3_7])

        ###########
        self.rconv3_6_=Conv2D(filters=128,kernel_size=(3,3), padding="same",name="rconv3_6__r", activation="relu",use_bias=False)(r_concat2)

        # rconv2_3=Conv2D(128,kernel_size=(3,3), dilation_rate=2, padding="same")(conv2_2)
        # rpool2_4=MaxPool2D(pool_size=(2,2),strides=(2,2))(rconv2_3)

        # self.conv5_input_shape=(3,3,512,256)

        #####

        self.conv5_3_t=Conv2D(filters=128, kernel_size=(3,3), padding="same", name="conv5_3_t_r", activation="relu",use_bias=True)(self.pool5)


        conv6_2=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=2, padding="same", name="conv6_2_r",activation="relu",use_bias=True)(self.conv5_3_t)

        pool6_3=MaxPool2D(pool_size=(5,5),strides=(1,1),padding="same", name="pool_6_3_r")(conv6_2)
        conv6_3=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=5, padding="same",name="conv6_3_r", activation="relu",use_bias=True)(pool6_3)
        conv6_3_=Conv2D(filters=128, kernel_size=(3,3), padding="same",name="conv6_3__r", activation="relu",use_bias=True)(concatenate([self.conv5_3_t, conv6_3],axis=3))



        pool6_4=MaxPool2D(pool_size=(9,9),strides=(1,1),padding="same", name="pool_6_4_r")(conv6_3_)
        conv6_4=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=9, padding="same",name="conv6_4_r", activation="relu",use_bias=True)(pool6_4)

        pool6_5=MaxPool2D(pool_size=(15,15),strides=(1,1),padding="same", name="pool_6_5_r")(conv6_4)
        conv6_5=Conv2D(filters=128, kernel_size=(3,3), dilation_rate=15, padding="same",name="conv6_5_r", activation="relu",use_bias=True)(pool6_5)

        r_concat3=concatenate([self.conv5_3_t, conv6_2, conv6_3, conv6_4, conv6_5], axis=3)

        ###########
        self.conv6_6=Conv2D(filters=128,kernel_size=(3,3), padding="same",name="conv6_6_r", activation="relu",use_bias=True)(r_concat3)

