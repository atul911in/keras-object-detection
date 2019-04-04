#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2017 - Limber Cheng <cheng@limberence.com> 
# @Time : 23/03/2018 17:19
# @Author : Limber Cheng
# @File : tiny-yolo-tl
# @Software: PyCharm
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D,Flatten,Dense
from keras.models import Sequential
from keras.utils import plot_model

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(448, 448, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(strides=(2, 2), padding="same"))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(strides=(2, 2), padding="same"))
model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4096))
# model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1470))
