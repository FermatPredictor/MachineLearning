# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist #手寫數字資料集
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD #stochastic gradient descent

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
資料前處理:
CNN專做圖形辨識的，不用將圖片拉成一維，
但input需要告訴CNN矩陣數量，
譬如一般圖片有RGB就有3個矩陣
"""
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
y_train = np_utils.to_categorical(y_train,10) #one-hot encoding
y_test = np_utils.to_categorical(y_test,10) #one-hot encoding


"""
範例: 做3次convolution, 每次都接max pooling
filter 大小都是3*3，max pooling大小都是2*2

做完CNN後會把output拉平，
送入一個正常的神經網路 NN

CNN的技巧是filter的數量會愈來愈多
"""
model = Sequential() #建立空的神經網路

"""
padding='same' 意思是做完convolution後矩陣不要縮水
"""
model.add(Conv2D(32, (3,3), padding='same', input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

"""
組裝神經網路
loss: 損失函數
optimizers: 訓練方式(lr是learning rate)
metrics: 評分標準
"""
model.compile(loss='mse', optimizer=SGD(lr=0.05), metrics=['accuracy'])
model.summary() #檢示神經網路的架構

"""
訓練神經網路
batch_size: 一次訓練幾筆資料(每幾筆資料調一次參數)
epochs: 6萬筆資料共訓練幾次
"""
model.fit(x_train, y_train, batch_size=100, epochs=20)

# 預測
predict = model.predict_classes(x_test)

# 將訓練好的模型存為HDF5格式
model.save('./handwrite_model.h5')

# 讀取模型與參數使用
model = tf.keras.models.load_model('./handwrite_model.h5')

# 重新預測(看載入的模型跟原本是不是一樣)
predict_2 = model.predict_classes(x_test)
