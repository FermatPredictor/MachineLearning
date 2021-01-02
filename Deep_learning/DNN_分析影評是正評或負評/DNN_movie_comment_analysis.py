# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb #影評資料集
from keras.preprocessing import sequence
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Activation, Flatten
from keras.layers import LSTM # Long-Short-term memory layer，有長期記憶的RNN模型
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD #stochastic gradient descent

import tensorflow as tf

# num_words表示取最常出現頻率前幾個字
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print(x_train[1])

"""
資料前處理:
DNN是有記憶的神經網路。
pad_sequences做「截長補短」，取影評的前一百個字，過短的字補0
"""
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)


model = Sequential() #建立空的神經網路

""" 
Embedding: 原本每個單字用一個1~10000的數字來表示，
(以one-hot encoding的角度來看就是一萬維向量)
Embedding 技術可以將它降維，並且語意相近的單字向量也會接近。
Embedding 可以將離散的數據變的連續
"""
model.add(Embedding(10000,128))
model.add(LSTM(150))
model.add(Dense(1))
model.add(Activation('sigmoid')) #因為輸出為0~1的數字，1:正評，0:負評


"""
組裝神經網路
loss: 損失函數
optimizers: 訓練方式
metrics: 評分標準
"""
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary() #檢示神經網路的架構

"""
訓練神經網路
batch_size: 一次訓練幾筆資料(每幾筆資料調一次參數)
epochs: 訓練幾次
"""
model.fit(x_train, y_train, batch_size=100, epochs=5)

# 預測
score = model.evaluate(x_test, y_test)
print('測試資料的loss', score[0])
print('測試資料的正確率', score[1])

# 將訓練好的模型存為HDF5格式
model.save('./idmb_model.h5')

# 讀取模型與參數使用
model = tf.keras.models.load_model('./idmb_model.h5')

# 重新預測(看載入的模型跟原本是不是一樣)
score2 = model.evaluate(x_test, y_test)
