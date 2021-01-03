# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


from keras.datasets import mnist #手寫數字資料集
from keras.utils import np_utils
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,784)
x_test_flat = x_test.reshape(10000,784)
y_test_one_hot = np_utils.to_categorical(y_test,10) #one-hot encoding

# 讀取模型與參數使用
model = tf.keras.models.load_model('./handwrite_model.h5')

# 重新預測(看載入的模型跟原本是不是一樣)
predict = model.predict(x_test_flat)
predict_class = np.argmax(predict, axis=-1)

# 預測
score = model.evaluate(x_test_flat, y_test_one_hot)
print('測試資料的loss', score[0])
print('測試資料的正確率', score[1])

plt.imshow(x_test[9487])
title = f'label={y_test[9487]}, predict={predict_class[9487]}'
plt.title(title)
plt.show()


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.3333 * r + 0.3333 * g + 0.3333 * b
    return 256-gray

from PIL import Image
for i in range(10):
    img = Image.open(f'{i}.png')
    numpydata = np.asarray(img)
    np_arr = rgb2gray(numpydata)

    np_x = np_arr.reshape(1,784)
    predict = model.predict(np_x)
    predict_class = np.argmax(predict, axis=-1)
    
    plt.imshow(np_arr)
    title = f'label={i}, predict={predict_class}'
    plt.title(title)
    plt.show()
