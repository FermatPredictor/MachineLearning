import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def softmax(x):
    """
    example:
    >>> softmax([1,2,3])
    [0.09003057, 0.24472847, 0.66524096]
    (explain: [e^1, e^2, e^3]/(e^1+e^2+e^3))
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def predict(X_data, w):
    """
    利用參數w預測X像各個類別的機率值
    X_data: 一維陣列，長度 = feature數量
    w: shape = 類別數量 * feature數量
    return 長度為類別數量的list
    
    例如return 值為[0.5, 0.2, 0.3]，
    表示資料
    像第0類的機率大概`0.5`，
    像第1類的機率大概`0.2`，
    像第2類的機率大概`0.3`，
    """
    return softmax(np.dot(w,X_data))

#畫圖函數
def Plot(errors, accuracy, epoch):
    plt.figure()
    plt.plot(epoch, errors, 'blue', label = 'errors' )
    plt.plot(epoch, accuracy, 'red', label = 'accuracy')
    plt.legend()
    plt.show()


def Training(x_train, y_train, training_times, plot=False):
    """
    用牛頓法做分類的參數訓練
    x_train: shape = data數量 * feature數量
    y_train: 格式為one-hot encoding， shape = data數量 * 類別數量
    training_times: 牛頓法的迭代次數
    在迭代過程中，
    我們同步將誤差和準確率印出來
    
    return: 一組參數w, shape =類別數量 * feature數量 (參數w用來預測資料像每個類別的機率)
    """
    data_num, feature = x_train.shape
    classNum = y_train.shape[1]
    
    w = np.zeros([classNum, feature])
    
    errors = []
    accuracys = []
    
    #牛頓法，迭代更新參數w
    for i in range(training_times):
        
        # 看準確率(注意第0次時即隨便亂分)
        yn = [predict(x_train[n], w) for n in range(data_num)]
        error = -np.sum(y_train * np.log(yn))
        accuracy = sum([y_train[n][np.argmax(yn[n])] for n in range(data_num)])/data_num
        errors.append(error)
        accuracys.append(accuracy)
        print(f"train{i}: error={error}, accuracy={accuracy}")
        
        # 送代更新參數
        for j in range(classNum):     
            gradient = np.zeros([1, feature]) # 梯度
            for n in range(data_num):
                gradient += (yn[n][j]-y_train[n][j])*x_train[n]
                
            H = np.zeros([feature, feature]) # Hessian矩陣
            for n in range(data_num):
                 H += yn[n][j]*(1-yn[n][j])*np.dot(x_train[n].reshape(-1,1),x_train[n].reshape(1,-1))
            w[j] -= np.dot(gradient, np.linalg.inv(H.T)).flatten()        
    
    if plot:
        Plot(errors, accuracys, range(training_times))
    return w

   
def Testing(x_test, y_test, w):
    """
    用來驗證參數w在testing data上的表現。
    x_test: shape = data數量 * feature數量
    y_test: 格式為one-hot encoding， shape = data數量 * 類別數量
    w:  shape = 類別數量 * feature數量
    return 準確度 = 正確分類/資料總數
    """
    data_num = x_test.shape[0]
    yn = [predict(x_test[n],w) for n in range(data_num)]
    accuracy = sum([y_test[n][np.argmax(yn[n])] for n in range(data_num)])/data_num
    print(f"testing data 的準確率={accuracy}")
    return accuracy

def main(x_data, y_data, iter_num):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=41)
    w = Training(x_train, y_train, iter_num, plot=True)
    Testing(x_test,y_test, w)
    
if __name__ == '__main__':
    iris = load_iris()
    x_data, y_data = iris['data'], iris['target']
    y_data = pd.get_dummies(y_data)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    main(x_data, y_data, 6)