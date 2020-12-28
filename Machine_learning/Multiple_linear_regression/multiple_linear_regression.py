from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations_with_replacement as CWR

import sys
sys.path.append('../..') # 添加相對路徑上層到sys.path，讓程式找到的模組_package
from _package._data_analysis.data_analysis import RMS, normalization

"""
 計算一個陣列透過idx的指定乘積
 例:idx=(i1,i2,...in)
 return nums[i1]*nums[i2]*...*nums[in]
"""
def prodOfArr(nums,idxs):
    return reduce(lambda x,y: x*nums[y], idxs, 1)

"""
 生成x的多項式函數，x資料維度=x_dim，次方最高到polyDeg
 例x_dim=4, polyDeg=2
 則X的一筆data形式為 [x1,x2,x3,x4]
 會得到函數: [1,x1,x2,x3,x4, x1*x2, x1*x3,x1*x4,x2*x3,x2*x4,x3*x4]
"""
def designPolyFunc(x_dim, polyDeg=1):
    funcs = [lambda X:1]
    for i in range(1,polyDeg+1):
        funcs += [lambda X, idx=c: prodOfArr(X,idx) for c in CWR(range(x_dim),i)]
    return funcs


"""
 透過函數們funcs將原始資料X=
 x01 x02 ... x0n (= vector x0)
 x11 x12 ... x1n (= vector x1)
 ...
 xm1 xm2 ... xmn (= vector xm)
 其中(m是data數量，n是x的feature數量)
 轉換成矩陣形式:
 1 f1(x0) f2(x0) ... fk(x0)
 1 f1(x1) f2(x1) ... fk(x1)
 ...
 1 f1(xm) f2(xm) ... fk(xm)
"""
def preprocess(funcs, X):
    X = normalization(X)
    mat = [[f(x) for f in funcs] for x in X]
    return np.array(mat)

# 公式解B= [(X_TX)^-1+ lamb\*I(單位矩陣)]X_TY
def parameter(X, Y, lamb=0):
    return np.dot(np.linalg.inv(np.dot(X.T,X)+lamb*np.eye(X.shape[1])),np.dot(X.T,Y))


# 把訓練資料、測試資料(預處理過的data)、lamb(若用MAP方法才要傳lamb)，做一次訓練
# 回傳值: 訓練得到的最佳參數、訓練資料的RMS、測試資料的RMS
def training(x_train,x_test, y_train, y_test, lamb=0):
    beta = parameter(x_train,y_train, lamb)
    
    ## 以下用train_data做predict，計算train_data的RMS
    train_perdict_Y = np.dot(x_train,beta)
    train_error= RMS(train_perdict_Y,y_train)
    
    ## 以下用test_data做predict，計算test_data的RMS
    perdict_Y = np.dot(x_test,beta)
    test_error= RMS(perdict_Y,y_test)
    return beta, train_error, test_error

    
if __name__ == '__main__':
    np.random.seed(931) # 這邊設固定的random seed，確保練習用的資料相同
    X = np.random.randint(low=4,high=1000,size=(100,3))
    Y = np.dot(X, np.array([0.5, 0.4, 0.7])) + 150
    
    # 資料預處理
    funcs = designPolyFunc(3, polyDeg=2)
    X_pro = preprocess(funcs, X)
    
    # 資料切割成訓練集、測試集
    x_train, x_test, y_train, y_test = train_test_split(X_pro, Y, test_size=0.2)
    beta, train_error, test_error = training(x_train, x_test, y_train, y_test)
    print(beta)
    print(train_error)
    print(test_error)
