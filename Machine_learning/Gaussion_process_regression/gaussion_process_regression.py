import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../..') # 添加相對路徑上兩層到sys.path，讓程式找到的模組_package
from _package._visualization.useful_matplot import plot_legend
from _package._data_analysis.data_analysis import RMS

def Kernel(xn, xm, par):
    """
    xn: may be array
    xm: scaler
    par: tuple of 4，不同的par為不同的kernel函數
    """
    return par[0]*np.exp(-par[1]/2*(xn-xm)**2)+par[2]+par[3]*xn*xm



def Gaussian_process(x_train, y_train, x_test, par):
    """
    函數功能: 輸入訓練資料x_train, y_train，
    以預測x_test的y值的可能分佈範圍(回傳平均值、標準差)，
    這邊的資料通通是一維
    
    Gaussian_process 與普通的linear regression差在
    linear regression直接給你一條最好的曲線，
    Gaussian_process 給出一個範圍上下界
    
    [TODO]雖然函式短，但研究很久沒理解裡面在計算什麼，
    有空時再找小一點的case run 看看
    """
    kernel = [[Kernel(x_train[i], x_train[j], par) for i in range(len(x_train))] for j in range(len(x_train))]
    

    k = [Kernel(x_train, x_test[i], par) for i in range(len(x_test))]
    k = np.array(k)
    
    c_plus = [Kernel(x_test[i], x_test[i], par) for i in range(len(x_test))]
    c_plus = np.array(c_plus)+1
    
    c = np.linalg.inv(kernel + np.eye(len(x_train)))

    mean = k.dot(c).dot(y_train)
    sigma = c_plus - np.sum(k.dot(c) * k, axis=1)
    return mean, sigma


if __name__ == '__main__':
    """
    用於kernal函數的theta參數，
    當參數為[0, 0, 0, 1]時是linear model
    """
    kernal_par = [[0, 0, 0, 1],
                  [1, 4, 0, 0],
                  [1, 4, 0, 5],
                  [1, 32, 5, 5]]
    
    df = pd.read_csv('gp.csv')
    x_data, y_data = np.array(df.iloc[:,0]), np.array(df.iloc[:,1])
    every_x =  np.linspace(np.min(x_data), np.max(x_data), 1500)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=41)
    for par in kernal_par:
        # 看train, test data 的誤差
        mean_train, sigma_train = Gaussian_process(x_train, y_train, x_train, par)
        mean_test, sigma_test = Gaussian_process(x_train, y_train, x_test, par)
        train_error = RMS(mean_train, y_train)
        test_error = RMS(mean_test, y_test)
        print(f'<kernal的參數> ={par}')
        print(f'train_error: {train_error:.2f}, test_error: {test_error:.2f}')
        
        mean, sigma = Gaussian_process(x_train, y_train, every_x, par)
        sigma_max = max(sigma)
        up = mean+sigma_max**0.5 # upper bound
        down = mean-sigma_max**0.5 # lower bound
        
        # 視覺化繪圖
        datas = [(x_data, y_data),
                 (every_x, mean),
                 (every_x, up),
                 (every_x,down)]
        legends = ['x_data', 'mean line', 'up boundary', 'down boundary']
        line_form = ['.', 'green', 'red', 'blue']
        plot_legend(datas, legends = legends,line_form = line_form,
                    fill = (every_x, up, down, 'pink'), 
                    title = f'Gaussian Process par={par}',
                    xlabel = 'x', ylabel='predictive value')