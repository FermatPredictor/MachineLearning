import math
import numpy as np

"""
注意: 函數預期吃到的參數為numpy.array
"""

def RMS(perdict, real):
    """
    函數功能: 計算預測值和真實值之間的root mean square
    使用時機: ML regression 分析對test data預測的誤差，值望小
    """
    return math.sqrt(sum((real-perdict)**2)*2/len(real))



def normalization(arr):
    """
    函數功能: arr 是numpy的2D陣列，
    假設資料為橫的每列是一筆資料，
    每行為一個資料特微，
    將資料依行做線性轉換，最小值映射至0，最大值映射至1
    
    使用時機: 當資料的特微之間範圍差距太大時就可能造成計算誤差，
    比如說一個feature的單位是「公斤」、另一個是「公克」，數字就會差很大
    將資料映射到0~1之間通常會好很多
    """
    return np.apply_along_axis(lambda x: (x-min(x)) / (max(x)-min(x)), axis=0, arr=arr)

if __name__ == '__main__':
    X = np.array([[1,10],
                  [20,100],
                  [50,30]])
    print(normalization(X))