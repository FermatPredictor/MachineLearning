# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist #參考資料: https://blog.csdn.net/kancy110/article/details/75675574

def K_means(photo, K, max_iter =100):
    """
    函數功能: 輸入原始資料photo(np array, 維度 = 照片大小*3(RGB))，將其分為K堆
    (參考資料: https://ithelp.ithome.com.tw/articles/10209058，內有k-mean算法的概念介紹)
    """
    
    def Mean(RGB_arr):
        """ 維度: (n,3)表示多個點的rgb值，回傳這些點的rgb平均值 """
        return np.sum(RGB_arr, axis=0)/RGB_arr.shape[0]

    # 不重複的隨機選取K個元素當做中心點
    center = photo[np.random.choice(photo.shape[0], K, replace=False)]
    c_new = np.zeros(len(photo))
    iterNum = 0
    while True:
        d = cdist(photo, center) # 計算所有點到每個中心的距離
        c = np.argmin(d,axis = 1) #記錄每個點比較靠近哪一堆 (np array, 維度 = 照片大小)
        center = [Mean(photo[c==i]) for i in range(K)] #迭代更新對新的分堆求新的中心點
        if all(c_new == c) or iterNum == max_iter: #檢查中心點是否不再移動，或迭代次數上限時結束迴圈
            break
        c_new = c
        iterNum+=1
    return center, c


def drawPicture(center, c, K):
    """
    函數功能: 
    center: 記錄每一堆的中心點顏色
    c: 記錄每個點比較靠近哪一堆 (np array, 維度 = 照片大小)
    K: int，表示分類分為K堆
    利用資訊center, c對圖片著色
    """
    img = Image.new('RGB', (height, width))
    for i in range(height):
        for j in range(width):
            index = i*width+j
            img.putpixel((i,j),tuple(center[c[index]]))
    plt.figure()               
    plt.imshow(img)
    plt.title(f'K = {K}')
    plt.show()

if __name__ == '__main__':
    """
    程式功能: 將照片的每個點上的(R,G,B)當做data，
    實作分類演算法(K_means)進行分堆，
    並以限定數量(K)的顏色對原始照片著色。
    """

    K = [2, 3, 5, 7, 10]
    filename = r'pic.jpeg'
    im = Image.open(filename)
    img = im.load()
    im.close()
    height, width = im.size
    photo = np.array([img[i,j] for i in range(height) for j in range(width)])
    print(photo.shape)

    for k in K:
        print(f"用{k}種顏色給照片著色")
        center, c = K_means(photo, k)
        center = np.array(center,dtype=int)
        
        drawPicture(center, c, k) #畫出照片
        

