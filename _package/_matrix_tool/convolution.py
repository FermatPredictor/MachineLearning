from pprint import pprint
import random

"""
關於什麼是convolution，
參考: http://www.songho.ca/dsp/convolution/convolution2d_example.html
"""

#創一個裡面元素均為0的二維陣列
def create2Darr(column,row):
    return [[0]*column for _ in range(row)]

def convolution(arr, kernal):
    arr_row, arr_col=len(arr), len(arr[0])
    ker_row, ker_col=len(kernal), len(kernal[0])
    out = create2Darr(arr_col-ker_col+1, arr_row-ker_row+1)
    for i in range(len(out)):
        for j in range(len(out[0])):
            for k in range(ker_row):
                for m in range(ker_col):
                    out[i][j]+=arr[i+k][j+m]*kernal[k][m]
    return out

if __name__ == '__main__':
    ker=[[1,1,1],
         [0,1,0],
         [1,1,1]]
    arr=[[random.randrange(3) for _ in range(5)] for _ in range(6)]     
    pprint(arr)
    pprint(ker)
    pprint(convolution(arr,ker))
