# -*- coding: utf-8 -*-
"""
程式功能: 記錄幾個好用的位元運算技巧
1. n&(n-1) : 將二進位數字最右邊的1改為0
2. n|(n+1): 將二進位數字最右邊的0改為1
"""

def removeRightestOne(n):
    return n&(n-1)

def addRightestOne(n):
    return n|(n+1)

# -x= ~x+1
def negative(x):
    return ~x + 1

# 取得最低位元的1
# 例如n= 11010 (二進位)
# -n = 00110(-n的算法為0, 1互換再加1)
# 取 n & -n即得到n的最低位
def lowbit(n):
    return n & -n

#取得最高位元的1
#想法: 透過一連串 n |= … 運算使n最高位以下全變成1
#一開始 n=1xxxxx
#n |= (n >>  1) 後變為11xxxx
#再做 n |= (n >>  1) 後變為1111xx
#以此類推，最後n ^ (n >> 1)把最高位以下的 1消去 
def hibit(n): 
    n |= (n >>  1)
    n |= (n >>  2)
    n |= (n >>  4) 
    n |= (n >>  8)
    n |= (n >> 16)
    return n ^ (n >> 1)

def next_2Pow(n):
    return hibit(n) << 1

#求n的二進位補數，單純0,1互換
def complement(n):
    return next_2Pow(n)-1-n

n=10**3
print(f'The binary number of {n} is {bin(n)[2:]}')
print(bin(removeRightestOne(n)))
print(bin(addRightestOne(n)))
print(bin(next_2Pow(n)))
print(bin(complement(n)))
print(bin(lowbit(n)))
print(bin(hibit(n)))
print(bin(negative(n)))
