import math
from collections import Counter

def average(L):
    return sum(L)/len(L)

def median(L):
    L=sorted(L)
    return (L[len(L)//2]+ L[len(L)//2-1])/2 if len(L)%2==0 else L[len(L)//2]
 
 # 注意: 若有多個出現次數最多的數，回傳最小的那個   
def mode(L):
    return Counter(L).most_common(1)[0][0]

def std(L):
    mean = average(L)
    return math.sqrt(average([(y-mean)**2 for y in L]))

def quartile1(L):
    L=sorted(L)
    small=L[:(len(L)-len(L)%2)//2]
    return median(small)

def quartile3(L):
    L=sorted(L)
    large=L[(len(L)+len(L)%2)//2:]
    return median(large)
