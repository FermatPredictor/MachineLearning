import math
from functools import reduce
from itertools import count

def isPrime(n):
    """
    函數功能: 判斷一個整數是不是質數
    example:
    >>> isPrime(0)
    False
    >>> isPrime(1)
    False
    >>> isPrime(2)
    True
    >>> isPrime(97)
    True
    >>> isPrime(121)
    False
    """
    return n==2 or (n > 2 and n%2 and all(n%x for x in range(3,int(n**.5)+1,2)))


def nextPrime(n):
    """
    函數功能: 找n的下一個質數是誰，
    count是無窮迭代器，從n+1開始一直往後生成{n+1, n+2, n+3,...}
    example:
    >>> nextPrime(15)
    17
    >>> nextPrime(2)
    3
    """
    return next(x for x in count(n+1) if isPrime(x))

def factorPrime(n):
    """
    函數功能: 質因數分解，回傳列表
    example:
    >>> factorPrime(108)
    [2, 2, 3, 3, 3]
    """
    factor=[]
    while n>0:
        for i in range(2,int(math.sqrt(n))+1):
            if n%i==0:
                n//=i
                factor.append(i)
                break
        else:
            factor.append(n)
            break
    return factor

def euler_phi(n):
    """
    函數功能: 計算euler phi的值，即小於n且與n互質的數字個數
    example:
    >>> [euler_phi(i) for i in range(1, 10)]
    [0, 1, 2, 2, 4, 2, 6, 4, 6]
    """
    primes= set(factorPrime(n))
    return reduce(lambda x,y: x//y*(y-1), primes, n)


if __name__ == "__main__":
    import doctest
    doctest.testmod()