def remove_duplicate(x):
    """
    函數功能: 去除列表中重複的元素，保持列表元素原來的順序
    >>> remove_duplicate(["a", "b", "a", "c", "c"])
    ['a', 'b', 'c']
    """
    return sorted(set(x), key = x.index)

def merge_dicts(*dict_args):
    """
    函數功能: 合併多個字典，若key值相同新的會覆蓋舊的
    >>> merge_dicts({'a': 1, 'b': 2}, {'b': 3, 'c': 4}, {'e': 10})
    {'a': 1, 'b': 3, 'c': 4, 'e': 10}
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_rank(X):
    """
    函數功能: 給定一個整數陣列，回傳每個位置元素的排名(最小的元素為1)
    >>> get_rank([1,1,2,5,-6,0,2])
    [3, 3, 4, 5, 1, 2, 4]
    """
    x_rank = dict((x, i+1) for i, x in enumerate(sorted(set(X))))
    return [x_rank[x] for x in X]


def flat_list(arr):
    """
    函數功能: 將嵌套list拉平成一維
    >>> flat_list([{1,2},{1:'A'},(1,"H",3),[5,[2,1],(3,4)],50,(1,['54',0])])
    [1, 2, 1, 1, 'H', 3, 5, 2, 1, 3, 4, 50, 1, '54', 0]
    """
    return [x for sub in arr for x in flat_list(sub)] if isinstance(arr, (list,tuple,set,dict)) else [arr]

def sepList(L,step):
    """
    函數功能: 將一個列表k個元素分成一組，變成兩維列表
    >>> sepList([1,2,3,4,5,6,7,8,9,10,11,20,66],3)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 20], [66]]
    """
    return [L[i:i+step] for i in range(0,len(L),step)]

def backSepList(L,step):
    """
    函數功能: 將一個列表k個元素分成一組(從後面分組)，變成兩維列表
    >>> backSepList([1,2,3,4,5,6,7,8,9,10,11,20,66],3)
    [[1], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 20, 66]]
    """
    return [L[max(0,i-step+1):i+1] for i in range(len(L)-1,-1,-step)][::-1]


def rotateRight(s, offset):
    """
    函數功能: 把一個字串(or陣列)向右旋轉offset個長度
    例: abcdefg, offset = 3+7k => efgabcd
    >>> rotateRight("",3)
    ''
    >>> rotateRight("abcdefg",3)
    'efgabcd'
    """
    return s[-offset%len(s):]+s[:-offset%len(s)] if s else s


def rotateLeft(s, offset):
    """
    函數功能: 把一個字串(or陣列)向左旋轉offset個長度
    >>> rotateLeft("abcdefg",3) 
    'defgabc'
    >>> rotateLeft([1,2,3,4],1)
    [2, 3, 4, 1]
    """
    return s[offset%len(s):]+s[:offset%len(s)] if s else s

if __name__ == "__main__":
    import doctest
    doctest.testmod()