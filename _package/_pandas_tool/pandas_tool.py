import random
import pandas as pd
import numpy as np
import time
from datetime import datetime

import sys
sys.path.append('../..') # 添加相對路徑上層到sys.path，讓程式找到的模組_package
from _package._random_tool.random_tool import rand_time

def to_csv(df, file_name):
    df.to_csv(file_name, index=False)
    
def read_csv(file_name):
    for enc in ['big5', 'utf-8']:
        try:
            return pd.read_csv(file_name, encoding=enc)
        except:
            pass
    raise RuntimeError('We try all encoding but cannot read the csv')
    
def ini_df(data_dict):
    """
    caller example:
    scores = {"姓名":["小華","小明","小李"],
             "國文":[80,55,75],
             "數學":[90,70,45]}
    df = ini_df(scores)
    """
    return pd.DataFrame.from_dict(data_dict)
    
def rand_df(rand_dict, row_num, miss_rate=0, na_values = None):
    """
    函數功能: 依指定規則生成一個隨機dataframe
    參數:
        rand_dict: key為column名稱，value為自定義隨機函數
        row_num: 總共有幾筆資料
        miss_rate: 資料丟失的機率
        na_values: 可迭代容器，當資料丟失時產生的值 (默認為 np.nan)
    caller example:
        rand_dict = {'status': lambda: random.choice(['Wait', 'Start'])}
        print(rand_df(rand_dict, 20, miss_rate=0.2, na_values=['None', 'NoData']))  
    """
    def miss_judge(val):
        if random.random()<= miss_rate:
            return random.choice(na_values) if na_values else np.nan
        return val
        
    assert 0 <= miss_rate <= 1, 'probability must in [0,1]'
    df = pd.DataFrame()
    for k,v in rand_dict.items():
        df[k] = [miss_judge(v()) for _ in range(row_num)]
    return df

def df_groupby(df, col, inplace = False):
    """
    函數功能: 
    將dataframe依指定的col group起來，
    回傳一個字典
    參數:
        inplace: 是否原地修改df
    caller example:
    >>> df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
                       'Max Speed': [380., 370., 24., 26.],
                       'Value':[3,3,2,6]})
    >>> print(df_groupby(df, 'Animal'))
    {'Falcon':    Animal  Max Speed  Value
                0  Falcon      380.0      3
                1  Falcon      370.0      3,
     'Parrot':    Animal  Max Speed  Value
                2  Parrot       24.0      2
                3  Parrot       26.0      6}
    """
    if not inplace:
        df = df.copy()
    return {name:group for name, group in df.groupby([col])}
    
def df_groupby_list(df, col, inplace = False, remove_dup = True):
    """
    函數功能: 
    將dataframe依指定的col group起來，
    其它欄位會group成list
    參數:
        inplace: 是否原地修改df
        remove_dup: 是否去除重複值
    caller example:
    >>> df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
                       'Max Speed': [380., 370., 24., 26.],
                       'Value':[3,3,2,6]})
    >>> print(df_groupby_list(df, 'Animal'))
    
                 Max Speed   Value
    Animal                        
    Falcon  [370.0, 380.0]     [3]
    Parrot    [24.0, 26.0]  [2, 6]
    """
    if not inplace:
        df = df.copy()
    other_cols = list(filter(lambda x:x!=col, df.columns))
    df[other_cols] = df[other_cols].applymap(lambda x:[x])
    df = df.groupby([col]).sum()
    if remove_dup:
        df[other_cols] = df[other_cols].applymap(lambda x:list(set(x)))
    return df


def time_diff(series_1, series_2):
    """
    函數功能: 
    回傳兩個series的時間差，單位為秒(其中一個變數可以是常量)
    """
    arr = [series_1, series_2]
    for i, e in enumerate(arr):
        try:
            arr[i] = e.apply(lambda x: int(time.mktime(pd.Timestamp(x).timetuple())))
        except:
            arr[i] = int(time.mktime(pd.Timestamp(e).timetuple()))
    return arr[0] - arr[1]    
    
        
if __name__=='__main__':
    rand_dict = {'status': lambda: random.choice(['Wait', 'Start'])}
    random_df = rand_df(rand_dict, 20, miss_rate=0.2, na_values=['None', 'NoData'])
    
    df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
                                  'Max Speed': [380., 370., 24., 26.],
                                  'Value':[3,3,2,6]})
    print(df_groupby_list(df, 'Animal'))
    print(df_groupby(df, 'Animal'))
    
    scores = {"姓名":["小華","小明","小李"],
             "國文":[80,55,75],
             "數學":[90,70,45]}
    score_df = ini_df(scores)
    
    
    min_time = (2020,10,23,0,0,0)
    max_time = (2020,10,23,2,0,0)
    now_time = (2020,10,23,4,0,0)
    rand_dict = {'time_1': lambda: rand_time(min_time, max_time),
                 'time_2': lambda: rand_time(min_time, max_time)}
    df = rand_df(rand_dict, 20)
    df['time_diff_minutes'] =  time_diff(df['time_1'], df['time_2'])//60
    df['time_diff_now_minutes'] =  time_diff(datetime(*now_time), df['time_2'])//60
    