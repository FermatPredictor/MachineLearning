from datetime import datetime
import time
import random

def rand_str(length):
    return ''.join([chr(ord('a')+random.randrange(26)) for _ in range(length)])

def rand_re(regex):
    """
    函數功能: 根據正則表達式隨機生成字串
    範例: random_re(r'([a-z])(\d{1,2}|[AJQK])')
    """
    try:
        from xeger import Xeger
    except Exception as e:
        print(e)
        print("Use 'pip install xeger' first")
        return ''
    _x = Xeger()
    return  _x.xeger(regex)



def rand_time(min_time, max_time):
    """
    函數功能: 回傳時間區間內的隨機時間，
    格式為年、月、日、時、分、秒，
    example:
        print(rand_time((2019,8,6,8,14,59), (2020,8,6,8,14,59)))
    """

    mintime_ts = int(time.mktime(datetime(*min_time).timetuple()))
    maxtime_ts = int(time.mktime(datetime(*max_time).timetuple()))
    random_ts = random.randint(mintime_ts, maxtime_ts)
    return datetime.fromtimestamp(random_ts)


if __name__=='__main__':
    print(rand_re(r'([a-z])(\d{1,2}|[AJQK])'))
    print(rand_time((2019,8,6,8,14,59), (2020,8,6,8,14,59)))