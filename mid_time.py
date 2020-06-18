from datetime import datetime, timedelta, time


mId = 0
MAX_VALUE = 9223372036854775807



def next():
    a = round(increase())
    a = hex(a).rstrip("L").lstrip("0x") or "0"
    return a

def increase():
    global mId
    now = round(datetime.now().timestamp() * 1000)
    if now > mId:
        mId = now
    else:
        ++mId
        now = mId
        if now == MAX_VALUE:
            mId = 0

    return now




