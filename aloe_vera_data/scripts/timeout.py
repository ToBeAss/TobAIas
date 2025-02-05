import time
from datetime import datetime, timedelta

def next_hour():
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_time = (next_hour - now).total_seconds()

    time.sleep(wait_time)
    return(next_hour)