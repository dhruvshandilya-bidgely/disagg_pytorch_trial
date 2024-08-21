"""Time vectors"""
from datetime import datetime
import numpy as np


def unixtime2datevec(unix_time):
    """Time vectors"""
    dt = datetime.utcfromtimestamp(unix_time)
    dv = datetime.timetuple(dt)
    time = np.array(dv[0:6])
    return time
