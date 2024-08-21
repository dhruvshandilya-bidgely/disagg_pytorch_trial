"""Time utility"""

from datetime import datetime, timedelta, date
import numpy as np


def unixtime2date(unix_time):
    """ Unix to date"""
    dt = datetime.utcfromtimestamp(unix_time)
    dv = datetime.timetuple(dt)
    date = np.array(dv[0:6])
    return date


def matlab_to_python_datetime(matlab_datenum):
    """ MA to PY date"""
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days=366)


def datetime2matlabdn(dt):
    """ MA date"""
    mdn = dt + timedelta(days=366)
    frac = (dt-datetime(dt.year, dt.month, dt.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac


def unixtime2datestr(unix_time, fmt="%b"):
    """returns month"""
    dt = datetime.fromtimestamp(unix_time)
    dv = datetime.timetuple(dt)
    month = date(int(dv[0]), int(dv[1]), int(dv[2])).strftime(fmt)
    return month
