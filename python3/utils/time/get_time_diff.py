"""get time diff returns the time in seconds from datetime objects"""

# Import python packages

import numpy as np


def get_time_diff(t0, t1):

    """
    Parameters:
        t0                  (datetime.datetime) : Initial time datetime object
        t1                  (datetime.datetime) : Final time datetime object

    Output:
        Time difference as seconds
    """

    return (t1 - t0).seconds + np.round((t1 - t0).microseconds / 1000000, 3)
