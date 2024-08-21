"""
Author: Mayank Sharan
Created: 8-Apr-2020
Have different utility functions related to time based operations
"""

# Import python packages

import numpy as np


def get_time_diff(t0, t1):

    """
    Compute the time in seconds from datetime objects
    Parameters:
        t0                  (datetime.datetime) : Initial time datetime object
        t1                  (datetime.datetime) : Final time datetime object
    Output:
        t_diff              (float)             : Time difference as seconds
    """

    t_diff = (t1 - t0).seconds + np.round((t1 - t0).microseconds / 1000000, 3)
    return t_diff
