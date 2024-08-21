"""
Author - Mayank Sharan
Date - 26/11/2019
percentile horizontal applies percentile filter on the data across the day
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile


def percentile_horizontal(day_data, pd, window, perc):
    """Return perc_Data

    PERCENTILEHORZ applies horizontal percentile(across epochs) filter to
    daywise data

    Input:
        day_Data (double matrix) : 2d matrix with each row representing a day
        window (double)         : The length of the vertical window in days
        perc (double)           : Percentile to be used for the filtering
        pd (double)             : Periodicity of data in seconds
    Output:
        perc_Data (double matrix)    : 2d output matrix with smoothed data
    """
    # Do Stuff

    num_ptw = int(window * 3600 / pd)
    num_pd = int(86400 / pd)
    perc_data = np.empty(shape=day_data.shape)
    perc_data[:] = np.nan

    for i in range(num_pd):
        idx_start = int(np.maximum(0, i - (num_ptw / 2)))
        idx_end = int(np.minimum(num_pd, i + num_ptw / 2)) + 1
        perc_data[:, i] = superfast_matlab_percentile(day_data[:, idx_start:idx_end], perc, 1)

    return perc_data
