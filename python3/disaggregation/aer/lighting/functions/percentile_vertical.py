"""
Author - Mayank Sharan
Date - 26/11/2019
percentile vertical applies percentile filter on the data across days
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.utils.maths_utils.matlab_utils import percentile_1d, superfast_matlab_percentile


def percentile_vertical(day_data, window, perc):
    """Return perc_data

    PERCENTILEVERT applies vertical percentile(across days) filter to daywise
    data. Ignores the days with all values 0

    Input:
        day_data (double matrix) : 2d matrix with each row representing a day
        window (double)         : The length of the vertical window in days
        perc (double)           : Percentile to be used for the filtering

    Output:
        perc_data (double matrix)    : 2d output matrix with smoothed data
    """

    # Do Stuff

    num_days = day_data.shape[0]

    perc_data = np.empty(shape=day_data.shape)
    perc_data[:] = np.nan

    # For each day get the data in the window remove zero consumption days and take percentile

    for i in range(num_days):

        idx_start = int(max(0, i - window / 2))
        idx_end = int(min(num_days, i + window / 2 + 1))

        m = np.nanmax(day_data[idx_start: idx_end, :], axis=1)

        temp = day_data[idx_start: idx_end, :]
        temp = temp[m > 0, :]

        # Creating a bug here to match MATLAB

        if not temp.shape[0] == 1:
            perc_data[i, :] = superfast_matlab_percentile(temp, perc)
        else:
            perc_data[i, :] = percentile_1d(temp[0], perc)

    return perc_data
