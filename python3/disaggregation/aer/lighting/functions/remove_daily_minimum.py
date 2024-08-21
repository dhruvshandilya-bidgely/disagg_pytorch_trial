"""
Author - Mayank Sharan
Date - 26/11/2019
remove daily minimum removes daily minimum value from each value in the day
"""

# Import python packages

import numpy as np


def remove_daily_minimum(day_data, exclude_zero):
    """Return day_data_mr

    REMDAILYMIN subtracts the minimum among day's value from all values for that day

    Input:
        day_data (double matrix) : 2d matrix with each row representing a day
        exclude_zero (double)    : Flag deciding whether zero should be included
    Output:
        day_data_mr (double matrix)  : 2d matrix with each row representing a day
    """

    # Do Stuff

    if exclude_zero == 1:
        day_data[day_data == 0] = np.nan

    daymin = np.nanmin(day_data, axis=1)

    rm2 = np.tile(daymin, (day_data.shape[1], 1)).transpose()
    day_data_mr = day_data - rm2

    if exclude_zero == 1:
        # Bug recreated to match with matlab should actually be 0
        day_data_mr[np.isnan(day_data_mr)] = np.nan
        day_data_mr[day_data_mr < 0] = 0

    return day_data_mr
