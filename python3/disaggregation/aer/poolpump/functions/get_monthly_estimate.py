"""
Author - Mayank Sharan
Date - 23/1/2019
get monthly estimate collates consumption by billing cycle
"""

# Import python packages

import numpy as np


def get_monthly_estimate(month_ts, data_est):
    """ Return monthly_Estimate

    GETMONTHLYESTIMATE generates estimated usage per month

    Input:
      month_ts (double matrix)    : 2d matrix with month timestamps
      data_est (double matrix)    : 2d matrix containing pool pump estimates

    Output:
      monthly_Estimate (double matrix) : Matrix with month timestamp and estimate
    """

    # Do Stuff

    col_size = month_ts.size

    month_ts_col = np.reshape(a=month_ts, newshape=(col_size, 1))
    data_est_col = np.reshape(a=data_est, newshape=(col_size, 1))

    val_indices = ~np.isnan(month_ts_col)

    month_ts_col = month_ts_col[val_indices]
    data_est_col = data_est_col[val_indices]

    ts, _, idx = np.unique(month_ts_col, return_index=True, return_inverse=True)

    monthly_estimate = np.bincount(idx, weights=data_est_col)
    monthly_estimate = np.c_[ts, monthly_estimate]

    return monthly_estimate
