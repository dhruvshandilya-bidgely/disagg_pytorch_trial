"""
Author - Mayank Sharan
Date - 26/11/2019
remove vacation sets all consumption values during vacation to 0
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def remove_vacation(in_data, vacation_periods):

    """ Return out_data

    REMOVEVACATION makes all vacation day values in the input data zero.

    Input:
        in_data (double matrix)                 : 21 column Input data matrix
        vacation_periods (struct)               : 3 column matrix with start
                                                 and end timestamps of vacation

    Output:
        out_data (double matrix)                : 21 column data matrix
    """

    # Do stuff

    out_data = in_data

    for i in range(vacation_periods.shape[0]):
        out_data[np.logical_and(out_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= vacation_periods[i, 0],
                                out_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= vacation_periods[i, 1]),
                 Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    return out_data
