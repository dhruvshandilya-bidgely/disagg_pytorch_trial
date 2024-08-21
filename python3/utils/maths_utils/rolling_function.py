"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Function to calculate mean/min/max in a rolling window exactly like MATLAB
function "lemire_nd_minengine". The logic is to stack the data by shifting place by place
vertically and taking the row min/max/mean/median/sum/std

Example:
# >>> Given x = [2 5 6 NaN -1 4 0 23 NaN NaN 3 NaN NaN NaN NaN 2 7]
# >>> roll_minmax(x, 5, 'min')
# >>> 2     2    -1    -1    -1    -1    -1     0     0     3     3     3     3     2     2     2     2
"""

# Import python packages

import numpy as np
from copy import deepcopy


# List of metrics supported by the rolling function

metrics = {
    'min': np.nanmin,
    'max': np.nanmax,
    'mean': np.nanmean,
    'sum': np.nansum,
    'std': np.nanstd,
    'median': np.nanmedian,
}


def rolling_function(in_data, window, out_metric='both'):
    """
    Parameters:
        in_data         (np.ndarray)        : Input 21-column matrix
        window          (int)               : Moving window size
        out_metric      (string)            : Type of metric

    Returns:
        output          (np.ndarray)        : Output array based on the asked metric
    """

    # Taking deep copy of the input data and replacing NaN with zeros

    data = deepcopy(in_data)
    data[np.isnan(data)] = 0

    # Making sure window is int

    window = int(window)

    # Creating an empty array to do row-wise calculations

    n_rows = data.shape[0]
    temp = np.zeros(shape=[n_rows, int(window)])

    # Treat odd / even sized windows separately

    if window % 2 == 0:
        # If the window size is n (even) data points

        # Filling columns before the middle column with backward shifted arrays

        for row, column in zip(range(-(window - 3) // 2, 0), range((window - 1) // 2)):
            temp[abs(row):, column] = data[:row]

        # Filling the middle column with original array

        temp[:, (window - 1) // 2] = data

        # Filling columns after the middle column with forward shifted arrays

        for row, column in zip(range(1, (window + 2) // 2), range(window // 2, int(window))):
            temp[:-row, column] = data[row:]
    else:
        # If the window size is n (odd) data points

        # Filling columns before the middle column with backward shifted arrays

        for row, column in zip(range(-(window - 1) // 2, 0), range((window - 1) // 2)):
            temp[abs(row):, column] = data[:row]

        # Filling the middle column with original array

        temp[:, (window - 1) // 2] = data

        # Filling columns after the middle column with forward shifted arrays

        for row, column in zip(range(1, (window + 1) // 2), range((window + 1) // 2, int(window))):
            temp[:-row, column] = data[row:]

    # Calculating the output metric on the data

    if out_metric == 'both':
        # If both min and max are required

        out1 = np.min(temp, axis=1)
        out2 = np.max(temp, axis=1)

        return out1, out2
    else:
        # If a particular output metric is required (from list of functions above)

        output = metrics[out_metric](temp, axis=1)

        return output
