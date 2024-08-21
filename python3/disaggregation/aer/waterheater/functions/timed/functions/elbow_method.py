"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Using elbow method find the optimum breakpoint
"""

# Import python packages

import numpy as np
from copy import deepcopy


def elbow_method(input_arr, elbow_threshold):
    """
    Finding optimum threshold to remove insignificant hour boxes
    Parameters:
        input_arr           (np.ndarray)        : Input array of fractions
        elbow_threshold     (float)             : Minimum threshold for elbow

    Returns:
        threshold           (float)             : Optimum breakpoint
    """

    # Taking deepcopy of input data to avoid scoping issues

    arr = deepcopy(input_arr)

    # Sort array in descending order

    arr = np.sort(arr)[::-1]

    # Initially mark time division below lower limit as unsafe

    unsafe_hours = (arr < (elbow_threshold * np.max(arr)))

    # Take moving diff of fraction array (fraction curve)

    abs_diff = np.diff(np.r_[arr[0], arr])

    # Get slope of each section of fractions curve

    theta = np.arctan(abs_diff)

    # Calculate change in slope at each section of fraction curve

    angular_change = np.diff(np.r_[theta, theta[-1]])
    angular_change[np.isnan(angular_change)] = 0

    # Find the section with highest change in slope

    thres_idx = np.argmax(angular_change * unsafe_hours)

    # Making sure the first section is not the breakpoint

    if thres_idx == 0:
        thres_idx = np.argmin(arr) + 1

    # Return the threshold (breakpoint) with added error (0.01)

    error_bound = 0.01

    thres_idx = np.fmax(thres_idx, 1)
    thres_idx = np.fmin(thres_idx, len(arr) - 1)

    threshold = arr[thres_idx] + error_bound

    return threshold
