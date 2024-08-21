"""
Author - Sahana M
Date - 20/07/2021
This file contains all the util files used in timed wh algorithm
"""

# Import python packages
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg


def tod_filler(col_data):
    """Creating a matrix with each column representing the time division of the day
    Parameters:
        col_data        (np.ndarray)    : Column data
    Returns:
        col_data        (np.ndarray)    : Column data
    """

    for i in range(col_data.shape[1]):
        col_data[:, i] = i

    return col_data


def exp_moving_average(values, window):
    """ Numpy implementation of EMA
    Parameters:
        values          (np.ndarray)    : Array to convolve
        window          (int)           : Window
    Returns:
        a               (array)         : Convolved array
    """

    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights)[:len(values)]
    a[:window] = a[window]
    return a


def get_start_end_idx(boolean_arr, end_idx_exclusive=False):
    """This function get the starting and ending index of all the True value boxes in a boolean array
    Parameters:
        boolean_arr         (np.ndarray)    : Input Boolean array to get start and end time stamps
        end_idx_exclusive   (Bool)          : End indexes to be included or not
    Returns:
        box_start_idx       (np.ndarray)    : Box start indexes
        box_end_idx         (np.ndarray)    : Box end indexes
    """

    box_energy_idx_diff = np.diff(np.r_[0, boolean_arr.astype(int), 0])
    box_start_idx = np.where(box_energy_idx_diff[:-1] > 0)[0]

    if end_idx_exclusive:
        box_end_idx = np.where(box_energy_idx_diff[1:] < 0)[0]
    else:
        box_end_idx = np.where(box_energy_idx_diff[1:] < 0)[0] + 1

    return box_start_idx, box_end_idx


def vertical_filter(data, window_size, percentile):
    """
    This function performs vertical filtering on a 2D matrix on a given window size and percentile
    Note: Make sure that the data provided in a 2D matrix where a row corresponds to a day, and column corresponds to time
    Parameters:
        data                    (np.ndarray)     :      Input 2D matrix
        window_size             (int)            :      Window size for filtering
        percentile              (int)            :      Percentile value for filtering

    Returns:
        vertical_filtered_data  (np.ndarray)     :      Vertically filtered data
    """

    # Load the required data

    data_matrix = deepcopy(data)

    # Initialise an empty 2d array to fill the filtered data

    vertical_filtered_data = np.full_like(data_matrix, fill_value=0.0)

    # Iterate over window_size*2 + 1 days(row wise) and perform percentile filter

    for row in range(data_matrix.shape[0]):

        # get the start and end row indexes

        start_idx = int(max(0, row - window_size))
        end_idx = int(min(data_matrix.shape[0], row + window_size + 1))

        # Extract the data between the start and end index

        window_data = data_matrix[start_idx:end_idx, :]

        # Perform percentile filtering over the window data and fill the vertical data matrix with the new values

        vertical_filtered_data[row, :] = np.percentile(window_data, axis=0, q=percentile)

    return vertical_filtered_data


def horizontal_filter(data, window_size, percentile, sampling_rate):
    """
    This function performs horizontal filtering on a 2D matrix on a given window size and percentile
    Parameters:
        data                           (np.ndarray)     :      Input 2D matrix
        window_size                    (int)            :      Window size for filtering
        percentile                     (int)            :      Percentile value for filtering
        sampling_rate                  (float)          :      Sampling rate of the data

    Returns:
        horizontal_filtered_data       (np.ndarray)     :       Horizontal filtered data
    """

    # Load the necessary data

    data_matrix = deepcopy(data)
    factor = Cgbdisagg.SEC_IN_HOUR/sampling_rate

    # Scale the window size according to the sampling rate

    window_size = window_size*factor

    # Initialise an empty horizontal_filtered_data array to fill the filtered values

    horizontal_filtered_data = np.full_like(data_matrix, fill_value=0.0)

    # Iterate over the window size*2 + 1 time interval(column wise) and perform percentile filter

    for time in range(data_matrix.shape[1]):

        # get the start and end column indexes

        start_idx = int(max(0, time - window_size))
        end_idx = int(min(data_matrix.shape[1], time + window_size + 1))

        # Extract the data between the start and end index

        window_data = data_matrix[:, start_idx:end_idx]

        # Perform percentile filtering over the window data and fill the horizontal data matrix with the new values

        horizontal_filtered_data[:, time] = np.percentile(window_data, axis=1, q=percentile)

    return horizontal_filtered_data
