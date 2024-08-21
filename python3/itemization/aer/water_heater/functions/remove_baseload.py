"""
Author - Sahana M
Date - 2/3/2021
Removes baseload from the input data
"""

# Import python packages

from copy import deepcopy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.rolling_function import rolling_function


def remove_baseload(input_matrix, window_hours, sampling_rate):
    """
    Parameters:
        input_matrix        (np.ndarray)        : Input data
        window_hours        (int)               : Moving window size
        sampling_rate       (int)               : Sampling rate of the data

    Returns:
        input_data          (np.ndarray)        : Updated input data
    """

    # Taking copies of input matrix for new input and baseload data

    input_data = deepcopy(input_matrix)
    baseload_data = deepcopy(input_matrix)

    # Finding number of data points required for the given window size

    window_size = window_hours * Cgbdisagg.SEC_IN_HOUR / sampling_rate
    window_size = np.floor(window_size / 2) * 2 + 1

    # Passing a minimum filter over consumption data

    base_load = rolling_function(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], window_size, 'min')

    # Passing a maximum filter over the min values calculated in previous step

    base_load = rolling_function(base_load, window_size, 'max')

    # Save base-load to debug object for future reference

    baseload_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = base_load

    # Base-load removed from the input data

    input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.fmax(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - base_load,
                                                             0)
    return input_data
