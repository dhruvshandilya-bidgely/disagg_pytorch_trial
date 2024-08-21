"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Convert hour of day to time division of day.
Example: 1800 sampling rate will have 48 time divisions
"""

# Import python packages

import numpy as np
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def hour_to_time_division(in_data, sampling_rate, offset=0):
    """
    Parameters:
        in_data         (np.ndarray)    : Input 21-column matrix
        sampling_rate   (int)           : Time interval between energy data points (in seconds)
        offset          (int)           : Default time offset

    Returns:
        input_data         (np.ndarray)    : Input data with updated hour of day column
    """

    # Taking copy of the input data

    input_data = deepcopy(in_data)

    # Get number of data points in offset hours

    n_offset = offset * Cgbdisagg.SEC_IN_HOUR

    # Update the epoch timestamp based on time zone if offset provided

    input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] + n_offset

    # Extract the hour of day from the epoch timestamp

    adjusted_hod = [datetime.fromtimestamp(x).hour for x in input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]]

    # Update the hour of day column in input data

    input_data[:, Cgbdisagg.INPUT_HOD_IDX] = adjusted_hod

    # Find the number of divisions in each hour

    n = Cgbdisagg.SEC_IN_HOUR // sampling_rate

    # Get the number of steps in each hour

    hour_steps = np.arange(0, 1, 1 / n)

    # Subset the hour of day column

    hod_column = input_data[:, Cgbdisagg.INPUT_HOD_IDX]

    # Find the offset point to get time divisions

    hod_offset = np.where(np.diff(hod_column) != 0)[0][0] + 1

    # Find the number of complete hours in the data

    full_steps = ((len(hod_column) - hod_offset) // n) + 1

    # Find the time division of all timestamps based on first timestamp

    first = hour_steps[-hod_offset:]
    rest = np.array(list(hour_steps) * int(full_steps))

    # Update hour of day column with the time divisions

    new_hod = np.r_[first, rest][:len(hod_column)]
    input_data[:, Cgbdisagg.INPUT_HOD_IDX] = input_data[:, Cgbdisagg.INPUT_HOD_IDX] + new_hod

    return input_data
