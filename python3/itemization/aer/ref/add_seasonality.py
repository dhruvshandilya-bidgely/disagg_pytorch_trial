
"""
Author - Nisha Agarwal
Date - 10/9/20
convert ref estimate to epoch level and add seasonality
"""

# Import python packages

import numpy as np
from scipy.stats import mode

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff


def add_seasonality(day_estimate, item_input_object):

    """
    convert ref estimate to epoch level and add seasonality

    Parameters:
        day_estimate            (int)              : Estimated ref day level consumption
        item_input_object     (dict)             : Dictionary containing all input

    Returns:
        epoch_estimate          (numpy.ndarray)    : Epoch level ref output
    """

    input_data = item_input_object.get("input_data")

    # Calculate sampling rate of the user

    sampling_rate = int(mode(np.diff(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]))[0][0])

    # Convert day estimate to epoch level estimates of all available days

    epoch_estimate = (day_estimate * sampling_rate) / (Cgbdisagg.HRS_IN_DAY * Cgbdisagg.SEC_IN_HOUR)

    epoch_estimate = np.ones(input_data[:, 0].shape) * epoch_estimate

    # Safety check for epoch level estimates

    input_data = item_input_object["input_data"][:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    input_data = np.fmax(input_data, 0)

    epoch_estimate = np.minimum(epoch_estimate, input_data)

    epoch_estimate = np.nan_to_num(epoch_estimate)

    day_input_data = item_input_object.get("item_input_params").get("day_input_data")

    day_level_estimate = np.ones_like(day_input_data) * (day_estimate * sampling_rate) / (Cgbdisagg.HRS_IN_DAY * Cgbdisagg.SEC_IN_HOUR)

    day_level_estimate = np.minimum(day_level_estimate, day_input_data)

    day_level_estimate = np.nan_to_num(day_level_estimate)

    return epoch_estimate, day_level_estimate
