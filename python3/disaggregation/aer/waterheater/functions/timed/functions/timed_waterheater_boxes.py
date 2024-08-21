"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to find high energy boxes in consumption data
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.rolling_function import rolling_function


def get_timed_waterheater_boxes(in_data, wh_config, wh_min, logger_base):
    """
    Find the boxes in the consumption data

    Parameters:
        in_data             (np.ndarray)    : The input data (21-column matrix)
        wh_config           (dict)          : Configuration dictionary for timed water heater
        wh_min              (int)           : The minimum energy value per data point
        logger_base         (logger)        : Logger object to log values

    Returns:
        input_data             (np.ndarray)    : The box data output at epoch level
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_timed_waterheater_boxes')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking deepcopy of input data to avoid scoping issues

    input_data = deepcopy(in_data)

    # Find the window size of the boxes required

    window_size = wh_config['minimum_box_points']
    window_half = window_size // 2

    logger.info('Windows size for timed water heater boxes | {}'.format(window_size))

    # Subset consumption data for finding boxes

    energy = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Mark all the consumption points above minimum as valid

    valid_idx = np.array(energy >= wh_min)

    # Accumulate the valid count over the window

    moving_sum = rolling_function(valid_idx, window_size, 'sum')

    # Consolidate the chunks of valid high energy consumption points

    valid_sum_bool = (moving_sum >= window_size)

    sum_idx = np.where(valid_sum_bool)[0]
    sum_final = deepcopy(sum_idx)

    # Padding the boxes for the first and last window

    for i in range(1, window_half + 1):
        sum_final = np.r_[sum_final, sum_idx + i]

        if i != window_half:
            sum_final = np.r_[sum_final, sum_idx - i]

    # Updating the valid sum bool

    valid_sum_bool[sum_final[sum_final < input_data.shape[0]]] = True

    # Make all invalid consumption points as zero in input data

    input_data[~valid_sum_bool, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    return input_data
