"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module finds the potential active usage of Water Heater (referred as Fat pulse)
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.rolling_function import rolling_function


def get_fat_boxes(in_data, wh_config, lower_amp, logger_base):
    """
    Find the boxes in the raw data

    Parameters:
        in_data             (np.ndarray)    : Input 21-column matrix
        wh_config           (dict)          : Config params
        lower_amp           (float)         : Lower fat bound
        logger_base         (dict)          : Logger object

    Returns:
        input_data          (np.ndarray)    : Potential fat pulse data
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_fat_boxes')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking a deepcopy of input data to keep local instances

    input_data = deepcopy(in_data)

    # Check if input data is empty

    if input_data.shape[0] == 0:
        return np.array([], dtype=np.int64).reshape(0, Cgbdisagg.INPUT_DIMENSION + 1)

    # Maximum duration of fat pulse allowed

    max_duration = wh_config['thermostat_wh']['estimation']['max_fat_pulse_duration']

    # Calculate the window size for box fitting

    min_window = int(wh_config['thermostat_wh']['detection']['minimum_box_size'] / wh_config['sampling_rate'])

    # Making sure that the window is at least of size 1 unit

    min_window = np.fmax(min_window, 1)

    # Calculate the half window size (in terms of data-points)

    window_half = int(min_window // 2)

    # Define the max window size based on max duration

    max_window = int(max_duration * (Cgbdisagg.SEC_IN_HOUR / wh_config.get('sampling_rate')))

    # Extract the energy values from raw data

    energy = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Mask energy values above min threshold as valid

    valid_idx = np.array(energy >= lower_amp)

    # Accumulate the valid count over the window

    mov_sum = rolling_function(valid_idx, min_window, 'sum')

    # Filter the chunks of length in the defined window range

    valid_sum_bool = (mov_sum >= min_window) & (mov_sum <= max_window)

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

    logger.info('Initial total box consumption | {}'.format(np.nansum(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])))

    return input_data
