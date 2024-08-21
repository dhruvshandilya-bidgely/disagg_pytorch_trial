"""
Author - Nikhil Singh Chauhan
Date - 15-May-2020
Remove baseload in a certain hour size max-min window
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.rolling_function import rolling_function


def remove_baseload(input_matrix, debug, ev_config, logger_base):
    """
    Parameters:
        input_matrix        (np.ndarray)        : Input data
        debug               (dict)              : Algorithm steps output
        ev_config           (dict)              : EV config
        logger_base         (dict)              : Dictionary containing the logger object and logging dict

    Returns:
        input_data          (np.ndarray)        : Updated input data
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('remove_baseload')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the params from config

    window_hours = ev_config['baseload_window']
    sampling_rate = ev_config['sampling_rate']

    # Taking copies of input matrix for new input and baseload data

    input_data = deepcopy(input_matrix)
    baseload_data = deepcopy(input_matrix)

    # Finding number of data points required for the given window size

    window_size = window_hours * Cgbdisagg.SEC_IN_HOUR / sampling_rate
    window_size = np.floor(window_size / 2) * 2 + 1

    logger.debug('Window size used for baseload removal | {}'.format(window_size))

    # Passing a minimum filter over consumption data

    base_load = rolling_function(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], window_size, 'min')

    # Passing a maximum filter over the min values calculated in previous step

    base_load = rolling_function(base_load, window_size, 'max')

    # Save base-load to debug object for future reference

    baseload_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = base_load
    debug['baseload_data'] = baseload_data

    # Base-load removed from the input data

    input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.fmax(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - base_load,
                                                             0)

    debug['input_after_baseload'] = deepcopy(input_data)

    return input_data, debug
