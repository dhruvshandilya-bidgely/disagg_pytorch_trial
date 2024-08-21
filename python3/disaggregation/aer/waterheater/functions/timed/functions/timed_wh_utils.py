"""
Author - Nikhil Singh Chauhan
Date - 02/11/18
Handy functions for timed water heater module
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def check_rounding(in_data, timed_config, logger):
    """
    Check if energy data is rounded (for Japan)

    Parameters:
        in_data         (np.ndarray)    : Input 21-column matrix
        timed_config    (dict)          : Timed config params
        logger          (logger)        : Logger object

    Returns:
        timed_config    (dict)          : Updated timed config params
    """

    # Taking deepcopy of input data and consumption values

    input_data = deepcopy(in_data)
    energy = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Extract the energy threshold for rounding check

    min_rounding_bar = timed_config['min_rounding_bar']
    rounding_balance = timed_config['rounding_balance']

    # Check remainder with minimum rounding bar

    if np.sum(energy % min_rounding_bar) < rounding_balance:
        # If rounding present, update the config

        timed_config['std_thres'] -= timed_config['std_thres_delta']

        logger.info('Rounding present at 100 Watt level | ')

    return timed_config


def default_timed_debug(input_data, debug, valid_pilot):
    """
    Parameters:
        input_data              (np.ndarray)    : Input 21-column matrix
        debug                   (dict)          : Module intermediate stage output
        valid_pilot             (bool)          : Check if valid pilot to run timed wh

    Returns:
        timed_wh_signal         (np.ndarray)    : Timed water heater output
        debug                   (dict)          : Updated debug object
    """
    timed_wh_signal = deepcopy(input_data)
    timed_wh_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    # Saving default values to the debug object

    debug['timed_wh_signal'] = timed_wh_signal[:, :Cgbdisagg.INPUT_DIMENSION]

    debug['timed_hld'] = 0
    debug['timed_wh_amplitude'] = 0

    # Define the default timed debug object

    debug['timed_debug'] = {
        'valid_pilot': valid_pilot,
        'box_data': timed_wh_signal[:, :Cgbdisagg.INPUT_DIMENSION],
        'timed_confidence': 0
    }

    return timed_wh_signal, debug
