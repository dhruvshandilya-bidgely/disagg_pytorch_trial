"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to remove timed appliances from input data
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def remove_pool_pump(in_data, debug, ev_config, logger_base):
    """
    Function to remove pool pump output from input data

    Parameters:
        in_data             (np.ndarray)            : Input matrix
        debug               (dict)                  : Dictionary containing output of each step
        logger_base         (logger)                : The logger object
        ev_config            (dict)                  : Configuration for the algorithm

    Returns:
        input_data          (np.ndarray)            : Updated input matrix
    """
    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('remove_pool_pump')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking local copy of input data

    input_data = deepcopy(in_data)

    # Retrieve the pool pump info

    pp_output = debug['other_output']['pp']
    pp_confidence = debug['other_output']['pp_confidence']

    if pp_confidence is not None:

        # Remove pool pump estimated output if detection confidence is above threshold

        if pp_confidence >= ev_config['pp_removal_conf']:

            input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= pp_output[:, 1]

            input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.fmax(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], 0)

            logger.info('Pool pump output removed from input data | ')
        else:
            logger.info('Pool pump confidence not enough | {}'.format(pp_confidence))
    else:
        logger.info('Pool pump output not present | ')

    return input_data


def remove_timed_wh(in_data, debug, logger_base):
    """
    Function to remove timed water heater output from input data

    Parameters:
        in_data             (np.ndarray)            : Input matrix
        debug               (dict)                  : Dictionary containing output of each step
        logger_base         (logger)                : The logger object

    Returns:
        input_data          (np.ndarray)            : Updated input matrix
    """
    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('remove_timed_wh')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking local copy of input data

    input_data = deepcopy(in_data)

    # Retrieve the timed water heater output

    timed_wh_output = debug['other_output']['timed_wh']

    if timed_wh_output is not None:
        # If timed water heater output present
        if np.nansum(timed_wh_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]) > 0:

            input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= timed_wh_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

            input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.fmax(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], 0)

            logger.info('Timed water heater output removed from input data | ')
        else:
            logger.info('Timed water heater output not present | ')

    return input_data
