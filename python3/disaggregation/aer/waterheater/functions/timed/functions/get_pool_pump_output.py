"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Gather pool pump information to check if need to be removed from input data
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.water_heater_utils import get_day_ts


def get_pool_pump_area(pool_pump_output, factor):
    """
    Calculate the area under curve for pool pump consumption

    Parameters:
        pool_pump_output        (np.ndarray)    : 2-column data (Epoch, pp_energy)
        factor                  (int)           : Time divisions per hour

    Returns:
        pp_area                 (float)         : Pool pump consumption per day
        pp_duration             (float)         : Pool pump run duration per day
    """

    pp_output = deepcopy(pool_pump_output)

    # Get day timestamp from epoch timestamp

    day_ts = list(map(get_day_ts, pp_output[:, 0]))
    pp_output[:, 0] = day_ts

    # Use non-zero pool pump output timestamps to get consumption per day

    pp_idx = pp_output[:, 1] > 0
    num_pp_days = len(np.unique(pp_output[pp_idx, 0]))
    pp_area = np.sum(pp_output[:, 1]) / num_pp_days

    # Use non-zero pool pump output timestamps to get run duration per day

    total_duration = np.sum(pp_idx)
    pp_duration = total_duration / (num_pp_days * factor)

    return pp_area, pp_duration


def get_pool_pump_output(raw_input, debug, timed_config, logger_base):
    """
    Module to remove pool pump output from input data

    Parameters:
        raw_input       (np.ndarray)    : Input 21-column matrix
        debug           (dict)          : Output of Pool pump, AO and vacation
        timed_config    (dict)          : Config params for timed water heater
        logger_base     (logger)        : Logger object to log values

    Returns:
        input_data      (np.ndarray)    : Updated 21-column matrix

    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_pool_pump_output')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Take copy of the raw data

    input_data = deepcopy(raw_input)

    # Get sampling rate from the timed config

    sampling_rate = timed_config.get('sampling_rate')

    # Pool pump columns

    pp_columns = {
        'epoch_ts': 0,
        'energy': 1
    }

    # Other appliance output

    other_output = debug.get('other_output')

    # Read pool pump output and confidence value

    pp_output = other_output.get('pp')
    pp_confidence = other_output.get('pp_confidence')

    # Extract the pool pump thresholds to decide if need to be removed

    pp_thresholds = timed_config['pp_thresholds']

    pp_area_bar = pp_thresholds['pp_area_bar']
    pp_duration_bar = pp_thresholds['pp_duration_bar']
    pp_confidence_bar = pp_thresholds['pp_confidence_bar']

    # Check if valid pool pump output and confidence values

    if (pp_output is not None) and (pp_confidence is not None):
        # If valid pool pump output, find area and duration per day

        factor = Cgbdisagg.SEC_IN_HOUR / sampling_rate
        pp_area, pp_duration = get_pool_pump_area(pp_output, factor)

        # Log relevant pool pump values

        logger.info('Pool pump area | {}'.format(pp_area))
        logger.info('Pool pump duration | {}'.format(pp_duration))
        logger.info('Pool pump confidence | {}'.format(pp_confidence))

        # Check if pool pump good enough to be removed from raw data

        if (pp_confidence >= pp_confidence_bar) and (pp_area > pp_area_bar) and (pp_duration > pp_duration_bar):
            # If pool pump is significant enough, remove it from input data

            input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= pp_output[:, pp_columns['energy']]

            # Making sure that negative values don't occur after removing pool pump

            input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.fmax(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], 0)

            logger.info('Pool pump data removed from the input data | ')
        else:
            logger.info('Pool pump data not removed from the input data | ')
    else:
        logger.info('Pool pump data not found | ')

    return input_data
