"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
The module detects and returns the estimated consumption of a timed water heater
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.waterheater.functions.remove_baseload import remove_baseload
from python3.disaggregation.aer.waterheater.functions.timed.functions.timed_wh_utils import check_rounding
from python3.disaggregation.aer.waterheater.functions.timed.functions.timed_wh_utils import default_timed_debug
from python3.disaggregation.aer.waterheater.functions.timed.functions.write_monthly_log import write_monthly_log
from python3.disaggregation.aer.waterheater.functions.timed.functions.get_pool_pump_output import get_pool_pump_output
from python3.disaggregation.aer.waterheater.functions.timed.functions.hour_to_time_division import hour_to_time_division
from python3.disaggregation.aer.waterheater.functions.timed.timed_waterheater_detection import timed_waterheater_detection


def timed_waterheater_module(input_data, wh_config, global_config, debug, error_list, logger_base):
    """
    The module detects and returns the estimated consumption of timed water heater

    Parameters:
        input_data          (np.ndarray)        : The input 21-column matrix
        wh_config           (dict)              : Configuration for the algorithm
        global_config       (dict)              : Configuration for the user
        debug               (dict)              : Output of algorithm at each step
        error_list          (list)              : The list of handled errors
        logger_base         (dict)              : The logger object to save updates

    Returns:
        new_data            (np.ndarray)        : The updated consumption values
        debug               (dict)              : Output of algorithm at each step
        error_list          (list)              : Timed water heater energy per data point
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('timed_waterheater_module')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking a deepcopy of input data

    in_data = deepcopy(input_data)
    new_data = deepcopy(input_data)

    # Reading hsm for MTD mode and pilot id

    hsm_in = debug.get('hsm_in')

    # #------------------------ Check if timed water heater absent from HSM (only for MTD mode)----------------------- #

    if (hsm_in is not None) and (hsm_in.get('attributes') is not None):
        # If water heater absent, return timed water heater consumption and amplitude as zero

        if hsm_in.get('attributes').get('timed_hld') == 0:
            # Saving default values to the debug object

            timed_wh_signal, debug = default_timed_debug(input_data, debug, True)

            # Add the relevant logs

            logger.info('Timed water heater module not run since hld | 0')
            write_monthly_log(timed_wh_signal, global_config.get('disagg_mode'), logger)

            return in_data, debug, error_list

    # #-------------------------- Running timed water heater module for the user --------------------------------------#

    logger.info('Running timed water heater module for this pilot | {}'.format(global_config.get('pilot_id')))

    # Load timed water heater config from overall config

    timed_config = wh_config['timed_wh']

    # Add sampling rate to the timed config

    sampling_rate = global_config.get('sampling_rate')
    timed_config['sampling_rate'] = sampling_rate

    # Remove pool pump based on significance score

    in_data = get_pool_pump_output(in_data, debug, timed_config, logger_pass)

    # Hour factor based on the sampling rate (factor for sampling rate 900 is: 3600 / 900 = 4)

    factor = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)
    num_hours = Cgbdisagg.HRS_IN_DAY * factor

    # Convert hour of day to time division of the day

    in_data = hour_to_time_division(in_data, sampling_rate)
    in_data[:, Cgbdisagg.INPUT_HOD_IDX] = (in_data[:, Cgbdisagg.INPUT_HOD_IDX] * factor).astype(int)

    # Check if energy consumption is rounded to a multiple of '100' (JAPAN)

    timed_config = check_rounding(in_data, timed_config, logger)

    # Initialize debug object for timed water heater module

    timed_debug = {
        'time_factor': factor,
        'valid_pilot': True
    }

    # Remove baseload (moving min-max in 8 hours window)

    in_data, debug = remove_baseload(in_data, debug, timed_config['baseload_window'], sampling_rate,
                                     'timed', logger_pass)

    # Initialize the features matrix

    features = np.empty(shape=(1, num_hours))

    # Run the timed water heater detection module

    timed_wh_signal, timed_debug = timed_waterheater_detection(in_data, features, timed_config,
                                                               timed_debug, logger_pass)

    # Add the relevant logs

    logger.info('Final timed water heater detection | {}'.format(timed_debug['timed_hld']))

    write_monthly_log(timed_wh_signal, global_config.get('disagg_mode'), logger)

    # Restore hour of day from the original input matrix

    timed_wh_signal[:, Cgbdisagg.INPUT_HOD_IDX] = deepcopy(input_data[:, Cgbdisagg.INPUT_HOD_IDX])

    # Saving amplitude and hld to the main debug object

    debug['timed_debug'] = timed_debug
    debug['timed_hld'] = timed_debug['timed_hld']

    # Save the timed water heater consumption and amplitude to debug

    debug['timed_wh_signal'] = timed_wh_signal[:, :Cgbdisagg.INPUT_DIMENSION]
    debug['timed_wh_amplitude'] = timed_debug['timed_wh_amplitude']
    debug['timed_confidence_score'] = timed_debug['timed_confidence']

    # Save modified data matrix with timed water heater consumption removed from energy consumption

    if timed_debug['timed_hld'] == 1:
        new_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= timed_wh_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    return new_data, debug, error_list
