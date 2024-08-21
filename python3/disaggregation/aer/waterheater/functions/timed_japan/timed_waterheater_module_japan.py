"""
Author - Sahana M
Date - 10/10/2018
The module detects and returns the estimated consumption of a timed water heater
"""

# Import python packages

import logging
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.waterheater.functions.remove_baseload import remove_baseload
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.timed_wh_utils import default_timed_debug
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.write_monthly_log import write_monthly_log
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.get_pool_pump_output import get_pool_pump_output
from python3.disaggregation.aer.waterheater.functions.timed_japan.twh_detection_module import timed_waterheater_detection


def timed_waterheater_module_japan(input_data, wh_config, global_config, debug, error_list, logger_base):
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

    # Reading hsm for MTD mode

    hsm_in = debug.get('hsm_in')

    # #------------------------ Check if timed water heater absent from HSM (only for MTD mode)----------------------- #

    if (hsm_in is not None) and (hsm_in.get('attributes') is not None) and (hsm_in.get('attributes').get('timed_hld') == 0):

        # Saving default values to the debug object

        timed_wh_signal, debug = default_timed_debug(input_data, debug, True)

        # Add the relevant logs

        logger.info('Timed water heater module not run since hld | 0')
        write_monthly_log(timed_wh_signal, global_config.get('disagg_mode'), logger)

        return in_data, debug, error_list

    # #-------------------------- Running timed water heater module for the user --------------------------------------#

    logger.info('Running Japan timed water heater module for this pilot | {}'.format(global_config.get('pilot_id')))

    # Load timed water heater config from overall config

    timed_config = wh_config['timed_wh']

    # Add sampling rate to the timed config

    sampling_rate = global_config.get('sampling_rate')
    timed_config['sampling_rate'] = sampling_rate

    # Remove pool pump based on significance score

    in_data = get_pool_pump_output(in_data, debug, timed_config, logger_pass)

    # Initialize debug object for timed water heater module

    timed_debug = deepcopy(debug)

    # Remove baseload (moving min-max in 8 hours window)

    in_data, debug = remove_baseload(in_data, debug, 16, sampling_rate,
                                     'timed', logger_pass)

    # Timed WH Detection & Estimation

    timed_wh_signal, timed_debug = timed_waterheater_detection(in_data, timed_debug, sampling_rate, logger_pass)

    # Add the relevant logs

    logger.info('Final timed water heater detection | {}'.format(timed_debug['timed_hld']))

    write_monthly_log(timed_wh_signal, global_config.get('disagg_mode'), logger)

    # Restore hour of day from the original input matrix

    timed_wh_signal[:, Cgbdisagg.INPUT_HOD_IDX] = deepcopy(input_data[:, Cgbdisagg.INPUT_HOD_IDX])

    # Saving amplitude and hld to the main debug object

    debug['timed_debug'] = timed_debug
    debug['timed_hld'] = timed_debug.get('timed_hld')

    # Save the timed water heater consumption and amplitude to debug

    debug['timed_wh_signal'] = timed_wh_signal[:, :Cgbdisagg.INPUT_DIMENSION]
    debug['timed_wh_amplitude'] = timed_debug.get('timed_wh_amplitude')
    debug['timed_confidence_score'] = timed_debug.get('timed_confidence')

    # Save modified data matrix with timed water heater consumption removed from energy consumption

    residual = deepcopy(new_data)

    if timed_debug['timed_hld'] == 1:
        new_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= timed_wh_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
        residual = new_data

    debug['timed_debug']['twh_residual'] = residual

    return new_data, debug, error_list
