"""
Author - Nikhil Singh Chauhan
Date - 16/10/18
This is the module which takes the input data and HSM(if present) as input and
return the epoch level estimate of water heater consumption
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.waterheater.functions.remove_baseload import remove_baseload
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.post_process import post_process
from python3.disaggregation.aer.waterheater.functions.thermostat.thermostat_waterheater_estimation import get_seasonal_estimation
from python3.disaggregation.aer.waterheater.functions.thermostat.thermostat_waterheater_detection import thermostat_waterheater_detection
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.overall_consumption_check import check_final_consumption_percentage


def thermostat_waterheater_module(input_data_original, wh_config, debug, error_list, logger_base):
    """
    Parameters:
        input_data_original     (np.ndarray)    : input data
        wh_config               (dict)          : the water heater params
        debug                   (dict)          : The object to save all important data and values
        error_list              (list)          : List of known errors found during the code run
        logger_base             (logger)        : Logger object to save important logs during the run

    Returns:
        wh_estimate             (np.ndarray)    : Monthly level Water Heater consumption
        debug                   (dict)          : The updated object with epoch level estimation and HSM
        error_list              (dict)          : Updated error list
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('thermostat_waterheater_module')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking a deepcopy of input data to keep local instances

    input_data = deepcopy(input_data_original)

    # Get config params from wh_config

    sampling_rate = wh_config.get('sampling_rate')
    thermostat_config = wh_config.get('thermostat_wh')

    # Remove baseload from input data

    input_data, debug = remove_baseload(input_data, debug, thermostat_config['baseload_window'], sampling_rate,
                                        'thermostat', logger_pass)

    # #--------------------------------------------- Detection --------------------------------------------------------#
    # Call home level detection module

    debug = thermostat_waterheater_detection(input_data, debug, wh_config, logger_pass)

    # Check detection before moving on to estimation

    if (debug.get('thermostat_hld') is None) or (debug['thermostat_hld'] == 0):
        # If no detection of thermostat water heater, return default zeros

        # Residual is equal to input data

        residual = deepcopy(input_data_original)

        # Water heater output made zero

        final_output = deepcopy(input_data_original)
        final_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        # Saving default zero values for thin, fat, total output

        debug['final_thin_output'] = final_output
        debug['final_fat_output'] = final_output
        debug['final_wh_signal'] = final_output

        debug['residual'] = residual

        # Aggregate consumption at bill cycle level is zero

        bill_cycle_ts = debug['bill_cycle_ts']

        wh_estimate = np.c_[bill_cycle_ts, np.zeros(shape=(len(bill_cycle_ts),))]

        # Add the output to debug object and skip estimation

        debug['wh_estimate'] = wh_estimate

        logger.info('No estimation since water heater not detected | ')

        return wh_estimate, debug, error_list

    # #--------------------------------------------- Estimation -------------------------------------------------------#
    # If water heater detected, get consumption for each season

    debug = get_seasonal_estimation(debug, wh_config, logger_pass)

    # Post processing steps

    debug = post_process(debug, wh_config, logger_pass)

    # Check if water heater consumption sufficient enough

    debug = check_final_consumption_percentage(debug, wh_config, logger_pass)

    # Retrieve the final bill cycle level water heater consumption

    wh_estimate = debug['wh_estimate']

    return wh_estimate, debug, error_list
