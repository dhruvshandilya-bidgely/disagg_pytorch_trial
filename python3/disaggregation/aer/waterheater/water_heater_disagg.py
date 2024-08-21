"""
Author - Nikhil Singh Chauhan
Date - 16/10/18
This file contains the main function that calls the Water Heater algorithm and get the monthly, epoch level estimate
for Timed/Non-timed storage water heater
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.water_heater_module import water_heater_module

from python3.disaggregation.aer.waterheater.functions.water_heater_utils import get_bill_cycle_info

from python3.disaggregation.aer.waterheater.functions.thermostat.functions.hsm_utils import check_hsm_validity
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.hsm_utils import make_hsm_from_debug

from python3.disaggregation.aer.waterheater.functions.waterheater_data_checks import check_downsampling
from python3.disaggregation.aer.waterheater.functions.waterheater_data_checks import check_number_days_validity


def water_heater_disagg(input_data, wh_config, global_config, wh_present, debug, exit_status, error_list, logger_base):
    """
    Wrapper function over the water heater disaggregation module

    Parameters:
        input_data      (np.ndarray)        : Raw data input for the user
        wh_config       (dict)              : The configuration for the algorithm
        global_config   (dict)              : The configuration for the user
        wh_present      (bool)              : Boolean for app profile
        debug           (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
        exit_status     (dict)              : Has info about code failures
        error_list      (list)              : List of known errors detected during code run
        logger_base     (logger)            : Logging object to log important steps and values in the run

    Returns:
        wh_usage        (np.ndarray)        : Monthly level consumption of WH
        debug           (object)            : Object containing all important data/values as well as HSM
        hsm_out         (dict)              : The updated HSM to be saved
        exit_status     (dict)              : The error code of the run
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('water_heater_disagg')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Calculating sampling rate (if Not present)

    sampling_rate = global_config.get('sampling_rate')

    # #---------------------------------- Checks for data and hsm validity --------------------------------------------#

    # Down-sampling data if sampling rate < 900

    input_data = check_downsampling(input_data, sampling_rate, wh_config, logger_pass)

    # Check validity of the hsm

    hsm_is_valid = check_hsm_validity(debug, logger_pass)

    # Check number of days required for historical mode

    n_days_is_valid, data_num_days = check_number_days_validity(input_data, global_config, wh_config)

    # Get unique bill cycle timestamps and their indices

    debug = get_bill_cycle_info(input_data, debug)

    bill_cycle_ts = debug['bill_cycle_ts']
    bill_cycle_idx = debug['bill_cycle_idx']

    # Retrieve GT information from debug object

    gt_data = debug.get('waterheater_gt')

    # #--------------------------------- Run water heater disagg if checks passed -------------------------------------#

    if wh_present and n_days_is_valid and hsm_is_valid:
        # Take only the non-NaN energy values for the algorithm

        input_data_wh = deepcopy(input_data[~np.isnan(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]), :])

        # If water heater ground truth is present, compute output at bill cycle level

        if (gt_data is not None) and (gt_data.shape[0] > 0):
            # If GT present, aggregate at bill cycle level

            logger.info('Water heater actual consumption present for the user | ')

            bill_cycle_consumption = np.bincount(bill_cycle_idx, gt_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
            wh_usage = np.c_[bill_cycle_ts, bill_cycle_consumption]

        else:
            # If no ground truth, call the water heater module

            logger.info('No water heater actual consumption present for the user | ')

            wh_usage, debug, error_list = water_heater_module(input_data_wh, wh_config, global_config,
                                                              debug, error_list, logger_pass)

        # Make hsm using the parameters learned by the module

        hsm_out, debug, error_list = make_hsm_from_debug(input_data, debug, wh_config, error_list, logger)
    else:
        # Write logs if water heater module is not run

        logger.info('NOT running water heater module | ')

        hsm_out = None

        # Make default water heater output filled with zeros

        wh_usage = np.c_[bill_cycle_ts, np.zeros(shape=(len(bill_cycle_ts)))]

        # Log possible reasons for not running module

        logger.info('The data_num_days is | {}'.format(data_num_days))
        logger.info('The wh_present flag is | {}'.format(wh_present))
        logger.info('The hsm validity flag is | {}'.format(hsm_is_valid))

    # Check if any error encountered during the run

    if len(error_list) > 0:
        logger.warning('Handled failures received in the code run | ')
        exit_status['exit_code'] = 0
    else:
        exit_status['exit_code'] = 1

    return wh_usage, debug, hsm_out, exit_status
