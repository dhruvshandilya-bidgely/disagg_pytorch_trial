"""
Author - Nikhil Singh Chauhan
Date - 16/10/18
Module with calls for Timed and Non-timed water heater sub-modules
"""

# Import python packages

import logging
import datetime
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants
from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aer.waterheater.functions.save_output import save_output
from python3.disaggregation.aer.waterheater.functions.timed.timed_waterheater_module import timed_waterheater_module
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.timed_wh_utils import default_timed_debug
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.write_monthly_log import write_monthly_log
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_waterheater_module_japan import timed_waterheater_module_japan
from python3.disaggregation.aer.waterheater.functions.thermostat.thermostat_waterheater_module import thermostat_waterheater_module


def water_heater_module(input_data, wh_config, global_config, debug, error_list, logger_base):
    """
    Module for Timed and Non-timed storage water heater

    Parameters:
        input_data          (np.ndarray)            : Input 21-column matrix
        wh_config           (dict)                  : Configuration for the algorithm
        global_config       (dict)                  : Configuration for the user
        debug               (dict)                  : Dictionary containing output of each step
        error_list          (list)                  : The list of handled errors
        logger_base         (logger)                : The logger object

    Returns:
        wh_estimate         (np.ndarray)            : Bill cycle level estimate
        debug               (dict)                  : Output at each algo step
        error_list          (dict)                  : List of handled errors encountered
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('water_heater_module')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Replace NaNs with zeros

    nan_input_idx = np.isnan(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    input_data[nan_input_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    # Extract bill cycle information

    bill_cycle_ts = debug['bill_cycle_ts']
    bill_cycle_idx = debug['bill_cycle_idx']

    # Saving input data to debug object

    debug['input_data'] = input_data

    # Get the pilot id of the user

    pilot_id = global_config.get('pilot_id')

    # Start time of the water heater module

    t0 = datetime.datetime.now()

    # ---------------------------------- Timed water heater module for Japan pilots ---------------------------------#

    if pilot_id in PilotConstants.TIMED_WH_JAPAN_PILOTS:

        new_data, debug, error_list = timed_waterheater_module_japan(input_data, wh_config, global_config, debug,
                                                                     error_list, logger_pass)

    # ---------------------------------- Timed water heater module of European & Australian pilots ------------------#

    if pilot_id in PilotConstants.TIMED_WH_PILOTS:

        new_data, debug, error_list = timed_waterheater_module(input_data, wh_config, global_config, debug,
                                                               error_list, logger_pass)

    # ---------------------------------- For Non-timed wh pilots give default timed values --------------------------#

    if (pilot_id not in PilotConstants.TIMED_WH_JAPAN_PILOTS) and (pilot_id not in PilotConstants.TIMED_WH_PILOTS):

        # Saving default values to the debug object

        timed_wh_signal, debug = default_timed_debug(input_data, debug, False)

        # Add the relevant logs

        logger.info('Timed water heater module not run for this pilot | {}'.format(pilot_id))
        write_monthly_log(timed_wh_signal, global_config.get('disagg_mode'), logger)

    # End time of Timed water heater module

    t1 = datetime.datetime.now()

    debug['timed_runtime'] = get_time_diff(t0, t1)

    # #--------------------------------- Thermostat water heater module -----------------------------------------------#

    if (debug['timed_hld'] != 1) or (pilot_id in PilotConstants.NEW_ZEALAND_PILOTS):
        # Run thermostat water heater module if no timed water heater detected

        if pilot_id in PilotConstants.NEW_ZEALAND_PILOTS:
            logger.info('Running storage non-timed water heater algo since New Zealand pilot | ')
        else:
            logger.info('Running storage non-timed water heater algo since timed water heater absent | ')

        wh_estimate, debug, error_list = thermostat_waterheater_module(input_data, wh_config, debug,
                                                                       error_list, logger_pass)

        if debug['timed_hld'] == 1:
            # Aggregate at the bill cycle level

            total_monthly = np.bincount(bill_cycle_idx, debug['timed_wh_signal'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

            wh_estimate = np.c_[bill_cycle_ts, total_monthly]

            logger.info('Timed water heater consumption aggregated at bill cycle level | ')
        else:
            # If the timed water heater is not found / overwritten

            logger.info('Aggregating the non-timed water heater output | ')

    else:
        logger.info('Not running storage non-timed water heater algo since timed water heater present | ')

        # Aggregate at the bill cycle level

        total_monthly = np.bincount(bill_cycle_idx, debug['timed_wh_signal'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        wh_estimate = np.c_[bill_cycle_ts, total_monthly]

        debug['thermostat_hld'] = 0

    # End time of the Non-timed water heater module

    t2 = datetime.datetime.now()

    debug['thermostat_runtime'] = get_time_diff(t1, t2)

    # #--------------------------------- Saving plots and output files ------------------------------------------------#

    # Retrieve the bill cycle timestamps for which output is to written

    out_bill_cycles = debug['out_bill_cycles']

    # Dump plots and csv files

    save_output(global_config, debug, out_bill_cycles, wh_config['timezone_hours'], logger_pass)

    return wh_estimate, debug, error_list
