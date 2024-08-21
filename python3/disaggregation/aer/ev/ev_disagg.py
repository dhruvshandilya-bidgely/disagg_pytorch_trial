"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module with calls for Electric Vehicle sub module
"""

# Import python packages

import logging
import datetime
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aer.ev.functions.ev_utils import cap_input_data
from python3.disaggregation.aer.ev.functions.ev_detection import ev_detection
from python3.disaggregation.aer.ev.functions.save_ev_output import save_ev_output
from python3.disaggregation.aer.ev.functions.ev_estimation import estimate_ev_consumption
from python3.disaggregation.aer.ev.functions.ev_l1.ev_l1_detection import ev_l1_detection
from python3.disaggregation.aer.ev.functions.ev_l1.init_ev_l1_params import init_ev_l1_params
from python3.disaggregation.aer.ev.functions.detection.data_sanity_checks import final_hld_sanity_checks
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_day_data_2d
from python3.disaggregation.aer.ev.functions.deep_learning.refine_detection_info import combine_detection_info
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_detection import deeplearning_detection
from python3.disaggregation.aer.ev.functions.detection.remove_timed_appliances import remove_pool_pump, remove_timed_wh


def ev_disagg(in_data, ev_config, global_config, debug, error_list, logger_base):
    """
    Module for EV disaggregation

    Parameters:
        in_data          (np.ndarray)            : Input 13-column matrix
        ev_config           (dict)                  : Configuration for the algorithm
        global_config       (dict)                  : Configuration for the user
        debug               (dict)                  : Dictionary containing output of each step
        error_list          (list)                  : The list of handled errors
        logger_base         (logger)                : The logger object

    Returns:
        ev_estimate         (np.ndarray)            : Bill cycle level estimate
        debug               (dict)                  : Output at each algo step
        error_list          (dict)                  : List of handled errors encountered
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('ev_disagg')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Replace NaNs with zeros
    input_data = deepcopy(in_data)

    nan_input_idx = np.isnan(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    input_data[nan_input_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    # Saving the original data before any pre-processing

    debug['original_input_data'] = deepcopy(input_data)
    debug['uuid'] = ev_config['uuid']
    debug['l1'] = {}
    debug['l2'] = {}

    # Removing Pool pump output from data

    input_data = remove_pool_pump(input_data, debug, ev_config, logger_pass)

    # Remove timed water heater

    input_data = remove_timed_wh(input_data, debug, logger_pass)

    # Extract bill cycle information

    bill_cycle_ts = debug['bill_cycle_ts']
    bill_cycle_idx = debug['bill_cycle_idx']

    # Start time of the EV module

    t_start = datetime.datetime.now()

    # Capping data

    input_data = cap_input_data(input_data, ev_config)

    # Saving input data to debug object

    debug['input_data'] = input_data

    # ---------------------------------- L2 Detection ----------------------------------------------------------------#

    debug, error_list = ev_detection(input_data, ev_config, debug, error_list, logger_pass)

    # ---------------------------------- L1 Detection ----------------------------------------------------------------#

    # Initialise EV L1 configurations

    ev_l1_config = init_ev_l1_params(ev_config)
    debug['l1_config'] = ev_l1_config

    if debug.get('ev_hld') == 0 and debug.get('ev_app_profile_yes') and ev_config.get('disagg_mode') != 'mtd':

        logger.info('EV L2 not detected but app profile is yes, running L1 detection | ')

        debug, error_list = ev_l1_detection(input_data, ev_l1_config, debug, error_list, logger_pass)

    # End time for EV detection

    t_detection = datetime.datetime.now()
    debug['detection_runtime'] = get_time_diff(t_start, t_detection)
    logger.info('Detection time | {}'.format(debug['detection_runtime']))

    # ---------------------------------- EV Deep learning -------------------------------------------------------------#

    if ev_config.get('disagg_mode') != 'mtd':

        input_data = debug.get('hvac_removed_data_l2')
        data_matrices, row_idx, col_idx = get_day_data_2d(input_data, ev_config)

        t_dl_start = datetime.datetime.now()

        debug_partitions = deeplearning_detection(data_matrices, debug, ev_config, row_idx, col_idx, logger_pass)
        debug = combine_detection_info(debug, debug_partitions)

        t_dl_end = datetime.datetime.now()
        logger.info('Deep learning model detection time | %s', get_time_diff(t_dl_start, t_dl_end))

    # ---------------------------------- Estimation ------------------------------------------------------------------#

    negate_detection_status = False

    if debug['ev_hld'] == 1:

        debug, error_list = estimate_ev_consumption(input_data, debug, ev_config, error_list, logger_pass)

        debug, negate_detection_status = final_hld_sanity_checks(debug, negate_detection_status, ev_config, logger_pass)

        # Aggregate at the bill cycle level

        monthly_ev = np.bincount(bill_cycle_idx, debug['final_ev_signal'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        ev_estimate = np.c_[bill_cycle_ts, monthly_ev]

        logger.info('EV consumption aggregated at bill cycle level | ')

    else:
        logger.info('Not running EV estimation since EV hld is zero | ')

        # Aggregate at the bill cycle level

        monthly_ev = np.zeros(shape=(len(bill_cycle_ts),))

        ev_estimate = np.c_[bill_cycle_ts, monthly_ev]

        debug['ev_amplitude'] = 0

        debug['final_ev_signal'] = deepcopy(in_data)
        debug['final_ev_signal'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        debug['residual_data'] = deepcopy(debug['original_input_data'])

    if negate_detection_status:

        logger.info('EV estimation changed to zero since Data sanity issue encountered | ')

        # Aggregate at the bill cycle level

        monthly_ev = np.zeros(shape=(len(bill_cycle_ts),))

        ev_estimate = np.c_[bill_cycle_ts, monthly_ev]

        debug['ev_amplitude'] = 0

        debug['final_ev_signal'] = deepcopy(in_data)
        debug['final_ev_signal'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        debug['residual_data'] = deepcopy(debug['original_input_data'])

    # Cumulative logs

    logger.info('Final EV Detection | {}'.format(debug.get('ev_hld')))
    logger.info('Final EV Probability | {}'.format(debug.get('ev_probability')))
    logger.info('Final EV Charger type detected | {}'.format(debug.get('charger_type')))
    logger.info('Final EV Amplitude | {}'.format(debug.get('ev_amplitude')))
    logger.info('Final EV Duration | {}'.format(debug.get('mean_duration')))

    # End time of the EV module

    t_end = datetime.datetime.now()

    debug['estimation_runtime'] = get_time_diff(t_detection, t_end)
    logger.info('Estimation time | {}'.format(debug['estimation_runtime']))

    debug['ev_runtime'] = get_time_diff(t_start, t_end)

    # #--------------------------------- Saving plots and output files ------------------------------------------------#

    # Retrieve the bill cycle timestamps for which output is to written

    out_bill_cycles = debug.get('out_bill_cycles')

    # Dump plots and csv files

    save_ev_output(global_config, debug, out_bill_cycles, ev_config, logger_pass)

    return ev_estimate, debug, error_list
