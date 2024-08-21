"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Call the Electric Vehicle disaggregation module after all the data checks
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.disaggregation.aer.ev.ev_disagg import ev_disagg

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.ev.functions.ev_utils import get_bill_cycle_info

from python3.disaggregation.aer.ev.functions.ev_hsm_utils import check_hsm_validity
from python3.disaggregation.aer.ev.functions.ev_hsm_utils import make_hsm_from_debug

from python3.disaggregation.aer.ev.functions.ev_data_checks import check_downsampling
from python3.disaggregation.aer.ev.functions.ev_data_checks import check_number_days_validity


def ev_module(raw_data, ev_config, global_config, ev_present, debug, exit_status, error_list, logger_base):
    """
        Wrapper function over the Electric Vehicle disaggregation module

        Parameters:
            raw_data        (np.ndarray)        : Raw data input for the user
            ev_config        (dict)              : The configuration for the algorithm
            global_config    (dict)              : The configuration for the user
            ev_present      (bool)              : Boolean for app profile
            debug           (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
            exit_status     (dict)              : Has info about code failures
            error_list      (list)              : List of known errors detected during code run
            logger_base     (logger)            : Logging object to log important steps and values in the run

        Returns:
            ev_usage        (np.ndarray)        : Monthly level consumption of EV
            debug           (object)            : Object containing all important data/values as well as HSM
            hsm_out         (dict)              : The updated HSM to be saved
            exit_status     (dict)              : The error code of the run
        """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('ev_module')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Calculating sampling rate (if Not present)

    sampling_rate = global_config.get('sampling_rate')

    # #---------------------------------- Checks for data and hsm validity --------------------------------------------#

    # Down-sampling data if sampling rate < 900

    input_data = check_downsampling(raw_data, sampling_rate, ev_config, logger_pass)

    # Check validity of the hsm

    hsm_is_valid = check_hsm_validity(debug, logger_pass)

    # Check number of days required for historical mode

    n_days_is_valid, data_num_days = check_number_days_validity(input_data, global_config, ev_config)

    # Get unique bill cycle timestamps and their indices

    debug = get_bill_cycle_info(input_data, debug)

    bill_cycle_ts = debug['bill_cycle_ts']
    bill_cycle_idx = debug['bill_cycle_idx']

    # Retrieve GT information from debug object

    gt_data = debug.get('ev_gt')

    # #------------------------------ Run electric vehicle disagg if checks passed ---------------------------------#

    if ev_present and n_days_is_valid and hsm_is_valid:
        # Take only the non-NaN energy values for the algorithm

        input_data_ev = deepcopy(input_data[~np.isnan(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]), :])

        # If EV ground truth is present, compute output at bill cycle level

        if (gt_data is not None) and (gt_data.shape[0] > 0):
            # If GT present, aggregate at bill cycle level

            logger.info('EV ground truth present for the user | ')

            bill_cycle_consumption = np.bincount(bill_cycle_idx, gt_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
            ev_usage = np.c_[bill_cycle_ts, bill_cycle_consumption]

        else:
            # If no ground truth, call the EV module

            logger.info('No EV ground truth present for the user | ')

            ev_usage, debug, error_list = ev_disagg(input_data_ev, ev_config, global_config, debug, error_list,
                                                    logger_pass)

        # Make hsm using the parameters learned by the module

        hsm_out, debug, error_list = make_hsm_from_debug(input_data, debug, ev_config, error_list, logger)
    else:
        # Write logs if EV module is not run

        logger.info('NOT running EV module | ')

        hsm_out = None

        # Make default EV output filled with zeros

        ev_usage = np.c_[bill_cycle_ts, np.zeros(shape=(len(bill_cycle_ts)))]

        # Log possible reasons for not running module

        logger.info('The data_num_days is | {}'.format(data_num_days))
        logger.info('The ev_present flag is | {}'.format(ev_present))
        logger.info('The hsm validity flag is | {}'.format(hsm_is_valid))

    # Check if any error encountered during the run

    if len(error_list) > 0:
        logger.warning('Handled failures received in the code run | ')
        exit_status['exit_code'] = 0
    else:
        exit_status['exit_code'] = 1

    return ev_usage, debug, hsm_out, exit_status
