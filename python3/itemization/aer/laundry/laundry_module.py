"""
Author - Nisha Agarwal
Date - 8th Oct 20
Master file for estimating laundry hybrid capacity
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.laundry.functions.laundry_detection import detect_laundry
from python3.itemization.aer.laundry.functions.get_laundry_delta import get_laundry_delta
from python3.itemization.aer.laundry.functions.get_laundry_consumption import get_laundry_estimate


def get_monthly_estimate(month_ts, data_est):

    """ Return monthly estimate

    This function generates estimated usage per month

    Input:
      month_ts (double matrix)    : 2d matrix with month timestamps
      data_est (double matrix)    : 2d matrix containing laundry estimates

    Output:
      monthly_Estimate (double matrix) : Matrix with month timestamp and estimate
    """

    # preparing list of billing cycles of the user

    col_size = month_ts.size
    month_ts_col = np.reshape(month_ts, [col_size, 1], order='F').copy()
    data_est_col = np.reshape(data_est, [col_size, 1], order='F').copy()

    val_indices = ~np.isnan(month_ts_col)

    month_ts_col = month_ts_col[val_indices]
    data_est_col = data_est_col[val_indices]

    ts, _, idx = np.unique(month_ts_col, return_index=True, return_inverse=True)

    # aggregating billing cycle level appliance cons of the user

    monthly_estimate = np.bincount(idx, weights=data_est_col)
    dop = np.zeros(shape=ts.shape)
    monthly_estimate = np.c_[ts, monthly_estimate, dop, dop, dop]

    return monthly_estimate


def push_laundry_hsm_attributes(ld_hsm, ld_detection):

    """
    Push Laundry HSM Attributes

    Parameters:
        ld_hsm                (dict)      : Laundry HSM
        ld_detection          (int)       : laundry detection tag

    Returns:
        ld_hsm                (dict)      : Laundry HSM
    """

    if ld_hsm is not None:

        ld_hsm.update({
            'ld_present':ld_detection
        })

    else:
        ld_hsm = {
            'ld_present': ld_detection
        }

    return ld_hsm


def run_laundry_module(item_input_object, item_output_object, disagg_mode, hsm_in, logger_pass):

    """
    Master function for laundry modules

    Parameters:
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        disagg_mode                 (str)       : disagg mode
        hsm_in                      (dict)      : HSM data
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        hsm_in                      (dict)      : modified hsm dictionary
        monthly_estimate            (np.ndarray) : Monthly level estimate of laundry
        laundry_ts_estimate         (np.ndarray) : Time stamp level estimate of laundry
    """

    logger_base = logger_pass.get('logger_base').getChild('run_laundry_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    logger.debug("Start of laundry hybrid module")

    # Fetch required input parameters

    t_laundry_module_start = datetime.now()

    activity_curve = item_input_object.get("activity_curve")
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")
    weekend_energy_profile = item_output_object.get("energy_profile").get("weekend_energy_profile")
    weekday_energy_profile = item_output_object.get("energy_profile").get("weekday_energy_profile")
    pilot = item_input_object.get("config").get("pilot_id")

    input_data = copy.deepcopy(item_input_object.get("item_input_params").get('day_input_data'))

    input_data[input_data > np.percentile(input_data, 95)] = np.percentile(input_data, 95)

    # prepare energy change of user in each timme slot

    weekday_delta, weekend_delta = get_laundry_delta(item_input_object, pilot, weekday_energy_profile, weekend_energy_profile, samples_per_hour, activity_curve)

    hsm_ld_detection = 1

    # checking HSM info of laundry app

    hsm_info_absent = hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0 or hsm_in.get('attributes') is None

    if hsm_info_absent:
        valid_hsm = False
    else:
        hsm_ld_detection = hsm_in.get('attributes').get('ld_present')

        valid_hsm = (hsm_ld_detection is not None)

        if isinstance(hsm_ld_detection, list):
            hsm_ld_detection = hsm_ld_detection[0]

    if not valid_hsm:
        logger.warning('Valid HSM not present, running historical mode')

    ts_list = item_input_object.get("item_input_params").get("ts_list")

    # Run the hsm based output

    if valid_hsm and (disagg_mode == "incremental" or disagg_mode == "mtd"):
        laundry_detection = hsm_ld_detection
        logger.info("Laundry detection using HSM | ")

    else:

        # HSM information not present, running laundry detection module

        laundry_detection = detect_laundry(item_input_object, item_output_object, weekday_delta, weekend_delta, logger_pass)

        hsm_in = {'timestamp':  ts_list[-1],
                  'attributes': push_laundry_hsm_attributes(hsm_in, int(laundry_detection))}

        logger.info("Laundry detection calculated in historical mode | ")

    logger.info("Laundry detection | %s", laundry_detection)

    # Estimate laundry

    if laundry_detection:
        laundry_estimate, dishwasher_cons = \
            get_laundry_estimate(item_input_object, item_output_object, input_data, weekday_delta, weekend_delta, logger)
    else:
        laundry_estimate = np.zeros(input_data.shape)
        dishwasher_cons = np.zeros_like(input_data)

    debug = dict()

    month_ts = item_input_object.get("item_input_params").get("month_ts")

    monthly_estimate = get_monthly_estimate(month_ts, laundry_estimate)

    debug.update({
        "monthly_estimate": monthly_estimate
    })

    shape = laundry_estimate.shape[0] * laundry_estimate.shape[1]
    ts_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_EPOCH_IDX, :, :]

    laundry_ts_estimate = np.vstack((np.reshape(ts_list, shape), np.reshape(laundry_estimate, shape))).T

    # preparing debug dict

    logger.debug("End of laundry hybrid module")

    avg_capacity = laundry_estimate.sum(axis=1)
    avg_capacity = avg_capacity[avg_capacity > 0]
    avg_capacity = np.mean(avg_capacity)

    debug.update({
        "laundry_ts_estimate": laundry_ts_estimate,
        "avg_consumption": avg_capacity,
        "weekday_delta": weekday_delta,
        "dish_washer_cons": dishwasher_cons
    })

    item_output_object.get('debug').update({
        "laundry_module_dict": debug
    })

    t_laundry_module_end = datetime.now()

    logger.info("Running main laundry module took | %.3f s",
                get_time_diff(t_laundry_module_start, t_laundry_module_end))

    logger.info("Average daily laundry consumption | %.2f ", avg_capacity)

    return item_input_object, item_output_object, hsm_in, monthly_estimate, laundry_ts_estimate
