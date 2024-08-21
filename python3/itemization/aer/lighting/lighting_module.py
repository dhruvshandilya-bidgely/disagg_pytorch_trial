
"""
Author - Nisha Agarwal
Date - 8th Oct 20
Master file for estimating lighting hybrid capacity
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.lighting.functions.lighting_utils import remove_daily_min
from python3.itemization.aer.lighting.functions.lighting_utils import check_to_remove_min
from python3.itemization.aer.lighting.functions.lighting_utils import postprocess_sleep_hours

from python3.itemization.aer.lighting.config.init_lighting_config import init_lighting_config

from python3.itemization.aer.lighting.functions.get_top_clean_days import get_top_clean_days
from python3.itemization.aer.lighting.functions.get_lighting_usage_hours import get_lighting_hours
from python3.itemization.aer.lighting.functions.get_lighting_capacity import get_lighting_capacity
from python3.itemization.aer.lighting.functions.get_lighting_estimation import get_lighting_estimate
from python3.itemization.aer.lighting.functions.post_process_lighting_output import smoothen_lighting_hours
from python3.itemization.aer.lighting.functions.post_process_lighting_output import post_process_lighting_tou
from python3.itemization.aer.lighting.functions.prepare_sunrise_sunset_data import preprare_sunrise_sunset_data


def get_monthly_estimate(month_ts, data_est):

    """

    GET MONTHLY ESTIMATE generates estimated usage per month

    Parameters:
      month_ts          (np.ndarray)    : 2d matrix with month timestamps
      data_est          (np.ndarray)    : 2d matrix containing lighting estimates

    Returns:
      monthly_estimate (np.ndarray)     : Matrix with month timestamp and estimate
    """

    col_size = month_ts.size
    month_ts_col = np.reshape(month_ts, [col_size, 1], order='F').copy()
    data_est_col = np.reshape(data_est, [col_size, 1], order='F').copy()

    val_indices = ~np.isnan(month_ts_col)

    month_ts_col = month_ts_col[val_indices]
    data_est_col = data_est_col[val_indices]

    ts, _, idx = np.unique(month_ts_col, return_index=True, return_inverse=True)

    monthly_estimate = np.bincount(idx, weights=data_est_col)
    dop = np.zeros(shape=ts.shape)
    monthly_estimate = np.c_[ts, monthly_estimate, dop, dop, dop]

    return monthly_estimate


def push_lighting_hsm_attributes(results):

    """Push Lighting HSM Attributes

    Parameters:
      results           (dict)       : prepared lighting results

    Returns:
      attributes        (dict)       : prepared lighting hsm

    """

    attributes = {
        'sleep_hours': results.get('sleep_hours').astype(int),
        'lighting_capacity': results.get('lighting_capacity'),
    }

    return attributes


def extract_hsm(hsm_in):

    """
    Utility to pull out HSM attributes

    Parameters:
      hsm_in             (dict)    : received lighting hsm

    Returns:
      hsm_temp           (dict)    : fetched required lighting hsm attributes
    """

    # Extract attributes from hsm for hsm run

    is_hsm_empty = (hsm_in is None) or (len(hsm_in) == 0)

    hsm_attr = None

    # Extract attributes if possible

    if not is_hsm_empty:
        hsm_attr = hsm_in.get('attributes')

    # Extract parameters from within hsm

    hsm_temp = None

    if hsm_attr is not None and len(hsm_attr) > 0:

        hsm_temp = {
            'sleep_hours': np.array(hsm_attr.get('sleep_hours')).astype(int),
            'lighting_capacity': float(hsm_attr.get('lighting_capacity')[0])
        }

    return hsm_temp


def run_lighting_module(item_input_object, item_output_object, disagg_mode, hsm_in, logger_pass):

    """
    Master function for lighting modules - which calculates the lighting usage hours and ts level capacity
    using either hsm inputs or prepared activity profile

    Parameters:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        disagg_mode                 (str)       : disagg mode
        hsm_in                      (dict)      : HSM data
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
    """

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('run_lighting_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    logger.debug("Start of lighting hybrid module")

    # Fetch required input parameters

    ts_list = item_input_object.get("item_input_params").get("ts_list")
    active_hours = item_output_object.get("profile_attributes").get("sleep_hours")
    vacation = item_input_object.get("item_input_params").get("vacation_data")
    input_data = item_input_object.get("item_input_params").get("day_input_data")
    clean_days_score = item_input_object.get("clean_day_score_object").get("clean_day_score")
    samples_per_hour = int(item_input_object.get("item_input_params").get("samples_per_hour"))

    vacation = np.sum(vacation, axis=1).astype(bool)

    scale_factor = 1

    # initialize debug object

    debug = dict()

    t_lighting_module_start = datetime.now()

    lighting_config = init_lighting_config(samples_per_hour)

    # preprocess and prepare sunrise sunset data

    sunrise_sunset_data = preprare_sunrise_sunset_data(item_input_object,  lighting_config, logger_pass)

    debug.update({
        "sunrise_sunset_data": sunrise_sunset_data
    })

    # check if valid hsm attributes are present

    if hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0 or hsm_in.get('attributes') is None:
        valid_hsm = False
    else:
        hsm_active_hours = hsm_in.get('attributes').get('sleep_hours')
        hsm_lighting_capacity = hsm_in.get('attributes').get('lighting_capacity')

        valid_hsm = (hsm_active_hours is not None) and (hsm_lighting_capacity is not None) and (np.sum(hsm_active_hours) != 0)

        if valid_hsm and samples_per_hour*Cgbdisagg.HRS_IN_DAY != len(hsm_active_hours):
            valid_hsm = False
            logger.warning('Sampling rate of HSM data is different from the current run')

    if not valid_hsm:
        logger.warning('Valid HSM not present, running historical mode')

    # Run the hsm based output

    if valid_hsm and (disagg_mode == "incremental" or disagg_mode == "mtd"):

        logger.info('Entering HSM usage mode |')

        # Extract parameters out of HSM for the run

        active_hours = np.array(hsm_active_hours)

        if isinstance(hsm_lighting_capacity, list):
            lighting_capacity = int(hsm_lighting_capacity[0])
        else:
            lighting_capacity = int(hsm_lighting_capacity)

        # Calculate lighting hours and lighting potential

        debug.update({
            "lighting_capacity": lighting_capacity*samples_per_hour,
        })

        lighting_potential, sunrise_sunset_potential, debug = \
            get_lighting_hours(item_input_object, input_data, active_hours, sunrise_sunset_data, lighting_config, debug,
                               logger_pass)

        # Smooth lighting potential

        lighting_estimate_potential = \
            smoothen_lighting_hours(item_input_object, lighting_potential, sunrise_sunset_data, lighting_config, logger_pass)

        debug.update({
            "lighting_potential_after_smoothing": lighting_estimate_potential
        })

        logger.info('HSM usage mode completed |')

    # Now, run fresh lighting for historical mode or when hsm is absent

    else:

        logger.info('Entering HSM creation mode |')

        # remove non zero min consumption from input data

        remove_min = check_to_remove_min(clean_days_score, vacation, lighting_config)

        logger.info("Remove minimum consumption | %s", bool(remove_min))

        if remove_min:
            input_data = remove_daily_min(input_data)

        # Calculate lighting usage potential

        lighting_potential, sunrise_sunset_potential, debug = \
            get_lighting_hours(item_input_object, input_data, active_hours, sunrise_sunset_data,
                               lighting_config, debug, logger_pass)

        debug.update({
            "lighting_potential_before_smoothing": lighting_potential
        })

        # Smooth lighting potential

        lighting_estimate_potential = \
            smoothen_lighting_hours(item_input_object, lighting_potential, sunrise_sunset_data, lighting_config, logger_pass)

        debug.update({
            "lighting_potential_after_smoothing": lighting_estimate_potential
        })

        # Fetch top cleanest days for lighting estimation

        top_clean_days = get_top_clean_days(item_input_object, lighting_config, logger_pass)

        # Get lighting capacity

        lighting_capacity, debug, scale_factor = \
            get_lighting_capacity(item_input_object, item_output_object, top_clean_days, lighting_config, debug, logger_pass)

        debug.update({
            "lighting_capacity": lighting_capacity*samples_per_hour,
            "top_cleanest_days": len(top_clean_days)
        })

        logger.info('HSM creation mode completed |')

        postprocess_sleep_hours(active_hours, lighting_config)

        # prepare HSM attributes

        results = {
            'sleep_hours': active_hours,
            'lighting_capacity': lighting_capacity
        }
        hsm_in = {
            'timestamp':  ts_list[-1],
            'attributes': push_lighting_hsm_attributes(results),
        }
        logger.info('Created lighting HSM is | %s', str(hsm_in).replace('\n', ' '))

    # calculate tou level lighting estimate

    lighting_hours_count = lighting_estimate_potential > 0
    lighting_hours_count = np.sum(lighting_hours_count, axis=1)
    lighting_hours_count = lighting_hours_count[lighting_hours_count > 0]
    lighting_hours_count = np.mean(lighting_hours_count)

    # calculate tou lighting usage

    lighting_estimate = \
        get_lighting_estimate(item_input_object, lighting_estimate_potential, lighting_capacity, samples_per_hour,
                              sunrise_sunset_potential, lighting_config, logger_pass)

    debug.update({
        "lighting_day_estimate": lighting_estimate
    })

    # post process of lighting estimate

    lighting_estimate = \
        post_process_lighting_tou(item_input_object, lighting_estimate, sunrise_sunset_data, lighting_config,
                                  logger_pass)

    debug.update({
        "lighting_day_estimate_after_postprocess": lighting_estimate
    })

    lighting_output = {
        "lighting_estimate": lighting_estimate,
        "sunrise_sunset_data": sunrise_sunset_data
    }

    item_output_object.update({
        "lighting_output": lighting_output
    })

    month_ts = item_input_object.get("item_input_params").get("month_ts")

    # prepare monthly estimate

    monthly_estimate = get_monthly_estimate(month_ts, lighting_estimate)

    debug.update({
        "monthly_estimate": monthly_estimate
    })

    shape = lighting_estimate.shape[0] * lighting_estimate.shape[1]
    ts_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_EPOCH_IDX, :, :]

    lighting_ts_estimate = np.vstack((np.reshape(ts_list, shape), np.reshape(lighting_estimate, shape))).T

    t_lighting_module_end = datetime.now()

    logger.debug("End of lighting hybrid module")

    logger.info("Running main lighting module took | %.3f s",
                get_time_diff(t_lighting_module_start, t_lighting_module_end))

    # calculate average lighting consumption

    avg_capacity = lighting_estimate.sum(axis=1)
    avg_capacity = avg_capacity[avg_capacity > 0]
    avg_capacity = np.mean(avg_capacity)

    # prepare debug object

    debug.update({
        "lighting_ts_estimate": lighting_ts_estimate,
        "avg_lighting_hours": lighting_hours_count,
        "avg_consumption": avg_capacity,
        'scale_factor': scale_factor
    })

    item_output_object.get('debug').update({
        "lighting_module_dict": debug
    })

    logger.info("Average lighting usage hours | %.2f ", (lighting_hours_count/samples_per_hour))
    logger.info("Average daily lighting consumption | %.2f ", avg_capacity)

    return item_input_object, item_output_object, hsm_in, monthly_estimate, lighting_ts_estimate
