"""
Author - Nisha Agarwal
Date - 8th Oct 20
Master file for estimating entertainment hybrid capacity
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.entertainment.functions.get_entertainment_consumption import get_entertainment_estimate


def get_monthly_estimate(month_ts, data_est):

    """ Return monthly estimate of the appliance

    This function generates estimated usage per month

    Input:
      month_ts (double matrix)    : 2d matrix with month timestamps
      data_est (double matrix)    : 2d matrix containing entertainment estimates

    Output:
      monthly_Estimate (double matrix) : Matrix with month timestamp and estimate
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


def run_entertainment_module(item_input_object, item_output_object, logger_pass):

    """
    Master function for entertainment modules

    Parameters:
        item_input_object           (dict)           : Dict containing all hybrid inputs
        item_output_object          (dict)           : Dict containing all hybrid outputs
        logger_pass                 (dict)           : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object           (dict)           : Dict containing all hybrid inputs
        item_output_object          (dict)           : Dict containing all hybrid outputs
        monthly_estimate            (np.ndarray)     : monthly level ent estimate
        entertainment_ts_estimate   (np.ndarray)     : timestamp level ent estimate
    """

    logger_base = logger_pass.get('logger_base').getChild('run_entertainment_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    logger.debug("Start of entertainment hybrid module")

    # Fetch required input parameters

    weekday_profile = item_output_object.get("weekday_profile")
    ent = item_input_object.get("appliance_profile").get("ent")
    vacation = item_input_object.get("item_input_params").get("vacation_data")
    user_attributes = item_output_object.get("occupants_profile").get("occupants_features")

    # run module that initializes ts level estimate of entertainment

    entertainment_estimate = \
        get_entertainment_estimate(item_output_object, item_input_object, user_attributes, vacation[:, 0],
                                   ent, weekday_profile, logger_pass)

    debug_object = dict()

    t_entertainment_module_start = datetime.now()

    month_ts = item_input_object.get("item_input_params").get("month_ts")

    # prepare month level estimate for entertainment based on ts level values

    monthly_estimate = get_monthly_estimate(month_ts, entertainment_estimate)

    debug_object.update({
        "monthly_estimate": monthly_estimate
    })

    shape = entertainment_estimate.shape[0] * entertainment_estimate.shape[1]
    ts_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_EPOCH_IDX, :, :]

    entertainment_ts_estimate = np.vstack((np.reshape(ts_list, shape), np.reshape(entertainment_estimate, shape))).T

    t_entertainment_module_end = datetime.now()

    logger.debug("End of entertainment hybrid module")

    logger.info("Running main entertainment module took | %.3f s",
                get_time_diff(t_entertainment_module_start, t_entertainment_module_end))

    avg_capacity = entertainment_estimate.sum(axis=1)
    avg_capacity = avg_capacity[avg_capacity > 0]
    avg_capacity = np.mean(avg_capacity)

    debug_object.update({
        "entertainment_ts_estimate": entertainment_ts_estimate,
        "avg_consumption": avg_capacity
    })

    item_output_object.get('debug').update({
        "entertainment_module_dict": debug_object
    })

    logger.info("Average daily entertainment consumption | %.2f ", avg_capacity)

    return item_input_object, item_output_object, monthly_estimate, entertainment_ts_estimate
