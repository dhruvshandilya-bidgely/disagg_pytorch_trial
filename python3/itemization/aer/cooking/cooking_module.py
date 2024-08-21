
"""
Author - Nisha Agarwal
Date - 9th Feb 20
Master file for estimating cooking hybrid capacity
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.cooking.functions.get_cooking_consumption import get_cooking_estimate


def get_monthly_estimate(month_ts, data_est):

    """ Return monthly estimate

    This function generates estimated usage per month

    Input:
      month_ts (double matrix)    : 2d matrix with month timestamps
      data_est (double matrix)    : 2d matrix containing laundry estimates

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


def run_cooking_module(item_input_object, item_output_object, logger_pass):

    """
    Master function for cooking modules

    Parameters:
        item_input_object           (dict)           : Dict containing all hybrid inputs
        item_output_object          (dict)           : Dict containing all hybrid outputs
        logger_pass                 (dict)           : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object           (dict)           : Dict containing all hybrid inputs
        item_output_object          (dict)           : Dict containing all hybrid outputs
        monthly_estimate            (np.ndarray)     : monthly level cooking estimate
        cooking_ts_estimate         (np.ndarray)     : timestamp level cooking estimate
    """

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('run_cooking_module')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    logger.debug("Start of cooking hybrid module")

    # Fetch required input parameters

    weekend_day = item_input_object.get("item_input_params").get("weekend_days")
    occupants_features = item_output_object.get("occupants_profile").get("occupants_features")

    cooking_estimate = get_cooking_estimate(item_input_object, item_output_object, occupants_features, weekend_day, logger_pass)

    t_cooking_module_start = datetime.now()

    # Prepare monthly estimate

    month_ts = item_input_object.get("item_input_params").get("month_ts")

    monthly_estimate = get_monthly_estimate(month_ts, cooking_estimate)

    # Prepare ts level estimate

    shape = cooking_estimate.shape[0] * cooking_estimate.shape[1]
    ts_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_EPOCH_IDX, :, :]

    cooking_ts_estimate = np.vstack((np.reshape(ts_list, (shape)), np.reshape(cooking_estimate, (shape)))).T

    t_cooking_module_end = datetime.now()

    logger.debug("End of cooking hybrid module")

    logger.info("Running main cooking module took | %.3f s", get_time_diff(t_cooking_module_start, t_cooking_module_end))

    # Calculate average daily consumption

    avg_capacity = cooking_estimate.sum(axis=1)
    avg_capacity = avg_capacity[avg_capacity > 0]
    avg_capacity = np.mean(avg_capacity)

    logger.info("Average daily cooking consumption | %.2f ", avg_capacity)

    return item_input_object, item_output_object, monthly_estimate, cooking_ts_estimate
