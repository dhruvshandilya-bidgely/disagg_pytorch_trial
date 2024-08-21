
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Initialize max and min ranges of appliance consumption (stage 1 of itemization)
"""

# Import python packages

import copy
import logging
import numpy as np
from sklearn.linear_model import LinearRegression

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

def initialize_ts_level_ranges(item_input_object, item_output_object, logger_pass):

    """
    Initialize appliance consumption ranges

    Parameters:
        item_input_object           (dict)      : Dict containing all inputs
        item_output_object          (dict)      : Dict containing all outputs
        logger_pass                 (dict)      : logger dictionary

    Returns:
        regression                 (np.ndarray)  : dict containing regression parameters
        total_consumption          (np.ndarray)  : total consumption
        min_range                  (np.ndarray)  : min app ranges
        mid_range                  (np.ndarray)  : mid app ranges
        max_range                  (np.ndarray)  : max app ranges
        app_cons                   (np.ndarray)  : app consumption
        app_conf                   (np.ndarray)  : app confidence
        app_pot                    (np.ndarray)  : app potential
    """

    # fetch required parameters

    app_cons = item_output_object.get("inference_engine_dict").get("app_cons")
    app_pot = item_output_object.get("inference_engine_dict").get("appliance_pot")
    app_conf = item_output_object.get("inference_engine_dict").get("appliance_conf")
    residual = item_output_object.get("inference_engine_dict").get("residual_without_detected_sig")
    app_names = item_output_object.get("inference_engine_dict").get("appliance_list")
    max_range = item_output_object.get("inference_engine_dict").get("appliance_max_values")
    min_range = item_output_object.get("inference_engine_dict").get("appliance_min_values")
    mid_range = item_output_object.get("inference_engine_dict").get("appliance_mid_values")
    input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('run_itemization_modules')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    app_count = mid_range.shape[2]

    regression = dict()

    total_consumption = np.zeros(max_range[:, :, 0].shape)

    data_shape = app_cons[:, :, 0].shape

    over_est_ts = residual <= 0
    under_est_ts = residual > 0

    over_est_ts = over_est_ts.flatten()
    under_est_ts = under_est_ts.flatten()

    for app in range(app_count):

        logger.debug("Initializing ranges for appliance %s ", app_names[app])

        # Preparing data for regression fitting

        pot_data = app_pot[:, :, app].flatten()
        cons_data = app_cons[:, :, app].flatten()

        pot_data = pot_data[cons_data > 0]
        cons_data = cons_data[cons_data > 0]

        cons_data = cons_data[pot_data > 0]
        pot_data = pot_data[pot_data > 0]

        # No regression fitting for vacation and non appliance users

        if len(cons_data) == 0 or len(pot_data) == 0:
            continue

        if app_names[app] in ['va1', 'va2']:
            continue

        # Fit regression

        regression[app_names[app]] = LinearRegression().fit(pot_data.reshape(-1, 1), cons_data.reshape(-1, 1))

        slope, intercept = regression[app_names[app]].coef_, regression[app_names[app]].intercept_

        dist = (pot_data * slope + intercept - cons_data) / (np.sqrt(slope * slope + 1))

        dist = dist.T[:, 0]

        pos_dis_intercept = 0
        neg_dis_intercept = 0

        if len(dist[dist > 0]):
            pos_dis_intercept = np.percentile(dist[dist > 0], 97)

        if len(dist[dist < 0]):
            neg_dis_intercept = np.percentile(dist[dist < 0], 3)

        pot_data = app_pot[:, :, app].flatten()
        cons_data = app_cons[:, :, app].flatten()
        conf_data = app_conf[:, :, app].flatten()

        pred = regression[app_names[app]].predict(np.nan_to_num(pot_data.reshape(-1, 1)))[:, 0]

        conf_data[conf_data == 0] = 1

        pred[cons_data == 0] = 0
        pred[conf_data == 0] = 0

        increase = - conf_data * (10/7) + (10/7)

        # Calculate intercept for upper and lower limit of appliance consumption

        pos_dis_intercept = np.ones(pot_data.shape) * pos_dis_intercept
        neg_dis_intercept = np.ones(pot_data.shape) * neg_dis_intercept

        # Low variation in case of certain appliances

        if app_names[app] not in ['pp']:
            increased_range = copy.deepcopy(increase)

            if app_names[app] not in ["ref", "ao"]:
                increased_range[over_est_ts] = 0

            pos_dis_intercept = pos_dis_intercept * (1 + increased_range)

            decreased_range = copy.deepcopy(increase)

            if app_names[app] not in ["ref", "ao"]:
                decreased_range[under_est_ts] = 0

            neg_dis_intercept = neg_dis_intercept * (1 + decreased_range)

        min_val = ((pot_data + neg_dis_intercept) * slope + intercept).flatten()
        max_val = ((pot_data + pos_dis_intercept) * slope + intercept).flatten()

        # Final sanity checks

        min_val[cons_data == 0] = 0
        max_val[cons_data == 0] = 0
        min_val[pot_data == 0] = 0
        max_val[pot_data == 0] = 0

        mid_range[:, :, app] = np.minimum(input_data, np.reshape(pred, data_shape))
        min_range[:, :, app] = np.minimum(input_data, np.reshape(min_val, data_shape))
        max_range[:, :, app] = np.minimum(input_data, np.reshape(max_val, data_shape))

    min_range = np.fmax(0, min_range)
    max_range = np.fmax(0, max_range)
    mid_range = np.fmax(0, mid_range)

    mid_range = np.swapaxes(mid_range, 0, 2)
    mid_range = np.fmax(0, np.swapaxes(mid_range, 1, 2))

    min_range = np.swapaxes(min_range, 0, 2)
    min_range = np.fmax(0, np.swapaxes(min_range, 1, 2))

    max_range = np.swapaxes(max_range, 0, 2)
    max_range = np.fmax(0, np.swapaxes(max_range, 1, 2))

    app_pot = np.swapaxes(app_pot, 0, 2)
    app_pot = np.fmax(0, np.swapaxes(app_pot, 1, 2))

    app_conf = np.swapaxes(app_conf, 0, 2)
    app_conf = np.fmax(0, np.swapaxes(app_conf, 1, 2))

    app_cons = np.swapaxes(app_cons, 0, 2)
    app_cons = np.fmax(0, np.swapaxes(app_cons, 1, 2))

    return regression, total_consumption, min_range, mid_range, max_range, app_cons, app_conf, app_pot
