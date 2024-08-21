"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update lighting consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def get_li_inference(app_index, item_input_object, item_output_object, logger_pass):

    """
    Update lighting consumption ranges using inference rules

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_li_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    li_config = get_inf_config().get("li")

    season = item_output_object.get("season")
    app_pot = item_output_object.get("inference_engine_dict").get("appliance_pot")[app_index]
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    max_cons = item_output_object.get("inference_engine_dict").get("appliance_max_values")[app_index, :, :]
    min_cons = item_output_object.get("inference_engine_dict").get("appliance_min_values")[app_index, :, :]
    li_scale_factor = item_output_object.get('debug').get('lighting_module_dict').get("scale_factor")

    ########################### RULE 1 - Check seasonality in lighting output ######################################

    min_cons, mid_cons, max_cons = adjust_li_seasonality(min_cons, mid_cons, max_cons, season, disagg_cons, logger)

    ########################### RULE 2 - Handle underestimation in lighting output ######################################

    li_scale_factor = min(li_config.get('max_scaling_factor'), max(li_scale_factor, li_config.get('min_scaling_factor')))

    mid_cons = mid_cons * li_scale_factor
    min_cons = min_cons * li_scale_factor
    max_cons = max_cons * li_scale_factor

    ########################### RULE 3 - Adjust lighting output based on room count of the user ########################

    room_count = item_input_object.get('home_meta_data').get('totalRooms')

    if room_count is None:
        room_count = 0

    multiplier_factor = li_config.get('room_count_factor')[np.digitize(room_count, li_config.get('room_count_bucket'))]

    mid_cons = mid_cons * multiplier_factor
    max_cons = max_cons * multiplier_factor
    min_cons = min_cons * multiplier_factor

    ########################### RULE 4 - Adjust lighting output based on app profile input of the user  ##############

    app_profile = item_input_object.get('app_profile').get('li')

    if app_profile is None:
        app_profile = item_input_object.get('app_profile').get('71')

    efficiency = '1'

    if (app_profile is not None) and (app_profile.get('size') is not None):
        logger.info('Li app prof present | ')
        efficiency = app_profile.get('size')

    efficiency_based_scaling_factor = 1

    day_level_cap_for_ineff_li = li_config.get('day_level_cap_for_ineff_li')

    if efficiency == '3' and np.sum(mid_cons, axis=1).mean() > day_level_cap_for_ineff_li:
        logger.info("Scaling lighting output, since user has all the li as efficient")
        efficiency_based_scaling_factor = li_config.get('scaling_factor_3')

    if efficiency == '2' and np.sum(mid_cons, axis=1).mean() > day_level_cap_for_ineff_li:
        logger.info("Scaling lighting output, since user has all the li as efficient")
        efficiency_based_scaling_factor = li_config.get('scaling_factor_2')

    if efficiency == '0':
        logger.info("Scaling lighting output, since user has all the li as inefficient")
        efficiency_based_scaling_factor = li_config.get('scaling_factor_0')

    mid_cons = mid_cons * efficiency_based_scaling_factor
    max_cons = max_cons * efficiency_based_scaling_factor
    min_cons = min_cons * efficiency_based_scaling_factor

    ########################### RULE 5 - Remove lighting on low activity days ######################################

    lighting_estimate = mid_cons
    lighting_max_estimate = max_cons
    lighting_min_estimate = min_cons

    lighting_estimate[np.logical_and(app_pot > 0, lighting_estimate == 0)] = \
        app_pot[np.logical_and(app_pot > 0, lighting_estimate == 0)] * np.max(lighting_estimate)

    lighting_max_estimate[np.logical_and(app_pot > 0, lighting_max_estimate == 0)] = \
        app_pot[np.logical_and(app_pot > 0, lighting_max_estimate == 0)] * np.max(lighting_max_estimate)

    lighting_estimate[lighting_estimate == 0] = disagg_cons[lighting_estimate == 0]
    lighting_min_estimate[lighting_min_estimate == 0] = disagg_cons[lighting_min_estimate == 0]
    lighting_max_estimate[lighting_max_estimate == 0] = disagg_cons[lighting_max_estimate == 0]

    zero_lighting_days = np.sum(disagg_cons, axis=1) == 0
    lighting_min_estimate[zero_lighting_days] = 0
    lighting_max_estimate[zero_lighting_days] = 0
    lighting_estimate[zero_lighting_days] = 0

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1) > 0

    lighting_max_estimate[vacation_days] = 0
    lighting_estimate[vacation_days] = 0
    lighting_min_estimate[vacation_days] = 0

    lighting_estimate = np.fmax(0, lighting_estimate)
    lighting_min_estimate = np.fmax(0, lighting_min_estimate)
    lighting_max_estimate = np.fmax(0, lighting_max_estimate)

    mid_cons = lighting_estimate
    min_cons = lighting_min_estimate
    max_cons = lighting_max_estimate

    amp = np.max(mid_cons) * 2

    ########################### RULE 6 - Maintain minimum month level lighting output #################################

    min_cons = apply_bc_level_min_limit(item_input_object, min_cons, vacation_days, li_config)
    mid_cons = apply_bc_level_min_limit(item_input_object, mid_cons, vacation_days, li_config)
    max_cons = apply_bc_level_min_limit(item_input_object, max_cons, vacation_days, li_config)

    min_cons = np.fmin(min_cons, amp)
    max_cons = np.fmin(max_cons, amp)
    mid_cons = np.fmin(mid_cons, amp)

    lighting_estimate = mid_cons
    lighting_min_estimate = min_cons
    lighting_max_estimate = max_cons

    # Updating the values in the original dictionary

    item_output_object = update_hybrid_object(app_index, item_output_object, lighting_estimate, lighting_max_estimate, lighting_min_estimate)

    item_output_object["inference_engine_dict"]["output_data"][app_index, :, :] = disagg_cons

    t_end = datetime.now()

    logger.debug("Lighting inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def apply_bc_level_min_limit(item_input_object, app_cons, vacation, li_config):
    """
    apply mininum bc level lighting consumption

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        app_cons                  (np.ndarray))   : lighting ts level output
        vacation                  (np.ndarray)    : day wise vacation tags
        li_config                 (dict)          : lighting config

    Returns:
        final_tou_consumption       (np.ndarray)    : updated final ts level estimation of appliances
        stat_app_array                (np.ndarray)    : updated dev qa metrics array
    """

    limit = li_config.get('min_li_cons')

    days_in_a_month = Cgbdisagg.DAYS_IN_MONTH

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc = np.unique(bc_list)

    # increase the consumption of lighting if it is less than a certain threshold

    for bc in unique_bc:

        if np.sum(bc_list == bc) < 5:
            continue

        scaling_factor = app_cons[bc_list == bc].sum()

        if np.sum(vacation[bc_list == bc]) > 0:
            scaling_factor = scaling_factor * days_in_a_month / (np.sum(vacation[bc_list == bc]))
        else:
            scaling_factor = scaling_factor * days_in_a_month / (np.sum(bc_list == bc))

        scaling_factor = limit / scaling_factor
        scaling_factor = max(1, scaling_factor)

        app_cons[bc_list == bc] = scaling_factor * app_cons[bc_list == bc]

    return np.nan_to_num(app_cons)


def adjust_li_seasonality(min_cons, mid_cons, max_cons, season, disagg_cons, logger):

    """
    Adjust lighting consumption incase of reverse seasonality

    Parameters:
        min_cons       (np.ndarray)    : ts level min lighting consumption
        mid_cons       (np.ndarray)    : ts level avg lighting consumption
        max_cons       (np.ndarray)    : ts level max lighting consumption
        season         (np.ndarray)    : day wise season tag
        disagg_cons    (np.ndarray)    : lighting disagg output
        logger         (logger)        : logger object

    Returns:
        min_cons       (np.ndarray)    : ts level min lighting consumption
        mid_cons       (np.ndarray)    : ts level avg lighting consumption
        max_cons       (np.ndarray)    : ts level max lighting consumption
    """

    samples_per_hour = int(min_cons.shape[1]/24)

    season[season > 0] = 1
    season[season < 0] = -1

    season_consumption = np.zeros(len(np.unique(season)))

    season_list = np.unique(season)

    day_hours = np.arange(6*samples_per_hour, Cgbdisagg.HRS_IN_DAY*samples_per_hour)

    for index, season_type in enumerate(season_list):
        consumption = copy.deepcopy(mid_cons[season == season_type][:, day_hours])
        season_consumption[index] = np.mean(consumption[consumption > 0])

    opp_seasonality_present_in_li = season_consumption[season_list == 1] > season_consumption[season_list == -1]

    if opp_seasonality_present_in_li:
        logger.debug("Reverse seasonality detected for lighting")
        diff = (season_consumption[season_list == 1] - season_consumption[season_list == -1])[0]

        target_epoch = np.zeros(min_cons.shape)
        target_epoch[season == 1] = 1
        target_epoch[:, : 12 * samples_per_hour] = 0
        target_epoch[mid_cons < diff] = 0
        target_epoch = target_epoch.astype(bool)

        mid_cons[target_epoch] = mid_cons[target_epoch] - diff
        min_cons[target_epoch] = mid_cons[target_epoch] - diff
        max_cons[target_epoch] = max_cons[target_epoch] - diff

        max_cons = np.fmax(max_cons, 0)
        min_cons = np.fmax(min_cons, 0)
        min_cons = np.fmax(min_cons, 0)

    max_cons[disagg_cons == 0] = 0
    mid_cons[disagg_cons == 0] = 0
    min_cons[disagg_cons == 0] = 0

    return min_cons, mid_cons, max_cons
