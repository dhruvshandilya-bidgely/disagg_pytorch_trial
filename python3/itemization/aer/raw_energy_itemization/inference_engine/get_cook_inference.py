"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update cooking consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.raw_energy_itemization.utils import update_stat_app_tou
from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val_for_valid_boxes

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config2 import get_inf_config2


def get_cook_inference(app_index, item_input_object, item_output_object, date_list, logger_pass):

    """
    Update cooking consumption ranges using inference rules

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        date_list                   (np.ndarray): list of target dates for heatmap dumping
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_cook_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    max_cons = item_output_object.get("inference_engine_dict").get("appliance_max_values")[app_index, :, :]
    min_cons = item_output_object.get("inference_engine_dict").get("appliance_min_values")[app_index, :, :]

    samples_per_hour = int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)
    cook_config = get_inf_config2(item_input_object, samples_per_hour).get("cook")

    residual = copy.deepcopy(item_output_object.get("inference_engine_dict").get("residual_data"))
    activity_curve = item_output_object.get('debug').get("profile_attributes_dict").get("activity_curve")

    min_cons = np.fmax(cook_config.get("min_cons_factor")*disagg_cons, min_cons)

    if np.all(disagg_cons == 0):
        return item_output_object

    ########################### RULE 1 - Shifting tou at overestimation hours  ######################################

    min_cons, mid_cons, max_cons = \
        adjust_cooking_hours_to_reduce_overestimation(item_output_object, disagg_cons, min_cons, mid_cons, max_cons,
                                                      activity_curve, cook_config, logger)

    ########################## RULE 2 - Add additional weekend consumption in cooking output  ##########################

    min_cons, mid_cons, max_cons = add_weekend_activity(item_output_object, min_cons, mid_cons, max_cons, samples_per_hour, logger)

    ########################### RULE 3 - Add cooking type box signature in cooking consumption  #########################

    min_cons, mid_cons, max_cons = \
        add_additional_cooking_box_from_disagg_residual(item_output_object, residual, cook_config, min_cons, mid_cons, max_cons, logger)

    ########################### RULE 4 - Zero consumption in sleep hours  ######################################

    min_cons, mid_cons, max_cons = \
        remove_cooking_in_sleeping_hours(item_output_object, disagg_cons, cook_config, min_cons, mid_cons, max_cons)

    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    ############################# RULE 5 - Monthly consumption limit ##########################################

    min_cons, mid_cons, max_cons = apply_monthly_cons_limit(min_cons, mid_cons, max_cons, hybrid_config, cook_config, input_data, logger)

    # Updating the values in the original dictionary

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons)

    item_output_object["inference_engine_dict"]["output_data"][app_index, :, :] = disagg_cons

    t_end = datetime.now()

    logger.debug("Cooking inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def add_additional_cooking_box_from_disagg_residual(item_output_object, residual, cook_config, min_cons, mid_cons, max_cons, logger):

    """
    Update cooking consumption ranges using inference rules

    Parameters:
        item_output_object          (dict)          : Dict containing all hybrid outputs
        residual                    (np.ndarray)    : disagg residual data
        cook_config                 (dict)          : config
        min_cons                    (np.ndarray)    : ts level min laundry consumption
        mid_cons                    (np.ndarray)    : ts level avg laundry consumption
        max_cons                    (np.ndarray)    : ts level max laundry consumption
        logger                      (dict)          : logger object

    Returns:
        min_cons                    (np.ndarray)    : ts level min laundry consumption
        mid_cons                    (np.ndarray)    : ts level avg laundry consumption
        max_cons                    (np.ndarray)    : ts level max laundry consumption
    """

    residual = np.fmax(0, residual)
    cooking_hours = cook_config.get("cooking_hours")
    cooking_hours = cooking_hours.astype(int)

    samples_per_hour = int(mid_cons.shape[1]/Cgbdisagg.HRS_IN_DAY)

    residual_copy = copy.deepcopy(residual)

    residual_copy = np.fmax(residual_copy, 0)

    # fetching cooking type boxes present in disagg residual

    item_output_object, res_boxes_idx = get_cook_box_signature(item_output_object, residual_copy)

    residual_cons = copy.deepcopy(residual_copy) - np.min(residual_copy, axis=1)[:, None]

    residual_cons[np.logical_not(res_boxes_idx)] = 0

    residual_cons = np.minimum(residual_cons, np.fmax(0, residual))

    if np.sum(residual_cons[:, cooking_hours]):
        logger.info("Cooking activity found in residual data | ")

    residual[:, cooking_hours] = residual[:, cooking_hours] - residual_cons[:, cooking_hours]

    item_output_object["inference_engine_dict"].update({
        "residual_data": (residual + np.fmin(0, item_output_object.get("inference_engine_dict").get("residual_data")))
    })

    # adding cooking type boxes present in disagg residual into cooking estimates

    max_cons[:, cooking_hours] = max_cons[:, cooking_hours] + residual_cons[:, cooking_hours]
    mid_cons[:, cooking_hours] = mid_cons[:, cooking_hours] + residual_cons[:, cooking_hours]
    min_cons[:, cooking_hours] = min_cons[:, cooking_hours] + residual_cons[:, cooking_hours]

    residual_wh = copy.deepcopy(item_output_object.get("hvac_dict").get("wh"))
    residual_wh[residual_wh < cook_config.get('min_res_wh') / samples_per_hour] = 0
    residual_wh[residual_wh > cook_config.get('max_res_wh') / samples_per_hour] = 0

    max_cons = max_cons - residual_wh
    min_cons = min_cons - residual_wh
    mid_cons = mid_cons - residual_wh

    max_cons = np.fmax(0, max_cons)
    mid_cons = np.fmax(0, mid_cons)
    min_cons = np.fmax(0, min_cons)

    return min_cons, mid_cons, max_cons


def remove_cooking_in_sleeping_hours(item_output_object, disagg_cons, cook_config, min_cons, mid_cons, max_cons):

    """
    Removing cooking during sleep hours

    Parameters:
        item_output_object          (dict)          : Dict containing all hybrid inputs
        disagg_cons                 (np.ndarray)    : intialized consumption for cooking
        cook_config                 (dict)          : cooking config
        min_cons                    (np.ndarray)    : ts level min lighting consumption
        mid_cons                    (np.ndarray)    : ts level avg lighting consumption
        max_cons                    (np.ndarray)    : ts level max lighting consumption
        samples_per_hour            (int)           : samples in an hour

    Returns:
        min_cons                    (np.ndarray)    : ts level min laundry consumption
        mid_cons                    (np.ndarray)    : ts level avg laundry consumption
        max_cons                    (np.ndarray)    : ts level max laundry consumption
    """

    sleep_hours = item_output_object.get("profile_attributes").get("sleep_hours")

    default_wakeup_time = cook_config.get('default_wakeup_time')
    additional_cook_hours = cook_config.get('additional_cook_hours')
    sleep_hours_thres = cook_config.get('sleep_hours_thres')

    # removing cooking during sleep hours

    if np.all(sleep_hours == 0):
        sleep_hours = np.ones_like(sleep_hours)
        sleep_hours[np.arange(0, default_wakeup_time)] = 0

    if mid_cons[:, sleep_hours].sum() == 0 and np.sum(sleep_hours) < sleep_hours_thres:
        sleep_hours[additional_cook_hours] = 1

    mid_cons[:, np.logical_not(sleep_hours)] = 0
    min_cons[:, np.logical_not(sleep_hours)] = 0
    max_cons[:, np.logical_not(sleep_hours)] = 0

    max_cons = max_cons * cook_config.get('max_range_multiplier')

    max_cons = np.maximum(max_cons, disagg_cons)

    max_cons[:, np.logical_not(sleep_hours)] = 0

    # removing cooking during 0 cooking hours

    zero_cook_hours = cook_config.get('zero_cook_hours')

    max_cons[:, zero_cook_hours] = 0
    mid_cons[:, zero_cook_hours] = 0
    min_cons[:, zero_cook_hours] = 0

    return min_cons, mid_cons, max_cons


def get_cook_box_signature(item_output_object, residual):

    """
      Calculate cooking consumption in the leftover residual data

      Parameters:
          item_output_object          (dict)        : Dict containing all hybrid outputs
          residual                    (np.ndarray)  : ts level residual data

      Returns:
          item_output_object          (dict)        : Dict containing all hybrid outputs
          valid_idx                   (dict)        : cooking BOX tou
      """

    # Either based on app profile, else based on defined config

    box_seq = item_output_object.get("box_dict").get("box_seq_wh")

    valid_idx = np.zeros(residual.size)

    # Filtering boxes in inactive time of the day

    box_seq = box_seq.astype(int)

    boxes_score = item_output_object.get("box_score")
    cook_boxes = boxes_score[:, 2] == np.max(boxes_score, axis=1)
    cook_boxes[boxes_score[:, 2] == 0] = 0

    valid_idx = fill_arr_based_seq_val_for_valid_boxes(box_seq, cook_boxes, valid_idx, 1, 1)

    item_output_object["box_dict"]["box_seq_wh"] = box_seq

    valid_idx = np.reshape(valid_idx, residual.shape)

    return item_output_object, valid_idx


def adjust_cooking_hours_to_reduce_overestimation(item_output_object, disagg_cons, min_cons, mid_cons, max_cons,
                                                  activity_curve, cook_config, logger):

    """
    Adjust cooking usage hours to reduce overlap with laundry consumption

    Parameters:
        item_output_object          (dict)          : Dict containing all hybrid inputs
        disagg_cons                 (np.ndarray)    : cooking statistical disagg output
        min_cons                    (np.ndarray)    : ts level min lighting consumption
        mid_cons                    (np.ndarray)    : ts level avg lighting consumption
        max_cons                    (np.ndarray)    : ts level max lighting consumption
        activity_curve              (np.ndarray)    : actvity profile of the user
        cook_config                 (dict)          : cooking config
        logger                      (logger)        : logger object

    Returns:
        min_cons                    (np.ndarray)    : ts level min laundry consumption
        mid_cons                    (np.ndarray)    : ts level avg laundry consumption
        max_cons                    (np.ndarray)    : ts level max laundry consumption
    """

    neg_residual = item_output_object.get("inference_engine_dict").get("residual_data")

    neg_residual = -np.fmin(0, neg_residual)

    overest_points = neg_residual > 0

    overest_tou = np.sum(overest_points, axis=0) > len(disagg_cons) * cook_config.get("app_tou_factor")
    app_tou = np.sum(disagg_cons > 0, axis=0) > len(disagg_cons) * cook_config.get("overest_tou_factor")

    overlap_tou = np.multiply(overest_tou, app_tou)
    app_seq = find_seq(app_tou, np.zeros(len(overlap_tou)), np.zeros(len(overlap_tou)))

    overest_bool = np.any(app_seq[:, 0] == 1)

    samples_per_hour = int(len(disagg_cons[0]) / Cgbdisagg.HRS_IN_DAY)

    min_disagg_frac = cook_config.get('min_disagg_frac')

    mid_cons = np.maximum(mid_cons, disagg_cons*min_disagg_frac)
    max_cons = np.maximum(max_cons, disagg_cons*min_disagg_frac)
    min_cons = np.maximum(min_cons, disagg_cons*min_disagg_frac)

    # If overestimation exists at cooking tou, shift the cooking band by 1-2 hours

    if overest_bool:
        logger.debug("Overestimation points present with cooking consumption, updating TOU")
        min_cons, mid_cons, max_cons = update_stat_app_tou(app_seq, app_tou, overest_tou, samples_per_hour, activity_curve, mid_cons, min_cons, max_cons)

    return min_cons, mid_cons, max_cons


def add_weekend_activity(item_output_object, min_cons, mid_cons, max_cons, samples_per_hour, logger):

    """
    Adjust extra weekend consumption into cooking min/mid/max consumption

    Parameters:
        item_output_object          (dict)          : Dict containing all hybrid inputs
        min_cons                    (np.ndarray)    : ts level min lighting consumption
        mid_cons                    (np.ndarray)    : ts level avg lighting consumption
        max_cons                    (np.ndarray)    : ts level max lighting consumption
        samples_per_hour            (int)           : samples in an hour
        logger                      (logger)        : logger object

    Returns:
        min_cons                    (np.ndarray)    : ts level min laundry consumption
        mid_cons                    (np.ndarray)    : ts level avg laundry consumption
        max_cons                    (np.ndarray)    : ts level max laundry consumption
    """

    weekend_activity = item_output_object.get("inference_engine_dict").get("weekend_activity")

    # Add additional cooking consumption, present only on weekends

    if weekend_activity is not None:

        cooking_hours_tmp = np.arange(7 * samples_per_hour, 9 * samples_per_hour + 1).astype(int)

        if np.sum(weekend_activity[:, cooking_hours_tmp]):
            logger.info("Adding extra cooking on weekends | ")

        max_cons[:, cooking_hours_tmp] = max_cons[:, cooking_hours_tmp] + weekend_activity[:, cooking_hours_tmp]
        mid_cons[:, cooking_hours_tmp] = mid_cons[:, cooking_hours_tmp] + weekend_activity[:, cooking_hours_tmp]
        min_cons[:, cooking_hours_tmp] = min_cons[:, cooking_hours_tmp] + weekend_activity[:, cooking_hours_tmp]

    return min_cons, mid_cons, max_cons


def apply_monthly_cons_limit(min_cons, mid_cons, max_cons, hybrid_config, cook_config, input_data, logger):

    """
    adjust cooking consumption based on min and max monthly limit

    Parameters:
        item_output_object          (dict)          : Dict containing all hybrid inputs
        min_cons                    (np.ndarray)    : ts level min lighting consumption
        mid_cons                    (np.ndarray)    : ts level avg lighting consumption
        max_cons                    (np.ndarray)    : ts level max lighting consumption
        hybrid_config               (dict)          : hybrid v2 pilot config
        cook_config                 (dict)          : cooking config
        input_data                  (np.ndarray)    : input data
        logger                      (logger)        : logger object

    Returns:
        min_cons                    (np.ndarray)    : ts level min laundry consumption
        mid_cons                    (np.ndarray)    : ts level avg laundry consumption
        max_cons                    (np.ndarray)    : ts level max laundry consumption
    """

    monthly_cons_max_limit = 0
    monthly_cons_min_limit = 0

    ld_idx = hybrid_config.get("app_seq").index('cook')
    have_min_cons = hybrid_config.get("have_min_cons")[ld_idx]
    min_cons_limit = hybrid_config.get("min_cons")[ld_idx]

    if have_min_cons and min_cons_limit > 0:
        monthly_cons_min_limit = min_cons_limit

    have_mid_cons = hybrid_config.get("have_max_cons")[ld_idx]
    mid_cons_lim = hybrid_config.get("max_cons")[ld_idx]

    # updating cooking ts level estimates based on monthly min/max limit

    if have_mid_cons and mid_cons_lim > 0:
        monthly_cons_max_limit = mid_cons_lim

    if monthly_cons_max_limit and monthly_cons_max_limit <= cook_config.get('max_cook'):
        monthly_cons = ((np.sum(max_cons) / len(max_cons)) * Cgbdisagg.DAYS_IN_MONTH) / Cgbdisagg.WH_IN_1_KWH
        if monthly_cons > monthly_cons_max_limit:
            factor = monthly_cons_max_limit / monthly_cons
            max_cons = max_cons * factor
            mid_cons = np.minimum(mid_cons, max_cons)
            min_cons = np.maximum(max_cons, min_cons)

    if monthly_cons_min_limit and monthly_cons_min_limit >= cook_config.get('min_cook'):
        monthly_cons = ((np.sum(min_cons) / len(min_cons)) * Cgbdisagg.DAYS_IN_MONTH) / Cgbdisagg.WH_IN_1_KWH
        if monthly_cons < monthly_cons_min_limit:
            factor = monthly_cons_min_limit / monthly_cons
            min_cons = min_cons * factor
            mid_cons = np.maximum(mid_cons, min_cons)
            max_cons = np.maximum(max_cons, min_cons)

    min_cons = np.minimum(min_cons, mid_cons)
    max_cons = np.maximum(max_cons, mid_cons)

    max_cons = np.minimum(max_cons, input_data)
    min_cons = np.minimum(min_cons, input_data)
    mid_cons = np.minimum(mid_cons, input_data)

    if np.sum(mid_cons.sum(axis=1) > 0) < len(mid_cons) * cook_config.get('zero_cook_days_frac'):
        logger.info("Blocked cooking consumption since days count were less than 5% | ")
        mid_cons[:, :] = 0
        min_cons[:, :] = 0
        max_cons[:, :] = 0

    return min_cons, mid_cons, max_cons
