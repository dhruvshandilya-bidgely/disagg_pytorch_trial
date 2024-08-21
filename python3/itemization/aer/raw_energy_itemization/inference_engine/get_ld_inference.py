"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update laundry consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.init_itemization_config import init_itemization_params

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config2 import get_inf_config2


def get_ld_inference(app_index, item_input_object, item_output_object, date_list, logger_pass):

    """
    Update laundry consumption ranges using inference rules

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

    logger_base = logger_pass.get('logger_base').getChild('get_ld_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    max_cons = item_output_object.get("inference_engine_dict").get("appliance_max_values")[app_index, :, :]
    min_cons = item_output_object.get("inference_engine_dict").get("appliance_min_values")[app_index, :, :]

    samples_per_hour = int(len(disagg_cons[0]) / Cgbdisagg.HRS_IN_DAY)
    ld_config = get_inf_config2(item_input_object, samples_per_hour).get("ld")

    residual = copy.deepcopy(item_output_object.get("inference_engine_dict").get("residual_data"))
    activity_curve = item_output_object.get('debug').get("profile_attributes_dict").get("activity_curve")

    if np.sum(disagg_cons) == 0:
        return item_output_object

    residual = np.fmax(0, residual)

    mid_cons = np.maximum(mid_cons, disagg_cons*ld_config.get('min_disagg_frac'))
    max_cons = np.maximum(max_cons, disagg_cons*ld_config.get('min_disagg_frac'))
    min_cons = np.maximum(min_cons, disagg_cons*ld_config.get('min_disagg_frac'))

    if np.all(disagg_cons == 0):
        return item_output_object

    ###################### RULE 1 - Add weekend residual consumption into laundry consumption ######################

    min_cons, mid_cons, max_cons, = add_weekend_cons_to_laundry(item_output_object, min_cons, mid_cons, max_cons, samples_per_hour, ld_config, logger)

    logger.debug("Added residual weekend component in laundry ranges | ")

    #################### RULE 2 - adding laundry activity boxes into laundry consumption  ##########################

    # Calculate laundry consumption boxes present in residual data

    min_cons, mid_cons, max_cons = \
    add_laundry_type_boxes_from_disagg_residual(activity_curve, item_output_object, min_cons, mid_cons, max_cons,
                                                samples_per_hour, residual, logger)

    ############################# RULE 3 - giving 0 laundry during inactive hours of the day #################################

    min_cons, mid_cons, max_cons = \
        remove_laundry_in_inactive_hours(item_input_object, item_output_object, min_cons, mid_cons, max_cons, samples_per_hour, ld_config)

    ############################# RULE 4 - Monthly consumption limit ##########################################

    min_cons, mid_cons, max_cons = apply_monthly_cons_limit(min_cons, mid_cons, max_cons, hybrid_config, ld_config, logger)

    # Updating the values in the original dictionary

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons)

    item_output_object["inference_engine_dict"]["output_data"][app_index, :, :] = disagg_cons

    t_end = datetime.now()

    logger.debug("Laundry inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def add_laundry_type_boxes_from_disagg_residual(activity_curve, item_output_object, min_cons, mid_cons, max_cons,
                                                samples_per_hour, disagg_residual, logger):

    """
    Add laundry type activity boxes from disagg residual into laundry min/mid/max consumption

    Parameters:
        activity_curve              (np.ndarray)    : user activity curve
        item_output_object          (dict)          : Dict containing all hybrid inputs
        min_cons                    (np.ndarray)    : ts level min lighting consumption
        mid_cons                    (np.ndarray)    : ts level avg lighting consumption
        max_cons                    (np.ndarray)    : ts level max lighting consumption
        samples_per_hour            (int)           : samples in an hour
        disagg_residual             (np.ndarray)    : ts level residual data
        logger                      (logger)        : logger object

    Returns:
        min_cons                    (np.ndarray)    : ts level min laundry consumption
        mid_cons                    (np.ndarray)    : ts level avg laundry consumption
        max_cons                    (np.ndarray)    : ts level max laundry consumption
    """

    normalized_curve = activity_curve / (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    # fetching laundry type boxes from residual

    residual_copy = copy.deepcopy(disagg_residual)

    residual_copy = np.fmax(residual_copy, 0)

    item_output_object, valid_idx = get_residual_signature(item_output_object, residual_copy, samples_per_hour, normalized_curve)

    valid_idx = valid_idx.astype(bool)

    residual_cons = copy.deepcopy(residual_copy) - np.min(residual_copy, axis=1)[:, None]

    residual_cons[np.logical_not(valid_idx)] = 0

    mid_cons = mid_cons + residual_cons
    max_cons = max_cons + residual_cons
    min_cons = min_cons + residual_cons

    if np.sum(residual_cons):
        logger.info("Adding laundry usage boxes found in residual data | ")

    residual_cons = np.minimum(residual_cons, np.fmax(0, disagg_residual))

    item_output_object["inference_engine_dict"].update({
        "residual_data": disagg_residual - residual_cons
    })

    logger.debug("Added residual box type consumption in laundry ranges")

    return min_cons, mid_cons, max_cons


def remove_laundry_in_inactive_hours(item_input_object, item_output_object, min_cons, mid_cons, max_cons, samples_per_hour, ld_config):

    """
    block laundry in sleeping hours and vacation days

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        min_cons                    (np.ndarray)    : ts level min lighting consumption
        mid_cons                    (np.ndarray)    : ts level avg lighting consumption
        max_cons                    (np.ndarray)    : ts level max lighting consumption
        samples_per_hour            (int)           : samples in an hour
        ld_config                   (dict)          : laundry config

    Returns:
        min_cons                    (np.ndarray)    : ts level min laundry consumption
        mid_cons                    (np.ndarray)    : ts level avg laundry consumption
        max_cons                    (np.ndarray)    : ts level max laundry consumption
    """

    sleep_hours = copy.deepcopy(item_output_object.get("profile_attributes").get("sleep_hours"))

    sleep_hours[get_index_array(item_output_object.get("profile_attributes").get("sleep_time") * samples_per_hour,
                                item_output_object.get("profile_attributes").get("sleep_time") * samples_per_hour +
                                int(samples_per_hour / 2) + 1,
                                samples_per_hour * Cgbdisagg.HRS_IN_DAY)] = 1

    sleep_hours[ld_config.get("ld_hours")] = 1
    mid_cons[:, np.logical_not(sleep_hours)] = 0
    min_cons[:, np.logical_not(sleep_hours)] = 0
    max_cons[:, np.logical_not(sleep_hours)] = 0

    zero_ld_hours = ld_config.get('zero_ld_hours').astype(int)

    max_cons[:, zero_ld_hours] = 0
    mid_cons[:, zero_ld_hours] = 0
    min_cons[:, zero_ld_hours] = 0

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    max_cons[vacation_days] = 0
    min_cons[vacation_days] = 0
    mid_cons[vacation_days] = 0

    return min_cons, mid_cons, max_cons


def add_weekend_cons_to_laundry(item_output_object, min_cons, mid_cons, max_cons, samples_per_hour, ld_config, logger):

    """
    Adjust extra weekend consumption into laundry min/mid/max consumption

    Parameters:
        item_output_object          (dict)          : Dict containing all hybrid inputs
        min_cons                    (np.ndarray)    : ts level min lighting consumption
        mid_cons                    (np.ndarray)    : ts level avg lighting consumption
        max_cons                    (np.ndarray)    : ts level max lighting consumption
        samples_per_hour            (int)           : samples in an hour
        ld_config                   (dict)          : laundry config
        logger                      (logger)        : logger object

    Returns:
        min_cons                    (np.ndarray)    : ts level min laundry consumption
        mid_cons                    (np.ndarray)    : ts level avg laundry consumption
        max_cons                    (np.ndarray)    : ts level max laundry consumption
    """

    weekend_activity = item_output_object.get("inference_engine_dict").get("weekend_activity")

    laundry_hours = np.setdiff1d(np.arange(0, (Cgbdisagg.HRS_IN_DAY-1) * samples_per_hour + 1), ld_config.get('cooking_hours'))

    laundry_hours = laundry_hours.astype(int)

    if weekend_activity is None:
        weekend_activity = np.zeros_like(max_cons)
        logger.info("Weekend activity not found | ")

    elif np.sum(weekend_activity[:, laundry_hours]):
        logger.info("Adding extra Laundry on weekends | ")

    max_cons[:, laundry_hours] = max_cons[:, laundry_hours] + weekend_activity[:, laundry_hours]
    mid_cons[:, laundry_hours] = mid_cons[:, laundry_hours] + weekend_activity[:, laundry_hours]
    min_cons[:, laundry_hours] = min_cons[:, laundry_hours] + weekend_activity[:, laundry_hours]

    return min_cons, mid_cons, max_cons


def apply_monthly_cons_limit(min_cons, mid_cons, max_cons, hybrid_config, ld_config, logger):

    """
    adjust laundry consumption based on min and max monthly limit

    Parameters:
        min_cons       (np.ndarray)    : ts level min lighting consumption
        mid_cons       (np.ndarray)    : ts level avg lighting consumption
        max_cons       (np.ndarray)    : ts level max lighting consumption
        hybrid_config  (dict)          : hybrid v2 pilot config
        ld_config      (dict)          : laundry config
        logger         (logger)        : logger object

    Returns:
        min_cons       (np.ndarray)    : ts level min laundry consumption
        mid_cons       (np.ndarray)    : ts level avg laundry consumption
        max_cons       (np.ndarray)    : ts level max laundry consumption
    """

    monthly_cons_max_limit = 0
    monthly_cons_min_limit = 0

    ld_idx = hybrid_config.get("app_seq").index('ld')
    have_min_cons = hybrid_config.get("have_min_cons")[ld_idx]
    min_cons_limit = hybrid_config.get("min_cons")[ld_idx]

    if have_min_cons and min_cons_limit > 0:
        monthly_cons_min_limit = min_cons_limit

    have_mid_cons = hybrid_config.get("have_max_cons")[ld_idx]
    mid_cons_lim = hybrid_config.get("max_cons")[ld_idx]

    if have_mid_cons and mid_cons_lim > 0:
        monthly_cons_max_limit = mid_cons_lim

    max_cons = max_cons * ld_config.get('max_range_multiplier')

    if monthly_cons_max_limit and monthly_cons_max_limit <= ld_config.get('max_ld'):
        monthly_cons = ((np.sum(max_cons) / len(max_cons)) * Cgbdisagg.DAYS_IN_MONTH) / Cgbdisagg.WH_IN_1_KWH
        if monthly_cons > monthly_cons_max_limit:
            scaling_factor = monthly_cons_max_limit / monthly_cons
            max_cons = max_cons * scaling_factor
            mid_cons = np.minimum(mid_cons, max_cons)
            min_cons = np.maximum(max_cons, min_cons)

    if monthly_cons_min_limit and monthly_cons_min_limit >= ld_config.get('min_ld'):
        monthly_cons = ((np.sum(min_cons) / len(min_cons)) * Cgbdisagg.DAYS_IN_MONTH) / Cgbdisagg.WH_IN_1_KWH
        if monthly_cons < monthly_cons_min_limit:
            scaling_factor = monthly_cons_min_limit / monthly_cons
            min_cons = min_cons * scaling_factor
            mid_cons = np.maximum(mid_cons, min_cons)
            max_cons = np.maximum(max_cons, min_cons)

    min_cons = np.minimum(min_cons, mid_cons)
    max_cons = np.maximum(max_cons, mid_cons)

    if np.sum(mid_cons > 0):
        mid_cons = np.fmin(mid_cons, np.percentile(mid_cons[mid_cons > 0], 98))
    if np.sum(min_cons > 0):
        min_cons = np.fmin(min_cons, np.percentile(min_cons[min_cons > 0], 98))
    if np.sum(max_cons > 0):
        max_cons = np.fmin(max_cons, np.percentile(max_cons[max_cons > 0], 98))

    if np.sum(mid_cons.sum(axis=1) > 0) < len(mid_cons)*ld_config.get('zero_ld_days_frac'):
        logger.info("Blocked laundry consumption since days count were less than 5% | ")
        mid_cons[:, :] = 0
        min_cons[:, :] = 0
        max_cons[:, :] = 0

    return min_cons, mid_cons, max_cons


def get_residual_signature(item_output_object, residual, samples_per_hour, activity_curve):

    """
    fetch laundry consumption from the leftover boxes in residual data

    Parameters:
        item_output_object          (dict)        : Dict containing all hybrid inputs
        residual                    (np.ndarray)  : ts level residual data
        samples_per_hour            (int)         : samples in an hour
        activity_curve              (np.ndarray)  : user activity curve

    Returns:
        item_output_object          (dict)        : Dict containing all hybrid outputs
        valid_idx                   (dict)        : Laundry BOX tou
    """

    # Pick boxes from left over residual that has the higher chances of being a laundry appliace

    box_seq = item_output_object.get("box_dict").get("box_seq_wh")

    valid_idx = np.zeros(residual.size)

    box_seq = box_seq.astype(int)

    valid_tou = np.where(activity_curve > 0.2)[0]

    seq_config = init_itemization_params().get("seq_config")

    boxes_score = item_output_object["box_score"]
    ld_boxes = boxes_score[:, 1] == np.max(boxes_score, axis=1)
    ld_boxes[boxes_score[:, 1] == 0] = 0

    for i in range(len(box_seq)):
        if ld_boxes[i] and (box_seq[i, 1] % (samples_per_hour * Cgbdisagg.HRS_IN_DAY) in valid_tou):
            valid_idx[box_seq[i, seq_config.get("start")]: box_seq[i, seq_config.get("end")] + 1] = 1

    item_output_object["box_dict"]["box_seq_wh"] = box_seq

    valid_idx = np.reshape(valid_idx, residual.shape)

    return item_output_object, valid_idx
