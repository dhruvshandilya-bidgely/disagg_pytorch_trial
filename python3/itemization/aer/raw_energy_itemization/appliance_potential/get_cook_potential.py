"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update cooking confidence and potential values
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

from python3.itemization.aer.raw_energy_itemization.appliance_potential.config.get_pot_conf import get_pot_conf


def get_cook_potential(app_index, item_input_object, item_output_object, sampling_rate, logger_pass):

    """
    Calculate cooking confidence and potential values

    Parameters:
        app_index                   (int)           : Index of app in the appliance list
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        sampling_rate               (int)           : sampling rate
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_cook_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetching required inputs

    cook_disagg = item_output_object.get("updated_output_data")[app_index, :, :]
    activity_curve = copy.deepcopy(item_output_object.get("debug").get("profile_attributes_dict").get("activity_curve"))
    dow = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_DOW_IDX, :, 0]
    original_input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    config = get_pot_conf(samples_per_hour).get('cook')

    cooking_potential_score_offset = config.get('cooking_potential_score_offset')
    activity_curve_thres = config.get('activity_curve_thres')
    activity_curve_cons_thres = config.get('activity_curve_cons_thres')
    non_cooking_hours = config.get('non_cooking_hours')
    cooking_pot_score_at_nonzero_disagg_points = config.get('cooking_pot_score_at_nonzero_disagg_points')
    cooking_cons_scaling_factor = config.get('cooking_cons_scaling_factor')
    cooking_cons_scaling_factor_offset = config.get('cooking_cons_scaling_factor_offset')

    # Calculate cooking potential using laundry tou and activity curve

    cook_confidence = np.ones(original_input_data.shape)

    # capping activity profile of the user to avoid outlier points

    activity_curve = (activity_curve - np.percentile(activity_curve, 3)) / (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    cook_confidence[np.logical_and(cook_disagg > 0, cook_confidence == 0)] = 0.7

    normalized_input_data = (original_input_data[:] - np.percentile(original_input_data, 5,  axis=1)[:, None]) / \
                             (np.percentile(original_input_data, 95, axis=1) - np.percentile(original_input_data, 5, axis=1))[:, None]

    normalized_input_data = np.round(normalized_input_data, 2)

    activity_curve[non_cooking_hours] = 0

    # adding pattern of user living load activity into cooking potential

    cook_confidence = np.multiply(cook_confidence, activity_curve[None, :])

    cook_potential = np.multiply(normalized_input_data, activity_curve[None, :])

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    total_cons = original_input_data[np.logical_not(vacation)]
    length = np.sum(np.logical_not(vacation))
    total_cons = ((np.sum(total_cons) / length) * (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH))

    # higher potential given to users with high consumption level

    threshold = (activity_curve_thres[0] * (total_cons > activity_curve_cons_thres[1])) + activity_curve_thres[1] * (total_cons <= activity_curve_cons_thres[1])

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    cook_idx = hybrid_config.get("app_seq").index('cook')

    # adjustment done in cooking ts level potential score, inorder to adjust cooking consumption if cooking average is high for the given pilot

    scale_cons = hybrid_config.get("scale_app_cons")[cook_idx]
    cons_factor = hybrid_config.get("scale_app_cons_factor")[cook_idx]

    if scale_cons:
        if cons_factor >= cooking_cons_scaling_factor[0]:
            threshold = threshold - cooking_cons_scaling_factor_offset
        if cons_factor >= cooking_cons_scaling_factor[1]:
            threshold = threshold - cooking_cons_scaling_factor_offset

    cook_potential[:, activity_curve < threshold] = 0
    cook_potential[cook_disagg > 0] = cooking_pot_score_at_nonzero_disagg_points

    # higher cooking potential on weekends

    weekend_days = np.logical_or(dow == 1, dow == 7)

    cook_potential[weekend_days][cook_potential[weekend_days] > 0] = \
        cook_potential[weekend_days][cook_potential[weekend_days] > 0] + cooking_potential_score_offset

    # Final sanity checks

    cook_confidence[cook_potential == 0] = 0

    cook_confidence = np.nan_to_num(cook_confidence)
    cook_confidence = np.fmin(1, cook_confidence)

    cook_potential = np.nan_to_num(cook_potential)
    cook_potential = np.fmin(1, cook_potential)

    cook_confidence[vacation] = 0
    cook_potential[vacation] = 0

    item_output_object["app_confidence"][app_index, :, :] = cook_confidence
    item_output_object["app_potential"][app_index, :, :] = cook_potential

    t_end = datetime.now()

    logger.info("Cook potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object
