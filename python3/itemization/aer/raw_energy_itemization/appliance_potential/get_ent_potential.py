"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update entertainment confidence and potential values
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.utils import postprocess_conf_arr

from python3.itemization.aer.raw_energy_itemization.appliance_potential.config.get_pot_conf import get_pot_conf


def get_ent_potential(app_index, item_input_object, item_output_object, sampling_rate, logger_pass):

    """
    Calculate entertainment confidence and potential values

    Parameters:
        app_index                   (int)           : Index of app in the appliance list
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        sampling_rate               (int)           : sampling rate
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object          (dict)          : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    config = get_pot_conf().get("ent")

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_ent_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetching required inputs

    ent_disagg = item_output_object.get("updated_output_data")[app_index, :, :]
    dow = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_DOW_IDX, :, 0]
    original_input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    if not np.sum(ent_disagg):
        return item_output_object

    ent_cons_score_for_late_night_hours = config.get('ent_cons_score_for_late_night_hours')
    ent_cons_potential_score_offset = config.get('ent_cons_potential_score_offset')
    activity_curve_thres = config.get('activity_curve_thres')
    activity_curve_cons_thres = config.get('activity_curve_cons_thres')

    ent_disagg = ent_disagg - np.percentile(ent_disagg[ent_disagg > 0], 5)

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    ent_confidence = np.ones(original_input_data.shape) * 0.1

    activity_curve = item_input_object.get("weekday_activity_curve")

    # Calculating ent confidence using activity curve for weekdays and weekends

    activity_curve = (activity_curve - np.percentile(activity_curve, 3)) / (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    normalized_input_data = (original_input_data[:] - np.percentile(original_input_data, 5,  axis=1)[:, None]) / \
                            (np.percentile(original_input_data, 95, axis=1) - np.percentile(original_input_data, 5, axis=1))[:, None]

    normalized_input_data = np.round(normalized_input_data, 2)

    weekend_days = np.logical_or(dow == 1, dow == 7)

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    total_cons = original_input_data[np.logical_not(vacation)]
    length = np.sum(np.logical_not(vacation))
    total_cons = ((np.sum(total_cons) / length) * (Cgbdisagg.DAYS_IN_MONTH / 1000))

    # for each time of the weekend days
    # assigning a score for ent being present, based on activity profile and consumption level of the user

    ent_confidence = np.multiply(ent_confidence, activity_curve[None, :])

    ent_potential = np.multiply(normalized_input_data, activity_curve[None, :]) + ent_cons_potential_score_offset

    threshold = (activity_curve_thres[0] * (total_cons > activity_curve_cons_thres[1])) + \
                activity_curve_thres[1] * (total_cons <= activity_curve_cons_thres[1])

    ent_potential[:, activity_curve < threshold] = 0

    activity_curve = item_input_object.get("weekend_activity_curve")
    activity_curve = (activity_curve - np.percentile(activity_curve, 3)) / (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    # for each time of the weekday days
    # assigning a score for ent being present, based on activity profile and consumption level of the user

    ent_potential_weekdays = np.multiply(normalized_input_data, activity_curve[None, :]) + ent_cons_potential_score_offset

    ent_potential_weekdays[:, activity_curve < threshold] = 0

    ent_potential[weekend_days] = 0
    ent_potential_weekdays[np.logical_not(weekend_days)] = 0

    ent_potential = ent_potential + ent_potential_weekdays

    ent_confidence[ent_potential == 0] = 0

    # Increasing ent potential on weekends

    ent_potential[weekend_days][ent_potential[weekend_days] > 0] = \
        ent_potential[weekend_days][ent_potential[weekend_days] > 0] + config.get("weekend_day_inc")

    postprocess_conf_arr(ent_confidence, ent_potential, ent_disagg, vacation)

    ent_confidence[np.logical_and(ent_disagg > 0, ent_confidence == 0)] = config.get("base_potential")
    ent_potential[np.logical_and(ent_disagg > 0, ent_potential == 0)] = config.get("base_potential")

    late_night_ent_usage_hours = get_index_array(item_output_object.get("profile_attributes").get("sleep_time") * samples_per_hour -
                                                 int(samples_per_hour / 2),
                                                 item_output_object.get("profile_attributes").get("sleep_time") * samples_per_hour +
                                                 int(samples_per_hour / 2) + 1,
                                                 samples_per_hour * Cgbdisagg.HRS_IN_DAY)

    # adding additional potential for ent during late night hours,
    # because usually other two stat app categories are not used during this time of the day

    ent_potential[:, late_night_ent_usage_hours] = ent_cons_score_for_late_night_hours
    ent_confidence[:, late_night_ent_usage_hours] = ent_cons_score_for_late_night_hours

    ent_potential = np.fmin(1, ent_potential)
    ent_confidence = np.fmin(1, ent_confidence)

    ent_potential = np.fmax(0, ent_potential)
    ent_confidence = np.fmax(0, ent_confidence)

    ent_potential[ent_disagg == 0] = 0
    ent_confidence[ent_disagg == 0] = 0

    # Dumping appliance confidence and potential values

    item_output_object["app_confidence"][app_index, :, :] = ent_confidence
    item_output_object["app_potential"][app_index, :, :] = ent_potential

    t_end = datetime.now()

    logger.info("Ent potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object
