
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Calculate laundry confidence and potential values
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.appliance_potential.config.get_pot_conf import get_pot_conf


def get_ld_potential(app_index, item_input_object, item_output_object, sampling_rate, vacation,  logger_pass):

    """
    Calculate laundry confidence and potential values

    Parameters:
        app_index                   (int)           : Index of app in the appliance list
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        sampling_rate               (int)           : sampling rate
        vacation                    (np.ndarray)    : array of vacation days
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_ld_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    config = get_pot_conf().get('ld')

    conf_score_at_non_zero_disagg = config.get('conf_score_at_non_zero_disagg')
    weekdays_ld_potential_score_buffer = config.get('weekdays_ld_potential_score_buffer')
    weekends_ld_potential_score_buffer = config.get('weekends_ld_potential_score_buffer')

    # Fetching required inputs

    ld_disagg = item_output_object.get("updated_output_data")[app_index, :, :]
    sleep_hours = item_output_object.get("profile_attributes").get("sleep_hours")
    dow = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.DAYS_IN_WEEK, :, 0]
    original_input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    ld_confidence = np.ones(original_input_data.shape) * 0.1

    activity_curve = item_output_object.get("debug").get("profile_attributes_dict").get("activity_curve")

    activity_curve = (activity_curve - np.percentile(activity_curve, 3)) / (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))
    activity_curve = np.fmax(0, np.fmin(1, activity_curve))

    # Calculate laundry potential using laundry tou and activity curve

    ld_confidence[np.logical_and(ld_disagg > 0, ld_confidence == 0)] = conf_score_at_non_zero_disagg

    normalized_input_data = (original_input_data[:] - np.min(original_input_data, axis=1)[:, None]) / \
                            (np.max(original_input_data, axis=1) - np.min(original_input_data, axis=1))[:, None]

    normalized_input_data = np.round(normalized_input_data, 2)

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')
    sleep_hours[get_index_array(item_output_object.get("profile_attributes").get("sleep_time") * samples_per_hour,
                                item_output_object.get("profile_attributes").get("sleep_time") * samples_per_hour +
                                int(samples_per_hour) + 1,
                                samples_per_hour * Cgbdisagg.HRS_IN_DAY)] = 1

    activity_curve[sleep_hours] = 0

    ld_confidence = np.multiply(ld_confidence, activity_curve[None, :]/np.max(activity_curve))

    ld_potential = np.multiply(normalized_input_data, activity_curve[None, :]/np.max(activity_curve))

    weekend_days = np.logical_or(dow == 1, dow == 7)

    ld_potential[ld_potential > 0] = ld_potential[ld_potential > 0] + weekdays_ld_potential_score_buffer

    # Giving higher laundry potential on weekends

    ld_potential[weekend_days][ld_potential[weekend_days] > 0] = \
        ld_potential[weekend_days][ld_potential[weekend_days] > 0] + weekends_ld_potential_score_buffer

    # Final sanity checks

    ld_confidence[ld_potential == 0] = 0

    ld_potential = np.fmax(0, ld_potential)
    ld_potential = np.fmin(1, ld_potential)

    ld_confidence = np.nan_to_num(ld_confidence)
    ld_confidence = np.fmin(1, ld_confidence)

    ld_potential = np.nan_to_num(ld_potential)
    ld_potential = np.fmin(1, ld_potential)

    ld_confidence[vacation] = 0
    ld_potential[vacation] = 0

    item_output_object["app_confidence"][app_index, :, :] = ld_confidence
    item_output_object["app_potential"][app_index, :, :] = ld_potential

    t_end = datetime.now()

    logger.info("LD potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object
