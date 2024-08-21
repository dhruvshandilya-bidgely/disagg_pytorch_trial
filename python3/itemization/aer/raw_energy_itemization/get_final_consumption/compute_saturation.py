
"""
Author - Mayank Sharan
Date - 4th April 2021
Computes expected saturation at each time of day based on the extracted activity curve
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.init_final_item_config import init_final_item_conf


def compute_saturation(item_input_object, item_output_object, item_adjustment_dict):

    """
    Compute expected saturation at each time of day based on the extracted activity curve

    Parameters:
        item_input_object           (dict)      : Dictionary containing all inputs needed for hybrid modules
        item_output_object          (dict)      : Dictionary containing all outputs from hybrid modules run so far
        item_adjustment_dict        (dict)      : Dictionary containing variables for adjusting itemization

    Returns:
        item_adjustment_dict        (dict)      : Dictionary containing variables for adjusting itemization
    """

    # Extract the profile attributes for the user

    profile_attributes = item_output_object.get('profile_attributes')
    activity_sequences = profile_attributes.get('activity_sequences')
    activity_curve = (profile_attributes.get('activity_curve'))
    sleep_hours = profile_attributes.get('sleep_hours')
    processed_input_data = copy.deepcopy(item_output_object.get('original_input_data'))
    processed_input_data[processed_input_data == 0] = 1

    # Extract additional data needed

    sampling_rate = item_input_object.get('config').get('sampling_rate')

    config = init_final_item_conf().get("sat_conf")

    num_samples_per_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)

    # Initialize the saturation arrays

    min_saturation_arr = np.zeros(shape=(num_samples_per_day,))
    mid_saturation_arr = np.zeros(shape=(num_samples_per_day,))
    max_saturation_arr = np.zeros(shape=(num_samples_per_day,))

    # Apply individual domain based rules to fill saturation array

    # For the sleeping hours set the saturation to high

    sleep_bool = sleep_hours == 0

    min_saturation_arr[sleep_bool] = config.get("sleep_min")
    mid_saturation_arr[sleep_bool] = config.get("sleep_mid")
    max_saturation_arr[sleep_bool] = config.get("sleep_max")

    # For all other activity hours in the day allow lower saturation
    # Lower further for time periods with rising activity and higher in falling activity

    increasing_activity_bool = np.logical_and(np.logical_not(sleep_bool), activity_sequences == 1)
    decreasing_activity_bool = np.logical_and(np.logical_not(sleep_bool), activity_sequences == -1)
    constant_activity_bool = np.logical_and(np.logical_not(sleep_bool), activity_sequences == 0)

    min_saturation_arr[increasing_activity_bool] = config.get("inc_min")
    mid_saturation_arr[increasing_activity_bool] = config.get("inc_mid")
    max_saturation_arr[increasing_activity_bool] = config.get("inc_max")

    min_saturation_arr[decreasing_activity_bool] = config.get("dec_min")
    mid_saturation_arr[decreasing_activity_bool] = config.get("dec_mid")
    max_saturation_arr[decreasing_activity_bool] = config.get("dec_max")

    min_saturation_arr[constant_activity_bool] = config.get("cons_min")
    mid_saturation_arr[constant_activity_bool] = config.get("cons_mid")
    max_saturation_arr[constant_activity_bool] = config.get("cons_max")

    # Adjust saturation values using the activity curve

    left_shift_ac = np.r_[activity_curve[1:], activity_curve[0]]
    right_shift_ac = np.r_[activity_curve[-1], activity_curve[:-1]]

    left_delta = activity_curve - right_shift_ac
    right_delta = activity_curve - left_shift_ac

    zero_arr = np.zeros_like(activity_curve)
    ones_arr = np.ones_like(activity_curve)

    max_pos_delta = np.maximum(zero_arr, np.maximum(left_delta, right_delta))
    max_neg_delta = np.minimum(zero_arr, np.minimum(left_delta, right_delta))

    factor = 0.03

    max_saturation_arr = np.minimum(max_saturation_arr + factor * max_pos_delta, ones_arr)
    min_saturation_arr = np.maximum(min_saturation_arr + factor * max_neg_delta, zero_arr)
    mid_saturation_arr = np.minimum(np.maximum(mid_saturation_arr + factor * (max_neg_delta + max_pos_delta),
                                               min_saturation_arr),
                                    max_saturation_arr)

    item_adjustment_dict['max_saturation_arr'] = max_saturation_arr
    item_adjustment_dict['min_saturation_arr'] = min_saturation_arr
    item_adjustment_dict['mid_saturation_arr'] = mid_saturation_arr

    return item_adjustment_dict
