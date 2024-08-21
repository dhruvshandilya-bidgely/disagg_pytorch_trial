
"""
Author - Nisha Agarwal
Date - 10th Nov 2020
Few post processing steps to modify lighting TOU
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile


def smoothen_lighting_hours(item_input_object, lighting_usage_potential, sunrise_sunset_data, lighting_config, logger_pass):

    """
    Smooth/Postprocess lighting usage potential output

    Parameters:
        item_input_object         (dict)        : Dict containing all hybrid inputs
        lighting_usage_potential    (np.ndarray)  : lighting usage potential (Normalized)
        sunrise_sunset_data         (np.ndarray)  : sunrise/sunset information
        lighting_config             (dict)        : dict containing lighting config values
        logger_pass                 (dict)        : Contains the logger and the logging dictionary to be passed on

    Returns:
        smooth_light_hours          (np.ndarray)  : post-processed lighting usage potential (Normalized)
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('smoothen_lighting_hours')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_lighting_start = datetime.now()

    lighting_usage_potential = np.nan_to_num(lighting_usage_potential)

    tou_percentile_cap = lighting_config.get('smooth_estimate_config').get('tou_percentile_cap')
    final_percentile_cap = lighting_config.get('smooth_estimate_config').get('final_percentile_cap')
    smoothing_window = lighting_config.get('smooth_estimate_config').get('smoothing_window')

    factor = int(item_input_object.get("item_input_params").get("samples_per_hour"))

    if np.all(lighting_usage_potential == 0):
        logger.info("lighting usage potential is zero throughout the timeframe")
        return lighting_usage_potential

    sunset_data = np.where(sunrise_sunset_data == 1)[1]

    lighting_usage_potential = np.nan_to_num(lighting_usage_potential)

    lighting_hours_copy = copy.deepcopy(lighting_usage_potential)

    # cap the values at 70th percentile for each timestamp

    lighting_usage_potential[lighting_usage_potential == 0] = np.nan

    tou_percentile = superfast_matlab_percentile(lighting_usage_potential, tou_percentile_cap, axis=0)

    tou_percentile = np.nan_to_num(tou_percentile)

    lighting_usage_potential = np.minimum(lighting_usage_potential, tou_percentile)

    lighting_usage_potential = np.nan_to_num(lighting_usage_potential)

    logger.info("Capped the potential values")

    # for each timestamp near sunrise / sunset , cap at 70th percentile

    lighting_usage_potential[lighting_hours_copy == 0] = 0

    length = len(lighting_usage_potential)

    for i in range(-factor, 2*factor):

        data_at_ts = lighting_hours_copy[np.arange(length), sunset_data+i]

        if not np.all(data_at_ts == 0):
            data_at_ts = data_at_ts[data_at_ts > 0]
            value = np.percentile(data_at_ts, tou_percentile_cap)
            lighting_usage_potential[np.arange(length), sunset_data + i][lighting_usage_potential[np.arange(length), sunset_data + i] > value] = value

    logger.info("Capped the sunrise-sunset neighbouring points")

    # smooth consumption using nearby timestamps estimates

    # This is done by replacing the value at a ts by average of values at (ts, ts+1, ts-1)

    smooth_light_hours = np.zeros(lighting_usage_potential.shape)

    lighting_usage_potential_right = np.roll(lighting_usage_potential, 1, axis=1)
    lighting_usage_potential_left = np.roll(lighting_usage_potential, -1, axis=1)
    valid_bool = lighting_usage_potential > 0

    smooth_light_hours[valid_bool] = (lighting_usage_potential + lighting_usage_potential_right + lighting_usage_potential_left)[valid_bool]

    division_factor = np.ones(smooth_light_hours.shape)*smoothing_window
    division_factor[lighting_usage_potential_right == 0] = division_factor[lighting_usage_potential_right == 0] - 1
    division_factor[lighting_usage_potential_left == 0] = division_factor[lighting_usage_potential_left == 0] - 1

    smooth_light_hours = np.divide(smooth_light_hours, division_factor)

    logger.info("Smoothen the potential values using neighbouring points")

    smooth_light_hours = smooth_light_hours / np.percentile(smooth_light_hours[np.nonzero(smooth_light_hours)], final_percentile_cap)

    smooth_light_hours = np.fmin(smooth_light_hours, 1)

    t_lighting_end = datetime.now()

    logger.debug("Smoothing of lighting potential done")

    logger.info("Smoothing of lighting potential took | %.3f s",
                get_time_diff(t_lighting_start, t_lighting_end))

    return smooth_light_hours


def post_process_lighting_tou(item_input_object, lighting_estimate, sunrise_sunset_data, lighting_config, logger_pass):

    """
    Smooth/Postprocess lighting estimate to decrease values away from sunrise/sunset hours

    Parameters:
        item_input_object         (dict)        : Dict containing all hybrid inputs
        lighting_estimate           (np.ndarray)  : lighting TOU estmates
        sunrise_sunset_data         (np.ndarray)  : sunrise/sunset information
        lighting_config             (dict)        : dict containing all lighting config values
        logger_pass                 (dict)        : Contains the logger and the logging dictionary to be passed on

    Returns:
        lighting_estimate           (np.ndarray)  : postprocessed lighting TOU estmates
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('post_process_lighting_tou')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_lighting_start = datetime.now()

    multiplier = lighting_config.get('postprocess_lighting_config').get('multiplier')
    decrement_factor = lighting_config.get('postprocess_lighting_config').get('decreament_factor')

    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")

    # decrease the values of estimate far from sunrise and sunset hours

    total_samples = samples_per_hour * Cgbdisagg.HRS_IN_DAY

    sunset_hours = np.where(sunrise_sunset_data == lighting_config.get('sunrise_sunset_config').get('sunset_val'))[1]
    sunrise_hours = np.where(sunrise_sunset_data == lighting_config.get('sunrise_sunset_config').get('sunrise_val'))[1]

    morn_buffer_inc = lighting_config.get('usage_potential_config').get('morn_buffer_inc')
    eve_buffer_inc = lighting_config.get('usage_potential_config').get('eve_buffer_inc')
    morn_buffer_buc = lighting_config.get('usage_potential_config').get('morn_buffer_buc')
    eve_buffer_buc = lighting_config.get('usage_potential_config').get('eve_buffer_buc')

    for day in range(len(lighting_estimate)):

        # For each day , the lighting consumption is decreased before the sunset hours and after the sunrise hours

        # The decreament factor is a function of distance of target timestamp
        # from sunrise/sunset ts of the particaular day

        # Post process morning band

        after_sunrise_hours = lighting_config.get('usage_potential_config').get('after_sunrise_hours')

        sunrise_sunset_diff = (sunset_hours[day] - sunrise_hours[day])
        morn_buffer_increment = morn_buffer_inc[np.digitize([sunrise_sunset_diff], morn_buffer_buc)[0]]

        after_sunrise_hours = (after_sunrise_hours + int(morn_buffer_increment)) * samples_per_hour

        sunrise = np.where(sunrise_sunset_data[day] == lighting_config.get('sunrise_sunset_config').get('sunrise_val'))[0]

        # List of ts after sunrise ts
        index_list = np.arange(int(sunrise + samples_per_hour + 1), int(sunrise + after_sunrise_hours + 1)) % total_samples

        # Array of decrement factors
        temp_multiplier = np.arange(multiplier - (len(index_list)*decrement_factor / samples_per_hour), multiplier,
                                    decrement_factor / samples_per_hour)[::-1][:len(index_list)]
        lighting_estimate[day, index_list] = np.multiply(lighting_estimate[day, index_list], temp_multiplier)

        before_sunset_hours = lighting_config.get('usage_potential_config').get('before_sunset_hours')

        # Post process evening band

        sunrise_sunset_diff = (sunset_hours[day] - sunrise_hours[day])
        eve_buffer_increment = eve_buffer_inc[np.digitize([sunrise_sunset_diff], eve_buffer_buc)[0]]

        before_sunset_hours = (before_sunset_hours + int(eve_buffer_increment)) * samples_per_hour

        sunset = np.where(sunrise_sunset_data[day] == lighting_config.get('sunrise_sunset_config').get('sunset_val'))[0]

        # List of ts before sunset ts
        index_list = np.arange(int(sunset - before_sunset_hours), int(sunset - samples_per_hour)) % total_samples

        # Array of decrement factors
        temp_multiplier = np.arange(multiplier - (len(index_list)*decrement_factor / samples_per_hour), multiplier,
                                    decrement_factor / samples_per_hour)[:len(index_list)]
        lighting_estimate[day, index_list] = np.multiply(lighting_estimate[day, index_list], temp_multiplier)

    t_lighting_end = datetime.now()

    logger.info("Post processing of lighting estimate took | %.3f s",
                get_time_diff(t_lighting_start, t_lighting_end))

    lighting_estimate = np.fmax(lighting_estimate, 0)

    return lighting_estimate
