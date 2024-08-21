
"""
Author - Nisha Agarwal
Date - 10th Nov 2020
Calculate timestamp level lighting usage capacity
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
from python3.itemization.aer.functions.itemization_utils import fill_array

from python3.itemization.init_itemization_config import init_itemization_params


def get_usage_potential(sunrise_sunset, samples_per_hours, lighting_config, logger):

    """
    Calculate lighting usage potential using sunrise/sunset data

    Parameters:
        sunrise_sunset           (np.ndarray)      : sunrise/sunset data
        samples_per_hours        (int)             : Number of samples in an hour
        lighting_config          (dict)            : dict containing lighting config values
        logger                   (logger)          : logger object

    Returns:
        usage_potential          (np.ndarray)      : lighting usage potential using sunrise/sunset data
    """

    # Calculate lighting usage potential using sunrise sunset data

    usage_potential = np.zeros(sunrise_sunset.shape)

    sunset_hours = np.where(sunrise_sunset == lighting_config.get('sunrise_sunset_config').get('sunset_val'))[1]

    sunrise_hours = np.where(sunrise_sunset == lighting_config.get('sunrise_sunset_config').get('sunrise_val'))[1]

    morn_buffer_inc = lighting_config.get('usage_potential_config').get('morn_buffer_inc')
    eve_buffer_inc = lighting_config.get('usage_potential_config').get('eve_buffer_inc')
    morn_buffer_buc = lighting_config.get('usage_potential_config').get('morn_buffer_buc')
    eve_buffer_buc = lighting_config.get('usage_potential_config').get('eve_buffer_buc')

    for day in range(min(len(sunset_hours), len(sunrise_hours))):

        logger.debug("Calculating usage hours for day | %d", day)

        before_sunset_hours = lighting_config.get('usage_potential_config').get('before_sunset_hours')
        after_sunrise_hours = lighting_config.get('usage_potential_config').get('after_sunrise_hours')

        sunrise_sunset_diff = (sunset_hours[day] - sunrise_hours[day])
        morn_buffer_increment = morn_buffer_inc[np.digitize([sunrise_sunset_diff], morn_buffer_buc)[0]]
        eve_buffer_increment = eve_buffer_inc[np.digitize([sunrise_sunset_diff], eve_buffer_buc)[0]]

        after_sunrise_hours = (after_sunrise_hours + morn_buffer_increment)
        before_sunset_hours = (before_sunset_hours + eve_buffer_increment)

        usage_potential[day, int(sunset_hours[day] - int(before_sunset_hours * samples_per_hours)):] = 1

        usage_potential[day, np.arange(0, sunrise_hours[day] + int(after_sunrise_hours * samples_per_hours) + 1)] = 1

    return usage_potential


def get_lighting_hours(item_input_object, input_data, sleeping_hours, sunrise_sunset, lighting_config, debug, logger_pass):

    """
    Calculate lighting usage potential using sunrise/sunset data

    Parameters:
        input_data                        (np.ndarray)      : input data
        sleeping_hours                    (np.ndarray)      : sleeping hours boolean array
        sunrise_sunset                    (np.ndarray)      : sunrise/sunset data
        lighting_config                   (dict)            : dict containing lighting config values
        debug                             (dict)            : debug object
        logger_pass                       (dict)            : Contains the logger and the logging dictionary to be passed on

    Returns:
        combined_lighting_potential       (np.ndarray)      : lighting usage potential (Normalized)
        lighting_potential                (np.ndarray)      : Lighting potential using only sunrise sunset data
        debug                             (dict)            : debug object
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_lighting_hours')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_lighting_start = datetime.now()

    samples_per_hour = int(input_data.shape[1] / Cgbdisagg.HRS_IN_DAY)
    activity_curve = item_input_object.get("activity_curve")

    # Normalize daily input data

    input_data_copy = copy.deepcopy(input_data) + item_input_object.get("item_input_params").get("timed_cons")

    input_data_copy = np.minimum(input_data_copy, np.percentile(input_data_copy, 95))

    normalized_input_data = (input_data_copy[:] - np.min(input_data_copy, axis=1)[:, None]) / \
                             (np.max(input_data_copy, axis=1) - np.min(input_data_copy, axis=1))[:, None]

    normalized_input_data = np.round(normalized_input_data, 2)

    # get lighting usage potential

    lighting_potential = get_usage_potential(sunrise_sunset, samples_per_hour, lighting_config, logger)

    debug.update({
        "lighting_potential_using_sunrise_sunset": lighting_potential
    })

    logger.debug("Calculated lighting usage potential using sunrise-sunset")

    # Post processing for more than 2 lighting bands

    sleeping_hours = fill_sleeping_hours(sleeping_hours, samples_per_hour, lighting_config)

    # No lighting during inactive hours

    normalized_input_data[:, sleeping_hours == 0] = 0

    # normalize activity curve

    activity_curve = (activity_curve - np.min(activity_curve)) / (np.max(activity_curve) - np.min(activity_curve))
    activity_curve[activity_curve == 0] = 0.01

    activity_curve_2d = np.zeros(normalized_input_data.shape)

    activity_curve_2d[:, :] = activity_curve

    # lighting potential using input data / active hours

    active_usage_hours = np.minimum(normalized_input_data, activity_curve_2d)

    logger.debug("Calculated lighting usage potential using activity profile")

    # combine both potential

    combined_lighting_potential = np.multiply(lighting_potential, active_usage_hours)

    debug.update({
        "lighting_potential_using_sunrise_sunset": active_usage_hours
    })

    t_lighting_end = datetime.now()

    logger.debug("Calculated final lighting usage hours")

    logger.info("Calculating of lighting usage hours took | %.3f s",
                get_time_diff(t_lighting_start, t_lighting_end))

    return combined_lighting_potential, lighting_potential, debug


def fill_sleeping_hours(sleeping_hours, samples_per_hour, lighting_config):

    """
    Fill sleeping hours to avoid more than 2 lighting bands

    Parameters:
        sleeping_hours                    (np.ndarray)      : sleeping hours boolean array
        samples_per_hour                  (int)             : Number of samples in an horu
        lighting_config                   (dict)            : dict containing lighting config values

    Returns:
        sleeping_hours                    (np.ndarray)      : updated sleeping hours boolean array
    """

    # Handle multiple bands scenario

    length = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    seq_config = init_itemization_params().get("seq_config")

    morning_band_hours = lighting_config.get("general_config").get("morn_band_hours")
    evening_band_hours = lighting_config.get("general_config").get("eve_band_hours")

    morning_tou = sleeping_hours[morning_band_hours]
    evening_tou = sleeping_hours[evening_band_hours % length]

    morning_seq = find_seq(morning_tou, np.zeros(morning_tou.shape), np.zeros(morning_tou.shape), overnight=0)
    evening_seq = find_seq(evening_tou, np.zeros(evening_tou.shape), np.zeros(evening_tou.shape), overnight=0)

    # to identify whether morning or evening hours have 2 lighting bands
    # If less than fill the gap if the length of the gap is less than a certain limit

    if len(morning_seq) > 1 and morning_seq[-1, 0] == 0:
        morning_seq = morning_seq[:-1]

    if len(morning_seq) > 1 and evening_seq[-1, 0] == 0:
        evening_seq = evening_seq[:-1]

    if len(morning_seq) > 1 and morning_seq[0, 0] == 0:
        morning_seq = morning_seq[1:]

    if len(evening_seq) > 1 and evening_seq[0, 0] == 0:
        evening_seq = evening_seq[1:]

    morning_seq = morning_seq[np.logical_and(morning_seq[:, seq_config.get("label")] == 0,
                                             morning_seq[:, seq_config.get("length")] <=
                                             lighting_config.get("general_config").get("morn_band_limit"))]
    evening_seq = evening_seq[np.logical_and(evening_seq[:, seq_config.get("label")] == 0,
                                             evening_seq[:, seq_config.get("length")] <=
                                             lighting_config.get("general_config").get("eve_band_limit"))]

    start = 0
    end = len(morning_seq)

    # Fill the sleep hours boolean array using the updated array of sequences

    for i in range(start, end):
        sleeping_hours = fill_array(sleeping_hours, int(morning_seq[i, seq_config.get("start")] + morning_band_hours[0]),
                                    int(morning_seq[i, seq_config.get("end")] + morning_band_hours[0]), 1)

    start = 0
    end = len(evening_seq)

    for i in range(start, end):
        sleeping_hours = fill_array(sleeping_hours, int(evening_seq[i, seq_config.get("start")] + evening_band_hours[0]),
                                    int(evening_seq[i, seq_config.get("end")] + evening_band_hours[0]), 1)

    return sleeping_hours
