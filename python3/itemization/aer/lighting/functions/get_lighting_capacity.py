
"""
Author - Nisha Agarwal
Date - 10th Nov 2020
Calculate lighting usage capacity
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

from python3.itemization.init_itemization_config import init_itemization_params


def get_living_load(average_sunrise, average_sunset, active_hours, clean_days_consumption, samples_per_hour, activity_curve, config, logger):

    """
    Calculate morning and evening living load activity

    Parameters:
        average_sunrise                 (float)          : Average sunrise hours of the user
        average_sunset                  (float)          : Average sunset hours of the user
        active_hours                    (np.ndarray)     : active/non active hours bool array
        clean_days_consumption          (np.ndarray)     : consumption of top cleanest days of a user
        samples_per_hour                (int)            : samples in an hour
        activity_curve                  (np.ndarray)     : living load activity profile
        config                          (dict)           : dict containing lighting config values

    Returns:
        morning_delta                   (int)             : Morning living load activity
        evening_delta                   (int)             : Evening living load activity
    """

    # Initialize morning and evening delta

    morning_delta = 0
    evening_delta = 0

    if np.all(clean_days_consumption == 0):
        logger.info("Clean day consumption is zero throughout, Not calculating activity delta ")
        return morning_delta, evening_delta, morning_delta, evening_delta

    itemization_config = init_itemization_params().get('seq_config')

    morning_hours = np.zeros(len(active_hours))
    morning_hours[config.get('morning_start') * samples_per_hour: int(average_sunrise + config.get('after_sunrise_hours') * samples_per_hour)] = 1
    morning_hours = np.logical_and(morning_hours, active_hours)

    # Calculate morning living load using morning activity change

    label_idx = itemization_config.get('label')
    low_perc_idx = itemization_config.get('low_perc')
    high_perc_idx = itemization_config.get('high_perc')

    if not(np.all(morning_hours == 0)):

        morning_seq = find_seq(morning_hours, clean_days_consumption, np.zeros(len(morning_hours)))

        index = np.where(morning_seq[:, label_idx] == 1) if \
            np.sum(morning_seq[:, label_idx] == 1) < 1 else np.where(morning_seq[:, label_idx] == 1)[0]

        morning_delta = np.max(morning_seq[index, high_perc_idx] -
                               np.minimum(np.roll(morning_seq[(index-1)%len(morning_seq), low_perc_idx], 1),
                                          np.roll(morning_seq[(index+1)%len(morning_seq), low_perc_idx], -1)))

    evening_hours = np.zeros(len(active_hours))
    evening_hours[int(average_sunset - config.get('before_sunset_hours') * samples_per_hour):] = 1
    evening_hours[: int(config.get('evening_end') * samples_per_hour)] = 1
    evening_hours = np.logical_and(evening_hours, active_hours)

    # Calculate evening living load using evening activity change

    if not (np.all(evening_hours == 0)):
        evening_seq = find_seq(evening_hours, clean_days_consumption, np.zeros(len(morning_hours)))

        index = np.where(evening_seq[:, label_idx] == 1) \
            if np.sum(evening_seq[:, label_idx] == 1) < 1 else np.where(evening_seq[:, label_idx] == 1)[0]

        evening_delta = np.max(evening_seq[index, high_perc_idx] -
                               np.minimum(np.roll(evening_seq[(index-1)%len(evening_seq), low_perc_idx], 1),
                                          np.roll(evening_seq[(index+1)%len(evening_seq), low_perc_idx], -1)))

    logger.info("calculated evening delta | %d", evening_delta)
    logger.info("calculated morning delta | %d", morning_delta)

    original_evening_delta = copy.deepcopy(evening_delta)
    original_morning_delta = copy.deepcopy(morning_delta)

    evening_delta = evening_delta if evening_delta > 0 else morning_delta
    morning_delta = morning_delta if morning_delta > 0 else evening_delta

    activity_curve_diff = np.max(activity_curve) - np.min(activity_curve)

    clean_days_cons_diff = np.max(clean_days_consumption) - np.min(clean_days_consumption)

    # Safety checks to resolve underestimation cases

    even_delta_bool = evening_delta < config.get('safety_check_max_limit')/samples_per_hour
    morn_delta_bool = morning_delta < config.get('safety_check_max_limit')/samples_per_hour
    clean_day_cons_bool = clean_days_cons_diff < config.get('safety_check_fraction')*activity_curve_diff

    if even_delta_bool and clean_day_cons_bool:
        evening_delta = evening_delta * config.get('safety_check_multiplier')

    if morn_delta_bool and clean_day_cons_bool:
        morning_delta = morning_delta * config.get('safety_check_multiplier')

    even_delta_bool = original_evening_delta < config.get('safety_check_max_limit')/samples_per_hour
    morn_delta_bool = original_morning_delta < config.get('safety_check_max_limit')/samples_per_hour

    if even_delta_bool and clean_day_cons_bool:
        original_evening_delta = original_evening_delta * config.get('safety_check_multiplier')

    if morn_delta_bool and clean_day_cons_bool:
        original_morning_delta = original_morning_delta * config.get('safety_check_multiplier')

    return morning_delta, evening_delta, original_morning_delta, original_evening_delta


def get_lighting_capacity(item_input_object, item_output_object, top_clean_days, lighting_config, debug, logger_pass):

    """
    Run modules to calculate lighting usage capacity

    Parameters:
        item_input_object             (dict)            : Dict containing all hybrid inputs
        item_output_object            (dict)            : Dict containing all hybrid outputs
        top_clean_days                  (int)             : Indexes of top cleanest days
        levels_count                    (int)             : Number of levels of activity of the user
        lighting_config                 (dict)            : dict containing lighting config values
        debug                           (dict)            : debug object
        logger_pass                     (logger_pass)     : Contains the logger and the logging dictionary to be passed on

    Returns:
        estimate                        (int)             : lighting usage capacity
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_lighting_capacity')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_lighting_start = datetime.now()

    # Fetch required input parameters

    activity_curve = item_input_object.get('activity_curve')
    active_hours = item_output_object.get('profile_attributes').get('sleep_hours')
    input_data = item_input_object.get("item_input_params").get("input_data")
    day_input_data = item_input_object.get("item_input_params").get("day_input_data")
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")
    levels_count = len(np.unique(item_output_object.get('profile_attributes').get('activity_levels')))

    living_load_config = lighting_config.get('living_load_config')
    capacity_config = lighting_config.get('lighting_capacity_config')

    # Calculate average sunrise sunset times of the user

    average_sunrise = np.mean(input_data[Cgbdisagg.INPUT_SUNRISE_IDX, :, 0])
    average_sunset = np.mean(input_data[Cgbdisagg.INPUT_SUNSET_IDX, :, 0])

    # Normalize activity curve

    activity_curve = (activity_curve - np.min(activity_curve)) / (np.max(activity_curve) - np.min(activity_curve))

    # percentile of top clean days consumption

    clean_days_consumption = day_input_data[top_clean_days]

    if np.sum(clean_days_consumption) == 0:
        clean_days_consumption = np.zeros(len(day_input_data[0]))
    else:
        clean_days_consumption = clean_days_consumption[~np.all(clean_days_consumption == 0, axis=1)]

    percentile_consumption = np.percentile(clean_days_consumption, living_load_config.get('consumption_perc'), axis=0)

    debug.update({
        "cleanest_days_consumption": percentile_consumption
    })

    activity_curve = activity_curve * np.max(clean_days_consumption.mean(axis=0))

    # Get morning and evening living load consumption

    morning_delta, evening_delta, original_morning_delta, original_evening_delta = \
        get_living_load(average_sunrise, average_sunset, active_hours, percentile_consumption, samples_per_hour, activity_curve, living_load_config, logger)

    logger.info("Morning delta | %s ", morning_delta)
    logger.info("Evening delta | %s ", evening_delta)

    # get lighting capacity

    capacity = calculate_lighting_capacity(item_input_object, morning_delta, evening_delta, levels_count,
                                           samples_per_hour, capacity_config, logger)

    original_capacity = calculate_lighting_capacity(item_input_object, original_morning_delta,
                                                    original_evening_delta, levels_count,
                                                    samples_per_hour, capacity_config, logger)

    factor = original_capacity/capacity

    logger.info("Final hourly lighting capacity | %s ", (capacity * samples_per_hour))

    t_lighting_end = datetime.now()

    logger.info("Calculation of lighting capacity took | %.3f s",
                get_time_diff(t_lighting_start, t_lighting_end))

    return capacity, debug, factor


def calculate_lighting_capacity(item_input_object, morning_delta, evening_delta, levels, samples_per_hour, config, logger):

    """
    Calculate lighting usage capacity

    Parameters:
        item_input_object             (dict)            : Dict containing all hybrid inputs
        morning_delta                   (int)             : Morning living load activity
        evening_delta                   (int)             : Evening living load activity
        levels                          (int)             : Number of levels of activity of the user
        samples_per_hour                (int)             : samples in an hour
        config                          (dict)            : dict containing lighting config values
        logger                          (logger)          : logger object

    Returns:
        estimate                        (int)             : lighting usage capacity
    """

    # Relation of morning and evening delta with lighting capacity

    multiplier = max(1, samples_per_hour/config.get('multiplier_factor'))

    limit = config.get('max_limit') / samples_per_hour

    # upper limit of lighting capacity

    morning = np.fmin(config.get('max_limit'), config.get('morning_val'))
    evening = np.fmin(config.get('max_limit'), config.get('evening_val'))

    if morning_delta*multiplier > config.get('max_limit')/config.get('last_bucket'):
        morning_estimate = limit
    else:
        morning_estimate = morning[int(np.round(morning_delta * multiplier, -1) / 10)] / multiplier

    if evening_delta*multiplier > config.get('max_limit')/config.get('last_bucket'):
        evening_estimate = limit
    else:
        evening_estimate = evening[int(np.round(evening_delta*multiplier, -1) / 10)] / multiplier

    logger.info("Calculated hour level morning capacity | %d", (morning_estimate*samples_per_hour))
    logger.info("Calculated hour level evening capacity | %d", (evening_estimate*samples_per_hour))

    estimate = morning_estimate * config.get('morning_estimate_weightage') + evening_estimate * config.get('evening_estimate_weightage')

    logger.info("Calculated hour level lighting capacity | %d", (estimate*samples_per_hour))

    # Modification based on available meta features

    num_of_occupants = item_input_object.get("home_meta_data").get("numOccupants")

    if num_of_occupants == 'default':
        num_of_occupants = 0

    if num_of_occupants is not None and num_of_occupants >= 1:
        occupants_index = np.digitize(num_of_occupants, config.get('occupants_bucket'))
        percentage = config.get('occupants_multiplier')[min(len(config.get('occupants_bucket')) - 1, occupants_index)]
        estimate = estimate + (percentage / 100) * estimate

    num_of_rooms = item_input_object.get("home_meta_data").get("bedrooms")

    if num_of_rooms == 'default':
        num_of_rooms = 0

    if num_of_rooms is not None and num_of_rooms >= 1:
        rooms_index = np.digitize(num_of_rooms, config.get('rooms_bucket'))
        percentage = config.get('rooms_multiplier')[min(len(config.get('rooms_bucket')) - 1, rooms_index)]
        estimate = estimate + (percentage / 100) * estimate

    if (num_of_occupants is None or num_of_occupants == 0) and (num_of_rooms is None or num_of_rooms == 0):
        level_index = np.digitize(levels, config.get('levels_bucket'))
        percentage = config.get('levels_multiplier')[min(len(config.get('levels_bucket')) - 1, level_index)]
        estimate = estimate + (percentage / 100) * estimate

    logger.info("Number of occupants | %s", num_of_occupants)
    logger.info("Number of Rooms | %s", num_of_rooms)

    logger.info("Calculated hour level lighting capacity after scaling | %d", (estimate*samples_per_hour))

    max_limit = config.get('max_limit') / samples_per_hour
    min_limit = config.get('min_limit') / samples_per_hour

    # final sanity checks

    estimate = min(max_limit, max(min_limit, estimate))

    return estimate
