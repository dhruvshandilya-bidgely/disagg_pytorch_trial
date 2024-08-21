
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Estimation dinner cooking consumption
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import add_noise_to_stat_app_est

from python3.itemization.aer.behavioural_analysis.home_profile.config.occupancy_profile_config import get_occupancy_profile_config


def get_dinner_usage(input_data, occupancy_profile, user_parameters, cooking_app_count, cooking_type, dinner_delta,
                     default_cooking_flag, cooking_config, logger_pass):

    """
       Estimate tou consumption of dinner cooking

       Parameters:
           input_data            (np.ndarray)         : Day input data
           occupancy_profile     (np.ndarray)         : Occupancy profile
           user_parameters       (dict)               : dict containing User attributes
           cooking_app_count     (np.ndarray)         : count of various cooking appliance categories
           cooking_type          (np.ndarray)         : arr of type for all cooking appliances (1 if electric , 0 if gas)
           dinner_delta          (float)              : energy delta during dinner hours
           default_cooking_flag  (bool)               : True if the given cooking app count is default
           cooking_config        (dict)               : Config containing lunch parameters
           logger_pass           (dict)               : Contains the logger and the logging dictionary to be passed on

       Returns:
           consumption           (np.ndarray)         : TOU level dinner estimate
       """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_dinner_usage')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    office_goer_index = get_occupancy_profile_config().get("general_config").get("office_goer_index")

    t_cooking_start = datetime.now()

    samples_per_hour = int(input_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    appliance_consumption = cooking_config.get("appliance_consumption")

    # Calculate appliance count using delta if default app count is available

    if default_cooking_flag:

        logger.debug("Using default cooking app count for dinner usage")

        cooking_app_count = cooking_config.get("cooking_app_count_arr")[
            np.digitize(dinner_delta, cooking_config.get("delta_arr"))]

    cooking_app_count = np.array(cooking_app_count)

    logger.info("Dinner energy delta | %d", dinner_delta)

    # Minor cooking consumption for gas based cooking

    cooking_app_count[cooking_type == 0] = cooking_app_count[cooking_type == 0] * 0

    appliance_consumption = np.dot(appliance_consumption, cooking_app_count)

    # Fetch cooking usage hours

    usage_hours = cooking_config.get("usage_hours") * samples_per_hour

    total_samples = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    # Modify sleeping hours for late night active users

    if user_parameters.get("sleep_time") == 0:
        user_parameters["sleep_time"] = total_samples - 1

    if user_parameters.get("sleep_time") < cooking_config.get("late_night_hour") * samples_per_hour:
        user_parameters['sleep_time'] = user_parameters.get('sleep_time') + Cgbdisagg.HRS_IN_DAY * samples_per_hour

    # If high office goers, extra dinner usage is provided

    if occupancy_profile[office_goer_index] == cooking_config.get("high_office_goers_limit"):
        usage_hours = np.array([cooking_config.get("high_office_goers_usage_hours")]) * samples_per_hour

    if np.sum(occupancy_profile) > cooking_config.get("occupants_limit"):
        appliance_consumption = appliance_consumption * cooking_config.get("high_occupants_multiplier")

    consumption = np.zeros(input_data.shape)

    # If not office arrival time is found, default is assumed

    if user_parameters.get('office_coming_time') == -1:
        user_parameters['office_coming_time'] = cooking_config.get("default_dinner_start") * samples_per_hour

    night_time_consistent_activity_absent = user_parameters.get('office_coming_time') > cooking_config.get("default_dinner_end") * samples_per_hour

    if night_time_consistent_activity_absent:
        appliance_consumption = cooking_config.get("absent_dinner_consumption")

    # Calculate dinner cooking hours

    start_time = user_parameters.get('dinner_start')
    end_time = start_time + usage_hours

    night_time_consistent_activity_absent = user_parameters.get("dinner_start") == -1

    if night_time_consistent_activity_absent:
        logger.info("Dinner time not found, taking default | ")
        appliance_consumption = cooking_config.get("absent_dinner_consumption")
        start_time = cooking_config.get("default_dinner_start") * samples_per_hour

    if samples_per_hour == 1:
        appliance_consumption = appliance_consumption * cooking_config.get("60_min_multiplier")

    logger.info("Dinner start time | %d", start_time)
    logger.info("Dinner end time | %d", end_time)

    logger.debug("Initialization done for calculation of dinner usage")

    consumption = add_noise_to_stat_app_est(consumption, start_time, end_time, usage_hours, total_samples,
                                            appliance_consumption / samples_per_hour)

    t_cooking_end = datetime.now()

    logger.info("Calculation of dinner usage took | %.3f s", get_time_diff(t_cooking_start, t_cooking_end))

    return consumption
