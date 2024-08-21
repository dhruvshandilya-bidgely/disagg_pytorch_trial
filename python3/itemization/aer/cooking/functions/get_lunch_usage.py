
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Estimate consumption for lunch usage
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


def get_lunch_usage(input_data, occupancy_profile, user_parameters, cooking_app_count, cooking_type, lunch_present,
                    lunch_delta, default_cooking_flag, cooking_config, logger_pass):

    """
       Estimate tou consumption of lunch cooking

       Parameters:
           input_data            (np.ndarray)         : Day input data
           occupancy_profile     (np.ndarray)         : Occupancy profile
           user_parameters       (dict)               : dict containing User attributes
           cooking_app_count     (np.ndarray)         : count of various cooking appliance categories
           cooking_type          (np.ndarray)         : arr of type for all cooking appliances (1 if electric , 0 if gas)
           lunch_present         (bool)               : True if lunch is detected for the user
           lunch_delta           (float)              : energy delta during lunch hours
           default_cooking_flag  (bool)               : True if the given cooking app count is default
           cooking_config        (dict)               : Config containing lunch parameters
           logger_pass           (dict)               : logger dictionary

       Returns:
           consumption           (np.ndarray)         : TOU level lunch estimate
       """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_lunch_usage')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_cooking_start = datetime.now()

    stay_at_home_index = get_occupancy_profile_config().get("general_config").get("stay_at_home_index")

    samples_per_hour = input_data.shape[1] / Cgbdisagg.HRS_IN_DAY

    appliance_consumption = cooking_config.get("appliance_consumption")

    # Calculate appliance count using delta if defualt app count is available

    if default_cooking_flag:
        cooking_app_count = cooking_config.get("cooking_app_count_arr")[np.digitize(lunch_delta, cooking_config.get("delta_arr"))]

    logger.info("Lunch energy delta | %d", lunch_delta)

    cooking_app_count = np.array(cooking_app_count)

    # Minor cooking consumption for gas based cooking

    cooking_app_count[cooking_type == 0] = cooking_app_count[cooking_type == 0] * 0

    appliance_consumption = np.dot(appliance_consumption, cooking_app_count)

    # Fetch cooking usage hours

    usage_hours = cooking_config.get("usage_hours") * samples_per_hour

    total_samples = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    consumption = np.zeros(input_data.shape)

    # If the user is not stay at home, no lunch cooking is provided

    lunch_type_activity_absent = (not occupancy_profile[stay_at_home_index]) and (not lunch_present)

    if lunch_type_activity_absent:
        return consumption

    # If more stay at home people at present, higher estimate is given

    if occupancy_profile[stay_at_home_index] == cooking_config.get("occupants_limit"):
        usage_hours = np.array([cooking_config.get("high_stay_at_home_usage_hours")]) * samples_per_hour
        appliance_consumption = appliance_consumption * cooking_config.get("high_stay_at_home_multiplier")

    # If lunch is absent for the user, a minor consumption is given

    if not lunch_present:
        logger.info("User is not given consumption during lunch hours | ")
        appliance_consumption = cooking_config.get("absent_lunch_consumption")

    # Calculate lunch cooking hours

    start_time = user_parameters.get("lunch_start")

    if user_parameters["lunch_start"] == -1:
        start_time = cooking_config.get("default_lunch_start") * samples_per_hour

    end_time = start_time + usage_hours

    logger.info("Lunch start time | %d", start_time)
    logger.info("Lunch end time | %d", end_time)

    logger.debug("Initialization done for calculation of lunch usage")

    consumption = add_noise_to_stat_app_est(consumption, start_time, end_time, usage_hours, total_samples,
                                            appliance_consumption / samples_per_hour)

    t_cooking_end = datetime.now()

    logger.info("Calculation of lunch usage took | %.3f s", get_time_diff(t_cooking_start, t_cooking_end))

    return consumption
