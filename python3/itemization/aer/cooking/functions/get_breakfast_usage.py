
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Estimate breakfast cooking usage
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime
from numpy.random import RandomState

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.aer.functions.itemization_utils import add_noise_to_stat_app_est


def get_breakfast_usage(input_data, occupancy_profile, user_parameters, cooking_app_count,
                        cooking_app_type, breakfast_delta, default_cooking_flag, cooking_config, logger_pass):

    """
       Estimate tou consumption of lunch cooking

       Parameters:
           input_data            (np.ndarray)         : Day input data
           occupancy_profile     (np.ndarray)         : Occupancy profile
           user_parameters       (dict)               : dict containing User attributes
           cooking_app_count     (np.ndarray)         : count of various cooking appliance categories
           cooking_app_type      (np.ndarray)         : arr of type for all cooking appliances (1 if electric , 0 if gas)
           breakfast_delta       (float)              : energy delta during breakfast hours
           default_cooking_flag  (bool)               : True if the given cooking app count is default
           cooking_config        (dict)               : Config containing lunch parameters
           logger_pass           (dict)               : logger dictionary

       Returns:
           consumption           (np.ndarray)         : TOU level lunch estimate
       """

    # Initialize the logger

    seed = RandomState(random_gen_config.seed_value)

    logger_base = logger_pass.get('logger_base').getChild('get_breakfast_usage')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_cooking_start = datetime.now()

    samples_per_hour = input_data.shape[1] / Cgbdisagg.HRS_IN_DAY

    consumption = np.zeros(input_data.shape)

    appliance_consumption = cooking_config.get("appliance_consumption")

    logger.info("Breakfast energy delta | %d", breakfast_delta)

    # Calculate appliance count using delta if default app count is available

    if default_cooking_flag:

        logger.debug("Using default cooking app count for breakfast usage calculation")
        cooking_app_count = cooking_config.get("cooking_app_count_arr")[np.digitize(breakfast_delta, cooking_config.get("delta_arr"))]

    cooking_app_count = np.array(cooking_app_count)

    # Minor cooking consumption for gas based cooking
    cooking_app_count[cooking_app_type == 0] = cooking_app_count[cooking_app_type == 0] * cooking_config.get("gas_cooking_multiplier")

    # Total amplitude of cooking

    appliance_consumption = np.dot(appliance_consumption, cooking_app_count)

    # Fetch cooking usage hours

    usage_hours = cooking_config.get("usage_hours") * samples_per_hour

    total_samples = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    breakfast_end_time = cooking_config.get("breakfast_end_time")

    # If the wakeup time is late, no breakfast estimate is given

    late_wakeup_time_flag = user_parameters['morning_start'] >= breakfast_end_time * samples_per_hour

    if late_wakeup_time_flag:

        logger.debug("Late wakeup hours for the user, not calculating breakfast usage")
        return consumption

    # If no morning end time is found, default is assumed

    if user_parameters.get('morning_end') == -1:
        logger.info("No morning activity end time found, adding default")
        user_parameters['morning_end'] = breakfast_end_time * samples_per_hour

    if np.sum(occupancy_profile) > cooking_config.get("occupants_limit"):
        appliance_consumption = appliance_consumption * cooking_config.get("high_occupants_multiplier")

    # If no morning start is found, default is assumed

    if user_parameters.get('morning_start') == -1:
        user_parameters['morning_start'] = (breakfast_end_time - cooking_config.get("usage_hours")) * samples_per_hour

    early_wake_up_time_flag = user_parameters.get('morning_start') >= \
                             (breakfast_end_time - cooking_config.get("usage_hours")) * samples_per_hour

    if early_wake_up_time_flag:
        appliance_consumption = cooking_config.get("late_wakeup_consumption")

    # Morning end is modified, with assumption of buffer time between breakfast cooking and leaving the house

    user_parameters['morning_end'] = min(user_parameters['morning_end'] -
                                         cooking_config.get("breakfast_buffer_hours") * samples_per_hour,
                                         breakfast_end_time * samples_per_hour)

    if samples_per_hour == 1:
        appliance_consumption = appliance_consumption * cooking_config.get("60_min_multiplier")

    morning_activity_window = (user_parameters.get('morning_end') - user_parameters.get('morning_start'))

    # Calculate breakfast cooking hours

    if morning_activity_window <= usage_hours:
        logger.debug("Using default breakfast hours for the user")
        start_time = user_parameters.get('morning_start')
        end_time = user_parameters.get('morning_end')

    else:
        logger.debug("Calculating breakfast hours from activity curve information")
        tou_list = np.arange(user_parameters.get('morning_start'), user_parameters.get('morning_end') - usage_hours)
        start_time = seed.choice(tou_list)
        end_time = start_time + usage_hours

    logger.info("Breakfast start time | %d", start_time)
    logger.info("Breakfast end time | %d", end_time)

    logger.debug("Initialization done for breakfast usage calculation")

    consumption = add_noise_to_stat_app_est(consumption, start_time, end_time, usage_hours, total_samples,
                                            appliance_consumption / samples_per_hour)

    t_cooking_end = datetime.now()

    logger.info("Calculation of breakfast usage took | %.3f s", get_time_diff(t_cooking_start, t_cooking_end))

    return consumption
