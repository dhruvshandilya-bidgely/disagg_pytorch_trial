
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Estimate on demand entertainment high usage
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import add_noise_to_stat_app_est

from python3.itemization.aer.entertainment.functions.entertainment_utils import get_start_end_time

from python3.itemization.aer.behavioural_analysis.home_profile.config.occupancy_profile_config import get_occupancy_profile_config


def get_ent_appliance_consumption(devices, start_time, end_time, usage_hours, average_television_usage, samples_per_hour, consumption):

    """
       calculate tou consumption using start, end time and appliance amplitude

       Parameters:
           devices                  (list)               : number of devices on on demand high usage cat
           start_time               (float)              : Possible start time of appliance
           end_time                 (float)              : Possible end time of the appliance
           usage_hours              (int)                : usage hours of the appliance
           average_television_usage (int)                : amplitude of television consumption
           samples_per_hour         (int)                : samples in an hour
           consumption              (np.ndarray)         : Previously estimated appliance consumption

       Returns:
           consumption              (np.ndarray)         : Estimated appliance consumption
       """

    total_samples = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    # Calculate tou for all appliances

    for i in range(devices[0]):

        start_time, end_time = get_start_end_time(start_time, end_time, usage_hours)

        # Add noise to the estimated consumption

        consumption = add_noise_to_stat_app_est(consumption, start_time, end_time, usage_hours, total_samples,
                                                average_television_usage / samples_per_hour)

    return consumption


def get_on_demand_high_usage(input_data, samples_per_hour, occupancy_profile, user_parameters, ent_app_count, default_ent,
                             usage_hours, ent_config, logger_pass, weekend=0):

    """
       Estimate on demand entertainment high usage

       Parameters:
           input_data            (np.ndarray)         : Day input data
           samples_per_hour      (int)                : samples in an hour
           occupancy_profile     (np.ndarray)         : Occupancy profile
           user_parameters       (dict)               : dict containing User attributes
           ent_app_count         (np.ndarray)         : count of various cooking appliance categories
           default_ent           (bool)               : this flag denote whether ent output is given based on app profile
           usage_hours           (np.array)           : usage hours of ent appliances
           ent_config            (dict)               : Config containing entertainment parameters
           logger_pass           (dict)               : Contains the logger and the logging dictionary to be passed on
           weekend               (bool)               : flag that denotes whether consumption is calculated for weekend days

       Returns:
           consumption           (np.ndarray)         : TOU level estimate
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_on_demand_high_usage')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    t_entertainment_start = datetime.now()

    # Initialize amplitude

    average_television_usage = ent_config.get("television_amp") * ent_app_count[1]

    if not default_ent:
        average_television_usage = ent_config.get("television_amp") * (ent_app_count[1] + 0.8*ent_app_count[2])

    usage_hours = np.fmax(1, usage_hours)

    # Modify sleep hours for late night active users

    if user_parameters.get('sleep_time') < ent_config.get("late_night_hours") * samples_per_hour:
        user_parameters['sleep_time'] = user_parameters.get('sleep_time') + Cgbdisagg.HRS_IN_DAY * samples_per_hour

    # Initialized devices count

    devices = ent_config.get("devices")
    buffer_hours = ent_config.get("buffer_hours")

    occupancy_profile_config = get_occupancy_profile_config().get("general_config")

    morning_index = 0
    mid_day_index = 1
    early_evening_index = 2
    evening_index = 3

    # Modify devices count based on occupancy profile

    if not weekend:
        if occupancy_profile[occupancy_profile_config.get("stay_at_home_index")] == 0:
            devices[mid_day_index] = [0]

        if occupancy_profile[occupancy_profile_config.get("early_arrival_index")] == 0:
            devices[early_evening_index] = [1]

    # No morning estimate in case of late wakeup hours

    if user_parameters.get('morning_start') >= ent_config.get("default_morning_end") * samples_per_hour:
        devices[morning_index] = [0]

    logger.debug("Entertainment on demand low usage devices count %s", devices)

    # Default morning activity end hours

    if user_parameters.get('morning_end') == -1:
        logger.info("Morning activity not found")
        user_parameters['morning_end'] = ent_config.get("default_morning_end") * samples_per_hour

    consumption = np.zeros(input_data.shape)

    # calculate morning entertainment usage

    morning_activity_start_end_present = user_parameters.get('morning_start') != -1 and user_parameters.get('morning_end') != -1

    if morning_activity_start_end_present:

        logger.debug("Calculating on demand high usage consumption for morning hours")

        consumption = get_ent_appliance_consumption(devices[morning_index],
                                                    user_parameters.get('morning_start') + buffer_hours[morning_index] * samples_per_hour,
                                                    user_parameters.get('morning_end') - buffer_hours[morning_index] * samples_per_hour,
                                                    usage_hours[morning_index], average_television_usage,
                                                    samples_per_hour, consumption)

    mid_day_start = ent_config.get("default_mid_day_start") * samples_per_hour
    mid_day_end = ent_config.get("default_mid_day_end") * samples_per_hour

    # calculate mid day entertainment usage

    consumption = get_ent_appliance_consumption(devices[mid_day_index], mid_day_start, mid_day_end,
                                                usage_hours[mid_day_index], average_television_usage,
                                                samples_per_hour, consumption)

    logger.debug("Calculated on demand high usage consumption for low usage for mid day hours")

    if user_parameters.get('office_coming_time') == -1:
        logger.info("No office coming time found for the user")
        user_parameters['office_coming_time'] = ent_config.get("default_evening_start") * samples_per_hour

    # calculate evening entertainment usage

    consumption = get_ent_appliance_consumption(devices[evening_index], user_parameters['office_coming_time'] + buffer_hours[evening_index] * samples_per_hour,
                                                user_parameters.get('sleep_time') - buffer_hours[evening_index] * samples_per_hour,
                                                usage_hours[evening_index], average_television_usage,
                                                samples_per_hour, consumption)

    logger.debug("Calculated on demand high usage consumption for low usage for evening hours")

    # calculate early evening entertainment usage

    kids_usage_end_time = ent_config.get("default_evening_start") * samples_per_hour

    if user_parameters.get('early_activity_start_time') != -1:

        logger.info("No early activity found in the user data")

        consumption = get_ent_appliance_consumption(devices[early_evening_index], user_parameters.get('early_activity_start_time'),
                                                    kids_usage_end_time, usage_hours[early_evening_index], average_television_usage,
                                                    samples_per_hour, consumption)

    logger.debug("Calculated on demand high usage consumption for low usage for early evening hours")

    t_entertainment_end = datetime.now()

    logger.info("Calculation of on demand low entertainment consumption took | %.3f s",
                get_time_diff(t_entertainment_start, t_entertainment_end))

    return consumption
