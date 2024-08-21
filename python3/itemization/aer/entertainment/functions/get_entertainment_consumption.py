
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Estimation entertainment tou consumption
"""

# Import python packages

import copy
import logging
import numpy as np

# import functions from within the project

from python3.itemization.aer.functions.itemization_utils import cap_app_consumption

from python3.itemization.aer.entertainment.config.get_entertainment_config import get_entertainment_config

from python3.itemization.aer.entertainment.functions.get_on_demand_low_usage import get_on_demand_low_usage
from python3.itemization.aer.entertainment.functions.get_on_demand_high_usage import get_on_demand_high_usage


def prepare_entertainment_tou_consumption(item_input_object, occupancy_profile, user_parameters, vacation, ent_app_count,
                                          default_ent, usage_hours, weekend, logger_pass):

    """
       Prepare tou consumption

       Parameters:
           pilot_config          (dict)               : hybrid v2 pilot config
           input_data            (np.ndarray)         : Day input data
           samples_per_hour      (int)                : samples in an hour
           occupants_count       (int)                : Estimated occupants count
           occupancy_profile     (np.ndarray)         : Occupancy profile
           user_parameters       (dict)               : dict containing User attributes
           vacation              (np.ndarray)         : vacation masked array
           ent_app_count         (np.ndarray)         : count of various entertainment appliance categories
           default_ent           (bool)               : this flag denote whether ent output is given based on app profile
           usage_hours           (np.ndarray)         : entertainment usage hours
           weekend               (bool)               : True if extra consumption is to be given on weekends
           logger_pass           (dict)               : Contains the logger and the logging dictionary to be passed on

       Returns:
           total_consumption     (np.ndarray)         : TOU level dinner estimate
       """

    pilot_config = item_input_object.get('pilot_level_config')
    input_data = copy.deepcopy(item_input_object.get("item_input_params").get('day_input_data'))
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")

    ent_config = get_entertainment_config(samples_per_hour, pilot_config)

    # calculate on demand low usage consumption

    od_low_consumption = get_on_demand_low_usage(input_data, samples_per_hour, occupancy_profile, user_parameters,
                                                 ent_config.get("od_low_config"),  ent_app_count, default_ent, logger_pass)

    # calculate on demand high usage consumption

    od_high_consumption = get_on_demand_high_usage(input_data, samples_per_hour, occupancy_profile, user_parameters,
                                                   ent_app_count, default_ent, usage_hours, ent_config.get("od_high_config"), logger_pass, weekend)

    # Take minimum with input data

    od_low_consumption = np.minimum(input_data, od_low_consumption)
    od_high_consumption = np.minimum(input_data, od_high_consumption)

    total_consumption = od_low_consumption + od_high_consumption

    total_consumption = np.minimum(input_data, total_consumption)

    # 0 consumption on vacation days

    vacation = vacation.astype(bool)

    od_low_consumption[vacation] = 0
    od_high_consumption[vacation] = 0
    total_consumption[vacation] = 0

    # Cap high percentile consumption points

    total_consumption = cap_app_consumption(total_consumption)

    return total_consumption


def modify_usage_hours(pilot_config, ent_app_count, samples_per_hour, weekday_profile, usage_hours, weekend_flag):

    """
       Modified usage hours based on weekday/weekend comparison

       Parameters:
           pilot_config          (dict)               : hybrid v2 pilot config
           ent_app_count         (np.ndarray)         : count of various cooking appliance categories
           samples_per_hour      (int)                : samples in an hour
           weekday_profile       (dict)               : weekday weekend comparison profile
           usage_hours           (np.ndarray)         : entertainment usage hours
           weekend_flag          (bool)               : True if extra consumption is to be given on weekends

       Returns:
           usage_hours           (np.ndarray)         : modified entertainment usage hours
           weekend               (bool)               : True if extra consumption is to be given on weekends (modified)
           ent_app_count         (np.ndarray)         : modified count of various cooking appliance categories
       """

    ent_config = get_entertainment_config(samples_per_hour, pilot_config).get("general_config")

    # Modify usage hours if extra activity is present on weekends

    if weekday_profile["weekend_morning_present"]:
        usage_hours[0] = ent_config.get("weekend_present_usage_hours")[0] * samples_per_hour

    if weekday_profile["weekend_mid_day_present"]:
        usage_hours[1] = ent_config.get("weekend_present_usage_hours")[1] * samples_per_hour

    if weekday_profile["weekend_early_eve_present"]:
        usage_hours[2] = ent_config.get("weekend_present_usage_hours")[2] * samples_per_hour

    if weekday_profile["weekend_night_present"]:
        usage_hours[3] = ent_config.get("weekend_present_usage_hours")[3] * samples_per_hour

    # Modify usage hours if lesser activity is present on weekends

    if weekday_profile["weekend_morning_absent"]:
        usage_hours[0] = ent_config.get("weekend_absent_usage_hours")[0] * samples_per_hour

    if weekday_profile["weekend_mid_day_absent"]:
        usage_hours[1] = ent_config.get("weekend_absent_usage_hours")[1] * samples_per_hour

    if weekday_profile["weekend_early_eve_absent"]:
        usage_hours[2] = ent_config.get("weekend_absent_usage_hours")[2] * samples_per_hour

    if weekday_profile["weekend_night_absent"]:
        usage_hours[3] = ent_config.get("weekend_absent_usage_hours")[3] * samples_per_hour

    if weekday_profile["weekend_mid_day_present"]:
        weekend_flag = True

    return usage_hours, weekend_flag, ent_app_count


def get_entertainment_estimate(item_output_object, item_input_object, occupancy_profile,
                               vacation, ent_app_count, weekday_profile, logger_pass):

    """
       Estimation entertainment tou consumption

       Parameters:
           pilot_config          (dict)               : hybrid v2 pilot config
           input_data            (np.ndarray)         : Day input data
           samples_per_hour      (int)                : samples in an hour
           occupants_count       (int)                : Estimated occupants count
           occupancy_profile     (np.ndarray)         : Occupancy profile
           user_parameters       (dict)               : dict containing User attributes
           vacation              (np.ndarray)         : vacation masked array
           ent_app_count         (np.ndarray)         : count of various cooking appliance categories
           default_ent           (bool)               : this flag denote whether ent output is given based on app profile
           weekday_profile       (dict)               : weekend/weekday comparison profile
           weekend_day           (np.ndarray)         : weekend masked array
           logger_pass           (dict)               : Contains the logger and the logging dictionary to be passed on

       Returns:
           total_consumption     (np.ndarray)         : TOU level estimate
       """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_entertainment_estimate')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    pilot_config = item_input_object.get('pilot_level_config')
    weekend_day = item_input_object.get("item_input_params").get("weekend_days")
    user_parameters = item_output_object.get("occupants_profile").get("user_attributes")
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")
    default_ent = item_input_object.get("appliance_profile").get("default_ent_flag")

    ent_config = get_entertainment_config(samples_per_hour, pilot_config)

    usage_hours = ent_config.get("general_config").get("usage_hours")

    weekend = False

    # Calculate weekday consumption

    total_consumption_weekday = prepare_entertainment_tou_consumption(item_input_object, occupancy_profile,
                                                                      user_parameters, vacation, ent_app_count,default_ent,
                                                                      usage_hours, weekend, logger_pass)

    logger.debug("Calculated entertainment consumption for weekdays")

    usage_hours, weekend, ent_app_count = modify_usage_hours(pilot_config, ent_app_count, samples_per_hour, weekday_profile, usage_hours, weekend)

    logger.debug("Modified usage hours based on weekend/weekday comparison")

    # calculate weekend consumption

    total_consumption_weekends = prepare_entertainment_tou_consumption(item_input_object, occupancy_profile,
                                                                       user_parameters, vacation, ent_app_count, default_ent,
                                                                       usage_hours, weekend, logger_pass)

    logger.debug("Calculated entertainment consumption for weekends")

    # Prepare final tou level entertainment estimate

    total_consumption = np.zeros(total_consumption_weekday.shape)
    total_consumption[weekend_day] = total_consumption_weekends[weekend_day]
    total_consumption[np.logical_not(weekend_day)] = total_consumption_weekday[np.logical_not(weekend_day)]

    return total_consumption
