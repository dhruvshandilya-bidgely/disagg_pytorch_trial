

"""
Author - Nisha Agarwal
Date - 3rd Feb 21
estimate tou level cooking estimate
"""

# Import python packages

import copy
import logging
import numpy as np

# import functions from within the project

from python3.itemization.aer.cooking.config.get_cooking_config import get_cooking_config

from python3.itemization.aer.functions.itemization_utils import cap_app_consumption

from python3.itemization.aer.cooking.functions.get_lunch_usage import get_lunch_usage
from python3.itemization.aer.cooking.functions.get_dinner_usage import get_dinner_usage
from python3.itemization.aer.cooking.functions.get_cooking_delta import get_cooking_delta
from python3.itemization.aer.cooking.functions.get_breakfast_usage import get_breakfast_usage


def prepare_cooking_tou_consumption(item_input_object, item_output_object, occupancy_profile, energy_profile,
                                    lunch_present, logger_pass):

    """
       Prepare tou estimate of cooking appliances for weekday and weekends individually

        Parameters:
           item_input_object          (dict)               : Dict containing all hybrid inputs
           item_output_object         (dict)               : Dict containing all hybrid outputs
           occupancy_profile          (np.ndarray)         : Occupancy profile
           energy_profile             (np.ndarray)         : energy profile using for delta calculation
           lunch_present              (bool)               : True if lunch is present for the user
           logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

       Returns:
           total_consumption          (np.ndarray)         : TOU level lunch estimate
       """

    cooking_app_count = item_input_object.get("appliance_profile").get("cooking")
    cooking_type = item_input_object.get("appliance_profile").get("cooking_type")
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    input_data = copy.deepcopy(item_input_object.get("item_input_params").get('day_input_data'))

    user_parameters = item_output_object.get("occupants_profile").get("user_attributes")
    default_cooking_flag = item_input_object.get("appliance_profile").get("default_cooking_flag")
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")
    pilot_level_config = item_input_object.get('pilot_level_config')

    vacation = vacation.astype(bool)
    cooking_app_count = [cooking_app_count, cooking_app_count, cooking_app_count]

    # fetch config dict

    cooking_config = get_cooking_config(item_input_object,pilot_level_config, samples_per_hour)

    # Calculate cooking delta using energy profile

    weekday_cooking_delta = get_cooking_delta(energy_profile, samples_per_hour, cooking_config.get("general_config"))

    # Calculate breakfast usage

    breakfast_consumption = \
        get_breakfast_usage(input_data, occupancy_profile, user_parameters, cooking_app_count[0], cooking_type,
                            weekday_cooking_delta[0], default_cooking_flag,
                            cooking_config.get("breakfast_config"), logger_pass)

    # Calculate lunch usage

    lunch_consumption = \
        get_lunch_usage(input_data, occupancy_profile, user_parameters, cooking_app_count[1], cooking_type,
                        lunch_present, weekday_cooking_delta[1], default_cooking_flag,
                        cooking_config.get("lunch_config"), logger_pass)

    # Calculate dinner usage

    dinner_consumption = \
        get_dinner_usage(input_data, occupancy_profile, user_parameters, cooking_app_count[2], cooking_type,
                         weekday_cooking_delta[2], default_cooking_flag, cooking_config.get("dinner_config"), logger_pass)

    # zero consumption on vacation days , and minumum with input data

    breakfast_consumption = np.minimum(input_data, breakfast_consumption)
    lunch_consumption = np.minimum(input_data, lunch_consumption)
    dinner_consumption = np.minimum(input_data, dinner_consumption)

    total_consumption = breakfast_consumption + lunch_consumption + dinner_consumption

    total_consumption = np.minimum(input_data, total_consumption)
    total_consumption = np.fmax(total_consumption, 0)

    total_consumption[vacation, :] = 0
    breakfast_consumption[vacation, :] = 0
    lunch_consumption[vacation, :] = 0
    dinner_consumption[vacation, :] = 0

    # Capping high percentile values

    total_consumption = cap_app_consumption(total_consumption)

    return total_consumption


def modify_app_count(weekday_profile, cooking_app_count, lunch_present, cooking_config):

    """
       Modify cooking app count based on weekday/weekend comparison

       Parameters:
           cooking_app_count          (np.ndarray)         : count of various cooking appliance categories
           weekday_profile            (dict)               : weekday/weekend profile comparison
           lunch_present              (bool)               : True if lunch is present for the user
           cooking_config             (dict)               : cooking config dictionary

       Returns:
           cooking_app_count          (np.ndarray)         : modified count of various cooking appliance categories
           lunch_present              (bool)               : True if lunch is present for the user (modified)
       """

    breakfast_cooking = copy.deepcopy(cooking_app_count)
    lunch_cooking = copy.deepcopy(cooking_app_count)
    dinner_cooking = copy.deepcopy(cooking_app_count)

    # Modify breakfast app count

    if weekday_profile["weekend_morning_present"] and weekday_profile["weekend_morning_absent"]:
        breakfast_cooking = breakfast_cooking * 1
    elif weekday_profile["weekend_morning_present"]:
        breakfast_cooking = breakfast_cooking * cooking_config.get("weekend_present_multiplier")
    elif weekday_profile["weekend_morning_absent"]:
        breakfast_cooking = breakfast_cooking * cooking_config.get("weekend_absent_multiplier")
    elif weekday_profile["weekend_breakfast_absent"]:
        breakfast_cooking = breakfast_cooking * cooking_config.get("cooking_absent_multiplier")

    # Modify lunch app count

    if weekday_profile["weekend_mid_day_present"] and weekday_profile["weekend_mid_day_absent"]:
        lunch_cooking = lunch_cooking * 1
    elif weekday_profile["weekend_mid_day_present"]:
        lunch_present = 1
        lunch_cooking = lunch_cooking * cooking_config.get("weekend_present_multiplier")
    elif weekday_profile["weekend_mid_day_absent"]:
        lunch_cooking = lunch_cooking * cooking_config.get("weekend_absent_multiplier")
    elif (not lunch_present) and weekday_profile["weekend_lunch_absent"]:
        lunch_cooking = lunch_cooking * cooking_config.get("cooking_absent_multiplier")

    # Modify dinner app count

    if weekday_profile["weekend_night_present"] and weekday_profile["weekend_night_absent"]:
        dinner_cooking = dinner_cooking * 1
    elif weekday_profile["weekend_night_present"]:
        dinner_cooking = dinner_cooking * cooking_config.get("weekend_present_multiplier")
    elif weekday_profile["weekend_night_absent"]:
        dinner_cooking = dinner_cooking * cooking_config.get("weekend_absent_multiplier")
    elif weekday_profile["weekend_dinner_absent"]:
        dinner_cooking = dinner_cooking * cooking_config.get("cooking_absent_multiplier")

    cooking_app_count = [breakfast_cooking, lunch_cooking, dinner_cooking]

    return cooking_app_count, lunch_present


def get_cooking_estimate(item_input_object, item_output_object, occupancy_profile, weekend_days, logger_pass):

    """
       Estimate tou consumption of cooking appliances

       Parameters:
           item_input_object          (dict)               : Dict containing all hybrid inputs
           item_output_object         (dict)               : Dict containing all hybrid outputs
           occupancy_profile          (np.ndarray)         : Occupancy profile
           weekend_days               (np.ndarray)         : bool array for masking weekend days
           logger_pass                (dict)               : Contains the logger and the logging dictionary to be passed on

       Returns:
           total_consumption          (np.ndarray)         : TOU level lunch estimate
       """

    weekday_profile = item_output_object.get("weekday_profile")
    pilot_level_config = item_input_object.get('pilot_level_config')
    cooking_app_count = item_input_object.get("appliance_profile").get("cooking")
    user_parameters = item_output_object.get("occupants_profile").get("user_attributes")
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")
    weekend_energy_profile = item_output_object.get("energy_profile").get("weekend_energy_profile")
    weekday_energy_profile = item_output_object.get("energy_profile").get("weekday_energy_profile")

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_cooking_estimate')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    lunch_present = user_parameters.get("lunch_present")

    # Calculate weekday consumption

    total_consumption_weekday = \
        prepare_cooking_tou_consumption(item_input_object, item_output_object, occupancy_profile,
                                        weekday_energy_profile, lunch_present, logger_pass)

    logger.debug("Calculated cooking consumption for weekdays")

    cooking_config = get_cooking_config(item_input_object, pilot_level_config, samples_per_hour).get("general_config")

    # Modify app count based on weekend/weekday comparison

    cooking_app_count, lunch_present = modify_app_count(weekday_profile, cooking_app_count, lunch_present, cooking_config)

    logger.debug("Modified app count based on weekday/weekend comparison")

    # Calculate weekend consumption

    total_consumption_weekend = \
        prepare_cooking_tou_consumption(item_input_object, item_output_object, occupancy_profile,
                                        weekend_energy_profile, lunch_present, logger_pass)

    logger.debug("Calculated cooking consumption for weekends")

    # Prepare final cooking estimate

    total_consumption = np.zeros(total_consumption_weekday.shape)
    total_consumption[weekend_days] = total_consumption_weekend[weekend_days]
    total_consumption[np.logical_not(weekend_days)] = total_consumption_weekday[np.logical_not(weekend_days)]

    total_consumption = np.fmax(total_consumption, 0)

    return total_consumption
