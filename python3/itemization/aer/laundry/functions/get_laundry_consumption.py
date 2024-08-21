
"""
Author - Nisha Agarwal
Date - 9th Feb 20
Prepare laundry TOU estimate
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.itemization.aer.functions.itemization_utils import rolling_func

from python3.itemization.aer.functions.itemization_utils import cap_app_consumption

from python3.itemization.aer.laundry.config.get_estimation_config import get_estimation_config

from python3.itemization.aer.laundry.functions.get_dishwasher_usage import get_dishwasher_usage

from python3.itemization.aer.laundry.functions.get_washing_machine_usage import get_washing_machine_usage


def get_consumption(item_input_object, delta, input_data, laundry_app_count, default_laundry_flag,
                    energy_diff_copy, vacation, logger):

    """
       Prepare laundry TOU estimate

       Parameters:
           item_input_object             (dict)               : hybrid input parameters dict
           delta                         (np.ndarray)         : time stamp level weekend energy delta
           input_data                    (np.ndarray)         : Day input data
           laundry_app_count             (np.ndarray)         : count of various laundry appliance categories
           default_laundry_flag          (bool)               : True if laundry app count are default values
           weekday_delta                 (np.ndarray)         : time stamp level weekday energy delta
           energy_diff                   (np.ndarray)         : weekend/weekday energy diff
           vacation                      (np.ndarray)         : Vacation days masked array
           logger                        (logger)             : logger object

       Returns:
           total_consumption             (np.ndarray)         : TOU level estimate
           total_dishwasher_component    (np.ndarray)         : dishwasher component of total estimate

    """

    pilot_level_config = item_input_object.get('pilot_level_config')

    # Calculate washing machine estimate

    consumption_wash_mach = get_washing_machine_usage(item_input_object, pilot_level_config, delta,
                                                      input_data, laundry_app_count, default_laundry_flag,
                                                      energy_diff_copy, logger)

    # Calculate dish washer estimate

    consumption_dish_wash = get_dishwasher_usage(item_input_object, pilot_level_config, delta,
                                                 input_data, logger)

    # take minimum with input data

    consumption_dish_wash = np.minimum(input_data, consumption_dish_wash)
    consumption_wash_mach = np.minimum(input_data, consumption_wash_mach)

    total_consumption = consumption_dish_wash + consumption_wash_mach

    total_consumption = np.minimum(input_data, total_consumption)

    # Zero consumption in vacation

    total_consumption[vacation, :] = 0
    consumption_dish_wash[vacation, :] = 0
    consumption_wash_mach[vacation, :] = 0

    # Cap laundry estimate

    total_consumption = cap_app_consumption(total_consumption)

    return total_consumption, consumption_dish_wash


def get_laundry_estimate(item_input_object, item_output_object, input_data, weekday_delta, weekend_delta, logger):

    """
       Prepare laundry TOU estimate

       Parameters:
           item_input_object             (dict)               : hybrid input parameters dict
           item_output_object            (dict)               : hybrid output parameters dict
           input_data                    (np.ndarray)         : Day input data
           weekend_day                   (np.ndarray)         : weekend days masked array
           logger                        (logger)             : logger object

       Returns:
           total_consumption             (np.ndarray)         : TOU level estimate
           total_dishwasher_component    (np.ndarray)         : dishwasher component of total estimate
    """

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    weekend_day = item_input_object.get("item_input_params").get("weekend_days")
    user_attributes = item_output_object.get("occupants_profile").get("occupants_features")
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")
    default_laundry_flag = item_input_object.get("appliance_profile").get("default_laundry_flag")

    laundry_app_count = item_input_object.get("appliance_profile").get("laundry")
    weekend_weekday_activity_diff = item_output_object.get("weekend_weekday_diff")
    weekend_weekday_energy_diff = item_output_object.get("energy_profile").get("energy_diff")

    # Zero laundry consumption in sleep hours

    sleep_hours = item_output_object.get("profile_attributes").get("sleep_hours")
    weekend_weekday_energy_diff[np.logical_not(sleep_hours)] = 0

    energy_diff_copy = rolling_func(weekend_weekday_energy_diff, int(samples_per_hour/2), avg=0)

    pilot_level_config = item_input_object.get('pilot_level_config')
    ld_config = get_estimation_config(pilot_level_config, samples_per_hour)

    office_going_hours = ld_config.get('office_going_hours')
    weekend_weekday_energy_diff_limit = ld_config.get('weekend_weekday_energy_diff_limit')

    # If user is an office goer, no consumption in day time

    if not user_attributes[2]:
        weekday_delta[office_going_hours] = 0
    else:

        # If user is stay at home , similar consumption on weekday and weekends

        total_consumption, dish_washer_cons = \
            get_consumption(item_input_object, weekday_delta, input_data, laundry_app_count, default_laundry_flag,
                            np.zeros(len(weekend_weekday_energy_diff)), vacation, logger)

        total_consumption = np.nan_to_num(total_consumption)

        return total_consumption, dish_washer_cons

    # If user is office goer, different laundry consumption and tou on weekdays and weekends

    # if laundry type activity is found additionally on weekdays,
    # then laundry is not given at times where extra weekend activity was detected rather than on weekdays

    if np.any(weekday_delta[weekend_weekday_activity_diff == -1] > weekend_weekday_energy_diff_limit):
        weekday_delta[weekend_weekday_activity_diff > -1] = 0

    elif np.any(weekday_delta[weekend_weekday_activity_diff <= 0] > weekend_weekday_energy_diff_limit):
        weekday_delta[weekend_weekday_activity_diff > 0] = 0

    # estimating laundry consumption for weekday days

    total_consumption_weekday, dish_washer_cons1 = \
        get_consumption(item_input_object, weekday_delta, input_data, laundry_app_count, default_laundry_flag, energy_diff_copy, vacation, logger)

    # if laundry type activity is found additionally on weekends,
    # then laundry is not given at times where extra weekdays activity was detected rather than on weekends

    if np.any(weekend_delta[weekend_weekday_activity_diff == 1] > weekend_weekday_energy_diff_limit):
        weekend_delta[weekend_weekday_activity_diff != 1] = 0

    elif np.any(weekend_delta[weekend_weekday_activity_diff >= 0] > weekend_weekday_energy_diff_limit):
        weekend_delta[weekend_weekday_activity_diff < 0] = 0

    # estimating laundry consumption for weekend days

    total_consumption_weekend, dish_washer_cons2 = \
        get_consumption(item_input_object, weekend_delta, input_data, laundry_app_count, default_laundry_flag, energy_diff_copy, vacation, logger)

    total_consumption = np.zeros(total_consumption_weekday.shape)
    total_consumption[weekend_day] = total_consumption_weekend[weekend_day]
    total_consumption[np.logical_not(weekend_day)] = total_consumption_weekday[np.logical_not(weekend_day)]

    # 0 laundry consumption on vacation days

    total_consumption[vacation] = 0

    total_consumption = np.nan_to_num(total_consumption)

    # these two components denotes the weekday and weekend consumption

    total_dishwasher_component = dish_washer_cons1 + dish_washer_cons2

    return total_consumption, total_dishwasher_component
