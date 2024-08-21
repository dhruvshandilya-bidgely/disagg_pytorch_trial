
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Estimate on demand entertainment low usage
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


def get_ent_appliance_consumption(devices, start_time, end_time, charging_hours, charging_consumption,
                                  samples_per_hour, consumption):

    """
       calculate tou consumption using start, end time and appliance amplitude

       Parameters:
           devices                  (list)               : number of devices on on demand low usage cat
           start_time               (float)              : Possible start time of appliance
           end_time                 (float)              : Possible end time of the appliance
           charging_hours           (int)                : usage hours of the appliance
           charging_consumption     (int)                : amplitude of television consumption
           samples_per_hour         (int)                : samples in an hour
           consumption              (np.ndarray)         : Previously estimated appliance consumption

       Returns:
           consumption              (np.ndarray)         : Estimated appliance consumption
       """

    total_samples = samples_per_hour * Cgbdisagg.HRS_IN_DAY

    for device_idx in range(len(devices)):

        for idx in range(int(devices[device_idx])):

            start_time, end_time = \
                get_start_end_time(start_time, end_time, charging_hours)

            consumption = add_noise_to_stat_app_est(consumption, start_time, end_time, charging_hours, total_samples,
                                                    charging_consumption[device_idx] / samples_per_hour)

    return consumption


def get_on_demand_low_usage(input_data, samples_per_hour, occupancy_profile, user_parameters, ent_config, ent_app_count, default_ent, logger_pass):

    """
       Estimate on demand entertainment low usage

       Parameters:
           input_data            (np.ndarray)         : Day input data
           samples_per_hour      (int)                : samples in an hour
           occupancy_profile     (np.ndarray)         : Occupancy profile
           user_parameters       (dict)               : dict containing User attributes
           ent_config            (dict)               : Config containing entertainment parameters
           ent_app_count         (np.ndarray)         : count of various entertainment appliance categories
           default_ent           (bool)               : this flag denote whether ent output is given based on app profile
           logger_pass           (dict)               : Contains the logger and the logging dictionary to be passed on

       Returns:
           consumption           (np.ndarray)         : TOU level estimate
       """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_on_demand_low_usage')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    t_entertainment_start = datetime.now()

    # Modify sleep hours for late night active users

    if user_parameters.get('sleep_time') < ent_config.get("late_night_hours") * samples_per_hour:
        user_parameters['sleep_time'] = user_parameters.get('sleep_time') + Cgbdisagg.HRS_IN_DAY * samples_per_hour

    devices = ent_config.get("devices")

    charging_consumption = ent_config.get("charging_amp")

    charging_hours = ent_config.get("charging_hours") * samples_per_hour

    occupancy_profile_config = get_occupancy_profile_config().get("general_config")

    morning_index = 0
    mid_day_index = 1
    evening_index = 3

    # Modify devices count based on occupancy profile

    if occupancy_profile[occupancy_profile_config.get("office_goer_index")] == 2:
        devices[morning_index] = [2, 1]
        devices[evening_index] = [2, 1]

    if occupancy_profile[occupancy_profile_config.get("stay_at_home_index")] == 2:
        devices[mid_day_index] = [2, 1]

    if not occupancy_profile[occupancy_profile_config.get("stay_at_home_index")]:
        devices[mid_day_index] = [0, 0]

    if user_parameters['morning_start'] >= ent_config.get("default_morning_end") * samples_per_hour:
        devices[morning_index] = [0, 0]

    if not default_ent:
        devices[morning_index] = np.fmax(np.array(devices[morning_index]), ent_app_count[0] - 2)
        devices[2] = np.fmax(np.array(devices[2]), ent_app_count[0] - 1)
        devices[1] = np.fmax(np.array(devices[1]), ent_app_count[0] - 1)
        devices[3] = np.fmax(np.array(devices[3]), ent_app_count[0] - 1)
        devices[4] = np.fmax(np.array(devices[4]), ent_app_count[0] - 1)

    logger.debug("Entertainment on demand low usage devices count %s", devices)

    morning_cons = np.zeros(input_data.shape)

    # calculate morning entertainment usage

    if user_parameters.get('morning_start') != -1 and user_parameters.get('morning_end') != -1 :

        logger.debug("Calculating on demand low usage consumption for morning hours")

        morning_cons = get_ent_appliance_consumption(devices[morning_index], user_parameters.get('morning_start'),
                                                     user_parameters.get('morning_end'), charging_hours, charging_consumption,
                                                     samples_per_hour, morning_cons)

    mid_day_start = ent_config.get("default_mid_day_start") * samples_per_hour
    mid_day_end = ent_config.get("default_mid_day_end") * samples_per_hour

    # calculate mid day entertainment usage

    mid_day_consumption = \
        get_ent_appliance_consumption(devices[mid_day_index], mid_day_start, mid_day_end, charging_hours, charging_consumption,
                                      samples_per_hour, morning_cons)

    logger.debug("Calculated on demand low usage consumption for low usage for mid day hours")

    # calculate evening entertainment usage

    consumption = get_ent_appliance_consumption(devices[evening_index], user_parameters.get('office_coming_time'),
                                                user_parameters.get('sleep_time'), charging_hours, charging_consumption,
                                                samples_per_hour, mid_day_consumption)

    logger.debug("Calculated on demand low usage consumption for low usage for evening hours")

    consumption = np.nan_to_num(consumption)

    # zero estimate during late night hours

    t_entertainment_end = datetime.now()

    logger.info("Calculation of on demand low entertainment consumption took | %.3f s",
                get_time_diff(t_entertainment_start, t_entertainment_end))

    return consumption
