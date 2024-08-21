
"""
Author - Nisha Agarwal
Date - 9th Feb 20
Calculates washing machine tou estimate
"""

# Import python packages

import copy
import numpy as np

from numpy.random import RandomState

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.aer.functions.itemization_utils import add_noise_in_laundry

from python3.itemization.aer.laundry.config.get_estimation_config import get_estimation_config


def get_washing_machine_usage(item_input_object, pilot_level_config, energy_delta_array, input_data,
                              laundry_app_count, default_laundry_flag, energy_diff, logger):

    """
       Calculate washing machine tou estimate

       Parameters:
           item_input_object     (dict)               : hybrid input parameters dict
           pilot_level_config    (dict)               : hybrid config for the pilot
           energy_delta_array    (np.ndarray)         : array of energy delta at each timestamp
           input_data            (np.ndarray)         : Day input data
           laundry_app_count     (np.ndarray)         : count of laundry appliances in app profile
           default_laundry_flag  (bool)               : True if the laundry app count are default values
           energy_diff           (np.ndarray)         : array of difference in weekend/weekday energy profile
           logger                (logger)             : logger object

       Returns:
           consumption           (np.ndarray)         : TOU level estimate
    """

    laundry_type = item_input_object.get("appliance_profile").get("laundry_type")

    logger.debug("Starting estimation of washing machine usage consumption")

    samples_per_hour = int(input_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    config = get_estimation_config(pilot_level_config, samples_per_hour,
                                   item_input_object.get("appliance_profile").get('drier_present')).get("washing_machine_config")

    usage_hours = config.get("default_hours")

    consumption = np.zeros(input_data.shape)

    energy_delta_array = copy.deepcopy(energy_delta_array)

    # washing machine consumption is not added during late night hours

    energy_delta_array[: 4 * samples_per_hour] = 0

    energy_delta_array[int((Cgbdisagg.HRS_IN_DAY - 1.5) * samples_per_hour):] = 0

    delta = np.max(energy_delta_array)
    start_time = np.argmax(energy_delta_array) + samples_per_hour

    extra_device = 0

    # If laundry app count is default

    if default_laundry_flag:

        logger.debug("Default laundry app count to be used")

        # If difference in weekday and weekend profile is significant, laundry is assumed to be mostly used on weekends

        if np.any(energy_diff > config.get("energy_diff_limit")):

            logger.debug("laundry estimation to be done using weekday weekend comparison")

            appliance_consumption = config.get("default_amp")
            devices = config.get("default_devices")
            start_time = np.argmax(energy_diff)
            days_in_week = config.get("default_dow")

        # Else laundry amplitude is assumed based on the energy delta

        else:

            logger.debug("laundry estimation to be done using daily consumption delta")

            devices = config.get("devices_arr")[np.digitize(delta, config.get("delta_arr"))]
            days_in_week = config.get("dow_arr")[np.digitize(delta, config.get("delta_arr"))]
            appliance_consumption = config.get("amp_arr")[np.digitize(delta, config.get("delta_arr"))]
            usage_hours = config.get("hours_arr")[np.digitize(delta, config.get("delta_arr"))]

    else:

        logger.debug("laundry app count to be calculated from given app profile")
        # if the laundry app count is already provided by the user

        days_in_week = 5
        appliance_consumption = 1000
        usage_hours = config.get("default_hours")
        devices = max(1, np.ceil(laundry_app_count[0]))

        extra_device = min(1, laundry_app_count[2] * laundry_type[2])

    total_samples = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    # Calculate timestamp level estimation for all the laundry devices

    devices = int(devices)

    for dev_count in range(devices):

        end_time = start_time + usage_hours - 1

        consumption = add_noise_in_laundry(consumption, start_time, end_time, usage_hours, total_samples, appliance_consumption / samples_per_hour)

    energy_delta_array_copy = copy.deepcopy(energy_delta_array)
    energy_delta_array_copy[np.arange(np.argmax(energy_delta_array_copy) - 2 * samples_per_hour,
                                      np.argmax(energy_delta_array_copy) + 2 * samples_per_hour) % len(energy_delta_array_copy)] = 0

    for dev_count in range(int(extra_device)):

        start_time = np.argmax(energy_delta_array_copy) + samples_per_hour

        end_time = start_time + usage_hours - 1

        consumption = consumption + add_noise_in_laundry(consumption, start_time, end_time, usage_hours, total_samples,
                                                         appliance_consumption / samples_per_hour)

    seed = RandomState(random_gen_config.seed_value)

    non_laundry_days = np.array(seed.choice(list(np.arange(len(input_data))),
                                            int((Cgbdisagg.DAYS_IN_WEEK - days_in_week) / Cgbdisagg.DAYS_IN_WEEK * len(input_data))))
    consumption[non_laundry_days.astype(int)] = 0

    logger.debug("Estimation of washing machine usage done")

    return consumption
