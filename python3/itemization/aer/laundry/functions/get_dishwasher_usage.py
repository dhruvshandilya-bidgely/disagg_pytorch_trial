
"""
Author - Nisha Agarwal
Date - 9th Feb 20
Calculate dishwasher tou estimate
"""

# Import python packages

import numpy as np
from numpy.random import RandomState

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import random_gen_config
from python3.itemization.aer.functions.itemization_utils import add_noise_in_laundry

from python3.itemization.aer.laundry.config.get_estimation_config import get_estimation_config


def get_dishwasher_usage(item_input_object, pilot_level_config, energy_delta_array, input_data, logger):

    """
       Calculate dishwasher tou estimate

       Parameters:
           item_input_object     (dict)               : hybrid input parameters dict
           pilot_level_config     (dict)           : hybrid config for the given pilot user
           energy_delta_array    (np.ndarray)         : array of energy delta at each timestamp
           input_data            (np.ndarray)         : Day input data
           logger                (logger)             : logger object

       Returns:
           consumption           (np.ndarray)         : TOU level estimate
    """

    logger.debug("Starting estimation of dishwasher consumption")

    samples_per_hour = int(input_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    config = get_estimation_config(pilot_level_config, samples_per_hour, item_input_object.get("appliance_profile").get('drier_present')).get("dishwasher_config")

    # typical dishwasher usage hours are assumed to be during night

    night_hours = config.get("night_hours")

    delta = np.max(energy_delta_array[night_hours]) * samples_per_hour
    start_time = np.argmax(energy_delta_array[night_hours]) + night_hours[0]

    if item_input_object.get("app_profile") is not None and \
        item_input_object.get("app_profile").get(31) is not None and \
        item_input_object.get("app_profile").get(31).get('number') is not None:
        app_profile = item_input_object.get("app_profile").get(31).get('number')
        app_profile = min(app_profile, 4)
    else:
        app_profile = -1

    # If laundry app count is default

    if ("31" in pilot_level_config.get('ld_config').get("drop_app")) or (
            "33" in pilot_level_config.get('ld_config').get("drop_app")):
        logger.debug("Default laundry app count to be used")

        devices = 0
        days_in_week = 0
        appliance_consumption = 0
        usage_hours = config.get("hours_arr")[np.digitize(delta, config.get("delta_arr"))]

    elif app_profile != -1:
        logger.debug("Dishwasher app profile present")

        days_in_week = config.get("default_dow")
        usage_hours = config.get("default_hours")
        devices = app_profile
        appliance_consumption = np.fmax(1500/samples_per_hour, config.get("amp_arr")[np.digitize(delta+300, config.get("delta_arr"))])

    else:

        logger.debug("Default laundry app count to be used")

        devices = config.get("devices_arr")[np.digitize(delta, config.get("delta_arr"))]
        days_in_week = config.get("dow_arr")[np.digitize(delta, config.get("delta_arr"))]
        appliance_consumption = config.get("amp_arr")[np.digitize(delta, config.get("delta_arr"))]
        usage_hours = config.get("hours_arr")[np.digitize(delta, config.get("delta_arr"))]

    total_samples = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    consumption = np.zeros(input_data.shape)

    devices = int(devices)

    logger.info("Dishwasher app count %d | ", devices)

    # Calculate timestamp level estimation for all the laundry devices

    for device in range(devices):

        end_time = start_time + usage_hours - 1

        consumption = add_noise_in_laundry(consumption, start_time, end_time, usage_hours, total_samples,
                                           appliance_consumption / samples_per_hour)

    # block laundry estimation for fraction of days

    seed = RandomState(random_gen_config.seed_value)

    non_laundry_days = np.array(seed.choice(list(np.arange(len(input_data))), int((Cgbdisagg.DAYS_IN_WEEK - days_in_week) / Cgbdisagg.DAYS_IN_WEEK * len(input_data))))

    consumption[(non_laundry_days).astype(int)] = 0

    logger.debug("Calculated dishwasher consumption")

    return consumption
