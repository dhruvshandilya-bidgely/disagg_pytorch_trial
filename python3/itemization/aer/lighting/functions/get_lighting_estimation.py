
"""
Author - Nisha Agarwal
Date - 10th Nov 2020
Calculate timestamp level lighting estimation
"""

# Import python packages

import logging
import numpy as np
from numpy.random import RandomState
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.init_itemization_config import random_gen_config


def get_lighting_estimate(item_input_object, lighting_estimate_potential, lighting_capacity, samples_per_hour,
                          lighting_usage_potential, lighting_config, logger_pass):

    """
    Calculates tou level lighting estimation

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        lighting_estimate_potential (np.ndarray)    : Tou level lighting usage potential (Normalized)
        lighting_capacity           (float)         : calculated lighting capacity
        samples_per_hour            (int)           : samples in an hour
        lighting_usage_potential    (np.ndarray)    : tou level lighting usage potential  (based on sunrise sunset data)
        lighting_config             (dict)          : dict containing lighting config values
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        lighting_estimate           (np.ndarray)      : lighting TOU level estimate
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_lighting_estimate')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_lighting_start = datetime.now()

    # Fetch required input data

    vacation = item_input_object.get("item_input_params").get("vacation_data")

    vacation = np.sum(vacation, axis=1).astype(bool)

    lighting_estimate_potential = np.nan_to_num(lighting_estimate_potential)

    # Get lighting tou estimate

    lighting_estimate = lighting_estimate_potential * lighting_capacity

    default_cap = lighting_config.get("estimation_config").get("default_cap") / samples_per_hour

    if np.all(lighting_estimate == 0):

        logger.info("Lighting consumption is zero throughout the year, providing default output")
        lighting_estimate[:, lighting_config.get("estimation_config").get("default_morn")] =\
            lighting_usage_potential[:, lighting_config.get("estimation_config").get("default_morn")] * default_cap

        lighting_estimate[:, lighting_config.get("estimation_config").get("default_eve")] =\
            lighting_usage_potential[:, lighting_config.get("estimation_config").get("default_eve")] * default_cap

    lighting_estimate = post_process_lighting_est(lighting_estimate, samples_per_hour, lighting_config)

    logger.debug("Added noise to lighting ts level estimate")

    # Safety check

    lighting_estimate = np.minimum(lighting_estimate,
                                   item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :])

    non_lighting_hours = lighting_config.get("postprocess_lighting_config").get('non_lighting_hours')
    non_lighting_hours = np.arange(non_lighting_hours[0]*samples_per_hour, non_lighting_hours[1]*samples_per_hour + 1).astype(int)

    lighting_estimate[:, non_lighting_hours] = 0

    # zero lighting output on vacation days

    lighting_estimate[vacation, :] = 0

    t_lighting_end = datetime.now()

    logger.info("Calculation of lighting estimate took | %.3f s", get_time_diff(t_lighting_start, t_lighting_end))

    return lighting_estimate


def post_process_lighting_est(lighting_estimate, samples_per_hour, lighting_config):

    """
    Adding noise in the lighting usage end time

    Parameters:
        lighting_estimate           (np.ndarray)    : lighting TOU level estimate
        samples_per_hour            (int)           : samples in an hour
        lighting_config             (dict)          : dict containing lighting config values

    Returns:
        lighting_estimate           (np.ndarray)      : lighting TOU level estimate
    """

    length = samples_per_hour * Cgbdisagg.HRS_IN_DAY

    night_hours = np.arange(16*samples_per_hour, 26*samples_per_hour) % length

    lighting_end_tou = np.zeros(len(lighting_estimate))

    for index in range(len(lighting_estimate)):

        if not np.all(lighting_estimate[index] == 0):

            # Calculate lighting ending tou of each day, if any lighting consumption is present

            # if lighting not present during night
            if np.all(lighting_estimate[index, night_hours] == 0):
                continue

            # No lighting after 12 am
            elif lighting_estimate[index, 0] == 0:
                lighting_end_tou[index] = np.where(lighting_estimate[index] > 0)[0][-1]

            # lighting is present after 12 am
            else:
                lighting_end_tou[index] = (np.where(lighting_estimate[index] == 0)[0][0] - 1) % length

    lighting_end_tou = lighting_end_tou.astype(int)

    lighting_end_cons = lighting_estimate[np.arange(0, len(lighting_estimate)), lighting_end_tou]

    # Based on randomized approach, add/delete lighting consumption at last ts for all days,
    # in order to add variation in lighting end time

    seed = RandomState(random_gen_config.seed_value)

    rand_array = seed.normal(0.5, 0.5, lighting_end_tou.shape)

    buckets = lighting_config.get("estimation_config").get("random_arr_buc")

    lighting_estimate[rand_array < buckets[0], lighting_end_tou[rand_array < buckets[0]]] = 0
    lighting_estimate[rand_array > buckets[1], (lighting_end_tou[rand_array > buckets[1]]+1)%length] =\
        lighting_end_cons[rand_array > buckets[1]]

    return lighting_estimate
