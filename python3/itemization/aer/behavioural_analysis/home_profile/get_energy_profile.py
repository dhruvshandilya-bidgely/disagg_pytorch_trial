
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Calculate energy profile for weekdays/weekends and difference energy profile
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import rolling_func
from python3.itemization.aer.functions.itemization_utils import rolling_func_along_row

from python3.itemization.aer.behavioural_analysis.home_profile.config.get_general_config import get_general_config


def get_energy_profile(item_input_object, item_output_object, logger_pass):

    """
    Calculate energy profile for weekdays/weekends by picking clean days consumption

    Parameters:
        item_input_object         (dict)           : Dict containing all hybrid inputs
        item_output_object        (dict)           : Dict containing all hybrid outputs
        logger_pass               (dict)           : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)           : Dict containing all hybrid inputs
        item_output_object        (dict)           : Dict containing all hybrid outputs
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_energy_profile')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    input_data = copy.deepcopy(item_input_object.get("item_input_params").get('day_input_data'))

    weekend_days = item_input_object.get("item_input_params").get('weekend_days')

    clean_day_mask = item_input_object.get("clean_day_score_object").get("clean_day_masked_array")
    clean_day_score = copy.deepcopy(item_input_object.get("clean_day_score_object").get("clean_day_score"))

    config = get_general_config()

    min_days_req_to_prepare_avg_based_prof = config.get('min_days_req_to_prepare_avg_based_prof')

    clean_day_score[clean_day_mask[:, 0] == config.get("non_clean_day_label")] = 0

    input_data = np.multiply(input_data, clean_day_score[:, None])

    t0 = datetime.now()

    # Calculate weekday energy profile

    weekday_energy_profile = \
        np.sum(input_data[np.logical_not(weekend_days)], axis=0) / np.sum(clean_day_score[np.logical_not(weekend_days)])

    # Calculate weekend energy profile

    weekend_energy_profile = np.sum(input_data[weekend_days], axis=0) / np.sum(clean_day_score[weekend_days])

    t1 = datetime.now()

    logger.info("Calculating individual energy profile took | %.3f s", get_time_diff(t0, t1))

    days_in_week = Cgbdisagg.DAYS_IN_WEEK

    # Higher weightage is given to cleaner day

    if len(clean_day_score) > days_in_week:
        avg_score = rolling_func(clean_day_score, days_in_week/2)[np.arange(0, len(clean_day_score), days_in_week)]
    else:
        avg_score = np.ones(len(clean_day_score)) * (1/len(clean_day_score))

    if len(input_data) > min_days_req_to_prepare_avg_based_prof:
        weekday_energy = rolling_func_along_row(input_data[np.logical_not(weekend_days)],
                                                len(config.get("weekdays"))/2)[np.arange(0, np.logical_not(weekend_days).sum(), len(config.get("weekdays")))]
        weekend_energy = rolling_func_along_row(input_data[weekend_days],
                                                len(config.get("weekends"))/2)[np.arange(0, weekend_days.sum(), len(config.get("weekends")))]
    else:
        weekend_energy = np.mean(input_data[weekend_days], axis=0)
        weekday_energy = np.mean(input_data[np.logical_not(weekend_days)], axis=0)

    week_energy_diff = weekday_energy[: min(len(weekday_energy), len(weekend_energy))] - weekend_energy[: min(len(weekday_energy), len(weekend_energy))]

    week_energy_diff = np.multiply(week_energy_diff, avg_score[: min(len(weekday_energy), len(weekend_energy))][:, None])
    week_energy_diff = np.sum(week_energy_diff, axis=0) / np.sum(avg_score)

    week_energy_diff = np.nan_to_num(week_energy_diff)

    t2 = datetime.now()

    logger.info("Calculating weekday/weekend difference energy profile took | %.3f s", get_time_diff(t1, t2))

    energy_profile = dict({
        "weekend_energy_profile": weekend_energy_profile,
        "weekday_energy_profile": weekday_energy_profile,
        "energy_diff": week_energy_diff
    })

    item_output_object.update({
        "energy_profile": energy_profile
    })

    return item_input_object, item_output_object
