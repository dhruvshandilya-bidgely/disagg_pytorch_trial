
"""
Author - Nisha Agarwal
Date - 10th Nov 2020
Calculate top cleanest days required for lighting estimation
"""

# Import python packages

import logging
import numpy as np


def get_top_clean_days(item_input_object, lighting_config, logger_pass):

    """
    Identify top cleanest days using distribution of clean days score

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        lighting_config             (dict)          : dict containing lighting config values
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        top_clean_days              (np.ndarray)    : Indexes of top cleanest days to be used for lighting estimation
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_top_clean_days')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # Identify top cleanest days using distribution of clean days score

    score_limit = lighting_config.get('top_clean_days_config').get('score_limit')
    days_count = lighting_config.get('top_clean_days_config').get('days_count')

    clean_days_score = item_input_object.get("clean_day_score_object").get("clean_day_score")

    sorted_clean_days = (-clean_days_score).argsort()

    # Each if statement determines the number of top clean days to be considered
    # This threhsold value is calculated using the information of number of days lying a particular range of cleanliness score
    # If high number of days are clean, clean day score threshold is higher
    # All the days having higher score than the threshold are considered as clean day
    # These clean days are further used for lighting capacity calculation

    min_clean_days = lighting_config.get('top_clean_days_config').get('min_days')

    if len(clean_days_score) < min_clean_days:
        top_clean_days = sorted_clean_days[:int(len(clean_days_score)/2)]

    elif np.sum(clean_days_score >= score_limit[0]) >= days_count[0]:
        top_clean_days = np.where(clean_days_score >= score_limit[0])[0]

    elif np.sum(clean_days_score >= score_limit[1]) >= days_count[1]:
        top_clean_days = sorted_clean_days[:days_count[1]]

    elif np.sum(clean_days_score >= score_limit[2]) >= days_count[2]:
        top_clean_days = sorted_clean_days[:days_count[2]]

    else:
        top_clean_days = \
            (-clean_days_score).argsort()[:lighting_config.get('top_clean_days_config').get('default_days_count')]

    logger.info("Number of top cleanest days | %d", len(top_clean_days))

    logger.debug("Calculate top cleanest days for lighting estimation")

    return top_clean_days
