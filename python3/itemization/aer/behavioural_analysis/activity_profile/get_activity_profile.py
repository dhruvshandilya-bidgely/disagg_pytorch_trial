
"""
Author - Nisha Agarwal
Date - 8th Oct 20
Prepare activity curve of a user using input data
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.itemization.aer.behavioural_analysis.activity_profile.prepare_activity_profile import prepare_activity_curve


def get_activity_profile(item_input_object, item_output_object, logger_pass):

    """
    Prepare living load activity profile of the user

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_input_object         (dict)      : Dict containing all outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('prepare_activity_profile')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    vacation = item_input_object.get("item_input_params").get('vacation_data')
    input_data = item_input_object.get("item_input_params").get('day_input_data')
    weekend_days = item_input_object.get("item_input_params").get('weekend_days')

    logger.info("Calculating all days activity curve")

    item_input_object, item_output_object, activity_curve, perc_value, chunks_data = \
        prepare_activity_curve(input_data, vacation, item_input_object, item_output_object, logger)

    weekday_vacation = vacation[np.logical_not(weekend_days)]
    weekday_input_data = input_data[np.logical_not(weekend_days)]

    logger.info("Calculating weekday days activity curve")

    if np.sum(np.logical_not(weekend_days)):

        # Calculating weekday days activity curve
        item_input_object, item_output_object, weekday_activity_curve, weekday_perc_value, weekday_chunks_data = \
            prepare_activity_curve(weekday_input_data, weekday_vacation, item_input_object, item_output_object, logger)

    else:
        weekday_activity_curve = np.zeros_like(activity_curve)

    weekend_vacation = vacation[weekend_days]
    weekend_input_data = input_data[weekend_days]

    logger.info("Calculating weekend days activity curve")

    if np.sum(weekend_days):
        # Calculating weekend days activity curve

        item_input_object, item_output_object, weekend_activity_curve, weekend_perc_value, weekend_chunks_data = \
            prepare_activity_curve(weekend_input_data, weekend_vacation, item_input_object, item_output_object, logger)

    else:
        weekend_activity_curve = np.zeros_like(activity_curve)

    item_input_object.update({

        "activity_curve": activity_curve,
        "weekday_activity_curve": weekday_activity_curve,
        "weekend_activity_curve": weekend_activity_curve,
        "perc_value": perc_value,
        "chunks_data": chunks_data

    })

    return item_input_object, item_output_object
