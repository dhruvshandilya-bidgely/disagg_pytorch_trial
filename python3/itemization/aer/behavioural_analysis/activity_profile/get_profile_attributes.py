
"""
Author - Nisha Agarwal
Date - 8th Oct 20
Calculate various attributes using activity curve
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.behavioural_analysis.activity_profile.get_sleep_hours import get_sleep_hours
from python3.itemization.aer.behavioural_analysis.activity_profile.get_active_hours import get_active_hours
from python3.itemization.aer.behavioural_analysis.activity_profile.get_activity_levels import get_activity_levels
from python3.itemization.aer.behavioural_analysis.activity_profile.get_activity_segments import get_activity_segments
from python3.itemization.aer.behavioural_analysis.activity_profile.get_wakeup_sleep_time import get_wakeup_sleep_time
from python3.itemization.aer.behavioural_analysis.activity_profile.get_activity_sequences import get_activity_sequences
from python3.itemization.aer.behavioural_analysis.activity_profile.postprocess_for_active_hours import extend_inactive_hours

from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_activity_profile_config import init_activity_profile_config


def get_profile_attributes(item_input_object, item_output_object, logger_pass):

    """
    Runs modules to calculate required attributes of activity profile

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_input_object         (dict)      : Dict containing all outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_profile_attributes')
    logger_activity_profile = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_activity_profile_start = datetime.now()

    debug = dict()

    activity_curve = item_input_object.get("activity_curve")
    samples_per_hour = item_input_object.get("item_input_params").get('samples_per_hour')

    config = init_activity_profile_config(samples_per_hour)

    activity_curve_diff = np.percentile(activity_curve, config.get('general_config').get('activity_curve_max_perc')) - \
                          np.percentile(activity_curve, config.get('general_config').get('activity_curve_min_perc'))

    if np.max(activity_curve) == np.min(activity_curve):
        logger_activity_profile.warning("The activity profile is invalid")
        return item_input_object, item_output_object

    logger_activity_profile.info("Difference in max and min of activity curve | %.3f ", activity_curve_diff)

    # calculate activity sequences as increasing, decreasing, or constant

    activity_sequences, debug = get_activity_sequences(activity_curve, samples_per_hour, activity_curve_diff, debug, logger_pass)

    logger_activity_profile.debug("Calculated activity sequences")

    # calculate activity segments as plain, uphill, downhill, plateau, mountain

    activity_segments, active_hours = get_activity_segments(samples_per_hour, activity_curve, activity_curve_diff, activity_sequences, logger_pass)

    logger_activity_profile.debug("Calculated activity segments")

    # calculate levels of activity

    activity_levels, range, lowest_level = get_activity_levels(samples_per_hour, activity_curve, activity_curve_diff, activity_segments, logger_pass)

    logger_activity_profile.debug("Calculated activity levels")

    # Calculate active/non active hours

    debug.update({
        "activity_curve": activity_curve,
        "activity_curve_diff": activity_curve_diff,
        "activity_sequences": activity_sequences,
        "activity_segments": activity_segments
    })

    active_hours = get_active_hours(samples_per_hour, active_hours, range, lowest_level, debug, logger_pass)

    debug.update({
        "initial_active_hours": active_hours,
    })

    logger_activity_profile.debug("Calculated active hours")

    # Post process for calculating inactive hours

    active_hours = extend_inactive_hours(activity_curve, samples_per_hour, activity_sequences, active_hours, logger_pass)

    debug.update({
        "active_hours_after_extension": active_hours
    })

    logger_activity_profile.debug("Extended inactive hours")

    # Calculate sleeping hours of the user

    sleep_hours = get_sleep_hours(activity_curve, samples_per_hour, activity_sequences, active_hours, logger_pass)

    logger_activity_profile.debug("Calculated final sleep time")

    wakeup_time, sleep_time, sleep_time_int, wakeup_time_int = get_wakeup_sleep_time(samples_per_hour, sleep_hours, config)

    if wakeup_time == -1:
        wakeup_time_int = config.get("default_wakeup_int")
        logger_activity_profile.info("Wakeup time of the user | %s ", config.get("default_wakeup"))
        logger_activity_profile.warning("Providing default wakeup time for the user")
    else:
        logger_activity_profile.info("Wakeup time of the user | %s ", wakeup_time)

    if sleep_time == -1:
        sleep_time_int = config.get("default_sleep_int")
        logger_activity_profile.info("Sleep time of the user | %s ", config.get("default_sleep"))
        logger_activity_profile.warning("Providing default sleep time for the user")
    else:
        logger_activity_profile.info("Sleep time of the user | %s ", sleep_time)

    profile_attributes = {
        "activity_sequences": activity_sequences,
        "activity_segments": activity_segments,
        "activity_levels": activity_levels,
        "active_hours": active_hours,
        "sleep_hours": sleep_hours,
        "activity_curve": activity_curve,
        "sleep_time": sleep_time_int,
        "wakeup_time": wakeup_time_int
    }

    debug.update(profile_attributes)

    item_output_object.update({
        "profile_attributes": profile_attributes
    })

    item_output_object.get('debug').update({
        "profile_attributes_dict": debug
    })

    t_activity_profile_end = datetime.now()

    logger_activity_profile.info('Calculation of activity profile took | %.3f s ',
                                 get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return item_input_object, item_output_object
