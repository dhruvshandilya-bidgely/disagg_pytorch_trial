
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Master file for calculation of occupancy profile and occupants count
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.behavioural_analysis.home_profile.get_user_attributes import get_users_parameters
from python3.itemization.aer.behavioural_analysis.home_profile.get_office_goer_prob import get_office_goer_prob
from python3.itemization.aer.behavioural_analysis.home_profile.get_stay_at_home_prob import get_stay_at_home_prob
from python3.itemization.aer.behavioural_analysis.home_profile.get_early_arrival_prob import get_early_arrival_prob

from python3.itemization.aer.behavioural_analysis.home_profile.config.occupancy_profile_config import get_occupancy_profile_config


def get_occupancy_profile(item_input_object, item_output_object, logger_pass):

    """
    Master function for calculation of occupancy profile and occupants count

    Parameters:
        item_input_object         (dict)           : Dict containing all hybrid inputs
        item_output_object        (dict)           : Dict containing all hybrid outputs
        logger_pass                 (dict)           : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)           : Dict containing all hybrid inputs
        item_output_object        (dict)           : Dict containing all hybrid outputs
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_occupancy_profile')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    activity_curve = item_input_object.get("weekday_activity_curve")
    activity_seq = item_output_object.get("profile_attributes").get('activity_sequences')
    sleep_hours = item_output_object.get("profile_attributes").get('sleep_hours')
    samples_per_hour = item_input_object.get("item_input_params").get('samples_per_hour')
    activity_curve_range = np.percentile(activity_curve, 97) - np.percentile(activity_curve, 5)

    occupants_profile_config = get_occupancy_profile_config()

    general_config = occupants_profile_config.get("general_config")

    occupancy_feature = np.zeros(general_config.get("occupants_type_count"))

    logger.info("Activity curve range is %s | ", activity_curve_range)

    office_goer_index = general_config.get("office_goer_index")
    early_arrival_index = general_config.get("early_arrival_index")
    stay_at_home_index = general_config.get("stay_at_home_index")

    # If difference between max and min of activity curve is too low (not much living load present in the homes)

    if activity_curve_range <= general_config.get("activity_curve_range_limit")[0]:

        logger.info("Activity curve range is too low, assigning default values to occupancy profile")

        occupancy_feature = general_config.get("low_range_count")
        occupants_prob = general_config.get("low_range_prob")
        occupants_count = np.sum(occupancy_feature)

    else:
        occupancy_feature, occupants_prob, occupants_count = \
            calculate_prob_of_various_occupants_type(item_input_object, item_output_object, occupancy_feature, logger)

    # Calculating user attributes to be used in statistical appliances estimation

    t4 = datetime.now()

    user_attributes = \
        get_users_parameters(activity_curve, samples_per_hour, activity_seq, sleep_hours, occupancy_feature)

    t5 = datetime.now()

    logger.info("Calculating user attributes took | %.3f s", get_time_diff(t4, t5))

    occupants_profile = dict({

        "occupants_count": occupants_count,
        "occupants_features": occupancy_feature,
        "occupants_prob": occupants_prob,
        "user_attributes": user_attributes,
        "office_goer_index": office_goer_index,
        "early_arrival_index": early_arrival_index,
        "stay_at_home_index": stay_at_home_index

    })

    logger.info("Office going prob | %s", occupants_prob[office_goer_index])
    logger.info("Early arrival prob | %s", occupants_prob[early_arrival_index])
    logger.info("Home stayer prob | %s", occupants_prob[stay_at_home_index])
    logger.info("Calculated occupants count | %s", occupancy_feature)

    item_output_object.update({
        "occupants_profile": occupants_profile
    })

    return item_input_object, item_output_object


def calculate_prob_of_various_occupants_type(item_input_object, item_output_object, occupancy_feature, logger):

    """
    Master function for calculation of occupancy profile and occupants count

    Parameters:
        item_input_object          (dict)           : Dict containing all hybrid inputs
        item_output_object         (dict)           : Dict containing all hybrid outputs
        occupancy_feature          (np.ndarray)     : array containing information of presence of each occupants type
        logger                     (logger)         : Contains the logger and the logging dictionary to be passed on

    Returns:
        occupancy_feature          (np.ndarray)     : array containing information of presence of each occupants type
        occupants_prob             (np.ndarray)     : array containing information of probability of each occupants type
        occupants_count            (np.ndarray)     : array containing information of count of each occupants type
    """

    activity_curve = item_input_object.get("weekday_activity_curve")
    levels = item_output_object.get("profile_attributes").get('activity_levels')
    levels = np.unique(levels)
    activity_curve_range = np.percentile(activity_curve, 97) - np.percentile(activity_curve, 5)

    occupants_profile_config = get_occupancy_profile_config()

    general_config = occupants_profile_config.get("general_config")
    office_goer_index = general_config.get("office_goer_index")
    early_arrival_index = general_config.get("early_arrival_index")
    stay_at_home_index = general_config.get("stay_at_home_index")

    levels = np.sort(levels)

    # Calculating time stamp level activity level mapping

    if len(levels):
        levels_mapping = np.argmin(np.abs(activity_curve[:, None] - levels[None, :]), axis=1)
    else:
        levels_mapping = np.zeros(len(activity_curve))

    logger.debug("Calculated levels mapping")

    t0 = datetime.now()

    # Calculating prob of presence of early arrivals

    occupancy_feature[early_arrival_index], early_arrival = get_early_arrival_prob(item_input_object,
                                                                                   item_output_object)

    t1 = datetime.now()

    logger.info("Calculating probability of presence of early arrivals took | %.3f s", get_time_diff(t0, t1))

    # Calculating prob of stay at home people

    occupancy_feature[stay_at_home_index], home_stayer = \
        get_stay_at_home_prob(item_input_object, item_output_object, occupancy_feature[early_arrival_index])

    t2 = datetime.now()

    logger.info("Calculating probability of stay at home people took | %.3f s", get_time_diff(t1, t2))

    # Calculating prob of office going adults

    occupancy_feature[office_goer_index], office_goer, multi_office_goers, morning_activity = \
        get_office_goer_prob(item_input_object, item_output_object, levels_mapping,
                             occupancy_feature[stay_at_home_index], home_stayer)

    t3 = datetime.now()

    logger.info("Calculating probability of office going adults took | %.3f s", get_time_diff(t2, t3))

    # Post processing rules on number of occupants of each type

    if multi_office_goers and occupancy_feature[0]:
        occupancy_feature[office_goer_index] = general_config.get("max_occupants_type_count")

    if occupancy_feature[office_goer_index] == general_config.get("max_occupants_type_count") \
            and occupancy_feature[stay_at_home_index] == general_config.get("max_occupants_type_count"):
        occupancy_feature[stay_at_home_index] = 1

    if occupancy_feature[office_goer_index] == 0 and occupancy_feature[stay_at_home_index] == 0:
        occupancy_feature[stay_at_home_index] = general_config.get("default_occupants_count")

    if activity_curve_range <= general_config.get("activity_curve_range_limit")[1]:
        occupancy_feature[stay_at_home_index] = min(occupancy_feature[stay_at_home_index], 1)
        occupancy_feature[office_goer_index] = min(occupancy_feature[office_goer_index], 1)

    occupants_count = np.sum(occupancy_feature)

    if occupants_count == 0 and multi_office_goers:
        occupants_count = general_config.get("max_occupants_type_count")

    max_score = 0.99

    early_arrival = np.round(min(max_score, early_arrival), general_config.get("max_occupants_type_count"))
    office_goer = np.round(min(max_score, office_goer), general_config.get("max_occupants_type_count"))
    home_stayer = np.round(min(max_score, home_stayer), general_config.get("max_occupants_type_count"))

    occupants_prob = [office_goer, early_arrival, home_stayer]

    return occupancy_feature, occupants_prob, occupants_count
