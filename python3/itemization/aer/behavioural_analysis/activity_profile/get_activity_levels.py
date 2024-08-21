
"""
Author - Nisha Agarwal
Date - 8th Oct 20
Calculate levels of activity
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_activity_levels_config import init_activity_levels_config
from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_activity_profile_config import init_activity_profile_config


def get_activity_levels(samples_per_hour, activity_curve, activity_curve_diff, activity_segments, logger_pass):

    """
    Divide the activity curve into different zones of activity levels

    Parameters:
        samples_per_hour           (int)            : samples in an hour
        activity_curve_diff        (float)          : diff in max and min of activity curve
        activity_segments          (np.ndarray)     : array containing information for individual segments
        logger_pass                (dict)           : Contains the logger and the logging dictionary to be passed on

    Returns:
        merged_levels              (np.ndarray)     : list of levels of activity
        range                      (float)          : cluster distance used for calculating levels
        lowest_levels              (float)          : lowest level of activity
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_activity_levels')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_activity_profile_start = datetime.now()

    activity_level_config = init_activity_levels_config()
    segments_index_config = init_activity_profile_config(samples_per_hour).get('segments_config')

    # fetch required activity segments information

    segment_levels = activity_segments[:, segments_index_config.get('level')]
    segment_levels[segment_levels < 0] = 0
    morning_segment = activity_segments[:, segments_index_config.get('morning')]
    segment_location = activity_segments[:, segments_index_config.get('location')]
    segment_list = activity_segments[:, segments_index_config.get('type')]

    segment_levels = segment_levels[segment_list <= 3]
    morning_segment = morning_segment[segment_list <= 3]
    segment_location = segment_location[segment_list <= 3]

    # Merge levels into groups

    merged_levels, range = merge_levels(samples_per_hour, activity_curve_diff, segment_levels, morning_segment,
                                        segment_location, activity_level_config)

    # Calculate value of lowest level

    merged_levels = merged_levels[merged_levels > np.min(activity_curve)]

    logger.debug("Merging of levels done | ")

    if len(merged_levels):
        lowest_level = np.min(merged_levels)
    else:
        lowest_level = 0

    merged_levels = np.unique(merged_levels)

    logger.info("Lowest level of activity | %d ", lowest_level)
    logger.info("Final merged levels | %s ", np.round(merged_levels, 3))
    logger.info("Cluster distance for levels calculation | %d", range)

    t_activity_profile_end = datetime.now()

    logger.info("Calculation of activity levels took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return merged_levels, range, lowest_level


def merge_levels(samples_per_hours, activity_curve_diff, segment_levels, morning_segments, location, config):

    """
    divide the levels iof activity into groups

    Parameters:
        samples_per_hour           (int)            : samples in an hour
        activity_curve_diff        (float)          : diff in max and min of activity curve
        segment_levels             (np.ndarray)     : levels of activity
        morning_segments           (np.ndarray)     : bool array - 1 if the segment is present in the morning hours
        location                   (np.ndarray)     : tod of the corresponding segment
        config                     (dict)           : dict containing activity levels config values

    Returns:
        levels                     (np.ndarray)     : final merged levels
        range                      (float)          : cluster distance used to merge levels
    """

    levels_array = np.vstack((segment_levels, morning_segments, location)).T

    # Sort the array based on increasing levels

    levels_array = levels_array[levels_array[:, 0].argsort()]

    location = levels_array[:, 2]
    segment_levels = levels_array[:, 0]
    morning_segments = levels_array[:, 1]

    levels = copy.deepcopy(segment_levels)

    segments_copy = np.arange(len(segment_levels))

    # cluster distance calculated based on difference in max and min of the input load curve

    win_strt_idx = np.digitize(activity_curve_diff, config.get("levels_config").get('activity_curve_diff_arr'))

    range = config.get("levels_config").get('threshold_array')[win_strt_idx]

    # To relax the limits for change in levels in morning hours

    morning_range = max(range - config.get("levels_config").get('decreament'), config.get("levels_config").get('min_range'))

    # To handle users with sampling rate greater than 15 min

    if samples_per_hours < 4:
        range = range + config.get("levels_config").get('increament')/samples_per_hours
        morning_range = morning_range + config.get("levels_config").get('increament')/samples_per_hours

    win_strt_idx = 0

    # divide the levels into groups using calculated cluster distance

    while win_strt_idx < len(segment_levels):

        cluster_distance = morning_range if morning_segments[win_strt_idx] else range

        intersection = np.where(segment_levels[: win_strt_idx] > segment_levels[win_strt_idx]-cluster_distance)[0]

        # The indices where the difference of previous points are less than defined distance,update the labels

        for sim_cls_idx in intersection:

            position_diff = np.absolute(location[sim_cls_idx] - location[segments_copy[sim_cls_idx]]) > \
                            np.absolute(location[sim_cls_idx] - location[win_strt_idx])

            level_diff = np.absolute(levels[sim_cls_idx] - levels[segments_copy[sim_cls_idx]]) > \
                         np.absolute(levels[sim_cls_idx] - levels[win_strt_idx])

            if position_diff and level_diff:
                levels[sim_cls_idx] = levels[win_strt_idx]
                segments_copy[sim_cls_idx] = win_strt_idx

        win_end_idx = win_strt_idx + 1

        # Update the pointer till the points that lie within the distance

        while win_end_idx < len(segment_levels) and (segment_levels[win_strt_idx] + cluster_distance) > segment_levels[win_end_idx]:

            levels[win_end_idx] = levels[win_strt_idx]
            segments_copy[win_end_idx] = win_strt_idx
            win_end_idx = win_end_idx + 1

        win_strt_idx = win_end_idx

    return levels, range
