"""
Author - Nisha Agarwal
Date - 8th Oct 20
Mask active/non active timestamps in the activity curve
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import fill_array
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.init_itemization_config import init_itemization_params
from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_active_hours_config import init_active_hours_config
from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_activity_profile_config import init_activity_profile_config


def get_active_hours(samples_per_hour, active_hours, range, lowest_level, debug, logger_pass):

    """
    Run modules to calculate active/nonactive hours for a user

      Parameters:
          samples_per_hour           (int)            : samples in an hour
          activity_curve             (np.ndarray)     : living load activity profile
          activity_curve_diff        (float)          : difference in max and min of activity curve
          activity_sequences         (np.ndarray)     : labels of activity sequences of the user
          active_hours               (np.ndarray)     : active/nonactive mask array
          activity_segments          (np.ndarray)     : array containing information for individual segments
          range                      (float)          : distance used to merge levels
          lowest_level               (float)          : lowest level of activity
          logger_pass                (dict)           : Contains the logger and the logging dictionary to be passed on

      Returns:
          active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    activity_curve = debug.get("activity_curve")
    activity_segments = debug.get("activity_segments")
    activity_sequences = debug.get("activity_sequences")
    activity_curve_diff = debug.get("activity_curve_diff")

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_active_hours')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # fetch config required to calculate active/nonactive hours

    active_hour_config = init_active_hours_config(int(samples_per_hour))

    t_activity_profile_start = datetime.now()

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    activity_seq_chunks = find_seq(activity_sequences, activity_curve, activity_curve_derivative)

    # calculate active / non-active hours
    active_hours = calculate_active_hours(samples_per_hour, activity_curve, activity_curve_diff, activity_seq_chunks,
                                          active_hours, activity_segments, range, lowest_level, active_hour_config)

    logger.debug("Calculated active hours")

    # post process to filter out small active hours sequence
    active_hours = filter_active_hours(active_hours, samples_per_hour, activity_curve, active_hour_config)

    logger.debug("Filtered active hours")

    t_activity_profile_end = datetime.now()

    logger.info("Calculation of active hours took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return active_hours


def calculate_active_hours(samples_per_hour, activity_curve, activity_curve_diff, activity_seq_chunks, active_hours,
                           activity_segments, levels_range, lowest_level, config):

    """
    Run modules to calculate active/nonactive hours for a user

      Parameters:
          samples_per_hour           (int)            : samples in an hour
          activity_curve             (np.ndarray)     : living load activity profile
          activity_curve_diff        (float)          : difference in max and min of activity curve
          activity_seq_chunks        (np.ndarray)     : labels of activity sequences of the user
          active_hours               (np.ndarray)     : active/nonactive mask array
          activity_segments          (np.ndarray)     : array containing information for individual segments
          levels_range               (float)          : distance used to merge levels
          lowest_level               (float)          : lowest level of activity
          config                     (dict)           : dict containing active hours config values
          logger_pass                (dict)           : Contains the logger and the logging dictionary to be passed on

      Returns:
          active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    seq_config = init_itemization_params().get('seq_config')
    segments_index_config = init_activity_profile_config(samples_per_hour).get('segments_config')

    lowest_is_plain_flag = 0

    score = np.zeros(len(activity_segments))

    # A given segment is marked active/inactive based on 4 factors
    # If plain has less variation
    # If the plain is in pattern Decrease -> Constant -> Increase
    # If the plain has less net slope
    # If the plain is a lower level plain
    # If atleast 3 of these are false, the segment is likely to be marked as active

    # variation based score calculation

    score = score + config.get('active_hours_config').get('weightage')[0] * \
            (activity_segments[:, segments_index_config.get('variation')] < config.get('active_hours_config').get(
                'var_limit')).astype(int)

    # pattern (Decrease -> Constant -> Increase)  based score calculation

    score = score + config.get('active_hours_config').get('weightage')[1] * activity_segments[:, segments_index_config.get('pattern')]

    # slope based score calculation

    score = score + config.get('active_hours_config').get('weightage')[2] * \
            (np.abs(activity_segments[:, segments_index_config.get('slope')]) < config.get(
                'active_hours_config').get('slope_limit')).astype(int)

    # level based score calculation

    score = score + config.get('active_hours_config').get('weightage')[3] * \
            (np.abs(activity_segments[:, segments_index_config.get('level')] - lowest_level) < activity_curve_diff).astype(int)

    for i in range(len(activity_segments)):

        # Mark a plain as an inactive phase based on factors

        if (activity_segments[i, 0] == 1) * int(np.abs(activity_segments[i, segments_index_config.get('level')] - lowest_level) < levels_range):
            lowest_is_plain_flag = 1

        # If atleast 3 factors are false, it is marked as an active segment

        # Or if level of the segment is greater than a certain limit

        if activity_segments[i, 0] == 1 and score[i] < config.get('active_hours_config').get('score_limit') or\
                (activity_segments[i, segments_index_config.get('level')] >
                 (np.min(activity_curve) + config.get('active_hours_config').get('diff_factor') * activity_curve_diff)):

            active_hours = fill_array(active_hours, activity_segments[i, segments_index_config.get('start')],
                                      activity_segments[i, segments_index_config.get('end')], 1)

    # If the lowest segment of the curve is a plain, mark the points on lower levels as inactive

    if lowest_is_plain_flag and lowest_level < (np.min(activity_curve) +
                                                config.get('active_hours_config').get('diff_factor') * activity_curve_diff):

        for i in range(len(activity_seq_chunks)):

            index_array = get_index_array(activity_seq_chunks[i, seq_config.get('start')],
                                          activity_seq_chunks[i, seq_config.get('end')], len(active_hours))

            # Possible scenarios where the sequence is active

            if len(index_array) <= config.get("active_hours_config").get('hour_limit')*samples_per_hour and \
                    activity_seq_chunks[i, seq_config.get('net_deri')] > config.get("active_hours_config").get('derivative_limit') and \
                    activity_seq_chunks[i, seq_config.get('label')] == 1:
                continue

            # different threshold for scenarios - decreasing seq, constant seq
            level = lowest_level + levels_range if activity_seq_chunks[i, seq_config.get('label')] == 0 else (lowest_level + levels_range / 2)

            # if the values are less than the level threshold, assign them as inactive hours

            active_hours_values = active_hours[index_array]

            final_curve_values = activity_curve[index_array]

            active_hours_values[final_curve_values < level] = 0

            active_hours[index_array] = active_hours_values

    return active_hours


def filter_active_hours(active_hours, samples_per_hour, activity_curve, config):

    """
    Filter smaller chunks of active/nonactive hours

      Parameters:
          active_hours               (np.ndarray)     : active/nonactive mask array
          samples_per_hour           (int)            : samples in an hour
          activity_curve             (np.ndarray)     : living load activity profile
          config                     (dict)           : dict containing active hours config values

      Returns:
          active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    limit = int(samples_per_hour / 2)

    seq_config = init_itemization_params().get('seq_config')

    # No filtering required for 60 min data

    if limit < 1:
        return active_hours

    threshold = config.get('active_hours_config').get('filter_active_hours_limit')

    # Adjusting threshold for lower activity range users
    if (np.max(activity_curve) - np.min(activity_curve)) < config.get('active_hours_config').get('diff_limit'):
        threshold = threshold - config.get('active_hours_config').get('small_diff_decreament')

    derivative = activity_curve - np.roll(activity_curve, 1)

    old_active_hours_seq = find_seq(active_hours, activity_curve, derivative)

    # filter out less significant chunk of inactive hours
    insignificant_seq = np.logical_and(old_active_hours_seq[:, seq_config.get('label')] == 0,
                                       old_active_hours_seq[:, seq_config.get('length')] <= limit)
    insignificant_seq = np.logical_and(insignificant_seq, old_active_hours_seq[:, seq_config.get('max_deri')] <= threshold)
    old_active_hours_seq[insignificant_seq, 0] = 1

    for i in range(len(old_active_hours_seq)):
        active_hours = fill_array(active_hours,
                                  old_active_hours_seq[i, seq_config.get('start')],
                                  old_active_hours_seq[i, seq_config.get('end')],
                                  old_active_hours_seq[i, seq_config.get('label')])

    new_active_hours_seq = find_seq(active_hours, activity_curve, derivative)

    # filter out less significant chunk of active hours
    insignificant_seq = np.logical_and(old_active_hours_seq[:, seq_config.get('label')] == 1,
                                       old_active_hours_seq[:, seq_config.get('length')] <= limit)
    insignificant_seq = np.logical_and(insignificant_seq, old_active_hours_seq[:, seq_config.get('max_deri')] <= threshold)
    old_active_hours_seq[insignificant_seq, 0] = 0

    for i in range(len(new_active_hours_seq)):
        active_hours = fill_array(active_hours,
                                  new_active_hours_seq[i, seq_config.get('start')],
                                  new_active_hours_seq[i, seq_config.get('end')],
                                  new_active_hours_seq[i, seq_config.get('label')])

    return active_hours
