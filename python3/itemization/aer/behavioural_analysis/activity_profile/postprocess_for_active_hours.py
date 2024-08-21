
"""
Author - Nisha Agarwal
Date - 8th Oct 20
Post processing steps for extension of inactive hours
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.functions.itemization_utils import rolling_func

from python3.itemization.init_itemization_config import init_itemization_params

from python3.itemization.aer.behavioural_analysis.activity_profile.extend_inactive_segments import fill_active_hours_array
from python3.itemization.aer.behavioural_analysis.activity_profile.extend_inactive_segments import extend_inactive_segments

from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_postprocess_active_hours_config import init_postprocess_active_hours_config


def extend_inactive_hours(activity_curve, samples_per_hour, activity_seq, active_hours, logger_pass):

    """
    Run modules to extend night inactive hours using TOD and derivative information

        Parameters:
            activity_curve             (np.ndarray)     : living load activity profile
            samples_per_hour           (int)            : samples in an hour
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            active_hours               (np.ndarray)     : active/nonactive mask array
            logger_pass                (dict)           : Contains the logger and the logging dictionary to be passed on

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('extend_inactive_hours')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_activity_profile_start = datetime.now()

    activity_curve_diff = np.max(activity_curve) - np.min(activity_curve)

    activity_curve_derivative = np.abs(activity_curve - np.roll(activity_curve, 1))

    activity_curve_derivative = activity_curve_derivative + np.roll(activity_curve_derivative, -1)

    config = init_postprocess_active_hours_config(activity_curve, activity_curve_derivative, activity_curve_diff,
                                                  int(samples_per_hour)).get("postprocess_active_hour_config")

    active_hours = 1 - active_hours

    active_hours_copy = copy.deepcopy(active_hours)

    # general sleeping hours

    probable_sleeping_hours = config.get('sleeping_hours')

    non_sleeping_hours = list(set(np.arange(len(activity_seq))) - set(probable_sleeping_hours))

    active_hours_copy[non_sleeping_hours] = 0

    seq = find_seq(active_hours_copy, np.zeros(len(activity_curve)), np.zeros(len(activity_curve)))
    seq = seq[seq[:, 0] == 1]

    seq_config = init_itemization_params().get("seq_config")

    if len(seq):
        max_length = seq[np.argmax(seq[:, seq_config.get("length")]), seq_config.get("length")]
    else:
        max_length = 0

    # Initialize scoring

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    activity_curve_derivative[np.abs(activity_curve_derivative) < config.get("deri_scoring_limit")] = 0

    inactivity_score =  get_inactivity_score(activity_curve, activity_seq, probable_sleeping_hours, active_hours,
                                             activity_curve_derivative, samples_per_hour, config)

    length = int(config.get('min_sleeping_hours') * samples_per_hour)

    logger.debug("Calculated sleeping hours score score for each timestamp")

    # lowest inactivity score to non sleeping hours

    inactivity_score[non_sleeping_hours] = -10

    zigzag_points = np.zeros(len(activity_curve))

    zigzag_len_limit = samples_per_hour*2 + 1

    # check if zigzag pattern is present in the activity curve

    if np.all(inactivity_score < 10):
        zigzag_present = 0
    else:
        zigzag_points[inactivity_score > 10] = 1

        zigzag_seq = find_seq(zigzag_points, np.zeros(len(activity_curve)), np.zeros(len(activity_curve)))

        zigzag_seq = zigzag_seq[zigzag_seq[:, 0] == 1, seq_config.get("length")]

        zigzag_present = np.max(zigzag_seq) >= zigzag_len_limit

    zigzag_absent = 1

    if zigzag_present:
        zigzag_absent, active_hours_copy = \
            zigzag_pattern_inactivity(inactivity_score, active_hours, samples_per_hour, activity_curve, max_length, seq_config,
                                      config, activity_seq, logger)

    if zigzag_absent:

        # restoring the score since no definite zigzag pattern present for the user
        inactivity_score = inactivity_score - config.get('zigzag_score')*2*(inactivity_score > config.get('zigzag_score')*2 - 5).astype(int)
        inactivity_score = inactivity_score - config.get('zigzag_score')*(inactivity_score > config.get('zigzag_score') - 5).astype(int)

    # Calculate chunk score

    chunk_score = np.roll(rolling_func(inactivity_score, (length)/2 , 0), int(-length/2))

    logger.debug("Calculated score for each chunk")

    seq = find_seq(active_hours[probable_sleeping_hours], np.zeros(len(activity_curve[probable_sleeping_hours])),
                   np.zeros(len(activity_curve[probable_sleeping_hours])))

    if np.any(seq[:, seq_config.get("label")] == 1):
        seq = seq[seq[:, seq_config.get("label")] == 1]
        continous_inactive_samples = np.max(seq[:, seq_config.get("length")])
    else:
        continous_inactive_samples = 0

    if continous_inactive_samples < config.get('min_sleeping_hours') * samples_per_hour:

        start = int(np.argmax(chunk_score))

        logger.info("Start time for minimum hours of inactive segment | %.2f", start)

        index_array = get_index_array(start, start + length, len(activity_curve))

        for index in range(len(index_array)):

            if not (not zigzag_absent and
                    activity_seq[index_array[index]] and
                    activity_curve_derivative[index_array[index]] > config.get('derivative_limit')):
                active_hours[int(index_array[index])] = 1

    else:
        logger.info("Minimum hours of inactivity present")

    # extend inactivity to the neighbourhood timestamps

    seq = find_seq(active_hours, activity_curve, np.zeros(len(activity_curve)))

    if np.any(seq[:, seq_config.get("label")] == 1):
        seq = seq[seq[:, seq_config.get("label")] == 1]

    for index in range(len(seq)):
        start = seq[index, seq_config.get("start")]
        max_length = seq[index, seq_config.get("length")]

        index_array = get_index_array(start, start + max_length, len(activity_curve))

        active_hours = extend_inactive_segments(active_hours, index_array, activity_curve, activity_seq, max_length,
                                                samples_per_hour, config, flag=True)

    active_hours = np.logical_or(active_hours, active_hours_copy)

    morning_hours = np.arange(int(4 * samples_per_hour), int(9.5 * samples_per_hour))

    # Fill morning segments as active hours, in case of reasonable evidence

    morning_seq = find_seq(activity_seq[morning_hours], activity_curve[morning_hours], np.zeros(len(activity_curve[morning_hours])))

    active_hours = fill_active_hours_array(active_hours, morning_hours, morning_seq, config, activity_seq, samples_per_hour)

    night_hours = np.arange(int(-2 * samples_per_hour), int(1 * samples_per_hour)) % len(activity_curve)

    night_seq = find_seq(activity_seq[night_hours], activity_curve[night_hours], np.zeros(len(activity_curve[night_hours])))

    active_hours = fill_active_hours_array(active_hours, night_hours, night_seq, config, activity_seq, samples_per_hour)

    active_hours = 1 - active_hours

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    active_hours[np.logical_and(activity_seq == 1, activity_curve_derivative >= config.get('derivative_limit'))] = 1

    t_activity_profile_end = datetime.now()

    logger.info("Post processing of active hours took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return active_hours


def zigzag_pattern_inactivity(score, active_hours, samples_per_hour, activity_curve, max_length, seq_config, config, activity_seq, logger):

    """
    Detect whether zigzag activity is present in the activity curve of the user (consecutive inc-dec pattern)

        Parameters:
            score                       (np.ndarray)        : tou level inactivity score
            active_hours                (np.ndarray)        : Array containing active/inactive hours information
            samples_per_hour            (int)               : samples in an hour
            activity_curve              (np.ndarray)        : Activity curve of the user
            max_length                  (int)               : max possible length of an inactive sequence
            seq_config                  (dict)              : config dict for seq array indices
            config                      (config)            : Config dictionary
            activity_seq                (np.ndarray)        : Array containing activity sequence information
            logger                      (logger)            : logger object

        Returns:
            zigzag_absent               (bool)              : bool for whether zigzag activity is present
            active_hours_copy           (np.ndarray)        : updated active hours array
    """

    zigzag_absent = 0

    active_hours_copy = copy.deepcopy(active_hours)

    index_array = np.where(score > 10)[0]

    index_array_copy = copy.deepcopy(index_array)
    index_array_copy = index_array_copy[index_array_copy < 12 * samples_per_hour]

    if not len(index_array_copy):
        zigzag_absent = 1
    else:
        logger.info("Found zigzag pattern")

        active_hours_copy[index_array_copy[-1]: index_array_copy[-1] + 2 * samples_per_hour] = 1
        active_hours_copy[index_array_copy[0] - 2 * samples_per_hour: index_array_copy[0]] = 1

        active_hours_copy[score > 10] = 1

        if samples_per_hour > 1:

            old_seq = find_seq(active_hours_copy, activity_curve, np.zeros(len(activity_curve)))
            old_seq[np.logical_and(old_seq[:, seq_config.get("label")] == 0,
                                   old_seq[:, seq_config.get("length")] <= 1), 0] = 1

            for index in range(len(old_seq)):
                active_hours_copy[
                    get_index_array(old_seq[index, seq_config.get("start")],
                                    old_seq[index, seq_config.get("end")], len(activity_curve))] = \
                    old_seq[index, seq_config.get("label")]

        active_hours_copy = extend_inactive_segments(active_hours_copy, index_array, activity_curve, activity_seq,
                                                     max_length, samples_per_hour, config, flag=False)

    return zigzag_absent, active_hours_copy


def get_inactivity_score(activity_curve, activity_seq, probable_sleeping_hours, active_hours,
                         activity_curve_derivative, samples_per_hour, config):

    """
        Calculate tou level inactivity score

        Parameters:
            activity_curve              (np.ndarray)        : Activity curve of the user
            activity_seq                (np.ndarray)        : Array containing activity sequence information
            probable_sleeping_hours     (np.ndarray)        : defined probable sleeping hours
            active_hours                (np.ndarray)        : Array containing active/inactive hours information
            activity_curve_derivative   (np.ndarray)        : Activity curve derivative
            samples_per_hour            (int)               : samples in an hour
            config                      (np.ndarray)        : Config dictionary

        Returns:
            inactivity_score            (dict)              : Calculated tou level inactivity score

    """

    activity_curve_buckets = config.get("activity_curve_buckets")

    derivative_buckets = config.get("derivative_buckets")

    zigzag_pattern_interval = config.get("zigzag_hour_interval") * samples_per_hour
    length = len(activity_curve)

    inactivity_score = config.get('active_hours_score')

    zigzag_pattern_thres = 0.1 if samples_per_hour == 1 else 0.25

    for index in probable_sleeping_hours:

        if active_hours[index]:
            inactivity_score[index] = inactivity_score[index] + config.get('non_active_tod_score')

        else:

            # assign score to each timestamp, higher the score, higher the chances of it being inactive

            index_array1 = activity_curve_derivative[
                get_index_array(index % length, (index + zigzag_pattern_interval) % length, len(activity_curve))]
            index_array_copy = activity_curve_derivative[
                get_index_array((index - zigzag_pattern_interval) % length, index % length, len(activity_curve))]

            deri_sum = np.abs(np.sum(index_array1) + np.sum(index_array_copy))

            low_deri_bool1 = not np.all(np.abs(index_array1) < 0.03)
            low_deri_bool2 = not np.all(np.abs(index_array_copy) < 0.03)

            # higher inactivity score to zigzag score

            bool1 = low_deri_bool1 and \
                    deri_sum < 0.1 and \
                    (np.sum(index_array1 > 0) / np.sum(index_array1 < 0)) > 1-zigzag_pattern_thres and \
                    (np.sum(index_array1 > 0) / np.sum(index_array1 < 0)) < 1+zigzag_pattern_thres

            bool2 = low_deri_bool2 and \
                    deri_sum < 0.1 and \
                    (np.sum(index_array_copy > 0) / np.sum(index_array_copy < 0)) > 1-zigzag_pattern_thres and \
                    (np.sum(index_array_copy > 0) / np.sum(index_array_copy < 0)) < 1+zigzag_pattern_thres

            inactivity_score[index] = inactivity_score[index] + config.get('zigzag_score') * int(bool1)
            inactivity_score[index] = inactivity_score[index] + config.get('zigzag_score') * int(bool2)

            # inactivity based on activity seq information

            inactivity_score[index] = inactivity_score[index] + config.get('increasing_score') * (activity_seq[index] == 1)

            inactivity_score[index] = inactivity_score[index] + config.get('decreasing_score') * (activity_seq[index] == -1)
            inactivity_score[index] = inactivity_score[index] - 0.1*np.digitize(activity_curve[index], activity_curve_buckets) * (activity_seq[index] == -1)

            inactivity_score[index] = inactivity_score[index] + config.get('constant_score') * (activity_seq[index] == 0)
            inactivity_score[index] = inactivity_score[index] - 0.1*np.digitize(activity_curve[index], activity_curve_buckets) * (activity_seq[index] == 0)

        inactivity_score[index] = inactivity_score[index] - 0.3*np.digitize(np.abs(activity_curve_derivative[index]), derivative_buckets)

    return inactivity_score
