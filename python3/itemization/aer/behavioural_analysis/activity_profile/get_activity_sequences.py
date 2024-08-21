
"""
Author - Nisha Agarwal
Date - 8th Oct 20
Divide activity curve into sequences - increasing / decreasing / constant
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import rolling_func
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.init_itemization_config import init_itemization_params

from python3.itemization.aer.behavioural_analysis.activity_profile.postprocess_for_activity_sequences import post_process_using_neighbour_seq
from python3.itemization.aer.behavioural_analysis.activity_profile.postprocess_for_activity_sequences import post_process_to_remove_small_seqs

from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_activity_sequence_config import init_activity_sequences_config


def get_activity_sequences(activity_curve, samples_per_hour, activity_curve_diff, debug, logger_pass):

    """
    Runs modules to divide activity curve into sequences - increasing / decreasing / constant

        Parameters:
            activity_curve             (np.ndarray)     : living load activity profile
            samples_per_hour           (int)            : samples in an hour
            activity_curve_diff        (float)          : difference in max and min of activity curve
            debug                      (dict)           : debug dict containing intermediate hybrid results
            logger_pass                (dict)           : Contains the logger and the logging dictionary to be passed on

        Returns:
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            debug                      (dict)           : updated debug dict containing intermediate hybrid results
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_activity_sequences')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_activity_profile_start = datetime.now()

    smoothen_activity_curve = copy.deepcopy(activity_curve)

    config = init_activity_sequences_config()

    alpha = config.get('smooth_curve_config').get('alpha')

    higher_sampling_rate_factor = 4

    # exponential averaging of activity curve

    exp_avg_start_index = 3

    if samples_per_hour >= higher_sampling_rate_factor:

        logger.debug("Smoothning the activity curve since higher sampling rate")

        smoothen_activity_curve[: exp_avg_start_index] = activity_curve[: exp_avg_start_index]

        for i in range(exp_avg_start_index, len(activity_curve)):
            smoothen_activity_curve[i] = alpha * activity_curve[i] + (1 - alpha) * alpha * activity_curve[i - 1] + \
                                    (1 - alpha) * (1 - alpha) * activity_curve[i - 1]

    activity_curve_derivative = smoothen_activity_curve - np.roll(smoothen_activity_curve, 1)
    activity_curve_double_derivative = smoothen_activity_curve - np.roll(smoothen_activity_curve, 2)
    activity_curve_backward_derivative = np.roll(activity_curve_derivative, -1)

    # calculate score for individual data point suing their derivative information

    if samples_per_hour >= higher_sampling_rate_factor:
        weights = config.get('seq_config').get('30_min_weights')
    else:
        weights = config.get('seq_config').get('non_30_min_weights')

    # Score being calculated using cube of individual element

    temp_activity_seq = 1000 * (weights[0] * np.power(activity_curve_derivative, 3) +
                                weights[1] * np.power(activity_curve_double_derivative, 3) +
                                weights[2] * np.power(activity_curve_backward_derivative, 3))

    activity_seq = np.sign(temp_activity_seq)

    # Calculate threshold for the score

    # Taking cube in order to normalize the threshold value with respect to calculated score

    if samples_per_hour < config.get('seq_config').get('higher_samples_per_hour_limit'):

        threshold = (np.power(np.max(activity_curve) - np.min(activity_curve), 3) * config.get('seq_config').get('threshold_multiplier'))/samples_per_hour

    else:

        threshold = (np.power(np.max(activity_curve) - np.min(activity_curve), 3) * config.get('seq_config').get('threshold_multiplier'))/samples_per_hour

        threshold = max(config.get('seq_config').get('min_threshold'), threshold)

    # using threshold and score values, mark each point as increasing, decreasing or constant

    activity_seq[np.abs(temp_activity_seq) < threshold] = 0

    logger.debug("Calculated activity sequences")

    debug.update({
        "initial_activity_seq": activity_seq
    })

    # post processing steps to handle all corner cases

    if samples_per_hour > config.get('seq_config').get('samples_per_hour_limit'):

        logger.debug("Postprocessing steps on activity sequences since higher sampling rate")

        activity_seq_slow_change = post_process_to_identify_slow_change(activity_seq, activity_curve, samples_per_hour, config, logger)

        debug.update({
            "activity_seq_slow_change": activity_seq_slow_change
        })

        logger.debug("Post process to identify slow increase/decrease done")

        activity_seq = merge_activity_sequences(activity_seq, activity_seq_slow_change, activity_curve_derivative,
                                                samples_per_hour, config, logger)

        debug.update({
            "activity_seq_after_merging": activity_seq
        })

        logger.debug("Post process to merge sequences done")

        activity_seq = post_process_to_remove_small_seqs(activity_seq, activity_curve, activity_curve_diff, samples_per_hour, config, logger)

        debug.update({
            "activity_seq_after_small_seq_removal": activity_seq
        })

        logger.debug("Post process to remove small sequences done")

        activity_seq = post_process_using_neighbour_seq(activity_seq, activity_curve, samples_per_hour, activity_curve_diff, config, logger)

        debug.update({
            "activity_seq_using_neighbour_seq": activity_seq
        })

        logger.debug("Post process using neighbour sequences done")

    t_activity_profile_end = datetime.now()

    logger.info("Calculation of activity sequences took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    debug.update({
        "final_activity_seq": activity_seq
    })

    return activity_seq, debug


def post_process_to_identify_slow_change(activity_sequences, activity_curve, samples_per_hour, config, logger):

    """
    post process to identify sequences with slow change

        Parameters:
            activity_sequences         (np.ndarray)     : labels of activity sequences of the user
            activity_curve             (np.ndarray)     : living load activity profile
            samples_per_hour           (int)            : samples in an hour
            config                     (dict)           : dict containing activity seq config values
            logger                     (logger)         : logger object

        Returns:
            activity_seq_slow_change   (np.ndarray)     : labels of activity sequences of the user
    """

    increasing_label = 1
    decreasing_label = -1

    t_activity_profile_start = datetime.now()

    activity_curve = np.array(activity_curve)

    # Initialize threshold

    threshold = config.get("slow_change_detection_config").get('max_min_diff_threshold')

    activity_curve_copy = copy.deepcopy(activity_curve)

    activity_curve_copy[activity_curve < (np.min(activity_curve) +
                                          config.get("slow_change_detection_config").get('diff_multiplier') *
                                          (np.max(activity_curve) - np.min(activity_curve)))] = 0

    # increase threshold if more high consumption points are available

    activity_curve_range = np.max(activity_curve) - np.min(activity_curve)
    total_samples = len(activity_curve_copy)

    high_cons_frac_thres = config.get('slow_change_detection_config').get('high_cons_fraction')
    high_cons_diff_thres = config.get('slow_change_detection_config').get('high_cons_diff_threshold')

    if np.count_nonzero(activity_curve_copy)/total_samples > high_cons_frac_thres and (activity_curve_range) > high_cons_diff_thres:
        threshold = threshold + config.get("slow_change_detection_config").get('max_min_diff_threshold_increament')

    # initialize activity sequences
    activity_seq_slow_change = copy.deepcopy(activity_sequences)
    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    higher_sampling_rate_factor = 4

    # Calculate rolling average and related parameters
    if samples_per_hour >= higher_sampling_rate_factor:
        logger.debug("Smoothning the activity curve since higher sampling rate")
        rolling_avg = rolling_func(activity_curve, int(samples_per_hour / 4))

    else:
        rolling_avg = copy.deepcopy(activity_curve)

    rolling_avg_derivative = rolling_avg - np.roll(rolling_avg, 1)
    rolling_avg_pos_label = np.zeros(len(activity_curve))
    rolling_avg_neg_label = np.zeros(len(activity_curve))

    limit = config.get('slow_change_detection_config').get('min_threshold')

    rolling_avg_pos_label[rolling_avg_derivative > -limit] = increasing_label
    rolling_avg_neg_label[rolling_avg_derivative < limit] = decreasing_label

    interval = 1 + samples_per_hour

    length = len(activity_seq_slow_change)

    # identify sequences with slow increase / decrease

    deri_threshold = config.get('slow_change_detection_config').get('derivative_threshold')

    for i in range(0, length):

        # logger.debug("Calculating activity sequence for slow change of timestamp %d | ", i)

        index_array = get_index_array(i, i+interval, len(activity_seq_slow_change))

        # mark the chunk as increasing or decreasing if all required conditions are true

        greater_len_seq = (np.sum(rolling_avg_pos_label[index_array] > 0) >= interval-1)
        seq_range_bool = activity_curve[(i + interval - 1) % length] - activity_curve[(i - 1) % length] > threshold
        seq_var_bool = (not (activity_curve_derivative[index_array][activity_curve_derivative[index_array] < 0].sum() < -deri_threshold))
        rolling_avg_bool = (rolling_avg_pos_label[i % length] == increasing_label)

        increasing_seq_bool = greater_len_seq and seq_range_bool and seq_var_bool and rolling_avg_bool

        greater_len_seq = (np.sum(rolling_avg_neg_label[index_array] < 0) >= interval-1)
        seq_range_bool = activity_curve[(i + interval - 1) % length] - activity_curve[(i - 1) % length] < -threshold
        seq_var_bool = (not (activity_curve_derivative[index_array][activity_curve_derivative[index_array] > 0].sum() > deri_threshold))
        rolling_avg_bool = (rolling_avg_neg_label[i % length] == decreasing_label)

        decreasing_seq_bool = greater_len_seq and seq_range_bool and seq_var_bool and rolling_avg_bool

        if increasing_seq_bool != decreasing_seq_bool:
            activity_seq_slow_change[index_array] = increasing_label * increasing_seq_bool + decreasing_label * decreasing_seq_bool

    t_activity_profile_end = datetime.now()

    logger.info("Post process to identify slow increase/decrease took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return activity_seq_slow_change


def merge_activity_sequences(activity_seq, activity_seq_slow_change, activity_curve_derivative, samples_per_hour, config, logger):

    """
    Merge the activity sequences capturing different information

        Parameters:
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            activity_seq_slow_change   (np.ndarray)     : labels of activity sequences of the user
            activity_curve_derivative  (np.ndarray)     : 1st order derivative of activity curve
            samples_per_hour           (int)            : samples in an hour
            config                     (dict)           : dict containing activity seq config values
            logger                     (logger)         : logger object

        Returns:
            final_activity_seq         (np.ndarray)     : labels of activity sequences of the user
    """

    increasing_label = 1
    decreasing_label = -1

    t_activity_profile_start = datetime.now()

    # merge original final_activity_seq sequence and final_activity_seq sequence derived using slow increase/decrease algorithm

    total_samples = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    seq_config = init_itemization_params().get('seq_config')

    activity_seq_chunks = find_seq(activity_seq, np.zeros(total_samples), np.zeros(total_samples))
    activity_seq_slow_change_chunks = find_seq(activity_seq_slow_change, np.zeros(total_samples), np.zeros(total_samples))

    index = 0

    while index < len(activity_seq):

        val = activity_seq_slow_change[index]

        # Merge increasing or decreasing sequences

        if val != 0 and activity_seq[index] == val:

            start1, end1 = get_start_end_of_activity_chunk(index, activity_seq, activity_seq_chunks, seq_config)

            start2, end2 = get_start_end_of_activity_chunk(index, activity_seq_slow_change, activity_seq_slow_change_chunks, seq_config)

            threshold1 = samples_per_hour

            threshold2 = int(samples_per_hour / 2)

            # If the sequences strongly overlap, make slow increasing seq as 0

            if (end1-start1+1) > threshold1:

                bool = start1 == start2 and not (end2-end1) >= threshold1
                bool = bool or (end1 == end2 and not (start1 - start2) >= threshold1)
                bool = bool or (not ((start1 - start2) > threshold2 and (end2 - end1) > threshold2))

                if bool:
                    activity_seq_slow_change[start2: end2 + 1] = 0
                    index = end2 + 1

        # If values are opposite and derivative is stronger, follow previous label

        elif val != 0 and activity_seq[index] == -val and np.abs(activity_curve_derivative[index]) > \
                config.get('merge_seq_config').get('threshold'):
            activity_seq_slow_change[index] = activity_seq[index]

        index = index + 1

    final_activity_seq = activity_seq_slow_change + activity_seq

    final_activity_seq[final_activity_seq < -1] = decreasing_label
    final_activity_seq[final_activity_seq > 1] = increasing_label

    t_activity_profile_end = datetime.now()

    logger.info("Post process to merge sequences took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return final_activity_seq


def get_start_end_of_activity_chunk(index, activity_seq, activity_seq_chunks, seq_config):

    """
    Get start and end of an activity chunk given index of a point in activity seq

        Parameters:
            index                   (int)            : index of ts
            activity_seq            (np.ndarray)     : labels of activity sequences of the user
            activity_seq_chunks     (np.ndarray)     : chunks of labels of activity sequences of the user
            seq_config              (dict)           : config dictionary

        Returns:
            start                   (int)            : start index
            end                     (int)            : end index
    """

    start_index = 0
    end_index = -1

    if index <= activity_seq_chunks[end_index, seq_config.get('end')]:
        start = int(activity_seq_chunks[np.where(
            activity_seq_chunks[:, seq_config.get('end')] >= index)[0][start_index], seq_config.get('start')])
        end = int(activity_seq_chunks[np.where(
            activity_seq_chunks[:, seq_config.get('end')] >= index)[0][start_index], seq_config.get('end')])

    else:
        start = int(activity_seq_chunks[start_index, seq_config.get('start')])
        end = int(activity_seq_chunks[end_index, seq_config.get('end')] + len(activity_seq))

    return start, end
