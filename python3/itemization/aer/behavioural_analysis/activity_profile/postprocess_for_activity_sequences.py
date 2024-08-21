"""
Author - Nisha Agarwal
Date - 8th Oct 20
post processing steps for calculation of increasing/decreasing sequences
"""

# Import python packages

import copy
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff
from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import fill_array
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.init_itemization_config import init_itemization_params


def post_process_to_remove_small_seqs(activity_seq, activity_curve, activity_curve_diff, samples_per_hour, config, logger):

    """
    post process to remove small activity sequences

        Parameters:
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            activity_curve             (np.ndarray)     : living load activity profile
            activity_curve_diff        (int)            : range of derived activity curve
            samples_per_hour           (int)            : samples in an hour
            config                     (dict)           : dict containing activity seq config values
            logger                     (logger)         : logger object

        Returns:
            activity_seq              (np.ndarray)     : labels of activity sequences of the user
    """

    t_activity_profile_start = datetime.now()

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    activity_curve_diff = max(activity_curve_diff, 0.1)

    seq_config = init_itemization_params().get('seq_config')
    threshold = config.get('remove_small_seq_config').get("threshold")

    threshold = threshold[0] if samples_per_hour == 2 else threshold[1]

    # calculate previously derived sequences of activity

    activity_curve_chunks = find_seq(activity_seq, activity_curve, activity_curve_derivative)

    label = activity_curve_chunks[:, seq_config.get('label')]
    length = activity_curve_chunks[:, seq_config.get('length')]
    derivative_strength = activity_curve_chunks[:, seq_config.get('deri_strength')]
    net_derivative = activity_curve_chunks[:, seq_config.get('net_deri')]
    start = activity_curve_chunks[:, seq_config.get('start')]
    end = activity_curve_chunks[:, seq_config.get('end')]

    weights = config.get('remove_small_seq_config').get("weightage")

    # filter out with sequences with lower length or score

    for index in range(len(length)):

        index_array = get_index_array(start[index], end[index], len(activity_curve))

        val = label[index]

        # filter positive sequence

        if val == 1 and length[index] < config.get('remove_small_seq_config').get("length_limit"):

            score = np.power(derivative_strength[index], 0.5)/2 * weights[0] + \
                    np.power(length[index], 0.5)/2 * weights[1] + \
                    net_derivative[index]/activity_curve_diff * weights[2]

            if score < threshold and net_derivative[index]/activity_curve_diff < \
                    config.get('remove_small_seq_config').get("derivative_threshold"):
                activity_seq[index_array] = 0

        # filter negative sequence

        if val == -1 and length[index] < config.get('remove_small_seq_config').get("length_limit"):

            score = np.power(derivative_strength[index], 0.5) / 2 * weights[0] + \
                    np.power(length[index], 0.5) / 2 * weights[1] - \
                    net_derivative[index] / activity_curve_diff * weights[2]

            if score < threshold and net_derivative[index] / activity_curve_diff > \
                    -config.get('remove_small_seq_config').get("derivative_threshold"):
                activity_seq[index_array] = 0

    t_activity_profile_end = datetime.now()

    logger.info("Post process to remove small sequences took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return activity_seq


def post_process_using_neighbour_seq(activity_seq, activity_curve, samples_per_hour, activity_curve_diff, config, logger):

    """
    Post process to modify sequences using neighbouring points

        Parameters:
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            activity_curve             (np.ndarray)     : living load activity profile
            samples_per_hour           (int)            : samples in an hour
            activity_curve_diff        (float)          : diff in max and min of activity curve
            config                     (dict)           : dict containing activity seq config values
            logger                     (logger)         : logger object

        Returns:
            activity_seq              (np.ndarray)     : labels of activity sequences of the user
    """

    t_activity_profile_start = datetime.now()

    seq_config = init_itemization_params().get('seq_config')

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    activity_curve_chunks = find_seq(activity_seq, activity_curve, activity_curve_derivative)

    label = activity_curve_chunks[:, seq_config.get('label')]
    start = activity_curve_chunks[:, seq_config.get('start')]
    end = activity_curve_chunks[:, seq_config.get('end')]
    count = activity_curve_chunks[:, seq_config.get('length')]
    max_derivative = activity_curve_chunks[:, seq_config.get('max_deri')]
    low_perc = activity_curve_chunks[:, seq_config.get('low_perc')]
    high_perc = activity_curve_chunks[:, seq_config.get('high_perc')]

    threshold = np.digitize([activity_curve_diff], np.arange(0.1, 1, 0.1))[0]

    threshold = config.get('neighbour_seq_config').get('threshold_array')[threshold]
    length_factor = config.get('neighbour_seq_config').get('length_factor')

    label_copy = copy.deepcopy(label)

    length = len(count)

    if np.min(activity_seq) == np.max(activity_seq):
        return activity_seq

    for index in range(length):

        index_array = get_index_array(index, index+2, length)

        if len(index_array) != 3:
            continue

        label_copy = filter_middle_constant_seq(label_copy, index_array, index, count, max_derivative, length_factor, threshold)

        # filter decreasing seq between two constant seq

        shorter_len_seq = count[(index+1) % length] <= samples_per_hour*length_factor
        lower_range_seq = (max_derivative[(index+1) % length] > -threshold) and (low_perc[(index+2) % length]-high_perc[index] > -(threshold-0.01))

        bool3 =  (shorter_len_seq and (label_copy[index_array] == [0, -1, 0]).all()) and lower_range_seq

        # filter increasing seq between two constant seq

        lower_range_seq = (max_derivative[(index+1) % length] < threshold) and (high_perc[(index+2) % length]-low_perc[index] < (threshold-0.01))

        bool4 =  (shorter_len_seq and (label_copy[index_array] == [0, 1, 0]).all()) and lower_range_seq

        label_copy[(index+1) % length] = 0 if (bool3 or bool4) else label_copy[(index+1) % length]

    activity_seq = update_activity_seq(start, end, label_copy, activity_seq)

    # final post processing step to filter out zigzag pattern (increase-decrease-increase-decrease)

    activity_seq = filter_zigzag_pattern(activity_seq, activity_curve, threshold, config, seq_config)

    t_activity_profile_end = datetime.now()

    logger.info("Post process using neighbour sequences took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return activity_seq


def filter_middle_constant_seq(label_copy, index_array, index, count, max_derivative, length_factor, threshold):

    """
    Filter patterns like 1, 0, 1 or -1, 0, -1 or -1, 0, 1 or 1, 0, -1

        Parameters:
            label_copy              (np.ndarray)    : copy of array of labels of activity seq
            index_array             (np.ndarray)    : list of tou indexes contained in the target sequence
            index                   (int)           : index of the target seq to be verified
            count                   (np.ndrray)     : array of length of all activity seq
            max_derivative          (int)           : array of derivative of all activity seq
            length_factor           (int)           : length threshold to be used for filtering
            threshold               (int)           : derivative threshold to be used for filtering

        Returns:
            label_copy              (np.ndarray)    : updated copy of array of labels of activity seq

    """

    # filter constant seq between two decreasing seq

    samples_per_hour = len(label_copy) / Cgbdisagg.HRS_IN_DAY

    length = len(count)

    bool1 = ((label_copy[index_array] == [-1, 0, -1]).all() or (label_copy[index_array] == [1, 0, 1]).all()) \
            and count[(index + 1) % length] <= samples_per_hour * length_factor

    label_copy[(index + 1) % length] = \
        (np.sign(max_derivative[(index + 1) % length]) if
         max_derivative[(index + 1) % length] >= (threshold - 0.03) else label_copy[index]) \
        if bool1 else label_copy[(index + 1) % length]

    # filter constant seq between decreasing-increasing seq

    bool2 = ((label_copy[index_array] == [-1, 0, 1]).all() or (label_copy[index_array] == [1, 0, -1]).all()) \
            and count[(index + 1) % length] <= samples_per_hour * length_factor

    label_copy[(index + 1) % length] = np.sign(max_derivative[(index + 1) % length] + 0.0000001) if bool2 else \
    label_copy[(index + 1) % length]

    return label_copy


def update_activity_seq(start, end, label, activity_seq):

    """
    Update activity seq based on post processing changes

        Parameters:
            start                     (np.ndarray)     : array of start index for which label has to updated
            end                       (np.ndarray)     : array of end index for which label has to updated
            label                     (int)            : final activity seq label to be assigned
            activity_seq              (np.ndarray)     : original labels of activity sequences of the user

        Returns:
            activity_seq              (np.ndarray)     : updated labels of activity sequences of the user

    """

    for index in range(len(start)):
        activity_seq = fill_array(activity_seq, int(start[index]), int(end[index]) + 1, label[index])

    return activity_seq


def filter_zigzag_pattern(activity_seq, activity_curve, threshold, config, seq_config):

    """
    final post processing step to filter out zigzag pattern (increase-decrease-increase-decrease)

        Parameters:
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            activity_curve             (np.ndarray)     : living load activity profile
            threshold                  (int)            : threshold to be used to determine whether the zigzag seq is active
            config                     (dict)           : dict containing activity seq config values
            seq_config                 (dict)           : dict containing seq columns labels information

        Returns:
            activity_seq              (np.ndarray)     : labels of activity sequences of the user

    """

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)
    samples_per_hour = int(len(activity_curve) / Cgbdisagg.HRS_IN_DAY)

    activity_curve_chunks = find_seq(activity_seq, activity_curve, activity_curve_derivative)

    label = activity_curve_chunks[:, seq_config.get('label')]
    start = activity_curve_chunks[:, seq_config.get('start')]
    end = activity_curve_chunks[:, seq_config.get('end')]
    count = activity_curve_chunks[:, seq_config.get('length')]
    max_derivative = activity_curve_chunks[:, seq_config.get('max_deri')]

    win_strt_idx = 1

    while win_strt_idx < len(count)-1:

        if label[win_strt_idx]:

            val = label[win_strt_idx]

            win_end_idx = win_strt_idx

            while (win_end_idx < len(count) - 1) and (label[win_end_idx] == val):

                val = -val
                win_end_idx = win_end_idx + 1

            # If length of zigzag pattern is greater than two

            if (win_end_idx-win_strt_idx > config.get('neighbour_seq_config').get('zigzag_pattern_limit')) and \
                    (np.abs(np.max(max_derivative[win_strt_idx: win_end_idx])) <
                     threshold+config.get('neighbour_seq_config').get('zigzag_threshold_increament')) and\
                    np.max(count[win_strt_idx: win_end_idx]) < samples_per_hour:

                # combine zigzag pattern into constant - increase - decrease sequence

                if np.abs(activity_curve[int(start[win_end_idx])] - activity_curve[int(start[win_strt_idx])]) < 0.01:
                    label[win_strt_idx: win_end_idx] = 0
                else:
                    label[win_strt_idx] = np.sign(activity_curve[int(start[win_end_idx])] - activity_curve[int(start[win_strt_idx])])

            win_strt_idx = win_strt_idx + 1

        else:
            win_strt_idx = win_strt_idx + 1

    for win_strt_idx in range(len(count)):
        activity_seq[int(start[win_strt_idx]): int(end[win_strt_idx] + 1)] = label[win_strt_idx]

    return activity_seq
