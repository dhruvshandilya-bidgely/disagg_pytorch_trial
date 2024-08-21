"""
Author - Nisha Agarwal
Date - 8th Oct 20
Post processing steps for extension of inactive hours
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.init_itemization_config import init_itemization_params


def fill_active_hours_array(active_hours, target_hours, target_hours_seq, config, activity_seq, samples_per_hour):

    """
    Fill active/inactive hours using activity sequence information

        Parameters:
            active_hours               (np.ndarray)     : active/nonactive mask array
            target_hours               (np.ndarray)     : target hours for filling active/inactive hours
            target_hours_seq           (np.ndarray)     : target activity sequence for filling active/inactive hours
            config                     (dict)           : dict containing active hours config values
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            samples_per_hour           (int)            : samples in an hour

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    # This function was added to handle cases like slow activity increase in morning and tag them as active seq

    morning_active_hours = active_hours[target_hours]

    seq_config = init_itemization_params().get("seq_config")

    min_columns_req_in_seq_arr = 4

    if not np.shape(target_hours_seq)[1] <= min_columns_req_in_seq_arr:

        if np.all(activity_seq[target_hours] == 1):
            active_hours[target_hours] = 0

        else:
            for index in range(len(target_hours_seq)):

                tag_as_active_segment = target_hours_seq[index, seq_config.get('label')] == 1 and \
                                        (target_hours_seq[index, seq_config.get('length')] > samples_per_hour / 2 or
                                         target_hours_seq[index, seq_config.get('max_deri')] >= config.get('derivative_limit'))

                if tag_as_active_segment:

                    # Tag the target time band as active sequence

                    index_array = get_index_array(target_hours_seq[index, seq_config.get('start')],
                                                  target_hours_seq[index, seq_config.get('end')],
                                                  len(morning_active_hours))
                    morning_active_hours[index_array] = 0

            active_hours[target_hours] = morning_active_hours

    return active_hours


def extend_inactive_segments(active_hours, sleeping_hours, activity_curve, activity_seq, max_length, samples_per_hour, config, flag=False):

    """
    Extend inactive hours for a user

        Parameters:
            active_hours               (np.ndarray)     : active/nonactive mask array
            sleeping_hours             (np.ndarray)     : sleeping hours
            activity_curve             (np.ndarray)     : living load activity profile
            activity_seq               (np.ndarray)     : labels of activity sequences of the user
            max_length                 (int)            : max length of the inactive hours
            samples_per_hour           (int)            : samples in an hour
            config                     (dict)           : dict containing active hours config values

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    count = max_length

    start = sleeping_hours[-1] + 1
    end = sleeping_hours[0] - 1

    # This functions extends the inactive segment of the user
    # this extension can be in either direction depending upon the derivative values
    # This extension is continue uptill a max inactive seq length threshold is met

    # threshold_for_inc_seq - indicates threshold for positive sequences (compares derivative of consecutive points)
    # net_threshold_for_inc_seq - indicates threshold for positive sequences (compares derivative of
    # current point to the starting point of extension)

    # threshold_for_dec_seq - indicates threshold for negative sequences (compares derivative of consecutive points)
    # net_threshold_for_dec_seq - indicates threshold for negative sequences (compares derivative of
    # current point to the starting point of extension)

    # Initializing threshold array, since the threshold is different at different time of the day

    inc_seq_ext_thres_factor = config.get('thres_for_inc_seq')
    dec_seq_ext_thres_factor = config.get('thres_for_dec_seq')

    inc_seq_ext_net_thres_factor = config.get('net_thres_for_inc_seq')
    dec_seq_ext_net_thres_factor = config.get('net_thres_for_dec_seq')

    threshold_for_inc_seq = np.ones(len(activity_curve)) * inc_seq_ext_thres_factor
    threshold_for_dec_seq = np.ones(len(activity_curve)) * dec_seq_ext_thres_factor

    net_threshold_for_inc_seq = np.ones(len(activity_curve)) * inc_seq_ext_net_thres_factor
    net_threshold_for_dec_seq = np.ones(len(activity_curve)) * dec_seq_ext_net_thres_factor

    morning_hours = config.get('morning_hours')
    night_hours = config.get('night_hours')

    thres_for_morn_hours = config.get('thres_for_morn_hours')
    thres_for_night_hours = config.get('thres_for_night_hours')

    threshold_for_inc_seq[morning_hours] = threshold_for_inc_seq[morning_hours] - thres_for_morn_hours
    net_threshold_for_inc_seq[morning_hours] = net_threshold_for_inc_seq[morning_hours] - thres_for_morn_hours

    threshold_for_inc_seq[night_hours] = threshold_for_inc_seq[night_hours] - thres_for_night_hours
    net_threshold_for_inc_seq[night_hours] = net_threshold_for_inc_seq[night_hours] - thres_for_night_hours

    threshold_for_dec_seq[night_hours] = threshold_for_dec_seq[night_hours] - thres_for_night_hours
    net_threshold_for_dec_seq[night_hours] = net_threshold_for_dec_seq[night_hours] - thres_for_night_hours

    late_evening_hours = config.get('late_evening_hours')

    diff = np.max(activity_curve) - np.min(activity_curve)

    threshold_for_inc_seq[late_evening_hours] = 0
    threshold_for_dec_seq[late_evening_hours] = 0
    net_threshold_for_inc_seq[late_evening_hours] = 0
    net_threshold_for_dec_seq[late_evening_hours] = 0

    constant_threshold = np.ones(len(activity_curve)) * 0.2 * diff
    constant_threshold[late_evening_hours] = -1

    # max limit - represents the time till which the inactive segment is to be extended
    limit = samples_per_hour * (10 - int(samples_per_hour/2))

    threshold_params = [threshold_for_inc_seq, net_threshold_for_inc_seq, threshold_for_dec_seq, net_threshold_for_dec_seq, constant_threshold]

    while flag and count < max(limit, max_length):

        start_check = 0
        end_check = 0

        # checking whether we can extend the inactive segment from right or left direction

        start_bool = get_segments_start_inactivity_check(active_hours, activity_seq, activity_curve, start,
                                                         sleeping_hours, threshold_params)

        end_bool = get_segments_end_inactivity_check(active_hours, activity_seq, activity_curve, end,
                                                     sleeping_hours, threshold_params)

        # If either of the two sides extension check is successful, the active hours array is updated

        active_hours, count, start, end, start_check, end_check = \
            update_start_end(start_bool, end_bool, active_hours, count, start, end, start_check, end_check)

        # If both side extension checks fail, the loop breaks

        if not start_check and not end_check:
            break

    # Extended inactive timestamps are tagged as -3

    active_hours[active_hours == -3] = 1

    morning_start_hour = 4
    morning_end_hour = 9.5

    morning_hours = np.arange(int(morning_start_hour * samples_per_hour), int(morning_end_hour * samples_per_hour))

    morning_seq = find_seq(activity_seq[morning_hours], activity_curve[morning_hours], np.zeros(len(activity_curve[morning_hours])))

    active_hours = fill_active_hours_array(active_hours, morning_hours, morning_seq, config, activity_seq, samples_per_hour)

    return active_hours


def update_start_end(start_bool, end_bool, active_hours, count, start, end, start_check, end_check):

    """
    Update start and end of inactive segments after extension

        Parameters:
            start_bool                 (int)            : to check whether we can extend the seq on
                                                          the direction of start of original inactive seq
            end_bool                   (int)            : to check whether we can extend the seq on
                                                          the direction of end of original inactive seq
            active_hours               (np.ndarray)     : active/nonactive mask array
            count                      (int)            : length of inactive segment
            start                      (int)            : start index of inactive segment
            end                        (int)            : end index of inactive segment
            start_check                (int)            : bool to track that the extension is preformed in
                                                          the start index direction  index of inactive segment
            end_check                  (int)            : bool to track that the extension is preformed in
                                                          the end index direction  index of inactive segment

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    length = len(active_hours)

    if start < end:

        if start_bool:
            # extension is preformed in the start index direction  index of inactive segment
            active_hours[start % length] = -3
            start = start + 1
            count = count + 1
            start_check = 1

        elif end_bool:
            # extension is preformed in the end index direction  index of inactive segment
            active_hours[end % length] = -3
            end = end - 1
            count = count + 1
            end_check = 1
    else:

        if end_bool:
            # extension is preformed in the end index direction  index of inactive segment
            active_hours[end % length] = -3
            end = end - 1
            count = count + 1
            end_check = 1

        elif start_bool:
            # extension is preformed in the start index direction  index of inactive segment
            active_hours[start % length] = -3
            start = start + 1
            count = count + 1
            start_check = 1

    return active_hours, count, start, end, start_check, end_check


def get_segments_start_inactivity_check(active_hours, activity_seq, activity_curve, start, sleeping_hours, threshold_params):

    """
    Check inactivity at start of an activity segment

        Parameters:
            active_hours               (np.ndarray)     : active/nonactive mask array
            activity_seq               (np.ndarray)     : activity seq of the user
            activity_curve             (np.ndarray)     : activity curve of the user
            start                      (int)            : start index
            sleeping_hours             (np.ndarray)     : sleeping hours of the user
            threshold_params           (list)           : list of threshold parameters to be used

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    derivative = activity_curve - np.roll(activity_curve, 1)

    threshold_for_inc_seq = threshold_params[0]
    net_threshold_for_inc_seq = threshold_params[1]
    threshold_for_dec_seq = threshold_params[2]
    net_threshold_for_dec_seq = threshold_params[3]
    constant_threshold = threshold_params[4]

    length = len(active_hours)

    samples_per_hour = int(len(active_hours) / Cgbdisagg.HRS_IN_DAY)

    general_sleep_hours = np.arange(len(active_hours) - 2 * samples_per_hour, len(active_hours) + 10 * samples_per_hour) % len(active_hours)

    net_start = sleeping_hours[-1]

    # Boolean to check whether the start index lies in the expected sleeping hours window

    bool1 = active_hours[start % length] == 1 and start % length in general_sleep_hours

    # Boolean to check whether activity seq is constant and net derivative is lower than threshold

    bool2 = activity_seq[start % length] == 0 and \
            np.abs(derivative[start % length]) < constant_threshold[start % length] and start % length in general_sleep_hours

    # Boolean to check whether activity seq is increasing and net derivative is lower than threshold

    bool3 = activity_seq[start % length] == 1 and \
            np.abs(activity_curve[start % length] - activity_curve[(start - 1) % length]) < threshold_for_inc_seq[start % length] and \
            np.abs(activity_curve[net_start % length] - activity_curve[start % length]) < net_threshold_for_inc_seq[start % length] and \
            start % length in general_sleep_hours

    # Boolean to check whether activity seq is  decreasing and net derivative is lower than threshold

    bool4 = activity_seq[start % length] == -1 and \
            np.abs(activity_curve[start % length] - activity_curve[(start - 1) % length]) < threshold_for_dec_seq[start % length] and \
            np.abs(activity_curve[net_start % length] - activity_curve[start % length]) < net_threshold_for_dec_seq[start % length] and \
            start % length in general_sleep_hours

    return bool1 or bool2 or bool3 or bool4


def get_segments_end_inactivity_check(active_hours, activity_seq, activity_curve, end, sleeping_hours, threshold_params):

    """
    Check inactivity at end of an activity segment

        Parameters:
            active_hours               (np.ndarray)     : active/nonactive mask array
            activity_seq               (np.ndarray)     : activity seq of the user
            activity_curve             (np.ndarray)     : activity curve of the user
            end                        (int)            : end index
            sleeping_hours             (np.ndarray)     : sleeping hours of the user
            threshold_params           (list)           : list of threshold parameters to be used

        Returns:
            active_hours               (np.ndarray)     : Updated active/nonactive mask array
    """

    derivative = activity_curve - np.roll(activity_curve, 1)

    threshold_for_inc_seq = threshold_params[0]
    net_threshold_for_inc_seq = threshold_params[1]
    threshold_for_dec_seq = threshold_params[2]
    net_threshold_for_dec_seq = threshold_params[3]
    constant_threshold = threshold_params[4]

    length = len(active_hours)

    samples_per_hour = int(len(active_hours) / Cgbdisagg.HRS_IN_DAY)

    general_sleep_hours = np.arange(len(active_hours) - 2 * samples_per_hour,
                                    len(active_hours) + 10 * samples_per_hour) % len(active_hours)

    net_end = sleeping_hours[0]

    # Boolean to check whether the end index lies in the expected sleeping hours window

    bool1 = active_hours[end % length] == 1 and end % length in general_sleep_hours

    # Boolean to check whether activity seq is constant and net derivative is lower than threshold

    bool2 = activity_seq[end % length] == 0 and np.abs(derivative[end % length]) < constant_threshold[end % length] and \
            end % length in general_sleep_hours

    # Boolean to check whether activity seq is increasing and net derivative is lower than threshold

    bool3 = activity_seq[end % length] == 1 and \
            np.abs(activity_curve[end % length] - activity_curve[(end + 1) % length]) < threshold_for_inc_seq[end % length] and \
            np.abs(activity_curve[net_end % length] - activity_curve[end % length]) < net_threshold_for_inc_seq[end % length] and \
            end % length in general_sleep_hours

    # Boolean to check whether activity seq is decreasing and net derivative is lower than threshold

    bool4 = activity_seq[end % length] == -1 and \
            np.abs(activity_curve[end % length] - activity_curve[(end + 1) % length]) < threshold_for_dec_seq[end % length] and \
            np.abs(activity_curve[net_end % length] - activity_curve[end % length]) < net_threshold_for_dec_seq[end % length] and \
            end % length in general_sleep_hours

    return bool1 or bool2 or bool3 or bool4
