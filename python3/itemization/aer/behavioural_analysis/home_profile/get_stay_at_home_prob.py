
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Calculates count of home stayers
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.behavioural_analysis.home_profile.config.occupancy_profile_config import get_stay_at_home_config


def get_stay_at_home_prob(item_input_object, item_output_object, early_arrival):

    """
       Calculates count of home stayers

       Parameters:
            item_input_object             (dict)               : Dict containing all hybrid inputs
            item_output_object            (dict)               : Dict containing all hybrid outputs
            early_arrival                 (int)                : Calculated early arrival count

       Returns:
           early_arrivals                (int)                : Count of school going kids present in the house
           probability                   (float)              : Probability of school going kids present in the house
    """

    stay_at_home_prob = 0

    activity_curve = item_input_object.get("weekday_activity_curve")
    activity_seq = item_output_object.get("profile_attributes").get('activity_sequences')
    activity_segments = item_output_object.get("profile_attributes").get('activity_segments')
    active_hours = item_output_object.get("profile_attributes").get('active_hours')
    samples_per_hour = item_input_object.get("item_input_params").get('samples_per_hour')

    # Calculate activity curve derivative to be used to get the amplitude of change in activity

    mid_day_act_flag = 0

    multiple_mid_day_act_flag = 0

    #  Calculate mid day active hours, using activity curve, sampling rate, and early arrival count information

    activity_curve_range = np.percentile(activity_curve, 98) - np.percentile(activity_curve, 2)

    config = get_stay_at_home_config(early_arrival, samples_per_hour, activity_curve).get('stay_at_home_config')

    mid_day_hours = config.get('mid_day_hours')
    valid_act_segments = config.get('valid_act_segments')
    zero_mid_day_act_score = config.get('zero_mid_day_act_score')
    stay_at_home_prob_thres = config.get('stay_at_home_prob_thres')

    # Identify whether each activity segment is a potential mid day activity or not

    activity_segments = activity_segments[np.isin(activity_segments[:, 0], valid_act_segments)]

    for seg_idx in range(len(activity_segments)):
        stay_at_home_prob, multiple_mid_day_act_flag, mid_day_act_flag, break_flag = \
            calculate_mid_day_act_score(activity_segments, seg_idx, activity_curve, config, stay_at_home_prob,
                                        multiple_mid_day_act_flag, mid_day_act_flag, activity_curve_range)

        if break_flag:
            break

    # if no mid day activity is found, stay at home score is made 0
    if not mid_day_act_flag:
        stay_at_home_prob = min(stay_at_home_prob, zero_mid_day_act_score)

    stay_at_home_prob = np.round(stay_at_home_prob, 2)

    if np.all(activity_seq[mid_day_hours] == 0):
        stay_at_home_prob = min(stay_at_home_prob_thres[0] - 0.1, stay_at_home_prob)
        multiple_mid_day_act_flag = 0

    # number of stay at home occupants are estimated based on level of mid day activity score
    mid_day_act_flag = 0

    if stay_at_home_prob > stay_at_home_prob_thres[0]:
        mid_day_act_flag = 1
        if stay_at_home_prob >= stay_at_home_prob_thres[1]:
            multiple_mid_day_act_flag = 1

    mid_day_active_hours = np.multiply(activity_seq, active_hours)

    mid_day_active_hours[mid_day_active_hours <= 0] = 0
    mid_day_active_hours[mid_day_active_hours > 0] = 1

    mid_day_active_hours[np.abs(activity_curve - np.roll(activity_curve, 1)) < 0.02] = 0

    if np.sum(mid_day_active_hours[mid_day_hours]) > 2 * samples_per_hour and (not stay_at_home_prob > stay_at_home_prob_thres[1]):
        multiple_mid_day_act_flag = 1
        stay_at_home_prob = stay_at_home_prob_thres[1]

    multiple_mid_day_act_flag = multiple_mid_day_act_flag + mid_day_act_flag

    return multiple_mid_day_act_flag, stay_at_home_prob


def calculate_mid_day_act_score(activity_segments, seg_idx, activity_curve, config, stay_at_home_prob,
                                multiple_mid_day_act_flag, mid_day_act_flag, activity_curve_range):

    """
    Calculates probability of mid day activity

       Parameters:
           activity_segments             (np.ndarray)         : array containing information for individual segments
           seg_idx                       (int)                : index of current activity segment
           activity_curve                (np.ndarray)         : activity profile of the user
           config                        (dict)               : config dict
           stay_at_home_prob             (float)              : calculated probability of stay at home people
           multiple_mid_day_act_flag     (bool)               : calculated probability of higher range of mid day activity
           mid_day_act_flag              (bool)               : calculated probability of lower range of mid day activity
           activity_curve_range          (float)              : range of activity profile

       Returns:
           stay_at_home_prob             (float)              : calculated probability of stay at home people
           multiple_mid_day_act_flag     (bool)               : calculated probability of higher range of mid day activity
           mid_day_act_flag              (bool)               : calculated probability of lower range of mid day activity
    """

    samples_per_hour = int(len(activity_curve) / 24)

    break_flag = 0

    mid_day_hours = config.get('mid_day_hours')
    act_range_factor = config.get('threshold')
    act_len_score_offset = config.get('act_len_score_offset')

    temp_stay_at_home_score_thres = config.get('temp_stay_at_home_score_thres')
    min_act_prof_val = config.get('min_act_prof_val')
    temp_stay_at_home_score_offset = config.get('temp_stay_at_home_score_offset')
    temp_multi_stay_home_users_score_thres = config.get('temp_multi_stay_home_users_score_thres')
    multi_stay_home_range_score_thres = config.get('multi_stay_home_range_score_thres')
    multi_stay_home_len_score_thres = config.get('multi_stay_home_len_score_thres')
    zero_stay_at_home_score = config.get('zero_stay_at_home_score')

    check_3_thres = config.get('check3_thres')
    check_2_thres = config.get('check2_thres')
    stay_at_home_score_offset = config.get('stay_at_home_score_offset')

    start = activity_segments[seg_idx, 2] - int(samples_per_hour != 1)
    end = activity_segments[seg_idx, 3]

    index_array = get_index_array(start, end, len(activity_curve))
    index_array = np.intersect1d(index_array, mid_day_hours)

    if len(index_array) == 0:
        return stay_at_home_prob, multiple_mid_day_act_flag, mid_day_act_flag, break_flag

    # check if the current activity segment lies in mid days hours slot

    # this variable is to assign mid day activity score based on the range of activity

    check_based_on_act_range = np.abs(np.max(activity_curve[index_array]) - np.min(activity_curve[index_array])) * 0.1 / act_range_factor
    check_based_on_act_range = min(check_based_on_act_range, 0.2)

    act_len_factor = min(1 * samples_per_hour, len(index_array))
    act_len_factor = max(act_len_factor, 0.5 * samples_per_hour) / samples_per_hour

    act_len_factor = 2 * (samples_per_hour == 1) + act_len_factor * (samples_per_hour != 1)

    act_len_factor = act_len_factor + 4 * (activity_segments[seg_idx, 0] == 1)

    # this variable is to assign mid day activity score based on the length of activity segment

    check_based_on_hour_count = len(index_array) / samples_per_hour
    check_based_on_hour_count = ((np.exp(check_based_on_hour_count / act_len_factor) -
                                  np.exp(-check_based_on_hour_count / act_len_factor)) / (np.exp(-check_based_on_hour_count / act_len_factor) +
                                                                                          np.exp(check_based_on_hour_count / act_len_factor))) / 5 - act_len_score_offset

    activity_curve_range = max(activity_curve_range, min_act_prof_val)

    temp_stay_at_home_score = check_based_on_act_range + check_based_on_hour_count + temp_stay_at_home_score_offset

    # checking the possibility of a stay at home user based on parameters calculated for mid day activity

    stay_at_home_flag = (check_based_on_hour_count > check_3_thres[0]) or (check_based_on_hour_count > check_3_thres[1] and check_based_on_act_range > check_2_thres)

    if np.all(activity_curve[index_array] > (0.3 * activity_curve_range + np.min(activity_curve))):
        temp_stay_at_home_score = temp_stay_at_home_score + stay_at_home_score_offset

    if np.all(activity_curve[index_array] < (0.2 * activity_curve_range + np.min(activity_curve))):
        temp_stay_at_home_score = temp_stay_at_home_score - stay_at_home_score_offset

    # checking the value of stay at home score

    if temp_stay_at_home_score > temp_stay_at_home_score_thres and stay_at_home_flag:

        mid_day_act_flag = 1

        length_threshold = 10 * (activity_segments[seg_idx, 0] == 1) + 2 * (activity_segments[seg_idx, 0] != 1)

        # this variable is to assign high mid day activity score based on the length of activity segment

        hours_count = int(len(index_array) / samples_per_hour)
        check_based_on_mid_day_act = (np.exp(hours_count / length_threshold) - np.exp(-hours_count / length_threshold)) / \
                                     (np.exp(-hours_count / length_threshold) + np.exp(hours_count / length_threshold)) / 5

        # this variable is to assign high mid day activity score based on the range of activity

        check_based_on_mid_day_act_range = (np.max(activity_curve[index_array]) - np.min(activity_curve[index_array])) / (5 * activity_curve_range)
        check_based_on_mid_day_act_range = min(check_based_on_mid_day_act_range, min_act_prof_val)

        temp_multi_stay_home_users_score = temp_stay_at_home_score + check_based_on_mid_day_act + check_based_on_mid_day_act_range

        # based on the actvity profile, higher amount of daytime activity is detected

        if temp_multi_stay_home_users_score >= temp_multi_stay_home_users_score_thres and \
                check_based_on_mid_day_act_range >= multi_stay_home_range_score_thres and \
                check_based_on_mid_day_act >= multi_stay_home_len_score_thres:

            # if home stay occupants are detected, loop is exited

            multiple_mid_day_act_flag = 1

            if activity_segments[seg_idx, 0] == 1:
                temp_multi_stay_home_users_score = temp_multi_stay_home_users_score - 0.1

            stay_at_home_prob = temp_multi_stay_home_users_score
            break_flag = 1

        else:
            if activity_segments[seg_idx, 0] == 1:
                temp_stay_at_home_score = temp_stay_at_home_score - 0.1
            stay_at_home_prob = max(stay_at_home_prob, temp_stay_at_home_score)
            stay_at_home_prob = min(stay_at_home_prob, zero_stay_at_home_score)

    elif not stay_at_home_flag:
        stay_at_home_prob = max(stay_at_home_prob, temp_stay_at_home_score)

    return stay_at_home_prob, multiple_mid_day_act_flag, mid_day_act_flag, break_flag
