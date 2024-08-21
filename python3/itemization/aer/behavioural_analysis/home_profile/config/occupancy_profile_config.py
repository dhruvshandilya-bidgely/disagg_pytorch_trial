
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Master file for calculation of occupancy profile and occupants count
"""

import numpy as np


def get_occupancy_profile_config(samples_per_hour=1, home_stayer_count=1):

    """
       Initialize config dict containing config values for occupancy profile calculation

       Parameters:
           samples_per_hour             (int)               : samples in an hour

       Returns:
           occupancy_profile_config     (dict)              : prepared config dictionary
    """

    occupancy_profile_config = dict()

    general_config = dict({

        "occupants_type_count": 3,
        "max_occupants_type_count": 2,
        "activity_curve_range_limit": [0.05, 0.08],
        "low_range_prob": [0.4, 0.4, 0.7],
        "low_range_count": [0, 0, 2],
        "office_goer_index": 0,
        "early_arrival_index": 1,
        "stay_at_home_index": 2,
        "default_occupants_count": 2,

    })

    # "occupants_type_count": number of occupants category
    # "max_occupants_type_count": max number of users in a particular occupants count ,
    # "activity_curve_range_limit": thresholds based on activity curve ranges
    # "low_range_prob": probability threshold for users with lower activity curve range
    # "low_range_count": occupants counts for users with lower activity curve range
    # "office_goer_index": index of office goers in occupants array
    # "early_arrival_index": index of early arrivals in occupants array
    # "stay_at_home_index": index of home stayers in occupants array
    # "default_occupants_count": default occupants count

    occupancy_profile_config.update({
        "general_config": general_config
    })

    if samples_per_hour > 1:
        evening_hours = np.arange(16.5 * samples_per_hour, 22 * samples_per_hour).astype(int)
    else:
        evening_hours = np.arange(17 * samples_per_hour, 22 * samples_per_hour).astype(int)

    len_threshold = 2
    diff_threshold = 4

    if home_stayer_count == 1:
        len_threshold = 3
        diff_threshold = 5

    if home_stayer_count == 2:
        len_threshold = 5
        diff_threshold = 7

    office_goer_config = dict({

        'evening_hours': evening_hours,
        'len_threshold' : len_threshold,
        'diff_threshold' : diff_threshold,
        'min_office_goer_score_weights' : [0.5, 1.5],
        'eve_act_levels_thres' : [0.7, 0.15],
        'valid_segments' : [2, 3, 4],
        'late_night_hours' : np.arange(1 * samples_per_hour, 4 * samples_per_hour + 1),
        'morning_end_time' : 11 * samples_per_hour,
        'max_score_for_act_rise' : 0.25,
        'act_rise_score_thres' : 0.025,
        'stat_at_home_thres' : 0.5,
        'multi_office_goers_score_thres' : 0.3,
        'office_score_thres' : [0.5, 0.35],
        'office_score_weights' : [0.3, 0.7],
        'office_score_offset' : 0.35,
        'zero_office_score' : 0.69,
        'office_goer_prob_thres' : [0.5, 0.7],
        'morning_hours' : np.arange(5.5 * samples_per_hour, 10 * samples_per_hour).astype(int),
        'max_morn_act_der_thres' : 0.03,
        'morn_act_score_thres' : 0.3,
        'eve_time_slot_1': np.arange(17 * samples_per_hour, 18 * samples_per_hour + 1).astype(int),
        'eve_time_slot_2': np.arange(18 * samples_per_hour, 19 * samples_per_hour + 1).astype(int),
        'eve_time_slot_0': np.arange(16 * samples_per_hour, 17 * samples_per_hour + 1).astype(int),
        'eve_time_slot_3': np.arange(19 * samples_per_hour, 20 * samples_per_hour + 1).astype(int),
        'thres_for_inc_diff': 0.05,
        'office_goer_offset_for_inc_diff': [0.05, 0.1],
        'office_going_prob_offset': 0.5,
        'min_overlapping_frac': 0.3,
    })

    occupancy_profile_config.update({
        "office_goer_config": office_goer_config
    })

    stay_at_home_config = dict({

    })

    occupancy_profile_config.update({
        "stay_at_home_config": stay_at_home_config
    })

    early_arrivals_config = dict({
        'morning_hours': np.arange(6 * samples_per_hour, 7.5 * samples_per_hour + 1).astype(int),
        'noon_hours': np.arange(14 * samples_per_hour, 16 * samples_per_hour + 1),
        'early_eve_hours': np.arange(5 * samples_per_hour, 6.5 * samples_per_hour + 1).astype(int),
        'morn_dip_thres': 0.03,
        "morn_rise_thres": 0.045,
        "act_thres": 0.2,
        'deri_weight': 0.67,
        'offset': 0.47
    })

    occupancy_profile_config.update({
        "early_arrivals_config": early_arrivals_config
    })

    user_attributes = dict({

        'start_list': np.array([0, 6, 7, 11, 14, 17, 18, 21]) * samples_per_hour,
        'end_list': np.array([3, 7, 10, 13, 16, 21, 20, 23]) * samples_per_hour,
        'morning_hours': np.arange(5 * samples_per_hour, 12 * samples_per_hour).astype(int),
        'sleeping_hours': np.arange(20 * samples_per_hour, 26 * samples_per_hour).astype(int) % (samples_per_hour * 24),
        'lunch_probable_hours': np.arange(11.5 * samples_per_hour, 14 * samples_per_hour + 1).astype(int),
        'default_sleep_time': np.arange(19 * samples_per_hour, 27 * samples_per_hour + 1),
        'default_wake_time': np.arange(5 * samples_per_hour, 12 * samples_per_hour + 1),
        'dinner_hours': np.arange(18 * samples_per_hour, 20 * samples_per_hour + 1),
        'overlap_thres': 0.5 * samples_per_hour,
        'lunch_hours': np.arange(12 * samples_per_hour, 14 * samples_per_hour + 1),
        'default_score': 0.01,
        'default_lunch_slot': 12 * samples_per_hour,

    })

    occupancy_profile_config.update({
        "user_attributes": user_attributes
    })

    return occupancy_profile_config


def get_stay_at_home_config(early_arrival, samples_per_hour, activity_curve):

    """
       Initialize config dict containing config values for occupancy profile calculation

       Parameters:
           samples_per_hour             (int)               : samples in an hour

       Returns:
           occupancy_profile_config     (dict)              : prepared config dictionary
    """

    activity_curve_range = np.percentile(activity_curve, 98) - np.percentile(activity_curve, 2)

    increasing_flag = np.sum(activity_curve[np.arange(11 * samples_per_hour, 12 * samples_per_hour + 1).astype(int)]) > \
                      np.sum(activity_curve[np.arange(10 * samples_per_hour, 11 * samples_per_hour + 1).astype(int)])

    if increasing_flag:
        mid_day_hours = np.arange(10 * samples_per_hour, 15.5 * samples_per_hour + 1).astype(int)

        if early_arrival:
            mid_day_hours = np.arange(10 * samples_per_hour, 14 * samples_per_hour + 1).astype(int)

    else:
        mid_day_hours = np.arange(11.5 * samples_per_hour, 15.5 * samples_per_hour + 1).astype(int)

        if early_arrival:
            mid_day_hours = np.arange(11.5 * samples_per_hour, 14 * samples_per_hour + 1).astype(int)

    if samples_per_hour == 1:
        mid_day_hours = mid_day_hours[1:]

    threshold = 0.07

    if activity_curve_range < 0.35:
        threshold = 0.065

    if activity_curve_range < 0.3:
        threshold = 0.06

    if activity_curve_range < 0.25:
        threshold = 0.055

    if activity_curve_range < 0.2:
        threshold = 0.055

    if activity_curve_range <= 0.15:
        threshold = 0.04

    if samples_per_hour == 1:
        threshold = threshold - 0.05

    act_len_score_offset = 0.05

    if samples_per_hour == 1:
        act_len_score_offset = 0.1

    occupancy_profile_config = dict()

    stay_at_home_config = dict({
        'mid_day_hours': mid_day_hours,
        "threshold": threshold,
        "act_len_score_offset": act_len_score_offset,
        'temp_stay_at_home_score_thres': 0.5,
        'valid_act_segments': [1, 2, 3, 4, 5],
        'min_act_prof_val': 0.15,
        'temp_stay_at_home_score_offset': 0.3,
        'temp_multi_stay_home_users_score_thres': 0.7,
        'multi_stay_home_range_score_thres': 0.04,
        'multi_stay_home_len_score_thres': 0.08,
        'zero_stay_at_home_score': 0.69,
        'zero_mid_day_act_score': 0.49,
        'stay_at_home_prob_thres': [0.5, 0.7],
        'increasing_flag': increasing_flag,
        'check3_thres': [0.1, 0.07],
        'check2_thres': 0.15,
        'stay_at_home_score_offset': 0.05

    })

    occupancy_profile_config.update({
        "stay_at_home_config": stay_at_home_config
    })

    return occupancy_profile_config
