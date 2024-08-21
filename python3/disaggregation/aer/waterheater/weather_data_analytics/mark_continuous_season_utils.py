"""
Author: Sahana M
Created: 17-Sep-2021
Utility functions to help in marking continuous seasons
"""

# Import python packages

import copy
import numpy as np


def check_btwn_trn_smr_and_smr(pre_seq, post_seq, seq_arr, idx):
    """
    Check between transition summer and summer
    Args:
        pre_seq         (np.ndarray)        : Previous sequence array
        post_seq        (np.ndarray)        : Future sequence array
        seq_arr         (np.ndarray)        : Overall season sequence array
        idx             (int)               : Index to update
    Returns:
        seq_arr         (np.ndarray)        : Overall season sequence array
    """

    if (pre_seq[0] == 0.5 and post_seq[0] == 1) or (post_seq[0] == 0.5 and pre_seq[0] == 1):
        seq_arr[idx, 0] = 0.5

    return seq_arr


def check_btwn_trn_wtr_and_wtr(pre_seq, post_seq, seq_arr, idx):
    """
    Check between transition summer and summer
    Args:
        pre_seq         (np.ndarray)        : Previous sequence array
        post_seq        (np.ndarray)        : Future sequence array
        seq_arr         (np.ndarray)        : Overall season sequence array
        idx             (int)               : Index to update
    Returns:
        seq_arr         (np.ndarray)        : Overall season sequence array
    """
    if (pre_seq[0] == -0.5 and post_seq[0] == -1) or (post_seq[0] == -0.5 and pre_seq[0] == -1):
        seq_arr[idx, 0] = -0.5

    return seq_arr


def update_seasons(pre_seq, post_seq, curr_seq, seq_arr, idx, smooth_avg_curr, max_winter_temp, max_tr_temp, tr_gap):
    """
    Update season labels
    Args:
        pre_seq         (np.ndarray)        : Previous sequence array
        post_seq        (np.ndarray)        : Future sequence array
        curr_seq        (np.ndarray)        : Current sequence array
        seq_arr         (np.ndarray)        : Overall season sequence array
        idx             (int)               : Index to update
        smooth_avg_curr (np.ndarray)        : Smoothened average array
        max_winter_temp (float)             : Max winter temperature
        max_tr_temp     (float)             : Max transition temperature
        tr_gap          (int)               : Transition gap
    Returns:
        seq_arr         (np.ndarray)        : Overall season sequence array
    """

    # Anything between transition summer and summer is transition summer

    seq_arr = check_btwn_trn_smr_and_smr(pre_seq, post_seq, seq_arr, idx)

    # Anything between transition winter and winter is transition winter

    seq_arr = check_btwn_trn_wtr_and_wtr(pre_seq, post_seq, seq_arr, idx)

    # Anything except summer between 2 transition summers if qualifies is transition summer

    if (pre_seq[0] == 0.5 and post_seq[0] == 0.5 and not (curr_seq[0] == 1)) and \
            (smooth_avg_curr >= max_tr_temp - tr_gap):
        seq_arr[idx, 0] = 0.5

    # Anything except winter between 2 transition winters if qualifies is transition winter

    if (pre_seq[0] == -0.5 and post_seq[0] == -0.5 and not (curr_seq[0] == -1)) and \
            (smooth_avg_curr <= max_winter_temp + tr_gap):
        seq_arr[idx, 0] = -0.5

    # Transition winter between 2 transition seasons if qualifies mark as transition

    if (pre_seq[0] == 0 and post_seq[0] == 0 and curr_seq[0] == -0.5) and \
            (smooth_avg_curr >= max_winter_temp + tr_gap):
        seq_arr[idx, 0] = 0

    # Transition summer between 2 transition seasons if qualifies mark as transition

    if (pre_seq[0] == 0 and post_seq[0] == 0 and curr_seq[0] == 0.5) and \
            (smooth_avg_curr <= max_tr_temp - tr_gap):
        seq_arr[idx, 0] = 0

    return seq_arr


def get_all_seasons_label(model_info_dict, limit_temperatures, bonus_dict, daily_avg_dict, s_label, max_tr_temp,
                          max_winter_temp):
    """

    Args:
        model_info_dict          (dict)          : Dictionary containing all information about the models
        limit_temperatures       (dict)          : Dictionary containing limit temperatures
        bonus_dict               (dict)          : Dictionary containing season bonuses
        daily_avg_dict           (dict)          : Dictionary containing daily average
        s_label                  (np.ndarray)    : Season labels
        max_tr_temp              (float)         : Max transition temperature
        max_winter_temp          (float)         : Max winter temperature
    Returns:
        s_label                  (np.ndarray)    : Season label
        max_tr_temp              (float)         : Max transition temperature
        max_winter_temp          (float)         : Max winter temperature
    """

    winter_bonus = bonus_dict.get('winter_bonus')
    summer_bonus = bonus_dict.get('summer_bonus')
    daily_avg = daily_avg_dict.get('daily_avg')
    daily_day_avg = daily_avg_dict.get('daily_day_avg')
    daily_night_avg = daily_avg_dict.get('daily_night_avg')
    lim_tr_temp = limit_temperatures.get('lim_tr_temp')
    lim_winter_temp = limit_temperatures.get('lim_winter_temp')
    is_longer_summer = model_info_dict.get('is_longer_summer')
    is_longer_winter = model_info_dict.get('is_longer_winter')
    season_switch_thr = model_info_dict.get('season_switch_thr')
    season_switch_thr_d = model_info_dict.get('season_switch_thr_d')
    season_switch_thr_n = model_info_dict.get('season_switch_thr_n')

    # We have all 3 seasons and 2 boundaries Or we have 2 seasons without transition

    if is_longer_winter and is_longer_summer:

        max_tr_temp = max(season_switch_thr_d[-1], lim_tr_temp) - summer_bonus
        max_winter_temp = min(season_switch_thr_n[0], lim_winter_temp) + winter_bonus

        s_label[daily_day_avg > max_tr_temp] = 1
        s_label[daily_night_avg <= max_winter_temp] = -1
        s_label[np.logical_and(daily_day_avg <= max_tr_temp, daily_night_avg > max_winter_temp)] = 0

    elif is_longer_winter:

        max_tr_temp = max(season_switch_thr[-1], lim_tr_temp) - summer_bonus
        max_winter_temp = min(season_switch_thr_n[0], lim_winter_temp) + winter_bonus

        s_label[daily_avg > max_tr_temp] = 1
        s_label[daily_night_avg <= max_winter_temp] = -1
        s_label[np.logical_and(daily_avg <= max_tr_temp, daily_night_avg > max_winter_temp)] = 0

    elif is_longer_summer:

        max_tr_temp = max(season_switch_thr_d[-1], lim_tr_temp) - summer_bonus
        max_winter_temp = min(season_switch_thr[0], lim_winter_temp) + winter_bonus

        s_label[daily_day_avg > max_tr_temp] = 1
        s_label[daily_avg <= max_winter_temp] = -1
        s_label[np.logical_and(daily_day_avg <= max_tr_temp, daily_avg > max_winter_temp)] = 0

    else:

        max_tr_temp = max(season_switch_thr[-1], lim_tr_temp) - summer_bonus
        max_winter_temp = min(season_switch_thr[0], lim_winter_temp) + winter_bonus

        s_label[daily_avg > max_tr_temp] = 1
        s_label[daily_avg <= max_winter_temp] = -1
        s_label[np.logical_and(daily_avg <= max_tr_temp, daily_avg > max_winter_temp)] = 0

    return s_label, max_tr_temp, max_winter_temp


def get_s_label_without_wtr(model_info_dict, mark_season_config, limit_temperatures, daily_avg_dict, s_label, max_tr_temp,
                            max_winter_temp):
    """
    Marking seasons when there are 2 seasons without winter
    Args:
        model_info_dict         (dict)          : Dictionary containing all information about the models
        mark_season_config      (dict)          : Dictionary containing all config used to mark seasons
        limit_temperatures      (dict)          : Dictionary containing limit temperatures
        daily_avg_dict          (dict)          : Dictionary containing daily average
        s_label                 (np.ndarray)    : Season label
        max_tr_temp              (float)         : Max transition temperature
        max_winter_temp          (float)         : Max winter temperature
    Returns:
        s_label                  (np.ndarray)    : Season label
        max_tr_temp              (float)         : Max transition temperature
        max_winter_temp          (float)         : Max winter temperature
    """

    daily_avg = daily_avg_dict.get('daily_avg')
    lim_tr_temp = limit_temperatures.get('lim_tr_temp')
    daily_day_avg = daily_avg_dict.get('daily_day_avg')
    is_longer_summer = model_info_dict.get('is_longer_summer')
    season_switch_thr = model_info_dict.get('season_switch_thr')
    season_switch_thr_d = model_info_dict.get('season_switch_thr_d')
    mark_prelim_seasons_config = mark_season_config.get('mark_prelim_season')
    obv_winter_thr = mark_prelim_seasons_config.get('obv_winter_thr')
    min_season_length = mark_prelim_seasons_config.get('min_season_length')

    if is_longer_summer:
        max_tr_temp = max(season_switch_thr_d[-1], lim_tr_temp)

        s_label[daily_day_avg <= max_tr_temp] = 0
        s_label[daily_day_avg > max_tr_temp] = 1

    else:
        max_tr_temp = max(season_switch_thr[-1], lim_tr_temp)

        s_label[daily_avg <= max_tr_temp] = 0
        s_label[daily_avg > max_tr_temp] = 1

    if np.sum(daily_avg <= obv_winter_thr) >= min_season_length:
        max_winter_temp = obv_winter_thr
        s_label[daily_avg <= obv_winter_thr] = -1

    return s_label, max_tr_temp, max_winter_temp


def get_s_label_without_smr(model_info_dict, mark_season_config, limit_temperatures, daily_avg_dict, s_label, max_tr_temp,
                            max_winter_temp):
    """
    Marking seasons when there are 2 seasons without summer
    Args:
        model_info_dict         (dict)          : Dictionary containing all information about the models
        mark_season_config      (dict)          : Dictionary containing all config used to mark seasons
        limit_temperatures      (dict)          : Dictionary containing limit temperatures
        daily_avg_dict          (dict)          : Dictionary containing daily average
        s_label                 (np.ndarray)    : Season label
        max_tr_temp              (float)         : Max transition temperature
        max_winter_temp          (float)         : Max winter temperature
    Returns:
        s_label                  (np.ndarray)    : Season label
        max_tr_temp              (float)         : Max transition temperature
        max_winter_temp          (float)         : Max winter temperature
    """

    daily_avg = daily_avg_dict.get('daily_avg')
    daily_night_avg = daily_avg_dict.get('daily_night_avg')
    is_longer_winter = model_info_dict.get('is_longer_winter')
    lim_winter_temp = limit_temperatures.get('lim_winter_temp')
    season_switch_thr = model_info_dict.get('season_switch_thr')
    season_switch_thr_n = model_info_dict.get('season_switch_thr_n')
    mark_prelim_seasons_config = mark_season_config.get('mark_prelim_season')
    min_season_length = mark_prelim_seasons_config.get('min_season_length')
    obv_summer_thr = mark_prelim_seasons_config.get('obv_summer_thr')

    if is_longer_winter:

        max_winter_temp = min(season_switch_thr_n[0], lim_winter_temp)

        s_label[daily_night_avg <= max_winter_temp] = -1
        s_label[daily_night_avg > max_winter_temp] = 0

    else:
        max_winter_temp = min(season_switch_thr[0], lim_winter_temp)

        s_label[daily_avg <= max_winter_temp] = -1
        s_label[daily_avg > max_winter_temp] = 0

    if np.sum(daily_avg >= obv_summer_thr) >= min_season_length:
        max_tr_temp = obv_summer_thr
        s_label[daily_avg >= obv_summer_thr] = 1

    return s_label, max_tr_temp, max_winter_temp


def get_s_label_single_season(model_info_dict, mark_season_config, daily_avg_dict, s_label, max_tr_temp, max_winter_temp):
    """
    Mark s_label for single season
    Args:
        model_info_dict             (dict)          : Dictionary containing all information about the models
        mark_season_config          (dict)          : Dictionary containing all config used to mark seasons
        daily_avg_dict              (dict)          : Dictionary containing daily average
        s_label                     (np.ndarray)    : Season label
        max_tr_temp              (float)         : Max transition temperature
        max_winter_temp          (float)         : Max winter temperature
    Returns:
        s_label                  (np.ndarray)    : Season label
        max_tr_temp              (float)         : Max transition temperature
        max_winter_temp          (float)         : Max winter temperature
    """

    daily_avg = daily_avg_dict.get('daily_avg')
    model_weights = model_info_dict.get('model_weights')
    mark_prelim_seasons_config = mark_season_config.get('mark_prelim_season')
    min_season_length = mark_prelim_seasons_config.get('min_season_length')
    obv_summer_thr = mark_prelim_seasons_config.get('obv_summer_thr')
    obv_winter_thr = mark_prelim_seasons_config.get('obv_winter_thr')

    if np.argmax(model_weights) - 1 == 0:
        if np.sum(daily_avg >= obv_summer_thr) >= min_season_length:
            max_tr_temp = obv_summer_thr
            s_label[daily_avg >= obv_summer_thr] = 1

        if np.sum(daily_avg <= obv_winter_thr) >= min_season_length:
            max_winter_temp = obv_winter_thr
            s_label[daily_avg <= obv_winter_thr] = -1

    elif np.argmax(model_weights) - 1 == 1:

        if np.sum(daily_avg < obv_summer_thr) >= min_season_length:
            max_tr_temp = obv_summer_thr
            s_label[daily_avg < obv_summer_thr] = 0

    elif np.argmax(model_weights) - 1 == -1 and np.sum(daily_avg > obv_winter_thr) >= min_season_length:
        max_winter_temp = obv_winter_thr
        s_label[daily_avg > obv_winter_thr] = 0

    return s_label, max_tr_temp, max_winter_temp


def get_trn_gap(max_tr_temp, max_winter_temp, default_tr_gap):
    """
    Identify the transition gap
    Args:
        max_tr_temp             (float)         : Maximum temperature at which transition is marked
        max_winter_temp         (float)         : Maximum temperature at which winter is marked
        default_tr_gap          (int)           : Default transition gap
    Returns:
        tr_gap                  (int)           : Transition gap
    """
    if not(max_tr_temp == 'NA' or max_winter_temp == 'NA'):
        tr_gap = max(default_tr_gap, int((max_tr_temp - max_winter_temp) / 3))
    else:
        tr_gap = default_tr_gap
    return tr_gap


def get_season_tag(season):
    """
    This function is used to map season to its label
    Args:
        season          (string)        : Season
    Returns:
        season_tag      (int)           : Season tag
    """
    if season == 'summer':
        season_tag = 1
    elif season == 'winter':
        season_tag = -1
    else:
        season_tag = 0

    return season_tag


def get_poss_tr_bool(season, smooth_temp_arr, max_tr_temp, max_winter_temp, tr_gap):
    """
    identify possible transition
    Args:
        season              (str)           : The season to run unification for
        smooth_temp_arr     (np.ndarray)    : Smoothened temperature array
        max_tr_temp         (float)         : Maximum temperature at which transition is marked
        max_winter_temp      (float)         : Maximum temperature at which winter is marked
        tr_gap              (int)           : Transition gap
    Returns:
        poss_tr_bool        (Boolean)       : Boolean containing possible transition days
    """
    if season == 'summer':
        poss_tr_bool = smooth_temp_arr >= (max_tr_temp - tr_gap)
    elif season == 'winter':
        poss_tr_bool = smooth_temp_arr <= (max_winter_temp + tr_gap)
    else:
        poss_tr_bool = np.full_like(smooth_temp_arr, fill_value=True)

    return poss_tr_bool


def assign_season_tag(seq_arr, pre_idx, idx, post_idx, season_tr_tag, seq_score):
    """
    This function is used to assign the season tag
    Args:
        seq_arr             (np.ndarray)    : Array containing information about all sequences in the data
        pre_idx             (int)           : Previous index
        idx                 (int)           : Current index
        post_idx            (int)           : Future index
        season_tr_tag       (int)           : Season tag
        seq_score           (float)         : Sequence score
    Returns:
        seq_arr             (np.ndarray)    : Array containing information about all sequences in the data
    """

    prev_season_trn = seq_arr[pre_idx, 0] == season_tr_tag
    post_season_trn = seq_arr[post_idx, 0] == season_tr_tag
    post_season_neg_trn = seq_arr[post_idx, 0] == -season_tr_tag
    pre_season_neg_trn = seq_arr[pre_idx, 0] == -season_tr_tag
    above_threshold = seq_score >= 0.4
    prev_trn = seq_arr[pre_idx, 0] == 0
    post_trn = seq_arr[post_idx, 0] == 0
    below_threshold = seq_score < 0.4

    check_pre_trn_post_trn_above_thr = (prev_season_trn or post_season_trn) and above_threshold
    check_pre_post_trn_below_thr = (prev_trn or post_trn) and below_threshold
    check_prev_season_trn_post_season_trn = prev_season_trn and post_season_trn
    check_prev_trn_post_trn = prev_trn and post_trn
    check_prev_season_trn_post_neg_season_trn = prev_season_trn and post_season_neg_trn
    check_prev_season_neg_trn_post_season_trn = pre_season_neg_trn and post_season_trn

    cond_1 = cond_2 = cond_3 = cond_4 = 0

    if check_pre_trn_post_trn_above_thr:
        seq_arr[idx, 0] = copy.deepcopy(season_tr_tag)
    elif check_pre_post_trn_below_thr:
        seq_arr[idx, 0] = cond_1
    elif check_prev_season_trn_post_season_trn:
        seq_arr[idx, 0] = season_tr_tag
    elif check_prev_trn_post_trn:
        seq_arr[idx, 0] = cond_2
    elif check_prev_season_trn_post_neg_season_trn:
        seq_arr[idx, 0] = cond_3
    elif check_prev_season_neg_trn_post_season_trn:
        seq_arr[idx, 0] = cond_4

    return seq_arr
