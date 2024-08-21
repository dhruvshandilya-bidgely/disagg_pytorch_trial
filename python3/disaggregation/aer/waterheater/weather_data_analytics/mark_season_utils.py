"""
Author: Mayank Sharan
Created: 12-Jul-2020
Utility functions to help in marking seasons
"""

# Import python packages

import copy
import numpy as np

from sklearn.mixture import GaussianMixture

# Import project functions and classes

from python3.disaggregation.aer.waterheater.weather_data_analytics.math_utils import find_seq
from python3.disaggregation.aer.waterheater.weather_data_analytics.math_utils import rolling_sum
from python3.disaggregation.aer.waterheater.weather_data_analytics.nbi_data_constants import SeqData
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_continuous_season_utils import update_seasons
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_continuous_season_utils import get_all_seasons_label
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_continuous_season_utils import get_s_label_single_season
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_continuous_season_utils import get_s_label_without_smr
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_continuous_season_utils import get_s_label_without_wtr
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_continuous_season_utils import get_trn_gap
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_continuous_season_utils import get_season_tag
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_continuous_season_utils import get_poss_tr_bool
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_continuous_season_utils import assign_season_tag


def get_avg_data(day_wise_data_dict, start_idx, end_idx, mark_season_config):

    """
    Calculate day level average data
    Parameters:
        day_wise_data_dict      (dict)          : Dictionary containing all day wise data matrices
        start_idx               (int)           : Index at which the data for the chunk starts
        end_idx                 (int)           : Index at which the data for the chunk ends
        mark_season_config      (dict)          : Dictionary containing all config used to mark seasons
    Returns:
        daily_avg               (np.ndarray)    : Array containing day level average of temperature
        daily_day_avg           (np.ndarray)    : Array containing day level average of temperature in day hours
        daily_night_avg         (np.ndarray)    : Array containing day level average of temperature in night hours
        smooth_daily_avg        (np.ndarray)    : Array containing smoothed day level average
    """

    # Initialize the config to compute average

    avg_data_config = mark_season_config.get('avg_data')

    # Extract data for the chunk

    day_temp_data = day_wise_data_dict.get('temp')[start_idx:end_idx, :]

    # Compute the averages

    daily_avg = np.nanmean(day_temp_data, axis=1)

    daily_day_data = day_temp_data[:, avg_data_config.get('morn_day_bdry_hr'): avg_data_config.get('eve_day_bdry_hr')]
    daily_day_avg = np.nanmean(daily_day_data, axis=1)

    daily_night_data = np.c_[day_temp_data[:, :avg_data_config.get('morn_day_bdry_hr')],
                             day_temp_data[:, avg_data_config.get('eve_day_bdry_hr'):]]
    daily_night_avg = np.nanmean(daily_night_data, axis=1)

    # Compute running average of temperature to use as smooth average

    window_size = avg_data_config.get('smooth_window_size')
    pad_length = window_size // 2

    daily_avg_padded = np.r_[daily_avg[-pad_length:], daily_avg, daily_avg[:pad_length]]
    smooth_daily_avg = (rolling_sum(daily_avg_padded, window_size)[window_size - 1:]) / window_size

    return daily_avg, daily_day_avg, daily_night_avg, smooth_daily_avg


def fit_gmm(daily_avg, daily_day_avg, daily_night_avg, class_name, mark_season_config):

    """
    Fit GMM to identify seasons
    Parameters:
        daily_avg               (np.ndarray)        : Array containing day level average of temperature
        daily_day_avg           (np.ndarray)        : Array containing day level average of temperature in day hours
        daily_night_avg         (np.ndarray)        : Array containing day level average of temperature in night hours
        class_name              (str)               : The string representing the koppen class the user belongs too
        mark_season_config      (dict)              : Dictionary containing all config used to mark seasons
    Returns:
        model                   (GaussianMixture)   : Model on daily average data
        model_day               (GaussianMixture)   : Model on daily average data of day
        model_night             (GaussianMixture)   : Model on daily average data of night
    """

    # Initialize the config to compute average

    gmm_fit_config = mark_season_config.get('gmm_fit')
    num_components = gmm_fit_config.get('num_components')
    cluster_centres = gmm_fit_config.get('init_cluster_centres')
    random_state = gmm_fit_config.get('random_state')
    cov_type = gmm_fit_config.get('cov_type')
    dn_offset = gmm_fit_config.get('day_night_offset')

    # Initialize the models

    model = GaussianMixture(n_components=num_components, means_init=cluster_centres.get(class_name),
                            random_state=random_state, covariance_type=cov_type)

    model_day = GaussianMixture(n_components=num_components, means_init=cluster_centres.get(class_name) + dn_offset,
                                random_state=random_state, covariance_type=cov_type)

    model_night = GaussianMixture(n_components=num_components, means_init=cluster_centres.get(class_name) - dn_offset,
                                  random_state=random_state, covariance_type=cov_type)

    # Fit the models

    daily_avg = copy.deepcopy(daily_avg)
    daily_avg = daily_avg[~np.isnan(daily_avg)]
    model.fit(np.reshape(daily_avg, newshape=(len(daily_avg), 1)))

    daily_day_avg = copy.deepcopy(daily_day_avg)
    daily_day_avg = daily_day_avg[~np.isnan(daily_day_avg)]
    model_day.fit(np.reshape(daily_day_avg, newshape=(len(daily_day_avg), 1)))

    daily_night_avg = copy.deepcopy(daily_night_avg)
    daily_night_avg = daily_night_avg[~np.isnan(daily_night_avg)]
    model_night.fit(np.reshape(daily_night_avg, newshape=(len(daily_night_avg), 1)))

    return model, model_day, model_night


def identify_seasons(model_info_dict, y, mark_season_config):

    """
    Identify the seasons present for the data based on information from the model
    Parameters:
        model_info_dict         (dict)              : Dictionary containing all information about the models
        y                       (np.ndarray)        : Cluster predictions from the model
        mark_season_config      (dict)              : Dictionary containing all config used to mark seasons
    Returns:
        valid_season_bool       (np.ndarray)        : Array marking existence of major seasons
        is_longer_summer        (bool)              : Boolean marking a day summer
        is_longer_winter        (bool)              : Boolean marking a night winter
    """

    # Initialize variables and config

    model_weights = model_info_dict.get('model_weights')
    model_weights_d = model_info_dict.get('model_weights_d')
    model_weights_n = model_info_dict.get('model_weights_n')

    identify_seasons_config = mark_season_config.get('identify_seasons')

    min_season_length = identify_seasons_config.get('min_season_length')
    max_ext_length = identify_seasons_config.get('max_ext_length')
    longer_season_perc = identify_seasons_config.get('longer_season_perc')

    valid_season_bool = np.logical_and(model_weights >= min_season_length,
                                       np.sum(np.c_[y == 0, y == 1, y == 2], axis=0) > 0)

    # Identify is there is a night winter

    is_longer_winter = False

    winter_idx = 0

    overall_winter = model_weights[winter_idx]
    night_winter = model_weights_n[winter_idx]

    perc_extra_w = (night_winter - overall_winter + 1) / (overall_winter + 1)

    if (night_winter >= min_season_length and overall_winter <= max_ext_length) and perc_extra_w >= longer_season_perc:
        is_longer_winter = True
        valid_season_bool[winter_idx] = True

    # Identify is there is a day summer

    is_longer_summer = False

    summer_idx = 2

    overall_summer = model_weights[summer_idx]
    day_summer = model_weights_d[summer_idx]

    perc_extra_s = (day_summer - overall_summer + 1) / (overall_summer + 1)

    if (day_summer >= min_season_length and overall_summer <= max_ext_length) and perc_extra_s >= longer_season_perc:
        is_longer_summer = True
        valid_season_bool[summer_idx] = True

    return valid_season_bool, is_longer_summer, is_longer_winter


def mark_preliminary_seasons(s_label, daily_avg, daily_day_avg, daily_night_avg, class_name, model_info_dict,
                             mark_season_config):

    """
    Mark the major seasons
    Parameters:
        s_label                 (np.ndarray)    : Array containing labels with all days
        daily_avg               (np.ndarray)    : Array containing day level average of temperature
        daily_day_avg           (np.ndarray)    : Array containing day level average of temperature in day hours
        daily_night_avg         (np.ndarray)    : Array containing day level average of temperature in night hours
        class_name              (str)           : The string representing the koppen class the user belongs too
        model_info_dict         (dict)          : Dictionary containing all information about the models
        mark_season_config      (dict)          : Dictionary containing all config used to mark seasons
    Returns:
        s_label                 (np.ndarray)    : Array containing labels with all days
        max_winter_temp         (float)         : Maximum temperature at which winter is marked
        max_tr_temp             (float)         : Maximum temperature at which transition is marked
    """

    # Extract variables from dictionary

    is_longer_summer = model_info_dict.get('is_longer_summer')
    is_longer_winter = model_info_dict.get('is_longer_winter')

    model_weights = model_info_dict.get('model_weights')
    model_weights_d = model_info_dict.get('model_weights_d')
    model_weights_n = model_info_dict.get('model_weights_n')

    valid_season_bool = model_info_dict.get('valid_season_bool')

    # Initialize config and variable to use

    max_winter_temp = 'NA'
    max_tr_temp = 'NA'

    mark_prelim_seasons_config = mark_season_config.get('mark_prelim_season')

    lim_winter_temp_dict = mark_prelim_seasons_config.get('lim_winter_temp_dict')
    lim_tr_temp_dict = mark_prelim_seasons_config.get('lim_tr_temp_dict')

    bonus_base_days = mark_prelim_seasons_config.get('bonus_base_days')
    limit_base_days = mark_prelim_seasons_config.get('limit_base_days')

    # Initialize variables

    winter_bonus = 0
    summer_bonus = 0

    num_summer_days = model_weights[2]
    if is_longer_summer:
        num_summer_days = model_weights_d[2]

    num_winter_days = model_weights[0]
    if is_longer_winter:
        num_winter_days = model_weights_n[0]

    if class_name == 'D':
        winter_bonus = np.round(num_summer_days / bonus_base_days)

    if class_name == 'A':
        summer_bonus = np.round(num_winter_days / bonus_base_days)

    # Adjust limit temperature

    lim_winter_temp = lim_winter_temp_dict.get(class_name)
    lim_tr_temp = lim_tr_temp_dict.get(class_name)

    if model_weights[2] > limit_base_days:
        lim_tr_temp += np.round(model_weights[2] / limit_base_days)

    if model_weights[0] > limit_base_days:
        lim_winter_temp -= np.round(model_weights[0] / limit_base_days)

    bonus_dict = {
        'winter_bonus': winter_bonus,
        'summer_bonus': summer_bonus
    }
    daily_avg_dict = {
        'daily_avg': daily_avg,
        'daily_day_avg': daily_day_avg,
        'daily_night_avg': daily_night_avg
    }

    limit_temperatures = {
        'lim_tr_temp': lim_tr_temp,
        'lim_winter_temp': lim_winter_temp
    }

    # We have all 3 seasons and 2 boundaries Or we have 2 seasons without transition

    if np.sum(valid_season_bool) == 3 or (np.sum(valid_season_bool) == 2 and not (valid_season_bool[1])):

        s_label, max_tr_temp, max_winter_temp = \
            get_all_seasons_label(model_info_dict, limit_temperatures, bonus_dict, daily_avg_dict, s_label, max_tr_temp,
                                  max_winter_temp)

    elif np.sum(valid_season_bool) == 2 and not (valid_season_bool[0]):

        # We have 2 seasons without winter

        s_label, max_tr_temp, max_winter_temp = \
            get_s_label_without_wtr(model_info_dict, mark_season_config, limit_temperatures, daily_avg_dict, s_label,
                                    max_tr_temp, max_winter_temp)

    elif np.sum(valid_season_bool) == 2 and not (valid_season_bool[2]):

        # We have 2 seasons without summer

        s_label, max_tr_temp, max_winter_temp =\
            get_s_label_without_smr(model_info_dict, mark_season_config, limit_temperatures, daily_avg_dict, s_label,
                                    max_tr_temp, max_winter_temp)

    elif np.sum(valid_season_bool) == 1:

        s_label[:] = np.argmax(model_weights) - 1

        s_label, max_tr_temp, max_winter_temp =\
            get_s_label_single_season(model_info_dict, mark_season_config, daily_avg_dict, s_label, max_tr_temp,
                                      max_winter_temp)

    s_label_nan_bool = np.isnan(s_label)
    s_label[s_label_nan_bool] = 0

    return s_label, max_winter_temp, max_tr_temp


def mark_transition_seasons(s_label, daily_avg, daily_day_avg, daily_night_avg, max_winter_temp, max_tr_temp,
                            model_info_dict):
    """
    Mark the transition seasons
    Parameters:
        s_label                 (np.ndarray)    : Array containing labels with all days
        daily_avg               (np.ndarray)    : Array containing day level average of temperature
        daily_day_avg           (np.ndarray)    : Array containing day level average of temperature in day hours
        daily_night_avg         (np.ndarray)    : Array containing day level average of temperature in night hours
        max_winter_temp         (float)         : Maximum temperature at which winter is marked
        max_tr_temp             (float)         : Maximum temperature at which transition is marked
        model_info_dict         (dict)          : Dictionary containing all information about the models
    Returns:
        s_label                 (np.ndarray)    : Array containing labels with all days
    """

    # Extract variables from dictionary

    is_longer_summer = model_info_dict.get('is_longer_summer')
    is_longer_winter = model_info_dict.get('is_longer_winter')

    model = model_info_dict.get('model')
    model_day = model_info_dict.get('model_day')
    model_night = model_info_dict.get('model_night')

    s_label = s_label.astype(float)

    # Initialize probabilities and nan bool to use

    winter_prob = np.full_like(s_label, fill_value=0.)
    summer_prob = np.full_like(s_label, fill_value=0.)

    # Get probabilities for each day

    daily_avg_non_nan_bool = ~np.isnan(daily_avg)

    if np.sum(daily_avg_non_nan_bool):
        y_prob = model.predict_proba(np.reshape(daily_avg[daily_avg_non_nan_bool],
                                                newshape=(np.sum(daily_avg_non_nan_bool), 1)))

        if is_longer_winter:
            daily_n_avg_non_nan_bool = ~np.isnan(daily_night_avg)
            y_prob_n = model_night.predict_proba(np.reshape(daily_night_avg[daily_n_avg_non_nan_bool],
                                                            newshape=(np.sum(daily_n_avg_non_nan_bool), 1)))
            winter_prob[daily_n_avg_non_nan_bool] = y_prob_n[:, 0]
        else:
            winter_prob[daily_avg_non_nan_bool] = y_prob[:, 0]

        if is_longer_summer:
            daily_d_avg_non_nan_bool = ~np.isnan(daily_day_avg)
            y_prob_d = model_day.predict_proba(np.reshape(daily_day_avg[daily_d_avg_non_nan_bool],
                                                          newshape=(np.sum(daily_d_avg_non_nan_bool), 1)))
            summer_prob[daily_d_avg_non_nan_bool] = y_prob_d[:, 2]
        else:
            summer_prob[daily_avg_non_nan_bool] = y_prob[:, 2]

        tr_prob = 1 - (summer_prob + winter_prob)
        tr_prob[tr_prob < 0] = 0

        # Find boolean for days to be marked as transition winter and summer

        prob_thr = 0.7

        tr_winter_bool = np.logical_and(np.logical_or(s_label == 0, s_label == -1),
                                        np.logical_and(winter_prob <= prob_thr, tr_prob <= prob_thr))

        tr_winter_bool = np.logical_and(winter_prob > summer_prob, tr_winter_bool)

        tr_summer_bool = np.logical_and(np.logical_or(s_label == 0, s_label == 1),
                                        np.logical_and(summer_prob <= prob_thr, tr_prob <= prob_thr))

        tr_summer_bool = np.logical_and(winter_prob < summer_prob, tr_summer_bool)

        if not(max_winter_temp == 'NA'):
            s_label[tr_winter_bool] = -0.5

        if not(max_tr_temp == 'NA'):
            s_label[tr_summer_bool] = 0.5

    return s_label


def perform_hysteresis_smoothening(s_label, mark_season_config):

    """
    Perform hysteresis smoothening on the labels
    Parameters:
        s_label                 (np.ndarray)    : Array containing labels with all days
        mark_season_config      (dict)          : Dictionary containing all config used to mark seasons
    Returns:
        s_label                 (np.ndarray)    : Array containing labels with all days
    """

    # Initialize variables and config

    hysteresis_smoothening_config = mark_season_config.get('hysteresis_smoothening')

    past_period = hysteresis_smoothening_config.get('past_period')
    future_period = hysteresis_smoothening_config.get('future_period')

    past_weight = hysteresis_smoothening_config.get('past_weight')
    future_weight = hysteresis_smoothening_config.get('future_weight')

    # Create 2d past data to compute smoothed labels in one go

    past_labels = np.full(shape=(s_label.shape[0], past_period), fill_value=0.)

    for shift_idx in range(1, past_period + 1):

        past_col_idx = past_period - shift_idx
        shifted_label_arr = np.r_[s_label[-shift_idx:], s_label[:-shift_idx]]
        past_labels[:, past_col_idx] = shifted_label_arr

    # Create 2d future data to compute smoothed labels in one go

    future_labels = np.full(shape=(s_label.shape[0], future_period), fill_value=0.)

    for shift_idx in range(1, future_period + 1):
        future_col_idx = shift_idx - 1
        shifted_label_arr = np.r_[s_label[shift_idx:], s_label[:shift_idx]]
        future_labels[:, future_col_idx] = shifted_label_arr

    # Compute the smoothed labels

    past_labels_avg = np.mean(past_labels, axis=1)
    future_labels_avg = np.mean(future_labels, axis=1)
    avg_label = np.round((past_labels_avg * past_weight) + (future_labels_avg * future_weight) + s_label) / 2
    s_label = copy.deepcopy(avg_label)

    return s_label


def mark_short_temp_events(daily_avg, smooth_daily_avg, s_label, mark_season_config):

    """
    Identify temperature based seasonal events
    Parameters:
        daily_avg               (np.ndarray)    : Array containing day level average of temperature
        smooth_daily_avg        (np.ndarray)    : Array containing smoothed day level average
        s_label                 (np.ndarray)    : Array containing labels with all days
        mark_season_config      (dict)          : Dictionary containing all config used to mark seasons
    Returns:
        is_hot_event_bool       (np.ndarray)    : Array marking days that are hot events
        is_cold_event_bool      (np.ndarray)    : Array marking days that are cold events
    """

    # Initialize variables and config

    num_days = len(s_label)

    temp_event_det_config = mark_season_config.get('temp_event_det')
    event_window = temp_event_det_config.get('event_window')
    ev_thr = temp_event_det_config.get('min_ev_score')

    # Mark severe temperature events

    diff_arr = daily_avg - smooth_daily_avg
    hotter_days = diff_arr >= 0
    colder_days = diff_arr < 0

    hot_days_diff = copy.deepcopy(diff_arr)
    hot_days_diff[colder_days] = 0

    cold_days_diff = copy.deepcopy(diff_arr)
    cold_days_diff[hotter_days] = 0

    hot_days_diff = np.abs(hot_days_diff)
    cold_days_diff = np.abs(cold_days_diff)
    diff_arr = np.abs(diff_arr)

    diff_mean = np.mean(diff_arr)
    diff_std = np.std(diff_arr)

    # Get a score for a day to be marked as a temperature event

    hot_ev_score = (hot_days_diff - diff_mean) / diff_std
    cold_ev_score = (cold_days_diff - diff_mean) / diff_std

    padding = event_window // 2

    hot_ev_score_padded = np.r_[hot_ev_score[-padding:], hot_ev_score, hot_ev_score[:padding]]
    cold_ev_score_padded = np.r_[cold_ev_score[-padding:], cold_ev_score, cold_ev_score[:padding]]

    hot_ev_score = rolling_sum(hot_ev_score_padded, event_window)[event_window - 1:]
    cold_ev_score = rolling_sum(cold_ev_score_padded, event_window)[event_window - 1:]

    hot_ev_bool = np.array(hot_ev_score >= ev_thr)
    cold_ev_bool = np.array(cold_ev_score >= ev_thr)

    hot_events = find_seq(hot_ev_bool, min_seq_length=1)
    cold_events = find_seq(cold_ev_bool, min_seq_length=1)

    # Populate hot and cold event boolean arrays

    is_hot_event_bool = np.full_like(s_label, fill_value=False)

    for ev_idx in range(hot_events.shape[0]):
        if hot_events[ev_idx, 0] == 1:
            is_hot_event_bool[max(hot_events[ev_idx, 1] - 1, 0): min(hot_events[ev_idx, 2] + 2, num_days)] = True

    is_cold_event_bool = np.full_like(s_label, fill_value=False)

    for ev_idx in range(cold_events.shape[0]):
        if cold_events[ev_idx, 0] == 1:
            is_cold_event_bool[max(cold_events[ev_idx, 1] - 1, 0): min(cold_events[ev_idx, 2] + 2, num_days)] = True

    return is_hot_event_bool, is_cold_event_bool


def merge_seq_arr(seq_arr, num_days_data):

    """
    Utility function to merge seq arr
    Parameters:
        seq_arr                 (np.ndarray)    : Array containing information about all sequences in the data
        num_days_data           (int)           : Number of days in the data
    Returns:
        seq_arr                 (np.ndarray)    : Array containing information about all sequences in the data
    """

    # Loop through all sequences to merge consecutive sequences with same value

    for idx in range(seq_arr.shape[0] - 1):

        curr_seq = seq_arr[idx, :]
        next_seq = seq_arr[idx + 1, :]

        # Merge sequences if same

        if curr_seq[SeqData.seq_val_col] == next_seq[SeqData.seq_val_col]:

            seq_arr[idx + 1, SeqData.seq_start_col] = curr_seq[SeqData.seq_start_col]
            seq_arr[idx + 1, SeqData.seq_len_col] = next_seq[SeqData.seq_end_col] - curr_seq[SeqData.seq_start_col] + 1
            seq_arr[idx, SeqData.seq_val_col] = np.nan

        elif not(curr_seq[SeqData.seq_end_col] + 1 == next_seq[SeqData.seq_start_col]):

            seq_arr[idx, SeqData.seq_end_col] = next_seq[SeqData.seq_start_col] - 1
            seq_arr[idx, SeqData.seq_len_col] = next_seq[SeqData.seq_start_col] - curr_seq[SeqData.seq_start_col]

    seq_arr = seq_arr[~(np.isnan(seq_arr[:, SeqData.seq_val_col])), :]

    # Handle if we are continuing a sequence from end of data

    if not(seq_arr[0, SeqData.seq_start_col] == 0):

        if seq_arr[0, SeqData.seq_val_col] == seq_arr[-1, SeqData.seq_val_col]:

            seq_arr[0, SeqData.seq_start_col] = 0
            seq_arr[0, SeqData.seq_len_col] = seq_arr[0, SeqData.seq_end_col] + 1

        else:

            seq_arr = np.r_[[[seq_arr[-1, SeqData.seq_val_col], 0, seq_arr[0, SeqData.seq_start_col] - 1,
                              seq_arr[0, SeqData.seq_start_col]]], seq_arr]

    if not(seq_arr[-1, SeqData.seq_end_col] == (num_days_data - 1)):

        seq_arr[-1, SeqData.seq_end_col] = (num_days_data - 1)
        seq_arr[-1, SeqData.seq_len_col] = num_days_data - seq_arr[-1, SeqData.seq_start_col]

    return seq_arr


def modify_event_tags(seq_arr, is_hot_event_bool, is_cold_event_bool, num_days_data, mark_season_config):

    """
    Modify season labels for chunks with high percentage of temperature events
    Parameters:
        seq_arr                 (np.ndarray)    : Array containing information about all sequences in the data
        is_hot_event_bool       (np.ndarray)    : Array marking days that are hot events
        is_cold_event_bool      (np.ndarray)    : Array marking days that are cold events
        num_days_data           (int)           : Number of days in the data
        mark_season_config      (dict)          : Dictionary containing all config used to mark seasons
    Returns:
        seq_arr                 (np.ndarray)    : Array containing information about all sequences in the data
    """

    # Initialize config and variables

    num_seq = seq_arr.shape[0]

    modify_event_tag_config = mark_season_config.get('modify_event_tag')
    event_ratio_thr = modify_event_tag_config.get('event_ratio_thr')
    label_offset = modify_event_tag_config.get('label_offset')

    for idx in range(num_seq):

        if (idx == 0 or idx == num_seq - 1) and (seq_arr[0, 0] == seq_arr[-1, 0]):

            seq_info_1 = seq_arr[0, :].astype(int)
            seq_info_2 = seq_arr[-1, :].astype(int)

            hot_event_ratio = (np.sum(is_hot_event_bool[seq_info_1[SeqData.seq_start_col]:
                                                        seq_info_1[SeqData.seq_end_col] + 1]) +
                               np.sum(is_hot_event_bool[seq_info_2[SeqData.seq_start_col]:
                                                        seq_info_2[SeqData.seq_end_col] + 1]))

            hot_event_ratio = hot_event_ratio / (seq_info_1[SeqData.seq_len_col] + seq_info_2[SeqData.seq_len_col])

            cold_event_ratio = (np.sum(is_cold_event_bool[seq_info_1[SeqData.seq_start_col]:
                                                          seq_info_1[SeqData.seq_end_col] + 1]) +
                                np.sum(is_cold_event_bool[seq_info_2[SeqData.seq_start_col]:
                                                          seq_info_2[SeqData.seq_end_col] + 1]))

            cold_event_ratio = cold_event_ratio / (seq_info_1[SeqData.seq_len_col] + seq_info_2[SeqData.seq_len_col])

        else:

            seq_info = seq_arr[idx, :]

            hot_event_ratio = np.sum(is_hot_event_bool[int(seq_info[SeqData.seq_start_col]):
                                                       int(seq_info[SeqData.seq_end_col]) + 1])
            hot_event_ratio = hot_event_ratio / seq_info[SeqData.seq_len_col]

            cold_event_ratio = np.sum(is_cold_event_bool[int(seq_info[SeqData.seq_start_col]):
                                                         int(seq_info[SeqData.seq_end_col]) + 1])
            cold_event_ratio = cold_event_ratio / seq_info[SeqData.seq_len_col]

        if hot_event_ratio > event_ratio_thr:
            seq_arr[idx, SeqData.seq_val_col] -= label_offset

        if cold_event_ratio > event_ratio_thr:
            seq_arr[idx, SeqData.seq_val_col] += label_offset

    seq_arr = merge_seq_arr(seq_arr, num_days_data)

    return seq_arr


def get_season_seq_metrics(season_seq_idx, num_seq, seq_arr, smooth_daily_avg):
    """
    Calculate the season sequence metrics
    Args:
        season_seq_idx          (list)       : Season sequence index
        num_seq                 (int)        : Shape of seq arr
        seq_arr                 (np.ndarray) : Overall season sequence array
        smooth_daily_avg        (np.ndarray) : Smoothened daily average
    Returns:
        season_seq_metrics      (list)       : Sequence metrics list
    """

    season_seq_metrics = []

    for idx in season_seq_idx:

        if (idx == 0 or idx == num_seq - 1) and (seq_arr[0, 0] == seq_arr[-1, 0]):

            seq_info_1 = seq_arr[0, :].astype(int)
            seq_info_2 = seq_arr[-1, :].astype(int)

            sum_length = seq_info_1[SeqData.seq_len_col] + seq_info_2[SeqData.seq_len_col]
            avg_smooth_temp = np.mean(np.r_[smooth_daily_avg[seq_info_1[SeqData.seq_start_col]:
                                                             seq_info_1[SeqData.seq_end_col] + 1],
                                            smooth_daily_avg[seq_info_2[SeqData.seq_start_col]:
                                                             seq_info_2[SeqData.seq_end_col] + 1]])

        else:

            seq_info = seq_arr[idx, :].astype(int)
            sum_length = seq_info[SeqData.seq_len_col]
            avg_smooth_temp = np.mean(smooth_daily_avg[seq_info[SeqData.seq_start_col]:
                                                       seq_info[SeqData.seq_end_col] + 1])

        season_seq_metrics.append([idx, sum_length, avg_smooth_temp])

    return season_seq_metrics


def unite_mutliple_smr(num_valid_seasons, seq_arr, valid_season_seq_idx, max_gap_bridge, num_days_data, season_tag):
    """
    Unite multiple summer seasons
    Args:
        num_valid_seasons           (int)           : Number of valid seasons
        seq_arr                     (np.ndarray)    : Array containing information about all sequences in the data
        valid_season_seq_idx        (np.ndarray)    : Array containing the index of valid seasons
        max_gap_bridge              (int)           : maximum gap allowed
        num_days_data               (int)           : Number of days in the data
        season_tag                  (int)           : Season tag
    Returns:
        seq_arr                     (np.ndarray)    : Array containing information about all sequences in the data
    """
    if num_valid_seasons > 1:

        for idx in range(num_valid_seasons):

            next_idx = (idx + 1) % num_valid_seasons

            curr_season_info = seq_arr[valid_season_seq_idx[idx], :].astype(int)
            next_season_info = seq_arr[valid_season_seq_idx[next_idx], :].astype(int)

            if 0 <= (next_season_info[1] - curr_season_info[2]) <= max_gap_bridge:
                seq_arr[valid_season_seq_idx[idx]: valid_season_seq_idx[next_idx], 0] = season_tag

            elif 0 <= ((num_days_data - curr_season_info[2]) + next_season_info[1]) <= max_gap_bridge:
                seq_arr[valid_season_seq_idx[idx]:, 0] = season_tag
                seq_arr[:valid_season_seq_idx[next_idx], 0] = season_tag

    return seq_arr


def unite_season(seq_arr, smooth_daily_avg, max_tr_temp, max_winter_temp, num_days_data, mark_season_config, season):

    """
    Unite the given major season summer or winter as 1 continuous chunk
    Parameters:
        seq_arr                 (np.ndarray)    : Array containing information about all sequences in the data
        smooth_daily_avg        (np.ndarray)    : Array containing smoothed day level average
        max_winter_temp         (float)         : Maximum temperature at which winter is marked
        max_tr_temp             (float)         : Maximum temperature at which transition is marked
        num_days_data           (int)           : Number of days in the data
        mark_season_config      (dict)          : Dictionary containing all config used to mark seasons
        season                  (str)           : The season to run unification for
    Returns:
        seq_arr                 (np.ndarray)    : Array containing information about all sequences in the data
    """

    # Initialize variables and config

    num_seq = seq_arr.shape[0]

    if season == 'summer':

        if max_tr_temp == 'NA':
            return seq_arr

        season_tag = 1
        season_tr_tag = 0.5
        temp_thr = max_tr_temp - 2

    elif season == 'winter':

        if max_winter_temp == 'NA':
            return seq_arr

        season_tag = -1
        season_tr_tag = -0.5
        temp_thr = max_winter_temp + 2

    else:
        season_tag = 0
        season_tr_tag = 0
        temp_thr = np.nan

    unite_season_config = mark_season_config.get('unite_season')
    season_length_base = unite_season_config.get('season_length_base')
    season_temp_base = unite_season_config.get('season_temp_base')
    length_weight = unite_season_config.get('length_weight')
    temp_weight = unite_season_config.get('temp_weight')
    max_gap_bridge = unite_season_config.get('max_gap_bridge')

    # Unite the season

    season_seq_bool = seq_arr[:, SeqData.seq_val_col] == season_tag
    num_season_seq = np.sum(season_seq_bool)

    has_single_season = (num_season_seq <= 1) or (num_season_seq == 2 and season_seq_bool[0] and season_seq_bool[-1])

    if not has_single_season:

        season_seq_idx = np.where(season_seq_bool)[0]

        season_seq_metrics = get_season_seq_metrics(season_seq_idx, num_seq, seq_arr, smooth_daily_avg)

        # Score each season sequence

        season_seq_metrics = np.array(season_seq_metrics)
        season_score = np.full(shape=(len(season_seq_idx), 2), fill_value=0.)

        season_score[:, 0] = season_seq_metrics[:, 1] / season_length_base
        season_score[season_score[:, 0] > 1, 0] = 1

        if season == 'summer':
            season_score[:, 1] = 0.5 + ((season_seq_metrics[:, 2] - temp_thr) / season_temp_base)

        if season == 'winter':
            season_score[:, 1] = 0.5 + ((temp_thr - season_seq_metrics[:, 2]) / season_temp_base)

        season_score[season_score[:, 1] > 1, 1] = 1
        season_score[season_score[:, 1] < 0, 1] = 0

        season_seq_scores = np.c_[season_seq_metrics, season_score,
                                  length_weight * season_score[:, 0] + temp_weight * season_score[:, 1]]

        non_season_bool = season_seq_scores[:, 5] < 0.5

        # Evaluate each non season sequence and update appropriately to transition or transition season

        non_season_seq_idx = season_seq_scores[non_season_bool, 0]
        non_season_seq_scores = season_seq_scores[non_season_bool, 5]

        for non_idx in range(len(non_season_seq_idx)):

            idx = int(non_season_seq_idx[non_idx])
            seq_score = non_season_seq_scores[non_idx]

            pre_idx = (idx - 1) % num_seq
            post_idx = (idx + 1) % num_seq

            seq_arr = assign_season_tag(seq_arr, pre_idx, idx, post_idx, season_tr_tag, seq_score)

        # If multiple summer sequences are valid unite them

        num_valid_seasons = int(np.sum(~non_season_bool))
        valid_season_seq_idx = season_seq_scores[~non_season_bool, 0].astype(int)

        seq_arr = unite_mutliple_smr(num_valid_seasons, seq_arr, valid_season_seq_idx, max_gap_bridge, num_days_data,
                                     season_tag)

    seq_arr = merge_seq_arr(seq_arr, num_days_data)

    return seq_arr


def make_seasons_continuous(seq_arr, smooth_daily_avg, max_tr_temp, max_winter_temp, num_days_data):

    """
    Make seasons continuous by resolving sandwiched chunks
    Parameters:
        seq_arr                 (np.ndarray)    : Array containing information about all sequences in the data
        smooth_daily_avg        (np.ndarray)    : Array containing smoothed day level average
        max_winter_temp         (float)         : Maximum temperature at which winter is marked
        max_tr_temp             (float)         : Maximum temperature at which transition is marked
        num_days_data           (int)           : Number of days in the data
    Returns:
        seq_arr                 (np.ndarray)    : Array containing information about all sequences in the data
    """

    # Initialize variables

    num_seq = seq_arr.shape[0]

    default_tr_gap = 7

    if not(max_tr_temp == 'NA' or max_winter_temp == 'NA'):
        tr_gap = max(default_tr_gap, int((max_tr_temp - max_winter_temp) / 3))
    else:
        tr_gap = default_tr_gap

    # Fix sandwiched seasons

    if num_seq >= 2:
        for idx in range(num_seq):
            pre_idx = (idx - 1) % num_seq
            post_idx = (idx + 1) % num_seq

            pre_seq = seq_arr[pre_idx, :]
            curr_seq = seq_arr[idx, :]
            post_seq = seq_arr[post_idx, :]

            if idx == 0 and pre_seq[0] == curr_seq[0]:

                smooth_avg_curr = np.mean(np.r_[smooth_daily_avg[int(pre_seq[1]): int(pre_seq[2]) + 1],
                                                smooth_daily_avg[int(curr_seq[1]): int(curr_seq[2]) + 1]])
                seq_len = curr_seq[3] + pre_seq[3]

                pre_idx = (idx - 2) % num_seq
                pre_seq = seq_arr[pre_idx, :]

            elif idx == num_seq - 1 and curr_seq[0] == post_seq[0]:

                smooth_avg_curr = np.mean(np.r_[smooth_daily_avg[int(curr_seq[1]): int(curr_seq[2]) + 1],
                                                smooth_daily_avg[int(post_seq[1]): int(post_seq[2]) + 1]])
                seq_len = curr_seq[3] + post_seq[3]

                post_idx = (idx + 2) % num_seq
                post_seq = seq_arr[post_idx, :]

            else:

                seq_len = curr_seq[3]
                smooth_avg_curr = np.mean(smooth_daily_avg[int(curr_seq[1]): int(curr_seq[2]) + 1])

            if seq_len <= 60:

                seq_arr = update_seasons(pre_seq, post_seq, curr_seq, seq_arr, idx, smooth_avg_curr, max_winter_temp,
                                         max_tr_temp, tr_gap)

    seq_arr = merge_seq_arr(seq_arr, num_days_data)

    return seq_arr


def ensure_tr_season(seq_arr, smooth_daily_avg, max_tr_temp, max_winter_temp, season):

    """
    Ensure summer and winter are surrounded by transitions
    Parameters:
        seq_arr                 (np.ndarray)    : Array containing information about all sequences in the data
        smooth_daily_avg        (np.ndarray)    : Array containing smoothed day level average
        max_winter_temp         (float)         : Maximum temperature at which winter is marked
        max_tr_temp             (float)         : Maximum temperature at which transition is marked
        season                  (str)           : The season to run unification for
    Returns:
        seq_arr                 (np.ndarray)    : Array containing information about all sequences in the data
    """

    # Initialize variables and config

    num_seq = seq_arr.shape[0]

    default_tr_gap = 5

    tr_gap = get_trn_gap(max_tr_temp, max_winter_temp, default_tr_gap)

    season_tag = get_season_tag(season)

    # Identify the type of unified season

    season_seq_idx = np.where(seq_arr[:, 0] == season_tag)[0]
    num_season_seq = len(season_seq_idx)

    # If Season is absent return

    if num_season_seq == 0:
        return seq_arr

    if num_season_seq == 1:
        pre_season_idx = (season_seq_idx[0] - 1) % num_seq
    elif num_season_seq == 2:
        pre_season_idx = (season_seq_idx[1] - 1) % num_seq
    else:
        pre_season_idx = np.nan

    pre_season_seq = seq_arr[pre_season_idx, :]

    # Check if pre season is a transition else create one

    if not(pre_season_seq[0] == (season_tag / 2)) and (pre_season_seq[0] == 0):

        smooth_temp_arr = smooth_daily_avg[int(pre_season_seq[1]): int(pre_season_seq[2]) + 1]

        poss_tr_bool = get_poss_tr_bool(season, smooth_temp_arr, max_tr_temp, max_winter_temp, tr_gap)

        temp_seq_arr = find_seq(poss_tr_bool)
        temp_seq_arr = temp_seq_arr[temp_seq_arr[:, 0] == 1, :]

        if temp_seq_arr.shape[0] == 0:
            num_pts_extract = min(7, pre_season_seq[3] // 2)
        else:
            num_pts_extract = min(int(len(poss_tr_bool) - temp_seq_arr[-1, 1]), pre_season_seq[3] // 2)

        # Split pre season seq

        first_split = [0, pre_season_seq[1], pre_season_seq[2] - num_pts_extract, pre_season_seq[3] - num_pts_extract]
        second_split = [season_tag / 2, pre_season_seq[2] - num_pts_extract + 1, pre_season_seq[2], num_pts_extract]

        seq_arr = np.r_[seq_arr[: pre_season_idx], [first_split, second_split], seq_arr[pre_season_idx + 1:]]

    season_seq_idx = np.where(seq_arr[:, 0] == season_tag)[0]

    if num_season_seq == 1 or num_season_seq == 2:
        post_season_idx = (season_seq_idx[0] + 1) % num_seq
    else:
        post_season_idx = np.nan

    post_season_seq = seq_arr[post_season_idx, :]

    # Check if post season is a transition else create one

    if not(post_season_seq[0] == (season_tag / 2)) and (post_season_seq[0] == 0):

        smooth_temp_arr = smooth_daily_avg[int(post_season_seq[1]): int(post_season_seq[2]) + 1]

        poss_tr_bool = get_poss_tr_bool(season, smooth_temp_arr, max_tr_temp, max_winter_temp, tr_gap)

        temp_seq_arr = find_seq(poss_tr_bool)
        temp_seq_arr = temp_seq_arr[temp_seq_arr[:, 0] == 1, :]

        if temp_seq_arr.shape[0] == 0:
            num_pts_extract = min(7, post_season_seq[3] // 2)
        else:
            num_pts_extract = min(int(temp_seq_arr[0, 2] + 1), post_season_seq[3] // 2)

        # Split pre season seq

        first_split = [season_tag / 2, post_season_seq[1], post_season_seq[1] + num_pts_extract - 1, num_pts_extract]
        second_split = [0, post_season_seq[1] + num_pts_extract, post_season_seq[2],
                        post_season_seq[3] - num_pts_extract]

        seq_arr = np.r_[seq_arr[: post_season_idx], [first_split, second_split], seq_arr[post_season_idx + 1:]]

    return seq_arr
