"""
Author - Sahana M
Date - 15-Feb-2023
Module to get EV detection confidence
"""

# Import python packages
import numpy as np
import pandas as pd
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.find_seq import find_seq
from python3.utils.code_utils import get_2d_matrix


def get_day_data_2d(input_data, sampling_rate):

    """
    Covert 2 D timestamp data to day level data

        Parameters:
            input_data                   (np.ndarray)         : timestamp level input data
            sampling_rate                (int)                : sampling rate of the user

        Returns:
            day_input_data               (np.ndarray)         : day level input data
            day_output_data              (np.ndarray)         : day level disagg output data
    """

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    day_idx = Cgbdisagg.INPUT_DAY_IDX

    # Prepare day timestamp matrix and get size of all matrices

    day_ts, row_idx2, row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True,
                                          return_index=True)
    day_ts = np.tile(day_ts, reps=(num_pd_in_day, 1)).transpose()

    # Compute hour of day based indices to use

    col_idx = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[:, day_idx]
    col_idx = col_idx / Cgbdisagg.SEC_IN_HOUR
    col_idx = (pd_mult * (col_idx - col_idx.astype(int) + input_data[:, Cgbdisagg.INPUT_HOD_IDX])).astype(int)

    # Create day wise 2d arrays for each variable

    epochs_in_a_day = int(Cgbdisagg.HRS_IN_DAY * (Cgbdisagg.SEC_IN_HOUR / sampling_rate))
    no_of_days = len(day_ts)

    day_input_data = np.zeros((no_of_days, epochs_in_a_day, len(input_data[0])))

    day_input_data[row_idx, col_idx, :] = input_data

    day_input_data = np.swapaxes(day_input_data, 0, 1)
    day_input_data = np.swapaxes(day_input_data, 0, 2)

    return day_input_data, row_idx, col_idx


def get_daily_ev_mean(strikes_present_idx, ev_data):
    """
    This function is used to get the daily EV consumption data along with its relation to temperature
    Parameters:
        strikes_present_idx             (np.ndarray)            : Indexes where EV strikes are present
        ev_data                         (np.ndarray)            : 2D ev data matrix
    Returns:
        daily_ev_mean                   (np.ndarray)            : Daily ev average
        daily_diff_mean                 (np.ndarray)            : Daily temperature difference average
    """

    # for each day calculate the ev average and the difference with the temperature average

    daily_diff_mean = []
    daily_ev_mean = []
    for i in range(len(strikes_present_idx)):
        if np.sum(strikes_present_idx[i]):
            strike_idx = strikes_present_idx[i] > 0
            daily_diff_mean.append(np.nanmean(strikes_present_idx[i][strike_idx]))
            daily_ev_mean.append(np.nanmean(ev_data[i][strike_idx]))

    return daily_ev_mean, daily_diff_mean


def get_season_means(box_data, norm_diff_feels_like_data, ev_local_s_label, s_label_data):
    """
    Function to calculate the average temperatures for each season
    Parameters:
        box_data                    (np.ndarray)                : 2D matrix of ev data
        norm_diff_feels_like_data   (np.ndarray)                : Normalised feels like temperature data
        ev_local_s_label            (np.ndarray)                : Season label calculated within the ev module
        s_label_data                (np.ndarray)                : Season label obtained from the 28 column matrix
    Returns:
        season_means                (np.ndarray)                : Season average temperature
        daily_seasons               (np.ndarray)                : Season label for a day
        unique_seasons              (np.ndarray)                : Total unique seasons in the data
    """

    season_means = []
    new_box_data_cols = {
        'month_indexes': 28,
        'season_labels': 29,
        'vacation_index': 30,
    }

    # check if the season label from the 28 column input data matrix can be used

    if np.sum(np.isnan(box_data[:, Cgbdisagg.INPUT_S_LABEL_IDX])) == len(box_data):
        unique_seasons = np.unique(box_data[:, new_box_data_cols['season_labels']])
        unique_seasons = unique_seasons[~np.isnan(unique_seasons)]
        daily_seasons = np.max(ev_local_s_label, axis=1)
        for i in range(len(unique_seasons)):
            current_season = unique_seasons[i]
            season_mean = np.nanmean(norm_diff_feels_like_data[ev_local_s_label == current_season])
            season_means.append(season_mean)
    else:

        # using computed s_label from within the ev module

        unique_seasons = np.unique(s_label_data)
        unique_seasons = unique_seasons[~np.isnan(unique_seasons)]
        daily_seasons = np.max(s_label_data, axis=1)
        for i in range(len(unique_seasons)):
            current_season = unique_seasons[i]
            season_mean = np.nanmean(norm_diff_feels_like_data[s_label_data == current_season])
            season_means.append(season_mean)

    return season_means, daily_seasons, unique_seasons


def get_season_seq(season_means, daily_seasons):
    """
    This function is used to get the season sequence
    Parameters:
        season_means                (np.ndarray)                : Average temperature for a season
        daily_seasons               (np.ndarray)                : Season marked at daily level
    Returns:
        season_seq                  (np.ndarray)                : Season sequences
    """

    # Identify the season sequence and format them for avoiding continuous repetitions

    season_seq = find_seq(daily_seasons, min_seq_length=7)
    formatted_season_seq = np.asarray(season_seq[0, :]).reshape(1, -1)
    formatted_season_means = []
    for i in range(len(season_seq) - 1):
        last_index = int(len(formatted_season_seq) - 1)
        if (season_seq[i + 1, 0] == formatted_season_seq[last_index, 0]) and (
                season_seq[i + 1, 1] - formatted_season_seq[last_index, 2]) <= 10:
            formatted_season_seq[last_index, 2] = season_seq[i + 1, 2]
            formatted_season_seq[last_index, 3] = season_seq[i + 1, 3] + formatted_season_seq[last_index, 3]
            formatted_season_means.append(season_means)
        else:
            formatted_season_seq = np.r_[formatted_season_seq, season_seq[i + 1].reshape(1, -1)]

    season_seq = formatted_season_seq

    return season_seq


def get_vacation_percentage(season_seq, vacation_data):
    """
    Function to calculate the vacation percentage for a season
    Parameters:
        season_seq                  (np.ndarray)                : Season sequences
        vacation_data               (np.ndarray)                : Vacation data matrix
    Returns:
        vacation_percentage         (np.ndarray)                : Vacation percentage for each season
    """

    vacation_percentage = []
    for i in range(len(season_seq)):
        start_idx = int(season_seq[i, 1])
        end_idx = int(season_seq[i, 2])
        curr_vacation_data = vacation_data[start_idx: end_idx]
        curr_vacation_perc = np.sum(np.sum(curr_vacation_data, axis=1) > 0)/len(curr_vacation_data)
        vacation_percentage.append(curr_vacation_perc)

    vacation_percentage = np.asarray(vacation_percentage)

    return vacation_percentage


def get_season_proportions(season_means, daily_diff_mean, season_seq, unique_seasons, ev_data):
    """
    Function to calculate the EV proportion for each season
    Parameters:
        season_means                (np.ndarray)                : Average temperature for a season
        daily_diff_mean             (np.ndarray)                : Daily temperature difference average
        season_seq                  (np.ndarray)                : Season sequences
        unique_seasons              (np.ndarray)                : Total unique seasons in the data
        ev_data                     (np.ndarray)                : 2D ev data matrix
    Returns:
        season_proportions          (np.ndarray)                : Proportion of EV for each season
        scaled_season_proportions   (np.ndarray)                : Proportion of EV scaled for each season based on EV
                                                                  repetition for a unqiue season
    """

    season_proportions = []
    scaled_season_proportions = []
    daily_diff_mean = np.asarray(daily_diff_mean)

    # For each season calculate the season proportions

    for i in range(len(season_means)-1):
        season_proportion = np.sum((daily_diff_mean <= season_means[i+1]) &
                                   (daily_diff_mean > season_means[i]))/len(daily_diff_mean)
        season_proportions.append(season_proportion)
        if len(np.where(season_seq[:, 0] == unique_seasons[i])[0]) == 1:
            season_repeat = 1
        else:
            ev_present_each_season = []
            for j in range(len(season_seq)):

                # for each time the season is repeated find out the number of times the ev consumption was found for
                # each season while taking into account vacation information

                if season_seq[j, 0] == unique_seasons[i]:
                    start_day = int(season_seq[j, 1])
                    end_day = int(season_seq[j, 2])
                    ev_present = bool(np.sum(ev_data[start_day: end_day] > 0))
                    ev_presence_percentage = max((ev_present - season_seq[j, 4]), 0)
                    ev_present_each_season.append(ev_presence_percentage)
            season_repeat = np.sum(ev_present_each_season)
        scaled_season_proportions.append(season_proportion*season_repeat)

    return season_proportions, scaled_season_proportions


def ev_year_round_presence(ev_data):
    """
    This funciton is used to derive the proportion of EV present through the year
    Parameters:
        ev_data                 (np.ndarray)            : EV 2D matrix
    Returns:
        ev_year_round_prob      (float)                 : Proportion of EV in the data
    """

    # Initialise the required variables
    start = 0
    days_duration = 30
    end = min(start+days_duration, len(ev_data))

    # For every 30 days calculate the presence of EV in that month
    ev_present_arr = []
    while start <= len(ev_data):
        ev_present = int(bool(np.sum(ev_data[start: end] > 0)))
        ev_present_arr.append(ev_present)
        start = end
        end = min(start+days_duration, len(ev_data))
        if start == len(ev_data):
            break

    # get the year round ev presence probability
    ev_year_round_prob = np.sum(ev_present_arr)/len(ev_present_arr)
    return ev_year_round_prob


def multiseason_check(box_data, ev_config, debug, probability):
    """
    This function is used to perform checks on potential FP cases on users from high HVAC geographies
    Parameters:
        box_data                (np.ndarray)                : EV boxes data
        ev_config               (dict)                      : EV configurations dictionary
        debug                   (dict)                      : Debug object
        probability             (float)                     : EV probability
    Returns:
        seasonal_fp_check       (Boolean)                   : Overall seasonal FP check
        seasonal_var_check      (Boolean)                   : Seasonal count variation and multiple charging check
        seasonal_proportion_var_check (Boolean)             : Seasonal EV proportion variation check
    """

    # Extract required configurations

    frac_multi_charge = debug.get('frac_multi_charge')
    seasonal_var_thr = ev_config.get('seasonal_var_thr')
    year_round_ev_prob_thr = ev_config.get('year_round_ev_prob_thr')
    seasonal_count_variation = debug.get('box_monthly_count_var')
    hld_check_confidence_thr = ev_config.get('hld_check_confidence_thr')
    recent_ev_days_proportion = ev_config.get('recent_ev_days_proportion')
    prior_recent_ev_proportion = ev_config.get('prior_recent_ev_proportion')
    seasonal_proportion_var_thr = ev_config.get('seasonal_proportion_var_thr')
    seasonal_proportion_var_check_2_thr = ev_config.get('seasonal_proportion_var_check_2_thr')

    # Extract the required data

    vacation_data = debug.get('other_output').get('vacation_data')
    box_data = pd.DataFrame(box_data)
    box_data = box_data.ffill()
    box_data = np.asarray(box_data.ffill())
    if vacation_data is None:
        vacation_data = np.zeros(shape=(box_data.shape[0], 1))
    vacation_data = vacation_data[:len(box_data)]
    box_data = np.c_[box_data, vacation_data]

    # get the 2D matrix

    input_data, row_idx, col_idx = get_day_data_2d(box_data, ev_config.get('sampling_rate'))

    # Extract the required data from the input_data matrix

    ev_data = input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]
    feels_like_data = input_data[Cgbdisagg.INPUT_FEELS_LIKE_IDX, :, :]
    s_label_data = input_data[Cgbdisagg.INPUT_S_LABEL_IDX, :, :]
    ev_local_s_label = input_data[-2, :, :]
    vacation_data = input_data[-1, :, :]

    # Identify the set point

    set_point = np.nanmean(np.nanmean(feels_like_data, axis=1))

    # get the array with the difference from the set point

    diff_feels_like_data = feels_like_data - set_point

    # difference of temperature from the setpoint

    norm_diff_feels_like_data = (diff_feels_like_data - np.nanmin(diff_feels_like_data)) /\
                                (np.nanmax(diff_feels_like_data) - np.nanmin(diff_feels_like_data))

    # get the indexes where the ev strikes are present

    strikes_present_idx = norm_diff_feels_like_data * (ev_data > 0)

    # get the temperature differences where the ev strikes are present

    daily_ev_mean, daily_diff_mean = get_daily_ev_mean(strikes_present_idx, ev_data)

    # for each unique season identify the season temperature means

    season_means, daily_seasons, unique_seasons = get_season_means(box_data, norm_diff_feels_like_data,
                                                                   ev_local_s_label, s_label_data)

    # Get the season sequences & format them to avoid continuous repetitions due to vacation

    season_seq = get_season_seq(season_means, daily_seasons)

    # Get the vacation percentage for each season

    vacation_percentage = get_vacation_percentage(season_seq, vacation_data)

    season_seq = np.c_[season_seq, vacation_percentage]

    # sort the seasons based on their temperature means

    season_means = [x for x in season_means if ~np.isnan(x)]
    season_means = np.asarray(season_means)
    sorting_indexes = np.argsort(season_means)
    unique_seasons = unique_seasons[sorting_indexes]
    season_means = season_means[sorting_indexes]
    season_means = np.r_[0, season_means]

    # identifying the proportion of ev strikes in a season

    season_proportions, scaled_season_proportions = get_season_proportions(season_means, daily_diff_mean, season_seq,
                                                                           unique_seasons, ev_data)

    # check for ev presence throughout the year

    ev_year_round_prob = ev_year_round_presence(ev_data)

    # Identify recent EV & update the threshold accordingly

    recent_ev_start_day = int(len(ev_data) - recent_ev_days_proportion * len(ev_data))
    recent_ev_end_day = int(len(ev_data))
    proportion_before_recent_ev = np.sum(np.sum(ev_data[0:recent_ev_start_day, :] > 0, axis=1) > 0)/len(ev_data)
    proportion_after_recent_ev = \
        np.sum(np.sum(ev_data[recent_ev_start_day:recent_ev_end_day, :], axis=1) > 0)/len(ev_data)

    seasonal_var_check = seasonal_proportion_var_check = False

    # Don't perform further checks if the user is a recent EV user

    perform_check = True
    if proportion_before_recent_ev <= prior_recent_ev_proportion and proportion_after_recent_ev >= 0:
        perform_check = False

    # Multiple charging and seasonal count variation check if confidence is less than the threshold

    if probability <= hld_check_confidence_thr and \
            (1-probability)*frac_multi_charge*seasonal_count_variation >= seasonal_var_thr:
        seasonal_var_check = True

    # If the user has EV throughout the year mostly then keep the threshold high

    if ev_year_round_prob >= year_round_ev_prob_thr:
        seasonal_proportion_var_thr = seasonal_proportion_var_check_2_thr

    # Seasonal proportion & Scaled seasonal proportion variation check for users with high probability

    seasonal_proportion_var_check_value = (1-probability)*np.var(season_proportions)*np.var(scaled_season_proportions)
    if seasonal_proportion_var_check_value >= seasonal_proportion_var_thr:
        seasonal_proportion_var_check = True

    # Final FP check status

    seasonal_fp_check = False
    if perform_check and (seasonal_var_check or seasonal_proportion_var_check):
        seasonal_fp_check = True

    return seasonal_fp_check, seasonal_var_check, seasonal_proportion_var_check


def winter_seasonality_check(box_data, ev_config, debug):
    """
    Function to identify highly seasonal device popping up in winters only in NSP
    Parameters:
        box_data                (np.ndarray)            : EV box data identified
        ev_config               (Dict)                  : EV configurations
        debug                   (Dict)                  : Debug dictionary
    Returns:
        winter_device           (Boolean)               : Winter device flag
        debug                   (Dict)                  : Debug dictionary
    """

    # Extract required variables
    min_winter_days = 7
    min_summer_days = 7
    min_winter_ev_strikes = 10
    max_summer_ev_strikes = 3
    winter_proportion_thr = ev_config.get('nsp_winter_seasonality_configs').get('winter_proportion_thr')

    # Get the 2D matrix of the box data

    sampling_rate = ev_config.get('sampling_rate')
    data_matrix, _, _ = get_2d_matrix(box_data, sampling_rate)

    # Get the EV data, Seasons label data

    ev_box_data = data_matrix[Cgbdisagg.INPUT_CONSUMPTION_IDX]
    s_label_data = data_matrix[Cgbdisagg.INPUT_S_LABEL_IDX]
    ev_present_days = (np.sum(ev_box_data, axis=1) > 0).astype(int)

    # Identify the season at day level

    day_season = []
    for i in range(len(s_label_data)):
        values, counts = np.unique(s_label_data[i, :], return_counts=True)
        day_season.append(values[np.argsort(-counts)][0])
    day_season = np.asarray(day_season)

    # Find the sequence of the seasons

    season_seq = find_seq(day_season, min_seq_length=7)

    # Identify the number of days where EV is present for a season

    ev_proportion_for_each_season = []
    for i in range(len(season_seq)):
        ev_proportion = np.sum(ev_present_days[int(season_seq[i, 1]): int(season_seq[i, 2])])
        ev_proportion_for_each_season.append(ev_proportion)
    ev_proportion_for_each_season = np.asarray(ev_proportion_for_each_season)

    season_seq = np.c_[season_seq, ev_proportion_for_each_season]

    # Check for the presence of both Winter & Summer season
    seasons = np.unique(season_seq[:, 0])
    winter_presence = any(val in [-1, -0.5] for val in seasons)
    smr_trans_presence = any(val in [1, 0.5, 0] for val in seasons)

    winter_device = False

    # Check for 2 winters and 1 summer

    if winter_presence and smr_trans_presence:

        # Identify winter indexes and its EV proportion
        wtr_indexes = np.isin(season_seq[:, 0], [-1, -0.5])
        wtr_consumption_percentage = np.sum(season_seq[wtr_indexes, 4])/np.sum(season_seq[:, 4])

        # Check for 2 times Heavy winter in the data & EV usage in both the winters
        heavy_winter_indexes = np.isin(season_seq[:, 0], [-1])
        heavy_winter_indexes = np.logical_and(heavy_winter_indexes, season_seq[:, 3] >= min_winter_days)
        heavy_winter_count = np.sum(heavy_winter_indexes)
        heavy_winter_ev_presence_count = np.sum(season_seq[heavy_winter_indexes, 4] >= min_winter_ev_strikes)

        # Check for Heavy winter usage along with more than 2 winters in the data
        if wtr_consumption_percentage >= winter_proportion_thr and heavy_winter_count >= 2 and \
                heavy_winter_count == heavy_winter_ev_presence_count:
            winter_device = True

    # Check for 2 summers and 1 winter

    if winter_presence and smr_trans_presence:

        # Identify winter indexes and its EV proportion
        wtr_indexes = np.isin(season_seq[:, 0], [-1, -0.5])
        wtr_consumption_percentage = np.sum(season_seq[wtr_indexes, 4]) / np.sum(season_seq[:, 4])

        # Identify the Winter indexes & the corresponding EV proportion
        heavy_winter_indexes = np.isin(season_seq[:, 0], [-1])
        heavy_winter_indexes = np.logical_and(heavy_winter_indexes, season_seq[:, 3] >= min_winter_days)
        heavy_winter_count = np.sum(heavy_winter_indexes)
        heavy_winter_ev_presence_count = np.sum(season_seq[heavy_winter_indexes, 4] >= min_winter_ev_strikes)

        # Identify the Summer indexes & the corresponding EV proportion
        heavy_smr_indexes = np.isin(season_seq[:, 0], [0, 0.5, 1])
        heavy_smr_indexes = np.logical_and(heavy_smr_indexes, season_seq[:, 3] >= min_summer_days)
        heavy_smr_count = np.sum(heavy_smr_indexes)
        heavy_smr_ev_presence_count = np.sum(season_seq[heavy_smr_indexes, 4] <= max_summer_ev_strikes)

        # Check for  1 heavy winter along with more than 2 summers
        wtr_dominance = wtr_consumption_percentage >= winter_proportion_thr and heavy_winter_count == 1 and \
                        heavy_winter_count == heavy_winter_ev_presence_count
        smr_nondominance = heavy_smr_count >= 2 and heavy_smr_ev_presence_count == heavy_smr_count
        wtr_end = ~heavy_winter_indexes[-1]

        if wtr_dominance and smr_nondominance and wtr_end:
            winter_device = True

    return winter_device, debug
