"""
Author: Mayank Sharan
Created: 12-Jul-2020
Utility functions to help in detection season
"""

# Import python packages

import pytz
import copy
import numpy as np

from datetime import datetime

# Import project functions and classes

from python3.disaggregation.aer.waterheater.weather_data_analytics.nbi_constants import TimeConstants


def check_trn_smr_btwn_smr(previous_seq, middle_seq, future_seq, correcting_idx, correcting_seq):
    """
    This function is used to check if the intersection is Transition Summer sandwiched between 2 Summers, if so
    then update the season label accordingly
    Args:
        previous_seq            (np.ndarray)        : Sequence information of the previous year's chunk
        middle_seq              (np.ndarray)        : Sequence information of the intersecting future year's chunk
        future_seq              (np.ndarray)        : Sequence information of the future year's chunk
        correcting_idx          (int)               : Index where the season label needs to be corrected
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    Returns:
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    """

    if previous_seq[0] == 1 and middle_seq[0] == 0.5 and future_seq[0] == 1 and (middle_seq[3] <= 25):
        correcting_seq[correcting_idx, 0] = 1

    return correcting_seq


def check_trn_wtr_btwn_wtr(previous_seq, middle_seq, future_seq, correcting_idx, correcting_seq):
    """
    This function is used to check if the intersection is Transition Winter sandwiched between 2 Winters, if so
    then update the season label accordingly
    Args:
        previous_seq            (np.ndarray)        : Sequence information of the previous year's chunk
        middle_seq              (np.ndarray)        : Sequence information of the intersecting future year's chunk
        future_seq              (np.ndarray)        : Sequence information of the future year's chunk
        correcting_idx          (int)               : Index where the season label needs to be corrected
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    Returns:
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    """

    if previous_seq[0] == -1 and middle_seq[0] == -0.5 and future_seq[0] == -1 and (middle_seq[3] <= 25):
        correcting_seq[correcting_idx, 0] = -1

    return correcting_seq


def check_trn_btwn_trn_smr(previous_seq, middle_seq, future_seq, correcting_idx, correcting_seq):
    """
    This function is used to check if the intersection is Transition sandwiched between 2 Transition Summers, if so
    then update the season label accordingly
    Args:
        previous_seq            (np.ndarray)        : Sequence information of the previous year's chunk
        middle_seq              (np.ndarray)        : Sequence information of the intersecting future year's chunk
        future_seq              (np.ndarray)        : Sequence information of the future year's chunk
        correcting_idx          (int)               : Index where the season label needs to be corrected
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    Returns:
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    """

    if previous_seq[0] == 0.5 and middle_seq[0] == 0 and future_seq[0] == 0.5 and (middle_seq[3] <= 25):
        correcting_seq[correcting_idx, 0] = 0.5

    return correcting_seq


def check_trn_btwn_trn_wtr(previous_seq, middle_seq, future_seq, correcting_idx, correcting_seq):
    """
    This function is used to check if the intersection is Transition sandwiched between 2 Transition Winter, if so
    then update the season label accordingly
    Args:
        previous_seq            (np.ndarray)        : Sequence information of the previous year's chunk
        middle_seq              (np.ndarray)        : Sequence information of the intersecting future year's chunk
        future_seq              (np.ndarray)        : Sequence information of the future year's chunk
        correcting_idx          (int)               : Index where the season label needs to be corrected
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    Returns:
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    """

    if previous_seq[0] == -0.5 and middle_seq[0] == 0 and future_seq[0] == -0.5 and (middle_seq[3] <= 25):
        correcting_seq[correcting_idx, 0] = -0.5

    return correcting_seq


def check_trn_smr_btwn_trn(previous_seq, middle_seq, future_seq, correcting_idx, correcting_seq):
    """
    This function is used to check if the intersection is Transition Summer sandwiched between 2 Transitions, if so
    then update the season label accordingly
    Args:
        previous_seq            (np.ndarray)        : Sequence information of the previous year's chunk
        middle_seq              (np.ndarray)        : Sequence information of the intersecting future year's chunk
        future_seq              (np.ndarray)        : Sequence information of the future year's chunk
        correcting_idx          (int)               : Index where the season label needs to be corrected
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    Returns:
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    """

    if previous_seq[0] == 0 and middle_seq[0] == 0.5 and future_seq[0] == 0 and (middle_seq[3] <= 25):
        correcting_seq[correcting_idx, 0] = 0

    return correcting_seq


def check_trn_wtr_btwn_trn(previous_seq, middle_seq, future_seq, correcting_idx, correcting_seq):
    """
    This function is used to check if the intersection is Transition Winter sandwiched between 2 Transitions, if so
    then update the season label accordingly
    Args:
        previous_seq            (np.ndarray)        : Sequence information of the previous year's chunk
        middle_seq              (np.ndarray)        : Sequence information of the intersecting future year's chunk
        future_seq              (np.ndarray)        : Sequence information of the future year's chunk
        correcting_idx          (int)               : Index where the season label needs to be corrected
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    Returns:
        correcting_seq          (np.ndaray)         : Sequence where the season label needs to be updated
    """

    if previous_seq[0] == 0 and middle_seq[0] == -0.5 and future_seq[0] == 0 and (middle_seq[3] <= 25):
        correcting_seq[correcting_idx, 0] = 0

    return correcting_seq


def chunk_weather_data(day_ts):

    """
    Chunk weather data to allow for proper season detection
    Parameters:
        day_ts                  (np.ndarray)    : Array containing timestamps for start of each day in weather data
    Returns:
        season_dict             (dict)          : Dictionary containing season detection data
    """

    # Initialize variables needed

    days_left = len(day_ts)
    chunk_arr = []

    while days_left > 0:

        # If more than 10 months of data is available we can fit GMM
        # Otherwise the chunk will use the GMM from the nearest future chunk
        # Ensure type 1 chunks are at least 15 days long so that there are no errors

        if (days_left > 10 * TimeConstants.max_days_in_1_month) and (days_left < TimeConstants.days_in_1_year +
                                                                     (TimeConstants.max_days_in_1_month / 2)):

            chunk_arr.append([0, days_left, days_left, 0])
            days_left = 0

        elif days_left > 10 * TimeConstants.max_days_in_1_month:

            chunk_start_idx = max(days_left - TimeConstants.days_in_1_year, 0)
            chunk_arr.append([chunk_start_idx, days_left, days_left - chunk_start_idx, 0])
            days_left = chunk_start_idx

        else:

            chunk_arr.append([0, days_left, days_left, 1])
            days_left = 0

    chunk_arr = np.array(chunk_arr)

    return chunk_arr


def adjust_for_global_warming(perc_smr_spr_prec, annual_avg_temp_c):
    """
    This function is used to derive Type B precipitation threshold adjusted to Global warming
    Args:
        perc_smr_spr_prec           (float):        Summer Precipitation percentage
        annual_avg_temp_c           (float):        Annual Type C average temperature
    Returns:
        type_b_prec_thr             (float):        Type B precipitation threshold
    """

    if perc_smr_spr_prec >= 70:
        type_b_prec_thr = 20 * annual_avg_temp_c + 250
    elif perc_smr_spr_prec >= 30:
        type_b_prec_thr = 20 * annual_avg_temp_c + 120
    else:
        type_b_prec_thr = 20 * annual_avg_temp_c

    return type_b_prec_thr


def identify_koppen_class(day_wise_data_dict, meta_data, chunk_info):

    """
    Identify koppen class for the user
    Parameters:
        day_wise_data_dict      (dict)          : Dictionary containing all day wise data matrices
        meta_data               (dict)          : Dictionary containing meta data about the user
        chunk_info              (np.ndarray)    : Information about the latest 1 year chunk to be used
    Returns:
        class_name              (str)           : The string representing the koppen class the user belongs too
    """

    # Extract weather data and day timestamps for the chunk

    start_idx = chunk_info[0]
    end_idx = chunk_info[1]
    num_days = chunk_info[2]

    day_temp_data = day_wise_data_dict.get('temp')[start_idx:end_idx, :]
    day_snow_data = day_wise_data_dict.get('snow')[start_idx:end_idx, :]
    day_prec_data = day_wise_data_dict.get('prec')[start_idx:end_idx, :]
    day_ts = day_wise_data_dict.get('day_ts')[start_idx:end_idx]

    tz = pytz.timezone(meta_data.get('timezone', 'UTC'))

    # Assign a month to each day

    day_month = []

    for day_idx in range(num_days):

        day_dt_time = datetime.fromtimestamp(day_ts[day_idx], tz)
        day_month.append(day_dt_time.month)

    day_month = np.array(day_month)

    # Get temperature and precipitation values for each calendar month

    avg_temp = []
    avg_prec = []
    days_count = []

    daily_temp_avg = np.nanmean(day_temp_data, axis=1)

    for month_idx in range(1, 13):
        month_bool = day_month == month_idx

        avg_temp.append(np.nanmean(daily_temp_avg[month_bool]))
        avg_prec.append(np.sum(day_prec_data[month_bool, :]) + np.sum(day_snow_data[month_bool, :]))
        days_count.append(int(np.sum(month_bool)))

    avg_data = np.c_[avg_temp, avg_prec, days_count]

    class_name = None

    # Initialize thresholds for koppen classification. Adjusted for global warming

    type_a_temp_thr = 66.2

    type_b_hot_thr = 64.4

    type_c_cold_temp_min = 32
    type_c_cold_temp_max = 66.2

    type_c_hot_temp_min = 50

    type_d_cold_temp_max = 32
    type_d_hot_temp_min = 50

    type_e_max_temp = 50

    # Attempt Koppen-Geiger classification class A

    if np.sum(avg_data[:, 0] >= type_a_temp_thr) == 12:
        class_name = 'A'

    # Attempt Koppen-Geiger classification class B

    annual_temp_avg = np.sum(np.multiply(avg_data[:, 0], avg_data[:, 2])) / np.sum(avg_data[:, 2])
    annual_avg_temp_c = (annual_temp_avg - 32) * 5 / 9

    # Calculate the percentage of precipitation in 6 warmest months of the year

    months_considered = np.argsort(avg_data[:, 0])[-6:]
    smr_spr_bool = np.full(shape=(12,), fill_value=False)
    smr_spr_bool[months_considered] = True

    smr_spr_prec = np.sum(avg_data[smr_spr_bool, 2])
    total_prec = np.sum(avg_data[:, 2])

    # calculate the precipitation in summer

    perc_smr_spr_prec = smr_spr_prec * 100 / total_prec

    # Adjusted for global warming

    type_b_prec_thr = adjust_for_global_warming(perc_smr_spr_prec, annual_avg_temp_c)

    if total_prec * 25.4 <= type_b_prec_thr and class_name is None:
        if annual_temp_avg >= type_b_hot_thr:
            class_name = 'Bh'
        else:
            class_name = 'Bk'

    # Attempt Koppen-Geiger classification class C

    coldest_month_avg_temp = np.min(avg_data[:, 0])
    hottest_month_avg_temp = np.max(avg_data[:, 0])

    if (type_c_cold_temp_max >= coldest_month_avg_temp >= type_c_cold_temp_min) and (
            hottest_month_avg_temp >= type_c_hot_temp_min) and class_name is None:
        class_name = 'C'

    # Sub classify class C

    if class_name == 'C':

        c_sub_temp_thr = 65
        perc_thr = 0.7

        perc_above_thr = float(np.sum(daily_temp_avg > c_sub_temp_thr)) / num_days
        perc_below_thr = float(np.sum(daily_temp_avg <= c_sub_temp_thr)) / num_days

        if perc_above_thr > perc_thr:
            class_name = 'Ch'
        elif perc_below_thr > perc_thr:
            class_name = 'Ck'

    # Attempt Koppen-Geiger classification class D

    if (coldest_month_avg_temp <= type_d_cold_temp_max) and (
            hottest_month_avg_temp >= type_d_hot_temp_min) and class_name is None:
        class_name = 'D'

    # Attempt Koppen-Geiger classification class E

    if np.sum(avg_data[:, 0] <= type_e_max_temp) == 12 and class_name is None:
        class_name = 'E'

    return class_name


def merge_chunks(chunk_idx, chunk_info, season_dict, chunk_season_dict, num_days_data):

    """
    Merge chunks coming from season detection
    Parameters:
        chunk_idx               (int)           : Index of the chunk that was currently processed
        chunk_info              (np.ndarray)    : Information about the latest 1 year chunk to be used
        season_dict             (dict)          : Dictionary containing season detection data
        chunk_season_dict       (dict)          : Dictionary containing chunk season detection data
        num_days_data           (int)           : Number of days with data
    Returns:
        season_dict             (dict)          : Dictionary containing season detection data
    """

    chunk_season_dict = copy.deepcopy(chunk_season_dict)

    # If the processing is on the first chunk initialize the season dictionary

    if chunk_idx == 0:

        chunk_start_idx = chunk_info[0]
        chunk_end_idx = chunk_info[1]

        # Populate season label

        s_label_overall = np.full(shape=(num_days_data,), fill_value=np.nan)
        s_label_overall[chunk_start_idx:chunk_end_idx] = chunk_season_dict.get('s_label')

        # Populate hot event bool

        is_hot_event_bool = np.full(shape=(num_days_data,), fill_value=np.nan)
        is_hot_event_bool[chunk_start_idx:chunk_end_idx] = chunk_season_dict.get('is_hot_event_bool')

        # Populate cold event bool

        is_cold_event_bool = np.full(shape=(num_days_data,), fill_value=np.nan)
        is_cold_event_bool[chunk_start_idx:chunk_end_idx] = chunk_season_dict.get('is_cold_event_bool')

        # Populate seq arr

        seq_arr = chunk_season_dict.get('seq_arr')
        seq_arr[:, 1:3] = seq_arr[:, 1:3] + chunk_start_idx

        # Populate the season dict

        season_dict = {
            's_label': s_label_overall,
            'seq_arr': seq_arr,
            'is_hot_event_bool': is_hot_event_bool,
            'is_cold_event_bool': is_cold_event_bool,
            'max_winter_temp': chunk_season_dict.get('max_winter_temp'),
            'max_tr_temp': chunk_season_dict.get('max_tr_temp'),
            'model_info_dict': chunk_season_dict.get('model_info_dict')
        }

    else:

        chunk_start_idx = chunk_info[0]
        chunk_end_idx = chunk_info[1]

        # Populate season label

        s_label_overall = season_dict.get('s_label')
        s_label_overall[chunk_start_idx:chunk_end_idx] = chunk_season_dict.get('s_label')
        season_dict['s_label'] = s_label_overall

        # Populate hot event bool

        is_hot_event_bool = season_dict.get('is_hot_event_bool')
        is_hot_event_bool[chunk_start_idx:chunk_end_idx] = chunk_season_dict.get('is_hot_event_bool')
        season_dict['is_hot_event_bool'] = is_hot_event_bool

        # Populate cold event bool

        is_cold_event_bool = season_dict.get('is_cold_event_bool')
        is_cold_event_bool[chunk_start_idx:chunk_end_idx] = chunk_season_dict.get('is_cold_event_bool')
        season_dict['is_cold_event_bool'] = is_cold_event_bool

        # Merge the sequence array

        chunk_seq_arr = chunk_season_dict.get('seq_arr')
        seq_arr = season_dict.get('seq_arr')

        future_seq_1 = seq_arr[0, :]
        past_seq_1 = chunk_seq_arr[-1, :]

        # If we have only one chunk available in the season chunk then check w.r.t that

        if chunk_seq_arr.shape[0] == 1 and seq_arr.shape[0] > 1:

            future_seq_2 = seq_arr[1, :]

            seq_arr = check_trn_smr_btwn_smr(past_seq_1, future_seq_1, future_seq_2, 0, seq_arr)

            seq_arr = check_trn_wtr_btwn_wtr(past_seq_1, future_seq_1, future_seq_2, 0, seq_arr)

            seq_arr = check_trn_btwn_trn_smr(past_seq_1, future_seq_1, future_seq_2, 0, seq_arr)

            seq_arr = check_trn_btwn_trn_wtr(past_seq_1, future_seq_1, future_seq_2, 0, seq_arr)

            seq_arr = check_trn_smr_btwn_trn(past_seq_1, future_seq_1, future_seq_2, 0, seq_arr)

            seq_arr = check_trn_wtr_btwn_trn(past_seq_1, future_seq_1, future_seq_2, 0, seq_arr)

        elif chunk_seq_arr.shape[0] > 1:

            past_seq_2 = chunk_seq_arr[-2, :]

            chunk_seq_arr = check_trn_smr_btwn_smr(past_seq_2, past_seq_1, future_seq_1, -1, chunk_seq_arr)

            chunk_seq_arr = check_trn_wtr_btwn_wtr(past_seq_2, past_seq_1, future_seq_1, -1, chunk_seq_arr)

            chunk_seq_arr = check_trn_btwn_trn_smr(past_seq_2, past_seq_1, future_seq_1, -1, chunk_seq_arr)

            chunk_seq_arr = check_trn_btwn_trn_wtr(past_seq_2, past_seq_1, future_seq_1, -1, chunk_seq_arr)

            chunk_seq_arr = check_trn_smr_btwn_trn(past_seq_2, past_seq_1, future_seq_1, -1, chunk_seq_arr)

            chunk_seq_arr = check_trn_wtr_btwn_trn(past_seq_2, past_seq_1, future_seq_1, -1, chunk_seq_arr)

        chunk_seq_arr[:, 1:3] = chunk_seq_arr[:, 1:3] + chunk_start_idx
        season_dict['seq_arr'] = np.r_[chunk_seq_arr, seq_arr]

    return season_dict
