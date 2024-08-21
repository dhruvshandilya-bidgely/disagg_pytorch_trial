"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to get the non-timed thermostat water heater features at bill cycle / monthly level
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.get_seasonal_segments import season_columns
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.thermostat_features import WhFeatures
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.get_consolidated_laps import lap_data_columns


def get_bill_cycle_features(in_data, laps, season_info, peak_index, wh_config, logger):
    """
    Parameters:
        in_data             (np.ndarray)    : Input 21-column matrix
        laps                (np.ndarray)    : Combined laps info
        season_info         (np.ndarray)    : Seasons information
        peak_index          (np.ndarray)    : Indices of single peaks
        wh_config           (dict)          : Config params
        logger              (logger)        : Logger object

    Returns:
        season_info         (np.ndarray)    : Updated season info with features
        single_peak_index   (np.ndarray)    : Indices of thin pulse peaks
    """

    # Check if input data and laps data is not empty

    if (len(in_data) == 0) or (len(laps) == 0):
        # If input data or laps data is empty, return blank output

        # Get the number of columns in the features set

        n_cols = WhFeatures.n_base + WhFeatures.n_features

        # Check if input data was non-empty

        if len(in_data) > 0:
            # If input data present but no thin pulses

            single_peak_index = np.array([False] * in_data.shape[0])
        else:
            # If input data is empty

            single_peak_index = np.array([])

        return np.array([], dtype=np.int64).reshape(0, n_cols), single_peak_index

    # Retrieve the required params from config

    sampling_rate = wh_config['sampling_rate']
    night_bounds = wh_config['thermostat_wh']['detection']['night_bounds']
    lap_half_width = wh_config['thermostat_wh']['detection']['lap_half_width']

    # Full lap width is double of the lap half width

    lap_full_width = lap_half_width * 2

    # Taking a deepcopy of input data to keep local instances

    input_data = deepcopy(in_data)

    # Extract the lap boolean column

    lap_index = input_data[:, Cgbdisagg.INPUT_DIMENSION]

    # Taking indices that are single point peaks and within laps

    single_peak_index = np.logical_and(lap_index > 0, peak_index)

    # Number of pre-existing columns and number of bill cycles

    n = WhFeatures.n_base

    n_bill_cycles = season_info.shape[0]

    # Append each new feature column to seasons info (contains row per bill cycle / month)

    season_info = np.hstack((season_info, np.zeros((n_bill_cycles, WhFeatures.n_features))))

    # Calculate peak factor

    pf, pf_std, two_peak_lap_count = lap_peak_factor(single_peak_index, lap_full_width, sampling_rate)

    # Assign values to the seasons info

    season_info[:, n + WhFeatures.PEAK_FACTOR] = pf
    season_info[:, n + WhFeatures.PEAK_FACTOR_STD] = pf_std
    season_info[:, n + WhFeatures.TWO_PEAKS_LAP_COUNT] = two_peak_lap_count

    # Get consistency factor

    season_info[:, n + WhFeatures.CONSISTENCY] = consistency_factor(input_data, single_peak_index, wh_config)

    # Get valid lap fraction

    season_info[:, n + WhFeatures.VALID_LAP_DAYS] = valid_lap_fraction(input_data, single_peak_index)

    # Get indices of the night hours and day hours

    night_time_index = np.logical_or(input_data[:, Cgbdisagg.INPUT_HOD_IDX] >= night_bounds[0],
                                     input_data[:, Cgbdisagg.INPUT_HOD_IDX] < night_bounds[1])

    day_time_index = ~night_time_index

    # Iterate over each bill cycle / month

    for i in range(n_bill_cycles):
        # Extract feature of current bill cycle / month

        month = season_info[i, season_columns['bill_cycle_ts']]

        # Get current month peaks indices

        month_idx = (input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == month)
        monthly_peaks_idx = np.logical_and(single_peak_index, month_idx)

        # Get current month laps info

        monthly_laps = laps[laps[:, lap_data_columns['month_ts']] == month, :]

        # Separating peaks for day and night in this month

        day_time_peaks_idx = np.logical_and(monthly_peaks_idx, day_time_index)
        night_time_peaks_idx = np.logical_and(monthly_peaks_idx, night_time_index)

        # Using baseload removed data for peaks energy in this month

        monthly_peaks_energy = input_data[monthly_peaks_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]
        monthly_lap_hours = np.sum(laps[laps[:, lap_data_columns['month_ts']] == month, lap_data_columns['duration']])

        # Calculate the time difference between peaks in this month (in hours)

        monthly_peaks_time_diff = np.diff(input_data[monthly_peaks_idx, Cgbdisagg.INPUT_EPOCH_IDX])

        # Calculate the time difference between peaks within same lap

        lap_window_size = lap_full_width * Cgbdisagg.SEC_IN_HOUR
        monthly_peaks_time_diff = monthly_peaks_time_diff[monthly_peaks_time_diff < lap_window_size]

        # Add features to season info

        if (np.sum(monthly_peaks_idx) != 0) and (len(monthly_laps) != 0):
            # If valid month data and laps, add features

            season_info[i, n + WhFeatures.ENERGY] = np.round(np.mean(monthly_peaks_energy), 2)
            season_info[i, n + WhFeatures.ENERGY_STD] = np.round(np.std(monthly_peaks_energy), 2)

            season_info[i, n + WhFeatures.COUNT_PEAKS] = len(monthly_peaks_energy)
            season_info[i, n + WhFeatures.PEAKS_PER_LAP] = len(monthly_peaks_energy) / monthly_laps.shape[0]

            season_info[i, n + WhFeatures.DAY_PEAKS] = np.sum(day_time_peaks_idx)
            season_info[i, n + WhFeatures.NIGHT_PEAKS] = np.sum(night_time_peaks_idx)

            season_info[i, n + WhFeatures.PEAKS_PER_HOUR] = np.round(season_info[i, n + WhFeatures.COUNT_PEAKS] /
                                                                     monthly_lap_hours, 3)

            season_info[i, n + WhFeatures.PEAK_DAYS_PER_MONTH] = peaks_per_month(input_data,
                                                                                 monthly_peaks_idx, month_idx)

            season_info[i, n + WhFeatures.NUM_DAYS] = len(np.unique(input_data[month_idx, Cgbdisagg.INPUT_DAY_IDX]))

        else:
            # if month or lap data empty

            logger.info('No features for bill cycle: | {}'.format(month))

        # Add the count of laps feature

        if len(monthly_laps) != 0:
            season_info[i, n + WhFeatures.COUNT_LAPS] = monthly_laps.shape[0]

        # Add features from time difference between peaks (if more than one peak)

        if len(monthly_peaks_time_diff) > 1:
            season_info[i, n + WhFeatures.MIN_TIME_DIFF] = np.min(monthly_peaks_time_diff)
            season_info[i, n + WhFeatures.MAX_TIME_DIFF] = np.max(monthly_peaks_time_diff)
            season_info[i, n + WhFeatures.MEDIAN_TIME_DIFF] = np.median(monthly_peaks_time_diff)

    return season_info, single_peak_index


def consistency_factor(input_data, single_peak_index, wh_config):
    """
    Parameters:
        input_data              (np.ndarray)    : Input 21-column matrix
        single_peak_index       (np.ndarray)    : Indices of thin pulse peaks
        wh_config               (np.ndarray)    : Water heater config params

    Returns:
        consistency_score       (float)         : Consistency factor
    """

    # Get the required params from config

    allowed_days_gap = wh_config['thermostat_wh']['detection']['allowed_days_gap']

    # Get the daily peaks count

    unq_days, days_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
    daily_peaks_count = np.bincount(days_idx, single_peak_index)

    # Select days with non-zero peaks

    peak_days_idx = np.where(daily_peaks_count > 0)[0]

    # Check the gap between peak days

    if len(peak_days_idx) > 1:
        # If multiple days found with peaks, find gap between peaks in days

        days_gap = np.abs(np.diff(peak_days_idx))

        # Filter out gaps above 10 days (Can be changed to 7 (1-week))

        valid_days_gap = days_gap[days_gap <= allowed_days_gap]

        # If valid data gapd left, find the score, else score zero

        if len(valid_days_gap) > 0:
            consistency_score = len(valid_days_gap) / np.sum(valid_days_gap)
        else:
            consistency_score = 0
    else:
        # If no day found with peaks, score zero

        consistency_score = 0

    consistency_score = np.round(consistency_score, 4)

    return consistency_score


def peaks_per_month(input_data, monthly_peaks_idx, month_idx):
    """
    Parameters:
        input_data              (np.ndarray)    : Input 21-column matrix
        monthly_peaks_idx       (np.ndarray)    : Current month thin pulse indices
        month_idx               (np.ndarray)    : Indices of current month

    Returns:
         peak_days              (float)         : Fraction of days with peaks
    """

    # Get daily peaks count

    unq_days, days_idx = np.unique(input_data[month_idx, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
    daily_peaks_count = np.bincount(days_idx, monthly_peaks_idx[month_idx])

    # Pick the days with non-zero number of peaks and calculate fraction

    peak_days = len(np.where(daily_peaks_count > 0)[0]) / len(unq_days)

    return peak_days


def valid_lap_fraction(input_data, single_peak_index):
    """
    Parameters:
        input_data              (np.ndarray)    : Input 21-column data matrix
        single_peak_index       (np.ndarray)    : Indices of thin pulse peaks

    Returns:
        lap_fraction            (float)         : Fraction of laps with peaks
    """

    # Get the indices of laps

    lap_index = input_data[:, Cgbdisagg.INPUT_DIMENSION] > 0

    # Get number of days with laps

    lap_days = len(np.unique(input_data[lap_index, Cgbdisagg.INPUT_DAY_IDX]))

    # Get number of days with peaks

    peak_days = len(np.unique(input_data[single_peak_index, Cgbdisagg.INPUT_DAY_IDX]))

    # Fraction of days with peaks and laps is the lap fraction

    lap_fraction = peak_days / lap_days

    return lap_fraction


def lap_peak_factor(single_peak_index, lap_window_size, sampling_rate):
    """
    Parameters:
        single_peak_index       (np.ndarray)    : Indices of thin pulses
        lap_window_size         (int)           : Window size of lap
        sampling_rate           (int)           : Sampling rate of data

    Returns:
        peak_factor             (float)         : Peak factor
        peak_factor_std         (float)         : Std of peak factor
        two_peak_lap_count      (int)           : Count of multiple peak laps
    """

    # Calculate the time difference between single peaks in hours

    time_diff_idx = np.diff(np.where(single_peak_index))[0]
    time_diff_hours = time_diff_idx * sampling_rate / Cgbdisagg.SEC_IN_HOUR

    # Select peaks that have gap less than lap window size (common lap)

    time_diff_hours = time_diff_hours[time_diff_hours <= lap_window_size]

    # Count the number of such laps

    two_peak_lap_count = len(time_diff_hours)

    # Calculate peak factor

    if two_peak_lap_count > 0:
        # If non-zero number of laps with multiple peaks

        peak_factor = np.round(1 / np.mean(time_diff_hours), 3)
    else:
        # If no laps with multiple peaks

        peak_factor = 0

    # Calculate the peak factor standard deviation

    if len(time_diff_hours) > 1:
        # If more than one lap with multiple peaks, get std value

        peak_factor_std = np.std(time_diff_hours)
    else:
        # If less than two lap with multiple peaks, default std value

        peak_factor_std = -1

    return peak_factor, peak_factor_std, two_peak_lap_count
