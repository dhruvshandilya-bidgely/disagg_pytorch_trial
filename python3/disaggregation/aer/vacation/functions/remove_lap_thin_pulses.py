"""
Author - Mayank Sharan
Date - 1/12/19
Remove thin WH pulses from the input data in Low Activity Period (LAP) regions
"""

# Import python packages

import copy
import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def remove_lap_thin_pulses(day_data_masked, day_peak_removal, day_peak_amp_arr, vacation_config, logger_pass):

    """
    Analyses the thin pulses in LAPs identified and removes thin pulses that qualify set criteria

    Parameters:
        day_data_masked         (np.ndarray)        : Day wise array with masked data
        day_peak_removal        (np.ndarray)        : Day wise boolean array with location of each peak
        day_peak_amp_arr        (np.ndarray)        : Day wise single peak amplitude array
        vacation_config         (dict)              : Contains parameters needed for vacation detection
        logger_pass             (dict)              : Contains the logger and the logging dictionary to be passed on

    Returns:
        day_data_processed      (np.ndarray)        : Day wise array with processed data
        vacation_config         (dict)              : Contains parameters needed for vacation detection
    """

    # Initialize logger

    logger_base = logger_pass.get('base_logger').getChild('remove_lap_thin_pulses')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Initialize day_data_processed to prepare output

    day_data_processed = copy.deepcopy(day_data_masked)

    # Initialize the config to be used

    thin_pulse_rem_config = vacation_config.get('thin_pulse_rem')
    pd_mult = Cgbdisagg.SEC_IN_HOUR / vacation_config.get('user_info').get('sampling_rate')

    # Reject days for wh removal where number of points is below the threshold

    num_potential_points_day = np.sum(day_peak_removal, axis=1)
    potential_days_for_peak_removal = np.sign(num_potential_points_day).astype(bool)

    logger.info('Number of days with potential peaks | %d', np.sum(potential_days_for_peak_removal))

    min_count_bool = num_potential_points_day < thin_pulse_rem_config.get('min_day_pts_for_removal')
    max_count_bool = num_potential_points_day > thin_pulse_rem_config.get('max_day_pts_for_removal')

    rejected_days_bool = np.logical_and(np.logical_or(min_count_bool, max_count_bool), potential_days_for_peak_removal)

    day_peak_removal[rejected_days_bool, :] = False
    potential_days_for_peak_removal[rejected_days_bool] = False

    logger.info('Number of days with invalid number of peaks | %d', np.sum(rejected_days_bool))

    # Reject days for wh removal where consistency check fails

    day_peak_location_arr = np.full(shape=(day_peak_removal.shape[0],
                                           thin_pulse_rem_config.get('max_day_pts_for_removal')), fill_value=np.nan)

    peak_location_row, peak_location_col = np.where(day_peak_removal)

    # If no peaks remain return the data with no peaks removed

    if len(peak_location_row) == 0:

        logger.info('Number of selected points for peak removal | 0')
        logger.info('User flag for thin pulse removal | False')

        return day_data_processed, vacation_config

    # Compute the sequence id of each peak within the day

    # Locate the start indices corresponding to each day

    seq_comp_arr = np.r_[1, np.diff(peak_location_row)]
    seq_comp_arr = seq_comp_arr.astype(float)

    pos_diff_idx = seq_comp_arr > 0
    pos_val_to_set = np.arange(1, np.sum(pos_diff_idx) + 1)

    # Modify start value at each day into sequential numbers as a sequential day idx as in 1, 2, 3, 4

    seq_comp_arr[pos_diff_idx] = pos_val_to_set
    seq_comp_arr[np.logical_not(pos_diff_idx)] = np.nan

    # Spread the sequence index of each day across all peak points corresponding to that day. Also used for division

    div_arr = np.fmax.accumulate(seq_comp_arr)

    # Take cumulative sum of this array to introduce sequential nature for index computation

    cum_sum_arr = np.cumsum(div_arr)

    # Compute array for subtraction from cumulative sum to extract multiple of index values

    pos_idx_arr = np.where(pos_diff_idx)[0]
    sub_arr = np.full(shape=seq_comp_arr.shape, fill_value=0)

    if len(pos_idx_arr) > 1:
        sub_arr[pos_idx_arr[1:]] = cum_sum_arr[(pos_idx_arr - 1)[1:]]
        sub_arr = np.fmax.accumulate(sub_arr)

    # Compute corresponding column index by modifying the cumulative sum array

    peak_day_seq_idx = np.divide(cum_sum_arr - sub_arr, div_arr) - 1
    peak_day_seq_idx = peak_day_seq_idx.astype(int)

    # Assign column index values to peaks in peak array using computed indices

    day_peak_location_arr[peak_location_row, peak_day_seq_idx] = peak_location_col

    # Using day peak location arr compute day level consistency of peaks and reject days

    potential_days_peak_location_arr = day_peak_location_arr[potential_days_for_peak_removal, :]
    day_peak_diff_arr = np.diff(potential_days_peak_location_arr, axis=1) / pd_mult

    # Compute consistency metrics to base rejections on

    day_peak_max_diff = np.nanmax(day_peak_diff_arr, axis=1)
    day_peak_med_diff = np.nanmedian(day_peak_diff_arr, axis=1)

    # Reject days based on consistency of peaks

    stage_2_rej_bool_1 = day_peak_max_diff > thin_pulse_rem_config.get('max_peak_diff_hrs')
    stage_2_rej_bool_2 = day_peak_med_diff < thin_pulse_rem_config.get('med_peak_diff_low_thr')
    stage_2_rej_bool_3 = day_peak_med_diff > thin_pulse_rem_config.get('med_peak_diff_high_thr')

    stage_2_rej_days_bool = np.logical_or(stage_2_rej_bool_1, np.logical_or(stage_2_rej_bool_2, stage_2_rej_bool_3))

    potential_days_for_peak_removal_idx = np.where(potential_days_for_peak_removal)[0]
    stage_2_rej_days_idx = potential_days_for_peak_removal_idx[stage_2_rej_days_bool]

    day_peak_removal[stage_2_rej_days_idx, :] = False
    potential_days_for_peak_removal[stage_2_rej_days_idx] = False

    logger.info('Number of days with inconsistent peaks invalid for removal | %d', len(stage_2_rej_days_idx))

    # Log information about selected peaks to be removed

    num_days_peak_removal = np.sum(potential_days_for_peak_removal)
    num_peaks_for_removal = np.sum(day_peak_removal)

    logger.info('Number of selected days for peak removal | %d', num_days_peak_removal)
    logger.info('Number of selected points for peak removal | %d', num_peaks_for_removal)

    day_data_processed[day_peak_removal] = day_data_processed[day_peak_removal] - day_peak_amp_arr[day_peak_removal]

    # Based on number of days and number of points of removal decide if we need to mark this as a wh user

    if num_days_peak_removal >= thin_pulse_rem_config.get('min_days_for_removal') \
            and num_peaks_for_removal >= thin_pulse_rem_config.get('min_peaks_for_removal'):

        logger.info('User flag for thin pulse removal | True')
        vacation_config['user_info']['thin_pulse_rem'] = True
    else:

        logger.info('User flag for thin pulse removal | False')

    return day_data_processed, vacation_config
