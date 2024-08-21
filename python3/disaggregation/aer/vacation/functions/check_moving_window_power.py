"""
Author - Mayank Sharan
Date - 11/12/19
Perform moving window power checks on probable vacation days for further selection
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def check_moving_window_power(day_data, day_wise_baseload, day_valid_mask_cons, probable_vac_bool, loop_break_idx,
                              loop_break_power, vacation_config):

    """
    Performs power checks on a moving window across the day to evaluate days for further selection

    Parameters:
        day_data                (np.ndarray)        : Day wise data matrix
        day_wise_baseload       (np.ndarray)        : Baseload computed corresponding to each day
        day_valid_mask_cons     (np.ndarray)        : Day wise boolean array of points masked as timed device usage
        probable_vac_bool       (np.ndarray)        : Boolean array marking days which are selected as probable
        loop_break_idx          (np.ndarray)        : Array containing the index at which the loop rejected the day
        loop_break_power        (np.ndarray)        : Array containing the power values that caused the loop break
        vacation_config         (dict)              : Contains all configuration variables needed for vacation

    Returns:
        power_check_passed_bool (np.ndarray)        : Boolean array denoting days that passed the power check
        sliding_power           (np.ndarray)        : Array containing window wise power till the day was in the loop
        sliding_power_bkp       (np.ndarray)        : Array containing window wise power for all days
        loop_break_idx          (np.ndarray)        : Array containing the index at which the loop rejected the day
        loop_break_power        (np.ndarray)        : Array containing the power values that caused the loop break
        power_chk_thr           (np.ndarray)        : Array containing the thresholds used to validate windows
    """

    # Initialize variables to be used

    num_days = day_data.shape[0]
    num_pts_day = day_data.shape[1]

    power_check_config = vacation_config.get('window_power_check')
    sampling_rate = vacation_config.get('user_info').get('sampling_rate')

    # Initialize the thresholds allowed for the moving power window

    power_chk_thr = np.full(shape=(num_days,), fill_value=np.nan)

    # Identify days in different baseload range values for assigning window thresholds

    lv_1 = day_wise_baseload < power_check_config.get('bl_lv_1')

    lv_2 = np.logical_and(day_wise_baseload >= power_check_config.get('bl_lv_1'),
                          day_wise_baseload < power_check_config.get('bl_lv_2'))

    lv_3 = np.logical_and(day_wise_baseload >= power_check_config.get('bl_lv_2'),
                          day_wise_baseload < power_check_config.get('bl_lv_3'))

    lv_4 = day_wise_baseload >= power_check_config.get('bl_lv_3')

    # Assign power check thresholds as per classification

    slope = power_check_config.get('bl_lv_4_slope')

    if vacation_config.get('user_info').get('thin_pulse_rem'):

        # Assign thresholds designed for users with thin pulse removal

        power_chk_thr[lv_1] = power_check_config.get('bl_lv_1_thr_tp')
        power_chk_thr[lv_2] = power_check_config.get('bl_lv_2_thr_tp')
        power_chk_thr[lv_3] = power_check_config.get('bl_lv_3_thr_tp')

        # For days in level 4 create threshold that grows linearly

        lv_3_thr = power_check_config.get('bl_lv_3_thr_tp')
        power_chk_thr[lv_4] = slope * (day_wise_baseload[lv_4] - power_check_config.get('bl_lv_3')) + lv_3_thr

    else:

        # Assign default thresholds

        power_chk_thr[lv_1] = power_check_config.get('bl_lv_1_thr')
        power_chk_thr[lv_2] = power_check_config.get('bl_lv_2_thr')
        power_chk_thr[lv_3] = power_check_config.get('bl_lv_3_thr')

        # For days in level 4 create threshold that grows linearly

        lv_3_thr = power_check_config.get('bl_lv_3_thr')
        power_chk_thr[lv_4] = slope * (day_wise_baseload[lv_4] - power_check_config.get('bl_lv_3')) + lv_3_thr

    # Add daily baseload to get the final threshold to be used for power check

    power_chk_thr = day_wise_baseload + power_chk_thr

    # Initialize the boolean array indicating days that will continue in the loop

    continue_day_bool = copy.deepcopy(probable_vac_bool)

    # Initialize parameters for the loop

    scaling_factor = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    window_size = int(power_check_config.get('window_size') * scaling_factor)
    sliding_size = int(power_check_config.get('slide_size') * scaling_factor)

    # Initialize arrays to store values computed in the loop

    num_windows = int(num_pts_day / sliding_size)

    is_nan_window = np.full(shape=(num_days, num_windows), fill_value=False)
    is_violating_thr = np.full(shape=(num_days, num_windows), fill_value=False)
    sliding_power = np.full(shape=(num_days, num_windows), fill_value=np.nan)
    sliding_power_bkp = np.full(shape=(num_days, num_windows), fill_value=np.nan)

    # Initialize column index variables

    loop_brk_col_1 = 0
    loop_brk_col_2 = 1

    # Loop over all windows and compute power while rejecting days that violate thresholds for 2 consecutive windows

    for window_start_idx in range(0, num_pts_day, sliding_size):

        # Compute the index of the sliding window

        sliding_idx = int(window_start_idx / sliding_size)

        # Extract data for the window

        window_end_idx = min(window_start_idx + window_size, num_pts_day)

        window_data = day_data[:, window_start_idx: window_end_idx]
        window_mask_bool = day_valid_mask_cons[:, window_start_idx: window_end_idx]

        # Compute and populate power for this index, also mark days for which this window is violating threshold
        # Windows which fail threshold check due to masking are excluded from being counted as violations

        window_power = np.round(np.nanmean(window_data, axis=1) * scaling_factor, 2)
        is_not_masked_window = np.sum(window_mask_bool, axis=1) < window_mask_bool.shape[1]

        is_nan_window[:, sliding_idx] = np.logical_and(np.isnan(window_power), is_not_masked_window)
        is_violating_thr[:, sliding_idx] = np.logical_or(window_power > power_chk_thr, is_nan_window[:, sliding_idx])

        # Exclude days that are not maintaining window level power thresholds

        if sliding_idx > 0:

            # Identify the days that violate thresholds for 2 consecutive windows and are still in the loop

            break_idx = np.logical_and(is_violating_thr[:, sliding_idx - 1], is_violating_thr[:, sliding_idx])
            break_idx = np.logical_and(break_idx, continue_day_bool)

            # Exclude the days which violate the thresholds, remove the violating

            continue_day_bool[break_idx] = False

            # Populate loop break index and loop break power for days that were rejected

            loop_break_idx[break_idx] = sliding_idx
            loop_break_power[break_idx, loop_brk_col_1] = sliding_power[break_idx, sliding_idx - 1]
            loop_break_power[break_idx, loop_brk_col_2] = window_power[break_idx]

            # Populate loop break power for days where loop broke due to nans with infinity

            nan_bool = np.logical_and(is_nan_window[:, sliding_idx - 1], is_nan_window[:, sliding_idx])
            loop_break_power[nan_bool, :] = np.inf

        # Populate sliding power arrays. The backup array for all days and main array for continuing days

        sliding_power_bkp[:, sliding_idx] = window_power
        sliding_power[continue_day_bool, sliding_idx] = window_power[continue_day_bool]

    # Identify days that passed the loop successfully

    power_check_passed_bool = continue_day_bool

    return power_check_passed_bool, sliding_power, sliding_power_bkp, loop_break_idx, loop_break_power, power_chk_thr
