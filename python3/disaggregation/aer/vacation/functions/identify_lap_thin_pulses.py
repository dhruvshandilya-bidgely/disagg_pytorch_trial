"""
Author - Mayank Sharan
Date - 1/12/19
Remove thin pulses from the input data in Low Activity Period (LAP) regions
"""

# Import python packages

import copy
import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.maths_utils import moving_sum
from python3.utils.maths_utils.maths_utils import convolve_function


def get_lap_bool(all_lap_bool, sampling_rate, lap_half_width):

    """
    Function to mark all points as whether they are in an LAP or not using middle points

    Parameters:
        all_lap_bool            (np.ndarray)        : Array containing boolean for all middle indices of LAPs
        sampling_rate           (int)               : Sampling rate of the data
        lap_half_width          (float)             : Half of the width of default LAP in hours

    Returns:
        lap_bool                (np.ndarray)        : Array containing boolean for all points in LAPs
    """

    # Initialize variables needed

    pd_mult = Cgbdisagg.SEC_IN_HOUR / sampling_rate
    range_points_to_set = np.arange(int(np.floor(- lap_half_width * pd_mult)) + 1, int(lap_half_width * pd_mult) + 1)

    lap_bool = np.full(shape=all_lap_bool.shape, fill_value=False)
    all_lap_mid_idx = np.where(all_lap_bool)[0]

    num_points = len(lap_bool)

    # For each LAP mark all points in the window as True

    for index_shift in range_points_to_set:
        mod_idx_list = all_lap_mid_idx + index_shift

        mod_idx_list = mod_idx_list[mod_idx_list < num_points]
        mod_idx_list = mod_idx_list[mod_idx_list >= 0]

        lap_bool[mod_idx_list] = True

    return lap_bool


def identify_lap_thin_pulses(input_data, vacation_config, logger_pass):

    """
    Function to identify single point peaks lying in LAPs that could be WH or multi ref

    Parameters:
        input_data              (np.ndarray)        : 21 column input data
        vacation_config         (dict)              : Contains all configuration variables needed for vacation
        logger_pass             (dict)              : Contains the logger and the logging dictionary to be passed on

    Returns:
        peak_amp_arr            (np.ndarray)        : Amplitude array result of convolution with thin peak filter
        peak_removal_bool       (np.ndarray)        : Boolean array indicating indices location of LAP wh thin pulses
    """

    # Initialize logger

    logger_base = logger_pass.get('base_logger').getChild('identify_lap_thin_pulses')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Initialize variables from config

    sampling_rate = vacation_config.get('user_info').get('sampling_rate')
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    thin_pulse_id_config = vacation_config.get('thin_pulse_id')

    # Initialize window size related variables

    trunc_gap = thin_pulse_id_config.get('trunc_gap')

    lap_half_width = thin_pulse_id_config.get('lap_half_width')
    window_size = int(lap_half_width * 2 * pd_mult)

    # Initialize amplitude related variables

    min_amp = thin_pulse_id_config.get('min_thin_pulse_amp')
    max_amp = thin_pulse_id_config.get('max_thin_pulse_amp')
    diff_amp = thin_pulse_id_config.get('thin_pulse_amp_std')

    # Initialize mask related variables

    amp_mask = thin_pulse_id_config.get('amplitude_mask')
    der_mask = thin_pulse_id_config.get('derivative_mask')

    # Initialize window movement related variables

    trunc_window_size = int(trunc_gap * pd_mult)
    shift_pts = window_size // 2

    # Extract consumption and compute a padded array containing consecutive differences

    cons_arr = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    nan_cons_idx = np.isnan(cons_arr)
    cons_arr[nan_cons_idx] = 0

    cons_diff_arr = np.r_[[0], np.diff(cons_arr), [0] * shift_pts]
    abs_diff_arr = np.abs(cons_diff_arr)

    # Identify indices with single peak consumption above minimum amplitude threshold

    arr_above_min = np.full(shape=abs_diff_arr.shape, fill_value=False, dtype=bool)
    peak_amp_arr = convolve_function(cons_arr, amp_mask)
    der_conv_arr = convolve_function(cons_arr, der_mask)

    valid_bool = np.logical_and(abs_diff_arr[: -shift_pts] > min_amp,
                                np.logical_and(der_conv_arr < diff_amp,
                                               np.logical_and(peak_amp_arr > min_amp, peak_amp_arr < max_amp)))
    valid_idx = np.where(valid_bool)[0]

    arr_above_min[valid_idx] = True
    arr_above_min[valid_idx + 1] = True

    above_min_moving_sum = moving_sum(arr_above_min, window_size)
    peak_above_min_bool_arr = (above_min_moving_sum == 2)[shift_pts:]

    # Locate LAPs

    arr_below_diff_amp = abs_diff_arr < diff_amp
    below_diff_moving_sum = moving_sum(arr_below_diff_amp, window_size)

    lap_with_peak_bool = (below_diff_moving_sum == window_size - 2)[shift_pts:]
    lap_without_peak_bool = (below_diff_moving_sum == window_size)[shift_pts:]

    # Identify LAPs with peak contained within it excluding the truncated edges

    cons_diff_arr[~arr_above_min] = 0
    cons_diff_moving_sum = np.r_[0, moving_sum(cons_diff_arr[1:], 2)]

    cons_lap_bool = (np.abs(cons_diff_moving_sum) < min_amp)[shift_pts:]
    peak_in_lap_limit_bool = np.logical_and(peak_above_min_bool_arr, np.logical_and(lap_with_peak_bool, cons_lap_bool))

    # Reject all LAPs that have peeks in truncation region

    if trunc_window_size > 0:
        below_diff_moving_sum_2 = moving_sum(arr_below_diff_amp[: -trunc_window_size],
                                             window_size - (2 * trunc_window_size))
    else:
        below_diff_moving_sum_2 = moving_sum(arr_below_diff_amp, window_size - (2 * trunc_window_size))

    lap_with_edge_peak_bool = \
        (below_diff_moving_sum_2 == ((window_size - 2 * trunc_window_size) - 2))[(shift_pts - trunc_window_size):]

    lap_valid_peak_bool = np.logical_and(peak_above_min_bool_arr, np.logical_and(lap_with_edge_peak_bool,
                                                                                 peak_in_lap_limit_bool))

    # Compute the boolean indicating the points that are parts of LAPs

    all_laps_mid_pt_bool = np.logical_or(lap_without_peak_bool, lap_valid_peak_bool)
    lap_pts_bool = get_lap_bool(all_laps_mid_pt_bool, sampling_rate, lap_half_width)

    peak_removal_bool = np.logical_and(valid_bool, lap_pts_bool)

    logger.info('Number of potential points for peak removal | %d', np.sum(peak_removal_bool))

    return peak_amp_arr, peak_removal_bool
