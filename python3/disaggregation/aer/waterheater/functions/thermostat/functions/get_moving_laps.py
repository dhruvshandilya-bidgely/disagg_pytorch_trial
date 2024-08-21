"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to get dynamic laps with and without single peaks
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.maths_utils import moving_sum
from python3.utils.maths_utils.maths_utils import convolve_function


def get_moving_laps(input_data, wh_config, logger):
    """
    Function to get dynamic LAPs

    Parameters:
        input_data          (np.ndarray)    : Input 21-column matrix
        wh_config           (dict)          : Config params
        logger              (logger)        : Logger object

    Returns:
        lap_mid_ts          (np.ndarray)    : Indices of lap mid-timestamps
        peaks_idx           (np.ndarray)    : Single peak indices
    """

    # Get detection config params from wh_config

    detection_config = wh_config['thermostat_wh']['detection']

    # Calculate window size for the LAP (half_width * 2)

    window_size = int(detection_config['lap_half_width'] * 2 * Cgbdisagg.SEC_IN_HOUR / wh_config['sampling_rate'])

    # Get moving LAPs

    if input_data.shape[0] > 0:
        # If input data size is valid

        lap_idx, peaks_idx = get_laps_indices(input_data, window_size, detection_config)
    else:
        # If input data is empty, return blank output

        lap_idx = np.array([])
        peaks_idx = np.array([])

    # If valid number of laps, return bill cycle, day and epoch timestamp for each

    if np.sum(lap_idx) > 0:
        # If number of laps more than zero

        lap_mid_ts = input_data[lap_idx][:, [Cgbdisagg.INPUT_BILL_CYCLE_IDX,
                                             Cgbdisagg.INPUT_DAY_IDX,
                                             Cgbdisagg.INPUT_EPOCH_IDX]]

        logger.info('Number of LAPs in the season | {}'.format(len(lap_mid_ts)))
    else:
        # If number of laps is zero

        lap_mid_ts = np.array([], dtype=np.int64).reshape(0, 3)

        logger.info('No LAPs found in the season | ')

    return lap_mid_ts, peaks_idx, lap_idx


def get_laps_indices(in_data, window_size, config):
    """
    Parameters:
        in_data             (np.ndarray)        : Input 21-column matrix
        window_size         (int)               : Window size for lap
        config              (dict)              : Config params

    Returns:
        all_laps            (np.ndarray)        : Lap mid-timestamps
        peaks_idx           (np.ndarray)        : Single peak indices
    """

    # Taking a deepcopy of input data to keep local instances

    input_data = deepcopy(in_data)

    # Extract required params for LAPs

    trunc_gap = config['end_gap']
    n_points = config['peak_points']
    half_gap = config['lap_half_width']

    # Get required amplitude bounds

    min_amp = config['min_thin_pulse_amp']
    max_amp = config['max_thin_pulse_amp']
    diff_amp = config['thin_pulse_amp_std']

    # Get the convolution filters

    amp_mask = config['amplitude_mask']
    der_mask = config['derivative_mask']

    # Calculate window size for truncated hours and window

    cut = int((trunc_gap * window_size) // (half_gap * 2))

    # Define shift window size for considering end points of the input data

    shift = window_size // n_points

    # Energy diff to find points with consecutive increase and decrease of above 250

    energy_diff = np.r_[[0], np.diff(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]), [0] * shift]
    abs_energy_diff = np.abs(energy_diff)

    # Create array to mark sharp peak found or not at each epoch

    single_peaks = np.array([False] * len(abs_energy_diff))

    # Convolve single peak detection and difference filters

    amp = convolve_function(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], amp_mask)
    der = convolve_function(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], der_mask)

    # Take the indices of peaks that satisfy boundary conditions

    single_peaks_bool = (amp > min_amp) & (amp < max_amp) & (der < diff_amp) & (abs_energy_diff[:-shift] > min_amp)

    # Get the single peaks indices

    single_peaks_idx = np.where(single_peaks_bool)[0]

    # Mask the corresponding indices as valid

    valid_idx = np.sort(np.r_[single_peaks_idx, single_peaks_idx + 1])
    single_peaks[valid_idx] = True

    # Accumulate the peaks info in a window to separate out very close peaks

    points_above_250 = moving_sum(single_peaks, window_size)
    valid_points_above_250 = (points_above_250 == n_points)[shift:]

    # Mask array below the certain amplitude

    arr_below_100 = (abs_energy_diff < diff_amp)

    # Doing calculations for full window

    b_100 = moving_sum(arr_below_100, window_size)

    # Lap with peaks

    lap_with_peaks = (b_100 == (window_size - n_points))[shift:]

    # Lap without peaks

    lap_without_peaks = (b_100 == window_size)[shift:]

    # Find the peaks within the truncated limit and remove them

    energy_diff[~single_peaks] = 0
    energy_cumsum = np.r_[0, moving_sum(energy_diff[1:], n_points)]
    bool_energy_lap = (np.abs(energy_cumsum) < min_amp)[shift:]

    # Filter the laps with peaks and not in truncated zone

    peaks_laps_1 = valid_points_above_250 & lap_with_peaks & bool_energy_lap

    # Doing calculations for window leaving first and last 'trunc_gap' hours

    if cut > 0:
        b_100 = moving_sum(arr_below_100[:-cut], window_size - n_points * cut)
    else:
        b_100 = moving_sum(arr_below_100[:], window_size - n_points * cut)

    # Filter peaks that are on the edges of laps

    lap_with_peaks = (b_100 == ((window_size - n_points * cut) - n_points))[(shift - cut):]

    # Filter laps within a certain window of a lap

    peaks_laps = valid_points_above_250 & lap_with_peaks & peaks_laps_1
    all_laps = lap_without_peaks | peaks_laps

    # Finding the sharp peaks indices and mask them true

    peaks_idx = np.array([False] * len(single_peaks[:-shift]))
    peaks_idx[single_peaks_bool] = True

    return all_laps, peaks_idx
