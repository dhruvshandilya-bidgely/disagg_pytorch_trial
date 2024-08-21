"""
Author - Sahana M
Date - 20/07/2021
The module detects the time intervals with good consistency of edges
"""

# Import python packages
import numpy as np
from copy import deepcopy
from python3.utils.maths_utils.rolling_function import rolling_function


def get_consistency_arr(input_data, wh_config, debug):
    """
    This function is used to identify the time intervals with high consistency edges
    Parameters:
        input_data              (np.ndarray)        : 2D matrix containing the cleaned data
        wh_config               (dict)              : contains all the twh configurations
        debug                   (dict)              : Debug dictionary

    Returns:
        concentration_arr_bool  (np.ndarray)        : Boolean array containing the time intervals
        concentration_arr       (np.ndarray)        : Numpy array containing the time intervals with probability
    """

    # Initialise the necessary variables

    factor = wh_config.get('factor')
    cleaned_data = deepcopy(input_data)
    num_box_days = cleaned_data.shape[0]
    min_amplitude = int(wh_config.get('min_amp') / factor)
    min_consistent_thr = wh_config.get('min_consistent_thr')
    base_consistent_thr = wh_config.get('base_consistent_thr')

    cleaned_data[cleaned_data <= wh_config.get('min_amp_bar')*min_amplitude] = 0

    # Taking only positive box boundaries for edge related calculations

    box_energy_idx_bool = (cleaned_data > 0)
    box_energy_idx_bool_flat = box_energy_idx_bool.flatten()

    # Accumulate the valid count over the window

    box_energy_idx_bool_flat = rolling_function(box_energy_idx_bool_flat, 2, 'sum')

    # Flatten to get the start and end edges of the boxes

    box_energy_idx_bool_flat = box_energy_idx_bool_flat >= 2

    # Identify the starting and the ending edges of boxes

    box_energy_idx = np.diff(np.r_[0, box_energy_idx_bool_flat])
    box_energy_idx[box_energy_idx < 0] = 0
    box_energy_idx = box_energy_idx.reshape(cleaned_data.shape[0], cleaned_data.shape[1])

    # Calculate the % of edges at each time interval

    concentration_arr = np.sum(box_energy_idx, axis=0) / num_box_days
    probable_start_idx = np.where(concentration_arr >= min_consistent_thr)[0]
    good_indexes = np.sum(box_energy_idx_bool, axis=0) / num_box_days

    # Identify the final indexes

    final_indexes = []
    for i in range(len(probable_start_idx)):
        idx = int(probable_start_idx[i])
        final_indexes.append(idx)

        # Find the time satisfying base_consistent_thr to the right of the idx

        k = idx + 1
        for j in range(k, len(good_indexes)):
            if good_indexes[j] <= base_consistent_thr:
                break
            final_indexes.append(j)

        # Find the time satisfying base_consistent_thr to the left of the idx

        k = idx - 1
        for j in range(k, -1, -1):
            if good_indexes[j] <= base_consistent_thr:
                break
            final_indexes.append(j)

    final_indexes = np.unique(final_indexes)

    # Create a boolean array denoting the high consistent time intervals

    concentration_arr_bool = np.full(shape=(concentration_arr.shape[0]), fill_value=False)
    if len(final_indexes):
        concentration_arr_bool[final_indexes] = True

    # To maintain continuity check for previous hsm

    concentration_arr_bool, concentration_arr = continuity_check(debug, final_indexes, concentration_arr, concentration_arr_bool)

    return concentration_arr_bool, concentration_arr


def continuity_check(debug, final_indexes, concentration_arr, concentration_arr_bool):
    """
    This function is used to make sure that a continuity in time band is maintain across historical,
    incremental and mtd runs
    Parameters:
        debug                   (dict)          : Debug dictionary
        final_indexes           (np.ndarray)    : Final timed wh indexes
        concentration_arr       (np.ndarray)    : Time frame for timed wh
        concentration_arr_bool  (np.ndarray)    : Boolean time frame for timed wh
    Returns:
        concentration_arr_bool  (np.ndarray)    : Boolean time frame for timed wh
        concentration_arr       (np.ndarray)    : Time frame for timed wh
    """

    if debug.get('hsm_in') is not None:
        if not len(final_indexes):
            hsm_in = debug.get('hsm_in')
            old_run_bands = np.array(hsm_in.get('twh_time_bands'))
            concentration_arr_bool = old_run_bands == 1

        if debug.get('disagg_mode') != 'historical':
            hsm_in = debug.get('hsm_in')
            old_run_bands = np.array(hsm_in.get('twh_time_bands'))
            old_run_bands_bool = old_run_bands == 1
            if np.sum(old_run_bands_bool):
                concentration_arr_bool |= old_run_bands_bool

    return concentration_arr_bool, concentration_arr
