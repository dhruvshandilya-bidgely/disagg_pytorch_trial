"""
Author - Sahana M
Date - 4/3/2021
Removes seasonal wh boxes which violates minimum and maximum duration criteria
"""

# Import python packages

import numpy as np
from copy import deepcopy


def get_duration(swh_data_matrix, start_time, end_time, seasonal_wh_config, debug):

    """
    Calculates duration of each seasonal wh box and removes boxes which violates min and max duration criteria
    Args:
        swh_data_matrix     (np.ndarray)    : Contains (day x time) shape consumption matrix
        start_time          (int)           : Denotes the start time index of the time zone in consideration
        end_time            (int)           : Denotes the end time index of the time zone in consideration
        seasonal_wh_config  (dict)          : Dictionary containing all needed configuration variables
        debug               (dict)          : Contains all variables required for debugging

    Returns:
        cleaned_data        (np.ndarray)    : Contains (day x time) shape cleaned data
        new_wh_present_idx  (np.array)      : Boolean array containing new wh present indexes
    """

    # Extract all the necessary data

    in_data = deepcopy(swh_data_matrix)
    new_wh_present_idx = deepcopy(debug['wh_present_idx'])
    start_idx = debug['swh_wh_pot_start']
    end_idx = debug['swh_wh_pot_end']
    padding_days = seasonal_wh_config.get('config').get('padding_days')
    min_amplitude = seasonal_wh_config.get('config').get('min_amplitude')
    max_amplitude = seasonal_wh_config.get('config').get('max_amplitude')
    factor = debug.get('factor')

    # Make all the indexes in the wh_potential days to True to avoid discontinuous days

    # end_idx < start_idx example pilot :- Dewa

    if end_idx < start_idx:
        new_wh_present_idx[end_idx: (start_idx + 1)] = True

    # end_idx > start_idx example pilot :- EESL

    else:
        new_wh_present_idx[end_idx: len(new_wh_present_idx)] = True
        new_wh_present_idx[0: (start_idx + 1)] = True

    # Add some padding days to improve estimation in both start and end

    new_wh_present_idx[start_idx: min(start_idx + 2*padding_days, len(new_wh_present_idx))] = True
    new_wh_present_idx[max(end_idx - 2*padding_days, 0): end_idx] = True

    # Consider only wh_present_days

    wh_data = in_data[new_wh_present_idx, start_time:end_time]

    # Get potential seasonal wh boxes based on amplitude

    wh_data[wh_data > (max_amplitude / factor)] = (max_amplitude / factor)
    wh_day_pulses = (wh_data >= (min_amplitude / factor))

    # Calculate the min (at least 30 minutes) and max duration (3 hours) scaled according to the sampling rate

    duration_min = max(factor / 2, 1)
    duration_max = factor * 3

    # Initialise a zero array and append as columns to wh_day_pulses

    zero_array = [0] * len(wh_day_pulses)
    wh_day_pulses = np.c_[zero_array, wh_day_pulses, zero_array]

    # Calculate the starting and ending index of the boxes

    box_energy_idx_diff = np.diff(wh_day_pulses)
    box_start_idx_row = np.where(box_energy_idx_diff[:] > 0)[0]
    box_start_idx_col = np.where(box_energy_idx_diff[:] > 0)[1]
    box_end_idx_col = np.where(box_energy_idx_diff[:] < 0)[1]

    cleaned_data = np.full(shape=swh_data_matrix.shape, fill_value=0.0)
    wh_idx = np.where(new_wh_present_idx)[0]

    # Create a matrix with the row, start idx, end idx and duration for each box

    matrix = np.c_[box_start_idx_row, box_start_idx_col, box_end_idx_col]
    matrix = np.c_[matrix, (matrix[:, 2] - matrix[:, 1])]

    # Remove the boxes with low duration and high duration

    matrix = matrix[(matrix[:, 3] >= duration_min) & (matrix[:, 3] < duration_max)]

    # Interpolate the new array

    for i in range(len(matrix)):
        start_band = start_time + matrix[i, 1]
        end_band = start_time + matrix[i, 2]
        cleaned_data[wh_idx[matrix[i, 0]], start_band:end_band] = swh_data_matrix[wh_idx[matrix[i, 0]], start_band:end_band]

    return cleaned_data, new_wh_present_idx
