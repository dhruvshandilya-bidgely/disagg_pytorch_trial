"""
Author - Sahana M
Date - 23/06/2021
Horizontal interpolation of timed wh
"""

# Import python packages
import numpy as np

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.twh_maths_utils import get_start_end_idx


def extend_time(raw_part, min_bar, max_bar, day_amp, side='Right'):
    """
    This function is used to mark the indexes with amplitude lying in a specified range
    Parameters:
        raw_part                (np.ndarray)        : Amplitude array
        min_bar                 (float)             : Minimum amplitude bar
        max_bar                 (float)             : Maximum amplitude bar
        day_amp                 (float)             : Day amplitude
        side                    (String)            : Right/ Left extension
    Returns:
        extension_arr           (np.ndarray)        : Indexes with satisfying amplitude
    """

    extension_arr = np.full_like(raw_part, fill_value=0.0)

    # Mark all the right near by points whose amplitudes lie within a considerable range of the amplitude at
    # original time interval

    if side == 'Right':

        for i in range(len(raw_part)):
            if (raw_part[i] > min_bar * day_amp) & (raw_part[i] < max_bar * day_amp):
                extension_arr[i] = raw_part[i]
            else:
                break

    if side == 'Left':

        # Mark all the left near by points whose amplitudes lie within a considerable range of the amplitude at
        # original time interval

        for i in range(len(raw_part) - 1, -1, -1):
            if (raw_part[i] > min_bar * day_amp) & (raw_part[i] < max_bar * day_amp):
                extension_arr[i] = raw_part[i]
            else:
                break

    return extension_arr


def horizontal_interpolation(original_data_matrix, final_twh_matrix, start_time, end_time, wh_config, overall_chunk_data):
    """
    This function is used to perform right & left side interpolation
    Parameters:
        original_data_matrix             (np.ndarray)        : Contains raw data
        final_twh_matrix                 (np.ndarray)        : Final timed wh estimation
        start_time                       (int)               : Start time of the band
        end_time                         (int)               : End time of the band
        wh_config                        (dict)              : Timed wh configurations
        overall_chunk_data               (np.ndarray)        : Contains data about each chunk

    Returns:
        final_twh_matrix                 (np.ndarray)        : Final timed wh estimation
    """

    # Initialise the required variables

    factor = wh_config.get('factor')
    expansion_hours = factor * wh_config.get('hi_time')
    min_bar = wh_config.get('hi_min_bar')
    max_bar = wh_config.get('hi_max_bar')
    expansion_days = wh_config.get('days')/2

    band_matrix = final_twh_matrix[:, start_time:end_time]

    # For each day perform horizontal interpolation of timed wh

    for day in range(band_matrix.shape[0]):
        if int(day/expansion_days) in overall_chunk_data[:, 0]:

            # Get the day's consumption & the time ranges to be expanded

            day_consumption = band_matrix[day, :]
            right_end_time_extended = int(min((end_time + expansion_hours), original_data_matrix.shape[1]))
            left_start_time_extended = int(max(0, (start_time - expansion_hours)))
            right_raw_consumption = original_data_matrix[day, start_time:right_end_time_extended]
            left_raw_consumption = original_data_matrix[day, left_start_time_extended:end_time]

            if np.sum(day_consumption) > 0:

                # Get the start & end time of the current days consumption

                day_consumption_bool = day_consumption > 0
                start_idx, end_idx = get_start_end_idx(day_consumption_bool)
                end_idx = end_idx[-1]

                # Get the raw data consumption for end time + expansion hours

                day_amp = np.median(day_consumption[day_consumption > 0])
                raw_right_part = right_raw_consumption[end_idx:]

                # Mark all the right near by points whose amplitudes lie within a considerable range of the amplitude at
                # original time interval

                extension_arr = extend_time(raw_right_part, min_bar, max_bar, day_amp, side='Right')

                # Expand the day's consumption on the right side

                fixed_start = start_time + end_idx
                fixed_end = right_end_time_extended
                final_twh_matrix[day, fixed_start:fixed_end] = extension_arr

                # Get the raw data consumption for start_time - expansion hours

                start_idx = start_idx[0]
                raw_left_part = left_raw_consumption[:((start_time + start_idx) - left_start_time_extended)]

                # Mark all the left near by points whose amplitudes lie within a considerable range of the amplitude at
                # original time interval

                extension_arr = extend_time(raw_left_part, min_bar, max_bar, day_amp, side='Left')

                # Expand the day's consumption on the left side

                fixed_start = left_start_time_extended
                fixed_end = start_time + start_idx
                final_twh_matrix[day, fixed_start:fixed_end] = extension_arr

    return final_twh_matrix
