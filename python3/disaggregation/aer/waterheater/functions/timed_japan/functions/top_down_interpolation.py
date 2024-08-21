"""
Author - Sahana M
Date - 07/06/2021
Post processing sub function
"""

# Import python packages
import numpy as np
from copy import deepcopy


def interpolate_amplitude(near_days_consumption, original_data_matrix, final_twh_matrix, day, wh_config, start_time, end_time):
    """
    Fill the days with 0 consumption with the corresponding raw data
    Parameters:
        near_days_consumption       (np.ndarray)        : Past/Future 15 days consumption
        original_data_matrix        (np.ndarray)        : Contains raw data
        final_twh_matrix            (np.ndarray)        : Contains final Wh estimation
        day                         (int)               : Day in consideration
        wh_config                   (dict)              : Timed wh configurations
        start_time                  (int)               : Start time of the band
        end_time                    (int)               : End time of the band

    Returns:
        final_twh_matrix            (np.ndarray)        : Contains final Wh estimation
        top_down_inter_done         (Bool)              : Boolean signifying the status of top down interpolation
    """

    # Extract all the necessary data

    window = wh_config.get('factor')
    window = int(window/2)
    min_bar = wh_config.get('td_min_bar')
    max_bar = wh_config.get('td_max_bar')

    # Identify the past/future days consumption

    if np.sum(near_days_consumption) == 0:
        near_days_amp = 0
    else:
        near_days_amp = np.median(near_days_consumption[near_days_consumption > 0])

    # Identify the time interval of timed wh in the past/future days

    near_days_time = np.sum(near_days_consumption > 0, axis=0)
    near_days_time = near_days_time > 0

    # Perform right and left shift of the near_days_time array to allow some shifting
    for k in range(1, (window + 1)):
        near_days_time[:-k] = near_days_time[:-k] | near_days_time[k:]

    # Right shift and expand the duration
    for k in range(1, (window + 1)):
        near_days_time[k:] = near_days_time[k:] | near_days_time[:-k]

    interpolation_check_needed = False
    if near_days_amp != 0:
        interpolation_check_needed = True

    top_down_inter_done = False
    if interpolation_check_needed:

        # Get the current day consumption from raw data for interpolation
        current_day_amp = 0
        day_consumption = deepcopy(original_data_matrix[day, start_time:end_time])
        interested_zone = day_consumption[near_days_time]
        if np.sum(interested_zone) > 0:
            current_day_amp = np.percentile(interested_zone[interested_zone > 0], q=90)

        current_days_time = day_consumption > 0

        # Identify the time overlap between the past/future days and the current day
        if np.sum(current_days_time) > np.sum(near_days_time):
            total_time_coverage = np.sum(near_days_time)
        else:
            total_time_coverage = np.sum(current_days_time)
        overlap = (np.sum(near_days_time & current_days_time)) / total_time_coverage

        # Interpolate if there is a good amplitude & time match
        interpolation_needed = (current_day_amp > min_bar * near_days_amp) & \
                               (current_day_amp < max_bar * near_days_amp) & \
                               (overlap > min_bar)

        # Perform interpolation
        if interpolation_needed:
            day_consumption[~near_days_time] = 0
            day_consumption[day_consumption < min_bar * near_days_amp] = 0
            day_consumption[day_consumption > max_bar * near_days_amp] = near_days_amp
            final_twh_matrix[day, start_time:end_time] = day_consumption
            top_down_inter_done = True

    return final_twh_matrix, top_down_inter_done


def top_down_interpolation(original_data_matrix, final_twh_matrix, start_time, end_time, wh_config, overall_chunk_data):
    """
    This function is used to perform interpolation for days with 0 consumption by comparing with past/future days
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

    # Extract all the necessary data

    window = wh_config.get('factor')
    start_time = int(max(start_time - window, 0))
    end_time = int(min(end_time + window, original_data_matrix.shape[0]))
    band_matrix = final_twh_matrix[:, start_time:end_time]

    rows = band_matrix.shape[0]
    expansion_days = wh_config.get('days')/2

    for day in range(rows):

        if int(day/expansion_days) in overall_chunk_data[:, 0]:
            day_consumption = band_matrix[day, :]

            # Top down interpolation
            if np.sum(day_consumption) == 0:

                # Get the last 15 days data
                past_days = int(max(day - expansion_days, 0))
                past_days_consumption = band_matrix[past_days: day, :]
                final_twh_matrix, top_down_inter_done = \
                    interpolate_amplitude(past_days_consumption, original_data_matrix, final_twh_matrix, day, wh_config,
                                          start_time, end_time)

                # Down to top interpolation
                if not top_down_inter_done:
                    future_days = int(min(day + expansion_days, band_matrix.shape[0]))
                    future_days_consumption = band_matrix[day: future_days, :]
                    final_twh_matrix, _ = interpolate_amplitude(future_days_consumption, original_data_matrix, final_twh_matrix,
                                                                day, wh_config, start_time, end_time)

    return final_twh_matrix
