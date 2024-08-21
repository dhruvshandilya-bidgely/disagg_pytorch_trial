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


# Define the columns for lap information

lap_data_columns = {
    'month_ts': 0,
    'day_ts': 1,
    'start_ts': 2,
    'end_ts': 3,
    'duration': 4
}


def get_consolidated_laps(in_data, lap_timestamps, laps_idx, wh_config):
    """
    Combines overlapping LAPs to get one LAP

    Parameters:
        in_data                 (np.ndarray)        : Input 21-column data matrix
        lap_timestamps          (np.ndarray)        : Mid timestamps of LAPs
        laps_idx                (np.ndarray)        : Boolean array marking LAP mid points
        wh_config               (dict)              : Config params

    Returns:
        combined_laps           (np.ndarray)        : Combined LAPs information
        input_data              (np.ndarray)        : Input data with LAP boolean appended (14-column)
    """

    # Taking a deepcopy of input data to keep local instances

    input_data = deepcopy(in_data)

    # Extract the detection params

    detection_config = wh_config['thermostat_wh']['detection']

    # Check if valid input data and lap information

    if (len(in_data) == 0) or (len(lap_timestamps) == 0):
        # If input data or lap data is empty, return blank array

        input_data = np.c_[input_data, np.array([0] * input_data.shape[0])]

        number_columns = len(lap_data_columns.keys())

        return np.array([], dtype=np.int64).reshape(0, number_columns), input_data

    # Retrieve the LAP mid timestamps

    lap_mid_timestamps = lap_timestamps[:, lap_data_columns['start_ts']]

    # Calculate the start and end timestamps of all laps

    low_limit = lap_mid_timestamps - ((detection_config['lap_half_width']) * Cgbdisagg.SEC_IN_HOUR) + \
                wh_config['sampling_rate']

    high_limit = lap_mid_timestamps + ((detection_config['lap_half_width']) * Cgbdisagg.SEC_IN_HOUR)

    # Convert start and end timestamp arrays to columns

    low_limit = low_limit[:, np.newaxis]
    high_limit = high_limit[:, np.newaxis]

    # Stack the start and end timestamps with lap info

    laps = np.hstack((low_limit, high_limit, lap_mid_timestamps[:, np.newaxis]))
    laps = laps[laps[:, lap_data_columns['day_ts']].argsort()]

    # Combine the laps by checking for overlapped portions

    merged_laps = combine_laps(laps, detection_config['lap_half_width'])

    # Create 14th column to mark if the data point is in lap

    input_data = np.hstack((input_data, np.zeros(shape=(len(input_data), 1))))

    # Iterate over combined laps

    for i, values in enumerate(merged_laps):
        # Get start & end times for current lap

        first_ts, last_ts = values

        # Mark the data points within that lap as 1

        input_data[(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= first_ts) &
                   (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= last_ts), Cgbdisagg.INPUT_DIMENSION] = 1

    # Get the start and end indices of the combined laps

    diff_x = np.diff(np.r_[0, input_data[:, Cgbdisagg.INPUT_DIMENSION], 0])

    start_idx = np.where(diff_x[:-1] > 0)[0]
    end_idx = np.where(diff_x[1:] < 0)[0]

    # Get the start and end timestamps of combined LAPs

    start_ts = input_data[start_idx, Cgbdisagg.INPUT_EPOCH_IDX]
    end_ts = input_data[end_idx, Cgbdisagg.INPUT_EPOCH_IDX]

    # Extract the bill cycle and day for combined LAPs

    month_ts = input_data[start_idx, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    day_ts = input_data[start_idx, Cgbdisagg.INPUT_DAY_IDX]

    # Stack the bill cycle ts, day ts, start ts, end ts and laps info

    combined_laps = np.hstack((month_ts.reshape(-1, 1), day_ts.reshape(-1, 1),
                               start_ts.reshape(-1, 1), end_ts.reshape(-1, 1)))

    combined_laps = np.hstack((combined_laps, np.zeros(shape=(len(combined_laps), 1))))

    # Calculate the duration of each LAP

    lap_duration = combined_laps[:, lap_data_columns['end_ts']] - combined_laps[:, lap_data_columns['start_ts']]

    # Append the LAP duration in hours

    combined_laps[:, lap_data_columns['duration']] = lap_duration / Cgbdisagg.SEC_IN_HOUR

    # Filter out LAPs greater than 24 hours

    combined_laps = combined_laps[combined_laps[:, lap_data_columns['duration']] <= Cgbdisagg.HRS_IN_DAY, :]

    return combined_laps, input_data


def combine_laps(all_laps, min_gap):
    """
    Parameters:
        all_laps            (np.ndarray)    : LAPs information
        min_gap             (int)           : Minimum gap between laps

    Returns:
        laps                (np.ndarray)    : Combined laps info
    """

    # Define the column indices for the laps info

    lap_start_idx = 0
    lap_end_idx = 1

    # Get the window size for lap

    gap_seconds = min_gap * 2 * Cgbdisagg.SEC_IN_HOUR

    # Calculate the time diff between given laps

    time_diff = np.diff(np.r_[all_laps[0, lap_data_columns['start_ts']], all_laps[:, lap_data_columns['start_ts']]])

    # If time diff is less than lap width, overlap is there

    cut_points = np.where(time_diff > gap_seconds)[0]

    # Find all the new start and end timestamps of combined laps

    start_idx = np.r_[0, cut_points]
    end_idx = np.r_[cut_points, all_laps.shape[0]] - 1

    # Stack the new timestamps with combined laps info

    laps = np.c_[all_laps[start_idx, lap_start_idx], all_laps[end_idx, lap_end_idx]]

    return laps
