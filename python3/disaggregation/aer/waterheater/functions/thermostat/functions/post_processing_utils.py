"""
Author - Sahana M
Date - 20/05/2021
This module contains all the utility functions used in post processing
"""

# Import python packages
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

fat_box_columns = {
    'day_ts': 0,
    'start_idx': 1,
    'end_idx': 2,
    'valid_fat_run': 3
}

fat_box_columns2 = {
    'day_ts': 0,
    'start_idx': 1,
    'end_idx': 2,
    'duration': 3,
    'close_thin_pulse': 4,
    'valid_fat_run': 5
}

def mark_fat_pulse(usages, missed_fat_input):
    """
    This function is used to get the consumption of the valid fat pulses in a new array
    Parameters
    usages                  (ndarray)       : Array containing information about the validity of a fat pulse
    missed_fat_input        (ndarray)       : 21 column matrix
    Returns
    fat_energy_values       (ndarray)       : Contains the final interpolated fat pulses
    """

    # final fat output array

    fat_energy_values = np.array([0] * len(missed_fat_input))

    # Mark the new fat pulses obtained as 1

    for index in range(len(usages)):

        # Get start & end times for current fat pulse
        first_ts = int(usages[index, fat_box_columns['start_idx']])
        last_ts = int(usages[index, fat_box_columns['end_idx']])

        # Mark the data points within that fat pulse as 1

        if usages[index, fat_box_columns['valid_fat_run']] == 1:
            for i in range(first_ts, (last_ts + 1)):
                fat_energy_values[i] = missed_fat_input[i, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    return fat_energy_values


def mark_fat_pulse2(usages, total_fat_output):
    """
    This function is used to get the consumption of the valid fat pulses in a new array
    Parameters :-
    usages                  (ndarray)       : Array containing information about the validity of a fat pulse
    total_fat_output        (ndarray)       : 21 column matrix containing fat pulse output
    Returns :-
    fat_output              (ndarray)       : Contains the final interpolated fat pulses
    """

    # Mark the new fat pulses obtained as 1

    fat_output = np.array([0] * len(total_fat_output))

    for index in range(len(usages)):

        # Get start & end times for current fat pulse
        first_ts = int(usages[index, fat_box_columns2['start_idx']])
        last_ts = int(usages[index, fat_box_columns2['end_idx']])

        # Mark the data points within that fat pulse as 1
        if (usages[index, fat_box_columns2['close_thin_pulse']] == 0) and (usages[index, fat_box_columns2['valid_fat_run']] == 1):
            for i in range(first_ts, (last_ts + 1)):
                fat_output[i] = total_fat_output[i, 6]

    return fat_output


def days_in_billcycle(input_data, bill_cycle_ts):
    """
    This function is used to identify the number of days in a bill cycle
    Parameters
    input_data                  (ndarray)        : 21 column matrix containing the input data
    bill_cycle_ts               (ndarray)        : Array containing the bill cycle time stamps
    Returns
    days_per_bill_cycle         (ndarray)        : Contains the days per bill cycle
    """

    # Get number of days in each bc

    days_per_bill_cycle = np.array([])

    for bc in bill_cycle_ts:
        days_current_bc = len(np.unique(input_data[input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == bc,
                                                   Cgbdisagg.INPUT_DAY_IDX]))
        days_per_bill_cycle = np.r_[days_per_bill_cycle, days_current_bc]

    return days_per_bill_cycle


def get_daily_pulses(winter_pulses, season_pulses, pulse_output, winter_bill_cycle_idx, season_bill_cycle_idx):
    """
    Function to get the number of pulses occuring each day
    Arguments:
        winter_pulses                 (np.array)        : Array containing Winter days consumption
        season_pulses                 (np.array)        : Array containing Summer/Intermediate consumption
        pulse_output                  (np.ndarray)      : 21 column matrix containing the thin/fat pulse output
        winter_bill_cycle_idx         (boolean)         : Boolean array containing the winter days indexes
        season_bill_cycle_idx         (boolean)         : Boolean array containing the Summer/Intermediate days indexes
    Returns:
        daily_winter_pulses           (np.array)        : Array containing the number of pulses in a day in winter
        daily_season_pulses           (np.array)        : Array containing the number of pulses in a day in other seasons
        pulse_output                  (np.ndarray)      : 21 column matrix containing the thin/fat pulse output
    """

    # Mark the indexes where consumption in Winter > 0

    winter_energy_idx = (winter_pulses > 0).astype(int)
    winter_idx_diff = np.diff(np.r_[0, winter_energy_idx])
    winter_idx_diff[winter_idx_diff < 0] = 0

    # Mark the indexes where consumption in Summer/Intermediate > 0

    season_energy_idx = (season_pulses > 0).astype(int)
    season_idx_diff = np.diff(np.r_[0, season_energy_idx])
    season_idx_diff[season_idx_diff < 0] = 0

    # Create a new temporary column to mark the start of the pulses
    pulse_output = np.hstack((pulse_output, np.zeros(pulse_output.shape[0]).reshape(-1, 1)))

    pulse_output[winter_bill_cycle_idx, -1] = winter_idx_diff
    pulse_output[season_bill_cycle_idx, -1] = season_idx_diff

    # Summation of the number of pulses in a day for each seasons

    daily_winter_pulses = np.array([(pulse_output[pulse_output[:, 2] == i, -1].sum()) for i in
                                    np.unique(pulse_output[winter_bill_cycle_idx, 2])])

    daily_season_pulses = np.array([(pulse_output[pulse_output[:, 2] == i, -1].sum()) for i in
                                    np.unique(pulse_output[season_bill_cycle_idx, 2])])

    # Remove the added temporary column

    pulse_output = pulse_output[:, :-1]

    return daily_winter_pulses, daily_season_pulses, pulse_output


def control_pulse_count(season_bill_cycle_idx, pulse_matrix, daily_season_pulses, winter_pulses, avg_winter_pulses):
    """
    Function to remove erroneous pulses and restrict the number of pulses in the season
    Parameters:
        season_bill_cycle_idx             (boolean)     : Boolean array containing True for current season days
        pulse_matrix                      (np.ndarray)  : 21 column matrix containing the thin/fat pulse output
        daily_season_pulses               (np.array)    : Array containing the number of pulses a day in the season
        winter_pulses                     (np.array)    : Array containing the winter consumption
        avg_winter_pulses                 (float)       : The average winter pulses
    Returns:
        pulse_output                      (np.ndarray)  : 21 column matrix containing the thin/fat pulse output
    """

    # Extract the needed data
    pulse_output = deepcopy(pulse_matrix)

    # Get the count buffer, season days epoch, days with higher pulse occurrence than the count buffer

    count_buffer = np.round((avg_winter_pulses + 1), 0)
    season_days = np.unique(pulse_output[season_bill_cycle_idx, 2])
    high_count_days = np.array(np.where(daily_season_pulses > count_buffer)[0])

    # Get the indexes where winter pulse consumption starts

    wtr_idx_diff_yearly = np.diff(np.r_[0, winter_pulses, 0])
    wtr_start_idx_yearly = np.where(wtr_idx_diff_yearly[:-1] > 0)[0]

    # Calculate the occurrence of pulses for each hour of day

    max_hod = int(1 * Cgbdisagg.HRS_IN_DAY) - 1
    edges = np.arange(0, max_hod + 2) - 0.5
    wtr_hod_count, _ = np.histogram(pulse_output[wtr_start_idx_yearly, Cgbdisagg.INPUT_HOD_IDX], bins=edges)

    # For each day with high pulse count perform curbing

    for i in range(len(high_count_days)):

        day = season_days[high_count_days[i]]
        day_idx = pulse_output[:, 2] == day

        # Identify the start & end indexes

        day_pulses = pulse_output[day_idx, 6]
        day_pulses_idx = (day_pulses > 0).astype(int)
        day_pulses_idx_diff = np.diff(np.r_[0, day_pulses_idx, 0])
        start_idx = np.array(np.where(day_pulses_idx_diff == 1)[0])
        end_idx = np.array(np.where(day_pulses_idx_diff == -1)[0])

        # Get the start time of each pulse

        start_time_of_pulse = pulse_output[day_idx][start_idx, 4]
        cleaned_array = np.zeros(shape=len(day_pulses))

        # Best times of day are those times which had high occurrence in Winter

        best_times_of_day = np.argsort(wtr_hod_count)[::-1]

        pulses_captured_count = 0

        # For each pulse in the day, find if the pulses occurs in the best time of day else discard

        for i in range(len(best_times_of_day)):
            if (best_times_of_day[i] in start_time_of_pulse) and (pulses_captured_count < count_buffer):
                pulse_idx = np.where(start_time_of_pulse == best_times_of_day[i])[0][0]
                cleaned_array[start_idx[pulse_idx]: end_idx[pulse_idx]] = day_pulses[start_idx[pulse_idx]: end_idx[pulse_idx]]
                pulses_captured_count += 1

        pulse_output[day_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = cleaned_array

    return pulse_output
