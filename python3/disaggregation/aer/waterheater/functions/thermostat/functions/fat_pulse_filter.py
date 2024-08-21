"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module filters out the outlier fat usage boxes
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


fat_box_columns = {
    'day_ts': 0,
    'start_idx': 1,
    'end_idx': 2,
    'duration': 3,
    'ideal_fat_energy': 4,
    'upper_fat_energy': 5,
    'box_fat_energy': 6,
}


def fat_pulse_filter(in_data, fat_output, usages, wh_config, logger_base):
    """
    Parameters:
        in_data             (np.ndarray)        : Input 21-column matrix
        fat_output          (np.ndarray)        : Fat consumption
        usages              (np.ndarray)        : Fat usage boxes
        wh_config           (dict)              : Config params
        logger_base         (dict)              : Logger object

    Returns:
        fat_output          (np.ndarray)        : Fat pulse consumption
        final_usages        (np.ndarray)        : Final fat usage boxes
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('fat_pulse_filter')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking a deepcopy of input data to keep local instances

    input_data = deepcopy(in_data)

    # Initialize the final box usages as empty

    final_usages = np.empty(shape=(1, usages.shape[1]))

    # Extract params for max number of runs and duration in hours

    max_runs = wh_config['thermostat_wh']['estimation']['max_runs']
    max_hours = wh_config['thermostat_wh']['estimation']['max_hours']

    # Max duration converted to max window size using sampling rate

    max_duration = max_hours * (Cgbdisagg.SEC_IN_HOUR / wh_config['sampling_rate'])

    # Take unique fat usage days

    unq_days, daily_count = np.unique(usages[:, fat_box_columns['day_ts']], return_counts=True)

    # Counter of number of issues related to run count or duration

    run_check_count, duration_check_count = 0, 0

    # Iterate over each day

    for day in unq_days:
        # Boxes of the current day

        day_usages = usages[usages[:, fat_box_columns['day_ts']] == day]

        # Calculate the number of runs

        valid_runs = (day_usages[:, fat_box_columns['box_fat_energy']] < day_usages[:, fat_box_columns['upper_fat_energy']])
        day_usages = day_usages[valid_runs, :]

        # Check if number of runs criterion is violated

        if np.sum(valid_runs) > max_runs:
            # Run count criterion violated, remove some usages of this day

            # Finding the energy diff of all boxes with respect to ideal energy

            diff = np.abs(day_usages[:, fat_box_columns['box_fat_energy']] -
                          day_usages[:, fat_box_columns['ideal_fat_energy']])

            # Keep the top usages up to number of runs allowed

            best_diff = np.argsort(diff)[:max_runs]
            day_usages = day_usages[best_diff, :]

            run_check_count += 1

        # Calculate the fat usage duration for the day

        total_duration = np.sum(day_usages[:, fat_box_columns['duration']])

        # Check if duration criterion is violated

        if total_duration > max_duration:
            # Duration check violated, keep only the usage upto acceptable duration

            # Find cumulative sum of energy boxes this day

            day_usages = day_usages[day_usages[:, fat_box_columns['duration']].argsort()[::-1]]
            duration_cumsum = np.cumsum(day_usages[:, fat_box_columns['duration']])

            # Stop at the duration just less than the max allowed duration

            stop_idx = np.where(duration_cumsum > max_duration)[0][0]
            day_usages = day_usages[:stop_idx, :]

            duration_check_count += 1

        # Append the left daily usages of this day to final usages

        final_usages = np.vstack((final_usages, day_usages))

    # Remove the top empty row

    final_usages = final_usages[1:, :]

    logger.info('Number of days fixed for run count | {}'.format(run_check_count))
    logger.info('Number of days fixed for run duration | {}'.format(duration_check_count))

    # Iterate over each boxes to populate final fat pulse consumption

    for idx, usage in enumerate(final_usages):
        # Get start and end index of the current box

        start, end = usage[fat_box_columns['start_idx']].astype(int), usage[fat_box_columns['end_idx']].astype(int)

        # Fill energy values between start and end index

        fat_output[start:(end + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = \
            np.min(input_data[start:(end + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX])

    return fat_output, final_usages
