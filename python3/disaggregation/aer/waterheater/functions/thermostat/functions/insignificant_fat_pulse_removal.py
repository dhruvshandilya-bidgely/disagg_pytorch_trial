"""
Author - Sahana M
Date - 20/05/2021
This remove fat pulses occuring in insignificant hours
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def remove_insignificant_fat_pulses(start_hod_count, start_hod_count_yearly, year_round_max_hod, usages, unique_days,
                                    missed_fat_input, wh_config):
    """
    Parameters
    start_hod_count            (ndarray)        : Contains the % of days with start boxes at each hour in the bill cycle
    start_hod_count_yearly     (ndarray)        : Contains the % of days with start boxes at each hour in the whole year
    year_round_max_hod         (ndarray)        : Contains the % of days with start boxes at each hour rounded in the whole year
    usages                     (ndarray)        : Contains the information about the each box (epoch, start time, end time)
    unique_days                (ndarray)        : Contains info about the unique days and the no. of fat pulses each day
    missed_fat_input           (ndarray)        : 21 column matrix contains estimated fat pulse
    wh_config                  (dict)           : Dictionary containing WH configurations

    Returns
    valid_fat_run               (ndarray)       : Array denoting whether the corresponding fat pulse is valid or not
    """

    # Extract the necessary variables

    night_hours = wh_config['thermostat_wh']['estimation']['night_hours']
    year_hod_thr = wh_config['thermostat_wh']['estimation']['year_hod_thr']
    start_hod_thr = wh_config['thermostat_wh']['estimation']['start_hod_thr']
    start_hod_min_thr = wh_config['thermostat_wh']['estimation']['start_hod_min_thr']
    max_fat_runs = wh_config['thermostat_wh']['estimation']['max_fat_runs']

    invalid_night_hrs = []
    for index in night_hours:
        if start_hod_count[index] > (start_hod_min_thr * start_hod_count_yearly[index]) or start_hod_count[index] < (
                start_hod_min_thr * start_hod_count_yearly[index]):
            invalid_night_hrs.append(index)

    valid_fat_run = []
    for index in range(len(usages)):

        no_of_runs_index = np.where(unique_days[0, :] == usages[index, 0])
        no_of_runs = unique_days[1, no_of_runs_index]

        # get the hour of day for this run

        hod = int(missed_fat_input[int(usages[index, 1]), Cgbdisagg.INPUT_HOD_IDX])

        # if more then 3 runs a day, remove the insignificant pulses

        if no_of_runs >= max_fat_runs:

            if (start_hod_count[hod] > (year_hod_thr * year_round_max_hod) and
                start_hod_count[hod] > start_hod_thr * np.max(start_hod_count)) and \
                (not np.isin(hod, invalid_night_hrs)):
                valid_fat_run.append(1)
            else:
                valid_fat_run.append(0)

        # remove invalid night hours

        elif np.isin(hod, invalid_night_hrs):
            valid_fat_run.append(0)

        else:
            valid_fat_run.append(1)

    return valid_fat_run
