"""
Author - Sahana M
Date - 20/07/2021
The module is used to get the derived features
"""

# Import python packages
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg


def get_auc_wh_pot_corr(wh_potential, auc_line, vacation_days_bool):
    """
    Function is used to get the correlation between Area under the curve (usage) and WH potential
    Parameters:
        wh_potential            (np.ndarray)    : Day level WH potential
        auc_line               (np.ndarray)     : Area under the curve of box fit data
        vacation_days_bool      (np.ndarray)    : Contains vacation days info

    Returns:
        auc_wh_pot_corr         (float)         : AUC & WH potential correlation
    """

    # Filter out the vacation days
    auc_line_wh_days = auc_line[~vacation_days_bool]
    wh_potential_wh_days = wh_potential[~vacation_days_bool]

    if np.sum(auc_line_wh_days) > 0:

        # If wh potential is not the same year round
        if not np.all(wh_potential_wh_days == wh_potential_wh_days[0]):
            auc_wh_pot_corr = np.round(np.corrcoef(auc_line_wh_days, wh_potential_wh_days)[0, 1], 2)

        # Else to avoid nan value change one of the wh potential values by a small amount
        else:
            wh_potential_wh_days[0] = max(wh_potential_wh_days[0] - 0.01, 0)
            auc_wh_pot_corr = np.round(np.corrcoef(auc_line_wh_days, wh_potential_wh_days)[0, 1], 2)
    else:
        auc_wh_pot_corr = 0

    return auc_wh_pot_corr


def get_deviations(line):
    """This function calculates the normalised deviation in the data
    Parameters:
        line            (np.ndarray)    : Line to perform normalised deviation
    Return:
        line_std        (np.ndarray)    : Normalised line
    """

    if np.sum(line) > 0:
        line_std = np.round((np.std(line) / np.mean(line)), 2)
    else:
        line_std = 0

    # Normalising with tanh
    line_std = np.round(((1 - (np.tanh((line_std - 3)/2) + 1))+1)/2, 2)

    return line_std


def get_max_median_consistency(max_consistency, consistency_arr, start_time, end_time):
    """
    This function calculates the max median consistency score
    Parameters:
        max_consistency             (float)     : Max consistency score
        consistency_arr             (array)     : Array containing time level consistency values
        start_time                  (int)       : Start time of the sequence
        end_time                    (int)       : End time of the sequence

    Returns:
        max_median_consistency      (float)     : max_median_consistency score
    """
    if max_consistency > 0:
        median_consistency = np.round(np.median(consistency_arr[start_time: end_time]), 2)
        max_median_consistency = np.round(1 - (1 - (max_consistency - median_consistency)), 2)
    else:
        max_median_consistency = 0

    return max_median_consistency


def get_tb_probability(start_time, end_time, wh_config):
    """
    Returns the time bands probability
    Parameters:
        start_time                  (int)       : Start time of the sequence
        end_time                    (int)       : End time of the sequence
        wh_config                   (dict)      : WH configurations dictionary

    Returns:
        final_tb_prob               (float)     : Final tb probability
    """

    factor = wh_config.get('factor')
    tb_probability_default = wh_config.get('tb_probability_default')

    # for non default tb probability pilots calculate the tb probability using tb_probability_default
    if start_time != -1:
        start = int(start_time / factor)
        end = int(end_time / factor) + 1
        tb_probability_value = tb_probability_default[start: end]
        max_prob = np.max(tb_probability_value)
        median_prob = np.median(tb_probability_value)
        final_tb_prob = np.round(((max_prob + median_prob) / 2), 2)
    else:
        final_tb_prob = 0

    return final_tb_prob


def convert_n_division_to_24_hour(in_data):
    """
    This function is used to convert n dimension data to 24 hour data
    Parameters:
        in_data         (np.ndarray)    : Input data
    Returns:
        converted_arr   (np.ndarray)    : Converted array
    """

    converted_arr = np.full(shape=Cgbdisagg.HRS_IN_DAY, fill_value=False)
    for i in range(0, len(in_data), 2):
        converted_arr[int(i/2)] = np.sum(in_data[i:i+2])

    return converted_arr
