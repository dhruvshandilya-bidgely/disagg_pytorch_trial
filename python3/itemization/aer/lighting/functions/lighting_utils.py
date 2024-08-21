
"""
Author - Nisha Agarwal
Date - 10th Nov 2020
Utils functions for lighting module
"""

# Import python packages

import copy
import numpy as np


def check_to_remove_min(clean_days_score, vacation, lighting_config):

    """
    check if to remove minimum consumption value from input data

    Parameters:
        clean_days_score            (np.ndarray)      : clean day score for all days
        vacation                    (np.ndarray)      : vacation boolean array
        lighting_config             (dict)            : dict containing lighting config values

    Returns:
        remove_min                  (bool)            : If true, remove minimum non zero consumption from input data
    """

    score_limit = lighting_config.get('remove_min_config').get('score_limit')
    days_fraction = lighting_config.get('remove_min_config').get('days_fraction')

    # This function was added in order to check whether the baseload should be removed before lighting estimation
    # It is usually not removed in the scenarios where there are clean days score is more towards higher side

    non_vac_days = np.sum(np.logical_not(vacation))

    bool1 = np.sum(clean_days_score[np.logical_not(vacation)] > score_limit[0]) > days_fraction[0] * non_vac_days and \
            np.sum(clean_days_score[np.logical_not(vacation)] > score_limit[1]) > days_fraction[1] * non_vac_days

    bool2 =  np.sum(clean_days_score[np.logical_not(vacation)] > score_limit[2]) >  days_fraction[2] * non_vac_days

    bool3 =  np.sum(clean_days_score[np.logical_not(vacation)] > score_limit[3]) > days_fraction[3] * non_vac_days

    remove_min = not (bool1 or bool2 or bool3)

    return remove_min


def postprocess_sleep_hours(sleep_hours, lighting_config):

    """
    fills sleeping hours, for users where 0 activity has been detected

    Parameters:
        sleep_hours                 (np.ndarray)      : sleep hours of user
        lighting_config             (dict)            : dict containing lighting config values

    Returns:
        sleep_hours                 (np.ndarray)      : updated sleep hours of user
    """

    if np.sum(sleep_hours) == 0:
        sleep_hours[lighting_config.get("estimation_config").get("default_morn")] = 1
        sleep_hours[lighting_config.get("estimation_config").get("default_eve")] = 1

    return sleep_hours


def remove_daily_min(input_data, exclude_zero=1):

    """
    Remove daily minimum from input data

    Parameters:
        input_data          (np.ndarray)      : day input data
        exclude_zero        (int)             : boolean for excluding zero for removing minimum consumption

    Returns:
        input_data_copy     (np.ndarray)      : day input data
    """

    # remove minimum value from input data

    input_data_copy = copy.deepcopy(input_data)

    min_data = np.nanmin(input_data_copy) if (not exclude_zero or (not len(np.nonzero(input_data_copy)[0]))) \
        else np.min(input_data_copy[np.nonzero(input_data_copy)])

    input_data_copy = input_data_copy - min_data

    input_data_copy[input_data_copy < 0] = 0

    return input_data_copy
