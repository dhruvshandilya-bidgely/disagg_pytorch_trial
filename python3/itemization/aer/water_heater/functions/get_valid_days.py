"""
Author - Sahana M
Date - 3/3/2021
Function returns the valid days which include buffer/transition days along with wh present days
"""

# Import python packages

from copy import deepcopy


def get_valid_days(valid_days_bool, padding_days, debug):

    """
    Function return valid days boolean which contain buffer/transition days + wh potential days
    Args:
        valid_days_bool             (np.array)      : Boolean array contains wh_present_indexes where wh_potential > 0
        padding_days                (int)           : Number of padding days required
        debug                       (dict)          : Contains all variables required for debugging

    Returns:
        padded_valid_days_bool      (np.array)      : Boolean array containing wh_present_indexes + buffer days padding
    """

    # Extract all the necessary variables required

    trn_start_1 = debug.get('trn_start_1')
    trn_end_1 = debug.get('trn_end_1')
    trn_start_2 = debug.get('trn_start_2')
    trn_end_2 = debug.get('trn_end_2')
    start_idx = debug.get('swh_wh_pot_start')
    end_idx = debug.get('swh_wh_pot_end')

    padded_valid_days_bool = deepcopy(valid_days_bool)

    padding_end = False
    padding_start = False

    # Include buffer days for correlation if present

    if trn_start_1 != 0 and trn_end_1 != 0:
        padded_valid_days_bool[trn_start_1: trn_end_1] = True
        padding_start = True

    if trn_start_2 != 0 and trn_end_2 != 0:
        padded_valid_days_bool[trn_start_2: trn_end_2] = True
        padding_end = True

    # If buffer days not present do padding on the wh potential day ends

    if (not padding_start) and start_idx != 0 and end_idx != 0:
        padded_valid_days_bool[start_idx: min(start_idx + padding_days, len(padded_valid_days_bool))] = True

    if (not padding_end) and start_idx != 0 and end_idx != 0:
        padded_valid_days_bool[max(end_idx - padding_days, 0): end_idx] = True

    return padded_valid_days_bool
