"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to get operational load of smb
"""

# Import python packages
import copy
import numpy as np

# Import functions from within the project
from python3.disaggregation.aes.hvac.get_smb_params import get_smb_params


def get_hour_wise_operational(valid_day_residue, operational_map):

    """
    Function to get initial reference-operational-load at hour level

    Parameters:

        valid_day_residue   (np.ndarray)          : Array containing residue consumption left at hour level
        operational_map     (pd.DataFrame)        : Contains operational consumption

    Returns:

        operational_map     (pd.DataFrame)        : Contains operational consumption
        hour_medians        (np.ndarray)          : Array containing hour level median residue left
        operational_values  (np.ndarray)          : Array containing operational consumption data

    """

    smb_params = get_smb_params()

    # getting hour level most representative base operational loads
    hour_medians = valid_day_residue[valid_day_residue > 0].median(axis=0).values

    # assigning hour medians as zero to invalid days
    hour_medians_invalid = (np.nan_to_num(valid_day_residue > 0).sum(axis=0) / valid_day_residue.shape[0]) < smb_params.get('operational').get('hour_median_condition')
    hour_medians[hour_medians_invalid] = 0

    # checking low level limit of operational load allowed, for failsafe
    hour_stds = valid_day_residue[valid_day_residue > 0].std(axis=0).values
    hour_median_low_limit = hour_medians - smb_params.get('operational').get('hour_median_low_lim') * hour_stds

    # ensuring no nan values flow in operational load
    hour_medians = np.nan_to_num(hour_medians)
    operational_values = copy.deepcopy(operational_map.values)

    # assigning operational loads and updating in operational map
    for idx in range(len(hour_medians)):

        hour_median_low_limit_idx = hour_median_low_limit[idx]
        curr_hour_med = valid_day_residue[valid_day_residue > hour_median_low_limit_idx].median(axis=0).values[idx]

        operational_values[operational_values[:, idx] > curr_hour_med, idx] = curr_hour_med

    operational_map[:] = operational_values

    return operational_map, hour_medians, operational_values
