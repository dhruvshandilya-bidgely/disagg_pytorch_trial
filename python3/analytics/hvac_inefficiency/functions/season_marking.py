"""
Author - Anand Kumar Singh
Date - 26th June 2021
HVAC inefficiency HSM related operations
"""

# Import functions from within the project
import copy

import numpy as np

from python3.utils.find_runs import find_runs
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def mark_and_analyse_season(s_label_pivot):
    """
        Parameters:
            s_label_pivot       (np.ndarray)    : Array containing season labels
        Returns:
            season dictionary   (dict)          : Dictionary containing season labels
    """

    static_params = hvac_static_params()

    daily_season_mark = s_label_pivot

    # Minimum days to consider a summer or winter season

    minimum_hvac_days = static_params.get('ineff').get('season_min_hvac_days')
    minimum_hvac_days_for_enough_summer = static_params.get('ineff').get('season_min_hvac_days_enough_summer')

    full_summer = 0
    full_winter = 0
    enough_summer = 0
    enough_winter = 0

    if daily_season_mark is None:
        season_dictionary = dict({})
        return season_dictionary

    nan_idx = np.isnan(daily_season_mark)
    daily_season_mark = daily_season_mark[~nan_idx]

    # Denoting summer by 1 and winter -1
    summer_season_mark = copy.deepcopy(daily_season_mark)
    winter_season_mark = copy.deepcopy(daily_season_mark)
    summer_season_mark[summer_season_mark > 0] = 1
    summer_season_mark[summer_season_mark < 0] = 0

    winter_season_mark[winter_season_mark < 0] = -1
    winter_season_mark[winter_season_mark > 0] = 0

    # Find runs

    winter_arr_val, winter_arr_start, winter_arr_length = find_runs(winter_season_mark)
    summer_arr_val, summer_arr_start, summer_arr_length = find_runs(summer_season_mark)
    val_summer_idx = (summer_arr_val == 1) & (summer_arr_length > minimum_hvac_days) | (summer_arr_val == 0)
    val_winter_idx = ((winter_arr_val == -1) & (winter_arr_length > minimum_hvac_days)) | (winter_arr_val == 0)

    winter_string_map = ''
    summer_string_map = ''

    winter_arr_val = winter_arr_val[val_winter_idx].astype(int)
    summer_arr_val = summer_arr_val[val_summer_idx].astype(int)

    winter_arr_length = winter_arr_length[val_winter_idx].astype(int)
    summer_arr_length = summer_arr_length[val_summer_idx].astype(int)

    for idx in range(0, winter_arr_val.shape[0]):
        value = winter_arr_val[idx]
        winter_string_map = winter_string_map + str(value)

    for idx in range(0, summer_arr_val.shape[0]):
        value = summer_arr_val[idx]
        summer_string_map = summer_string_map + str(value)

    if '0-10' in winter_string_map:
        full_winter = 1

    if '010' in summer_string_map:
        full_summer = 1

    val_summer_idx = (summer_arr_val == 1) & (summer_arr_length > minimum_hvac_days_for_enough_summer)
    val_winter_idx = (winter_arr_val == -1) & (winter_arr_length > minimum_hvac_days_for_enough_summer)

    if np.nansum(val_summer_idx) > 0:
        enough_summer = 1
    if np.nansum(val_winter_idx) > 0:
        enough_winter = 1

    season_dictionary = {'full_summer': full_summer,
                         'full_winter': full_winter,
                         'enough_summer': enough_summer,
                         'enough_winter': enough_winter}

    return season_dictionary
