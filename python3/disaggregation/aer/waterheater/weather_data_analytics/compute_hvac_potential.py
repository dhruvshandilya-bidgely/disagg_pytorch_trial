"""
Author: Mayank Sharan
Created: 13-Jul-2020
Compute HVAC potential at each data point
"""

# Import python packages

import numpy as np

# Import project functions and classes

from python3.disaggregation.aer.waterheater.weather_data_analytics.math_utils import find_seq
from python3.disaggregation.aer.waterheater.weather_data_analytics.math_utils import nan_percentile


def compute_hvac_potential(day_wise_data_dict, season_detection_dict):

    """
    Compute the heating and cooling potential at data point level
    Parameters:
        day_wise_data_dict      (dict)          : Dictionary containing all day wise data matrices
        season_detection_dict   (dict)          : Dictionary containing season detection data
    Returns:
        hvac_potential_dict     (dict)          : Dictionary containing all outputs regarding hvac potential
    """

    # Extract information from the dictionaries

    day_temp_data = day_wise_data_dict.get('temp')
    day_fl_data = day_wise_data_dict.get('fl')
    day_prec_data = day_wise_data_dict.get('prec')
    day_snow_data = day_wise_data_dict.get('snow')

    valid_season_bool = season_detection_dict.get('model_info_dict').get('valid_season_bool')
    s_label = season_detection_dict.get('s_label')

    max_winter_temp = season_detection_dict.get('max_winter_temp')
    max_tr_temp = season_detection_dict.get('max_tr_temp')

    class_name = season_detection_dict.get('class_name')

    # Attempt scoring each day with a heating potential

    heating_max_thr = 'NA'
    heating_pot = np.full_like(day_temp_data, fill_value=0.0)

    # Check if a valid winter exists

    if valid_season_bool[0]:

        # Add an offset based on the koppen class

        offset_dict = {
            'A': 0,
            'Bh': 0,
            'Bk': 3,
            'Ch': -5,
            'C': 0,
            'Ck': 3,
            'D': 5,
        }

        heating_max_thr = min(max_winter_temp + offset_dict.get(class_name), 53.6)

        heating_pot = heating_max_thr - day_fl_data
        heating_pot[heating_pot <= 0] = 0

        # Convert the raw temperature difference to a potential between 0 and 1

        exp_heating_pot = 0.184 * np.exp(heating_pot / 9)
        sqrt_heating_pot = np.sqrt(heating_pot / 36)

        low_diff_bool = np.logical_and(heating_pot <= 9, heating_pot > 0)
        heating_pot[low_diff_bool] = exp_heating_pot[low_diff_bool]
        heating_pot[~low_diff_bool] = sqrt_heating_pot[~low_diff_bool]

        # Add snowfall and precipitation bonuses

        pr_bonus = 1 + 0.05 * (1 - np.exp(-day_prec_data / 0.025))
        heating_pot = np.multiply(pr_bonus, heating_pot)

        sn_bonus = 1 + 0.1 * (1 - np.exp(-day_snow_data / 0.025))
        heating_pot = np.multiply(sn_bonus, heating_pot)

        heating_pot[heating_pot > 1] = 1

        # Exclude days with high inertia

        day_inertia_score = (np.multiply(np.sign(s_label), np.sqrt(np.abs(s_label))) + 1) / 4

        day_high_pot = nan_percentile(heating_pot, 75, axis=1)
        day_inertia_score = day_inertia_score + (0.5 * (1 - day_high_pot))

        is_heating_day = day_inertia_score <= 0.5

        heat_days_seq = find_seq(is_heating_day, min_seq_length=5)
        heat_days_seq = heat_days_seq[heat_days_seq[:, 0] == 1, :]

        is_heating_day[:] = False

        for idx in range(heat_days_seq.shape[0]):
            curr_seq = heat_days_seq[idx, :]
            is_heating_day[int(curr_seq[1]): int(curr_seq[2]) + 1] = True

        heating_pot[~is_heating_day, :] = 0

        # Assign a base heating potential

        base_heating_pot = 0.2 * (heating_pot > 0)
        heating_pot = base_heating_pot + 0.8 * heating_pot

    # Attempt scoring each day with a cooling potential

    cooling_min_thr = 'NA'
    cooling_pot = np.full_like(day_temp_data, fill_value=0.0)

    # Check if a valid winter exists

    if valid_season_bool[2]:

        # Add an offset based on the koppen class

        offset_dict = {
            'A': 0,
            'Bh': 0,
            'Bk': 0,
            'Ch': 2,
            'C': 3,
            'Ck': 3,
            'D': 5,
        }

        cooling_min_thr = max(max_tr_temp + offset_dict.get(class_name), 75.2)

        cooling_pot = day_fl_data - cooling_min_thr
        cooling_pot[cooling_pot <= 0] = 0

        # Convert the raw temperature difference to a potential between 0 and 1

        exp_cooling_pot = 0.184 * np.exp(cooling_pot / 9)
        sqrt_cooling_pot = np.sqrt(cooling_pot / 36)

        low_diff_bool = np.logical_and(cooling_pot <= 9, cooling_pot > 0)
        cooling_pot[low_diff_bool] = exp_cooling_pot[low_diff_bool]
        cooling_pot[~low_diff_bool] = sqrt_cooling_pot[~low_diff_bool]

        # Add snowfall and precipitation bonuses

        pr_penalty = 1 - 0.10 * (1 - np.exp(-day_prec_data / 0.025))
        cooling_pot = np.multiply(pr_penalty, cooling_pot)

        cooling_pot[cooling_pot > 1] = 1

        # Exclude days with high inertia

        day_inertia_score = (np.multiply(-np.sign(s_label), np.sqrt(np.abs(s_label))) + 1) / 4

        day_high_pot = nan_percentile(cooling_pot, 84, axis=1)
        day_inertia_score = day_inertia_score + (0.5 * (1 - day_high_pot))

        is_cooling = day_inertia_score <= 0.5

        cool_days_seq = find_seq(is_cooling, min_seq_length=5)
        cool_days_seq = cool_days_seq[cool_days_seq[:, 0] == 1, :]

        is_cooling[:] = False

        for idx in range(cool_days_seq.shape[0]):
            curr_seq = cool_days_seq[idx, :]
            is_cooling[int(curr_seq[1]): int(curr_seq[2]) + 1] = True

        cooling_pot[~is_cooling, :] = 0

        # Assign a base cooling potential

        base_cooling_pot = 0.2 * (cooling_pot > 0)
        cooling_pot = base_cooling_pot + 0.8 * cooling_pot

    # Populate the return dictionary

    hvac_potential_dict = {
        'heating_pot': heating_pot,
        'heating_max_thr': heating_max_thr,
        'cooling_pot': cooling_pot,
        'cooling_min_thr': cooling_min_thr,
    }

    return hvac_potential_dict
