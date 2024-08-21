"""
Author - Sahana M
Date - 4/3/2021
Identifies the amplitude range of the seasonal wh
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.utils.maths_utils.maths_utils import rotating_avg


def get_energy_range(cleaned_data, start_time, end_time, new_wh_present_idx, wrong_days_idx, debug, seasonal_config):

    """
    Calculates the amplitude range of seasonal wh for this particular time zone
    Args:
        cleaned_data            (np.ndarray)    : Contains (day x time) shape consumption matrix
        start_time              (int)           : Denotes the start time index of the time zone in consideration
        end_time                (int)           : Denotes the end time index of the time zone in consideration
        new_wh_present_idx      (np.array)      : Boolean array denoting + wh potential with padding
        wrong_days_idx          (np.array)      : Boolean array denoting erroneous days in + wh potential days
        debug                   (dict)          : Contains all variables required for debugging
        seasonal_config         (dict)          : Dictionary containing all needed configuration variables

    Returns:
        return_list             (list)          : List containing energy ranges and new wh present indexes

    """

    # Initialise all data required

    in_data = deepcopy(cleaned_data)
    padding_days = seasonal_config.get('config').get('padding_days')
    trn_start_1 = debug.get('trn_start_1')
    trn_start_2 = debug.get('trn_start_2')
    trn_end_1 = debug.get('trn_end_1')
    trn_end_2 = debug.get('trn_end_2')
    factor = debug.get('factor')
    start_idx = debug.get('swh_wh_pot_start')
    end_idx = debug.get('swh_wh_pot_start')
    buffer_amp_perc = seasonal_config['config']['buffer_amp_perc']
    wh_amp_perc = seasonal_config['config']['wh_amp_perc']
    energy_diff_perc = seasonal_config['config']['energy_diff_perc']
    amp_range_perc = seasonal_config['config']['amp_range_perc']
    min_base_amp = seasonal_config['config']['min_base_amplitude']
    max_base_energy = seasonal_config['config']['max_base_energy']
    max_amplitude = seasonal_config['config']['max_amplitude']

    # To identify energy, first get buffer days energy

    buffer_day_1 = in_data[trn_start_1:trn_end_1, start_time:end_time]
    buffer_day_2 = in_data[trn_start_2:trn_end_2, start_time:end_time]
    buffer_days = np.r_[buffer_day_1, buffer_day_2]

    # If buffer days are not found then use days at the ends of wh_potential days

    if not len(buffer_days):
        buffer_day_1 = in_data[start_idx: min(start_idx + padding_days, len(new_wh_present_idx)), start_time: end_time]
        buffer_day_2 = in_data[max(end_idx - padding_days, 0):end_idx, start_time:end_time]
        buffer_days = np.r_[buffer_day_1, buffer_day_2]

    # Remove the wrong days from wh_potential days to not affect the energy calculation

    right_days = ~wrong_days_idx & new_wh_present_idx
    wh_pot_days = in_data[right_days, start_time:end_time]

    # Get the 80th percentile energy in buffer days

    buffer_days_energy = np.percentile(buffer_days, buffer_amp_perc, axis=0)

    # Get the 80th percentile energy in wh potential days

    if not len(wh_pot_days):
        wh_days_energy = np.nanpercentile(wh_pot_days, wh_amp_perc, axis=0)
        wh_days_energy = np.nan_to_num(wh_days_energy)
    else:
        wh_days_energy = np.percentile(wh_pot_days, wh_amp_perc, axis=0)

    # Calculate the energy difference between buffer days and wh potential days

    comparison = np.c_[wh_days_energy, buffer_days_energy]

    # Get the 90th percentile of energy difference after smoothing

    energy_diff = np.percentile(rotating_avg((comparison[:, 0] - comparison[:, 1]), 3), energy_diff_perc)

    # Scale the energy_diff to hourly level

    energy_diff = energy_diff * factor

    # Get the base difference between buffer days and wh potential days

    base_diff = wh_pot_days - buffer_days_energy

    base_diff[base_diff < 0] = 0

    # Get the array of amplitude with minimum amplitude criteria satisfied

    amplitude_arr = base_diff[base_diff >= (min_base_amp / factor)]

    # Calculate the epoch level energy

    epoch_energy = energy_diff / factor

    # Get all the energy below the epoch energy and above to calculate energy range

    low_range = amplitude_arr[amplitude_arr < epoch_energy]
    high_range = amplitude_arr[amplitude_arr >= epoch_energy]

    # Low amplitude is 70th percentile of low_range values

    if len(low_range) != 0:
        low_amp = np.percentile(low_range, amp_range_perc)
    else:
        low_amp = max(epoch_energy - (min_base_amp/factor), (min_base_amp/factor))

    # High amplitude is 70th percentile of high_range values

    if len(high_range) != 0:
        high_amp = np.percentile(high_range, amp_range_perc)
    else:
        high_amp = min(epoch_energy + (max_base_energy/factor), (max_amplitude/factor))

    # Cap all the values consuming greater than high amp to high amp and less than low amp to 0

    in_data[in_data > high_amp] = high_amp
    in_data[(in_data < low_amp)] = 0

    return_list = [energy_diff, low_amp, high_amp, in_data, new_wh_present_idx]

    return return_list
