"""
Author - Sahana M
Date - 20/05/2021
Module to remove extra fat and thin pulses occuring in summer/ transition where the consumption in these months is > winter
"""

# Import python packages

import numpy as np

# Import packages from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.post_processing_utils import get_daily_pulses
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.post_processing_utils import control_pulse_count


def fix_summer(seasons, bill_cycle_ts, season_bill_cycle_idx, bill_cycle_idx, fat_output, thin_output,
               wh_config):
    """
    Parameters
    seasons                     (ndarray)             : Array containing estimation information about every billcycle
    bill_cycle_ts               (ndarray)             : Bill cycles as epoch time stamp
    season_bill_cycle_idx       (ndarray)             : Boolean index of season under consideration (summer/ transition)
    bill_cycle_idx              (ndarray)             : Array marking each day with their bill cycle index
    fat_output                  (ndarray)             : 21 column matrix containing fat pulse output
    thin_output                 (ndarray)             : 21 column matrix containing thin pulse output
    wh_config                   (dict)                : WH configuration dictionary
    Returns
    fat_output                  (ndarray)             : 21 column matrix containing fat pulse output
    thin_output                 (ndarray)             : 21 column matrix containing thin pulse output
    """

    # Extract necessary data

    capping_buffer = wh_config['thermostat_wh']['estimation']['capping_buffer']
    consumption_threshold = wh_config['thermostat_wh']['estimation']['wtr_consumption_threshold']

    # Get the average number of thin pulses and fat pulses in summer / intermediate months

    winter_bill_cycles = seasons[seasons[:, 2] == 1, 0]
    winter_bill_cycle_idx = np.where(np.in1d(bill_cycle_ts, winter_bill_cycles))[0]
    winter_bill_cycle_idx = np.in1d(bill_cycle_idx, winter_bill_cycle_idx)

    # Extract the Winter & the Current season fat pulse array

    winter_fat_pulses = fat_output[winter_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    season_fat_pulses = fat_output[season_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Extract the Winter & the Current season thin pulse array

    winter_thin_pulses = thin_output[winter_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    season_thin_pulses = thin_output[season_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Identify the number of fat pulses each day for both Winter & Current season

    daily_winter_fat_pulses, daily_season_fat_pulses, fat_output = \
        get_daily_pulses(winter_fat_pulses, season_fat_pulses, fat_output, winter_bill_cycle_idx,
                         season_bill_cycle_idx)

    # Identify the number of thin pulses each day for both Winter & Current season

    daily_winter_thin_pulses, daily_season_thin_pulses, thin_output = \
        get_daily_pulses(winter_thin_pulses, season_thin_pulses, thin_output, winter_bill_cycle_idx,
                         season_bill_cycle_idx)

    # Check if thin pulses exists in Winter

    if np.max(daily_winter_thin_pulses) > 0:

        # Get the average number of thin pulses throughout the seasons

        avg_winter_thin_pulses = np.mean(daily_winter_thin_pulses[daily_winter_thin_pulses > 0])
        avg_season_thin_pulses = np.mean(daily_season_thin_pulses[daily_season_thin_pulses > 0])

        # If the Summer/Intermediate seasons thin pulses > Winter then removing noise is necessary

        if avg_season_thin_pulses > avg_winter_thin_pulses and np.sum(winter_thin_pulses) > 0:
            # Curb the erroneous thin pulses by restricting the number of thin pulse counts

            thin_output = control_pulse_count(season_bill_cycle_idx, thin_output, daily_season_thin_pulses,
                                              daily_winter_thin_pulses, avg_winter_thin_pulses)

            # Perform amplitude capping

            winter_consumption_median = np.percentile(winter_thin_pulses[winter_thin_pulses > 0], consumption_threshold)
            season_pulses = thin_output[season_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]
            season_pulses[season_pulses > (winter_consumption_median + capping_buffer)] = \
                (winter_consumption_median + capping_buffer)

            thin_output[season_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = season_pulses

    # No thin pulses in Winter, hence unlikely of the occurrence of Summer fat pulses

    else:
        thin_output[season_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    # Check if fat pulse exists in winter

    if np.max(daily_winter_fat_pulses) > 0:

        # Get the average number of fat pulses throughout the seasons

        avg_winter_fat_pulses = np.mean(daily_winter_fat_pulses[daily_winter_fat_pulses > 0])
        avg_season_fat_pulses = np.mean(daily_season_fat_pulses[daily_season_fat_pulses > 0])

        # If the Summer/Intermediate seasons fat pulses > Winter then capping is necessary

        if avg_season_fat_pulses > avg_winter_fat_pulses:
            # Curb the erroneous fat pulses by restricting the number of thin pulse counts

            fat_output = control_pulse_count(season_bill_cycle_idx, fat_output, daily_season_fat_pulses,
                                             daily_winter_fat_pulses, avg_winter_fat_pulses)

            # Perform amplitude capping

            winter_consumption_median = np.percentile(winter_fat_pulses[winter_fat_pulses > 0], consumption_threshold)
            season_pulses = fat_output[season_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]
            season_pulses[season_pulses > (winter_consumption_median + capping_buffer)] = \
                (winter_consumption_median + capping_buffer)

            fat_output[season_bill_cycle_idx, 6] = season_pulses

    # No fat pulses in Winter, hence unlikely of the occurrence of Summer fat pulses

    else:
        fat_output[season_bill_cycle_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    return fat_output, thin_output
