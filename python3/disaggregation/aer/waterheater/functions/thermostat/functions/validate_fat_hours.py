"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module validates the active usage hours (fat pulse)
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def check_fat_hours(fat_hours, wh_config, logger):
    """
    Parameters:
        fat_hours           (np.ndarray)    : Fat usage hours
        wh_config           (dict)          : Config params
        logger              (logger)        : Logger object

    Returns:
        final_fat_hours     (np.ndarray)    : Filtered fat usage hours
    """

    # Making sure the fat hours array is numpy array

    fat_hours = np.array(fat_hours)

    # Take valid fat hours (between 0 and 23)

    fat_hours = fat_hours[(fat_hours >= 0) & (fat_hours < Cgbdisagg.HRS_IN_DAY)]

    # Get night hours bound from config

    night_hours = wh_config['thermostat_wh']['estimation']['night_hours']

    # Extract night start and end hour

    night_start = night_hours[0]
    night_end = night_hours[-1]

    # Get night buffer and proportion threshold from config

    night_buffer = wh_config['thermostat_wh']['estimation']['night_buffer']
    night_proportion = wh_config['thermostat_wh']['estimation']['night_proportion']

    # Calculate the proportion of fat hours in night hours

    night_fraction = np.sum(np.isin(fat_hours, night_hours)) / len(fat_hours)

    # Get fat hours that didn't occur in night hours

    non_night_hours = fat_hours[~np.isin(fat_hours, night_hours)]

    # Check if night usage hours proportion high

    if night_fraction >= night_proportion:
        logger.info('High proportion of night hours in fat hours | ')

        # Get difference of fat hours with night start and retain the ones in buffer

        night_diff_start = np.abs(fat_hours - night_start)
        valid_hours_idx_start = np.where(night_diff_start <= night_buffer)[0]

        # Get difference of fat hours with night end and retain the ones in buffer

        night_diff_end = np.abs(fat_hours - night_end)
        valid_hours_idx_end = np.where(night_diff_end <= night_buffer)[0]

        # Combine the valid hours with respect to start and end

        valid_hours_idx = np.unique(np.append(valid_hours_idx_start, valid_hours_idx_end))

        # Combine valid night hours with day usage hours

        final_fat_hours = np.unique(np.append(fat_hours[valid_hours_idx], non_night_hours))

    else:
        logger.info('Normal proportion of night hours in fat hours | ')

        final_fat_hours = fat_hours

    return final_fat_hours
