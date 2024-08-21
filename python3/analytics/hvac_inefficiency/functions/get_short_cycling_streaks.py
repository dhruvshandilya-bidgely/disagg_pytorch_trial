"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains code for finding short cycling regions in HVAC consumption
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.analytics.hvac_inefficiency.configs.init_cycling_based_config import get_short_cycling_config


def get_short_cycle_array(end_index, loop_idx, start_index, counter, short_cycling_duration_limit, short_cycling_array):
    """
    Function to get short cycle array

    Parameters:
        end_index                       (int)       : end_index
        loop_idx                        (int)       : loop index
        start_index                     (int)       : start_index
        counter                         (int)       : counter
        short_cycling_duration_limit    (int)       : short_cycling_duration_limit
        short_cycling_array             (np.ndarray): short_cycling_array

    Returns:
        end_index                       (int)       : end_index
        short_cycling_array             (np.ndarray): short_cycling_array
    """

    if (end_index is None) & (start_index is not None):
        end_index = loop_idx
        if counter <= short_cycling_duration_limit:
            short_cycling_array[start_index:end_index] = 0

    return end_index, short_cycling_array


def get_counter_start_idx(counter, start_index, i):

    """
    Function to return counter and start index

    Parameters:

        counter         (int)   : Counter
        start_index     (int)   : Start index
        i               (int)   : index

    Returns:
        counter         (int)   : Counter
        start_index     (int)   : Start index
    """

    counter += 1

    if start_index is None:
        start_index = i

    return counter, start_index


def handle_loop(counter, short_cycling_duration_limit, short_cycling_array, start_index, end_index, streak_counter,
                energy_array, average_energy):
    """
    Function to handle loop

    Parameters:
        counter                         : Counter
        short_cycling_duration_limit    : Short cycling duration limit
        short_cycling_array             : Short cycling array
        start_index                     : Start index
        end_index                       : End index
        streak_counter                  : Streak counter
        energy_array                    : Energy array
        average_energy                  : Average energy

    Returns:
        short_cycling_array             : Short cycling array
        streak_counter                  : Streak counter
        average_streak_energy           : Average_ treak energy
        average_energy                  : Average energy

    """

    if counter < short_cycling_duration_limit:
        short_cycling_array[start_index:end_index] = 0
        average_streak_energy = None
    else:
        streak_counter += 1
        average_streak_energy = np.nanmedian(energy_array[start_index:end_index])
        average_energy[start_index:end_index] = average_streak_energy

    return short_cycling_array, streak_counter, average_streak_energy, average_energy


def get_short_cycling_streaks(short_cycling_array, total_consumption, sampling_rate):

    """
        Get short cycling streaks based on total consumption

        Parameters:
            short_cycling_array     (numpy.ndarray)          array containing consumption meeting sc criterion
            total_consumption       (numpy.ndarray)          array containing total consumption
            sampling_rate           (int)                    sampling rate of the user
        Returns:
            short_cycling_array     (numpy.ndarray)          valid short cycling consumption array
            average_energy          (numpy.ndarray)          avergae consumption of short cycling array
    """

    start_index = None
    end_index = None
    counter = 0
    zero_counter = 0
    previous_value = 0
    streak_counter = 0
    average_energy = np.zeros_like(short_cycling_array, dtype=np.float)
    average_energy[:] = np.nan
    energy_array = total_consumption.ravel()

    config = get_short_cycling_config(sampling_rate)

    short_cycling_duration_limit = config.get('short_cycling_duration_limit')
    max_zeros_allowed = config.get('max_zeros_allowed')

    i = 0

    for i in range(0, short_cycling_array.shape[0]):

        if short_cycling_array[i] != 0:

            counter, start_index = get_counter_start_idx(counter, start_index, i)

        else:

            zero_counter += 1

            if (zero_counter > max_zeros_allowed) or (previous_value == 0):

                if (end_index is None) & (start_index is None):
                    counter = 0

                elif (end_index is None) & (start_index is not None):

                    end_index = i

                    short_cycling_array, streak_counter, average_streak_energy, average_energy = \
                        handle_loop(counter, short_cycling_duration_limit, short_cycling_array, start_index, end_index,
                                    streak_counter, energy_array, average_energy)

                    start_index = None
                    end_index = None
                    counter = 0
                    zero_counter = 0

        previous_value = short_cycling_array[i]

    end_index, short_cycling_array = get_short_cycle_array(end_index, i, start_index, counter,
                                                           short_cycling_duration_limit, short_cycling_array)

    return short_cycling_array, average_energy
