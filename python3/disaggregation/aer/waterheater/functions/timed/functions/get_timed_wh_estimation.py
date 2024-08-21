"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to get final estimation of timed water heater
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_timed_wh_estimation(timed_wh_box, wh_config, logger_base):
    """
    Fix the estimation energy values

    Parameters:
        timed_wh_box        (np.ndarray)    : Input 21-column box data
        wh_config           (dict)          : Timed water heater config params
        logger_base         (dict)          : Logger object

    Returns:
        box_data            (np.ndarray)    : Output 21-column box data
        timed_amplitude     (int)           : Energy per data point
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_estimation')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the amplitude params from config

    amplitude_bar = wh_config['amplitude_bar']
    amplitude_limits = wh_config['amplitude_limits']

    # Taking deepcopy of input data to keep local instances

    box_data = deepcopy(timed_wh_box)

    # Pick non zero energy data indices

    box_idx = (box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0)

    # Get energy values from indices

    all_energy = box_data[box_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    if len(all_energy) == 0:
        # If no non-zero values present

        logger.info('No valid energy data found for adjustment | ')

        return box_data, 0

    # Plot energy histogram with bucket size 100

    all_edges = np.arange(0, np.max(all_energy) + 100, 100)
    all_val, all_edges = np.histogram(all_energy, bins=all_edges)

    # Select bucket with highest frequency

    all_idx = np.argmax(all_val)

    # Take 2 adjacent buckets (if available)

    all_idx = np.r_[all_idx - 1, all_idx, all_idx + 1]
    all_idx = np.fmax(all_idx, 0)
    all_idx = np.fmin(all_idx, len(all_val) - 1)
    all_idx = np.unique(all_idx)

    # Bucket centres are the energy amplitude

    all_mid_edges = (all_edges[:-1] + all_edges[1:]) / 2

    # Take the weighed mean of all selected buckets above to get thin_peak_energy

    timed_amplitude = np.sum(all_mid_edges[all_idx] * all_val[all_idx]) / np.sum(all_val[all_idx])

    logger.info('Timed water heater thin_peak_energy | {}'.format(timed_amplitude))

    # Select all the valid energy data points with respect thin_peak_energy

    valid_energy = all_energy[(all_energy > amplitude_limits[0] * timed_amplitude) &
                              (all_energy < amplitude_limits[1] * timed_amplitude)]

    # Define the maximum energy per data point

    mu_max = np.percentile(valid_energy, amplitude_bar)

    logger.info('Maximum allowed energy for timed water heater | {}'.format(mu_max))

    # Subset the energy data based on max energy limit

    final_energy = np.fmin(all_energy, mu_max)

    # Update the box_data with new estimation

    box_data[box_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = final_energy

    return box_data, timed_amplitude
