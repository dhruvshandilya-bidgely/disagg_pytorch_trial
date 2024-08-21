"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module filters the bill cycle / month with abormally high / low fat pulse consumption
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.maths_utils import moving_sum
from python3.utils.maths_utils.maths_utils import convolve_function
from python3.utils.maths_utils.rolling_function import rolling_function

fat_box_columns = {
    'day_ts': 0,
    'start_idx': 1,
    'end_idx': 2,
    'valid_fat_run': 3
}


def filter_low_consumption_bill_cycle(in_data, wh_config, logger_base):
    """
    Checking if a bill cycle has an abnormal fraction of consumption

    Parameters:
        in_data             (np.ndarray)        : Input fat pulse data
        wh_config           (dict)              : Config params
        logger_base         (dict)              : Logger object

    Returns:
        fat_data            (np.ndarray)        : Updated fat pulse data
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('bill_cycle_filter')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Defining columns for filtering out the bill cycles

    bill_cycle_ts_col = 0
    fat_consumption_col = 1
    final_validity_col = 2

    # Taking a deepcopy of fat data to keep local instances

    fat_data = deepcopy(in_data)

    # Extract fat consumption limit from config

    num_bill_cycles = wh_config['thermostat_wh']['estimation']['num_compare_bc']
    consumption_limit = wh_config['thermostat_wh']['estimation']['consumption_limit']

    # Get unique bill cycles and its timestamps

    bill_cycle_ts, bill_cycle_idx = np.unique(fat_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_inverse=True)

    # Initialize the fractions table (fat data out of max consumption)

    bill_cycle_fractions = np.hstack([bill_cycle_ts.reshape(-1, 1), np.zeros((len(bill_cycle_ts), 2))])

    # Get fat consumption at each bill cycle

    bill_cycle_fractions[:, fat_consumption_col] = np.bincount(bill_cycle_idx,
                                                               fat_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    consumptions = bill_cycle_fractions[:, fat_consumption_col]

    # Number of bill cycles to compare with (expected 3, works with less if not present)

    bill_cycle_to_use = np.min([num_bill_cycles, len(bill_cycle_ts)])
    max_consumption = np.mean(np.sort(consumptions)[::-1][:bill_cycle_to_use])

    # Filter out bill cycles with abnormally low consumption as compared to top 3

    invalid_bill_cycles = consumptions < (consumption_limit * max_consumption)
    bill_cycle_fractions[:, final_validity_col] = invalid_bill_cycles.astype(int)

    # Removing the invalid bill cycle consumption

    for i, row in enumerate(bill_cycle_fractions):
        # Check for each bill cycle if valid

        if row[final_validity_col] == 1:
            fat_data[fat_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == row[bill_cycle_ts_col],
                     Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    logger.info('Bill cycles removed are | {}'.format(bill_cycle_ts[invalid_bill_cycles].astype(int)))

    return fat_data


def filter_noise_fat_usages(fat_output, wh_config):
    """
    Parameters:
        fat_output          (np.ndarray)        : Fat pulse consumption
        wh_config           (dict)              : Water heater params

    Returns:
        fat_signal          (np.ndarray)        : Filtered fat pulse consumption
    """

    # Taking a deepcopy of fat data to keep local instances

    fat_signal = deepcopy(fat_output)

    # Retrieve the params from config

    estimation_config = wh_config['thermostat_wh']['estimation']

    fat_bound_limit = estimation_config['fat_bound_limit']
    noise_window_size = estimation_config['noise_window_size']

    # Extract fat pulse energy values

    fat_energy_idx = fat_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0

    # Histogram of fat energy values in 20 bins

    count, edges = np.histogram(fat_signal[fat_energy_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX], bins=20)

    # Take the middle value of each bin from edge values

    bin_energy = (edges[1:] + edges[:-1]) / 2

    # Calculate the ideal fat pulse energy using all bins (except first and last)

    fat_peak_energy = np.sum(bin_energy[1:-1] * count[1:-1]) / np.sum(count[1:-1])

    # Define the lower fat pulse bound

    lower_bound = fat_bound_limit * fat_peak_energy

    # Check energy values adjacent to the fat pulse

    side_check = moving_sum(np.r_[0, fat_energy_idx, 0], noise_window_size)

    # If adjacent values significantly different than fat pulse and pulse energy far from ideal size, remove it

    invalid_fat_idx = (side_check[2:] == 1) & (fat_energy_idx == 1) & \
                      (fat_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] < lower_bound)

    fat_signal[invalid_fat_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    return fat_signal
