"""
Author - Nikhil Singh Chauhan
Date - 02/11/18
Handy functions for timed water heater module
"""

# Import python packages

import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime, date

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def check_rounding(in_data, timed_config, logger):
    """
    Check if energy data is rounded (for Japan)

    Parameters:
        in_data         (np.ndarray)    : Input 21-column matrix
        timed_config    (dict)          : Timed config params
        logger          (logger)        : Logger object

    Returns:
        timed_config    (dict)          : Updated timed config params
    """

    # Taking deepcopy of input data and consumption values

    input_data = deepcopy(in_data)
    energy = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Extract the energy threshold for rounding check

    min_rounding_bar = timed_config.get('min_rounding_bar')
    rounding_balance = timed_config.get('rounding_balance')

    # Check remainder with minimum rounding bar

    if np.sum(energy % min_rounding_bar) < rounding_balance:
        # If rounding present, update the config

        timed_config['std_thres'] -= timed_config['std_thres_delta']

        logger.info('Rounding present at 100 Watt level | ')

    return timed_config


def default_timed_debug(input_data, debug, valid_pilot):
    """
    Parameters:
        input_data              (np.ndarray)    : Input 21-column matrix
        debug                   (dict)          : Module intermediate stage output
        valid_pilot             (bool)          : Check if valid pilot to run timed wh

    Returns:
        timed_wh_signal         (np.ndarray)    : Timed water heater output
        debug                   (dict)          : Updated debug object
    """
    timed_wh_signal = deepcopy(input_data)
    timed_wh_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    # Saving default values to the debug object

    debug['timed_wh_signal'] = timed_wh_signal[:, :Cgbdisagg.INPUT_DIMENSION]

    debug['timed_hld'] = 0
    debug['timed_wh_amplitude'] = 0

    # Define the default timed debug object

    debug['timed_debug'] = {
        'valid_pilot': valid_pilot,
        'box_data': timed_wh_signal[:, :Cgbdisagg.INPUT_DIMENSION],
        'timed_confidence': 0
    }

    return timed_wh_signal, debug


def get_2d_matrix(input_data, sampling_rate, debug):
    """
    This function take the 21 column data and converts the data into 2D matrix where row represents each day and
    column represents the time of the day
    Args:
        input_data              (np.ndarray)        :  21 column data
        sampling_rate           (float)             :  Sampling rate
        debug                   (dict)              : Debug dictionary
    Returns:
        data_matrix             (np.ndarray)        : 2D days x time format matrix
        row_idx                 (np.array)          : Contains the row index mapping of each data point to its 21 column matrix
        col_idx                 (np.array)          : Contains the col index mapping of each data point to its 21 column matrix
    """

    vacation_output = debug.get('other_output').get('vacation')

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    # Prepare day timestamp matrix and get size of all matrices

    day_ts, row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
    day_ts = np.tile(day_ts, reps=(num_pd_in_day, 1)).transpose()

    # Initialize all 2d matrices with default value of nan except the boolean ones

    data_matrix = np.full(shape=day_ts.shape, fill_value=0.0)
    vacation_matrix = np.full(shape=day_ts.shape, fill_value=0.0)

    # Compute hour of day based indices to use

    col_idx = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[:, Cgbdisagg.INPUT_DAY_IDX]
    col_idx = col_idx / Cgbdisagg.SEC_IN_HOUR
    col_idx = (pd_mult * (col_idx - col_idx.astype(int) + input_data[:, Cgbdisagg.INPUT_HOD_IDX])).astype(int)

    data_matrix[row_idx, col_idx] = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    vacation_matrix[row_idx, col_idx] = np.sum(vacation_output, axis=1)
    debug['vacation_matrix'] = vacation_matrix

    # Get the y_ticks (dates)

    data_matrix_df = pd.DataFrame(data_matrix)

    # Initialise x ticks column names and assign the dataframes with the column names (time stamps)

    frac_pd = Cgbdisagg.HRS_IN_DAY / data_matrix_df.shape[1]

    hour_tick = np.arange(0, Cgbdisagg.HRS_IN_DAY, frac_pd)
    frac_tick = (hour_tick - np.floor(hour_tick)) * 0.6
    day_points = (np.floor(hour_tick) + frac_tick).astype(int)

    data_matrix_df.columns = day_points

    # Initialise y ticks native

    yticks = day_ts[:, 0]
    ytick_labels = []

    for j in range(0, len(yticks)):
        dt = datetime.fromtimestamp(yticks[j])
        dv = datetime.timetuple(dt)
        month = date(int(dv[0]), int(dv[1]), int(dv[2])).strftime('%d-%b-%y')
        ytick_labels.append(month)

    return data_matrix, row_idx, col_idx, ytick_labels, day_points
