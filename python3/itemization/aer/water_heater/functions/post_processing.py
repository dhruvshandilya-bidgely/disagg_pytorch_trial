"""
Author - Sahana M
Date - 4/3/2021
Post processing for noise removal on Final data
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.itemization.aer.water_heater.functions.get_consistency import get_consistency


def post_processing(final_data, debug, seasonal_config, logger_pass):
    """
    Performs post processing for noise removal on Final data
    Args:
        final_data              (np.ndarray)        : Contains (day x time) shape final consumption matrix
        debug                   (dict)              : Contains all variables required for debugging
        seasonal_config         (dict)              : Dictionary containing all needed configuration variables
        logger_pass             (Logger)            : Passed logger object
    Returns:
        wh_data                 (np.ndarray)        : Contains (day x time) shape final cleaned consumption matrix
        least_value             (float)             : Detected low amplitude
        highest_value           (float)             : Detected high amplitude
    """

    # Initialize logger

    logger_base = logger_pass.get('base_logger').getChild('post_processing')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Initialise all variables required

    wh_data = deepcopy(final_data)
    factor = debug['factor']
    time_zones_div = seasonal_config['time_zones_hour_div']
    time_band_prob = seasonal_config['tb_prob']
    max_runs = seasonal_config['config']['max_runs']
    buffer_amplitude = seasonal_config['config']['buffer_amplitude']
    lower_amp_cap = seasonal_config['config']['lower_amp_cap']
    upper_amp_cap = seasonal_config['config']['upper_amp_cap']

    # ------------------------------------------ STAGE 1: RESTRICT ENERGY RANGE ---------------------------------------

    # Compress the energy range based on percentile

    temp = wh_data[wh_data > 0]
    least_value = np.percentile(temp, lower_amp_cap) - (buffer_amplitude / factor)
    highest_value = np.percentile(temp, upper_amp_cap) + (buffer_amplitude / factor)

    logger.info('Lower and Upper bound amplitudes identified for Seasonal WH are | ({}, {}]'.format(least_value, highest_value))

    # Cap all the high values, makes all the low values to 0

    wh_data[wh_data < least_value] = 0
    wh_data[wh_data >= highest_value] = highest_value

    # ------------------------------------------ STAGE 2: REMOVE SMALL DURATION BOXES ---------------------------------

    if factor == 4 and np.sum(wh_data) > 0:
        false_arr = [False] * len(wh_data)
        wh_boxes = (wh_data > 0)
        left_shifted_arr = np.c_[false_arr, wh_data[:, :-1] > 0]

        wh_boxes_left_shifted = wh_boxes & left_shifted_arr

        right_shifted_arr = np.c_[wh_boxes_left_shifted[:, 1:], false_arr]
        wh_boxes_right_shifted = right_shifted_arr | wh_boxes_left_shifted

        wh_data *= wh_boxes_right_shifted.astype(int)

    logger.info('Removed boxes running for < 30 minutes | ')

    # ------------------------------------------ STAGE 3: MULTIPLE RUNS -----------------------------------------------

    start_time = 0
    end_time = Cgbdisagg.HRS_IN_DAY * factor

    # Get consistency of seasonal wh boxes

    _, consistency_array, _ = get_consistency(wh_data, start_time, end_time, debug, seasonal_config)

    consistency_array = consistency_array[:-1]

    # Initialise an empty time band probability array

    tb_prob_arr = [0] * (wh_data.shape[1])

    # Get time band probability for each time division

    for i in range(len(time_zones_div)):
        left = time_zones_div[i][0] * factor
        right = time_zones_div[i][1] * factor
        tb_prob_arr[left: right + 1] = [time_band_prob[i]] * ((right + 1) - left)

    tb_prob_arr = np.asarray(tb_prob_arr)

    # Find probability as a component of consistency and time band probability

    probability_arr = consistency_array * tb_prob_arr

    if np.sum(probability_arr) > 0:

        # Get the days with detected water heater boxes

        wh_day_pulses = (wh_data > 0)

        zero_array = [0] * len(wh_day_pulses)
        wh_day_pulses = np.c_[zero_array, wh_day_pulses, zero_array]

        # Identify the day, the start & end time/index of the water heater box

        box_energy_idx_diff = np.diff(wh_day_pulses)
        box_start_idx_row = np.where(box_energy_idx_diff[:] > 0)[0]
        box_start_idx_col = np.where(box_energy_idx_diff[:] > 0)[1]
        box_end_idx_col = np.where(box_energy_idx_diff[:] < 0)[1]

        cleaned_data = np.full(shape=wh_data.shape, fill_value=0.0)
        final_matrix = np.full(shape=(0, 4), fill_value=0.0)

        matrix = np.c_[box_start_idx_row, box_start_idx_col, box_end_idx_col, probability_arr[box_end_idx_col]]

        # For each water heater box day perform the following

        for i in np.unique(matrix[:, 0]):

            # For days with > 3 runs select the top 3 boxes with the highest probability value

            if len(np.where(matrix[:, 0] == i)[0]) > max_runs:
                temp = matrix[np.where(matrix[:, 0] == i), :][0]
                temp = temp[temp[:, 3].argsort()[::-1]][:max_runs, :]
                final_matrix = np.r_[final_matrix, temp]

            # For days with <= 3 runs append all the runs to the final matrix array

            else:
                temp = matrix[np.where(matrix[:, 0] == i), :][0]
                final_matrix = np.r_[final_matrix, temp]

        # Append the final filtered data into the cleaned_data matrix

        for i in range(len(final_matrix)):
            start_band = int(start_time + final_matrix[i, 1])
            end_band = int(start_time + final_matrix[i, 2])
            cleaned_data[int(final_matrix[i, 0]), start_band:end_band] = wh_data[int(final_matrix[i, 0]), start_band:end_band]

    return wh_data, least_value, highest_value
