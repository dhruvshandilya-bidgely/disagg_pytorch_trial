"""
Author - Sahana M
Date - 2/3/2021
Preprocessing input data to remove HVAC consumption and other noise
"""

# Import python packages

import scipy
import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.rolling_function import rolling_function


def preprocess_data(in_data, seasonal_wh_config, debug, logger_pass):

    """
    Preprocess the input data to remove HVAC consumption and other noise
    Args:
        in_data             (np.ndarray)        : 21 column input data matrix
        seasonal_wh_config  (dict)              : Dictionary containing all needed configuration variables
        debug               (dict)              : Contains all variables required for debugging
        logger_pass         (dict)              : Contains the logger and the logging dictionary to be passed on

    Returns:
        data                (np.ndarray)        : 21 column cleaned data matrix
        debug               (dict)              : Contains all variables required for debugging

    """

    # Initialize logger

    logger_base = logger_pass.get('base_logger').getChild('preprocess_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Initialise all the variables to be used

    sampling_rate = seasonal_wh_config['user_info'].get('sampling_rate')
    window_hours = seasonal_wh_config['config'].get('cleaning_window_size')
    percentile_filter_value = seasonal_wh_config['config'].get('percentile_filter_value')

    data = deepcopy(in_data)

    # Find the number of data points required for the given window size

    window_size = window_hours * Cgbdisagg.SEC_IN_HOUR / sampling_rate
    window_size = np.floor(window_size / 2) * 2 + 1

    # Smoothing using Median filter

    filtered_data = scipy.ndimage.median_filter(data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], size=int(window_size))

    # Applying Percentile filtering

    filtered_data = scipy.ndimage.percentile_filter(filtered_data, percentile=percentile_filter_value,
                                                    size=int(window_size))

    # Removing the HVAC consumption

    filtered_data = rolling_function(filtered_data, window_size, 'min')

    # Passing a maximum filter over the min values calculated in previous step

    filtered_data = rolling_function(filtered_data, window_size, 'max')

    # Removing unwanted consumption from the input data

    data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.fmax(data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - filtered_data, 0)

    logger.info('Data cleaning for Seasonal WH input completed')

    return data, debug
