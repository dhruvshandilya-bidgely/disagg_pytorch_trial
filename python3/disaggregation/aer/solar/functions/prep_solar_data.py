"""
Author - Paras Tehria
Date - 12/11/19
This module converts the data to suitable format for solar detection
"""

# Import python packages

import logging
import numpy as np
from sklearn import preprocessing

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.maths_utils import forward_fill
from python3.utils.maths_utils.maths_utils import create_pivot_table


def get_preprocessed_data(input_data, logger_base, solar_config):
    """
    This function pre-processes the data and create instances for solar detection

    Parameters:
        input_data          (np.ndarray)    :       input 21 column matrix
        logger_base         (dict)          :       dict used for logging
        solar_config        (dict)          :       config file

    Return:
        cnn_detection_array (np.ndarray)    :       final array containing instances required for solar detection
    """

    # Creating a local logger from the logger_base

    slide_len = solar_config.get('prep_solar_data').get('slide_len')
    instance_size = solar_config.get('prep_solar_data').get('instance_size')

    logger_local = logger_base.get('logger').getChild('get_preprocessed_data')

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Generating Raw consumption pivot table

    y_signal_raw, _, _ = create_pivot_table(data=input_data, index=Cgbdisagg.INPUT_DAY_IDX,
                                            columns=Cgbdisagg.INPUT_HOD_IDX, values=Cgbdisagg.INPUT_CONSUMPTION_IDX)

    if len(y_signal_raw) < instance_size:
        logger.info('Not enough data, skipping solar detection module (data required, data present)|  {}, {}'.format(
            instance_size, len(y_signal_raw)))

    # Replacing na values by neighborhood days (ffill)
    y_signal_raw = forward_fill(y_signal_raw)

    # bfill
    y_signal_raw = np.flipud(forward_fill(np.flipud(y_signal_raw)))

    # capping daily high value outliers

    percentile_cap = solar_config.get('prep_solar_data').get('percentile_cap')
    percentile_array = np.tile(np.percentile(y_signal_raw, percentile_cap, axis=1).reshape(-1, 1),
                               y_signal_raw.shape[1])
    y_signal_raw = np.round(np.minimum(y_signal_raw, percentile_array), 2)

    # Capping negative values to zero helps in capturing solar signals on normalised data

    y_signal_raw[y_signal_raw < 0] = 0

    # Generating solar presence pivot table
    # Sunlight present when time is between sunrise and sunset times
    sun_presence = np.logical_and(
        input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX],
        input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= input_data[:, Cgbdisagg.INPUT_SUNSET_IDX])

    logger.debug('Created one-hot encoded sunlight presence array | ')

    # array contains one hot encoded array containing presence of sunlight
    sun_array = np.ones((len(input_data), 1), dtype=input_data.dtype)

    # Adding the one-hot encoded sun_array to act as a new column
    sun_array[:, 0] = np.where(sun_presence, sun_array[:, 0], 0)
    nan_sunrise_sunset = np.logical_or(np.isnan(input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX]),
                                       np.isnan(input_data[:, Cgbdisagg.INPUT_SUNSET_IDX]))
    sun_array[:, 0] = np.where(~nan_sunrise_sunset, sun_array[:, 0], np.nan)

    input_data = np.hstack((input_data, sun_array))
    logger.debug('sunlight presence array added to 21 column matrix | ')

    sun_index = solar_config.get('prep_solar_data').get('sun_index')
    y_signal_sun, _, _ = create_pivot_table(data=input_data, index=Cgbdisagg.INPUT_DAY_IDX,
                                            columns=Cgbdisagg.INPUT_HOD_IDX, values=sun_index)

    # Replacing na values by neighborhood days (ffill)

    y_signal_sun = forward_fill(y_signal_sun)

    # bfill

    y_signal_sun = np.flipud(forward_fill(np.flipud(y_signal_sun)))

    # Normalising the data by applying min max scaler on daily level

    y_signal_raw = preprocessing.minmax_scale(y_signal_raw.T).T

    # stacked channels

    final_data = np.dstack((y_signal_raw, y_signal_sun))

    # Initializing array containing the instances for detection

    cnn_detection_array = []

    logger.debug('Starting creation of the instances | ')

    # Making chunks of 90 days from data with 30 days sliding. Every chunk will act as an instance to cnn

    chunk_start = []
    chunk_end = []
    chunk_dates = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX])
    for i in np.arange(0, len(final_data), slide_len):
        three_months_data = final_data[i:i + instance_size]

        # Break if less than  of data left

        if three_months_data.shape[0] < instance_size:
            break

        chunk_start.append(int(chunk_dates[i]))
        chunk_end.append(int(chunk_dates[i + instance_size - 1]))

        cnn_detection_array.append(three_months_data)

    logger.debug('Total number of chunks created | {}'.format(len(cnn_detection_array)))

    # Returning preprocessed data in numpy format

    cnn_detection_array = np.array(cnn_detection_array)

    chunk_times = {'chunk_start': chunk_start, 'chunk_end': chunk_end}

    return chunk_times, cnn_detection_array
