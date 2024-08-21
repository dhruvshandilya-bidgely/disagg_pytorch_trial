
"""
Author - Nisha Agarwal
Date - 8th Oct 20
Prepare and preprocess sunrise/sunset data for lighting estimation
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.init_itemization_config import init_itemization_params


def preprare_sunrise_sunset_data(item_input_object, lighting_config, logger_pass):

    """
    prepare sunrise/sunset data for lighting estimation

    Parameters:
        item_input_object         (dict)            : Dict containing all hybrid inputs
        lighting_config             (dict)            : dict containing lighting config values
        logger_pass                 (dict)            : Contains the logger and the logging dictionary to be passed on

    Returns:
        sunrise_sunset              (np.ndarray)      : Prepared/filtered sunrise/sunset data
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('prepare_sunrise_sunset_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_lighting_start = datetime.now()

    input_data = item_input_object.get("item_input_params").get("input_data")
    output_data = item_input_object.get("item_input_params").get("output_data")
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")

    sunrise_sunset = np.zeros(output_data[0].shape)

    # prepare sunrise sunset data

    input_data[Cgbdisagg.INPUT_SUNRISE_IDX, :, 0], input_data[Cgbdisagg.INPUT_SUNSET_IDX, :, 0] = \
        filter_sunrise_sunset_data(input_data[Cgbdisagg.INPUT_SUNRISE_IDX, :, 0],
                                   input_data[Cgbdisagg.INPUT_SUNSET_IDX, :, 0],
                                   samples_per_hour, lighting_config, logger)

    logger.debug("Prepared sunrise-sunset data")

    sunrise_sunset[np.arange(len(sunrise_sunset)), (input_data[Cgbdisagg.INPUT_SUNSET_IDX, :, 0]).astype(int)] =\
        lighting_config.get('sunrise_sunset_config').get('sunset_val')
    sunrise_sunset[np.arange(len(sunrise_sunset)), (input_data[Cgbdisagg.INPUT_SUNRISE_IDX, :, 0]).astype(int)] = \
        lighting_config.get('sunrise_sunset_config').get('sunrise_val')

    t_lighting_end = datetime.now()

    logger.info("Running main sunrise-sunset preparation took | %.3f s",
                get_time_diff(t_lighting_start, t_lighting_end))

    return sunrise_sunset


def filter_sunrise_sunset_data(sunrise_data, sunset_data, samples_per_hour, lighting_config, logger):

    """
    Clean sunrise sunset data

    Parameters:
        sunrise_data                (np.ndarray)         : Original sunrise data
        sunset_data                 (np.ndarray)         : Original sunset data
        samples_per_hour            (int)                : Samples in a hour
        lighting_config             (dict)               : dict containing lighting config values
        logger                      (logger)             : logger object

    Returns:
        sunrise_data                (np.ndarray)         : Filtered sunrise data
        sunset_data                 (np.ndarray)         : Filtered sunset data
    """

    default_sunrise = lighting_config.get('sunrise_sunset_config').get('default_sunrise')
    default_sunset = lighting_config.get('sunrise_sunset_config').get('default_sunset')

    sunrise_data = sunrise_data * samples_per_hour
    sunset_data = sunset_data * samples_per_hour

    # fetch days with faulty sunrise sunset data


    # Number of neighbouring hours (to the default value) considered to be valid sunrise/sunset hours
    default_buffer_hours = 3

    faulty_sunrise_data = np.logical_or(sunrise_data > (default_sunrise+default_buffer_hours) * samples_per_hour,
                                        sunrise_data < (default_sunrise-default_buffer_hours) * samples_per_hour)
    faulty_sunset_data = np.logical_or(sunset_data > (default_sunset+default_buffer_hours+1) * samples_per_hour,
                                       sunset_data < (default_sunset-default_buffer_hours) * samples_per_hour)

    sunset_data_labels = \
        find_seq(faulty_sunset_data, np.zeros(len(faulty_sunset_data)), np.zeros(len(faulty_sunset_data)))
    sunrise_data_labels = \
        find_seq(faulty_sunrise_data, np.zeros(len(faulty_sunrise_data)), np.zeros(len(faulty_sunrise_data)))

    logger.info("Total number of days | %d", len(sunrise_data))
    logger.info("Number of days with faulty sunrise data | %d", np.sum(faulty_sunrise_data))
    logger.info("Number of days with faulty sunset data | %d", np.sum(faulty_sunset_data))

    # replace faulty sunrise data with neighbour values or default values

    if np.all(faulty_sunrise_data):
        sunrise_data[:] = default_sunrise * samples_per_hour

        logger.info("All sunrise data are faulty")

    elif np.any(faulty_sunrise_data):

        sunrise_data = fill_sunrise_sunset_data(sunrise_data, sunrise_data_labels,
                                                lighting_config.get('sunrise_sunset_config').get('faulty_days_limit'),
                                                default_sunrise, samples_per_hour, logger)

    # replace faulty sunset data with neighbour values or default values

    if np.all(faulty_sunset_data):
        sunset_data[:] = default_sunset * samples_per_hour

        logger.info("All sunset data are faulty")

    elif np.any(faulty_sunset_data):

        sunset_data = fill_sunrise_sunset_data(sunset_data, sunset_data_labels,
                                               lighting_config.get('sunrise_sunset_config').get('faulty_days_limit'),
                                               default_sunset, samples_per_hour, logger)

    logger.debug("Cleaning of sunrise-sunset data done")

    return sunrise_data, sunset_data


def fill_sunrise_sunset_data(data, data_labels, limit, default_val, samples_per_hour, logger):

    """
    Fill empty sunrise sunset data

    Parameters:
        data                    (np.ndarray)        : sunrise/sunrise data
        data_labels             (np.ndarray)        : sunrise/sunset seq
        limit                   (int)               : faulty days length threshold
        default_val             (float)             : sunrise/sunset default tou
        samples_per_hour        (int)               : samples in an hour

    Returns:
        data                    (np.ndarray)        : filled sunrise/sunrise data
    """

    seq_config = init_itemization_params().get("seq_config")

    for i in range(len(data_labels)):

        logger.debug("Preparing sunrise sunset data for days chunk %d", i)

        # If missing chunk length is less than limit, fill using neighbouring points

        if data_labels[i][seq_config.get("label")] and data_labels[i][seq_config.get("length")] < limit:

            value = (data[int(data_labels[(i - 1) % len(data_labels)][2])] +
                     data[int(data_labels[(i + 1) % len(data_labels)][1])]) / 2

            index_array = get_index_array(data_labels[i][seq_config.get("start")], data_labels[i][seq_config.get("end")], len(data))
            data[index_array] = value

        # If missing chunk length is greater than limit, fill using default values

        elif data_labels[i][seq_config.get("label")] and data_labels[i][seq_config.get("length")] >= limit:

            index_array = get_index_array(data_labels[i][seq_config.get("start")], data_labels[i][seq_config.get("end")], len(data))
            data[index_array] = default_val * samples_per_hour

    return data
