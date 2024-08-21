
"""
Author - Nisha Agarwal
Date - 8th Oct 20
Prepare activity curve of a user using input data
"""

# Import python packages

import copy
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff
from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import rolling_func_along_row

from python3.itemization.init_itemization_config import init_itemization_params

from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_activity_profile_config import init_activity_profile_config


def prepare_activity_curve(input_data, vacation, item_input_object, item_output_object, logger):

    """
    Prepare living load activity profile of the user

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    t_activity_profile_start = datetime.now()

    samples_per_hour = int(len(input_data[0])/Cgbdisagg.HRS_IN_DAY)

    # Fetch config dictionaries

    seq_config = init_itemization_params().get('seq_config')
    config = init_activity_profile_config(samples_per_hour).get('prepare_activity_curve_config')

    chunk_length = config.get('chunk_length')

    # Calculate values to normalize input data

    min_val = np.min(input_data)
    max_val = np.max(input_data)

    # Initialize values

    calculate_perc = True
    activity_curve = np.zeros(input_data.shape[1])

    # Prepare activity curve for each chunk of data

    input_data = np.nan_to_num(input_data)

    chunks_data, valid_chunks = \
        get_chunks_of_input_data(input_data, vacation, chunk_length, max_val, min_val, config.get("vacation_limit1"))

    # get non zero chunk data
    chunks_data = chunks_data[~np.all(chunks_data == 0, axis=1)]

    # If very few valid chunks are available , decrease the threshold for minimum non vacation days requirement

    if valid_chunks < config.get('min_valid_chunk'):

        logger.info("Few valid chunks present, modifying the vacation days requirement")

        chunks_data, valid_chunks = \
            get_chunks_of_input_data(input_data, vacation, chunk_length, max_val, min_val, config.get("vacation_limit2"))

        # If no valid chunks are available, consider biggest non vacation days chunk

        if np.all(chunks_data == 0):

            logger.info("Vacation home")

            calculate_perc = False

            # vacation home

            seq = find_seq(np.logical_not(vacation[:, 0]), np.zeros(len(vacation[:, 0])), np.zeros(len(vacation[:, 0])))

            if np.all(np.logical_not(vacation[:, 0]) == 0):
                activity_curve = np.zeros(input_data.shape[1])

            else:
                seq = seq[seq[:, seq_config.get('label')] == 1]
                index = int(np.argmax(seq[:, seq_config.get('length')]))
                activity_curve = np.nanmean(input_data[int(seq[index, seq_config.get('start')]):
                                                       int(seq[index, seq_config.get('end')])+1, :], axis=0)

        chunks_data = chunks_data[~np.all(chunks_data == 0, axis=1)]

    logger.debug("Calculated profile of individual chunks")
    logger.info("Number of valid chunks | %s", valid_chunks)

    non_clean_day_fraction = 1 - item_input_object.get("clean_day_score_object").get("clean_day_fraction")

    # calculate percentile to be taken, to calculate final activity curve
    # to represent non HVAC days living load consumption pattern of the user

    perc_value = get_perc_value(non_clean_day_fraction, config)

    logger.info("Clean days fraction | %s", item_input_object.get("clean_day_score_object").get("clean_day_fraction"))
    logger.info("Percentile chosen | %s", perc_value)

    if calculate_perc and len(chunks_data):
        activity_curve = np.percentile(chunks_data, perc_value, axis=0)

    t_activity_profile_end = datetime.now()

    logger.info("Preparing of activity curve took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return item_input_object, item_output_object, activity_curve, perc_value, chunks_data


def get_perc_value(non_clean_day_fraction, config):

    """
    calculate percentile to be taken using the value of non clean fraction

    Parameters:
        non_clean_day_fraction       (float)      : fraction of number of non clean days in a year
        config                       (dict)       : clean day scoring config

    Returns:
        perc                         (float)      : Percentile to be taken to calculate activity curve
    """

    # If all the days of the user are unclean

    if non_clean_day_fraction >= 1 or np.isnan(non_clean_day_fraction):
        return config.get("non_clean_user_perc")

    non_clean_days_fraction = non_clean_day_fraction * 100

    # Calculate percentile value using non clean day fraction

    perc_array = config.get('perc_array')

    perc = perc_array[int(non_clean_days_fraction)]

    perc = np.round(perc, 2)

    return perc


def get_chunks_of_input_data(input_data, vacation, chunk_length, max_val, min_val, vacation_limit):

    """
    Identify chunks of data from the given input data

    Parameters:
        input_data                  (np.ndarray)    : 2d input day data
        vacation                    (np.ndarray)    : vacation day output
        chunk_length                (int)           : window size for calculating a chunk
        max_val                     (int)           : max value of input data used for min-max norm
        min_val                     (int)           : min value of input data used for min-max norm
        vacation_limit              (float)         : max fraction of vacation for a valid chunk

    Returns:
        chunks_data                 (np.ndarray)    : prepared chunks input data
        valid_chunks                (int)           : number of valid chunks

    """

    chunks_data = np.zeros((int(len(input_data) / chunk_length), len(input_data[0])))

    if len(input_data) > chunk_length:
        rolling_vacation = rolling_func_along_row(vacation, (chunk_length - 1) / 2, 0)
        rolling_input_data = rolling_func_along_row(input_data, (chunk_length - 1) / 2, 0)
    else:
        rolling_vacation = copy.deepcopy(vacation)
        rolling_input_data = copy.deepcopy(input_data)

    # Take chunk of days with major non vacation days

    valid_chunk_index = np.logical_not(rolling_vacation[:, 0] > vacation_limit)
    chunk_position = np.arange(len(vacation)) + int(chunk_length / 2)
    chunk_position = (chunk_position % chunk_length) == 0
    valid_chunk_index = np.logical_and(valid_chunk_index, chunk_position)
    valid_chunks = np.sum(valid_chunk_index)
    rolling_input_data = rolling_input_data / chunk_length

    if valid_chunks > 0:
        chunks_data = np.round((rolling_input_data - min_val) / (max_val - min_val), 2)
        chunks_data = chunks_data[valid_chunk_index][:-1]

    return chunks_data, valid_chunks
