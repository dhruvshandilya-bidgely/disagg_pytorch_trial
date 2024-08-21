"""
Author - Sahana M
Date - 20/07/2021
This function is used to remove chunks with a lower score
"""

# Import python packages
import logging
import numpy as np

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import chunk_indexes
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.twh_maths_utils import get_start_end_idx


def low_score_gaps(overall_chunk_data, concentration_arr, start_idx_arr, wh_config, logger_base):
    """
    Low score gaps function is used to remove the chunks with a lower score
    Parameters:
        overall_chunk_data     (np.ndarray)    : Contains info about all the chunks identified
        concentration_arr      (np.array)      : Boolean array containing the best times of interest
        start_idx_arr          (np.ndarray)    : Array which contains the starting indexes of all the instances throughout the year
        wh_config              (dict)          : WH configurations dictionary
        logger_base            (logger)             : Logger passed

    Returns:
        scored_data_matrix      (np.ndarray)    : 2D matrix containing the chunks with their score for each 30 day division
        overall_chunk_data      (np.ndarray)    : Contains info about all the chunks identified:
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('low_score_gaps')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Initialise the necessary data

    cols = wh_config.get('cols')
    gap_penalty = wh_config.get('chunk_gap_penalty')
    min_chunk_score = wh_config.get('min_chunk_score')
    scored_data_matrix = np.full(shape=(start_idx_arr.shape[0], cols), fill_value=0.0)

    if len(overall_chunk_data):

        # Normalise the score between 0 & 1
        overall_chunk_data[:, chunk_indexes['chunk_score']] = overall_chunk_data[:, chunk_indexes['chunk_score']] / np.max(
            overall_chunk_data[:, chunk_indexes['chunk_score']])

        # Remove the chunks with score <= 0.35
        overall_chunk_data = overall_chunk_data[overall_chunk_data[:, chunk_indexes['chunk_score']] > min_chunk_score]

        # Copy all the chunk scores into the scored data matrix
        for i in range(overall_chunk_data.shape[0]):
            start = int(overall_chunk_data[i, chunk_indexes['chunk_start']])
            end = int(overall_chunk_data[i, chunk_indexes['chunk_end']])
            window = int(overall_chunk_data[i, chunk_indexes['overall_index']])
            scored_data_matrix[window, start:end] = overall_chunk_data[i, chunk_indexes['chunk_score']]

        scored_data_matrix /= np.max(scored_data_matrix)

        block_start_idx, block_end_idx = get_start_end_idx(concentration_arr)

        for i in range(len(block_start_idx)):

            # Identify the missing chunks (gaps) in the 30 day division throughout the data

            gaps_arr = np.sum(scored_data_matrix[:, block_start_idx[i]: block_end_idx[i]], axis=1)
            gap_days = gaps_arr == 0
            gap_start_idx, _ = get_start_end_idx(gap_days)

            # Add penalty to the missing chunks

            if len(gap_start_idx):

                scored_data_matrix[:, block_start_idx[i]: block_end_idx[i]] = \
                    scored_data_matrix[:, block_start_idx[i]: block_end_idx[i]] - gap_penalty * len(gap_start_idx)

                scored_data_matrix[scored_data_matrix < 0] = 0

                logger.info('Penalty inflicted due to gaps found in the scored data matrix | ')

    return scored_data_matrix, overall_chunk_data
