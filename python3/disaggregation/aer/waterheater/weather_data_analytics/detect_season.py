"""
Author: Mayank Sharan
Created: 12-Jul-2020
Detect day level season tags and calculate associated information
"""

# Import python packages

import logging
import numpy as np

# Import project functions and classes

from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_seasons import mark_seasons
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import merge_seq_arr
from python3.disaggregation.aer.waterheater.weather_data_analytics.detect_season_utils import merge_chunks
from python3.disaggregation.aer.waterheater.weather_data_analytics.detect_season_utils import chunk_weather_data
from python3.disaggregation.aer.waterheater.weather_data_analytics.detect_season_utils import identify_koppen_class


def detect_season(day_wise_data_dict, meta_data, logger_pass):

    """
    Process weather data for use by the engine
    Parameters:
        day_wise_data_dict      (dict)          : Dictionary containing all day wise data matrices
        meta_data               (dict)          : Dictionary containing meta data about the user
        logger_pass             (dict)          : Dictionary containing objects needed for logging
    Returns:
        season_dict             (dict)          : Dictionary containing season detection data
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('detect_season')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Identify and split data into chunks

    day_ts = day_wise_data_dict.get('day_ts')
    num_days_data = len(day_ts)

    chunk_arr = chunk_weather_data(day_ts)

    logger.info('Number of weather data chunks | complete - %d, partial - %d', np.sum(chunk_arr[:, 3] == 0),
                np.sum(chunk_arr[:, 3] == 1))

    if np.sum(chunk_arr[:, 3] == 0) == 0:
        logger.warning('No complete chunks. Insufficient data for weather analytics |')
        return {}

    # Identify koppen class for the user using data from the latest chunk

    if meta_data['pilot_id'] == 5044:
        class_name = 'Ch'
    else:
        class_name = identify_koppen_class(day_wise_data_dict, meta_data, chunk_arr[0, :])

    logger.info('Koppen class for the user is | %s', class_name)

    # For each chunk mark season and compute associated attributes. Type 0 - fit GMM, Type 1 - use nearest GMM fit

    prev_chunk_dict = {}
    season_dict = {}

    for chunk_idx in range(chunk_arr.shape[0]):

        chunk_info = chunk_arr[chunk_idx]
        logger.info('Processing chunk | %s', ','.join(chunk_info.astype(str)))

        chunk_season_dict = mark_seasons(day_wise_data_dict, chunk_info, class_name, prev_chunk_dict, logger_pass)

        prev_chunk_dict = chunk_season_dict

        # Update overall information by merging chunks

        season_dict = merge_chunks(chunk_idx, chunk_info, season_dict, chunk_season_dict, num_days_data)

    # Merge seq array together

    final_seq_arr = merge_seq_arr(season_dict.get('seq_arr'), num_days_data)
    s_label = season_dict.get('s_label')

    for seq_idx in range(final_seq_arr.shape[0]):
        seq_info = final_seq_arr[seq_idx, :]
        s_label[int(seq_info[1]): int(seq_info[2]) + 1] = seq_info[0]

    season_dict['seq_arr'] = final_seq_arr
    season_dict['s_label'] = s_label
    season_dict['class_name'] = class_name

    return season_dict
