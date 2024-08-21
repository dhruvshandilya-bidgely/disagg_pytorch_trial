"""
Author - Sahana M
Date - 20/07/2021
Get the Band data probably containing Timed WH instance and remove erroneous blocks
"""

# Import python packages
import logging
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.twh_maths_utils import tod_filler
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.low_score_gaps import low_score_gaps
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.get_chunk_scoring import chunk_scoring
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.identify_seq_type import identify_seq_type
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.get_instances_chunk import get_instances_chunks


def get_gaussian_band_data(input_data, concentration_arr, debug, wh_config, logger_base):
    """
    This function is used to find probable timed wh instances and remove erroneous instances by running on 30 days of
    data with a sliding window of 15 days
    Parameters:
        input_data             (np.ndarray)        : 2D input matrix containing cleaned data
        concentration_arr      (np.array)          : Boolean array containing the best times of interest
        debug                  (dict)              : Contains algorithm output
        wh_config              (dict)              : WH configurations dictionary
        logger_base            (logger)             : Logger passed

    Returns:
        overall_chunk_data     (np.ndarray)        : Array containing the selected timed wh instances info
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('get_gaussian_band_data')
    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract the necessary data

    days = wh_config.get('days')
    sliding_window = wh_config.get('sliding_window')

    # Initialise necessary data

    aoi_data = deepcopy(input_data)

    # Initialise array which contains the starting indexes of all the instances throughout the year

    start_idx_arr = np.arange(0, aoi_data.shape[0]-sliding_window, sliding_window).astype(int)

    # Initialise a Consistency & Amplitude array

    overall_consistency = np.full(shape=(start_idx_arr.shape[0], aoi_data.shape[1]), fill_value=0)
    overall_amplitude = np.full(shape=(start_idx_arr.shape[0], aoi_data.shape[1]), fill_value=0)

    global_instance_seq = {}
    global_chunks_info = []
    overall_chunk_data = []

    logger.info('Identifying Gaussian band data for {} chunks | '.format(len(start_idx_arr)))

    # For each Instance of 30 days get the probable chunks and a scoring for those chunks

    for start_idx in start_idx_arr:

        # Extract the data of interest (batch_data)

        overall_idx = np.where(start_idx_arr == start_idx)[0]
        batch_data = aoi_data[start_idx:min(start_idx+days, aoi_data.shape[0]), :]
        batch_data_1d = batch_data.flatten()

        # Identify the time of each batch_data point

        col_data = np.full_like(batch_data, fill_value=np.nan)
        col_data = tod_filler(col_data)
        col_data_1d = col_data.flatten()

        # Get the scaling factor required for bucketisation

        scaling_factor = Cgbdisagg.HRS_IN_DAY/batch_data.shape[1]

        # Get the 2d histogram
        # Note: The 2D matrix has x axis as time and y axis as amplitude range, the number on each data point represents
        #        the number of days (out of 30 days) which contains the y amplitude at x time.

        energy_blocks_2d = np.flipud(np.histogram2d(col_data_1d, batch_data_1d,
                                                    bins=(batch_data.shape[1], int(np.max(batch_data) / (500 * scaling_factor)) + 1))[0].T)

        # get the amplitude ranges

        if np.max(batch_data) > 0:
            amplitude_ranges = np.array_split(np.arange(0, np.max(batch_data)),
                                              int(np.max(batch_data) / (500 * scaling_factor)) + 1)

            amplitude_range_arr = np.full(shape=(len(amplitude_ranges), 2), fill_value=0)
            for i in range(len(amplitude_ranges)):
                amplitude_range_arr[i][0] = amplitude_ranges[i][0]
                amplitude_range_arr[i][1] = amplitude_ranges[i][-1]

            amplitude_range_arr = np.flipud(amplitude_range_arr)

            # Ignore the last bin and concentrate on the upper bins only

            energy_blocks_2d_toi = energy_blocks_2d[:-1, :]
            amplitude_range_arr = amplitude_range_arr[:-1, :]

            # Get the best chunks from the energy_blocks_2d_toi

            chunk_data, max_consistencies_1d, percentile_amplitudes = \
                get_instances_chunks(start_idx, overall_idx[0], energy_blocks_2d_toi, amplitude_range_arr, debug, wh_config)

            # Identify the chunk type and allocate it to a sequence

            chunk_data, global_instance_seq, global_chunks_info = \
                identify_seq_type(chunk_data, global_instance_seq, global_chunks_info, wh_config, logger_pass)

            # Store the refined amplitudes and consistencies in the overall arrays

            overall_amplitude[overall_idx] = percentile_amplitudes
            overall_consistency[overall_idx] = max_consistencies_1d

            # Store the chunk as a part of the overall_chunk_data array

            if len(chunk_data):
                if not len(overall_chunk_data):
                    overall_chunk_data = chunk_data
                else:
                    overall_chunk_data = np.vstack((overall_chunk_data, chunk_data))

                # Store the overall info in a dictionary
                overall_info = {
                    'overall_amplitude': overall_amplitude,
                    'overall_consistency': overall_consistency,
                    'overall_chunk_data': overall_chunk_data
                }

                # Calculate the scores for these chunks

                overall_chunk_data = chunk_scoring(overall_idx[0], overall_info, chunk_data, aoi_data, wh_config)

    logger.info('The number of independent sequences identified are | {} '.format(len(global_instance_seq)))

    # Remove the low scoring boxes

    scored_data_matrix, overall_chunk_data = low_score_gaps(overall_chunk_data, concentration_arr, start_idx_arr, wh_config,
                                                            logger_pass)

    debug['scored_data_matrix'] = scored_data_matrix
    debug['overall_chunk_data'] = overall_chunk_data

    logger.info('Total number of chunks identified are | {} '.format(len(overall_chunk_data)))

    return scored_data_matrix, overall_chunk_data
