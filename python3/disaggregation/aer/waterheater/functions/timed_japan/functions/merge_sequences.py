"""
Author - Sahana M
Date - 20/07/2021
This function is used to merge sequences
"""

# Import python packages
import logging
import numpy as np

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import chunk_indexes


def get_seq_info(curr_seq, cleaned_data_matrix, curr_seq_time_bool, curr_amp, days):
    """
    get seq info function is used to get the amplitude and time coverage information about the sequence
    Parameters:
        curr_seq                (np.ndarray)    : chunk info about the current sequence
        cleaned_data_matrix     (np.ndarray)    : 2D matrix where containing the box fit data
        curr_seq_time_bool      (np.ndarray)    : Boolean array containing the time coverage of the current sequence
        curr_amp                (list)          : Empty list to be filled with current sequence amplitudes
        days                    (int)           : Number of days considered for a chunk

    Returns:
        curr_seq_time_bool      (np.ndarray)    : Boolean array containing the time coverage of the current sequence
        curr_amp                (list)          : contains all the amplitude boxes in the current sequence
    """

    for i in range(len(curr_seq)):
        s_row = int(curr_seq[i, chunk_indexes['window_start']])
        e_row = int(min((s_row + days), cleaned_data_matrix.shape[0]))
        s_col = int(curr_seq[i, chunk_indexes['chunk_start']])
        e_col = int(curr_seq[i, chunk_indexes['chunk_end']])
        temp = cleaned_data_matrix[s_row:e_row, s_col:e_col].flatten()
        curr_seq_time_bool[s_col:e_col] = True
        curr_amp.extend(temp)

    return curr_seq_time_bool, curr_amp


def merge_sequences(overall_chunk_data, cleaned_data_matrix, wh_config, logger_base):
    """
    Merge sequences function is used to merge similar the chunk instances/sequences
    Parameters:
        overall_chunk_data      (np.ndarray)        : Contains info about all the chunks identified
        cleaned_data_matrix     (np.ndarray)        : 2D matrix where containing the box fit data
        wh_config               (dict)              : WH configurations dictionary
        logger_base             (logger)            : Logger passed

    Returns:
        overall_chunk_data      (np.ndarray)        : Contains info about all the chunks identified
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('merge_sequences')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract the necessary data

    cols = wh_config.get('cols')
    days = wh_config.get('days')
    min_bar = wh_config.get('merge_seq_min_bar')
    max_bar = wh_config.get('merge_seq_max_bar')

    num_seq = np.unique(overall_chunk_data[:, chunk_indexes['merged_seq']])

    # For each sequence compare it with other sequence & merge if possible

    for curr_seq_id in num_seq:
        for other_seq_id in num_seq:
            if other_seq_id != curr_seq_id:
                curr_seq = overall_chunk_data[overall_chunk_data[:, chunk_indexes['merged_seq']] == curr_seq_id]
                other_seq = overall_chunk_data[overall_chunk_data[:, chunk_indexes['merged_seq']] == other_seq_id]
                curr_seq_time_bool = np.full(shape=cols, fill_value=False)
                other_seq_time_bool = np.full(shape=cols, fill_value=False)
                curr_amp = []
                other_amp = []

                # Get info about the current & other sequence

                curr_seq_time_bool, curr_amp = get_seq_info(curr_seq, cleaned_data_matrix, curr_seq_time_bool, curr_amp,
                                                            days)
                other_seq_time_bool, other_amp = get_seq_info(other_seq, cleaned_data_matrix, other_seq_time_bool,
                                                              other_amp, days)

                curr_amp = np.array(curr_amp)
                other_amp = np.array(other_amp)

                # get median amplitudes
                curr_amp_median = np.median(curr_amp[curr_amp > 0])
                other_amp_median = np.median(other_amp[other_amp > 0])

                # Get info about time overlap

                worthy_merge = (other_amp_median > min_bar * curr_amp_median) \
                               & (other_amp_median < max_bar * curr_amp_median) \
                               & (np.sum(other_seq_time_bool & curr_seq_time_bool) > 0) \
                               & (np.sum(np.isin(other_seq[:, chunk_indexes['window_start']],
                                                 curr_seq[:, chunk_indexes['window_start']])) == 0)

                # If all the above conditions meet then merge them

                if worthy_merge:
                    overall_chunk_data[overall_chunk_data[:, chunk_indexes['merged_seq']] == curr_seq_id, chunk_indexes[
                        'merged_seq']] = other_seq_id
                    break

    unique_sequences = len(np.unique(overall_chunk_data[:, chunk_indexes['merged_seq']]))
    logger.info('Final number of independent sequences identified are | {}'.format(unique_sequences))

    return overall_chunk_data
