"""
Author - Sahana M
Date - 20/07/2021
This function is used to identify the category of the chunk where the categories are Continued, ReScheduled, Independent
"""

# Import python packages
import logging
import numpy as np

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import chunk_indexes
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.identify_label import identify_label
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.merge_with_seq import merge_with_seq


def identify_seq_type(chunk_data, global_instance_seq, global_chunks_info, wh_config, logger_base):
    """
    This function is allot each chunk to its respective sequence category
    Parameters:
        chunk_data              (np.ndarray)        : Array containing information about the chunk/s
        global_instance_seq     (dict)              : Dictionary containing information about the different sequences identified
        global_chunks_info      (np.ndarray)        : Contains chunk data of all the chunks identified till now
        wh_config               (dict)              : WH configurations dictionary
        logger_base             (Logger)            :       Logger

    Returns:
        chunk_data              (np.ndarray)        : Array containing information about the chunk/s
        global_instance_seq     (dict)              : Dictionary containing information about the different sequences identified
        global_chunks_info      (np.ndarray)        : Contains chunk data of all the chunks identified till now
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('identify_seq_type')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract all the necessary data

    instance_seq_ids = list(global_instance_seq.keys())
    num_instance_seq = len(instance_seq_ids)
    num_inst_curr_chunk = chunk_data.shape[0]

    # Append the chunk data info to the global chunks info

    curr_global_chunk_count = len(global_chunks_info)
    chunk_global_idx_arr = []
    for chunk_inst_idx in range(num_inst_curr_chunk):
        global_chunks_info.append(chunk_data[chunk_inst_idx])
        chunk_global_idx_arr.append(curr_global_chunk_count)
        curr_global_chunk_count += 1

    # If there are is no chunk data available then pass
    if num_inst_curr_chunk == 0:
        pass

    # If this is the first chunk data then assign the chunk of label 0

    elif num_instance_seq == 0:
        for chunk_inst_idx in range(num_inst_curr_chunk):
            inst_seq_dict = dict()
            inst_seq_dict['chunk_inst_id_arr'] = [chunk_global_idx_arr[chunk_inst_idx]]
            inst_seq_dict['amplitude_arr'] = [chunk_data[chunk_inst_idx, chunk_indexes['amplitude_idx']]]
            inst_seq_dict['duration_arr'] = [chunk_data[chunk_inst_idx, chunk_indexes['duration_idx']]]
            inst_seq_dict['start_arr'] = [chunk_data[chunk_inst_idx, chunk_indexes['chunk_start']]]
            inst_seq_dict['end_arr'] = [chunk_data[chunk_inst_idx, chunk_indexes['chunk_end']]]
            inst_seq_dict['last_added_chunk'] = int(chunk_data[chunk_inst_idx, chunk_indexes['window_start']])
            inst_seq_dict['add_type_arr'] = [0]
            global_instance_seq[num_instance_seq] = inst_seq_dict
            chunk_data[chunk_inst_idx, chunk_indexes['label']] = 0
            chunk_data[chunk_inst_idx, chunk_indexes['previous_chunk_idx']] = -1
            chunk_data[chunk_inst_idx, chunk_indexes['merged_seq']] = num_instance_seq
            num_instance_seq += 1

    # If this is not the first chunk then assign the chunks label

    else:

        chunk_inst_allotment_arr = identify_label(num_inst_curr_chunk, chunk_data, instance_seq_ids, global_instance_seq,
                                                  global_chunks_info, wh_config)

        chunk_inst_allotment_arr = np.array(chunk_inst_allotment_arr)
        if len(chunk_inst_allotment_arr.shape) == 1:
            chunk_inst_allotment_arr = np.array([chunk_inst_allotment_arr])

        if not chunk_inst_allotment_arr.size:
            chunk_inst_allotment_arr = np.array([[-1, -1, -1, -1]])

        sort_idx = np.flip(np.argsort(chunk_inst_allotment_arr[:, -1], axis=None))
        chunk_inst_allotment_arr = chunk_inst_allotment_arr[sort_idx]

        chunk_inst_allotment_arr, chunk_data, global_instance_seq = merge_with_seq(chunk_inst_allotment_arr, chunk_data,
                                                                                   global_instance_seq, chunk_global_idx_arr, num_instance_seq)

    logger.info('Identifying sequence type completed | ')

    return chunk_data, global_instance_seq, global_chunks_info
