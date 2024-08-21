"""
Author - Sahana M
Date - 20/07/2021
This file is used to merge the chunk to one of the sequences formed
"""

# Import python packages
import numpy as np

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import chunk_indexes


def merge_with_seq(chunk_inst_allotment_arr, chunk_data, global_instance_seq, chunk_global_idx_arr, num_instance_seq):
    """
    This function is used to merge a chunk with a sequence
    Parameters:
        chunk_inst_allotment_arr        (list)          : Contains the chunk label and its score
        chunk_data                      (np.ndarray)    : Array containing the chunk information in the current 30 day division
        global_instance_seq             (dict)          : Dictionary containing chunks info for each sequence identified
        chunk_global_idx_arr            (list)          : The chunk index in the global chunk collection
        num_instance_seq                (int)           : Number of sequences identified till now
    Returns:
        chunk_inst_allotment_arr        (list)          : Contains the chunk label and its score
        chunk_data                      (np.ndarray)    : Array containing the chunk information in the current 30 day division
        global_instance_seq             (dict)          : Dictionary containing chunks info for each sequence identified
    """

    # get the number of probable sequences

    num_unique_seq = np.unique(chunk_inst_allotment_arr[:, 1])
    unalloted_chunk_id = list(chunk_data[:, chunk_indexes['chunk_id']] - 1)

    for seq_id in num_unique_seq:
        curr_chunk_allotment_info = chunk_inst_allotment_arr[chunk_inst_allotment_arr[:, 1] == seq_id]

        for i in range(len(curr_chunk_allotment_info)):
            cur_inst_info = curr_chunk_allotment_info[i, :]
            chunk_inst_idx = int(cur_inst_info[0])

            if chunk_inst_idx in unalloted_chunk_id:
                inst_seq_dict = global_instance_seq[seq_id]
                inst_seq_dict['chunk_inst_id_arr'].append(chunk_global_idx_arr[chunk_inst_idx])
                inst_seq_dict['amplitude_arr'].append(
                    chunk_data[chunk_inst_idx, chunk_indexes['amplitude_idx']])
                inst_seq_dict['duration_arr'].append(chunk_data[chunk_inst_idx, chunk_indexes['duration_idx']])
                inst_seq_dict['start_arr'].append(chunk_data[chunk_inst_idx, chunk_indexes['chunk_start']])
                inst_seq_dict['end_arr'].append(chunk_data[chunk_inst_idx, chunk_indexes['chunk_end']])
                inst_seq_dict['last_added_chunk'] = int(
                    chunk_data[chunk_inst_idx, chunk_indexes['window_start']])
                inst_seq_dict['add_type_arr'].append(cur_inst_info[2])
                global_instance_seq[seq_id] = inst_seq_dict

                chunk_data[chunk_inst_idx, chunk_indexes['label']] = cur_inst_info[2]
                chunk_data[chunk_inst_idx, chunk_indexes['previous_chunk_idx']] = \
                    inst_seq_dict['chunk_inst_id_arr'][-2]
                chunk_data[chunk_inst_idx, chunk_indexes['merged_seq']] = seq_id

                chunk_inst_allotment_arr[chunk_inst_allotment_arr[:, 0] == chunk_inst_idx, -1] = 0
                sort_idx = np.flip(np.argsort(chunk_inst_allotment_arr[:, -1], axis=None))
                chunk_inst_allotment_arr = chunk_inst_allotment_arr[sort_idx]
                unalloted_chunk_id.remove(chunk_inst_idx)

                break

    # Allot the unallocated sequence

    for i in range(len(unalloted_chunk_id)):
        chunk_inst_idx = int(unalloted_chunk_id[int(i)])
        inst_seq_dict = dict()
        inst_seq_dict['chunk_inst_id_arr'] = [chunk_global_idx_arr[chunk_inst_idx]]
        inst_seq_dict['amplitude_arr'] = [chunk_data[chunk_inst_idx, chunk_indexes['amplitude_idx']]]
        inst_seq_dict['duration_arr'] = [chunk_data[chunk_inst_idx, chunk_indexes['duration_idx']]]
        inst_seq_dict['start_arr'] = [chunk_data[chunk_inst_idx, chunk_indexes['chunk_start']]]
        inst_seq_dict['end_arr'] = [chunk_data[chunk_inst_idx, chunk_indexes['chunk_end']]]
        inst_seq_dict['last_added_chunk'] = int(chunk_data[chunk_inst_idx, chunk_indexes['window_start']])
        inst_seq_dict['add_type_arr'] = [0]

        chunk_data[chunk_inst_idx, chunk_indexes['label']] = 0
        chunk_data[chunk_inst_idx, chunk_indexes['previous_chunk_idx']] = -1
        chunk_data[chunk_inst_idx, chunk_indexes['merged_seq']] = num_instance_seq

        global_instance_seq[num_instance_seq] = inst_seq_dict
        num_instance_seq += 1

    return chunk_inst_allotment_arr, chunk_data, global_instance_seq
