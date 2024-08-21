"""
Author - Sahana M
Date - 20/07/2021
This file is used to allocat the chunk to one of the category of chunks - Continued, Rescheduled, Independant
"""

# Import python packages
import numpy as np

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import chunk_indexes, seq_type


def identify_label(num_inst_curr_chunk, chunk_data, instance_seq_ids, global_instance_seq, global_chunks_info, wh_config):
    """
    Identify label function is used to identify the continuity status of the current chunk Independent, Continued or Rescheduled
    Parameters:
        num_inst_curr_chunk       (int)           : Number of chunks in the current 30 day division
        chunk_data                (np.ndarray)    : Array containing the chunk information in the current 30 day division
        instance_seq_ids          (list)          : IDs of the sequences identified
        global_instance_seq       (dict)          : Dictionary containing chunks info for each sequence identified
        global_chunks_info        (list)          : list containing the chunks identified at every 30 day division
        wh_config                 (dict)          : WH configurations dictionary

    Returns:
        chunk_inst_allotment_arr  (list)          : Contains the chunk label and its score
    """

    # Initialise the required variables
    cols = wh_config.get('cols')
    factor = wh_config.get('factor')
    td_weight = wh_config.get('seq_td_weight')
    dur_weight = wh_config.get('seq_dur_weight')
    min_weight = wh_config.get('seq_min_weight')
    max_weight = wh_config.get('seq_max_weight')
    amp_weight = wh_config.get('seq_amp_weight')
    td_score_thr = wh_config.get('seq_td_score_thr')
    amp_score_thr = wh_config.get('seq_amp_score_thr')
    dur_score_thr = wh_config.get('seq_dur_score_thr')
    seq_overlap_thr = wh_config.get('seq_overlap_thr')
    max_hours = wh_config.get('seq_centered_max_time')
    seq_centered_amp = wh_config.get('seq_centered_amp')
    overall_score_thr = wh_config.get('seq_overall_score_thr')
    scaling_factor = Cgbdisagg.HRS_IN_DAY/cols

    # For each chunk within a 30 day division assign it to a Sequence type

    chunk_inst_allotment_arr = []

    for chunk_inst_idx in range(num_inst_curr_chunk):
        curr_chunk_info = chunk_data[chunk_inst_idx]

        # For each chunk check with each sequence and associate it with the most similar sequence

        for all_inst_idx in instance_seq_ids:

            # Get info about the last added chunk from the sequence in consideration
            global_inst_info = global_instance_seq[all_inst_idx]
            last_added_idx = global_inst_info['chunk_inst_id_arr'][-1]
            last_chunk_info = global_chunks_info[last_added_idx]

            # Calculate the amplitude difference & score between the current and the last added chunk

            amp_diff = abs(
                last_chunk_info[chunk_indexes['amplitude_idx']] - curr_chunk_info[chunk_indexes['amplitude_idx']])
            amp_score = np.exp(-amp_diff / seq_centered_amp)

            # Calculate the duration difference & score between the current and the last added chunk

            dur_diff = abs(
                last_chunk_info[chunk_indexes['duration_idx']] - curr_chunk_info[chunk_indexes['duration_idx']])
            dur_score = np.exp(-dur_diff / 2)

            # Extract time based info of the current and the last added chunk

            previous_start_time = int(last_chunk_info[chunk_indexes['chunk_start']])
            previous_end_time = int(last_chunk_info[chunk_indexes['chunk_end']])
            curr_start_time = int(curr_chunk_info[chunk_indexes['chunk_start']])
            curr_end_time = int(curr_chunk_info[chunk_indexes['chunk_end']])
            previous_duration = previous_end_time - previous_start_time
            current_duration = curr_end_time - curr_start_time

            # Calculate the time deviation score

            max_time_diff = min(abs(curr_start_time - previous_start_time),
                                abs(curr_end_time - previous_end_time))
            time_deviation_score = np.exp(-max_time_diff / (max_hours * factor))

            # Get the time overlap score

            if previous_duration >= current_duration:
                previous_time_bool = np.full(cols, fill_value=False)
                previous_time_bool[previous_start_time: previous_end_time] = True
                overlap_count = np.sum(previous_time_bool[curr_start_time: curr_end_time])
                overlap = np.round(overlap_count / (curr_end_time - curr_start_time), 2)
                test_overlap = overlap
            else:
                reverse_time_bool = np.full(cols, fill_value=False)
                reverse_time_bool[curr_start_time: curr_end_time] = True
                overlap_count = np.sum(reverse_time_bool[previous_start_time: previous_end_time])
                reverse_overlap = np.round(
                    overlap_count / (previous_end_time - previous_start_time), 2)
                test_overlap = reverse_overlap

            # Calculate the Overall score & identify if the chunk is a continuity or rescheduled

            is_continuation = (test_overlap > seq_overlap_thr) or (overlap_count * scaling_factor) >= 1
            overall_score = amp_weight * amp_score + td_weight * time_deviation_score + dur_weight * dur_score

            # Based on is_continuation value assign it to continuity or rescheduled type

            if is_continuation:
                overall_score = min(overall_score * max_weight, 1)
                amp_score = min(amp_score * max_weight, 1)
                dur_score = min(dur_score * max_weight, 1)
                addition_type = seq_type['continued']
            else:
                overall_score = min(overall_score * min_weight, 1)
                amp_score = min(amp_score * min_weight, 1)
                dur_score = min(dur_score * min_weight, 1)
                addition_type = seq_type['rescheduled']

            worthy_bool = (overall_score >= overall_score_thr) & (amp_score >= amp_score_thr) & (dur_score >= dur_score_thr) & \
                          (time_deviation_score >= td_score_thr)

            if worthy_bool:
                chunk_inst_allotment_arr.append([chunk_inst_idx, all_inst_idx, addition_type, overall_score])

    return chunk_inst_allotment_arr
