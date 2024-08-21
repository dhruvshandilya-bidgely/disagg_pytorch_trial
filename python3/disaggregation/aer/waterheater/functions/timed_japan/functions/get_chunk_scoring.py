"""
Author - Sahana M
Date - 20/07/2021
This function is used to get score for each chunk
"""

# Import python packages
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import chunk_indexes, seq_type


def handle_independent_chunk(chunk_data, window_consistency, window_amplitude, index, overall_idx, wh_config):
    """
    This function is used to score the independent chunks
    Parameters:
        chunk_data              (np.ndarray)    : Array containing the chunk information in the overall_idx division
        window_consistency      (np.ndarray)    : Contains time of day consistency pertaining to the current chunk
        window_amplitude        (np.ndarray)    : Contains time of day amplitude pertaining to the current chunk
        index                   (int)           : This is the index of the current chunk
        overall_idx             (int)           : This is the ith 30 day division of the whole data
        wh_config               (dict)          : Contains the twh configurations

    Returns:
        chunk_data              (np.ndarray)    : Array containing the chunk information in the overall_idx division
    """

    # Initialise the required variables

    amplitude_score = 0
    days = wh_config.get('days')
    end = int(chunk_data[index, chunk_indexes['chunk_end']])
    start = int(chunk_data[index, chunk_indexes['chunk_start']])
    default_independent_score = wh_config.get('default_independent_score')
    consistency_score = np.max(window_consistency[overall_idx, start:end]) / days

    # Score the chunk based on consistency & amplitude

    if np.max(window_amplitude[overall_idx, start:end]) != 0:
        amplitude_score = 1 - (np.max(window_amplitude[overall_idx, start:end]) -
                               np.median(window_amplitude[overall_idx, start:end])) / np.max(window_amplitude[overall_idx, start:end])
    within_chunk_score = (consistency_score + amplitude_score) / 2
    chunk_data[index, chunk_indexes['chunk_score']] = min(within_chunk_score, default_independent_score)

    return chunk_data


def get_connectivity_score(chunk_data, index, cols, time_info, wh_config):
    """
    This function is used to calculate the connectivity score
    Parameters:
        chunk_data          (np.ndarray)    : Array containing the chunk information in the overall_idx division
        index               (int)           : The chunk index in the current 30 day division chunk data
        cols                (int)           : Number of time divisions
        time_info           (dict)          : Contains all the time, duration based info from current & previous chunks
        wh_config           (dict)          : Contains the twh configurations

    Returns:
        connectivity_score  (float)         : Final Connectivity score
    """

    # Initialise the required variables

    curr_end_time = time_info.get('curr_end_time')
    curr_start_time = time_info.get('curr_start_time')
    current_duration = time_info.get('current_duration')
    previous_end_time = time_info.get('previous_end_time')
    previous_duration = time_info.get('previous_duration')
    previous_start_time = time_info.get('previous_start_time')
    default_connectivity_score = wh_config.get('default_connectivity_score')

    # If it a reschedule run then give it a connectivity score of 0.5 by default

    if chunk_data[index, chunk_indexes['label']] == seq_type['rescheduled']:
        connectivity_score = default_connectivity_score

    else:
        if previous_duration >= current_duration:
            previous_time_bool = np.full(cols, fill_value=False)
            previous_time_bool[previous_start_time: previous_end_time] = True
            overlap_count = np.sum(previous_time_bool[curr_start_time: curr_end_time])
            connectivity_score = np.round(overlap_count / (curr_end_time - curr_start_time), 2)
        else:
            reverse_time_bool = np.full(cols, fill_value=False)
            reverse_time_bool[curr_start_time: curr_end_time] = True
            overlap_count = np.sum(reverse_time_bool[previous_start_time: previous_end_time])
            connectivity_score = np.round(
                overlap_count / (previous_end_time - previous_start_time), 2)

    return connectivity_score


def backward_reinforcement(chunk_data, index, overall_chunk_data, wh_config):
    """
    This function provides reinforcement reward to the previous chunks belonging to the same merged seq based on the similarity
    Parameters:
        chunk_data              (np.ndarray)    : Array containing the chunk information in the overall_idx division
        index                   (int)           : The chunk index in the current 30 day division chunk data
        overall_chunk_data      (np.ndarray)    : Contains info about all the chunks identified
        wh_config               (dict)          : Contains the twh configurations

    Returns:
        overall_chunk_data      (np.ndarray)    : Contains info about all the chunks identified
    """

    # Initialise the required variables

    base_reward = wh_config.get('base_reward')
    propagating_reward = wh_config.get('propagating_reward')

    blocks_backward_rewarding = []
    previous_block_idx = int(chunk_data[index, chunk_indexes['previous_chunk_idx']])
    blocks_backward_rewarding.append(chunk_data[index, chunk_indexes['previous_chunk_idx']])

    # Get all the previous chunks indexes which belong to the same merged sequence

    while previous_block_idx >= 0:
        blocks_backward_rewarding.append(
            overall_chunk_data[previous_block_idx, chunk_indexes['previous_chunk_idx']])
        previous_block_idx = int(overall_chunk_data[previous_block_idx, chunk_indexes['previous_chunk_idx']])

    # Perform backwards rewarding

    blocks_backward_rewarding = blocks_backward_rewarding[:-1]
    moving_reward = base_reward
    for i in blocks_backward_rewarding:
        reward = propagating_reward * moving_reward
        moving_reward = moving_reward - reward
        overall_chunk_data[int(i), chunk_indexes['chunk_score']] += reward

    return overall_chunk_data


def chunk_scoring(overall_idx, overall_info, chunk_data, batch_data, wh_config):
    """
    Scoring function is intended to score each chunk based on its similarity with one of the sequences identified based
    on several features like connectivity, amplitude variability, time deviation, also includes backward score reinforcements.
    Parameters:
        overall_idx             (int)       : This is the ith 30 day division of the whole data
        overall_info            (dict)      : Dictionary contains all the info about amplitude, duration & consistency
        chunk_data              (np.ndarray): Array containing the chunk information in the overall_idx division
        batch_data              (np.ndarray): Input array containing the raw data in the current 30 day division
        wh_config               (dict)      : Contains the twh configurations

    Returns:
        overall_chunk_data      (np.ndarray): Array containing all the chunks information
    """

    # Initialise the required variables

    days = wh_config.get('days')
    cols = batch_data.shape[1]
    factor = wh_config.get('factor')
    overall_idx = int(overall_idx)
    min_amp_bar = wh_config.get('min_amp_bar')
    max_amp_bar = wh_config.get('max_amp_bar')
    td_score_weight = wh_config.get('td_score_weight')
    o_score_weight = wh_config.get('overall_score_weight')
    a_score_weight = wh_config.get('amplitude_score_weight')
    p_score_weight = wh_config.get('previous_score_weight')
    cr_score_weight_1 = wh_config.get('current_score_weight_1')
    cr_score_weight_2 = wh_config.get('current_score_weight_2')
    c_score_weight = wh_config.get('connectivity_score_weight')
    overall_chunk_data = overall_info['overall_chunk_data']
    con_score_weight = wh_config.get('consistency_score_weight')
    max_amp_percentile = wh_config.get('max_amplitude_percentile')
    window_amplitude = deepcopy(overall_info['overall_amplitude'])
    window_consistency = deepcopy(overall_info['overall_consistency'])

    # For each chunk in the chunk data do scoring

    for index in range(len(chunk_data)):

        # If the chunk is the starting chunk give it a independent chunk  status

        if chunk_data[index, chunk_indexes['previous_chunk_idx']] == -1:
            chunk_data = handle_independent_chunk(chunk_data, window_consistency, window_amplitude, index, overall_idx, wh_config)

        else:

            # Get the chunk data of the previous chunk to which the current chunk is merged with

            previous_chunk_idx = int(chunk_data[index, chunk_indexes['previous_chunk_idx']])
            previous_chunk_data = overall_chunk_data[previous_chunk_idx]

            # Get the start and end day information about current and previous chunks

            previous_start_day = int(previous_chunk_data[chunk_indexes['window_start']])
            previous_end_day = int(previous_start_day+ 30)
            current_start_day = int(chunk_data[index, chunk_indexes['window_start']])
            current_end_day = min(int(current_start_day + 30), batch_data.shape[0])

            # Get the start and end time, duration information about current and previous chunks

            curr_start_time = int(chunk_data[index, chunk_indexes['chunk_start']])
            curr_end_time = int(chunk_data[index, chunk_indexes['chunk_end']])
            previous_start_time = int(previous_chunk_data[chunk_indexes['chunk_start']])
            previous_end_time = int(previous_chunk_data[chunk_indexes['chunk_end']])
            previous_duration = previous_end_time - previous_start_time
            current_duration = curr_end_time - curr_start_time

            # Store the duration & time based info
            time_info = {
                'curr_end_time': curr_end_time,
                'curr_start_time': curr_start_time,
                'current_duration': current_duration,
                'previous_end_time': previous_end_time,
                'previous_duration': previous_duration,
                'previous_start_time': previous_start_time
            }

            # Connectivity score - determines the extent to which the current chunk is connected to its previous chunk

            connectivity_score = get_connectivity_score(chunk_data, index, cols, time_info, wh_config)

            # Get the previous chunk percentile & current chunk percentile

            previous_amplitude = np.percentile(batch_data[previous_start_day:(previous_end_day+1),
                                                          previous_start_time:previous_end_time], q=max_amp_percentile)
            current_amplitude = np.percentile(batch_data[current_start_day:(current_end_day+1),
                                                         curr_start_time:curr_end_time], q=max_amp_percentile)

            # Check if the current amplitude lies within the suitable range

            amplitude_score = 0
            if not (not (current_amplitude > min_amp_bar * previous_amplitude) or not (
                    current_amplitude < max_amp_bar * previous_amplitude)):
                amplitude_score = 1 - abs((1 - (current_amplitude/previous_amplitude)))

            # Calculate the overall score

            overall_score = (c_score_weight * connectivity_score + a_score_weight * amplitude_score)

            # Calculate time deviation score

            max_time_diff = max(abs(curr_start_time - previous_start_time), abs(curr_end_time - previous_end_time))
            time_deviation_score = max(1 - (max_time_diff/(2*factor)), 0)

            # Calculate the current chunk score

            previous_score = overall_chunk_data[previous_chunk_idx, chunk_indexes['chunk_score']]
            current_score = o_score_weight * overall_score + td_score_weight * time_deviation_score

            # Update the previous chunk score based on the current chunk score

            updated_previous_score = p_score_weight * previous_score + cr_score_weight_1 * current_score
            overall_chunk_data[previous_chunk_idx, chunk_indexes['chunk_score']] = updated_previous_score

            # Add the consistency score as well

            chunk_data[index, chunk_indexes['chunk_score']] = \
                cr_score_weight_2 * current_score + con_score_weight * (chunk_data[index, chunk_indexes['consistency_idx']]/days)

            # Backward Reinforcement

            overall_chunk_data = backward_reinforcement(chunk_data, index, overall_chunk_data, wh_config)

    replacing_indexes = np.where(overall_chunk_data[:, chunk_indexes['overall_index']] == overall_idx)[0]
    overall_chunk_data[replacing_indexes] = chunk_data

    return overall_chunk_data
