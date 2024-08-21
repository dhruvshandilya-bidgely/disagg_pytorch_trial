"""
Author - Sahana M
Date - 07/05/2021
Perform box cleaning
"""

# Import python packages
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import chunk_indexes, box_indexes, windows


def split_multiple_boxes(window_data):
    """This function is used to identify the discontinuous sequences
    Parameters:
        window_data         (np.ndarray)        : Window sequence
    Returns:
        split_window_data   (np.ndarray)        : split data
    """

    seq = window_data[:, box_indexes['start_col']]
    split_seq = seq[1:] - seq[:-1]
    sequences = np.split(seq, np.argwhere(split_seq>1).flatten() + 1)

    split_window_data = []
    for i in range(len(sequences)):
        indexes = np.isin(window_data[:, box_indexes['start_col']], sequences[i])
        split_window_data.append(window_data[indexes, :])

    return split_window_data


def single_point_handling(filtered_data, cleaned_data_matrix, overall_selected_boxes, chunk_info, amp_configs):
    """
    This function is used to fit best box on a single point data
    Parameters:
        filtered_data               (np.ndarray): 2D matrix containing the raw data corresponding to each chunk
        cleaned_data_matrix         (np.ndarray): 2D matrix where the best boxes will be present
        overall_selected_boxes      (np.ndarray): Best box candidates for the current day chunk
        chunk_info                  (dict)      : All the chunk related info
        amp_configs                 (dict)      : All the amplitude related configs

    Returns:
        overall_selected_boxes      (np.ndarray): Best box candidates for the current day chunk
    """

    # Extract the necessary data

    day = chunk_info['day_idx']
    end_idx = chunk_info['end_idx']
    min_bar = amp_configs['min_bar']
    max_bar = amp_configs['max_bar']
    start_idx = chunk_info['start_idx']
    buffer_amp = amp_configs['buffer_amp']
    current_chunk = chunk_info['current_chunk']

    # Define amplitude ranges for scoring

    single_point_amp = filtered_data[day, start_idx:end_idx]
    chunk_amp = current_chunk[chunk_indexes['amplitude_idx']]
    amp_low = chunk_amp * min_bar
    amp_high = (current_chunk[chunk_indexes['amplitude_idx']] + buffer_amp) * max_bar
    amp_mid = chunk_amp

    # If amplitude falls within the range perform scoring

    if (single_point_amp > amp_low) & (single_point_amp < amp_high):
        cleaned_data_matrix[day, start_idx:end_idx] = single_point_amp
        amp_score = np.exp((-abs(amp_mid - single_point_amp)) / ((amp_high - amp_low) / 2))
        box_info = np.array([day, start_idx, day, end_idx, single_point_amp, 1, amp_score])
        overall_selected_boxes.extend([box_info])

    return overall_selected_boxes


def get_boxes_within_chunk(window_size_array, box_fitting_data, chunk_info):
    """
    Function to identify the boxes within the time limits of the chunk in consideration
    Parameters:
        window_size_array       (list)      : Contains different window sizes for the sampling rate
        box_fitting_data        (dict)      : For each window size has the start row, start time, end row, end time, amplitude info of all the boxes
        chunk_info              (dict)      : All the chunk related info
    Returns:
        boxes_within_chunk      (np.ndarray): Best box candidates from different window size for the current day chunk
    """

    # Extract the necessary data

    day_idx = chunk_info['day_idx']
    end_idx = chunk_info['end_idx']
    start_idx = chunk_info['start_idx']
    boxes_within_chunk = []

    # For each window select the boxes occurring within the chunk start & end time

    for window in window_size_array:
        temp_1 = box_fitting_data[window]

        if not len(temp_1):
            continue

        temp_2 = np.logical_and((temp_1[:, box_indexes['start_row']] == day_idx),
                                (temp_1[:, box_indexes['end_row']] == day_idx))
        temp_3 = temp_1[temp_2, :]

        if not len(temp_3):
            continue

        temp_4 = np.logical_and((temp_3[:, box_indexes['start_col']] >= start_idx),
                                temp_3[:, box_indexes['end_col']] <= (end_idx + 1))
        temp_5 = temp_3[temp_4, :]

        boxes_within_chunk.extend(temp_5)

    boxes_within_chunk = np.array(boxes_within_chunk)

    return boxes_within_chunk


def get_best_box(boxes_within_chunk, cleaned_data_matrix, day, overall_selected_boxes, amp_config, wh_config):
    """
    This function is used to get the best boxes from each window size
    Args:
        boxes_within_chunk          (np.ndarray): Best box candidates from different window size for the current day chunk
        cleaned_data_matrix         (np.ndarray): 2D matrix where the best boxes will be present
        overall_selected_boxes      (np.ndarray): Best box candidates for the current day chunk
        day                         (int)       : Day under consideration
        amp_config                  (dict)      : All the amplitude related configs
        wh_config                   (dict)      : WH configurations dictionary

    Returns:
        cleaned_data_matrix         (np.ndarray): 2D matrix where the best boxes will be present
        overall_selected_boxes      (np.ndarray): Best box candidates for the current day chunk
    """

    # Extract the necessary data

    amp_low = amp_config.get('amp_low')
    amp_high = amp_config.get('amp_high')
    amp_mid = amp_config.get('amp_mid')
    dur_low = amp_config.get('dur_low')
    dur_high = amp_config.get('dur_high')
    dur_mid = amp_config.get('dur_mid')
    amp_weight = wh_config.get('box_fit_amp_weight')
    dur_weight = wh_config.get('box_fit_dur_weight')

    # Get the window size of each box
    boxes_within_chunk = np.c_[boxes_within_chunk, (boxes_within_chunk[:, box_indexes['end_col']] -
                                                    boxes_within_chunk[:, box_indexes['start_col']] + 1)]

    # Filter the boxes which fall within the amplitude ranges
    amp_correct_points = boxes_within_chunk[np.logical_and(boxes_within_chunk[:, box_indexes['amplitude']] >= amp_low,
                                                           boxes_within_chunk[:, box_indexes['amplitude']] <= amp_high), :]

    box_scores = []
    box_info_with_score = []

    subset_windows = np.unique(amp_correct_points[:, box_indexes['window_size']])

    # For each filtered window box, score these boxes

    for subset_window in subset_windows:
        window_data = amp_correct_points[amp_correct_points[:, box_indexes['window_size']] == subset_window]

        split_window_data = split_multiple_boxes(window_data)

        for i in range(len(split_window_data)):
            temp_data = split_window_data[i]
            box_amp = np.mean(temp_data[:, box_indexes['amplitude']])
            box_duration = abs(temp_data[-1, box_indexes['end_col']] - temp_data[0, box_indexes['start_col']])

            amp_score = np.exp((-abs(amp_mid - box_amp)) / ((amp_high - amp_low) / 2))
            duration_score = np.exp((-abs(dur_mid - box_duration)) / ((dur_high - dur_low) / 2))

            final_score = amp_weight * amp_score + dur_weight * duration_score

            final_score_arr = np.repeat(final_score, len(temp_data))
            temp_data = np.c_[temp_data, final_score_arr]
            box_info_with_score.append(temp_data)
            box_scores.append(final_score)

    # Select the best box with the highest score

    if len(box_scores):
        best_box_idx = np.argmax(box_scores)
        best_data_info = box_info_with_score[best_box_idx]

        box_start = int(best_data_info[0, box_indexes['start_col']])
        box_end = int(best_data_info[-1, box_indexes['end_col']])
        box_amp = best_data_info[0, box_indexes['amplitude']]

        cleaned_data_matrix[day, box_start:box_end] = box_amp

        overall_selected_boxes.extend(best_data_info)

    return cleaned_data_matrix, overall_selected_boxes


def get_amp_ranges(all_amplitudes, all_durations, amp_configs, factor):
    """
    Get the ranges of amplitude and duration for the current sequence
    Parameters:
        all_amplitudes          (np.ndarray)    : Amplitude of all the chunks detected in the sequence
        all_durations           (np.ndarray)    : Duration of all the chunks detected in the sequence
        amp_configs             (dict)          : All the amplitude related configs
        factor                  (float)         : Number of units in an hour

    Returns:
        amp_ranges              (list)          : Detected amplitude ranges
    """

    # Initialise the necessary local variables

    dur_hours = 4
    min_percentile = 25
    max_percentile = 95
    min_bar = amp_configs.get('min_bar')
    max_bar = amp_configs.get('max_bar')
    buffer_amp = amp_configs.get('buffer_amp')

    if len(all_amplitudes) > 1:
        amp_low = np.percentile(all_amplitudes, q=min_percentile) * min_bar
        amp_high = (np.percentile(all_amplitudes, q=max_percentile) + buffer_amp) * max_bar
        amp_mid = np.median(all_amplitudes)
        dur_low = np.min(all_durations) * factor
        dur_high = np.max(all_durations) * factor
        dur_mid = np.median(all_durations) * factor
    else:
        amp_low = np.min(all_amplitudes[0]) * min_bar
        amp_high = (np.max(all_amplitudes[0]) + buffer_amp) * max_bar
        amp_mid = all_amplitudes[0]
        dur_low = all_durations[0] * dur_hours * min_bar
        dur_high = all_durations[0] * dur_hours * max_bar
        dur_mid = all_durations[0] * dur_hours

    amp_ranges = [amp_low, amp_high, amp_mid, dur_low, dur_high, dur_mid]

    return amp_ranges


def special_single_point_handling(cleaned_data_matrix, filtered_data, overall_selected_boxes, all_amplitudes, chunk_info,
                                  amp_configs):
    """
    This function is used to handle the special case of a single point twh, but the chunk size is > 1
    Parameters:
        cleaned_data_matrix         (np.ndarray): 2D matrix where the best boxes will be present
        filtered_data               (np.ndarray): 2D matrix containing the raw data corresponding to each chunk
        overall_selected_boxes      (np.ndarray): Best box candidates for the current day chunk
        all_amplitudes              (np.ndarray): Amplitude of all the chunks detected in the sequence
        chunk_info                  (dict)      : All the chunk related info
        amp_configs                 (dict)      : All the amplitude related configs

    Returns:
        cleaned_data_matrix         (np.ndarray): 2D matrix where the best boxes will be present
        overall_selected_boxes      (np.ndarray): Best box candidates for the current day chunk
    """

    day = chunk_info['day_idx']
    start_idx = chunk_info['start_idx']
    end_idx = chunk_info['end_idx']
    min_bar = amp_configs.get('min_bar')
    max_bar = amp_configs.get('max_bar')
    buffer_amp = amp_configs.get('buffer_amp')

    # If a single point matching the amplitude ranges then consider adding it

    if np.sum(cleaned_data_matrix[day, start_idx:end_idx]) == 0:
        temp = filtered_data[day, start_idx: end_idx]
        amp_low = np.percentile(all_amplitudes, q=25) * min_bar
        amp_high = (np.percentile(all_amplitudes, q=95) + buffer_amp) * max_bar
        if np.sum((temp > amp_low) & (temp < amp_high)) == 1:
            within_temp_start = np.argwhere((temp > amp_low) & (temp < amp_high) > 0)[0][0]
            start = int(start_idx + within_temp_start)
            end = int(start + 1)
            single_point_amp = temp[within_temp_start]
            box_info = np.array([day, start, day, end, single_point_amp, 1, 1])
            cleaned_data_matrix[day, start:end] = single_point_amp
            overall_selected_boxes.extend([box_info])

    return cleaned_data_matrix, overall_selected_boxes


def n_point_handling(cleaned_data_matrix, filtered_data, overall_selected_boxes, chunk_info, wh_config):
    """
    This function is used to handle cases when several points in a box are missed due to improper boxes picked
    Parameters:
        cleaned_data_matrix         (np.ndarray): 2D matrix where the best boxes will be present
        filtered_data               (np.ndarray): 2D matrix containing the raw data corresponding to each chunk
        overall_selected_boxes      (np.ndarray): Best box candidates for the current day chunk
        chunk_info                  (dict)      : All the chunk related info
        wh_config                   (dict)      : WH configurations dictionary

    Returns:
        cleaned_data_matrix         (np.ndarray): 2D matrix where the best boxes will be present
        overall_selected_boxes      (np.ndarray): Best box candidates for the current day chunk
    """

    day = chunk_info['day_idx']
    factor = wh_config.get('factor')
    end_idx = chunk_info['end_idx']
    day_range = wh_config.get('day_range')
    start_idx = chunk_info['start_idx']
    amp_thr = wh_config.get('neighbour_days_amp_thr')

    # Check for the past and the next 6 days amplitude and decided whether to include the
    # current day box or not depending on the amplitude score

    if np.sum(cleaned_data_matrix[day, start_idx:end_idx]) == 0:
        temp = filtered_data[day, start_idx: start_idx + 2]
        previous_days = max(day - day_range, 0)
        next_days = min(day + day_range, filtered_data.shape[0]) + 1
        neighbour_days_amp = np.median(np.max(filtered_data[previous_days:next_days, start_idx:start_idx + 2], axis=1))
        amp_low = min(neighbour_days_amp - int(amp_thr / factor), 0)
        amp_high = neighbour_days_amp + int(amp_thr / factor)
        amp_mid = neighbour_days_amp
        if np.any(temp > amp_low) and np.any(temp < amp_high):
            index = int(np.where((temp > amp_low) & np.any(temp < amp_high) > 0)[0][0])
            n_point_amp = temp[index]
            amp_score = np.exp((-abs(amp_mid - n_point_amp)) / ((amp_high - amp_low) / 2))
            start = int(start_idx + index)
            end = int(start_idx + index + 1)
            box_info = np.array([day, start, day, end, n_point_amp, 1, amp_score])
            cleaned_data_matrix[day, start:end] = n_point_amp
            overall_selected_boxes.extend([box_info])

    return cleaned_data_matrix, overall_selected_boxes


def dynamic_box_cleaning(box_info, filtered_data, overall_chunk_data, wh_config):
    """
    From each window size for each box pick the best fitted box
    Parameters:
        box_info            (dict)      : For each window size has the start row, start time, end row, end time, amplitude info of all the boxes
        filtered_data       (np.ndarray): 2D matrix containing the raw data corresponding to each chunk
        overall_chunk_data  (np.ndarray): Contains info about all the chunks identified
        wh_config           (dict)      : WH configurations dictionary

    Returns:
        cleaned_data_matrix         (np.ndarray): 2D matrix where the best boxes will be present
        overall_selected_boxes      (np.ndarray): Best box candidates for the current day chunk
    """

    # Extract the necessary data

    cols = wh_config.get('cols')
    factor = wh_config.get('factor')
    rows = filtered_data.shape[0]
    window_size_array = windows[factor]
    box_fitting_data = deepcopy(box_info)
    min_bar = wh_config.get('box_min_amp_bar')
    max_bar = wh_config.get('box_max_amp_bar')
    sliding_window = wh_config.get('sliding_window')
    buffer_amp = int(wh_config.get('min_amp')/factor)

    overall_selected_boxes = []

    cleaned_data_matrix = np.full(shape=(rows, cols), fill_value=0.0)

    # For each day fit the best box data

    for day in range(rows):

        # Get the chunks present in the current day
        chunk_num = min(int(day/sliding_window), overall_chunk_data[-1, chunk_indexes['overall_index']])
        current_day_chunks = overall_chunk_data[overall_chunk_data[:, chunk_indexes['overall_index']] == chunk_num, :]

        # For each chunk obtained in the current day fit the best box data
        for idx in range(len(current_day_chunks)):

            current_chunk = current_day_chunks[idx, :]
            start_idx = int(current_chunk[chunk_indexes['chunk_start']])
            end_idx = int(current_chunk[chunk_indexes['chunk_end']])

            chunk_info = {
                'current_chunk': current_chunk,
                'day_idx': day,
                'start_idx': start_idx,
                'end_idx': end_idx
            }

            amp_configs = {
                'min_bar': min_bar,
                'max_bar': max_bar,
                'buffer_amp': buffer_amp,
                'factor': factor
            }

            # If single point chunks fit the box data accordingly

            if (end_idx - start_idx) == 1:
                overall_selected_boxes = single_point_handling(filtered_data, cleaned_data_matrix, overall_selected_boxes,
                                                               chunk_info, amp_configs)

            # If more than double point

            else:

                # Get the boxes within the chunks

                boxes_within_chunk = get_boxes_within_chunk(window_size_array, box_fitting_data, chunk_info)

                # Get the amplitude information of the current chunk

                merged_id = current_chunk[chunk_indexes['merged_seq']]
                all_amplitudes = overall_chunk_data[overall_chunk_data[:, chunk_indexes['merged_seq']] == merged_id, chunk_indexes['amplitude_idx']]
                all_durations = overall_chunk_data[overall_chunk_data[:, chunk_indexes['merged_seq']] == merged_id, chunk_indexes['duration_idx']]

                amp_ranges = get_amp_ranges(all_amplitudes, all_durations, amp_configs, factor)

                amp_low, amp_high, amp_mid, dur_low, dur_high, dur_mid = amp_ranges

                # Amp info =
                amp_info = {
                    'amp_low': amp_low,
                    'amp_high': amp_high,
                    'amp_mid': amp_mid,
                    'dur_low': dur_low,
                    'dur_high': dur_high,
                    'dur_mid': dur_mid
                }

                # Evaluate only if boxes are found

                if len(boxes_within_chunk):
                    cleaned_data_matrix, overall_selected_boxes = get_best_box(boxes_within_chunk, cleaned_data_matrix, day,
                                                                               overall_selected_boxes, amp_info, wh_config)

                # Special case handling when the single point lies somewhere in the middle in a larger chunk window

                cleaned_data_matrix, overall_selected_boxes = special_single_point_handling(cleaned_data_matrix, filtered_data,
                                                                                            overall_selected_boxes, all_amplitudes,
                                                                                            chunk_info, amp_configs)

                # N point special case handling

                cleaned_data_matrix, overall_selected_boxes = n_point_handling(cleaned_data_matrix, filtered_data,
                                                                               overall_selected_boxes, chunk_info, wh_config)

    overall_selected_boxes = np.array(overall_selected_boxes)

    return cleaned_data_matrix, overall_selected_boxes
