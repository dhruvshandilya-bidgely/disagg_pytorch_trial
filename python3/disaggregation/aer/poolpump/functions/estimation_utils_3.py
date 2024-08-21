"""
Author - Arpan Agrawal
Date - 09/04/2019
Utility functions to help compute the estimate
"""

# Import python packages

import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.poolpump.functions.cleaning_utils import find_edges
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_days_labeled
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import find_potential_matches_all
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_distance_from_masked_edge

from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import roll_array
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_schedule
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_schedule_arr
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import remove_pairs_by_amp
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import remove_overlapping_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_valid_uncontested_idx
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_all_possible_schedules
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import remove_small_uncontested_pairs


def match_duration(input_dict, days_label, duration_each_day, num_of_runs_each_day, filled_days,
                   filled_previous_iterations, section=0):
    """Utility to match duration"""

    # Extract constants from config

    min_time_mask_score = 1
    window_size = 30

    uncontested_matrix = input_dict['uncontested_matrix']
    all_pos_edges = input_dict['all_pos_edges']
    all_neg_edges = input_dict['all_neg_edges']
    all_pairs = input_dict['all_pairs']
    window = input_dict['window']
    amp_ballpark = input_dict['amp_ballpark']

    total_days, total_time_divisions = uncontested_matrix.shape

    multiple_days_label_copy = days_label[window:min(total_days, min(total_days, (window + window_size)))].copy()
    multiple_days_label_copy[multiple_days_label_copy != 2] = 0
    multiple_start_arr, multiple_end_arr = find_edges(multiple_days_label_copy)

    for idx in range(len(multiple_start_arr)):

        pairs_idx = np.where((all_pairs[:, 2] <= (window + multiple_end_arr[idx])) &
                             (all_pairs[:, 3] >= (window + multiple_start_arr[idx])))[0]
        pairs = all_pairs[pairs_idx]

        if np.sum(filled_previous_iterations) > 0 and len(pairs) > 0:
            pairs = remove_overlapping_pairs(pairs, all_pos_edges, filled_previous_iterations, total_time_divisions,
                                             window)

        if len(pairs) > 0:
            pairs = remove_pairs_by_amp(pairs, all_pos_edges, all_neg_edges, amp_ballpark)

        if len(pairs) == 0:
            return uncontested_matrix

        multiple_pair_window_data = np.zeros(((multiple_end_arr[idx] - multiple_start_arr[idx]), total_time_divisions))
        multiple_pair_duration_each_day = duration_each_day[(window + multiple_start_arr[idx]):
                                                            (window + multiple_end_arr[idx])]

        multiple_pair_days_label = days_label[(window + multiple_start_arr[idx]):(window + multiple_end_arr[idx])]

        past_duration, _, future_duration, _ = past_future_parameters(uncontested_matrix, multiple_start_arr,
                                                                      multiple_end_arr, idx, window, section)

        for pair in pairs:
            if (pair[5] != past_duration and pair[5] != future_duration) or (
                    pair[6] < min_time_mask_score or pair[7] < min_time_mask_score):
                continue

            start_day = max(pair[2] - window - multiple_start_arr[idx], 0)
            end_day = min(pair[3] - window - multiple_start_arr[idx], len(multiple_pair_window_data))
            multiple_pair_window_data[start_day:end_day, all_pos_edges[pair[0]][2]] = all_pos_edges[pair[0]][3]
            multiple_pair_window_data[start_day:end_day, all_neg_edges[pair[1]][2]] = -all_neg_edges[pair[1]][3]

            multiple_pair_duration_each_day[start_day:end_day] = pair[5]

        window_num_runs = num_of_runs_each_day[(window + multiple_start_arr[idx]):(window + multiple_end_arr[idx])]

        multiple_pair_window_days_labeled = get_days_labeled(multiple_pair_window_data, window_num_runs,
                                                             (multiple_end_arr[idx] - multiple_start_arr[idx]))

        multiple_pair_window_days_labeled = remove_small_uncontested_pairs(multiple_pair_window_days_labeled,
                                                                           minimum_uncontested_pair_length=5)

        pre_uncontested_idx = np.where(multiple_pair_window_days_labeled == 1)[0]
        uncontested_idx = get_valid_uncontested_idx(pre_uncontested_idx)

        bool_arr = np.zeros((multiple_end_arr[idx] - multiple_start_arr[idx]), dtype=bool)
        bool_arr[uncontested_idx] = True

        multiple_filled_days = filled_days[(window + multiple_start_arr[idx]):(window + multiple_end_arr[idx])]
        multiple_filled_days[bool_arr] = 1

        multiple_pair_duration_each_day[~bool_arr] = 0
        multiple_pair_days_label[bool_arr] = 1

        uncontested_data = uncontested_matrix[(window + multiple_start_arr[idx] + uncontested_idx), :].copy()
        uncontested_matrix[(window + multiple_start_arr[idx] + uncontested_idx), :] = \
            np.where(uncontested_data != 0, uncontested_data, multiple_pair_window_data[uncontested_idx, :])

    return uncontested_matrix


def scoring_multiple_pairs(input_dict, days_label, duration_each_day, num_of_runs_each_day, filled_days,
                           filled_previous_iterations, section=0):
    """Utility to score multiple pairs"""

    window_size = 30

    uncontested_matrix = input_dict['uncontested_matrix']
    all_pos_edges = input_dict['all_pos_edges']
    all_neg_edges = input_dict['all_neg_edges']
    all_pairs = input_dict['all_pairs']
    window = input_dict['window']
    amp_ballpark = input_dict['amp_ballpark']

    total_days, total_time_divisions = uncontested_matrix.shape

    multiple_days_label_copy = days_label[window:min(total_days, min(total_days, (window + window_size)))].copy()
    multiple_days_label_copy[multiple_days_label_copy != 2] = 0
    multiple_start_arr, multiple_end_arr = find_edges(multiple_days_label_copy)

    for idx in range(len(multiple_start_arr)):
        pairs_idx = np.where((all_pairs[:, 2] < (window + multiple_end_arr[idx])) &
                             (all_pairs[:, 3] >= (window + multiple_start_arr[idx])))[0]
        pairs = all_pairs[pairs_idx]

        if np.sum(filled_previous_iterations) > 0 and len(pairs) > 0:
            pairs = remove_overlapping_pairs(pairs, all_pos_edges, filled_previous_iterations, total_time_divisions,
                                             window)

        if len(pairs) > 0:
            pairs = remove_pairs_by_amp(pairs, all_pos_edges, all_neg_edges, amp_ballpark)

        if len(pairs) == 0:
            return uncontested_matrix

        multiple_pair_window_df = np.zeros(((multiple_end_arr[idx] - multiple_start_arr[idx]), total_time_divisions))
        multiple_pair_duration_each_day = duration_each_day[(window + multiple_start_arr[idx]):
                                                            (window + multiple_end_arr[idx])]

        multiple_pair_days_label = days_label[(window + multiple_start_arr[idx]):(window + multiple_end_arr[idx])]

        past_duration, past_amp_ratio, future_duration, future_amp_ratio = past_future_parameters(
            uncontested_matrix, multiple_start_arr, multiple_end_arr, idx, window, section, past_duration=36,
            future_duration=36, past_amp_ratio=2, future_amp_ratio=2)

        dimensions = list()

        for pair in pairs:
            time_div_mask_dimension = pair[6] + pair[7]
            length_dimension = pair[4]
            signal_score_dimension = pair[8]
            deviation_from_duration_dimension = min(abs(pair[5] - past_duration), abs(pair[5] - future_duration))

            pair_amp_ratio = min(all_pos_edges[pair[0]][3], all_neg_edges[pair[1]][3]) / max(
                all_pos_edges[pair[0]][3], all_neg_edges[pair[1]][3])
            deviation_from_amp_ratio_dimension = min(abs(past_amp_ratio - pair_amp_ratio),
                                                     abs(future_amp_ratio - pair_amp_ratio))
            amp_ballpark_deviation = min(abs(amp_ballpark[3] - all_pos_edges[pair[0]][3]),
                                         abs(amp_ballpark[3] - all_neg_edges[pair[1]][3]))
            dimensions.append([time_div_mask_dimension, length_dimension, signal_score_dimension,
                               deviation_from_duration_dimension, deviation_from_amp_ratio_dimension,
                               amp_ballpark_deviation])

        dimensions = np.array(dimensions) / np.sum(np.array(dimensions), axis=0)
        dimensions[np.isnan(dimensions)] = 0
        weights = [0.3, 0.1, 0.2, -0.2, -0.2, 0]
        if amp_ballpark[-1]:
            weights = [0.3, 0.1, 0.2, -0.2, -0.2, 0.1]

        best_pair = np.zeros(all_pairs.shape[1], dtype=int)
        if len(dimensions) != 0:
            scores = dimensions * weights
            best_pair = pairs[np.argmax(np.sum(scores, axis=1))]

        start_day = max(best_pair[2] - window - multiple_start_arr[idx], 0)
        end_day = min(best_pair[3] - window - multiple_start_arr[idx], len(multiple_pair_window_df))
        multiple_pair_window_df[start_day:end_day, all_pos_edges[best_pair[0]][2]] = all_pos_edges[best_pair[0]][3]
        multiple_pair_window_df[start_day:end_day, all_neg_edges[best_pair[1]][2]] = -all_neg_edges[best_pair[1]][3]

        multiple_pair_duration_each_day[start_day:end_day] = best_pair[5]

        window_num_runs = num_of_runs_each_day[(window + multiple_start_arr[idx]):(window + multiple_end_arr[idx])]

        multiple_pair_window_days_labeled = get_days_labeled(multiple_pair_window_df, window_num_runs,
                                                             (multiple_end_arr[idx] - multiple_start_arr[idx]))
        multiple_pair_window_days_labeled = remove_small_uncontested_pairs(multiple_pair_window_days_labeled,
                                                                           minimum_uncontested_pair_length=5)

        pre_uncontested_idx = np.where(multiple_pair_window_days_labeled == 1)[0]
        uncontested_idx = get_valid_uncontested_idx(pre_uncontested_idx)

        bool_arr = np.zeros((multiple_end_arr[idx] - multiple_start_arr[idx]), dtype=bool)
        bool_arr[uncontested_idx] = True

        multiple_filled_days = filled_days[(window + multiple_start_arr[idx]):(window + multiple_end_arr[idx])]
        multiple_filled_days[bool_arr] = 1

        multiple_pair_duration_each_day[~bool_arr] = 0
        multiple_pair_days_label[bool_arr] = 1

        uncontested_df = uncontested_matrix[(window + multiple_start_arr[idx] + uncontested_idx), :].copy()
        uncontested_matrix[(window + multiple_start_arr[idx] + uncontested_idx), :] = \
            np.where(uncontested_df != 0, uncontested_df, multiple_pair_window_df[uncontested_idx, :])

    return uncontested_matrix


def create_virtual_pos_edges(neg_edges, section_arr, virtual_pos_edges, past_duration, future_duration,
                             total_time_divisions, time_div_dict, pos_df, pp_config):
    """Create virtual pos_edges for neg_edges"""

    for edge in neg_edges:

        past_time_div = (edge[2] - int(past_duration)) % total_time_divisions
        past_pos_edge = np.zeros(5)
        past_pos_edge[0] = edge[0]
        past_pos_edge[1] = edge[1]
        past_pos_edge[2] = past_time_div
        past_pos_edge[3] = edge[3]
        past_pos_edge[4] = 0

        past_pos_edge[4] += 2 * time_div_dict['primary_pos'][past_time_div]
        past_pos_edge[4] += time_div_dict['secondary_pos'][past_time_div]

        start_arr, end_arr = find_edges(pos_df[:, past_time_div])
        distance_from_masked_edge = get_distance_from_masked_edge(edge[0], edge[1], pos_df[:, past_time_div],
                                                                  start_arr, end_arr, pp_config)
        past_pos_edge[4] /= distance_from_masked_edge

        pair_arr = np.zeros(total_time_divisions)
        duration = (edge[2] - past_time_div) % total_time_divisions
        if duration == 0:
            duration = total_time_divisions
        time_div_arr = (np.arange(past_time_div, past_time_div + duration + 1)) % total_time_divisions
        pair_arr[time_div_arr] = 1

        if np.sum(pair_arr * section_arr) == 0:
            virtual_pos_edges.append(past_pos_edge)

        # virtual_pos_edges.append(past_pos_edge)

        future_time_div = (edge[2] - int(future_duration)) % total_time_divisions
        future_pos_edge = np.zeros(5)
        future_pos_edge[0] = edge[0]
        future_pos_edge[1] = edge[1]
        future_pos_edge[2] = future_time_div
        future_pos_edge[3] = edge[3]
        future_pos_edge[4] = 0

        future_pos_edge[4] += 2 * time_div_dict['primary_pos'][future_time_div]
        future_pos_edge[4] += time_div_dict['secondary_pos'][future_time_div]

        start_arr, end_arr = find_edges(pos_df[:, future_time_div])
        distance_from_masked_edge = get_distance_from_masked_edge(edge[0], edge[1], pos_df[:, future_time_div],
                                                                  start_arr, end_arr, pp_config)
        future_pos_edge[4] /= distance_from_masked_edge

        pair_arr = np.zeros(total_time_divisions)
        duration = (edge[2] - future_time_div) % total_time_divisions
        if duration == 0:
            duration = total_time_divisions
        time_div_arr = (np.arange(future_time_div, future_time_div + duration + 1)) % total_time_divisions
        pair_arr[time_div_arr] = 1

        if np.sum(pair_arr * section_arr) == 0:
            virtual_pos_edges.append(future_pos_edge)

    return virtual_pos_edges


def create_virtual_neg_edges(pos_edges, section_arr, virtual_neg_edges, past_duration, future_duration,
                             total_time_divisions, time_div_dict, neg_df, pp_config):
    """Create virtual neg_edges for pos_edges"""

    for edge in pos_edges:
        past_time_div = (edge[2] + int(past_duration)) % total_time_divisions
        past_neg_edge = np.zeros(5)
        past_neg_edge[0] = edge[0]
        past_neg_edge[1] = edge[1]
        past_neg_edge[2] = past_time_div
        past_neg_edge[3] = edge[3]
        past_neg_edge[4] = 0

        past_neg_edge[4] += 2 * time_div_dict['primary_neg'][past_time_div]
        past_neg_edge[4] += time_div_dict['secondary_neg'][past_time_div]

        start_arr, end_arr = find_edges(neg_df[:, past_time_div])
        distance_from_masked_edge = get_distance_from_masked_edge(edge[0], edge[1], neg_df[:, past_time_div],
                                                                  start_arr, end_arr, pp_config)
        past_neg_edge[4] /= distance_from_masked_edge

        pair_arr = np.zeros(total_time_divisions)
        if edge[2] < past_time_div:
            pair_arr[edge[2]:past_time_div + 1] = 1
        else:
            pair_arr[edge[2]:] = 1
            pair_arr[:past_time_div] = 1

        if np.sum(pair_arr * section_arr) == 0:
            virtual_neg_edges.append(past_neg_edge)

        future_time_div = (edge[2] + int(future_duration)) % total_time_divisions
        future_neg_edge = np.zeros(5)
        future_neg_edge[0] = edge[0]
        future_neg_edge[1] = edge[1]
        future_neg_edge[2] = future_time_div
        future_neg_edge[3] = edge[3]
        future_neg_edge[4] = 0

        future_neg_edge[4] += 2 * time_div_dict['primary_neg'][future_time_div]
        future_neg_edge[4] += time_div_dict['secondary_neg'][future_time_div]

        start_arr, end_arr = find_edges(neg_df[:, future_time_div])
        distance_from_masked_edge = get_distance_from_masked_edge(edge[0], edge[1], neg_df[:, future_time_div],
                                                                  start_arr, end_arr, pp_config)

        future_neg_edge[4] /= distance_from_masked_edge

        pair_arr = np.zeros(total_time_divisions)
        if edge[2] < future_time_div:
            pair_arr[edge[2]:future_time_div + 1] = 1
        else:
            pair_arr[edge[2]:] = 1
            pair_arr[:future_time_div] = 1

        if np.sum(pair_arr * section_arr) == 0:
            virtual_neg_edges.append(future_neg_edge)

    return virtual_neg_edges


def create_virtual_pos_neg_edges(clean_union, pos_edges, neg_edges, past_duration, future_duration, time_div_dict,
                                 total_time_divisions, pp_config, section=0):
    """Utility to create virtual positive negative edges"""

    df_copy = clean_union.copy()
    pos_df = df_copy.copy()
    pos_df[pos_df < 0] = 0
    pos_df[pos_df > 0] = 1
    neg_df = df_copy.copy()
    neg_df[neg_df > 0] = 0
    neg_df[neg_df < 0] = 1

    section_start = 0
    section_end = total_time_divisions - 1
    if section != 0:
        section_start = int(section / 100)
        section_end = int(section % 100)

    section_arr = np.ones(total_time_divisions)
    section_duration = (section_end - section_start) % total_time_divisions
    section_time_div_arr = (np.arange(section_start, section_start + section_duration + 1)) % total_time_divisions
    section_arr[section_time_div_arr] = 0

    virtual_pos_edges = list()
    virtual_neg_edges = list()

    virtual_neg_edges = create_virtual_neg_edges(
        pos_edges, section_arr, virtual_neg_edges, past_duration, future_duration, total_time_divisions, time_div_dict,
        neg_df, pp_config)

    virtual_pos_edges = create_virtual_pos_edges(
        neg_edges, section_arr, virtual_pos_edges, past_duration, future_duration, total_time_divisions, time_div_dict,
        pos_df, pp_config)

    return np.array(virtual_pos_edges, dtype=int), np.array(virtual_neg_edges, dtype=int)


def score_all_pairs_utility(input_dict, idx, e_s_arr, e_e_arr, filled_previous_iterations, pp_config):
    """Utility function for score_all_pairs"""

    section = input_dict['section']
    time_div_dict = input_dict['time_div_dict']
    amp_ballpark = input_dict['amp_ballpark']
    data_bl_removed = input_dict['data_bl_removed']
    data_clean_edges = input_dict['data_clean_edges']
    uncontested_matrix = input_dict['uncontested_matrix']
    all_pos_edges = input_dict['all_pos_edges']
    all_neg_edges = input_dict['all_neg_edges']
    all_smooth_pos_edges = input_dict['all_smooth_pos_edges']
    all_smooth_neg_edges = input_dict['all_smooth_neg_edges']
    window = input_dict['window']

    total_time_divisions = uncontested_matrix.shape[1]
    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / pp_config.get('sampling_rate'))

    pos_edges_idx = np.array([])
    neg_edges_idx = np.array([])
    smooth_pos_edges_idx = np.array([])
    smooth_neg_edges_idx = np.array([])

    if len(all_pos_edges) > 0:
        pos_edges_idx = \
            np.where((all_pos_edges[:, 0] < (window + e_e_arr[idx])) & (all_pos_edges[:, 1] > (window + e_s_arr[idx])))[
                0]
    if len(all_neg_edges) > 0:
        neg_edges_idx = \
            np.where((all_neg_edges[:, 0] < (window + e_e_arr[idx])) & (all_neg_edges[:, 1] > (window + e_s_arr[idx])))[
                0]
    if len(all_smooth_pos_edges) > 0:
        smooth_pos_edges_idx = np.where((all_smooth_pos_edges[:, 0] < (window + e_e_arr[idx])) &
                                        (all_smooth_pos_edges[:, 1] > (window + e_s_arr[idx])))[0]
    if len(all_smooth_neg_edges) > 0:
        smooth_neg_edges_idx = np.where((all_smooth_neg_edges[:, 0] < (window + e_e_arr[idx])) &
                                        (all_smooth_neg_edges[:, 1] > (window + e_s_arr[idx])))[0]

    past_duration, _, future_duration, _ = \
        past_future_parameters(uncontested_matrix, e_s_arr, e_e_arr, idx, window, section, past_duration=0,
                               future_duration=0, past_amp_ratio=0, future_amp_ratio=0)

    virtual_pos_edges, virtual_neg_edges = create_virtual_pos_neg_edges(data_clean_edges,
                                                                        all_pos_edges[pos_edges_idx],
                                                                        all_neg_edges[neg_edges_idx],
                                                                        past_duration, future_duration,
                                                                        time_div_dict, total_time_divisions,
                                                                        pp_config, section=section)

    pos_edges_idx = list(pos_edges_idx)
    neg_edges_idx = list(neg_edges_idx)
    smooth_pos_edges_idx = list(smooth_pos_edges_idx)
    smooth_neg_edges_idx = list(smooth_neg_edges_idx)

    clean_pos_edges = all_pos_edges[pos_edges_idx]
    clean_neg_edges = all_neg_edges[neg_edges_idx]
    smooth_pos_edges = all_smooth_pos_edges[smooth_pos_edges_idx]
    smooth_neg_edges = all_smooth_neg_edges[smooth_neg_edges_idx]

    if len(virtual_pos_edges) == 0:
        virtual_pos_edges = np.empty(shape=(0, 5))
    if len(virtual_neg_edges) == 0:
        virtual_neg_edges = np.empty(shape=(0, 5))
    if len(clean_pos_edges) == 0:
        clean_pos_edges = np.empty(shape=(0, 5))
    if len(clean_neg_edges) == 0:
        clean_neg_edges = np.empty(shape=(0, 5))
    if len(smooth_pos_edges) == 0:
        smooth_pos_edges = np.empty(shape=(0, 5))
    if len(smooth_neg_edges) == 0:
        smooth_neg_edges = np.empty(shape=(0, 5))

    pos_edges = np.vstack((virtual_pos_edges, clean_pos_edges, smooth_pos_edges)).astype(int)
    neg_edges = np.vstack((virtual_neg_edges, clean_neg_edges, smooth_neg_edges)).astype(int)

    pairs, _, _ = find_potential_matches_all(data_bl_removed, pos_edges, neg_edges, total_time_divisions,
                                             samples_per_hour, pp_config, minimum_pair_length=1, empty_flag=1)

    if np.sum(filled_previous_iterations) > 0 and len(pairs) > 0:
        pairs = remove_overlapping_pairs(pairs, pos_edges, filled_previous_iterations, total_time_divisions,
                                         window)

    if len(pairs) > 0:
        pairs = remove_pairs_by_amp(pairs, pos_edges, neg_edges, amp_ballpark)

    return pairs, pos_edges, neg_edges


def get_all_schedules(data_col, day):
    """Utility to get all schedules"""

    df_col_copy = np.sign(data_col)

    day_arr = df_col_copy[df_col_copy != 0]
    day_td = np.where(df_col_copy != 0)[0]
    day_arr, day_td = roll_array(day_arr, day_td)

    if len(day_td) == 0:
        return [0, 0, 0, 0, 0]

    all_schedules = list()
    transition_idx = np.where(day_arr[:-1] < day_arr[1:])[0]
    transition_idx += 1
    transition_idx = np.r_[0, transition_idx, len(day_arr)]

    for i in range(len(transition_idx) - 1):
        schedules = get_all_possible_schedules(day_arr[transition_idx[i]:transition_idx[i + 1]],
                                               day_td[transition_idx[i]:transition_idx[i + 1]])
        for idx in range(len(schedules)):
            schedules[idx] = np.r_[day, schedules[idx]]
        all_schedules.extend(schedules)

    return all_schedules


def get_pairs_idx(uncontested_matrix, start_day, end_day, global_pairs_matrix):
    """Get idx for pairs"""

    total_time_divisions = uncontested_matrix.shape[1]
    pairs_in_days_idx = np.where((global_pairs_matrix[:, 0] < end_day) & (global_pairs_matrix[:, 1] > start_day) &
                                 (global_pairs_matrix[:, 5] < end_day) & (global_pairs_matrix[:, 6] > start_day))[0]

    schedules_list = list()
    for day in range(start_day, end_day):
        schedules = get_all_schedules(uncontested_matrix[day, :], day)
        schedules_list.extend(schedules)

    schedules_df = pd.DataFrame(np.array(schedules_list, dtype=int))
    grouped_schedules = schedules_df.groupby([1, 2]).groups

    pairs_in_time_div_list = list()
    grouped_schedules = combine_overlapping_schedules(list(grouped_schedules.keys()), total_time_divisions)
    for schedule in grouped_schedules:
        pairs_in_time_div = \
            np.where((global_pairs_matrix[:, 2] == schedule[0]) & (global_pairs_matrix[:, 7] == schedule[1]))[0]
        pairs_in_time_div_list.extend(list(pairs_in_time_div))

    pairs_in_time_div_idx = np.array(pairs_in_time_div_list)
    desired_pairs = np.intersect1d(pairs_in_time_div_idx, pairs_in_days_idx)

    return desired_pairs, grouped_schedules


def past_future_parameters(uncontested_matrix, multiple_start_arr, multiple_end_arr, idx, window, section,
                           past_duration=0, future_duration=0, past_amp_ratio=0, future_amp_ratio=0):
    """Utility to find parameters from past and future schedules"""

    past_schedule_idx = np.where(np.any(uncontested_matrix[:(window + multiple_start_arr[idx]), :] > 0, axis=1))[0]
    future_schedule_idx = np.where(np.any(uncontested_matrix[(window + multiple_end_arr[idx]):, :] > 0, axis=1))[0]

    if len(past_schedule_idx) > 0:
        past_idx = past_schedule_idx[-1]
        past_schedule = get_schedule(uncontested_matrix[past_idx, :], section=section)
        past_duration = past_schedule[2]
        past_amp_ratio = min(past_schedule[3], past_schedule[4]) / max(past_schedule[3], past_schedule[4])

    if len(future_schedule_idx) > 0:
        future_idx = (window + multiple_end_arr[idx] + future_schedule_idx[0])
        future_schedule = get_schedule(uncontested_matrix[future_idx, :], section=section)
        future_duration = future_schedule[2]
        future_amp_ratio = min(future_schedule[3], future_schedule[4]) / max(future_schedule[3], future_schedule[4])

    return past_duration, past_amp_ratio, future_duration, future_amp_ratio


def combine_overlapping_schedules(grouped_schedules, total_time_divisions):
    """Finds non-overlapping sections for multiple run PP"""

    grouped_schedules_matrix = np.array(grouped_schedules)
    duration_arr = (grouped_schedules_matrix[:, 1] - grouped_schedules_matrix[:, 0]) % total_time_divisions
    grouped_schedules_matrix = np.c_[grouped_schedules_matrix, duration_arr]
    grouped_schedules = [grouped_schedules[i] for i in grouped_schedules_matrix[:, 2].argsort()[::-1]]
    is_schedule_selected = np.zeros(len(grouped_schedules))
    sections = list()

    for idx in range(len(grouped_schedules)):

        if is_schedule_selected[idx] == 0:
            section_new = list()
            section_new.append(grouped_schedules[idx])
            section_new_boundaries = (grouped_schedules[idx][0], grouped_schedules[idx][1])
            is_schedule_selected[idx] = 1

            for i in range(idx + 1, len(grouped_schedules)):

                if found_sections_overlapping(section_new_boundaries, grouped_schedules[i],
                                              total_time_divisions) and is_schedule_selected[i] == 0:
                    section_new.append(grouped_schedules[i])
                    is_schedule_selected[i] = 1
                    section_new_boundaries = get_section_new_boundaries(section_new, total_time_divisions)

            sections.append(section_new_boundaries)
    return sections


def found_sections_overlapping(section_boundaries, new_section, total_time_divisions):
    """Checks whether sections are overlapping"""

    section_arr = get_schedule_arr(section_boundaries[0], section_boundaries[1], total_time_divisions)
    new_section_arr = get_schedule_arr(new_section[0], new_section[1], total_time_divisions)

    if np.sum(section_arr * new_section_arr) > 0:
        return True
    return False


def get_section_new_boundaries(section_new, total_time_divisions):
    """Utility to expand section boundaries"""
    arr = np.zeros(total_time_divisions, dtype=int)

    for section in section_new:
        section_arr = get_schedule_arr(section[0], section[1], total_time_divisions)
        arr = np.bitwise_or(arr, np.array(section_arr, dtype=int))

    start_arr, end_arr = find_edges(arr)

    pos_time_div = start_arr[0]
    neg_time_div = end_arr[0] - 1
    if len(start_arr) > 1:
        pos_time_div = start_arr[-1]

    return pos_time_div, neg_time_div
