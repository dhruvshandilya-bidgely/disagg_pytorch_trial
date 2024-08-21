"""
Author - Mayank Sharan
Date - 22/1/19
Utility functions to help compute the estimate
"""

# Import python packages
import copy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.poolpump.functions.cleaning_utils import find_edges

from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_days_labeled
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_consistency_val
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_distance_from_masked_edge

from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import roll_array
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_schedule
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_section_int
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import edge_exists_bool
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import found_overlapping
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import pair_primary_bool
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import remove_pairs_by_amp
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import remove_overlapping_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import combine_overlapping_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_valid_uncontested_idx
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_all_possible_schedules
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_section_time_divisions
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import find_past_future_pairs_matrix
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import remove_small_uncontested_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import remove_undesired_amplitude_pairs

from python3.disaggregation.aer.poolpump.functions.estimation_utils_3 import past_future_parameters
from python3.disaggregation.aer.poolpump.functions.estimation_utils_3 import score_all_pairs_utility

from python3.disaggregation.aer.poolpump.functions.calculate_pp_conf_for_hybrid import calculate_pp_conf_for_hybrid


def create_virtual_pairs(input_dict, schedule_dict, empty_pair_window_df, duration_each_day, new_added_pairs,
                         filled_previous_iterations, pp_config, section=0):
    """Utility to create virutal pairs"""

    minimum_day_signal_fraction = pp_config.get('minimum_day_signal_fraction')
    minimum_match_signal_fraction = pp_config.get('minimum_match_signal_fraction')
    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / pp_config.get('sampling_rate'))

    clean_union = input_dict['data_clean_edges']
    y_signal_values = input_dict['data_bl_removed']
    window = input_dict['window']
    time_div_dict = input_dict['time_div_dict']
    amp_ballpark = input_dict['amp_ballpark']

    past_schedule = schedule_dict['past_schedule']
    future_schedule = schedule_dict['future_schedule']
    start_idx = schedule_dict['start_idx']
    end_idx = schedule_dict['end_idx']

    df_copy = clean_union.copy()

    pos_df = df_copy.copy()
    pos_df[pos_df < 0] = 0
    pos_df[pos_df > 0] = 1

    neg_df = df_copy.copy()
    neg_df[neg_df > 0] = 0
    neg_df[neg_df < 0] = 1
    neg_df = np.abs(neg_df)

    total_time_divisions = y_signal_values.shape[1]

    section_start = 0
    section_end = total_time_divisions - 1
    if section != 0:
        section_start = int(section / 100)
        section_end = int(section % 100)

    pos_edges = list()
    neg_edges = list()

    if past_schedule[2] != 0:
        pos_edge = np.zeros(5)
        pos_edge[0] = start_idx
        pos_edge[1] = end_idx
        pos_edge[2] = past_schedule[0]
        pos_edge[3] = past_schedule[3]
        pos_edge[4] = 0

        pos_edge[4] += 2 * time_div_dict['primary_pos'][past_schedule[0]]
        pos_edge[4] += time_div_dict['secondary_pos'][past_schedule[0]]

        start_arr, end_arr = find_edges(pos_df[:, past_schedule[0]])
        dist_masked_edge = get_distance_from_masked_edge(window + start_idx, window + end_idx,
                                                         pos_df[:, past_schedule[0]], start_arr, end_arr, pp_config)
        pos_edge[4] /= dist_masked_edge
        pos_edges.append(pos_edge)

        neg_edge = np.zeros(5)
        neg_edge[0] = start_idx
        neg_edge[1] = end_idx
        neg_edge[2] = past_schedule[1]
        neg_edge[3] = past_schedule[4]
        neg_edge[4] = 0

        neg_edge[4] += 2 * time_div_dict['primary_neg'][past_schedule[1]]
        neg_edge[4] += time_div_dict['secondary_neg'][past_schedule[1]]

        start_arr, end_arr = find_edges(neg_df[:, past_schedule[1]])
        dist_masked_edge = get_distance_from_masked_edge(window + start_idx, window + end_idx,
                                                         neg_df[:, past_schedule[1]], start_arr, end_arr, pp_config)

        neg_edge[4] /= dist_masked_edge
        neg_edges.append(neg_edge)

    if future_schedule[2] != 0 and (future_schedule[0] != past_schedule[0] or future_schedule[1] != past_schedule[1]):
        pos_edge = np.zeros(5)
        pos_edge[0] = start_idx
        pos_edge[1] = end_idx
        pos_edge[2] = future_schedule[0]
        pos_edge[3] = future_schedule[3]
        pos_edge[4] = 0

        pos_edge[4] += 2 * time_div_dict['primary_pos'][future_schedule[0]]
        pos_edge[4] += time_div_dict['secondary_pos'][future_schedule[0]]

        start_arr, end_arr = find_edges(pos_df[:, future_schedule[0]])
        dist_masked_edge = get_distance_from_masked_edge(window + start_idx, window + end_idx,
                                                         pos_df[:, future_schedule[0]], start_arr, end_arr, pp_config)
        pos_edge[4] /= dist_masked_edge

        pos_edges.append(pos_edge)

        neg_edge = np.zeros(5)
        neg_edge[0] = start_idx
        neg_edge[1] = end_idx
        neg_edge[2] = future_schedule[1]
        neg_edge[3] = future_schedule[4]
        neg_edge[4] = 0

        neg_edge[4] += 2 * time_div_dict['primary_neg'][future_schedule[1]]
        neg_edge[4] += time_div_dict['secondary_neg'][future_schedule[1]]

        start_arr, end_arr = find_edges(neg_df[:, future_schedule[1]])
        dist_masked_edge = get_distance_from_masked_edge(window + start_idx, window + end_idx,
                                                         neg_df[:, future_schedule[1]], start_arr, end_arr, pp_config)

        neg_edge[4] /= dist_masked_edge
        neg_edges.append(neg_edge)

    pos_edges = np.array(pos_edges, dtype=int)
    neg_edges = np.array(neg_edges, dtype=int)

    virtual_pair = list()

    for idx in range(len(pos_edges)):

        match_start_day = max(pos_edges[idx][0], neg_edges[idx][0])
        match_end_day = min(pos_edges[idx][1], neg_edges[idx][1])
        amp = min(pos_edges[idx][3], neg_edges[idx][3])
        pos_col = pos_edges[idx][2]
        neg_col = neg_edges[idx][2]

        day_signal_fraction, match_signal_fraction = get_consistency_val(y_signal_values, match_start_day,
                                                                         match_end_day, amp, pos_col, neg_col,
                                                                         pp_config, window, samples_per_hour)

        if (day_signal_fraction < minimum_day_signal_fraction or match_signal_fraction < minimum_match_signal_fraction) \
            or (int(pos_edges[idx][2]) < section_start or int(neg_edges[idx][2]) > section_end):
            continue

        signal_score = int(((day_signal_fraction + match_signal_fraction) / 2) * 100)

        virtual_pair.append([idx, idx, window + start_idx, window + end_idx, (end_idx - start_idx),
                             (neg_edges[idx][2] - pos_edges[idx][2]) % total_time_divisions, pos_edges[idx][4],
                             neg_edges[idx][4], signal_score])

    filled_bool = False

    if len(virtual_pair) > 0:
        virtual_pair = remove_pairs_by_amp(np.array(virtual_pair), pos_edges, neg_edges, amp_ballpark)

    if len(virtual_pair) == 0 or len(virtual_pair) > 1:
        return empty_pair_window_df, duration_each_day, filled_bool, new_added_pairs

    if np.sum(filled_previous_iterations) > 0 and len(virtual_pair) > 0:
        virtual_pair = remove_overlapping_pairs(np.array(virtual_pair), pos_edges, filled_previous_iterations,
                                                total_time_divisions, window=0)

    if len(virtual_pair) != 1:
        return empty_pair_window_df, duration_each_day, filled_bool, new_added_pairs

    virtual_pair = np.array(virtual_pair[0])

    empty_pair_window_df[:(end_idx - start_idx), pos_edges[virtual_pair[0]][2]] = pos_edges[virtual_pair[0]][3]
    empty_pair_window_df[:(end_idx - start_idx), neg_edges[virtual_pair[1]][2]] = -neg_edges[virtual_pair[1]][3]

    duration_each_day[(window + start_idx):(window + end_idx)] = (neg_edges[virtual_pair[1]][2] -
                                                                  pos_edges[virtual_pair[0]][2]) % \
                                                                 (y_signal_values.shape[1])

    filled_bool = True

    pos_edge_start_day = virtual_pair[2]
    pos_edge_end_day = virtual_pair[3]
    pos_edge_time_div = pos_edges[virtual_pair[0]][2]
    pos_edge_amp = pos_edges[virtual_pair[0]][3]
    pos_edge_time_mask = pos_edges[virtual_pair[0]][4]

    neg_edge_start_day = virtual_pair[2]
    neg_edge_end_day = virtual_pair[3]
    neg_edge_time_div = neg_edges[virtual_pair[1]][2]
    neg_edge_amp = neg_edges[virtual_pair[1]][3]
    neg_edge_time_mask = neg_edges[virtual_pair[1]][4]

    signal_score = virtual_pair[-1]

    new_added_pairs.append(
        [pos_edge_start_day, pos_edge_end_day, pos_edge_time_div, pos_edge_amp, pos_edge_time_mask,
         neg_edge_start_day, neg_edge_end_day, neg_edge_time_div, neg_edge_amp, neg_edge_time_mask,
         signal_score])

    return empty_pair_window_df, duration_each_day, filled_bool, new_added_pairs


def fill_virtual_edges(input_dict, uncontested_matrix, new_added_pairs, days_label, duration_each_day,
                       num_of_runs_each_day, filled_days, filled_previous_iterations, pp_config, section=0):

    """Utility to fill virtual edges in empty days"""

    window_size = pp_config.get('window_size')

    window = input_dict['window']

    total_days, total_time_divisions = uncontested_matrix.shape
    empty_days_label_copy = filled_days[window:min(total_days, min(total_days, (window + window_size)))].copy()
    empty_days_label_copy[empty_days_label_copy != 0] = -1
    empty_start_arr, empty_end_arr = find_edges(empty_days_label_copy + 1)

    if np.sum(empty_days_label_copy) == 0:
        empty_start_arr, empty_end_arr = np.array([0]), np.array([(min(total_days, (window + window_size)) - window)])

    for idx in range(len(empty_start_arr)):
        past_schedule_idx = np.where(np.any(uncontested_matrix[:(window + empty_start_arr[idx]), :] > 0, axis=1))[0]
        future_schedule_idx = np.where(np.any(uncontested_matrix[(window + empty_end_arr[idx]):, :] > 0, axis=1))[0]

        if len(past_schedule_idx) > 0:
            past_idx = past_schedule_idx[-1]
            past_schedule = get_schedule(uncontested_matrix[past_idx, :], section=section)
        else:
            past_schedule = [0, 0, 0, 0, 0]

        if len(future_schedule_idx) > 0:
            future_idx = (window + empty_end_arr[idx] + future_schedule_idx[0])
            future_schedule = get_schedule(uncontested_matrix[future_idx, :], section=section)
        else:
            future_schedule = [0, 0, 0, 0, 0]

        empty_pair_window_df = np.zeros(((empty_end_arr[idx] - empty_start_arr[idx]), total_time_divisions))

        schedule_dict = dict()
        schedule_dict['past_schedule'] = past_schedule
        schedule_dict['future_schedule'] = future_schedule
        schedule_dict['start_idx'] = empty_start_arr[idx]
        schedule_dict['end_idx'] = empty_end_arr[idx]

        empty_pair_window_df, duration_each_day, filled_bool, new_added_pairs = create_virtual_pairs(
            input_dict, schedule_dict, empty_pair_window_df, duration_each_day, new_added_pairs,
            filled_previous_iterations, pp_config, section=section)

        if not filled_bool:
            duration_each_day[(window + empty_start_arr[idx]):(window + empty_end_arr[idx])] = 0
            continue
        filled_days[(window + empty_start_arr[idx]):(window + empty_end_arr[idx])] = 1

        empty_pair_duration_each_day = duration_each_day[(window + empty_start_arr[idx]):(window + empty_end_arr[idx])]

        empty_pair_days_label = days_label[(window + empty_start_arr[idx]):(window + empty_end_arr[idx])]

        window_num_runs = num_of_runs_each_day[(window + empty_start_arr[idx]):(window + empty_end_arr[idx])]

        empty_pair_window_days_labeled = get_days_labeled(empty_pair_window_df, window_num_runs,
                                                          (empty_end_arr[idx] - empty_start_arr[idx]))
        empty_pair_window_days_labeled = remove_small_uncontested_pairs(empty_pair_window_days_labeled,
                                                                        minimum_uncontested_pair_length=1)

        pre_uncontested_idx = np.where(empty_pair_window_days_labeled == 1)[0]
        uncontested_idx = get_valid_uncontested_idx(pre_uncontested_idx, min_length_for_validity=1)

        bool_arr = np.zeros((empty_end_arr[idx] - empty_start_arr[idx]), dtype=bool)
        bool_arr[uncontested_idx] = True

        empty_filled_days = filled_days[(window + empty_start_arr[idx]):(window + empty_end_arr[idx])]
        empty_filled_days[bool_arr] = 1

        empty_pair_duration_each_day[~bool_arr] = 0
        empty_pair_days_label[bool_arr] = 1

        uncontested_df = uncontested_matrix[(window + empty_start_arr[idx] + uncontested_idx), :].copy()
        uncontested_matrix[(window + empty_start_arr[idx] + uncontested_idx), :] = \
            np.where(uncontested_df != 0, uncontested_df, empty_pair_window_df[uncontested_idx, :])

    return uncontested_matrix, new_added_pairs


def score_all_pairs(input_dict, all_smooth_pos_edges, all_smooth_neg_edges, new_added_pairs, filled_days,
                    filled_previous_iterations, pp_config, section=0):
    """Utility to score all pairs"""
    window_size = pp_config.get('window_size')

    uncontested_matrix = input_dict['uncontested_matrix']
    data_clean_edges = input_dict['data_clean_edges']
    data_bl_removed = input_dict['data_bl_removed']
    all_pos_edges = input_dict['all_pos_edges']
    all_neg_edges = input_dict['all_neg_edges']
    time_div_dict = input_dict['time_div_dict']
    amp_ballpark = input_dict['amp_ballpark']
    window = input_dict['window']

    total_days, total_time_divisions = uncontested_matrix.shape

    window_filled_days = filled_days[window:min(total_days, (window + window_size))]
    iter_idx = 0

    weights = [0.05, 0.02, 0.4, -0.2, -0.2, 0]
    if amp_ballpark[-1]:
        weights = [0.05, 0.02, 0.4, -0.2, -0.2, 0.1]

    while np.sum(window_filled_days) < min(total_days - window, window_size) and iter_idx < 10:

        empty_days_label_copy = filled_days[window:min(total_days, (window + window_size))].copy()
        empty_days_label_copy[empty_days_label_copy != 0] = -1
        e_s_arr, e_e_arr = find_edges(empty_days_label_copy + 1)

        if np.sum(empty_days_label_copy) == 0:
            e_s_arr, e_e_arr = np.array([0]), np.array([(min(total_days, (window + window_size)) - window)])

        for idx in range(len(e_s_arr)):
            input_dict = dict()
            input_dict['section'] = section
            input_dict['time_div_dict'] = time_div_dict
            input_dict['amp_ballpark'] = amp_ballpark
            input_dict['data_bl_removed'] = data_bl_removed
            input_dict['data_clean_edges'] = data_clean_edges
            input_dict['uncontested_matrix'] = uncontested_matrix
            input_dict['all_pos_edges'] = all_pos_edges
            input_dict['all_neg_edges'] = all_neg_edges
            input_dict['all_smooth_pos_edges'] = all_smooth_pos_edges
            input_dict['all_smooth_neg_edges'] = all_smooth_neg_edges
            input_dict['window'] = window

            pairs, pos_edges, neg_edges = score_all_pairs_utility(input_dict, idx, e_s_arr, e_e_arr,
                                                                  filled_previous_iterations, pp_config)

            if len(pairs) == 0:
                filled_days[(window + e_s_arr[idx]):(window + e_e_arr[idx])] = 1
                continue

            dimensions = list()

            past_duration, past_amp_ratio, future_duration, future_amp_ratio = past_future_parameters(
                uncontested_matrix, e_s_arr, e_e_arr, idx, window, section, past_duration=36, future_duration=36,
                past_amp_ratio=2, future_amp_ratio=2)

            for pair in pairs:
                time_div_mask_dimension = pair[6] + pair[7]
                length_dimension = pair[4]
                signal_score_dimension = pair[8]
                deviation_from_duration_dimension = min(abs(pair[5] - past_duration),
                                                        abs(pair[5] - future_duration))

                pair_amp_ratio = min(pos_edges[pair[0]][3], neg_edges[pair[1]][3]) / max(pos_edges[pair[0]][3],
                                                                                         neg_edges[pair[1]][3])
                deviation_from_amp_ratio_dimension = min(abs(past_amp_ratio - pair_amp_ratio),
                                                         abs(future_amp_ratio - pair_amp_ratio))
                amp_ballpark_deviation = min(abs(amp_ballpark[3] - pos_edges[pair[0]][3]),
                                             abs(amp_ballpark[3] - neg_edges[pair[1]][3]))

                dimensions.append([time_div_mask_dimension, length_dimension, signal_score_dimension,
                                   deviation_from_duration_dimension, deviation_from_amp_ratio_dimension,
                                   amp_ballpark_deviation])

            dimensions = np.array(dimensions) / np.sum(np.array(dimensions), axis=0)
            dimensions[np.isnan(dimensions)] = 0

            scores = dimensions * weights
            best_pair = pairs[np.argmax(np.sum(scores, axis=1))]

            start_day = max((best_pair[2] - window - e_s_arr[idx]), 0)
            end_day = min((best_pair[3] - window - e_s_arr[idx]),
                          (e_e_arr[idx] - e_s_arr[idx]))

            uncontested_matrix[(window + e_s_arr[idx] + start_day):
                               (window + e_s_arr[idx] + end_day), pos_edges[best_pair[0]][2]] = \
                pos_edges[best_pair[0]][3]

            uncontested_matrix[(window + e_s_arr[idx] + start_day):
                               (window + e_s_arr[idx] + end_day), neg_edges[best_pair[1]][2]] = \
                -neg_edges[best_pair[1]][3]

            filled_days[(window + e_s_arr[idx] + start_day):(window + e_s_arr[idx] + end_day)] = 1

            iter_idx += 1

            pos_edge_start_day = window + e_s_arr[idx] + start_day
            pos_edge_end_day = window + e_s_arr[idx] + end_day
            pos_edge_time_div = pos_edges[best_pair[0]][2]
            pos_edge_amp = pos_edges[best_pair[0]][3]
            pos_edge_time_mask = pos_edges[best_pair[0]][4]
            neg_edge_start_day = window + e_s_arr[idx] + start_day
            neg_edge_end_day = window + e_s_arr[idx] + end_day
            neg_edge_time_div = neg_edges[best_pair[1]][2]
            neg_edge_amp = neg_edges[best_pair[1]][3]
            neg_edge_time_mask = neg_edges[best_pair[1]][4]
            signal_score = best_pair[-1]
            new_added_pairs.append(
                [pos_edge_start_day, pos_edge_end_day, pos_edge_time_div, pos_edge_amp, pos_edge_time_mask,
                 neg_edge_start_day, neg_edge_end_day, neg_edge_time_div, neg_edge_amp, neg_edge_time_mask,
                 signal_score])

    return uncontested_matrix, new_added_pairs


def get_pair_dimensions(uncontested_matrix, start_day, end_day, pair):
    """Utility to get pair dimensions"""

    dimensions = list()
    total_time_divisions = uncontested_matrix.shape[1]
    section = (pair[2], pair[6])
    section_int = get_section_int(section)
    past_schedule_idx = np.where(np.any(uncontested_matrix[:start_day, :] > 0, axis=1))[0]
    future_schedule_idx = np.where(np.any(uncontested_matrix[end_day:, :] > 0, axis=1))[0]
    past_duration = 36
    future_duration = 36
    past_amp_ratio = 2
    future_amp_ratio = 2

    if len(past_schedule_idx) > 0:
        past_idx = past_schedule_idx[-1]
        past_schedule = get_schedule(uncontested_matrix[past_idx, :], section=section_int)
        past_duration = past_schedule[2]
        past_amp_ratio = min(past_schedule[3], past_schedule[4]) / max(past_schedule[3], past_schedule[4])

    if len(future_schedule_idx) > 0:
        future_idx = (end_day + future_schedule_idx[0])
        future_schedule = get_schedule(uncontested_matrix[future_idx, :], section=section_int)
        future_duration = future_schedule[2]
        future_amp_ratio = min(future_schedule[3], future_schedule[4]) / max(future_schedule[3], future_schedule[4])

    time_div_mask_dimension = pair[4] + pair[9]
    length_dimension = min(pair[1], pair[6]) - max(pair[0], pair[5])
    signal_score_dimension = pair[10]

    pair_duration = (pair[7] - pair[2]) % total_time_divisions
    deviation_from_duration_dimension = min(abs(pair_duration - past_duration),
                                            abs(pair_duration - future_duration))

    pair_amp_ratio = min(pair[3], pair[8]) / max(pair[3], pair[8])
    deviation_from_amp_ratio_dimension = min(abs(past_amp_ratio - pair_amp_ratio),
                                             abs(future_amp_ratio - pair_amp_ratio))

    dimensions.extend([time_div_mask_dimension, length_dimension, signal_score_dimension,
                       deviation_from_duration_dimension,
                       deviation_from_amp_ratio_dimension])

    return dimensions


def get_best_scored_pairs(uncontested_matrix, start_day, end_day, time_div_list, pairs, grouped_schedules, common_runs):
    """Utility to get best scored pairs"""

    each_time_div_end_days_list = list()

    for time_div in np.unique(time_div_list):
        _, end_arr = find_edges(abs(uncontested_matrix[start_day:end_day, time_div]))
        end_arr += start_day
        each_time_div_end_days_list.append(end_arr[0])
    each_time_div_end_days_list = list(set([start_day] + list(set(each_time_div_end_days_list)) + [end_day]))
    dividing_days = np.sort(np.array(each_time_div_end_days_list))

    for day_idx in range(len(dividing_days) - 1):
        pairs_in_division = pairs[np.where(
            (pairs[:, 0] < dividing_days[day_idx + 1]) & (pairs[:, 1] > dividing_days[day_idx]) &
            (pairs[:, 5] < dividing_days[day_idx + 1]) & (pairs[:, 6] > dividing_days[day_idx]))[0]]

        pairs_in_division = combine_overlapping_pairs(pairs_in_division, grouped_schedules)

        dimensions = list()
        for pair in pairs_in_division:
            pair_dimensions = get_pair_dimensions(uncontested_matrix, start_day, end_day, pair)
            dimensions.append(pair_dimensions)

        dimensions = np.array(dimensions) / np.sum(np.array(dimensions), axis=0)
        dimensions[np.isnan(dimensions)] = 0
        weights = [0.05, 0.02, 0.4, -0.2, -0.2]

        if len(dimensions) == 0:
            continue

        scores = dimensions * weights
        sorted_scores_idx = np.argsort(np.sum(scores, axis=1))[::-1]

        keep_pairs_df = np.zeros(((dividing_days[day_idx + 1] - dividing_days[day_idx]), uncontested_matrix.shape[1]))
        for idx in sorted_scores_idx[:common_runs]:
            keep_pairs_df[:, pairs_in_division[idx][2]] = pairs_in_division[idx][3]
            keep_pairs_df[:, pairs_in_division[idx][7]] = -pairs_in_division[idx][8]

        uncontested_matrix[dividing_days[day_idx]:dividing_days[day_idx + 1], :] = keep_pairs_df

    return uncontested_matrix


def get_filled_previous_iterations(data):
    """Utility to get filled previous iterations"""

    data_rows, total_time_divisions = data.shape

    blocked_window_matrix = np.zeros_like(data)

    data_sign = np.sign(data)

    for day in range(data_rows):
        data_day = data_sign[day, :]
        day_arr = data_day[data_day != 0]
        day_td = np.where(data_day != 0)[0]

        if len(day_arr) == 0:
            continue

        day_arr, day_td = roll_array(day_arr, day_td)

        all_schedules = list()
        transition_idx = np.where(day_arr[:-1] < day_arr[1:])[0]
        transition_idx += 1
        transition_idx = np.r_[0, transition_idx, len(day_arr)]

        for i in range(len(transition_idx) - 1):
            schedules = get_all_possible_schedules(day_arr[transition_idx[i]:transition_idx[i + 1]],
                                                   day_td[transition_idx[i]:transition_idx[i + 1]])
            all_schedules.extend(schedules)
        all_run_schedules = np.array(all_schedules)

        for schedule in all_run_schedules:
            duration = (schedule[1] - schedule[0]) % total_time_divisions
            time_div_to_fill = np.arange(schedule[0], schedule[0] + duration + 1) % total_time_divisions
            blocked_window_matrix[day, time_div_to_fill] = 1

    return blocked_window_matrix


def get_overlapping_pairs_sections(window, window_size, all_pairs, all_pos_edges, all_neg_edges, total_time_divisions):
    """Utility to get Overlapping pair sections"""

    sections = list()
    pairs_idx = np.where((all_pairs[:, 2] < (window + window_size)) & (all_pairs[:, 3] > window))[0]
    pairs_list = all_pairs[pairs_idx]
    pairs_list = pairs_list[pairs_list[:, 5].argsort()][::-1]

    is_pair_selected = np.zeros(len(pairs_list))

    for i in range(len(is_pair_selected)):

        if is_pair_selected[i] == 0:
            section_new = list()
            section_new.append(pairs_list[i])
            section_new_boundaries = (all_pos_edges[pairs_list[i][0]][2], all_neg_edges[pairs_list[i][1]][2])
            is_pair_selected[i] = 1

            for j in range(i + 1, len(pairs_list)):

                if found_overlapping(section_new_boundaries, pairs_list[j], all_pos_edges, all_neg_edges,
                                     total_time_divisions) and is_pair_selected[j] == 0:
                    section_new.append(pairs_list[j])
                    is_pair_selected[j] = 1

                    section_new_boundaries = get_section_time_divisions(section_new, all_pos_edges, all_neg_edges,
                                                                        total_time_divisions)

            section_new_time_divisions = get_section_time_divisions(section_new, all_pos_edges,
                                                                    all_neg_edges, total_time_divisions)
            sections.append(section_new_time_divisions)

    return sections


def signal_consistency_bool(raw_data, pair, schedule, samples_per_hour, pp_config, minimum_day_signal_fraction=0.8,
                            minimum_match_signal_fraction=0.5):
    """Utility to get signal consistency bool"""

    day_signal_fraction, match_signal_fraction = get_consistency_val(raw_data, pair[2], pair[3], pair[4], schedule[0],
                                                                     schedule[1], pp_config, 0, samples_per_hour)

    if day_signal_fraction < minimum_day_signal_fraction or match_signal_fraction < minimum_match_signal_fraction:
        return False

    return True


def post_process_pairs(uncontested_matrix, clean_edges, raw_data, pairs, total_time_divisions, samples_per_hour,
                       pp_config, runs=1):
    """Utility to post process pairs"""

    # constants to be extracted from pp_config
    duration_weight_threshold = 0.2
    duration_difference = 1.5 * samples_per_hour

    if len(pairs) == 0:
        return pairs

    pairs = remove_undesired_amplitude_pairs(pairs)

    if len(pairs) == 0:
        return pairs

    pairs_info = np.zeros(shape=(len(pairs), 2), dtype=int)
    pairs_info[:, 0] = (pairs[:, 1] - pairs[:, 0]) % total_time_divisions
    pairs_info[:, 1] = (pairs[:, 3] - pairs[:, 2])

    duration_bins = np.bincount([np.int(x) for x in pairs_info[:, 0]], pairs_info[:, 1])
    duration_idx = np.where(duration_bins > 0)[0]
    duration_arr = np.multiply(duration_idx, duration_bins[duration_idx])
    duration_arr /= np.sum(duration_arr)

    weak_duration_idx = np.where(duration_arr < duration_weight_threshold)[0]
    weak_duration = duration_idx[weak_duration_idx]

    weak_duration_pairs_idx = np.where(np.isin(pairs_info[:, 0], weak_duration))[0]

    for idx in range(len(pairs)):
        pair_duration = (pairs[idx][1] - pairs[idx][0]) % total_time_divisions
        past_schedule, future_schedule = find_past_future_schedule(uncontested_matrix, pairs, idx, runs)

        if np.abs(pair_duration - past_schedule[2]) > duration_difference < np.abs(pair_duration - future_schedule[2]):
            weak_duration_pairs_idx = np.r_[weak_duration_pairs_idx, idx]

    for idx in np.unique(weak_duration_pairs_idx):
        pair = pairs[idx]

        past_schedule, future_schedule = find_past_future_schedule(uncontested_matrix, pairs, idx, runs)
        pairs[idx] = change_pair_duration(uncontested_matrix, raw_data, clean_edges, pair, past_schedule,
                                          future_schedule, total_time_divisions, pp_config)

    return pairs


def find_past_future_schedule(uncontested_matrix, pairs, idx, runs):
    """
    Utility to find past and future schedule from corresponding consistent pairs
    """

    past_matrix, past_median_distance, future_matrix, future_median_distance = find_past_future_pairs_matrix(
        uncontested_matrix, pairs, idx, runs)
    past_schedule = [0, 0, 0, 0, 0]
    future_schedule = [0, 0, 0, 0, 0]

    past_day_idx = -1
    while np.abs(past_day_idx) <= len(past_matrix):
        past_schedule = get_schedule(past_matrix[past_day_idx, :], get_section_int((pairs[idx][0], pairs[idx][1])),
                                     median_distance=past_median_distance)
        if past_schedule[2] == 0:
            past_day_idx -= 1
            continue
        break

    future_day_idx = 0
    while future_day_idx < len(future_matrix):
        future_schedule = get_schedule(future_matrix[future_day_idx, :],
                                       get_section_int((pairs[idx][0], pairs[idx][1])),
                                       median_distance=future_median_distance)
        if future_schedule[2] == 0:
            future_day_idx += 1
            continue
        break

    return past_schedule, future_schedule


def change_pair_duration(uncontested_matrix, raw_data, clean_edges, pair, past_schedule, future_schedule,
                         total_time_divisions, pp_config):
    """Truncate or extend pair duration based on past and future schedule"""

    weak_pair_length = pp_config.get('weak_pair_length')
    strong_pair_length = pp_config.get('strong_pair_length')
    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / pp_config.get('sampling_rate'))

    pair_length = pair[3] - pair[2]

    past_start_window = np.arange((past_schedule[0] - samples_per_hour),
                                  (past_schedule[0] + samples_per_hour + 1)) % total_time_divisions
    past_end_window = np.arange((past_schedule[1] - samples_per_hour),
                                (past_schedule[1] + samples_per_hour + 1)) % total_time_divisions
    future_start_window = np.arange((future_schedule[0] - samples_per_hour),
                                    (future_schedule[0] + samples_per_hour + 1)) % total_time_divisions
    future_end_window = np.arange((future_schedule[1] - samples_per_hour),
                                  (future_schedule[1] + samples_per_hour + 1)) % total_time_divisions

    if pair_length > strong_pair_length or past_schedule[2] == 0 or future_schedule[2] == 0 or (
            future_schedule[0] not in past_start_window or future_schedule[1] not in past_end_window):
        return pair

    past_schedule_filled = False
    future_schedule_filled = False
    if np.sum(uncontested_matrix[pair[2]:pair[3], past_schedule[0]]) != 0 and np.sum(
            uncontested_matrix[pair[2]:pair[3], past_schedule[1]]) != 0:
        past_schedule_filled = True
    if np.sum(uncontested_matrix[pair[2]:pair[3], future_schedule[0]]) != 0 and np.sum(
            uncontested_matrix[pair[2]:pair[3], future_schedule[1]]) != 0:
        future_schedule_filled = True

    past_signal_consistency_bool = signal_consistency_bool(raw_data, pair, past_schedule,
                                                           samples_per_hour, pp_config)
    future_signal_consistency_bool = signal_consistency_bool(raw_data, pair, past_schedule,
                                                             samples_per_hour, pp_config)

    truncation_duration = min(past_schedule[2], future_schedule[2])
    truncation_start_time_div = (pair[1] - truncation_duration) % total_time_divisions
    truncation_end_time_div = (pair[0] + truncation_duration) % total_time_divisions
    truncation_schedule_left = [pair[0], truncation_end_time_div, truncation_duration, pair[4], pair[5]]
    truncation_schedule_right = [truncation_start_time_div, pair[1], truncation_duration, pair[4], pair[5]]

    past_future_input_dict = dict()
    past_future_input_dict['past_start_window'] = past_start_window
    past_future_input_dict['past_end_window'] = past_end_window
    past_future_input_dict['future_end_window'] = future_end_window
    past_future_input_dict['future_start_window'] = future_start_window
    past_future_input_dict['past_signal_consistency_bool'] = past_signal_consistency_bool
    past_future_input_dict['future_signal_consistency_bool'] = future_signal_consistency_bool
    past_future_input_dict['past_schedule'] = past_schedule
    past_future_input_dict['future_schedule'] = future_schedule

    bool_val, modified_pair = past_future_schedule_matched_pair(pair, past_future_input_dict, clean_edges, pp_config)

    if bool_val:
        pair = modified_pair

    elif past_signal_consistency_bool and not past_schedule_filled:
        pair[0] = past_schedule[0]
        pair[1] = past_schedule[1]

    elif future_signal_consistency_bool and not future_schedule_filled:
        pair[0] = future_schedule[0]
        pair[1] = future_schedule[1]

    elif pair_length < weak_pair_length and signal_consistency_bool(raw_data, pair, truncation_schedule_left,
                                                                    samples_per_hour, pp_config):
        pair[1] = truncation_end_time_div
    elif pair_length < weak_pair_length and signal_consistency_bool(raw_data, pair, truncation_schedule_right,
                                                                    samples_per_hour, pp_config):
        pair[0] = truncation_start_time_div

    return pair


def past_future_schedule_matched_pair(pair, past_future_input_dict, clean_edges, pp_config):
    """Truncate or extend pair duration based on past and future schedule"""

    weak_pair_length = pp_config.get('weak_pair_length')
    bool_val = False

    pair_length = pair[3] - pair[2]

    past_start_window = past_future_input_dict['past_start_window']
    past_end_window = past_future_input_dict['past_end_window']
    future_end_window = past_future_input_dict['future_end_window']
    future_start_window = past_future_input_dict['future_start_window']
    past_signal_consistency_bool = past_future_input_dict['past_signal_consistency_bool']
    future_signal_consistency_bool = past_future_input_dict['future_signal_consistency_bool']
    past_schedule = past_future_input_dict['past_schedule']
    future_schedule = past_future_input_dict['future_schedule']

    if (pair[0] in past_start_window or pair[1] in past_end_window) or (
            pair[0] in future_start_window or pair[1] in future_end_window):

        if pair_length <= weak_pair_length and past_signal_consistency_bool:
            bool_val = pair_length <= weak_pair_length
            pair[0] = past_schedule[0]
            pair[1] = past_schedule[1]

        elif pair_length <= weak_pair_length and future_signal_consistency_bool:
            bool_val = pair_length <= weak_pair_length
            pair[0] = future_schedule[0]
            pair[1] = future_schedule[1]

        elif pair_length >= weak_pair_length and past_signal_consistency_bool and edge_exists_bool(
                clean_edges, past_schedule, past_start_window, past_end_window, pair):
            bool_val = pair_length >= weak_pair_length
            pair[0] = past_schedule[0]
            pair[1] = past_schedule[1]

        elif pair_length >= weak_pair_length and future_signal_consistency_bool and edge_exists_bool(
                clean_edges, future_schedule, future_start_window, future_end_window, pair):
            bool_val = pair_length >= weak_pair_length
            pair[0] = future_schedule[0]
            pair[1] = future_schedule[1]
        else:
            bool_val = True

    return bool_val, pair


def build_confidence_score(smooth_nms, pairs, pp_config, amp_low=500, amp_high=700, expected_pair_length=60,
                           longest_pair_length=100, best_two_longest_pairs_length=150, variable=False):
    """Utility to build confidence score"""

    total_days = smooth_nms.shape[0]
    pp_run_days_arr = np.zeros(total_days)
    pairs_in_smooth_edges = pairs.copy()

    samples = int(smooth_nms.shape[1]/24)

    if len(pairs) == 0:
        return 0, pp_run_days_arr

    idx = 0

    for pair in pairs:
        smooth_pos_edge = np.array([[pair[2], pair[3], pair[0], pair[4]]], dtype=int)
        smooth_neg_edge = np.array([[pair[2], pair[3], pair[1], pair[4]]], dtype=int)
        smooth_pair = np.array([0, 0, pair[2], pair[3]])

        _, smooth_start_day, smooth_end_day = pair_primary_bool(smooth_nms, smooth_pair, smooth_pos_edge,
                                                                smooth_neg_edge, pp_config, flag=False)
        pairs_in_smooth_edges[idx, 2] = smooth_start_day
        pairs_in_smooth_edges[idx, 3] = smooth_end_day

        idx += 1

        if max(pair[4], pair[5]) <= amp_low:
            # less than 500
            pp_run_days_arr[pair[2]:pair[3]] = np.where(pp_run_days_arr[pair[2]:pair[3]] > 1,
                                                        pp_run_days_arr[pair[2]:pair[3]], 1)
        elif min(pair[4], pair[5]) >= amp_high:
            # above 700
            pp_run_days_arr[pair[2]:pair[3]] = np.where(pp_run_days_arr[pair[2]:pair[3]] > 3,
                                                        pp_run_days_arr[pair[2]:pair[3]], 3)
        else:
            # between 500 and 700
            pp_run_days_arr[pair[2]:pair[3]] = np.where(pp_run_days_arr[pair[2]:pair[3]] > 2,
                                                        pp_run_days_arr[pair[2]:pair[3]], 2)

    if pp_config.get('input_data') is not None:
        input_data = np.abs(pp_config.get('input_data') - np.roll(pp_config.get('input_data'), 1, axis=1))
    else:
        return np.round(0.5, 3), pp_run_days_arr

    mean_val = np.mean(pairs[:, 4:])

    threshold = [800, 1000, 1200, 1400, 1600, 2000][np.digitize(mean_val*samples, [2500, 3000, 4000, 5000, 6000])]

    threshold = threshold/samples

    pairs_copy = copy.deepcopy(pairs)
    pairs = pairs_in_smooth_edges

    cv = calculate_pp_conf_for_hybrid(pairs, input_data, pp_config, samples, threshold, pp_run_days_arr)

    pp_config["hybrid_conf_val"] = cv

    total_pp_run_days = np.count_nonzero(pp_run_days_arr)

    # Score on Amplitude
    days_amp_score = 0

    if not variable:
        different_amp_days = np.array([len(np.where(pp_run_days_arr == 1)[0]), len(np.where(pp_run_days_arr == 2)[0]),
                                       len(np.where(pp_run_days_arr == 3)[0])], dtype=int)
        different_amp_weights = np.array([0.1, 0.5, 1])

        days_amp_score = np.sum(different_amp_weights * different_amp_days) / total_pp_run_days

    # Score on num_of_pp_run_days
    pp_run_days_score = total_pp_run_days / total_days

    # Score on avg_length_of_pairs

    pairs = pairs_copy
    avg_length_of_pairs = np.mean(pairs[:, 3] - pairs[:, 2])
    avg_length_of_pairs_score = min(1, avg_length_of_pairs / expected_pair_length)

    # Score on best_two_pairs_in_smooth_nms
    smooth_pair_length_diff_arr = np.sort(pairs_in_smooth_edges[:, 3] - pairs_in_smooth_edges[:, 2])
    smooth_pair_length_sum = np.sum(smooth_pair_length_diff_arr[-1:])
    smooth_pair_length_score = min(1, smooth_pair_length_sum / longest_pair_length)
    if len(smooth_pair_length_diff_arr) > 1:
        smooth_pair_length_sum = np.sum(smooth_pair_length_diff_arr[-2:])
        smooth_pair_length_score = min(1, smooth_pair_length_sum / best_two_longest_pairs_length)

    confidence_score_weights = np.array([0, 0.5, 0.1, 0.4])

    if not variable:
        confidence_score_weights = np.array([0.2, 0.3, 0.2, 0.3])

    confidence_score_parameters = np.array(
        [days_amp_score, pp_run_days_score, avg_length_of_pairs_score, smooth_pair_length_score])

    confidence_score = np.sum(confidence_score_weights * confidence_score_parameters)

    return np.round(confidence_score, 3), pp_run_days_arr
