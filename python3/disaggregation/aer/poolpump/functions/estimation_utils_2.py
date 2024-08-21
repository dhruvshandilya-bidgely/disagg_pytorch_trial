"""
Author - Arpan Agrawal
Date - 06/04/2019
Utility functions to help compute the estimate
"""

# Import python packages

import numpy as np
import pandas as pd

# Import functions from within the project

from python3.disaggregation.aer.poolpump.functions.cleaning_utils import find_edges

from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_days_labeled


def get_amp_ballpark_from_uncontested_matrix(uncontested_matrix, pp_config):
    """Utility to get rough estimate of amplitude"""

    # Extract constants out of config

    min_pp_amp = 0
    max_pp_amp = 3000
    amp_margin = 0.3
    filled_days_threshold = 0.2

    # If empty return default values

    filled_days_in_uncontested = np.count_nonzero(np.count_nonzero(uncontested_matrix, axis=1))
    filled_days_fraction = filled_days_in_uncontested / len(uncontested_matrix)

    if not np.any(uncontested_matrix) or filled_days_fraction < filled_days_threshold:
        return min_pp_amp, max_pp_amp, 0, False

    # If not empty compute min max and scale it to get amplitude ballpark

    uncontested_matrix_abs = np.abs(uncontested_matrix)

    non_zero_idx = uncontested_matrix_abs != 0

    min_amp = np.min(uncontested_matrix_abs[non_zero_idx])
    max_amp = np.max(uncontested_matrix_abs[non_zero_idx])

    min_amp *= (1 - amp_margin)
    max_amp *= (1 + amp_margin)

    non_zero_values = uncontested_matrix[uncontested_matrix != 0]
    median_amp = np.median(np.abs(non_zero_values))

    return min_amp, max_amp, median_amp, True


def remove_pairs_by_amp(pairs, pos_edges, neg_edges, amp_ballpark):
    """Remove pairs that violate the amplitude ballpark"""

    pos_edges_in_pairs = pos_edges[pairs[:, 0], :]
    neg_edges_in_pairs = neg_edges[pairs[:, 1], :]

    pair_min_amp_arr = np.minimum(pos_edges_in_pairs[:, 3], neg_edges_in_pairs[:, 3])
    pair_max_amp_arr = np.maximum(pos_edges_in_pairs[:, 3], neg_edges_in_pairs[:, 3])

    selected_pair_idx = np.logical_and(pair_min_amp_arr >= amp_ballpark[0], pair_max_amp_arr <= amp_ballpark[1])

    pairs_selected = pairs[selected_pair_idx, :]

    return pairs_selected.astype(int)


def remove_overlapping_pairs(pairs, pos_edges, filled_previous_iterations, total_time_divisions, window):
    """Utility to remove overlapping pairs from previous iterations"""

    rows = filled_previous_iterations.shape[0]
    pairs_selected = list()

    # For each pair check if it violates if not add it to non overlapping pairs

    for pair in pairs:

        pair_block_matrix = np.zeros_like(filled_previous_iterations)
        time_div_to_fill = np.arange(pos_edges[pair[0]][2],
                                     (pos_edges[pair[0]][2] + pair[5] + 1)) % total_time_divisions

        pair_block_matrix[max(0, pair[2] - window):min(rows, pair[3] - window), time_div_to_fill] = 1

        non_overlap_area = pair_block_matrix * (1 - filled_previous_iterations)
        overlap_days = np.where(np.sum(non_overlap_area, axis=1) != (pair[5] + 1))[0]

        if len(overlap_days) == rows:
            continue

        non_overlap_area[overlap_days, :] = 0

        start_arr, end_arr = find_edges(non_overlap_area[:, pos_edges[pair[0]][2]])

        for idx in range(len(start_arr)):
            new_pair = pair.copy()
            new_pair[2] = window + start_arr[idx]
            new_pair[3] = window + end_arr[idx]
            new_pair[4] = new_pair[3] - new_pair[2]
            pairs_selected.append(new_pair)

    return np.array(pairs_selected, dtype=int)


def get_valid_days_in_smooth(pos_data, pos_shift, neg_data, neg_shift):
    """Utility to get valid days in smooth data"""

    pos_data_copy = pos_data.copy()
    pos_data_copy[pos_data_copy > 0] = 1
    neg_data_copy = neg_data.copy()
    neg_data_copy[neg_data_copy < 0] = 1

    pos_argmax_start_idx = np.argmax(pos_data_copy, axis=0)
    pos_argmax_start_idx_values = pos_data_copy[pos_argmax_start_idx, np.arange(pos_data.shape[1])]
    pos_argmax_end_idx = pos_data.shape[0] - 1 - np.argmax(pos_data_copy[::-1], axis=0)
    pos_argmax_end_idx_values = pos_data_copy[pos_argmax_end_idx, np.arange(pos_data.shape[1])]
    pos_start_idx = pos_argmax_start_idx[pos_argmax_start_idx_values != 0]
    pos_end_idx = pos_argmax_end_idx[pos_argmax_end_idx_values != 0]

    neg_argmax_start_idx = np.argmax(neg_data_copy, axis=0)
    neg_argmax_start_idx_values = neg_data_copy[neg_argmax_start_idx, np.arange(neg_data.shape[1])]
    neg_argmax_end_idx = neg_data.shape[0] - 1 - np.argmax(neg_data_copy[::-1], axis=0)
    neg_argmax_end_idx_values = neg_data_copy[neg_argmax_end_idx, np.arange(neg_data.shape[1])]
    neg_start_idx = neg_argmax_start_idx[neg_argmax_start_idx_values != 0]
    neg_end_idx = neg_argmax_end_idx[neg_argmax_end_idx_values != 0]

    if min(len(pos_start_idx), len(pos_end_idx)) == 0 or min(len(neg_start_idx), len(neg_end_idx)) == 0:
        return 0, 0

    smooth_pos_start_day = np.min(pos_start_idx) + pos_shift
    smooth_pos_end_day = np.max(pos_end_idx) + pos_shift

    smooth_neg_start_day = np.min(neg_start_idx) + neg_shift
    smooth_neg_end_day = np.max(neg_end_idx) + neg_shift

    smooth_start_day, smooth_end_day = max(smooth_pos_start_day, smooth_neg_start_day), min(smooth_pos_end_day,
                                                                                            smooth_neg_end_day)

    return smooth_start_day, smooth_end_day


def remove_small_uncontested_pairs(days_label, minimum_uncontested_pair_length=7):
    """Utility to remove small uncontested pairs"""

    uncontested_labeled_days = days_label.copy()
    uncontested_labeled_days[uncontested_labeled_days != 1] = 0

    start_arr, end_arr = find_edges(uncontested_labeled_days)
    diff = end_arr - start_arr

    idx = np.where(diff < minimum_uncontested_pair_length)[0]

    if len(days_label) <= minimum_uncontested_pair_length:
        idx = np.array([])

    uncontested_labeled_days = np.where(days_label == 2, days_label, uncontested_labeled_days)

    for i in idx:
        if (start_arr[i] > 0 and days_label[start_arr[i] - 1] == 2) or (
                end_arr[i] < len(days_label) and days_label[end_arr[i]] == 2):
            uncontested_labeled_days[start_arr[i]:end_arr[i]] = 2
        else:
            uncontested_labeled_days[start_arr[i]:end_arr[i]] = 0

    return uncontested_labeled_days


def get_valid_uncontested_idx(uncontested_idx, min_length_for_validity=5):
    """Get Valid uncontested index"""

    uncontested_days = len(uncontested_idx)
    diff_arr = uncontested_idx[1:] - uncontested_idx[:-1]
    bool_arr = np.ones(uncontested_days, dtype=bool)

    diff_arr[diff_arr != 1] = 0
    start_arr, end_arr = find_edges(diff_arr)

    for idx in range(len(start_arr)):
        if (end_arr[idx] - start_arr[idx] + 1) >= min_length_for_validity:
            continue
        bool_arr[start_arr[idx]:(end_arr[idx] + 1)] = False

    if len(uncontested_idx) == 1:
        bool_arr[:] = False

    return uncontested_idx[bool_arr]


def get_schedule_arr(schedule_start, schedule_end, total_time_divisions):
    """Utility to get schedule array"""

    schedule_arr = np.zeros(total_time_divisions)

    if schedule_start < schedule_end:

        schedule_arr[schedule_start:schedule_end + 1] = 1
    else:

        schedule_arr[schedule_start:] = 1
        schedule_arr[:(schedule_end + 1)] = 1

    return schedule_arr


def get_all_possible_schedules(arr, time_div_arr):
    """Utility to get a list of all possible schedules"""

    schedules = list()

    idx_pos_one = np.where(arr == 1)[0]
    idx_neg_one = np.where(arr == -1)[0]

    for idx_pos in idx_pos_one:

        neg_idx_valid = idx_neg_one[idx_neg_one >= idx_pos]

        for idx_neg in neg_idx_valid:
            schedules.append([time_div_arr[idx_pos], time_div_arr[idx_neg]])

    return schedules


def get_section_int(section):
    """Utility to get integer of section"""

    if int(section[0] / 10) == 0:
        section_start = "0" + str(section[0])
    else:
        section_start = str(section[0])

    if int(section[1] / 10) == 0:
        section_end = "0" + str(section[1])
    else:
        section_end = str(section[1])

    return int(section_start + section_end)


def find_run_filled_days(df, common_runs, common_flag=False):
    """Utility to find run filled days"""

    df_copy = df.copy()
    df_copy[df_copy > 0] = 1
    df_copy[df_copy < 0] = 0
    runs_each_day = np.sum(df_copy, axis=1)

    if not common_flag:
        run_filled_days = np.where(runs_each_day > common_runs)[0]
    else:
        run_filled_days = np.where(runs_each_day == common_runs)[0]

    return run_filled_days


def combine_overlapping_pairs(pairs_matrix, grouped_schedules):
    """Utility to combine overlapping pairs"""

    pairs_list = list()
    for schedule in grouped_schedules:
        pair = np.zeros(pairs_matrix.shape[1])

        pairs_in_schedule = np.where((pairs_matrix[:, 2] == schedule[0]) & (pairs_matrix[:, 7] == schedule[1]))[0]
        pairs = pairs_matrix[pairs_in_schedule]

        if len(pairs) == 0:
            continue

        pair[0] = np.min(pairs[:, 0])
        pair[1] = np.max(pairs[:, 1])
        pair[2] = schedule[0]
        pair[3] = np.median(pairs[:, 3])
        pair[4] = np.max(pairs[:, 4])
        pair[5] = np.min(pairs[:, 5])
        pair[6] = np.max(pairs[:, 6])
        pair[7] = schedule[1]
        pair[8] = np.median(pairs[:, 8])
        pair[9] = np.max(pairs[:, 9])
        pair[10] = np.mean(pairs[:, 10])
        pairs_list.append(pair)

    return np.array(pairs_list, dtype=int)


def get_global_pairs_matrix(global_pos_edges, global_neg_edges, global_pairs):
    """Utility to get global pairs matrix"""

    global_pairs_matrix = list()
    for i in range(len(global_pairs)):
        pos_edge_start_day = global_pos_edges[global_pairs[i][0]][0]
        pos_edge_end_day = global_pos_edges[global_pairs[i][0]][1]

        pos_edge_time_div = global_pos_edges[global_pairs[i][0]][2]
        pos_edge_amp = global_pos_edges[global_pairs[i][0]][3]

        pos_edge_time_mask = global_pos_edges[global_pairs[i][0]][4]

        neg_edge_start_day = global_neg_edges[global_pairs[i][1]][0]
        neg_edge_end_day = global_neg_edges[global_pairs[i][1]][1]

        neg_edge_time_div = global_neg_edges[global_pairs[i][1]][2]
        neg_edge_amp = global_neg_edges[global_pairs[i][1]][3]

        neg_edge_time_mask = global_neg_edges[global_pairs[i][1]][4]

        signal_score = global_pairs[i][-1]

        global_pairs_matrix.append(
            [pos_edge_start_day, pos_edge_end_day, pos_edge_time_div, pos_edge_amp, pos_edge_time_mask,
             neg_edge_start_day, neg_edge_end_day, neg_edge_time_div, neg_edge_amp, neg_edge_time_mask,
             signal_score])

    return np.array(global_pairs_matrix, dtype=int)


def retain_time_mask_score(all_pos_edges, all_neg_edges, total_days, total_time_divisions):
    """Utility to retain time mask score"""

    pos_time_mask_data = np.zeros((total_days, total_time_divisions))
    neg_time_mask_data = np.zeros((total_days, total_time_divisions))

    for edge in all_pos_edges:
        pos_time_mask_data[edge[0]:edge[1], edge[2]] = edge[4]

    for edge in all_neg_edges:
        neg_time_mask_data[edge[0]:edge[1], edge[2]] = edge[4]

    return pos_time_mask_data, neg_time_mask_data


def restore_time_mask_score(pos_time_mask_data, neg_time_mask_data, all_pos_edges, all_neg_edges):
    """Utility to restore time mask score"""

    for edge in all_pos_edges:
        edge[4] = int(np.sum(pos_time_mask_data[edge[0]:edge[1], edge[2]]) / (edge[1] - edge[0]))

    for edge in all_neg_edges:
        edge[4] = int(np.sum(neg_time_mask_data[edge[0]:edge[1], edge[2]]) / (edge[1] - edge[0]))

    return all_pos_edges, all_neg_edges


def get_section_time_divisions(section_new, all_pos_edges, all_neg_edges, total_time_divisions):
    """Utility to get section time divisions"""

    arr = np.zeros(total_time_divisions)

    for pair in section_new:
        if all_pos_edges[pair[0]][2] < all_neg_edges[pair[1]][2]:
            arr[all_pos_edges[pair[0]][2]:all_neg_edges[pair[1]][2] + 1] = 1
        else:
            arr[all_pos_edges[pair[0]][2]:] = 1
            arr[:all_neg_edges[pair[1]][2] + 1] = 1

    start_arr, end_arr = find_edges(arr)

    pos_time_div = start_arr[0]
    neg_time_div = end_arr[0] - 1
    if len(start_arr) > 1:
        pos_time_div = start_arr[-1]

    return pos_time_div, neg_time_div


def found_overlapping(section_boundaries, pair_two, all_pos_edges, all_neg_edges, total_time_divisions):
    """Utility to decide if the sections are overlapping or ot"""

    pair_two_start_time_div = all_pos_edges[pair_two[0]][2]
    pair_two_end_time_div = all_neg_edges[pair_two[1]][2]

    section_arr = np.zeros(total_time_divisions)
    pair_two_arr = np.zeros(total_time_divisions)

    if section_boundaries[0] < section_boundaries[1]:
        section_arr[section_boundaries[0]:section_boundaries[1] + 1] = 1
    else:
        section_arr[section_boundaries[0]:] = 1
        section_arr[:section_boundaries[1] + 1] = 1

    if pair_two_start_time_div < pair_two_end_time_div:
        pair_two_arr[pair_two_start_time_div:pair_two_end_time_div + 1] = 1
    else:
        pair_two_arr[pair_two_start_time_div:] = 1
        pair_two_arr[:pair_two_end_time_div + 1] = 1

    if np.sum(section_arr * pair_two_arr) > 0:
        return True

    return False


def fill_uniform_amplitude(matrix):
    """Utility to fill uniform amplitude in uncontested matrix"""

    pos_mat = matrix.copy()
    pos_mat[pos_mat < 0] = 0
    neg_mat = matrix.copy()
    neg_mat[neg_mat > 0] = 0

    pos_time_divs = np.unique(np.where(pos_mat != 0)[1])
    neg_time_divs = np.unique(np.where(neg_mat != 0)[1])

    pos_sign_mat = np.sign(pos_mat)
    neg_sign_mat = np.sign(np.abs(neg_mat))

    for pos_time_div in pos_time_divs:
        start_arr, end_arr = find_edges(pos_sign_mat[:, pos_time_div])
        for idx in range(len(start_arr)):
            pos_mat[start_arr[idx]:end_arr[idx], pos_time_div] = np.median(
                pos_mat[start_arr[idx]:end_arr[idx], pos_time_div])

    for neg_time_div in neg_time_divs:
        start_arr, end_arr = find_edges(neg_sign_mat[:, neg_time_div])
        for idx in range(len(start_arr)):
            neg_mat[start_arr[idx]:end_arr[idx], neg_time_div] = -np.median(
                np.abs(neg_mat[start_arr[idx]:end_arr[idx], neg_time_div]))

    res_mat = pos_mat + neg_mat

    return res_mat


def change_day_arr(day_arr):
    """Changes day_arr in case of wrongly filled multiple run. Written for crash handling"""

    runs_idx = np.where(day_arr[:-1] < day_arr[1:])[0]
    runs_idx = np.r_[0, runs_idx, len(day_arr) - 1]
    inter_idx = runs_idx[1:-1] + 1
    runs_idx = np.sort(np.r_[runs_idx, inter_idx])
    all_runs_idx = np.split(runs_idx, len(runs_idx)/2)
    new_day_arr_idx = np.array([])
    for run_idx in all_runs_idx:
        if np.sum(day_arr[run_idx[0]:run_idx[1] + 1]) == 0:
            new_day_arr_idx = np.r_[new_day_arr_idx, np.arange(run_idx[0], run_idx[1] + 1)]
        else:
            new_day_arr_idx = np.r_[new_day_arr_idx, run_idx[0], run_idx[1]]

    return new_day_arr_idx


def find_pairs(matrix, pp_config):
    """Utility to find pairs"""

    max_gap_allowed = 2

    sign_matrix = np.sign(matrix)
    x = np.vstack((np.zeros(shape=matrix.shape[1]), sign_matrix, np.zeros(shape=matrix.shape[1])))

    y = np.apply_along_axis(np.diff, 0, x)
    diff_sum = np.sum(np.abs(y), 1)
    day_bounds = np.where(diff_sum > 0)[0]
    sections = [(i, j) for i, j in zip(day_bounds[:-1], day_bounds[1:])]

    pairs = np.empty(shape=(1, 6))

    for section in sections:
        df_day = sign_matrix[section[0], :]
        day_arr = df_day[df_day != 0]
        day_td = np.where(df_day != 0)[0]

        if len(day_arr) == 0:
            continue

        day_arr, day_td = roll_array(day_arr, day_td)

        if np.sum(np.abs(day_arr[1:] + day_arr[:-1])) != 0:
            new_day_arr_idx = change_day_arr(day_arr)
            day_td = day_td[new_day_arr_idx.astype(int)]

        day_amp = np.abs(matrix[section[0], day_td])

        all_schedules = np.vstack(np.split(day_td, (len(day_td) / 2)))
        all_days = np.tile(np.array([section[0], section[1]]), (len(all_schedules), 1))
        all_amps = np.vstack(np.split(day_amp, (len(day_td) / 2)))
        current_pair = np.hstack((all_schedules, all_days, all_amps))
        pairs = np.vstack((pairs, current_pair))

    all_section_pairs = np.array(pairs[1:, :], dtype=int)

    all_section_pairs_df = pd.DataFrame(all_section_pairs)
    schedule_pair_groups = all_section_pairs_df.groupby([0, 1]).groups

    final_pairs = np.empty(shape=(1, 6))

    for key in schedule_pair_groups.keys():

        pairs = all_section_pairs[schedule_pair_groups[key].values]

        if len(pairs) == 1:
            final_pairs = np.vstack((final_pairs, pairs))
            continue

        pairs = pairs[pairs[:, 2].argsort()]

        diff_arr = pairs[1:, 2] - pairs[:-1, 3]
        diff_arr[diff_arr < max_gap_allowed] = 0
        diff_arr = np.r_[1, diff_arr, 1]

        discontinuous_idx = np.where(diff_arr != 0)[0]

        for idx in range(len(discontinuous_idx) - 1):
            pair = [key[0], key[1], pairs[discontinuous_idx[idx]][2], pairs[discontinuous_idx[idx + 1] - 1][3],
                    pairs[discontinuous_idx[idx]][4], pairs[discontinuous_idx[idx + 1] - 1][5]]

            final_pairs = np.vstack((final_pairs, np.array(pair)))

    return np.array(final_pairs[1:], dtype=int)


def remove_undesired_amplitude_pairs(pairs, min_pp_amp=400, max_pp_amp=4000, amp_margin=0.4, amp_relaxation_factor=0.9):
    """Utility to remove undesired amplitude pairs"""

    all_pos_amps = np.repeat(pairs[:, 4], (pairs[:, 3] - pairs[:, 2]))
    all_neg_amps = np.repeat(pairs[:, 5], (pairs[:, 3] - pairs[:, 2]))
    median_amp = amp_relaxation_factor * np.median(np.r_[all_pos_amps, all_neg_amps])
    lower_threshold = max(min_pp_amp, (1 - amp_margin) * median_amp)

    within_amp_pairs_idx = np.unique(np.where((pairs[:, 4:] >= lower_threshold) & (pairs[:, 4:] < max_pp_amp))[0])

    return pairs[within_amp_pairs_idx]


def find_past_future_pairs_matrix(uncontested_matrix, pairs, idx, runs):
    """Find past and future matrices of consistent pair length """

    # constants
    consistent_pair_length = 10
    near_days = 30
    num_rows, num_cols = uncontested_matrix.shape

    past_median_distance = num_cols
    future_median_distance = num_cols

    pairs_in_past_idx = np.where((pairs[:, 2] < pairs[idx][2]) & (pairs[:, 3] - pairs[:, 2] >= consistent_pair_length))[
        0]
    past_pairs = pairs[pairs_in_past_idx]
    past_matrix = np.array([])
    if len(past_pairs) != 0:
        sorted_past_pairs = past_pairs[past_pairs[:, 3].argsort()]
        near_past_start_day = max(min(pairs[idx][2] - 1, sorted_past_pairs[-1][3]) - near_days, 0)
        near_past_pairs_idx = np.where((past_pairs[:, 3] >= near_past_start_day) & (past_pairs[:, 2] < pairs[idx][2]))[
            0]
        near_past_pairs = past_pairs[near_past_pairs_idx]
        past_matrix = np.zeros_like(uncontested_matrix)
        for pair in near_past_pairs:
            past_matrix[pair[2]:pair[3], pair[0]] = pair[4]
            past_matrix[pair[2]:pair[3], pair[1]] = -pair[5]
        past_matrix = past_matrix[near_past_start_day:pairs[idx][2], :]

    pairs_in_future_idx = \
        np.where((pairs[:, 3] >= pairs[idx][3]) & (pairs[:, 3] - pairs[:, 2] >= consistent_pair_length))[0]
    future_pairs = pairs[pairs_in_future_idx]
    future_matrix = np.array([])
    if len(future_pairs) != 0:
        sorted_future_pairs = future_pairs[future_pairs[:, 2].argsort()]
        near_future_end_day = min(max(pairs[idx][3], sorted_future_pairs[0][2]) + near_days, num_rows)
        near_future_pairs_idx = \
            np.where((future_pairs[:, 3] >= pairs[idx][3]) & (future_pairs[:, 2] < near_future_end_day))[0]
        near_future_pairs = future_pairs[near_future_pairs_idx]
        future_matrix = np.zeros_like(uncontested_matrix)
        for pair in near_future_pairs:
            future_matrix[pair[2]:pair[3], pair[0]] = pair[4]
            future_matrix[pair[2]:pair[3], pair[1]] = -pair[5]
        future_matrix = future_matrix[pairs[idx][3]: near_future_end_day, :]
    if runs == 1:
        return past_matrix, past_median_distance, future_matrix, future_median_distance

    if len(past_matrix) > 0:
        past_sign = np.sign(past_matrix)
        past_sign[past_sign < 0] = 0
        runs_each_day = np.sum(past_sign, axis=1)
        past_days_with_runs = np.where(runs_each_day >= runs)[0]
        past_schedules_distance = list()
        for day in past_days_with_runs:
            data_col_sign = np.sign(past_matrix[day, :])
            day_arr = data_col_sign[data_col_sign != 0]
            day_td = np.where(data_col_sign != 0)[0]

            day_arr, day_td = roll_array(day_arr, day_td)

            odd_idx = np.arange(1, len(day_arr), 2)
            even_idx = np.arange(0, len(day_arr), 2)
            past_schedules_distance = schedule_diff(odd_idx, even_idx, day_td, num_cols, past_schedules_distance)
        if len(past_schedules_distance) > 0:
            past_median_distance = np.median(np.array(past_schedules_distance))

    if len(future_matrix) > 0:
        future_sign = np.sign(future_matrix)
        future_sign[future_sign < 0] = 0
        runs_each_day = np.sum(future_sign, axis=1)
        future_days_with_runs = np.where(runs_each_day >= runs)[0]
        future_schedules_distance = list()
        for day in future_days_with_runs:
            data_col_sign = np.sign(future_matrix[day, :])
            day_arr = data_col_sign[data_col_sign != 0]
            day_td = np.where(data_col_sign != 0)[0]

            day_arr, day_td = roll_array(day_arr, day_td)

            odd_idx = np.arange(1, len(day_arr), 2)
            even_idx = np.arange(0, len(day_arr), 2)
            future_schedules_distance = schedule_diff(odd_idx, even_idx, day_td, num_cols, future_schedules_distance)
        if len(future_schedules_distance) > 0:
            future_median_distance = np.median(np.array(future_schedules_distance))

    return past_matrix, past_median_distance, future_matrix, future_median_distance


def schedule_diff(odd_idx, even_idx, day_td, num_cols, n_schedules_distance):
    """
    Function used to get the scheduled difference
    Parameters:
        odd_idx                 (np.array)       : Odd index
        even_idx                (np.array)       : Even index
        day_td                  (np.array)       : Day numbers
        num_cols                (int)            : Number of columns (time)
        n_schedules_distance    (list)           : list of distances between the runs
    Returns:
        n_schedules_distance    (list)           : list of distances between the runs
    """
    if len(odd_idx) == len(even_idx):
        time_diff = (day_td[odd_idx] - day_td[even_idx]) % num_cols
        start_td = day_td[even_idx]
        schedule_mid_pt_arr = (start_td + (time_diff / 2)) % num_cols
        schedule_diff = np.diff(schedule_mid_pt_arr) % num_cols
        n_schedules_distance.extend(list(schedule_diff))
    return n_schedules_distance


def roll_array(day_arr, day_td):
    """Rolls array by required number of positions"""

    if len(day_arr) != 0 and day_arr[0] < 0:

        roll_idx = np.argmax(day_arr)
        day_arr = np.roll(day_arr, -roll_idx)
        day_td = np.roll(day_td, -roll_idx)

    elif len(day_arr) != 0 and day_arr[-1] > 0:

        roll_idx = np.argmin(day_arr[::-1])
        day_arr = np.roll(day_arr, roll_idx)
        day_td = np.roll(day_td, roll_idx)

    return day_arr, day_td


def edge_exists_bool(data_clean_edges, schedule, start_window, end_window, pair):
    """Utility for post processing of pairs and checking presence of edge in data_clean_edges"""

    # constants
    edge_presence_fraction = 0.5

    total_time_divisions = data_clean_edges.shape[1]
    if pair[0] in start_window:
        time_div_range = np.arange(schedule[1] - 1, schedule[1] + 2) % total_time_divisions
        neg_edge_mat = data_clean_edges[pair[2]:pair[3], time_div_range]
        neg_edge_mat[neg_edge_mat > 0] = 0
        neg_edge_mat[neg_edge_mat < 0] = 1
        days_filled = np.count_nonzero(np.sum(neg_edge_mat, axis=1))
        if days_filled / (pair[3] - pair[2]) >= edge_presence_fraction:
            return True

    elif pair[1] in end_window:
        time_div_range = np.arange(schedule[0] - 1, schedule[0] + 2) % total_time_divisions
        pos_edge_mat = data_clean_edges[pair[2]:pair[3], time_div_range]
        pos_edge_mat[pos_edge_mat > 0] = 1
        pos_edge_mat[pos_edge_mat < 0] = 0
        days_filled = np.count_nonzero(np.sum(pos_edge_mat, axis=1))
        if days_filled / (pair[3] - pair[2]) >= edge_presence_fraction:
            return True
    return False


def reject_poolpump(day_seasons, pp_run_days_arr, min_summer_intr_days=40, ratio_threshold=0.5, min_summer_days=30):
    """Utility to decide if we are going to reject pool pump"""

    if np.sum(pp_run_days_arr) == 0:
        return False
    pp_run_days_idx = np.nonzero(pp_run_days_arr)[0]
    pp_region_start, pp_region_end = pp_run_days_idx[0], (pp_run_days_idx[-1] + 1)

    if len(day_seasons) < len(pp_run_days_arr):
        day_seasons = np.r_[day_seasons,
                            np.full(shape=(len(pp_run_days_arr) - len(day_seasons),), fill_value=day_seasons[-1])]
    elif len(day_seasons) > len(pp_run_days_arr):
        day_seasons = day_seasons[:len(pp_run_days_arr)]

    # Metrics from PP Area
    num_winter_days = max(1, len(np.where(day_seasons[pp_region_start:pp_region_end] == 1)[0]))
    num_summer_intr_days = max(1, len(np.where(day_seasons[pp_region_start:pp_region_end] != 1)[0]))
    num_summer_days = max(1, len(np.where(day_seasons[pp_region_start:pp_region_end] == 3)[0]))
    summer_intr_days_ratio = float('nan')
    winter_days_ratio = \
        len(np.where((pp_run_days_arr[pp_region_start:pp_region_end] != 0) &
                     (day_seasons[pp_region_start:pp_region_end] == 1))[0]) / num_winter_days

    if num_summer_intr_days >= min_summer_intr_days:
        summer_intr_days_ratio = \
            len(np.where((pp_run_days_arr[pp_region_start:pp_region_end] != 0) &
                         (day_seasons[pp_region_start:pp_region_end] != 1))[0]) / num_summer_intr_days

    # Metrics from Area above PP
    summer_intr_above_pp = False
    if len(np.where(day_seasons[:pp_region_start] != 1)[0]) > min_summer_intr_days:
        summer_intr_above_pp = True

    # Metrics from Area below PP
    summer_intr_below_pp = False
    if len(np.where(day_seasons[pp_region_end:] != 1)[0]) > min_summer_intr_days:
        summer_intr_below_pp = True

    bool_val = True
    cond_1 = summer_intr_days_ratio >= ratio_threshold or num_summer_days > min_summer_days or \
             winter_days_ratio < ratio_threshold
    cond_2 = np.isnan(summer_intr_days_ratio) and winter_days_ratio >= ratio_threshold and \
             not summer_intr_below_pp and summer_intr_above_pp

    if cond_1 or cond_2:
        bool_val = False

    return bool_val


def create_structure_for_hsm(processed_pairs, cons_threshold, samples_per_hr):
    """Utility creates a 21 column structure to return for hsm"""

    structures = np.empty(shape=(0, 10))

    for pair in processed_pairs:
        structures = np.vstack((structures, [pair[2], pair[3], pair[0], -1, -1, pair[1], pair[4], 0, 0, pair[5]]))

    # Modify structures to add consumption

    c1 = structures[:, 6]
    c4 = structures[:, 9]

    cons_cols = np.zeros(shape=(structures.shape[0], 3))
    cons_threshold *= samples_per_hr

    # Fill values case by case

    # Case 1 both in the middle are zero

    cons_cols[:, 1] = np.fmax((c1 + c4) / 2, cons_threshold)

    structures = np.c_[structures, cons_cols]

    return structures


def pair_primary_bool(data_nms, pair, pos_edges, neg_edges, pp_config, minimum_primary_length=30, flag=True):
    """Utility to compute primary values for a given pair"""

    # Extract constants from config

    min_time_mask_score = 1

    smooth_nms_copy = data_nms.copy()
    n = smooth_nms_copy.shape[1]
    pos_time_div = pos_edges[pair[0]][2]
    neg_time_div = neg_edges[pair[1]][2]

    neg_start, neg_end = max((neg_time_div - 1), 0), min((neg_time_div + 2), n)
    pos_start, pos_end = max((pos_time_div - 1), 0), min((pos_time_div + 2), n)

    pos_data = smooth_nms_copy[pos_edges[pair[0]][0]:pos_edges[pair[0]][1], pos_start:pos_end]
    pos_data[pos_data < 0] = 0
    neg_data = smooth_nms_copy[neg_edges[pair[1]][0]:neg_edges[pair[1]][1], neg_start:neg_end]
    neg_data[neg_data > 0] = 0

    pos_count = np.count_nonzero(np.count_nonzero(pos_data, axis=1))
    neg_count = np.count_nonzero(np.count_nonzero(neg_data, axis=1))

    bool_val = True

    if flag and (pair[6] < min_time_mask_score or pair[7] < min_time_mask_score) and (
            pos_count < minimum_primary_length or neg_count < minimum_primary_length):
        return False, 0, 0

    smooth_start_day, smooth_end_day = get_valid_days_in_smooth(pos_data, pos_edges[pair[0]][0], neg_data,
                                                                neg_edges[pair[1]][0])

    # check signal consistency here

    if smooth_start_day == 0 and smooth_end_day == 0:
        return False, 0, 0

    return bool_val, max(smooth_start_day, pair[2]), min(smooth_end_day, pair[3])


def check_pair_prim(input_dict, data_nms, days_label, duration_each_day, num_of_runs_each_day, filled_days,
                    filled_previous_iterations, pp_config):
    """Utility to check pairs in primary"""

    window_size = pp_config.get('window_size')
    uncontested_matrix = input_dict['uncontested_matrix']
    all_pos_edges = input_dict['all_pos_edges']
    all_neg_edges = input_dict['all_neg_edges']
    all_pairs = input_dict['all_pairs']
    window = input_dict['window']
    amp_ballpark = input_dict['amp_ballpark']

    total_days, total_time_divisions = data_nms.shape
    # Initialise start and end index arrays to run on

    multiple_days_label_copy = days_label[window:min(total_days, min(total_days, (window + window_size)))].copy()
    multiple_days_label_copy[multiple_days_label_copy != 2] = 0
    multiple_start_arr, multiple_end_arr = find_edges(multiple_days_label_copy)

    # For all multiple run possibilities on the same day run the following

    for idx in range(len(multiple_start_arr)):

        pairs_idx = np.where((all_pairs[:, 2] <= (window + multiple_end_arr[idx])) &
                             (all_pairs[:, 3] >= (window + multiple_start_arr[idx])))[0]
        pairs = all_pairs[pairs_idx]

        # Remove overlapping pairs from prev iterations

        if np.sum(filled_previous_iterations) > 0 and len(pairs) > 0:
            pairs = remove_overlapping_pairs(pairs, all_pos_edges, filled_previous_iterations, total_time_divisions,
                                             window)

        # Remove pairs violating rough amplitude

        if len(pairs) > 0:
            pairs = remove_pairs_by_amp(pairs, all_pos_edges, all_neg_edges, amp_ballpark)

        if len(pairs) == 0:
            return uncontested_matrix

        multiple_pair_window_data = np.zeros(((multiple_end_arr[idx] - multiple_start_arr[idx]), total_time_divisions))
        multiple_pair_duration_each_day = duration_each_day[(window + multiple_start_arr[idx]):
                                                            (window + multiple_end_arr[idx])]
        multiple_pair_days_label = days_label[(window + multiple_start_arr[idx]):(window + multiple_end_arr[idx])]

        # For each pair extract important values

        for pair in pairs:

            bool_val, pair_smooth_start, pair_smooth_end = pair_primary_bool(data_nms, pair, all_pos_edges,
                                                                             all_neg_edges, pp_config)

            if bool_val and pair_smooth_end > (window + multiple_start_arr[idx]):
                start_day = max((pair_smooth_start - window - multiple_start_arr[idx]), 0)
                end_day = min((pair_smooth_end - window - multiple_start_arr[idx]), len(multiple_pair_window_data))
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


def get_schedule(data_col, section=0, median_distance=0):
    """Utility to get schedule"""

    total_time_divisions = len(data_col)
    data_col_sign = np.sign(data_col)

    # for SINGLE run only

    section_start = 0
    section_end = len(data_col)
    section_mid = int(len(data_col) / 2)

    if section != 0:
        section_start = int(section / 100)
        section_end = int(section % 100)
        section_mid = (section_start + ((section_end - section_start) % total_time_divisions)) % total_time_divisions

    section_arr = get_schedule_arr(section_start, section_end, total_time_divisions)

    day_arr = data_col_sign[data_col_sign != 0]
    day_td = np.where(data_col_sign != 0)[0]
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
        all_schedules.extend(schedules)

    all_run_schedules = np.array(all_schedules)

    schedule_overlap_arr = np.zeros(len(all_run_schedules))

    for i in range(len(all_run_schedules)):
        schedule_arr = get_schedule_arr(all_run_schedules[i, 0], all_run_schedules[i, 1], total_time_divisions)
        schedule_arr *= section_arr
        schedule_overlap_arr[i] = np.sum(schedule_arr)

    distance_from_section = np.zeros(len(all_run_schedules))

    if np.max(schedule_overlap_arr) == 0:

        midpoint_duration_arr = np.zeros((len(all_run_schedules), 2))

        for i in range(len(all_run_schedules)):
            midpoint_duration_arr[i][0] = (all_run_schedules[i][1] - all_run_schedules[i][0]) % total_time_divisions
            midpoint_duration_arr[i][1] = (all_run_schedules[i][0] + midpoint_duration_arr[i][0]) % total_time_divisions
            distance_from_section[i] = min(abs(section_mid - midpoint_duration_arr[i][1]),
                                           abs(section_mid - midpoint_duration_arr[i][1]) % total_time_divisions)
        if median_distance != 0 and np.min(distance_from_section) <= median_distance:
            closest_schedule_idx = np.argmin(distance_from_section)
            closest_schedule = all_run_schedules[closest_schedule_idx]
        else:
            closest_schedule = [0, 0, 0, 0, 0]

    else:

        closest_schedule_idx = np.argmax(schedule_overlap_arr)
        closest_schedule = all_run_schedules[closest_schedule_idx]

    schedule_start, schedule_end = closest_schedule[0], closest_schedule[1]

    schedule_start_amp = data_col[schedule_start]
    schedule_end_amp = data_col[schedule_end]
    schedule_duration = (schedule_end - schedule_start) % len(data_col)

    schedule = np.zeros(5)
    schedule[0] = schedule_start
    schedule[1] = schedule_end
    schedule[2] = schedule_duration
    schedule[3] = schedule_start_amp
    schedule[4] = abs(schedule_end_amp)

    return np.array(schedule, dtype=int)
