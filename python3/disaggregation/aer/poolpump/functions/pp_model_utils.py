"""
Author - Mayank Sharan
Date - 21/1/19
All the utilities needed by get pp model
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.disaggregation.aer.poolpump.functions.cleaning_utils import find_edges


def label_edges(data_clean_edges, pp_config, min_edge_length=5):
    """Utility to label edges"""

    # Extract constants out of the config

    # Initialise positive and negative matrices

    data_clean_pos = copy.deepcopy(data_clean_edges)
    data_clean_pos[data_clean_pos < 0] = 0

    data_clean_neg = copy.deepcopy(data_clean_edges)
    data_clean_neg[data_clean_neg > 0] = 0
    data_clean_neg = np.abs(data_clean_neg)

    # Populate positive and negative edges

    pos_edges = []
    neg_edges = []

    # Loop over columns to get positive edges

    col_idx_pos = np.where(np.sum(data_clean_pos, axis=0) > 0)[0]

    for col_idx in col_idx_pos:

        pos_start_arr, pos_end_arr = find_edges(data_clean_pos[:, col_idx])

        for idx in range(len(pos_start_arr)):

            start_idx = pos_start_arr[idx]
            end_idx = pos_end_arr[idx]

            if end_idx - start_idx >= min_edge_length:
                pos_edges.append([start_idx, end_idx, col_idx,
                                  int(np.median(data_clean_pos[start_idx: end_idx, col_idx])), 1])

    # Loop over columns to get negative edges

    col_idx_neg = np.where(np.sum(data_clean_neg, axis=0) > 0)[0]

    for col_idx in col_idx_neg:

        neg_start_arr, neg_end_arr = find_edges(data_clean_neg[:, col_idx])

        for idx in range(len(neg_start_arr)):

            start_idx = neg_start_arr[idx]
            end_idx = neg_end_arr[idx]

            if end_idx - start_idx >= min_edge_length:
                neg_edges.append([start_idx, end_idx, col_idx,
                                  int(np.median(data_clean_neg[start_idx: end_idx, col_idx])), -1])

    return np.array(pos_edges), np.array(neg_edges)


def get_consistency_val(data, match_start_day, match_end_day, amp, pos_col, neg_col, pp_config, window, samples_per_hr):
    """Utility to get signal consistency matrix for given data"""

    # Initialise constants from config

    amplitude_margin = pp_config.get('amplitude_margin')
    relaxed_amp_margin = pp_config.get('relaxed_amp_margin')
    moderate_amp_margin = pp_config.get('moderate_amp_margin')
    minimum_duration_fraction = pp_config.get('minimum_duration_fraction')
    duration_one_hr = pp_config.get('duration_one_hr')

    # Initialise values needed here

    total_cols = data.shape[1]

    if pos_col < neg_col:
        data_reqd = data[window + match_start_day: window + match_end_day, pos_col: neg_col + 1]
    else:
        data_reqd = np.concatenate((data[window + match_start_day: window + match_end_day, pos_col: total_cols],
                                    data[window + match_start_day: window + match_end_day, : neg_col + 1]), axis=1)

    # Get min amp and data check

    amp_per_hr = amp / samples_per_hr
    duration = ((neg_col - pos_col) % total_cols) / samples_per_hr

    if samples_per_hr == 1:
        min_amp = relaxed_amp_margin * amp_per_hr
        data_check = copy.deepcopy(data_reqd)
        if duration >= duration_one_hr:
            data_check = copy.deepcopy(data_reqd[:, :-1])
    elif samples_per_hr == 2:
        min_amp = moderate_amp_margin * amp_per_hr
        data_check = copy.deepcopy(data_reqd[:, 1: -1])
    else:
        min_amp = amplitude_margin * amp_per_hr
        data_check = copy.deepcopy(data_reqd[:, 1: -2])

    check_rows, check_cols = data_check.shape

    # Compute final values

    data_check[data_check < min_amp] = 0
    data_check = np.sign(data_check)
    duration_fraction_arr = np.sum(data_check, axis=1)

    # Compute duration fraction array

    if samples_per_hr == 4:
        min_border_amp = relaxed_amp_margin * amp_per_hr
        second_last_time_div_data = data_reqd[:, -2]

        idx_1 = second_last_time_div_data < min_border_amp
        duration_fraction_arr[idx_1] = duration_fraction_arr[idx_1] / check_cols

        idx_2 = np.logical_not(idx_1)
        duration_fraction_arr[idx_2] = (duration_fraction_arr[idx_2] + 1) / (check_cols + 1)
    else:
        duration_fraction_arr = duration_fraction_arr / check_cols

    # Compute aggregate values

    idx_dur_valid = duration_fraction_arr >= minimum_duration_fraction
    duration_fraction_sum = np.sum(duration_fraction_arr[idx_dur_valid])
    days_carried = np.sum(idx_dur_valid)
    days_in_match = np.sum(duration_fraction_arr == 1)

    if days_carried == 0:
        day_fraction = 0
    else:
        day_fraction = duration_fraction_sum / days_carried

    match_fraction = days_in_match / (match_end_day - match_start_day + 1)

    return day_fraction, match_fraction


def find_potential_matches_all(data, pos_edges, neg_edges, num_cols, samples_per_hr, pp_config, minimum_pair_length=5,
                               empty_flag=0):
    """Utility to find all potential matches"""

    # Extract constants from the config

    min_duration = pp_config.get('min_duration')
    max_duration = pp_config.get('max_duration')
    min_amp_ratio_higher = pp_config.get('min_amp_ratio_higher')
    min_pair_length_lower = pp_config.get('min_pair_length_lower')
    min_pair_length_higher = pp_config.get('min_pair_length_higher')
    default_min_amp_ratio = pp_config.get('default_min_amp_ratio')
    min_amp_ratio_lower = pp_config.get('min_amp_ratio_lower')
    minimum_area_under_curve = pp_config.get('minimum_area_under_curve')
    minimum_match_signal_fraction = pp_config.get('minimum_match_signal_fraction')
    amp_ratio_reduction_factor = pp_config.get('amp_ratio_reduction_factor')
    minimum_day_signal_fraction = pp_config.get('minimum_day_signal_fraction')

    num_rows = data.shape[0]
    min_amp_ratio = default_min_amp_ratio * (amp_ratio_reduction_factor ** 4)

    # Initialise variables for use in the function

    num_pairs_day = np.zeros(shape=(num_rows,))
    duration_days = np.zeros(shape=(num_rows,))

    matches = []
    duration_list = []
    num_pairs_list = []

    if (len(pos_edges) == 0) or (len(neg_edges) == 0):
        return np.array(matches), num_pairs_day, duration_days

    for pos_edge_idx in range(len(pos_edges)):

        # Initialize values from the positive edge

        pos_edge_start = pos_edges[pos_edge_idx, 0]
        pos_edge_end = pos_edges[pos_edge_idx, 1]
        pos_edge_col = pos_edges[pos_edge_idx, 2]
        pos_edge_amp = pos_edges[pos_edge_idx, 3]

        # Extract eligible negative edges

        eligible_neg_edges_bool = np.logical_and(neg_edges[:, 0] < pos_edge_end, neg_edges[:, 1] > pos_edge_start)
        eligible_neg_edges_idx = np.where(eligible_neg_edges_bool)[0]
        eligible_neg_edges = neg_edges[eligible_neg_edges_bool, :]

        # Compute first rejection criteria for edges

        min_amp = np.minimum(eligible_neg_edges[:, 3], pos_edge_amp)
        max_amp = np.maximum(eligible_neg_edges[:, 3], pos_edge_amp)
        amp_ratios = np.divide(min_amp, max_amp)

        match_start_days = np.maximum(eligible_neg_edges[:, 0], pos_edge_start)
        match_end_days = np.minimum(eligible_neg_edges[:, 1], pos_edge_end)
        pair_lengths = match_end_days - match_start_days

        # Possible bug here in precedence of and and or

        rej_idx_1 = np.logical_or(np.logical_and(empty_flag == 0, np.logical_and(pair_lengths < min_pair_length_lower,
                                                                                 amp_ratios < min_amp_ratio_higher)),
                                  np.logical_and(pair_lengths < min_pair_length_higher,
                                                 amp_ratios < min_amp_ratio_lower))
        rej_idx_1 = np.logical_not(rej_idx_1)

        # Compute rejection criteria 2

        rej_idx_2 = np.logical_not(np.logical_and(empty_flag == 0, pair_lengths < minimum_pair_length))

        # Compute rejection criteria 3 based on area under curve

        duration_arr = ((eligible_neg_edges[:, 2] - pos_edge_col) % num_cols) / samples_per_hr
        auc_arr = np.multiply(pos_edge_amp + eligible_neg_edges[:, 3], duration_arr / 2)

        rej_idx_3 = np.logical_not(auc_arr < minimum_area_under_curve)

        # Compute rejection criteria 4 based on duration

        rej_idx_4 = np.logical_not(np.logical_or(duration_arr < min_duration, duration_arr > max_duration))

        selected_neg_edge_idx_arr = np.where(np.logical_and(np.logical_and(rej_idx_1, rej_idx_2),
                                                            np.logical_and(rej_idx_3, rej_idx_4)))[0]

        for neg_idx in selected_neg_edge_idx_arr:

            match_start_day = match_start_days[neg_idx]
            match_end_day = match_end_days[neg_idx]
            duration_pts = duration_arr[neg_idx] * samples_per_hr

            # Get signal consistency matrix for the given edges

            day_f, match_f = get_consistency_val(data, match_start_day, match_end_day, min_amp[neg_idx], pos_edge_col,
                                                 eligible_neg_edges[neg_idx, 2], pp_config, window=0,
                                                 samples_per_hr=samples_per_hr)

            if empty_flag == 0 and (day_f < minimum_day_signal_fraction or match_f < minimum_match_signal_fraction):
                continue

            try:
                signal_score = int(50 * (day_f + match_f))
            except ValueError:
                signal_score = 0

            if empty_flag == 0 and amp_ratios[neg_idx] < min_amp_ratio:
                continue

            pair_bool_arr = np.zeros(shape=(num_rows,))
            pair_bool_arr[match_start_day: match_end_day] = 1
            num_pairs_list.append(pair_bool_arr)

            duration_array = np.zeros(shape=(num_rows,))
            duration_array[match_start_day:match_end_day] = duration_pts
            duration_list.append(duration_array)

            matches.append([pos_edge_idx, eligible_neg_edges_idx[neg_idx], match_start_day, match_end_day,
                            pair_lengths[neg_idx], duration_pts, pos_edges[pos_edge_idx, 4],
                            eligible_neg_edges[neg_idx, 4], signal_score])

    matches = np.array(matches)

    if matches.shape[0] == 0:
        return matches, num_pairs_day, duration_days

    sorted_idx = matches[:, 2].argsort()
    sorted_matches = matches[sorted_idx, :].astype(int)

    num_pairs_day = np.sum(np.array(num_pairs_list), axis=0)
    duration_days = np.max(np.array(duration_list), axis=0)

    return sorted_matches, num_pairs_day, duration_days


def should_be_masked(data_col, pp_config):
    """Utility to decide if we should mask column"""

    # Extract constants from config

    minimum_edge_length = 30

    start_arr, end_arr = find_edges(data_col)

    if len(start_arr) == 0:
        return False

    diff = end_arr - start_arr

    # Based max sequence decide return value

    if np.max(diff) >= minimum_edge_length:
        return True

    return False


def duration_validity(time_div_arr, total_time_divisions, samples_per_hour, pp_config):
    """Utility to decide validity of duration"""

    # Extract constants from the config

    min_duration = pp_config.get('min_duration')
    max_duration = pp_config.get('max_duration')

    duration = ((time_div_arr[-1] - time_div_arr[0]) % total_time_divisions) / samples_per_hour

    if min_duration <= duration <= max_duration:
        return True

    return False


def fill_run_matrix(time_div_dict, data, num_cols, samples_per_hour, pp_config):
    """Utility to fill run matrix"""

    # Extract constants from config

    minimum_days_run = 45

    # Initialize variables

    data_rows, data_cols = data.shape

    primary_mask = np.tile(np.bitwise_or(time_div_dict['primary_pos'], time_div_dict['primary_neg']),
                           reps=(data_rows, 1))

    secondary_mask = np.tile(np.bitwise_or(time_div_dict['secondary_pos'], time_div_dict['secondary_neg']),
                             reps=(data_rows, 1))

    data = copy.deepcopy(data)
    data = np.sign(data)
    data_abs = np.abs(data)

    run_matrix = np.zeros(shape=(data_rows, 4))

    # Mask data

    masking_idx = np.logical_not(np.logical_or(data_abs == primary_mask, data_abs == secondary_mask))
    data[masking_idx] = 0

    data_reqd = data
    non_zero_idx = np.logical_not(data_reqd == 0)

    day_sum = np.sum(data_reqd, axis=1)
    day_sum_abs = np.abs(day_sum)

    day_abs_sum = np.sum(np.abs(data_reqd), axis=1)

    idx_run_mat_col_0 = np.logical_or(day_sum_abs == day_abs_sum, day_abs_sum == 0)
    idx_run_mat_col_1 = np.logical_and(np.logical_and(day_sum == 0, day_abs_sum == 2),
                                       np.logical_not(idx_run_mat_col_0))

    run_matrix[idx_run_mat_col_0, 0] = 1
    run_matrix[idx_run_mat_col_1, 1] = 2

    skip_idx = np.logical_or(idx_run_mat_col_0, idx_run_mat_col_1)

    for day in range(data_rows):

        day_data = data_reqd[day, :]

        day_arr = day_data[non_zero_idx[day, :]]
        day_td = np.where(non_zero_idx[day, :])[0]

        if skip_idx[day]:
            continue

        if day_arr[0] < 0:

            roll_idx = np.where(day_arr == 1)[0]
            day_arr = np.roll(day_arr, -roll_idx[0])
            day_td = np.roll(day_td, -roll_idx[0])

        elif day_arr[-1] > 0:

            roll_idx = np.where(day_arr[::-1] == -1)[0]
            day_arr = np.roll(day_arr, roll_idx[0])
            day_td = np.roll(day_td, roll_idx[0])

        num_of_runs = np.where(day_arr[:-1] < day_arr[1:])[0]

        if len(num_of_runs) == 0:

            run_matrix[day, 1] = 2
            awe_sum = np.sum(day_arr[:-1] * day_arr[1:])

            if awe_sum == (day_abs_sum[day] - 3) and duration_validity(day_td, num_cols, samples_per_hour, pp_config):
                run_matrix[day, 3] = 4
        else:
            run_matrix[day, 1] = 2
            run_matrix[day, 2] = 3

    multiple_start_arr, multiple_end_arr = find_edges(run_matrix[:, 2])
    multiple_diff_arr = multiple_end_arr - multiple_start_arr
    variable_start_arr, variable_end_arr = find_edges(run_matrix[:, 3])
    variable_diff_arr = variable_end_arr - variable_start_arr

    small_multiple_days = np.where(multiple_diff_arr < minimum_days_run)[0]
    small_variable_days = np.where(variable_diff_arr < minimum_days_run)[0]

    multiple_start_arr = multiple_start_arr[small_multiple_days]
    multiple_end_arr = multiple_end_arr[small_multiple_days]

    variable_start_arr = variable_start_arr[small_variable_days]
    variable_end_arr = variable_end_arr[small_variable_days]

    for i in range(len(multiple_start_arr)):
        run_matrix[multiple_start_arr[i]:multiple_end_arr[i], 2] = 0

    for i in range(len(variable_start_arr)):
        run_matrix[variable_start_arr[i]:variable_end_arr[i], 3] = 0

    return run_matrix


def get_run_probability_matrix(data_1, data_2, samples_per_hour, pp_config):
    """Utility to get run probability matrix data1 = clean_union data2 = smooth_edges"""

    num_cols = data_1.shape[1]
    time_div_dict = {
        'primary_strength': np.zeros(num_cols),
        'primary_pos': np.zeros(num_cols, dtype=int),
        'primary_neg': np.zeros(num_cols, dtype=int),
        'secondary_strength': np.zeros(num_cols),
        'secondary_pos': np.zeros(num_cols, dtype=int),
        'secondary_neg': np.zeros(num_cols, dtype=int),
    }

    # Get positive and negative arrays from data_2

    pos_data_2 = copy.deepcopy(data_2)
    pos_data_2[pos_data_2 < 0] = 0
    pos_data_2 = np.sign(pos_data_2)

    neg_data_2 = copy.deepcopy(data_2)
    neg_data_2[neg_data_2 > 0] = 0
    neg_data_2 = np.sign(neg_data_2)
    neg_data_2 = np.abs(neg_data_2)

    # Get positive and negative arrays from data_1

    pos_data_1 = copy.deepcopy(data_1)
    pos_data_1[pos_data_1 < 0] = 0
    pos_data_1 = np.sign(pos_data_1)

    neg_data_1 = copy.deepcopy(data_1)
    neg_data_1[neg_data_1 > 0] = 0
    neg_data_1 = np.sign(neg_data_1)
    neg_data_1 = np.abs(neg_data_1)

    for col_idx in range(num_cols):
        pos_bool_val_2 = should_be_masked(pos_data_2[:, col_idx], pp_config)
        neg_bool_val_2 = should_be_masked(neg_data_2[:, col_idx], pp_config)

        pos_bool_val_1 = should_be_masked(pos_data_1[:, col_idx], pp_config)
        neg_bool_val_1 = should_be_masked(neg_data_1[:, col_idx], pp_config)

        time_div_dict['primary_pos'][col_idx] = pos_bool_val_2
        time_div_dict['primary_neg'][col_idx] = neg_bool_val_2

        time_div_dict['secondary_pos'][col_idx] = pos_bool_val_1
        time_div_dict['secondary_neg'][col_idx] = neg_bool_val_1

    pos_strength_2 = np.sum(pos_data_2, axis=0)
    pos_strength_2[np.logical_not(time_div_dict['primary_pos'])] = 0

    neg_strength_2 = np.sum(neg_data_2, axis=0)
    neg_strength_2[np.logical_not(time_div_dict['primary_neg'])] = 0

    time_div_dict['primary_strength'] = np.maximum(pos_strength_2, neg_strength_2)

    pos_strength_1 = np.sum(pos_data_1, axis=0)
    pos_strength_1[np.logical_not(time_div_dict['secondary_pos'])] = 0

    neg_strength_1 = np.sum(neg_data_1, axis=0)
    neg_strength_1[np.logical_not(time_div_dict['secondary_neg'])] = 0

    time_div_dict['secondary_strength'] = np.maximum(pos_strength_1, neg_strength_1)

    run_matrix = fill_run_matrix(time_div_dict, data_1, num_cols, samples_per_hour, pp_config)

    return time_div_dict, run_matrix


def get_distance_from_masked_edge(start_day, end_day, data_col, start_arr, end_arr, pp_config):
    """Utility to get distance from masked edge"""

    min_length = 30

    if end_day - start_day >= min_length:
        return 1

    data_col_copy = data_col.copy()

    masked_edges_idx = np.where((end_arr - start_arr) >= min_length)[0]

    for i in masked_edges_idx:
        data_col_copy[start_arr[i]:end_arr[i]] = 1
    data_col_copy[data_col_copy != 1] = 0

    data_before_edge = data_col_copy[:start_day]
    data_after_edge = data_col_copy[end_day:]

    distance_before_edge = len(data_col)
    distance_after_edge = len(data_col)

    if np.sum(data_before_edge) != 0:
        distance_before_edge = np.argmax(data_before_edge[::-1])

    if np.sum(data_after_edge) != 0:
        distance_after_edge = np.argmax(data_after_edge)

    distance_before_edge = max(1, distance_before_edge)
    distance_after_edge = max(1, distance_after_edge)

    return min(distance_before_edge, distance_after_edge)


def label_edges_time_mask_score(data, time_div_dict, pp_config, min_edge_length=5, empty_flag=0):
    """Label edges in time by mask score"""

    data_copy = data.copy()

    pos_data = data_copy.copy()
    pos_data[pos_data < 0] = 0
    pos_data[pos_data > 0] = 1

    neg_data = data_copy.copy()
    neg_data[neg_data > 0] = 0
    neg_data[neg_data < 0] = 1
    neg_data = np.abs(neg_data)

    pos_edges = list()
    neg_edges = list()

    for time_div in range(data.shape[1]):

        pos_start_arr, pos_end_arr = find_edges(pos_data[:, time_div])
        neg_start_arr, neg_end_arr = find_edges(neg_data[:, time_div])

        for i in range(len(pos_start_arr)):

            if empty_flag == 0 and pos_end_arr[i] - pos_start_arr[i] < min_edge_length:
                continue

            pos_edge = np.zeros(5)
            pos_edge[0] = pos_start_arr[i]
            pos_edge[1] = pos_end_arr[i]
            pos_edge[2] = time_div
            pos_edge[3] = np.median(data_copy[pos_start_arr[i]:pos_end_arr[i], time_div])
            pos_edge[4] = time_div_dict['primary_pos'][time_div] * 2 + time_div_dict['secondary_pos'][time_div]

            distance_from_masked_edge = get_distance_from_masked_edge(pos_start_arr[i], pos_end_arr[i],
                                                                      pos_data[:, time_div], pos_start_arr, pos_end_arr,
                                                                      pp_config)

            pos_edge[4] /= distance_from_masked_edge

            pos_edges.append(pos_edge)

        for i in range(len(neg_start_arr)):

            if empty_flag == 0 and neg_end_arr[i] - neg_start_arr[i] < min_edge_length:
                continue

            neg_edge = np.zeros(5)
            neg_edge[0] = neg_start_arr[i]
            neg_edge[1] = neg_end_arr[i]
            neg_edge[2] = time_div
            neg_edge[3] = abs(np.median(data_copy[neg_start_arr[i]:neg_end_arr[i], time_div]))
            neg_edge[4] = time_div_dict['primary_neg'][time_div] * 2 + time_div_dict['secondary_neg'][time_div]

            distance_from_masked_edge = get_distance_from_masked_edge(neg_start_arr[i], neg_end_arr[i],
                                                                      neg_data[:, time_div], neg_start_arr, neg_end_arr,
                                                                      pp_config)

            neg_edge[4] /= distance_from_masked_edge
            neg_edges.append(neg_edge)

    return np.array(pos_edges, dtype=int), np.array(neg_edges, dtype=int)


def get_num_of_runs_each_day(all_pairs_matrix):
    """Utility to get number of runs per day"""

    num_rows = all_pairs_matrix.shape[0]

    all_pairs_matrix_sign = np.sign(all_pairs_matrix)

    num_of_runs_each_day = np.zeros(num_rows)

    for day in range(num_rows):
        day_df = all_pairs_matrix_sign[day, :]
        day_arr = day_df[day_df != 0]

        if len(day_arr) == 0:
            num_of_runs_each_day[day] = 0
            continue

        if day_arr[0] < 0:
            roll_idx = np.where(day_arr == 1)[0]
            day_arr = np.roll(day_arr, -roll_idx[0])
        elif day_arr[-1] > 0:
            roll_idx = np.where(day_arr[::-1] == -1)[0]
            day_arr = np.roll(day_arr, roll_idx[0])

        num_of_runs = np.where(day_arr[:-1] < day_arr[1:])[0]

        num_of_runs_each_day[day] = len(num_of_runs) + 1

    return num_of_runs_each_day


def get_days_labeled(all_pairs_matrix, num_of_runs_each_day, days):
    """Utility to label days"""

    num_of_runs_each_day_copy = num_of_runs_each_day.copy()

    all_pairs_pos_part = all_pairs_matrix.copy()
    all_pairs_pos_part[all_pairs_pos_part < 0] = 0
    all_pairs_neg_part = all_pairs_matrix.copy()
    all_pairs_neg_part[all_pairs_neg_part > 0] = 0

    pos_time_divisions = np.count_nonzero(all_pairs_pos_part, axis=1)
    neg_time_divisions = np.count_nonzero(all_pairs_neg_part, axis=1)

    days_label = np.zeros(days)

    uncontested_idx = np.where((pos_time_divisions == neg_time_divisions) &
                               (pos_time_divisions == num_of_runs_each_day_copy) & (num_of_runs_each_day != 0))[0]

    days_label[uncontested_idx] = 1

    multiple_idx = np.where(((pos_time_divisions > 0) & (pos_time_divisions != num_of_runs_each_day_copy)) |
                            ((neg_time_divisions > 0) & (neg_time_divisions != num_of_runs_each_day_copy)))[0]

    days_label[multiple_idx] = 2

    return days_label


def remove_small_uncontested_pairs(days_label, pp_config):
    """Utility to remove small uncontested pairs"""

    minimum_uncontested_pair_length = 7

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


def change_days_label_for_weak_runs(num_of_runs_each_day, days_label, pp_config):
    """Utility to change days label for weak runs"""

    min_run_days = 18

    distinct_runs = np.array(list(set(num_of_runs_each_day)))

    for run in distinct_runs:

        num_of_runs_each_day_copy = num_of_runs_each_day.copy()
        num_of_runs_each_day_copy[num_of_runs_each_day_copy != run] = -1
        start_arr, end_arr = find_edges(num_of_runs_each_day_copy + 1)
        weak_run_idx = np.where((end_arr - start_arr) < min_run_days)[0]

        for i in weak_run_idx:
            days_label[start_arr[i]:end_arr[i]] = 2

    return days_label


def get_runs_from_uncontested(uncontested_matrix, total_time_divisions, samples_per_hour, pp_config):
    """Utility to get runs fromo uncontested matrix"""

    num_rows, num_cols = uncontested_matrix.shape
    uncontested_matrix_copy = np.sign(uncontested_matrix)

    runs_matrix = np.zeros((num_rows, 6))

    for day in range(num_rows):
        day_df = uncontested_matrix_copy[day, :]
        day_arr = day_df[day_df != 0]
        day_td = np.where(day_df != 0)[0]

        if np.sum(abs(day_arr)) == 0:
            runs_matrix[day, 0] += 1
            continue

        if np.sum(day_arr) == 0 and len(day_arr) == 2:
            runs_matrix[day, 1] += 1
            continue

        if day_arr[0] < 0:
            roll_idx = np.argmax(day_arr)
            day_arr = np.roll(day_arr, -roll_idx)
            day_td = np.roll(day_td, -roll_idx)
        elif day_arr[-1] > 0:
            roll_idx = np.argmin(day_arr[::-1])
            day_arr = np.roll(day_arr, roll_idx)
            day_td = np.roll(day_td, roll_idx)

        num_of_runs = np.where(day_arr[:-1] < day_arr[1:])[0]

        runs_matrix[day, 1] += 1

        if len(num_of_runs) == 0:
            awe_sum = np.sum(day_arr[:-1] * day_arr[1:])
            if awe_sum == (len(day_arr) - 3) and duration_validity(day_td, total_time_divisions, samples_per_hour,
                                                                   pp_config):
                runs_matrix[day, 5] += 1

        else:
            runs_matrix[day, min((len(num_of_runs) + 1), 4)] += 1

    return runs_matrix


def get_direct_probability_value(pp_run_matrix, col, pp_config, first_longest_threshold=100,
                                 first_second_longest_threshold=150):
    """Utility to get direct probability value"""

    days_threshold = 30

    m = pp_run_matrix.shape[0]
    start_arr, end_arr = find_edges(pp_run_matrix[:, col])

    if len(start_arr) == 0 and len(end_arr) == 0:
        return 0

    diff = end_arr - start_arr
    diff = np.array(sorted(diff, reverse=True))
    diff = diff[np.where(diff > days_threshold)[0]]

    if (len(diff) > 0 and (diff[0] >= first_longest_threshold) or (
            len(diff) > 1 and (diff[0] + diff[1]) >= first_second_longest_threshold)) or np.sum(diff) > (m / 2):
        return 1

    return 0


def get_probability(pp_run_matrix, run_type, col, pp_config):
    """Get probability of a type of run"""

    minimum_data_days = 360

    m = pp_run_matrix.shape[0]

    if m >= minimum_data_days:
        direct_value = max(get_direct_probability_value(pp_run_matrix, col, pp_config),
                           get_direct_probability_value(pp_run_matrix[:int(m / 2), :], col, pp_config),
                           get_direct_probability_value(pp_run_matrix[int(m / 2):, :], col, pp_config))
    else:
        direct_value = get_direct_probability_value(pp_run_matrix, col, pp_config, first_longest_threshold=60,
                                                    first_second_longest_threshold=100)

    if run_type == 'single':
        return max(direct_value, np.count_nonzero(pp_run_matrix[:, col]) / m)

    return max(direct_value, np.count_nonzero(pp_run_matrix[:, col]) / np.count_nonzero(pp_run_matrix[:, 1]))


def get_raw_data_percentile(raw_data, pair, total_time_divisions, avg_consumption, pp_config):
    """Utility to get raw data percentile"""

    # Extract constants out of config

    pth_percentile = 50

    # Compute the raw data percentile

    duration = (pair[1] - pair[0]) % total_time_divisions
    time_div_reqd = np.arange(pair[0], pair[0] + duration + 1) % total_time_divisions
    df_required = raw_data[pair[0]:pair[1], time_div_reqd]

    try:
        percentile_value = np.percentile(df_required[df_required > 0], pth_percentile)
    except IndexError:
        # if no values in df_required are greater than zero
        percentile_value = avg_consumption

    return percentile_value


def get_day_and_time_division_array(df_row):
    """Utility to get day and time division array"""

    df_row_copy = df_row.copy()
    df_row_copy[df_row_copy > 0] = 1
    df_row_copy[df_row_copy < 0] = -1
    day_arr = df_row_copy[df_row_copy != 0]
    day_td = np.where(df_row_copy != 0)[0]

    if len(day_arr) != 0 and day_arr[0] < 0:
        roll_idx = np.argmax(day_arr)
        day_arr = np.roll(day_arr, -roll_idx)
        day_td = np.roll(day_td, -roll_idx)
    elif len(day_arr) != 0 and day_arr[-1] > 0:
        roll_idx = np.argmin(day_arr[::-1])
        day_arr = np.roll(day_arr, roll_idx)
        day_td = np.roll(day_td, roll_idx)

    return day_arr, day_td


def get_consumption_matrix(pairs, raw_data_processed, raw_data, samples_per_hour, pp_config):
    """Utility to get consumption matrix using pairs and raw data"""

    # Extract constants out of config

    consumption_threshold = 0.5
    min_days = 20

    # Initialize a few constants and variables needed for the function

    total_time_divisions = raw_data.shape[1]
    output_matrix = np.zeros_like(raw_data)

    if len(pairs) == 0:
        return output_matrix, 0

    added_params = list()

    # For each pair fill in the consumption

    for pair in pairs:
        avg_consumption = ((pair[4] + pair[5]) / 2) / samples_per_hour
        raw_data_percentile = get_raw_data_percentile(raw_data_processed, pair, total_time_divisions, avg_consumption,
                                                      pp_config)

        eligibility = 0
        if (pair[3] - pair[2]) >= min_days:
            eligibility = 1

        switch = 0
        if raw_data_percentile > avg_consumption or (
                np.abs(raw_data_percentile - avg_consumption) / raw_data_percentile) > consumption_threshold:
            switch = 1

        df_required = raw_data[pair[2]:pair[3], :]
        consumption_avg = np.sum(df_required) / (pair[3] - pair[2])
        if np.isnan(consumption_avg):
            consumption_avg = 0

        added_params.append([switch, consumption_avg, eligibility])

    added_params = np.array(added_params)
    eligible_pairs = added_params[np.where(added_params[:, -1] == 1)[0]]

    if len(eligible_pairs) == 0:
        eligible_pairs = added_params

    consumption_avg_df = eligible_pairs[:, 1]
    cons_threshold = np.percentile(consumption_avg_df, 30)
    desired_eligible_pairs = pairs[np.where(added_params[:, 1] <= cons_threshold)[0]]

    if len(desired_eligible_pairs) == 0:
        desired_eligible_pairs = pairs

    all_values = np.array([])

    for pair in desired_eligible_pairs:
        duration = (pair[1] - pair[0]) % total_time_divisions
        time_div_reqd = np.arange(pair[0], pair[0] + duration + 1) % total_time_divisions
        df_required = raw_data_processed[pair[2]:pair[3], time_div_reqd]
        all_values = np.r_[all_values, df_required.flatten()]

    raw_data_percentile = np.percentile(all_values, 35)

    for pair in pairs:
        avg_consumption = ((pair[4] + pair[5]) / 2) / samples_per_hour

        consumption = max(raw_data_percentile, avg_consumption)

        # fill output matrix

        duration = (pair[1] - pair[0]) % total_time_divisions
        time_div_reqd = np.arange(pair[0], pair[0] + duration + 1) % total_time_divisions

        df_required = raw_data[pair[2]:pair[3], time_div_reqd]
        output_matrix[pair[2]:pair[3], time_div_reqd] = np.where(df_required <= consumption, df_required, consumption)

    return output_matrix, raw_data_percentile
