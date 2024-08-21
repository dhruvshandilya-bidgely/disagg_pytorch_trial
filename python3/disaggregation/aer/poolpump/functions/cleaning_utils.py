"""
Author - Mayank Sharan
Date - 20/1/19
Smart merge utility functions, also contains find edges
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.utils.maths_utils.matlab_utils import rolling_sum


def find_edges(col_data):
    """Get start and end indices of value 1"""

    col_edges = np.r_[0, np.sign(col_data), 0]
    edges_diff = np.diff(col_edges)

    start_idx = np.where(edges_diff == 1)[0]
    end_idx = np.where(edges_diff == -1)[0]

    return start_idx, end_idx


def should_be_masked(data_col, pp_config):
    """Utility to decide masking and return strength"""

    # Extract constants from config

    minimum_edge_length = pp_config.get('cleaning_minimum_edge_length')

    # Compute masking

    start_arr, end_arr = find_edges(data_col)

    if len(start_arr) == 0:
        return False, 0

    # Return max edge length if greater than minimum

    diff = end_arr - start_arr

    if np.max(diff) >= minimum_edge_length:
        return True, np.sum(diff)

    return False, 0


def get_consistency_array(data, pp_config):
    """Utility to get consistency array for the given data"""

    # Extract thresholds from config

    window_size = pp_config.get('cleaning_consistency_window_size')
    min_sufficient_length = pp_config.get('cleaning_minimum_sufficient_length')

    edges = np.sign(data)
    required_window_sum = rolling_sum(edges, roll_window=window_size, axis=0)
    fill_consistency_idx = required_window_sum >= min_sufficient_length

    col_idx_iter = np.where(np.sum(fill_consistency_idx, axis=0) > 0)[0]
    consistency_arr = np.zeros(shape=data.shape)

    num_rows_max = data.shape[0] - window_size

    for col_idx in col_idx_iter:
        col_data = edges[:, col_idx]
        col_roll_bool = fill_consistency_idx[:, col_idx]

        row_idx = np.where(col_roll_bool)[0][0]

        while row_idx < num_rows_max:
            if col_roll_bool[row_idx]:
                one_val_idx = np.where(col_data[row_idx: row_idx + window_size] == 1)[0]
                first_occ_idx = row_idx + one_val_idx[0]
                last_occ_idx = row_idx + one_val_idx[-1] + 1
                consistency_arr[first_occ_idx: last_occ_idx, col_idx] = 1
            row_idx += 1

    return consistency_arr


def get_strength(col_data):
    """Utility to get strength for a column"""

    start_arr, end_arr = find_edges(col_data)
    diff = end_arr - start_arr

    if len(diff) == 0:
        return 0

    return np.max(diff)


def get_strength_col(col_data, window_size):
    """Utility to compute vectorised strength"""

    num_rows = len(col_data)
    num_rows_res = num_rows - window_size + 1

    start_arr, end_arr = find_edges(col_data)
    num_idx = len(start_arr)

    if num_idx == 0:
        return np.zeros(shape=(num_rows_res,))

    if num_rows_res <= 0:
        return np.zeros(shape=(0,))

    s_idx_1d = np.arange(num_rows_res)
    s_idx_2d = np.tile(s_idx_1d, reps=(num_idx, 1)).transpose()
    e_idx_2d = s_idx_2d + window_size

    start_idx_2d = np.tile(start_arr, reps=(num_rows_res, 1))
    end_idx_2d = np.tile(end_arr, reps=(num_rows_res, 1))

    res_2d_start = np.zeros(shape=s_idx_2d.shape)
    res_2d_end = np.zeros(shape=s_idx_2d.shape)

    idx_1 = np.logical_and(start_idx_2d <= s_idx_2d, end_idx_2d >= e_idx_2d)

    res_2d_start[idx_1] = s_idx_2d[idx_1]
    res_2d_end[idx_1] = e_idx_2d[idx_1]

    idx_2 = np.logical_and(start_idx_2d >= s_idx_2d, end_idx_2d <= e_idx_2d)

    res_2d_start[idx_2] = start_idx_2d[idx_2]
    res_2d_end[idx_2] = end_idx_2d[idx_2]

    idx_3 = np.logical_and(np.logical_and(end_idx_2d >= s_idx_2d, end_idx_2d <= e_idx_2d), start_idx_2d <= s_idx_2d)

    res_2d_start[idx_3] = s_idx_2d[idx_3]
    res_2d_end[idx_3] = end_idx_2d[idx_3]

    idx_4 = np.logical_and(np.logical_and(start_idx_2d >= s_idx_2d, start_idx_2d <= e_idx_2d), end_idx_2d >= e_idx_2d)

    res_2d_start[idx_4] = start_idx_2d[idx_4]
    res_2d_end[idx_4] = e_idx_2d[idx_4]

    res_diff_arr = res_2d_end - res_2d_start
    res_arr = np.max(res_diff_arr, axis=1)

    return res_arr


def get_consistency_array_strength(data, pp_config):
    """Utility to get consistency array for the given data using strength threshold"""

    # Extract thresholds from config

    passing_number = pp_config.get('cleaning_passing_number')
    window_size = pp_config.get('cleaning_consistency_window_size')
    min_sufficient_length = pp_config.get('cleaning_minimum_sufficient_length')

    edges = np.sign(data)
    required_window_sum = rolling_sum(edges, roll_window=window_size, axis=0)
    fill_consistency_idx = required_window_sum >= passing_number

    col_idx_iter = np.where(np.sum(edges, axis=0) > 0)[0]
    consistency_arr = np.zeros(shape=data.shape)

    num_rows_max = data.shape[0] - window_size

    for col_idx in col_idx_iter:
        col_data = edges[:, col_idx]
        col_roll_bool = fill_consistency_idx[:, col_idx]
        strength_roll_bool = get_strength_col(col_data, window_size) >= min_sufficient_length
        bool_for_col = np.logical_or(strength_roll_bool, col_roll_bool)
        row_idx = 0

        while row_idx < num_rows_max:
            if bool_for_col[row_idx]:
                one_val_idx = np.where(col_data[row_idx: row_idx + window_size] == 1)[0]
                first_occ_idx = row_idx + one_val_idx[0]
                last_occ_idx = row_idx + one_val_idx[-1] + 1
                consistency_arr[first_occ_idx: last_occ_idx, col_idx] = 1
            row_idx += 1

    return consistency_arr


def smart_merge(data_padded, col_idx_from, from_col_cons_arr, col_idx_to, to_col_cons_arr, pp_config):
    """Utility to perform smart merge"""

    # Initialise constants to use

    minimum_sufficient_length = pp_config.get('cleaning_minimum_sufficient_length')
    minimum_distance_from_edge = pp_config.get('cleaning_minimum_distance_from_edge')
    minimum_distance_for_consistency = pp_config.get('cleaning_minimum_distance_for_consistency')

    # Initialize arrays to use for loop

    start_idx_arr, end_idx_arr = find_edges(data_padded[:, col_idx_from])
    edge_sizes = end_idx_arr - start_idx_arr
    num_idx_arr = len(start_idx_arr)
    num_rows = data_padded.shape[0]

    for idx in range(num_idx_arr):

        if edge_sizes[idx] >= minimum_sufficient_length:
            continue

        min_dist_start_idx = max(start_idx_arr[idx] - minimum_distance_from_edge, 0)
        min_dist_end_idx = min(end_idx_arr[idx] + minimum_distance_from_edge, num_rows)

        if np.sum(from_col_cons_arr[min_dist_start_idx: min_dist_end_idx]) > 0:
            continue

        if np.sum(data_padded[min_dist_start_idx: min_dist_end_idx, col_idx_to]) == 0:
            continue

        start_idx = max(start_idx_arr[idx] - minimum_distance_for_consistency, 0)
        end_idx = min(end_idx_arr[idx] + minimum_distance_for_consistency, num_rows)

        if np.sum(to_col_cons_arr[start_idx: end_idx]) == 0:
            continue

        required_data = data_padded[start_idx: end_idx, col_idx_to]

        data_padded[start_idx_arr[idx]: end_idx_arr[idx], col_idx_to] = np.median(required_data[required_data > 0])
        data_padded[start_idx_arr[idx]: end_idx_arr[idx], col_idx_from] = 0

        # Modify consistency array in case it changes

        mod_cons_arr = get_consistency_array_strength(data_padded[:, [col_idx_from, col_idx_to]], pp_config)
        from_col_cons_arr = mod_cons_arr[:, 0]
        to_col_cons_arr = mod_cons_arr[:, 1]

    return data_padded


def get_min_dist(past, future, invalid_distance):
    """Utility to get min dist values"""

    if len(past) == 0:
        min_dist_past = 0
    elif np.sum(past) == 0:
        min_dist_past = invalid_distance
    else:
        min_dist_past = np.argmax(past[::-1])

    if len(future) == 0:
        min_dist_future = 0
    elif np.sum(future) == 0:
        min_dist_future = invalid_distance
    else:
        min_dist_future = np.argmax(future)

    return min(min_dist_past, min_dist_future)


def get_merge_col(col_left, col_right, min_dist_left, min_dist_right, invalid_distance):
    """Utility to get merge column"""

    if (min_dist_right == invalid_distance and min_dist_left < invalid_distance) or (
            min_dist_left > min_dist_right and min_dist_left != invalid_distance):
        merge_col = col_left

    else:
        merge_col = col_right

    return merge_col


def smart_merge_both(data_padded, col_left, col_mid, col_right, consistency_arr_slice, pp_config):
    """Utility to perform smart merge on both sides"""

    # Initialise constants to use

    invalid_distance = pp_config.get('cleaning_invalid_distance')
    minimum_sufficient_length = pp_config.get('cleaning_minimum_sufficient_length')
    minimum_distance_from_edge = pp_config.get('cleaning_minimum_distance_from_edge')
    minimum_distance_for_consistency = pp_config.get('cleaning_minimum_distance_for_consistency')

    # Initialize arrays to use for loop

    start_arr, end_arr = find_edges(data_padded[:, col_mid])
    edge_sizes = end_arr - start_arr
    num_idx_arr = len(start_arr)
    num_rows = data_padded.shape[0]

    # Extract consistency arrays

    consistency_left = consistency_arr_slice[:, 0]
    consistency_mid = consistency_arr_slice[:, 1]
    consistency_right = consistency_arr_slice[:, 2]

    for idx in range(num_idx_arr):

        edge_start, edge_end, edge_length = start_arr[idx], end_arr[idx], edge_sizes[idx]

        if edge_length > minimum_sufficient_length:
            continue

        overlap_consistency_start = max(edge_start - minimum_distance_from_edge, 0)
        overlap_consistency_end = min(edge_end + minimum_distance_from_edge, num_rows)

        if np.sum(consistency_mid[overlap_consistency_start:overlap_consistency_end]) > 0:
            continue

        merge_col = None

        found_edge_overlap_left = (np.sum(data_padded[overlap_consistency_start: overlap_consistency_end, col_left])
                                   > 0)

        found_edge_overlap_right = (np.sum(data_padded[overlap_consistency_start: overlap_consistency_end, col_right])
                                    > 0)

        if (found_edge_overlap_left <= 0) & (found_edge_overlap_right <= 0):
            continue

        if (found_edge_overlap_left <= 0) & (found_edge_overlap_right > 0):
            merge_col = col_right

        if (found_edge_overlap_left > 0) & (found_edge_overlap_right <= 0):
            merge_col = col_left

        consistency_window_start = max(edge_start - minimum_distance_for_consistency, 0)
        consistency_window_end = min(edge_end + minimum_distance_for_consistency, num_rows)

        if merge_col is None:

            left_past = consistency_left[consistency_window_start:edge_start]
            left_future = consistency_left[edge_end:consistency_window_end]
            min_dist_left = get_min_dist(left_past, left_future, invalid_distance)

            right_past = consistency_right[consistency_window_start:edge_start]
            right_future = consistency_right[edge_end:consistency_window_end]
            min_dist_right = get_min_dist(right_past, right_future, invalid_distance)

            if min_dist_left == min_dist_right == invalid_distance:
                continue

            merge_col = get_merge_col(col_left, col_right, min_dist_left, min_dist_right, invalid_distance)

        consistency_window = data_padded[consistency_window_start:consistency_window_end, merge_col]

        data_padded[edge_start: edge_end, merge_col] = np.median(consistency_window[consistency_window > 0])
        data_padded[edge_start: edge_end, col_mid] = 0

        # Modify consistency array in case it changes

        mod_cons_arr = get_consistency_array_strength(data_padded[:, [col_left, col_mid, col_right]], pp_config)
        consistency_left = mod_cons_arr[:, 0]
        consistency_mid = mod_cons_arr[:, 1]
        consistency_right = mod_cons_arr[:, 2]

    return data_padded


def delete_insufficient_edges(data_col, consistency_arr_col, pp_config):
    """Utility to delete insufficient edges"""

    # Extract constants from config

    small_edge_length = pp_config.get('cleaning_small_edge_length')
    minimum_distance_from_edge = pp_config.get('cleaning_minimum_distance_from_edge')

    # Initialize arrays to use

    start_arr, end_arr = find_edges(data_col)
    edge_sizes = end_arr - start_arr
    num_idx_arr = len(start_arr)

    # Delete edges as per required

    for idx in range(num_idx_arr):
        if edge_sizes[idx] < small_edge_length:

            overlap_consistency_delete_start = max(start_arr[idx] - 2 * minimum_distance_from_edge, 0)
            overlap_consistency_delete_end = min(end_arr[idx] + 2 * minimum_distance_from_edge, data_col.shape[0])

            if np.sum(consistency_arr_col[overlap_consistency_delete_start: overlap_consistency_delete_end]) <= 0:
                data_col[start_arr[idx]: end_arr[idx]] = 0

    return data_col
