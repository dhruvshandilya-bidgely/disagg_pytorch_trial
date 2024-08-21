"""
Author - Mayank Sharan
Date - 19/1/19
Apply smart union to the input data
"""

# Import python packages

import copy
import numpy as np


def fill_median_value(df, i, j, val_sign):
    """Utility to fill median value"""

    history_window = 100
    passing_window_size = 20

    m, _ = df.shape
    begin_window = max(0, i - history_window)
    end_window = min(m - 1, i + history_window)

    required_df = df[begin_window: end_window + 1, j - 1: j + 2]
    required_df_copy = copy.deepcopy(required_df)

    if val_sign < 0:
        required_df_copy[required_df_copy > 0] = 0
        required_df_copy[required_df_copy < 0] = 1
    else:
        required_df_copy[required_df_copy < 0] = 0
        required_df_copy[required_df_copy > 0] = 1

    max_length = 0
    required_col = -1

    for col in range(required_df.shape[1]):

        arr = required_df_copy[:, col]
        start_arr = np.where(arr[:-1] < arr[1:])[0]
        end_arr = np.where(arr[:-1] > arr[1:])[0]

        if len(start_arr) < len(end_arr):
            start_arr = np.r_[0, start_arr]

        if len(end_arr) < len(start_arr):
            end_arr = np.r_[end_arr, len(arr) - 1]

        diff = end_arr - start_arr

        if len(diff) > 0 and np.max(diff) >= passing_window_size and np.max(diff) > max_length:
            max_length = np.max(diff)
            required_col = col

    if required_col == -1:
        return -1, required_col + (j - 1)

    required_df = required_df[:, required_col]

    if val_sign > 0:
        median_value, col_idx = np.median(required_df[required_df > 0]), required_col + (j - 1)
    else:
        median_value, col_idx = np.median(required_df[required_df < 0]), required_col + (j - 1)
    return median_value, col_idx


def retain_in_nms(data_nms_3d, data_nms_padded, data_nms_raw_padded, val_sign_arr, data_smart_union, pp_config):
    """Retain in nms utility that works with 3d arrays"""

    # Extract parameters from config dictionary

    window_size = pp_config.get('smart_union_window_size')
    sliding_window = pp_config.get('smart_union_sliding_window')
    history_window = pp_config.get('smart_union_history_window')
    passing_window_size = pp_config.get('smart_union_passing_window_size')
    lower_threshold_fraction = pp_config.get('smart_union_lower_threshold_fraction')
    upper_threshold_fraction = pp_config.get('smart_union_upper_threshold_fraction')
    first_loose_passing_window = pp_config.get('smart_union_first_loose_passing_window')
    second_loose_passing_window = pp_config.get('smart_union_second_loose_passing_window')

    num_z_idx = 2 * history_window + 1

    # Initialise bool and median arrays to work with

    bool_val_array = np.full(shape=data_smart_union.shape, fill_value=False)
    median_val_array = np.full(shape=data_smart_union.shape, fill_value=-1.0)

    # Perform the analysis on NMS data

    values_set_idx_array = np.full(shape=data_smart_union.shape, fill_value=False)

    day_completed = np.full(shape=(data_nms_3d.shape[0],), fill_value=False)

    for idx in range(0, num_z_idx - window_size + sliding_window, sliding_window):
        # Get data to work with

        data_nms_curr = np.full(shape=(data_nms_3d.shape[0], window_size, data_nms_3d.shape[2]), fill_value=np.nan)
        data_nms_curr[:, :min(num_z_idx - idx, window_size), :] = data_nms_3d[:, idx: idx + window_size, :]
        valid_obs = np.sum(np.isfinite(data_nms_curr), axis=1)

        # Decide which days are in their last cycle and give them 35 points

        obs_count = valid_obs[:, 0]
        last_day_idx = np.where(np.logical_and(obs_count < window_size, np.logical_not(day_completed)))[0]

        for day_idx in last_day_idx:
            day_completed[day_idx] = True
            delta_window = window_size - obs_count[day_idx]
            data_nms_curr[day_idx, :, :] = data_nms_3d[day_idx, idx - delta_window:idx + window_size - delta_window, :]

        # Possible bug in the old code we do not consider exactly 35 point arrays here

        valid_obs = np.sum(np.isfinite(data_nms_curr), axis=1)

        useful_obs_idx = valid_obs[:, 0] >= window_size
        data_nms_useful = data_nms_curr[useful_obs_idx, :, :]
        valid_obs_useful = valid_obs[useful_obs_idx, :]

        data_nms_sorted = np.sort(data_nms_useful, axis=1)
        data_nms_sorted_abs = np.abs(data_nms_sorted)

        # Extract positive average
        pos_avg_col = np.divide(np.sum(data_nms_sorted > 0, axis=1), valid_obs_useful)

        # Extract negative average
        neg_avg_col = np.divide(np.sum(data_nms_sorted < 0, axis=1), valid_obs_useful)

        # Compute the median values

        num_med_vals = valid_obs_useful.size
        num_med_rows = valid_obs_useful.shape[0]
        num_med_cols = valid_obs_useful.shape[1]

        dim_1_idx = (np.arange(0, num_med_vals) / num_med_cols).astype(int)
        dim_3_idx = np.tile(np.arange(0, num_med_cols), reps=(num_med_rows,))

        median_idx = np.reshape((valid_obs_useful - 1) / 2, newshape=(num_med_vals,))
        dim_2_f_idx = np.floor(median_idx).astype(int)
        dim_2_c_idx = np.ceil(median_idx).astype(int)

        median_value_arr = np.reshape((data_nms_sorted[dim_1_idx, dim_2_f_idx, dim_3_idx] +
                                       data_nms_sorted[dim_1_idx, dim_2_c_idx, dim_3_idx]) / 2,
                                      newshape=(num_med_rows, num_med_cols))

        median_sign_arr = np.sign(median_value_arr)
        median_sign_arr_3d = np.swapaxes(np.tile(median_sign_arr, (window_size, 1, 1)), axis1=0, axis2=1)

        # Get upper and lower bound and start using conditions to enter median values

        upper_bound_arr = np.abs((1 + upper_threshold_fraction) * median_value_arr)
        upper_bound_arr = np.swapaxes(np.tile(upper_bound_arr, (window_size, 1, 1)), axis1=0, axis2=1)
        lower_bound_arr = np.abs((1 - lower_threshold_fraction) * median_value_arr)
        lower_bound_arr = np.swapaxes(np.tile(lower_bound_arr, (window_size, 1, 1)), axis1=0, axis2=1)

        data_nms_sorted_med_sign = np.multiply(median_sign_arr_3d, data_nms_sorted)

        count_idx_arr = np.logical_and(np.logical_and(data_nms_sorted_abs <= upper_bound_arr,
                                                      data_nms_sorted_abs >= lower_bound_arr),
                                       data_nms_sorted_med_sign > 0)

        count_arr = np.sum(count_idx_arr, axis=1)
        count_opp_sign_arr = np.sum(np.logical_and(data_nms_sorted_med_sign < 0,
                                                   np.logical_not(count_idx_arr)),
                                    axis=1)

        # Start writing values based on conditions

        consistency_bool_val_arr = np.full(shape=valid_obs_useful.shape, fill_value=False)
        consistency_median_val_arr = np.full(shape=valid_obs_useful.shape, fill_value=-1.0)

        set_consistency_idx_1 = count_arr >= passing_window_size
        set_consistency_idx_2 = np.logical_and(count_arr > first_loose_passing_window, count_opp_sign_arr <= 1)
        set_consistency_idx_3 = np.logical_and(count_arr > second_loose_passing_window, count_opp_sign_arr == 0)

        set_consistency_idx = np.logical_and(np.logical_or(np.logical_or(set_consistency_idx_1, set_consistency_idx_2),
                                                           set_consistency_idx_3),
                                             np.logical_not(median_value_arr == 0))

        consistency_bool_val_arr[set_consistency_idx] = True
        consistency_median_val_arr[set_consistency_idx] = median_value_arr[set_consistency_idx]

        # Extract values from overall matrices that apply only to the days needed

        data_smart_union_useful = data_smart_union[useful_obs_idx, :]
        val_sign_arr_useful = val_sign_arr[useful_obs_idx, :]
        data_nms_useful = data_nms_padded[useful_obs_idx, :]
        data_nms_raw_useful = data_nms_raw_padded[useful_obs_idx, :]
        median_value_useful = median_val_array[useful_obs_idx, :]
        bool_value_useful = bool_val_array[useful_obs_idx, :]

        values_set_idx_useful = values_set_idx_array[useful_obs_idx, :]

        valid_bool_val_arr = np.full(shape=valid_obs_useful.shape, fill_value=False)
        valid_median_val_arr = np.full(shape=valid_obs_useful.shape, fill_value=-1.0)

        pos_avg_col_idx = pos_avg_col >= 0.5
        neg_avg_col_idx = neg_avg_col >= 0.5

        middle_valid_arr = np.logical_or(np.logical_and(pos_avg_col_idx, val_sign_arr_useful > 0),
                                         np.logical_and(neg_avg_col_idx, val_sign_arr_useful < 0))

        set_valid_idx_1 = np.logical_and(middle_valid_arr, consistency_bool_val_arr)

        valid_bool_val_arr[set_valid_idx_1] = True
        valid_median_val_arr[set_valid_idx_1] = consistency_median_val_arr[set_valid_idx_1]

        # Left valid

        not_set_valid_idx = np.multiply(np.roll(data_smart_union_useful, 1, axis=1), val_sign_arr_useful) > 0
        skip_idx_valid = np.logical_or(set_valid_idx_1, not_set_valid_idx)

        left_valid_arr = np.logical_or(np.logical_and(np.roll(pos_avg_col_idx, 1, axis=1), val_sign_arr_useful > 0),
                                       np.logical_and(np.roll(neg_avg_col_idx, 1, axis=1), val_sign_arr_useful < 0))

        set_valid_idx_2 = np.logical_and(np.logical_and(left_valid_arr, np.logical_not(skip_idx_valid)),
                                         np.roll(consistency_bool_val_arr, 1, axis=1))

        valid_bool_val_arr[set_valid_idx_2] = True
        valid_median_val_arr[set_valid_idx_2] = np.roll(consistency_median_val_arr, 1, axis=1)[set_valid_idx_2]

        # Right valid

        skip_idx_valid = np.logical_or(skip_idx_valid, set_valid_idx_2)

        right_valid_arr = np.logical_or(np.logical_and(np.roll(pos_avg_col_idx, -1, axis=1), val_sign_arr_useful > 0),
                                        np.logical_and(np.roll(neg_avg_col_idx, -1, axis=1), val_sign_arr_useful < 0))

        set_valid_idx_3 = np.logical_and(np.logical_and(right_valid_arr, np.logical_not(skip_idx_valid)),
                                         np.roll(consistency_bool_val_arr, -1, axis=1))

        set_valid_idx_3 = \
            np.logical_and(
                np.logical_and(
                    np.multiply(np.roll(data_nms_useful, -1, axis=1), val_sign_arr_useful) == 0,
                    np.multiply(np.roll(data_nms_raw_useful, -1, axis=1), val_sign_arr_useful) == 0),
                set_valid_idx_3)

        valid_bool_val_arr[set_valid_idx_3] = True
        valid_median_val_arr[set_valid_idx_3] = np.roll(consistency_median_val_arr, -1, axis=1)[set_valid_idx_3]

        valid_median_val_arr[values_set_idx_useful] = median_value_useful[values_set_idx_useful]
        valid_bool_val_arr[values_set_idx_useful] = False

        bool_val_array[useful_obs_idx, :] = np.logical_or(valid_bool_val_arr, bool_value_useful)
        median_val_array[useful_obs_idx, :] = valid_median_val_arr

        values_set_idx_useful = np.logical_or(np.logical_or(np.logical_or(values_set_idx_useful, set_valid_idx_1),
                                                            set_valid_idx_2),
                                              set_valid_idx_3)
        values_set_idx_array[useful_obs_idx, :] = values_set_idx_useful

    return bool_val_array, median_val_array


def smart_union(data_nms, data_nms_raw, pp_config):
    """
    Parameters:
        data_nms            (np.ndarray)        : Day wise data matrix after nms
        data_nms_raw        (np.ndarray)        : Day wise data matrix after nms on raw data
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump

    Returns:
        data_smart_union    (np.ndarray)        : Day wise data after smart union
    """

    # Copy the input matrices to preserve values from previous step

    data_nms = copy.deepcopy(data_nms)
    data_nms_raw = copy.deepcopy(data_nms_raw)

    # Initialise smart union and get abs value for nms matrices

    num_rows, num_cols = data_nms.shape

    # Get padded Arrays

    data_nms_padded = np.zeros(shape=(num_rows, num_cols + 2))
    data_nms_padded[:, 1: -1] = data_nms
    data_nms_padded[1:, 0] = data_nms[0: -1, -1]
    data_nms_padded[0: -1, -1] = data_nms[1:, 0]

    data_nms_raw_padded = np.zeros(shape=(num_rows, num_cols + 2))
    data_nms_raw_padded[:, 1: -1] = data_nms_raw
    data_nms_raw_padded[1:, 0] = data_nms_raw[0: -1, -1]
    data_nms_raw_padded[0: -1, -1] = data_nms_raw[1:, 0]

    data_smart_union = np.zeros(shape=data_nms_padded.shape)

    data_nms_abs = np.abs(data_nms_padded)
    data_nms_raw_abs = np.abs(data_nms_raw_padded)

    val_sign_arr = np.sign(data_nms_raw_padded)

    num_rows_padded = num_rows
    num_cols_padded = num_cols + 2

    # Case A : NMS values in the region of zero limit and max value

    case_a_idx = np.logical_and(data_nms_abs > pp_config.get('zero_val_limit'),
                                data_nms_abs <= pp_config.get('smart_union_max_val'))

    data_smart_union[case_a_idx] = data_nms_padded[case_a_idx]

    # Prepare 3d data to run retain in nms

    history_window = pp_config.get('smart_union_history_window')
    num_z_idx = 2 * history_window

    data_nms_3d = np.full(shape=(num_rows_padded, num_z_idx + 1, num_cols_padded), fill_value=np.nan)
    data_nms_raw_3d = np.full(shape=(num_rows_padded, num_z_idx + 1, num_cols_padded), fill_value=np.nan)

    for idx in range(num_rows_padded):
        start_idx = max(0, idx - history_window)
        end_idx = min(num_rows_padded, idx + history_window + 1)

        data_nms_3d[idx, :(end_idx - start_idx), :] = data_nms_padded[start_idx: end_idx, :]
        data_nms_raw_3d[idx, :(end_idx - start_idx), :] = data_nms_raw_padded[start_idx: end_idx, :]

    # Run retain in nms for NMS

    bool_val_array, median_val_array = retain_in_nms(data_nms_3d, data_nms_padded, data_nms_raw_padded, val_sign_arr,
                                                     data_smart_union, pp_config)

    not_set_in_nms = np.logical_not(bool_val_array)

    bool_val_array_raw, median_val_array_raw = retain_in_nms(data_nms_raw_3d, data_nms_raw_padded, data_nms_padded,
                                                             val_sign_arr, data_smart_union, pp_config)

    bool_val_array[not_set_in_nms] = bool_val_array_raw[not_set_in_nms]
    median_val_array[not_set_in_nms] = median_val_array_raw[not_set_in_nms]

    # Case B : Raw NMS values in range zero limit and max value, NMS before and after with sign is less than zero limit

    case_b_idx = np.logical_and(np.logical_and(data_nms_raw_abs > pp_config.get('zero_val_limit'),
                                               data_nms_raw_abs <= pp_config.get('smart_union_max_val')),
                                np.logical_not(case_a_idx))

    case_b_idx = np.logical_and(np.logical_and(np.multiply(np.roll(data_nms_padded, 1, axis=1), val_sign_arr) <=
                                               pp_config.get('zero_val_limit'),
                                               np.multiply(np.roll(data_nms_padded, -1, axis=1), val_sign_arr) <=
                                               pp_config.get('zero_val_limit')),
                                case_b_idx)

    bool_val_array = np.logical_and(case_b_idx, bool_val_array)

    row_indices, col_indices = np.where(bool_val_array)
    num_indices = len(row_indices)

    minimum_median_threshold = pp_config.get('minimum_median_threshold')
    median_threshold_fraction = pp_config.get('median_threshold_fraction')

    for idx in range(num_indices):

        row_idx = row_indices[idx]
        col_idx = col_indices[idx]

        val_sign = val_sign_arr[row_idx, col_idx]
        median_value = median_val_array[row_idx, col_idx]

        modified_median_value, col_idx_return = fill_median_value(data_nms_padded, row_idx, col_idx, val_sign)

        if modified_median_value == -1:
            modified_median_value, col_idx_return = fill_median_value(data_nms_raw_padded, row_idx, col_idx, val_sign)

        if col_idx_return == col_idx:
            minimum_median_threshold /= 1.5

        median_value = max(abs(median_value), abs(modified_median_value))
        median_value *= val_sign

        if median_value == -1 or \
                abs(data_nms_raw_padded[row_idx, col_idx]) <= abs(minimum_median_threshold * median_value):
            continue

        if median_value * (1 + median_threshold_fraction) >= data_nms_raw_padded[row_idx, col_idx] >= \
                median_value * (1 - median_threshold_fraction):
            data_smart_union[row_idx, col_idx] = data_nms_raw_padded[row_idx, col_idx]
        else:
            data_smart_union[row_idx, col_idx] = median_value

    # Remove padding from data smart union

    data_smart_union = data_smart_union[:, 1: -1]

    return data_smart_union
