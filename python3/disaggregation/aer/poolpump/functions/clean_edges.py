"""
Author - Mayank Sharan
Date - 20/1/19
Merge edges and perform cleaning
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.disaggregation.aer.poolpump.functions.cleaning_utils import find_edges
from python3.disaggregation.aer.poolpump.functions.cleaning_utils import smart_merge
from python3.disaggregation.aer.poolpump.functions.cleaning_utils import smart_merge_both
from python3.disaggregation.aer.poolpump.functions.cleaning_utils import should_be_masked
from python3.disaggregation.aer.poolpump.functions.get_padded_signal import get_padded_signal
from python3.disaggregation.aer.poolpump.functions.cleaning_utils import get_consistency_array
from python3.disaggregation.aer.poolpump.functions.cleaning_utils import delete_insufficient_edges
from python3.disaggregation.aer.poolpump.functions.cleaning_utils import get_consistency_array_strength


def merge_adjacent_edges(data_smart_union_part, pp_config):
    """Utility to merge adjacent edges"""

    # Get padded signal from the input

    data_padded = get_padded_signal(data_smart_union_part, n_rows_pad=0, n_cols_pad=1)

    # Compute initial consistency array for each column

    consistency_array = get_consistency_array(data_padded, pp_config)
    consistency_col_sum = np.sum(consistency_array, axis=0)

    consistency_array_strength = get_consistency_array_strength(data_padded, pp_config)

    # Loop over columns and perform merge

    for col_idx in range(1, data_padded.shape[1] - 1):

        left_sum = consistency_col_sum[col_idx - 1]
        right_sum = consistency_col_sum[col_idx + 1]

        if left_sum > 0 and right_sum == 0:
            data_padded = smart_merge(data_padded, col_idx, consistency_array_strength[:, col_idx], col_idx - 1,
                                      consistency_array_strength[:, col_idx - 1], pp_config)

            # Reset consistency array and sum for modified array

            col_idx_modified = [col_idx - 1, col_idx]
            modified_cons_array = get_consistency_array(data_padded[:, col_idx_modified], pp_config)
            consistency_array[:, col_idx_modified] = modified_cons_array
            consistency_col_sum[col_idx_modified] = np.sum(modified_cons_array, axis=0)

            modified_cons_array_strength = get_consistency_array_strength(data_padded[:, col_idx_modified], pp_config)
            consistency_array_strength[:, col_idx_modified] = modified_cons_array_strength

        elif right_sum > 0 and left_sum == 0:
            data_padded = smart_merge(data_padded, col_idx, consistency_array_strength[:, col_idx], col_idx + 1,
                                      consistency_array_strength[:, col_idx + 1], pp_config)

            # Reset consistency array and sum for modified array

            col_idx_modified = [col_idx, col_idx + 1]
            modified_cons_array = get_consistency_array(data_padded[:, col_idx_modified], pp_config)
            consistency_array[:, col_idx_modified] = modified_cons_array
            consistency_col_sum[col_idx_modified] = np.sum(modified_cons_array, axis=0)

            modified_cons_array_strength = get_consistency_array_strength(data_padded[:, col_idx_modified], pp_config)
            consistency_array_strength[:, col_idx_modified] = modified_cons_array_strength

        elif right_sum > 0 and left_sum > 0:
            data_padded = smart_merge_both(data_padded, col_idx - 1, col_idx, col_idx + 1,
                                           consistency_array_strength[:, col_idx - 1: col_idx + 2], pp_config)

            # Reset consistency array and sum for modified array

            col_idx_modified = [col_idx - 1, col_idx, col_idx + 1]
            modified_cons_array = get_consistency_array(data_padded[:, col_idx_modified], pp_config)
            consistency_array[:, col_idx_modified] = modified_cons_array
            consistency_col_sum[col_idx_modified] = np.sum(modified_cons_array, axis=0)

            modified_cons_array_strength = get_consistency_array_strength(data_padded[:, col_idx_modified], pp_config)
            consistency_array_strength[:, col_idx_modified] = modified_cons_array_strength

    data_padded[:, 1] = np.maximum(data_padded[:, 1], data_padded[:, -1])
    data_padded[:, -2] = np.maximum(data_padded[:, -2], data_padded[:, 0])

    # Delete edges as necessary

    col_idx_modified = [1, -2]
    consistency_array_del = consistency_array_strength
    consistency_array_del_modified = get_consistency_array_strength(data_padded[:, col_idx_modified], pp_config)
    consistency_array_del[:, col_idx_modified] = consistency_array_del_modified

    for col_idx in range(1, data_padded.shape[1] - 1):
        data_padded[:, col_idx] = delete_insufficient_edges(data_padded[:, col_idx], consistency_array_del[:, col_idx],
                                                            pp_config)

    return data_padded[:, 1:-1]


def get_uniform_data(data_merged, pp_config):
    """Utility to get uniform data"""

    # Extract constants from the config

    zero_bin_size = pp_config.get('zero_val_limit')
    consistency_margin = pp_config.get('cleaning_consistency_margin')

    # Run the get uniform data

    consistency_array = get_consistency_array_strength(data_merged, pp_config)
    consistency_array_sum = np.sum(consistency_array, axis=0)

    col_idx_iter = np.where(np.logical_not(consistency_array_sum == 0))[0]
    num_rows = data_merged.shape[0]

    for col_idx in col_idx_iter:

        consistency_start_arr, consistency_end_arr = find_edges(consistency_array[:, col_idx])
        num_idx_arr = len(consistency_start_arr)

        for idx in range(num_idx_arr):
            start_idx = max(consistency_start_arr[idx] - consistency_margin, 0)
            end_idx = min(consistency_end_arr[idx] + consistency_margin, num_rows)

            required_data = data_merged[start_idx:end_idx, col_idx]
            to_be_retained = np.where(required_data > zero_bin_size)[0]

            median_val = np.median(required_data[to_be_retained])
            to_be_retained += start_idx

            data_merged[to_be_retained, col_idx] = median_val

    return data_merged


def remove_weak_time_divisions(data_merged, pp_config):
    """Utility to remove weak time divisions"""

    # Extract constants from config

    best_time_div_num = pp_config.get('cleaning_best_time_div_num')
    amplitude_fraction = pp_config.get('cleaning_amplitude_fraction')
    minimum_unmasked_length = pp_config.get('cleaning_minimum_unmasked_length')

    # Initialise values to be used

    num_rows, num_cols = data_merged.shape

    # Fill mask amplitudes

    col_amp = np.zeros(num_cols)

    for col_idx in range(num_cols):

        bool_val, strength = should_be_masked(data_merged[:, col_idx], pp_config)

        if bool_val:
            col_amp[col_idx] = strength

    # If no masking needed return value

    if len(np.where(col_amp > 0)[0]) < 2:
        return data_merged

    # Get median values for best 2 columns

    amplitude = []
    best_idx = np.argsort(-col_amp)[:best_time_div_num]

    for col_idx in best_idx:
        start_arr, end_arr = find_edges(data_merged[:, col_idx])
        diff = end_arr - start_arr

        idx = np.argmax(diff)
        amplitude.append(np.median(data_merged[start_arr[idx]:end_arr[idx], col_idx]))

    # Mask time divisions based on threshold

    threshold_amp = amplitude_fraction * np.min(amplitude)
    unmasked_col = np.where(col_amp == 0)[0]

    for col_idx in unmasked_col:

        start_arr, end_arr = find_edges(data_merged[:, col_idx])
        num_idx = len(start_arr)

        for idx in range(num_idx):
            if (np.median(data_merged[start_arr[idx]: end_arr[idx], col_idx]) < threshold_amp) or \
                    (end_arr[idx] - start_arr[idx] < minimum_unmasked_length):
                data_merged[start_arr[idx]:end_arr[idx], col_idx] = 0

    return data_merged


def clean_edges(data_smart_union, pp_config):
    """
    Parameters:
        data_smart_union    (np.ndarray)        : Day wise data matrix after smart union
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump

    Returns:
        data_clean_edges    (np.ndarray)        : Day wise data after merging edges
    """

    # Initialise positive and negative arrays

    data_smart_union_pos = copy.deepcopy(data_smart_union)
    data_smart_union_neg = copy.deepcopy(data_smart_union)

    data_smart_union_pos[data_smart_union_pos < 0] = 0
    data_smart_union_neg[data_smart_union_neg > 0] = 0
    data_smart_union_neg = np.abs(data_smart_union_neg)

    # Merge adjacent edges on both positive and negative data

    data_pos_merged = merge_adjacent_edges(data_smart_union_pos, pp_config)
    data_neg_merged = merge_adjacent_edges(data_smart_union_neg, pp_config)

    # Get uniform amplitude on both positive and negative data

    data_pos_merged = get_uniform_data(data_pos_merged, pp_config)
    data_neg_merged = get_uniform_data(data_neg_merged, pp_config)

    # Remove weak time divisions

    data_pos_merged = remove_weak_time_divisions(data_pos_merged, pp_config)
    data_neg_merged = remove_weak_time_divisions(data_neg_merged, pp_config)

    # Combine positive and negative arrays to get final array

    pos_idx = data_pos_merged > data_neg_merged
    neg_idx = data_neg_merged > data_pos_merged

    data_pos_merged[neg_idx] = 0
    data_neg_merged[pos_idx] = 0

    data_clean_edges = data_pos_merged - data_neg_merged

    return data_clean_edges
