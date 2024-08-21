"""
Author - Mayank Sharan
Date - 19/1/19
Apply non maximal suppression on the input data
"""

# Import python packages

import copy
import numpy as np


def get_consistency(matrix, row, col, consistency_window_size=10):
    """Utility to get consistency value"""

    threshold_fraction = 1 / consistency_window_size
    half_window_size = int(consistency_window_size / 2)

    count_lower = get_count_of_comparisons(matrix, row, col, row - consistency_window_size + 1, row + 1,
                                           threshold_fraction)

    count_upper = get_count_of_comparisons(matrix, row, col, row, row + consistency_window_size,
                                           threshold_fraction)

    count_middle = get_count_of_comparisons(matrix, row, col, row - half_window_size, row + half_window_size,
                                            threshold_fraction)

    return max(count_lower, count_upper, count_middle)


def get_count_of_comparisons(matrix, row, col, lower_idx, upper_idx, threshold_fraction):
    """Utility to get count of comp"""

    consistency_fraction = 0.25
    margin = consistency_fraction * matrix[row, col]
    upper_bound = matrix[row, col] + margin
    lower_bound = matrix[row, col] - margin

    # number of rows
    N = matrix.shape[0]

    # Find the column indices in matrix [row, 0/col-window_size : col+window_size/N-1] for comparison
    if lower_idx < 0:
        lower_idx = 0
    if upper_idx > (N - 1):
        upper_idx = N - 1

    required_vec = matrix[lower_idx:upper_idx, col]
    count = np.sum(np.logical_and(np.abs(required_vec) >= np.abs(lower_bound),
                                  np.abs(required_vec) <= np.abs(upper_bound)))

    count *= threshold_fraction
    return count


def non_maximal_suppression(data_grad, zero_val_limit, pp_config):
    """
    Parameters:
        data_grad           (np.ndarray)        : Day wise data matrix
        zero_val_limit      (int)               : Value below which everything will be set to zero
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump

    Returns:
        data_nms            (np.ndarray)        : Day wise data after nms calculation
    """

    # Set values in data grad below threshold to 0

    data_grad = copy.deepcopy(data_grad)
    data_grad_abs = np.abs(data_grad)
    data_grad[data_grad_abs < zero_val_limit] = 0

    # Create left val matrix

    left_val = np.zeros(shape=data_grad.shape)
    left_val[:, 1:] = data_grad[:, :-1]
    left_val[1:, 0] = data_grad[:-1, -1]
    left_val[0, 0] = left_val[0, 1]
    left_val_abs = np.abs(left_val)

    # Create right val matrix

    right_val = np.zeros(shape=data_grad.shape)
    right_val[:, :-1] = data_grad[:, 1:]
    right_val[:-1, -1] = data_grad[1:, 0]
    right_val[-1, -1] = right_val[-1, -2]
    right_val_abs = np.abs(right_val)

    # Prepare the signs for data matrix

    pos_idx = data_grad > 0
    neg_idx = data_grad < 0

    # Prepare the signs for left val matrix

    pos_idx_left = left_val > 0
    neg_idx_left = left_val < 0

    # Set signs based on centre val for zeros

    zero_idx_left = left_val == 0
    pos_idx_left[zero_idx_left] = pos_idx[zero_idx_left]
    neg_idx_left[zero_idx_left] = neg_idx[zero_idx_left]

    # Prepare the signs for left val matrix

    pos_idx_right = right_val > 0
    neg_idx_right = right_val < 0

    # Set signs based on centre val for zeros

    zero_idx_right = right_val == 0
    pos_idx_right[zero_idx_right] = pos_idx[zero_idx_right]
    neg_idx_right[zero_idx_right] = neg_idx[zero_idx_right]

    # Initialize the nms matrix

    data_nms = np.zeros(shape=data_grad.shape)

    # Populate the nms matrix based on conditions

    # Case B : Condition for left and right having same sign but middle having the opposite

    case_b_idx = np.logical_or(np.logical_and(np.logical_and(pos_idx_left == pos_idx_right, pos_idx_left), neg_idx),
                               np.logical_and(np.logical_and(neg_idx_left == neg_idx_right, neg_idx_left), pos_idx))

    data_nms[case_b_idx] = data_grad[case_b_idx]

    # Case C : Left and middle have same sign but right has opposite sign and abs val is not less than abs left

    case_c_idx = np.logical_and(np.logical_or(np.logical_and(np.logical_and(pos_idx_left == pos_idx, pos_idx),
                                                             neg_idx_right),
                                              np.logical_and(np.logical_and(neg_idx_left == neg_idx, neg_idx),
                                                             pos_idx_right)),
                                data_grad_abs >= left_val_abs)

    data_nms[case_c_idx] = data_grad[case_c_idx]

    # Case D : Right and middle have same sign but left has opposite sign and abs val is not less than abs right

    case_d_idx = np.logical_and(np.logical_or(np.logical_and(np.logical_and(pos_idx_right == pos_idx, pos_idx),
                                                             neg_idx_left),
                                              np.logical_and(np.logical_and(neg_idx_right == neg_idx, neg_idx),
                                                             pos_idx_left)),
                                data_grad_abs >= right_val_abs)

    data_nms[case_d_idx] = data_grad[case_d_idx]

    # Case A : All three have the same sign

    case_a_idx = np.logical_or(np.logical_and(np.logical_and(pos_idx_left == pos_idx_right, pos_idx_left), pos_idx),
                               np.logical_and(np.logical_and(neg_idx_left == neg_idx_right, neg_idx_left), neg_idx))

    # Case A1 : Middle val is greater in magnitude than other two

    case_a_1_idx = np.logical_and(np.logical_and(data_grad_abs > left_val_abs, data_grad_abs > right_val_abs),
                                  case_a_idx)

    data_nms[case_a_1_idx] = data_grad[case_a_1_idx]

    # The cases that happen rarely so we will do it with a for loop

    consistency_window_size = pp_config.get('consistency_window_size_nms')

    # Case A2 : Left and middle are equal and right is smaller

    case_a_2_idx = np.logical_and(np.logical_and(data_grad_abs == left_val_abs, data_grad_abs > right_val_abs),
                                  case_a_idx)

    idx_row, idx_col = np.where(case_a_2_idx)
    num_idx = len(idx_row)

    for idx in range(num_idx):
        row_num = idx_row[idx]
        col_num = idx_col[idx]

        mid_consistency = get_consistency(data_grad, row_num, col_num, consistency_window_size)
        left_consistency = get_consistency(left_val, row_num, col_num, consistency_window_size)

        if mid_consistency >= left_consistency:
            data_nms[row_num, col_num] = data_grad[row_num, col_num]

    # Case A3 : Left and middle are equal and right is smaller

    case_a_3_idx = np.logical_and(np.logical_and(data_grad_abs > left_val_abs, data_grad_abs == right_val_abs),
                                  case_a_idx)

    idx_row, idx_col = np.where(case_a_3_idx)
    num_idx = len(idx_row)

    for idx in range(num_idx):
        row_num = idx_row[idx]
        col_num = idx_col[idx]

        mid_consistency = get_consistency(data_grad, row_num, col_num, consistency_window_size)
        right_consistency = get_consistency(right_val, row_num, col_num, consistency_window_size)

        if mid_consistency >= right_consistency:
            data_nms[row_num, col_num] = data_grad[row_num, col_num]

    return data_nms
