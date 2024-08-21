"""
Author - Sahana M
Date - 07/05/2021
This function is used to get the boxes from the input data
"""

# Import python packages
import numpy as np


def get_box_data(filtered_data, window_size):
    """
    This function is used to get the boxes from the input data
    Parameters:
        filtered_data       (np.ndarray)    : Filtered data
        window_size         (int)           : Window size
    Returns:
        box_info            (np.ndarray)    : Box info
        box_val             (np.ndarray)    : Get min and max
    """

    num_cols = filtered_data.shape[1]

    row_idx_matrix = np.full_like(filtered_data, fill_value=0.0)
    col_idx_matrix = np.full_like(filtered_data, fill_value=0.0)

    for i in range(row_idx_matrix.shape[0]):
        row_idx_matrix[i, :] = i

    for i in range(col_idx_matrix.shape[1]):
        col_idx_matrix[:, i] = i

    # Flatten the input data

    temp = filtered_data.flatten()
    row_idx = row_idx_matrix.flatten()
    col_idx = col_idx_matrix.flatten()

    # Initialise an empty array

    shifting_arr = np.full(shape=(len(temp), window_size), fill_value=0.0)

    # Forward shifting

    for j in range(window_size):
        if j == 0:
            shifting_arr[:, j] = temp
        else:
            shifting_arr[:-j, j] = temp[j:]

    # Get the minimum at each point

    box_val = np.min(shifting_arr, axis=1)

    box_start_bool = box_val > 0
    num_boxes = np.sum(box_start_bool)
    box_info = np.zeros(shape=(num_boxes, 5))
    box_info[:, 0] = row_idx[box_start_bool]
    box_info[:, 1] = col_idx[box_start_bool]
    box_info[:, 2] = row_idx[box_start_bool]
    box_info[:, 3] = box_info[:, 1] + (window_size - 1)
    box_info[:, 4] = box_val[box_start_bool]

    overflow_boxes_bool = box_info[:, 3] >= num_cols

    box_info[overflow_boxes_bool, 2] += 1
    box_info[overflow_boxes_bool, 3] = box_info[overflow_boxes_bool, 3] - num_cols

    # Initialise an empty array

    shifting_arr = np.full(shape=(len(box_val), window_size), fill_value=0.0)

    # Backward shifting

    for j in range(window_size):
        if j == 0:
            shifting_arr[:, j] = box_val
        else:
            shifting_arr[j:, j] = box_val[:-j]

    # Get the maximum at each point

    box_val = np.max(shifting_arr, axis=1)

    return box_info, box_val
