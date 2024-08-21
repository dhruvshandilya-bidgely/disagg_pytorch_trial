"""
Author - Sahana M
Date - 07/12/2021
This function is used to convert a ndarray info about all the best fit boxes into its 2D format
"""

# Import python packages
import numpy as np

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import box_indexes


def box_data_to_2d(box_info, rows, cols):
    """
    This function is used to convert a ndarray info about all the best fit boxes into its 2D format
    Parameters:
        box_info                (np.ndarray): Best box candidates for the current day chunk
        rows                    (int)       : Number of days in the data
        cols                    (int)       : Number of time divisions

    Returns:
        box_2d                  (np.ndarray): Contains 2D matrix of box fit data
    """

    box_2d = np.zeros(shape=(rows, cols))

    # For each row compute the dynamic box fitting

    for i in range(box_info.shape[0]):
        current_box_info = box_info[i, :]
        current_amp = current_box_info[box_indexes['amplitude']]

        # If the box is present in the same day

        if current_box_info[box_indexes['start_row']] == current_box_info[box_indexes['end_row']]:
            day_idx = int(current_box_info[box_indexes['start_row']])
            start_idx = int(current_box_info[box_indexes['start_col']])
            end_idx = int(current_box_info[box_indexes['end_col']]+1)
            box_2d[day_idx, start_idx:end_idx] = np.fmax(current_amp, box_2d[day_idx, start_idx:end_idx])

        # If the box is continued to the next day

        else:
            day_idx_1 = int(current_box_info[box_indexes['start_row']])
            start_idx_1 = int(current_box_info[box_indexes['start_col']])
            end_idx_1 = cols
            day_idx_2 = int(current_box_info[box_indexes['end_row']])
            start_idx_2 = 0
            end_idx_2 = int(current_box_info[box_indexes['end_col']] + 1)
            box_2d[day_idx_1, start_idx_1:end_idx_1] = np.fmax(current_amp, box_2d[day_idx_1, start_idx_1:end_idx_1])
            box_2d[day_idx_2, start_idx_2:end_idx_2] = np.fmax(current_amp, box_2d[day_idx_2, start_idx_2:end_idx_2])

    return box_2d
