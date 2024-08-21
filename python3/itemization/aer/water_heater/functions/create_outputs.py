"""
Author - Sahana M
Date - 9/3/2021
This file contains the functions used to create Time stamp & Bill cycle level output
"""

# Import python packages
import numpy as np
from copy import deepcopy
from datetime import datetime

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg


def create_ts_output(input_data, debug):
    """
    Function used to create time stamp level output
    Parameters:
        input_data          (np.ndarray)        : 21 column input data
        debug               (dict)              : Contains all variables required for debugging
    Returns:
        swh_ts_estimate     (np.ndarray)        : Contains 2 column array with time stamp & consumption
        final_estimate      (np.ndarray)        : Contains 21 column matrix with Seasonal WH estimation output
    """

    # Create time stamp level output

    if len(input_data) == len(debug.get('final_estimation').flatten()):
        swh_estimate = debug.get('final_estimation').flatten()

    # For data with discontinuous data points map estimation values to their corresponding time stamps

    else:
        swh_estimation_matrix = debug.get('final_estimation')
        row_idx = debug.get('row_idx')
        col_idx = debug.get('col_idx')
        swh_estimate = swh_estimation_matrix[row_idx, col_idx].flatten()

    # Assign the swh estimate values to the final estimate array

    final_estimate = deepcopy(input_data)
    final_estimate[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = swh_estimate

    # Final ime stamp & Consumption array

    swh_ts_estimate = final_estimate[:, [Cgbdisagg.INPUT_EPOCH_IDX, Cgbdisagg.INPUT_CONSUMPTION_IDX]]
    final_estimate = final_estimate[:, :]

    return swh_ts_estimate, final_estimate


def create_monthly_output(final_estimate):
    """
    Returns bill cycle level output
    Parameters:
        final_estimate              (np.ndarray)            : Contains 21 column output matrix
    Returns:
        final_monthly_estimate      (np.ndarray)            : Contains 2 columns bill cycle & consumption array
        monthly_output_log          (dict)                  : Bill cycle consumption in Month-Year format
    """

    # Get the bill cycles time stamp

    bill_cycle_idx = final_estimate[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]

    # Get the unique bill cycles & their indexes

    bill_cycle, bc_index = np.unique(bill_cycle_idx, return_inverse=True)

    # Get the total number of bill cycles

    indexes = np.unique(bc_index)

    # Create an empty 2d array for writing bill cycle & their consumption

    final_monthly_estimate = np.full(shape=(bill_cycle.shape[0], 2), fill_value=0.0)

    # Iterate over each bill cycle estimate and aggregate

    for i in indexes:
        bc_ts_indexes_bool = bc_index == i
        bc_estimate = np.nansum(final_estimate[bc_ts_indexes_bool, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        final_monthly_estimate[i][0] = bill_cycle[i]
        final_monthly_estimate[i][1] = bc_estimate

    # Map the epoch Bill cycle to Month-Year format for logging

    monthly_output_log = [
        (datetime.utcfromtimestamp(final_monthly_estimate[i, 0]).strftime('%b-%Y'), final_monthly_estimate[i, 1])
        for i in range(final_monthly_estimate.shape[0])]

    return final_monthly_estimate, monthly_output_log
