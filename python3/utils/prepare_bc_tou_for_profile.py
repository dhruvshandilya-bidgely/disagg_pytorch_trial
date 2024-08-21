"""
Author - Mayank Sharan
Date - 14-Jun-2021
Function to prepare tou field for appliance profile for a given list of bill cycles
"""

# Import functions from within the project

import numpy as np

# Import packages from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def prepare_bc_tou_for_profile(input_data, appliance_1d_est, out_bill_cycles):

    """
    For each bill cycle in out_bill_cycles create an hourly array to be filled in TOU for appliance profile

    Parameters:
        input_data          (np.ndarray)            : 21 column matrix for reference
        appliance_1d_est    (np.ndarray)            : 1d timestamp level estimate for the appliance
        out_bill_cycles     (np.ndarray)            : Bill cycles for which output is to be given

    Returns:
        tou_by_bc_dict      (np.ndarray)            : Dictionary containing tou profile for each bill cycle
        success             (bool)                  : Boolean indicating if the data is correct for generation
    """

    # Initialize variables needed and check validity of parameters

    tou_by_bc_dict = {}

    if not out_bill_cycles.tolist() or not(input_data.shape[0] == len(appliance_1d_est)) or \
            len(appliance_1d_est.shape) != 1:
        return tou_by_bc_dict, False

    appliance_1d_est[np.isnan(appliance_1d_est)] = 0

    # For each bill cycle in out bill cycles create an hourly tou profile

    for bill_idx in range(out_bill_cycles.shape[0]):
        curr_bc_start = out_bill_cycles[bill_idx, 0]

        # Filter the data for the current bill cycle

        curr_bc_bool = input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == curr_bc_start

        curr_input_data = input_data[curr_bc_bool, :]
        curr_app_1d_estimate = appliance_1d_est[curr_bc_bool]

        bin_idx = curr_input_data[:, Cgbdisagg.INPUT_HOD_IDX].astype(int)
        bc_hr_profile = np.bincount(bin_idx, weights=curr_app_1d_estimate)

        total_cons = np.sum(bc_hr_profile)

        if total_cons > 0:
            bc_hr_profile = np.round(bc_hr_profile / total_cons, 4)

        tou_by_bc_dict[curr_bc_start] = bc_hr_profile

    return tou_by_bc_dict
