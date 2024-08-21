"""
Author - Mayank Sharan
Date - 12/12/19
Converts day level labels to timestamp level output for further usage
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.utils.maths_utils.find_seq import find_seq

from python3.config.Cgbdisagg import Cgbdisagg


def prepare_vacation_output(vac_label, day_ts, epoch_ts, debug, vacation_config):

    """
    Parameters:
        vac_label               (np.ndarray)        : Array marking each day as type 1 / type 2 vacation or non vacation
        day_ts                  (np.ndarray)        : Day wise array containing the day start timestamp
        epoch_ts                (np.ndarray)        : Day wise array containing the the epoch timestamp
        debug                   (dict)              : Contains all variables needed for debugging
        vacation_config         (dict)              : Contains parameters needed for vacation detection

    Returns:
        debug                   (dict)              : Contains all variables needed for debugging
        type_1_epoch            (np.ndarray)        : Array marking type 1 vacation at epoch level
        type_2_epoch            (np.ndarray)        : Array marking type 2 vacation at epoch level
    """

    # Initialize variables of dimensions

    num_days = epoch_ts.shape[0]
    num_pts_day = epoch_ts.shape[1]

    # Create a 2d arrays and mark days as vacation using day level labels

    day_type_1 = np.full_like(epoch_ts, fill_value=0)
    day_type_1[vac_label == 1, :] = 1

    day_type_2 = np.full_like(epoch_ts, fill_value=0)
    day_type_2[vac_label == 2, :] = 1

    # Convert day level values to 1d epoch level arrays

    epoch_1d = np.reshape(epoch_ts, newshape=(num_days * num_pts_day,))
    type_1_1d = np.reshape(day_type_1, newshape=(num_days * num_pts_day,))
    type_2_1d = np.reshape(day_type_2, newshape=(num_days * num_pts_day,))

    # Create output arrays wth epoch as first column and markings as second

    type_1_epoch = np.c_[epoch_1d, type_1_1d]
    type_2_epoch = np.c_[epoch_1d, type_2_1d]

    # Use only valid values

    valid_bool = np.logical_not(np.isnan(epoch_1d))

    type_1_epoch = type_1_epoch[valid_bool, :]
    type_2_epoch = type_2_epoch[valid_bool, :]

    # Convert vacation labels to vacation periods. Temporary right now. Will be removed before going to production
    # Code below this line is purely here for the purpose of the non dev qa so please do not consider as part of review

    sampling_rate = vacation_config.get('user_info').get('sampling_rate')

    vac_sequence = find_seq(vac_label, min_seq_length=0).astype(int)
    vac_sequence = vac_sequence[vac_sequence[:, 0] > 0]

    vac_start_idx = 0
    vac_end_idx = 1
    vac_type_idx = 2

    vacation_periods = np.zeros(shape=(vac_sequence.shape[0], 3))

    vacation_periods[:, vac_start_idx] = epoch_ts[vac_sequence[:, 1], 0]
    vacation_periods[:, vac_end_idx] = epoch_ts[vac_sequence[:, 2], -1]
    vacation_periods[:, vac_type_idx] = vac_sequence[:, 0]

    day_start_nan = np.isnan(vacation_periods[:, vac_start_idx])
    vacation_periods[day_start_nan, vac_start_idx] = day_ts[vac_sequence[day_start_nan, 1], 0]

    # This part is buggy it basically leaves out 1 hour from the day or adds an extra hour from next day in case of DST

    day_end_nan = np.isnan(vacation_periods[:, vac_end_idx])
    vacation_periods[day_end_nan, vac_end_idx] = \
        day_ts[vac_sequence[day_end_nan, 2], 0] + Cgbdisagg.SEC_IN_DAY - sampling_rate

    debug['vacation_periods'] = vacation_periods

    return debug, type_1_epoch, type_2_epoch
