"""
Author - Mayank Sharan
Date - 29/8/19
Convert 21 column data to day wise data and remove LAP thin WH pulses
"""

# Import python modules

import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_day_data(input_data, input_data_pp_masked, valid_mask_cons_bool, peak_amp_arr, peak_removal_bool,
                 vacation_config, logger_pass):

    """
    Converts 1d input data to day wise 2d matrix

    Parameters:
        input_data              (np.ndarray)        : 21 column input data
        input_data_pp_masked    (np.ndarray)        : 21 column input data with pool pump masked from consumption
        valid_mask_cons_bool    (np.ndarray)        : Array containing boolean indicating points masked as pp
        peak_amp_arr            (np.ndarray)        : Amplitude array result of convolution with thin peak filter
        peak_removal_bool       (np.ndarray)        : Array containing boolean where LAP WH peaks are present
        vacation_config         (dict)              : Contains parameters needed for vacation detection
        logger_pass             (dict)              : Contains the logger and the logging dictionary to be passed on

    Returns:
        return_dict             (dict)              : Dictionary containing all day wise matrices to be returned
    """

    # Initialize vacation logger

    logger_base = logger_pass.get('base_logger').getChild('get_day_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Prepare logger pass to pass to further sub-modules

    logger_pass['base_logger'] = logger_base

    # Get the timestamps for each day and initialize day wise matrices

    sampling_rate = vacation_config.get('user_info').get('sampling_rate')

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    # Prepare day timestamp matrix and get size of all matrices

    day_ts, row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
    day_ts = np.tile(day_ts, reps=(num_pd_in_day, 1)).transpose()

    # Initialize all 2d matrices with default value of nan except the boolean ones

    month_ts = np.full(shape=day_ts.shape, fill_value=np.nan)
    epoch_ts = np.full(shape=day_ts.shape, fill_value=np.nan)
    day_data = np.full(shape=day_ts.shape, fill_value=np.nan)
    day_data_masked = np.full(shape=day_ts.shape, fill_value=np.nan)

    day_valid_mask_cons = np.full(shape=day_ts.shape, fill_value=False)
    day_peak_amp_arr = np.full(shape=day_ts.shape, fill_value=np.nan)
    day_peak_removal = np.full(shape=day_ts.shape, fill_value=False)

    logger.debug("Shape of converted 2d day wise data is | %s", str(day_data_masked.shape))

    # Compute hour of day based indices to use

    col_idx = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[:, Cgbdisagg.INPUT_DAY_IDX]
    col_idx = col_idx / Cgbdisagg.SEC_IN_HOUR
    col_idx = (pd_mult * (col_idx - col_idx.astype(int) + input_data[:, Cgbdisagg.INPUT_HOD_IDX])).astype(int)

    # Create day wise 2d arrays for each variable

    day_data[row_idx, col_idx] = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    month_ts[row_idx, col_idx] = input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    epoch_ts[row_idx, col_idx] = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    day_data_masked[row_idx, col_idx] = input_data_pp_masked[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    day_valid_mask_cons[row_idx, col_idx] = valid_mask_cons_bool
    day_peak_amp_arr[row_idx, col_idx] = peak_amp_arr
    day_peak_removal[row_idx, col_idx] = peak_removal_bool

    # Populate the return dictionary

    return_dict = {
        'day_ts': day_ts,
        'day_data': day_data,
        'month_ts': month_ts,
        'epoch_ts': epoch_ts,
        'day_data_masked': day_data_masked,
        'day_peak_amp_arr': day_peak_amp_arr,
        'day_peak_removal': day_peak_removal,
        'day_valid_mask_cons': day_valid_mask_cons,
    }

    return return_dict
