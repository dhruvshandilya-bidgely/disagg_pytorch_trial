"""
Author - Mayank Sharan
Date - 29/8/19
Mask pool pump consumption from the input signal
"""

# Import python packages

import copy
import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def mask_timed_devices(input_data, timed_disagg_output, logger_pass):

    """
    Mask timed device consumption data points by replacing with nan

    Parameters:
        input_data              (np.ndarray)        : The 21 column raw data matrix
        timed_disagg_output     (dict)              : Contains all timed devices' data like pp and twh
        logger_pass             (dict)              : Contains the logger and the logging dictionary to be passed on

    Returns:
        input_data_masked       (np.ndarray)        : Consumption column from 21 column data without timed devices
        valid_mask_cons_bool    (np.ndarray)        : Boolean array marking indices where timed device has been detected
    """

    # Initialize logger

    logger_base = logger_pass.get('base_logger').getChild('mask_timed_devices')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Extract the consumption column from input data

    input_data_masked = copy.deepcopy(input_data)
    input_cons = input_data_masked[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Mask pool pump consumption from the input data

    num_pp_pts = timed_disagg_output.get('pp').get('num_pts')
    valid_pp_cons_bool = np.full_like(input_cons, fill_value=False)

    if num_pp_pts > 0:

        # Extract the consumption for pp

        pp_cons = timed_disagg_output.get('pp').get('cons')

        # Replace points with positive consumption by nan

        valid_pp_cons_bool = pp_cons > 0
        input_cons[valid_pp_cons_bool] = np.nan

        logger.info('Pool pump consumption masked for vacation | %d points', num_pp_pts)

    else:

        logger.info('Pool pump consumption not masked for vacation |')

    # Mask timed water heater consumption from the input data

    num_twh_pts = timed_disagg_output.get('twh').get('num_pts')
    valid_twh_cons_bool = np.full_like(input_cons, fill_value=False)

    if num_twh_pts > 0:

        # Extract the consumption for timed wh

        twh_cons = timed_disagg_output.get('twh').get('cons')

        # Replace points with positive consumption by nan

        valid_twh_cons_bool = twh_cons > 0
        input_cons[valid_twh_cons_bool] = np.nan

        logger.info('Timed water heater consumption masked for vacation | %d points', num_twh_pts)

    else:

        logger.info('Timed water heater consumption not masked for vacation |')

    # Compute boolean for points where masking has been performed

    valid_mask_cons_bool = np.logical_or(valid_pp_cons_bool, valid_twh_cons_bool)

    # Place modified consumption back in the 21 column matrix

    input_data_masked[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = input_cons

    return input_data_masked, valid_mask_cons_bool
