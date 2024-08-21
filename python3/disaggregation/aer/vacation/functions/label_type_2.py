"""
Author - Mayank Sharan
Date - 5/9/19
Label type 2 vacation days
"""

# Import python packages

import logging
import numpy as np


def label_type_2(day_data, day_nan_count, vac_confidence, vacation_config, logger_pass):

    """
    Parameters:
        day_data            (np.ndarray)        : Day wise data matrix
        day_nan_count       (np.ndarray)        : Day level array containing number of points which are nan
        vac_confidence      (np.ndarray)        : Confidence values for each day being vacation
        vacation_config     (dict)              : Contains all configuration variables needed for vacation
        logger_pass         (dict)              : Contains the logger and the logging dictionary to be passed on

    Returns:
        type_2_bool         (np.ndarray)        : Day wise boolean marking type 2 vacation days
        vac_confidence      (np.ndarray)        : Confidence values for each day being vacation
    """

    # Initialize logger

    logger_base = logger_pass.get('base_logger').getChild('label_type_2')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Initialize config to detect type 2 vacation days

    type_2_config = vacation_config.get('type_2')

    # All days where greater than 6 hours has values as nan are invalidated from being considered for type 2

    valid_days_idx = day_nan_count <= int(type_2_config.get('max_nan_ratio') * day_data.shape[1])

    # Label days with 0 std dev and more than 1 values as type 2 vacation

    day_std_dev = np.nanstd(day_data, axis=1)
    type_2_bool = np.logical_and(day_std_dev == 0, valid_days_idx)

    # Log number of days marked as type 2 vacation

    num_type_2_days = np.sum(type_2_bool)
    logger.info('Number of type 2 vacation days are | %d', num_type_2_days)

    # Mark confidence for days marked as type 2 vacation

    vac_confidence[type_2_bool] = type_2_config.get('conf_val')

    return type_2_bool, vac_confidence
