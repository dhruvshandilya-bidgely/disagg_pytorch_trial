"""
Author - Mayank Sharan
Date - 21st May 2021
Apply per column sanity checks to limit data
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.sanity_constants import SanityConstants


def enforce_column_sanity(input_data, logger_pass):

    """
    Per column enforce sanity values

    Parameters:
        input_data          (np.ndarray)        : 21 column data matrix
        logger_pass         (dict)              : Dictionary containing logger base and logging dict

    Returns:
        input_data          (np.ndarray)        : 21 column data matrix
    """

    # Initialize logger for the function

    logger_base = logger_pass.get('base_logger').getChild('enforce_column_sanity')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # If column index is in this list mark violating values as missing rather than replacing

    mark_missing_col = [Cgbdisagg.INPUT_DOW_IDX, Cgbdisagg.INPUT_HOD_IDX, Cgbdisagg.INPUT_TEMPERATURE_IDX,
                        Cgbdisagg.INPUT_SKYCOV_IDX, Cgbdisagg.INPUT_DEW_IDX, Cgbdisagg.INPUT_SUNRISE_IDX,
                        Cgbdisagg.INPUT_SUNSET_IDX, Cgbdisagg.INPUT_FEELS_LIKE_IDX, Cgbdisagg.INPUT_WET_BULB_IDX,
                        Cgbdisagg.INPUT_WIND_DIR_IDX]

    # Loop over each column and enforce sanity checks

    for col_idx in range(Cgbdisagg.INPUT_DIMENSION):

        min_lim = SanityConstants.sanity_min_limits.get(col_idx)
        max_lim = SanityConstants.sanity_max_limits.get(col_idx)

        if min_lim is None or max_lim is None:
            continue

        min_violation_bool = input_data[:, col_idx] < min_lim
        max_violation_bool = input_data[:, col_idx] > max_lim

        logger.info('Number of points in violation are | column - %d, min - %d, max - %d', col_idx,
                    np.sum(min_violation_bool), np.sum(max_violation_bool))

        # Adjust values to lie within limits or mark missing

        if col_idx in mark_missing_col:
            input_data[min_violation_bool, col_idx] = np.nan
            input_data[max_violation_bool, col_idx] = np.nan
        else:
            input_data[min_violation_bool, col_idx] = min_lim
            input_data[max_violation_bool, col_idx] = max_lim

    return input_data
