"""
Author - Mayank Sharan
Date - 24/12/18
Remove extra high consumption values that are erroneous or will interfere with disagg
"""

# Import python packages

import logging
import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def modify_high_consumption(input_data, sampling_rate, logger_pass):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column data matrix
        sampling_rate       (int)               : sampling rate at which the data is given
        logger_pass         (dict)              : Dictionary containing logger base and logging dict

    Returns:
        input_data          (np.ndarray)        : 21 column data matrix modified
    """

    # Initialize logger for the function

    logger_base = logger_pass.get('base_logger').getChild('modify_high_consumption')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Initialize threshold to 20 kWh on the hour level

    high_cons_diff_threshold = Cgbdisagg.HIGH_CONS_THRESHOLD * sampling_rate / Cgbdisagg.SEC_IN_HOUR

    # Get basic consumption thresholds

    max_consumption_baseline = np.percentile(a=input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], q=95)
    max_cons_value = np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Based on the threshold modify unusually high values

    if max_cons_value - max_consumption_baseline > high_cons_diff_threshold:
        logger.info('Abnormally high consumption values in data threshold diff is %.3f |',
                    max_cons_value - max_consumption_baseline)

        # Initialize basic threshold to remove the high consumption values

        high_cons_base = max_consumption_baseline + high_cons_diff_threshold
        high_cons_idx = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > high_cons_base

        num_pts_high_cons = np.sum(high_cons_idx)
        logger.info('Number of points above the high consumption threshold of %.3f are %d |', high_cons_base,
                    num_pts_high_cons)

        # Set high consumption values to nan and then interpolate

        cons_values = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
        cons_values[high_cons_idx] = np.nan

        cons_df = pd.DataFrame(data=cons_values)
        cons_df = cons_df.interpolate(method='linear')

        input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.reshape(a=cons_df.values, newshape=(len(cons_values, )))
    else:
        logger.info('No abnormally high consumption values in data |')

    return input_data
