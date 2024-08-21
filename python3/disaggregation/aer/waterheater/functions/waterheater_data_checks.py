"""
Author - Nikhil Singh Chauhan
Date - 16/10/18
Data checks for water heater module
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.master_pipeline.preprocessing.downsample_data import downsample_data


def check_downsampling(input_data, sampling_rate, wh_config, logger_base):
    """
    Parameters:
        input_data          (np.ndarray)    : Input 21-column matrix
        sampling_rate       (int)           : Data point interval in seconds
        wh_config           (dict)          : Config parameters for water heater
        logger_base         (dict)          : Object containing logger info

    Returns:
        input_data          (np.ndarray)    : Downsampled input data
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('check_downsampling')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Check if the downsampling is required

    if sampling_rate < wh_config['min_sampling_rate']:
        # Downsample the data to 15-min

        input_data = downsample_data(input_data, wh_config['min_sampling_rate'])

        logger.info('Downsampling data to 900 because of sampling rate | {}'.format(sampling_rate))
    else:
        logger.info('Downsampling not required for the data | ')

    return input_data


def check_number_days_validity(input_data, global_config, wh_config):
    """
    Parameters:
        input_data              (np.ndarray)    : Input 21-column matrix
        global_config           (dict)          : Config parameters for the user
        wh_config               (dict)          : Config parameters for the module

    Returns:
        is_valid                (bool)          : Boolean check for valid hsm
        data_num_days           (int)           : Number of days in data
    """

    # Retrieve the minimum number of days required from config

    num_days_threshold = wh_config.get('min_num_days')

    # Get number of days in data

    data_num_days = len(np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX]))

    # Retrieve the disagg mode for the run

    disagg_mode = global_config.get('disagg_mode')

    # If days < 270 for historical, run is invalid

    is_valid = (((disagg_mode == 'historical') or (disagg_mode == 'incremental')) and (data_num_days > num_days_threshold)) \
               or (disagg_mode == 'mtd')

    return is_valid, data_num_days
