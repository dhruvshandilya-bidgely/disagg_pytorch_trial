"""
Author - Sahana M
Date - 2/3/2021
Compute WH potential based on weather data analytics output
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_wh_potential(in_data, logger_pass):

    """
    Returns Water heater potential based on weather data analytics output
    Args:
        in_data                 (np.ndarray): Input data
        logger_pass             (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        wh_potential            (np.array)  : Water heater potential array
    """

    # Initialize logger

    logger_base = logger_pass.get('base_logger').getChild('get_wh_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Initialise all the variables to be used

    unique_days = np.unique(in_data[:, Cgbdisagg.INPUT_DAY_IDX])

    # For each day get the WH potential for that day

    new_wh_pot = []
    for epoch_day in unique_days:
        days = in_data[:, Cgbdisagg.INPUT_DAY_IDX] == epoch_day
        wh_potential_day = np.nanmedian(in_data[days, Cgbdisagg.INPUT_WH_POTENTIAL_IDX])
        new_wh_pot.append(wh_potential_day)

    wh_potential = np.asarray(new_wh_pot)

    wh_pot_indexes = (wh_potential > 0) & (wh_potential != np.nan)

    logger.info('Number of days marked as WH potential | %d ', np.sum(wh_pot_indexes))

    return wh_potential
