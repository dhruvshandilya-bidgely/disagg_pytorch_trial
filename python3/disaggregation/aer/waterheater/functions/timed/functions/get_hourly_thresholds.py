"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to find fraction threshold for each time division
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.timed.functions.find_significant_hours import get_before_after


def get_hourly_thresholds(edge_hod_count, max_fraction, max_hours, breakpoint_threshold, wh_config, debug, logger_base):
    """
    Finding each hours appropriate fraction threshold

    Parameters:
        edge_hod_count          (np.ndarray)    : Seasonal energy fraction per time division
        max_fraction            (float)         : Maximum energy fraction
        max_hours               (np.ndarray)    : Hours with hugh energy fraction
        breakpoint_threshold    (float)         : Threshold fraction
        wh_config               (dict)          : Config params
        debug                   (dict)          : Algorithm intermediate steps output
        logger_base             (dict)          : Logger object

    Returns:
        thresholds              (np.ndarray)    : Fraction thresholds for each time division
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_hourly_thresholds')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve information from debug object

    factor = debug['time_factor']
    n_divisions = Cgbdisagg.HRS_IN_DAY * factor

    # Retrieve the params from config

    min_fraction_allowed = wh_config['energy_threshold']

    # Calculate the minimum threshold

    breakpoint_threshold = np.min([breakpoint_threshold, min_fraction_allowed * max_fraction])

    logger.info('Breakpoint threshold | {}'.format(breakpoint_threshold))

    # Initialize the thresholds array with breakpoint threshold

    thresholds = np.array([breakpoint_threshold] * n_divisions)

    # Mark the significant time divisions that is to be preserved

    significant_hours = (edge_hod_count >= breakpoint_threshold)

    # Initialize the final important hours array

    final_imp_hours = np.array([False] * n_divisions)

    # Get the number of division before and after that is to be preserved

    before, after, count = get_before_after(max_fraction, factor, wh_config, logger_pass)

    # Accumulate the important time divisions

    important_hours = np.unique(np.sort(np.repeat(max_hours, count) +
                                        np.tile(np.arange(-before, after + 1), len(max_hours))))

    # Adjust the time divisions that are updated beyond the valid range

    important_hours[important_hours < 0] += n_divisions
    important_hours[important_hours > (n_divisions - 1)] -= n_divisions

    # Take the final unique important hours and mask them True

    important_hours = np.unique(important_hours)
    final_imp_hours[important_hours] = True

    # Take the common time divisions of important and significant time divisions

    final_imp_hours = final_imp_hours & significant_hours

    # Make threshold zero for the preserved time divisions

    thresholds[final_imp_hours] = 0

    return thresholds
