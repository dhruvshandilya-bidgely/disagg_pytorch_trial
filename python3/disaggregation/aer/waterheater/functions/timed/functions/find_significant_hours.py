"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Filter to remove insignificant / minor runs
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.timed.functions.elbow_method import elbow_method


def find_significant_hours(high_fraction_hours, max_energy_fraction, edge_hod_count, wh_config, debug, logger_base):
    """
    Find the significant hours

    Parameters:
        high_fraction_hours     (np.ndarray)        : High energy hours
        max_energy_fraction     (float)             : Maximum energy fraction
        edge_hod_count          (np.ndarray)        : Energy fraction at each time division
        wh_config               (dict)              : Config params
        debug                   (dict)              : Algorithm intermediate steps output
        logger_base             (dict)              : Logger object

    Returns:
        final_imp_hours_idx     (np.ndarray)        : Updated high energy hours
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('find_significant_hours')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve sampling rate factor and time divisions in a day

    factor = debug['time_factor']
    n_divisions = Cgbdisagg.HRS_IN_DAY * factor

    # Retrieve the energy threshold and elbow threshold from config

    elbow_threshold = wh_config['elbow_threshold']
    energy_threshold = wh_config['energy_threshold']

    # Using elbow method, find the breakpoint fraction

    breakpoint_threshold = elbow_method(edge_hod_count, elbow_threshold)
    breakpoint_threshold = np.min([breakpoint_threshold, energy_threshold * max_energy_fraction])

    logger.debug('Breakpoint threshold from elbow method is | {}'.format(breakpoint_threshold))

    # Mark time divisions above breakpoint as significant

    significant_hours = (edge_hod_count >= breakpoint_threshold)
    final_imp_hours = np.array([False] * n_divisions)

    # Get the number of hours to preserve adjacent to max energy fraction time divisions

    before, after, count = get_before_after(max_energy_fraction, factor, wh_config, logger_pass)

    logger.debug('Hours to conserve before, after | {}, {}'.format(before / factor, after / factor))

    imp_hours = np.unique(np.sort(np.repeat(high_fraction_hours, count) +
                                  np.tile(np.arange(-before, after + 1), len(high_fraction_hours))))

    imp_hours[imp_hours < 0] += n_divisions
    imp_hours[imp_hours > (n_divisions - 1)] -= n_divisions

    imp_hours = np.unique(imp_hours)
    final_imp_hours[imp_hours] = True

    final_imp_hours = final_imp_hours & significant_hours

    final_imp_hours_idx = np.where(final_imp_hours)[0]

    return final_imp_hours_idx


def get_before_after(max_fraction, factor, wh_config, logger_base):
    """
    Get before and after time division limits

    Parameters:
        max_fraction        (float)     : Max energy fraction
        factor              (int)       : Sampling rate factor
        wh_config           (dict)      : Timed waterheater config params
        logger_base         (dict)      : Logger object

    Returns:
        before              (int)       : Safe time divisions before
        after               (int)       : Safe time divisions after
        duration            (int)       : Duration of safe time divisions
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_before_after')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the max limits and adjacent hours from config

    limits = wh_config.get('max_limits')
    hours = wh_config.get('adjacent_hours')

    # Find safe divisions range for corresponding energy fraction bracket

    if max_fraction >= limits[0]:
        before = factor
        after = factor

    elif max_fraction >= limits[1]:
        before = factor + hours[0]
        after = factor + hours[0]

    elif max_fraction >= limits[2]:
        before = factor + hours[1]
        after = factor + hours[1]

    elif max_fraction >= limits[3]:
        before = factor + hours[2]
        after = factor + hours[2]

    else:
        before = factor + hours[3]
        after = factor + hours[3]

    logger.info('Max fraction is | {}'.format(max_fraction))

    logger.info('The (before, after) limits are as follows | {}, {}'.format(before, after))

    # Calculate the duration of sage time divisions

    duration = before + after + 1

    return before, after, duration
