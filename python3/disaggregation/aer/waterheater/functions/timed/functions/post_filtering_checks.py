"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Post filtering checks for energy consistency
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.timed.functions.post_process import bill_cycle_filter


def post_filtering_checks(box_data, debug, wh_config, logger_base):
    """
    Validate if the energy concentration is still consistent

    Parameters:
        box_data            (np.ndarray)    : Input 21-column box data
        debug               (dict)          : Algorithm intermediate steps output
        wh_config           (dict)          : Config params
        logger_base         (dict)          : Logger object

    Returns:
        box_data            (np.ndarray)    : Updated box data
        debug               (dict)          : Algorithm intermediate steps output
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('post_filtering_checks')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    if (debug.get('timed_hld_wtr') is not None) and (debug['timed_hld_wtr'] == 1):
        # If winter only timed water heater detected

        box_hod_min_threshold = wh_config['wtr_box_hod_min_threshold']
    else:
        # If across seasons water heater detected

        box_hod_min_threshold = wh_config['box_hod_min_threshold']

    # Number of days in the data

    num_days = len(np.unique(box_data[:, Cgbdisagg.INPUT_DAY_IDX]))

    # Defining time division brackets to find fractions

    bin_offset = 0.5

    edges = np.arange(0, debug['max_hod'] + 2) - bin_offset

    # Calculate the energy fractions from box data

    box_energy_idx = (box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0).astype(int)

    hod_count, _ = np.histogram(box_data[box_energy_idx, Cgbdisagg.INPUT_HOD_IDX], bins=edges)
    hod_count = hod_count / num_days

    box_max_hod = np.max(hod_count)

    # Add max fraction to debug object

    debug['max_box_hod_count'] = box_max_hod

    logger.info('Max energy fraction post filtering | {}'.format(box_max_hod))

    # Check if energy consistent enough

    if box_max_hod < box_hod_min_threshold:
        # If inconsistent energy, return module with detection zero

        debug['num_runs'] = 0
        debug['timed_hld'] = 0

        box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        logger.info('Rejecting timed water heater due to insufficient energy fraction post filtering | ')
    else:
        # If consistent energy, check for partial bill cycles

        box_data = bill_cycle_filter(box_data, box_energy_idx, wh_config, debug, logger_pass)

    return box_data, debug
