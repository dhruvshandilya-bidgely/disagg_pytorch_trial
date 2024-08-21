"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Filter to remove insignificant / minor runs
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.timed.functions.box_features import Boxes
from python3.disaggregation.aer.waterheater.functions.timed.functions.update_boxes_features import update_boxes_features
from python3.disaggregation.aer.waterheater.functions.timed.functions.find_significant_hours import find_significant_hours
from python3.disaggregation.aer.waterheater.functions.timed.functions.check_energy_concentration import check_energy_concentration


def inconsistent_runs_filter(input_box_data, boxes, edge_hod_count, wh_config, debug, logger_base):
    """
    Parameters:
        input_box_data          (np.ndarray)        : Input 21-column box data
        boxes                   (np.ndarray)        : Boxes features
        edge_hod_count          (np.ndarray)        : Fraction of energy at each time division
        wh_config               (dict)              : Config params for timed water heater
        debug                   (dict)              : Algorithm output of each step
        logger_base             (dict)              : Logger object

    Returns:
        box_data                (np.ndarray)        : Updated box data
        debug                   (dict)              : Algorithm output of each step
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('remove_inconsistent_runs')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking deepcopy of input data to avoid scoping issues

    box_data = deepcopy(input_box_data)

    # Retrieve the relevant thresholds (minimum gap between runs, energy fraction between runs)

    minimum_hour_gap = wh_config['minimum_hour_gap']

    energy_fraction_threshold = wh_config['energy_fraction_threshold']
    insignificant_run_threshold = wh_config['insignificant_run_threshold']

    # Max energy fraction at any time division

    max_energy_fraction = np.max(edge_hod_count)

    # Get time divisions above a certain fraction of highest energy fraction

    high_fraction_hours = np.where(edge_hod_count >= (insignificant_run_threshold * max_energy_fraction))[0]

    logger.debug('High energy fraction hours | {}'.format(high_fraction_hours))

    # Get energy concentration of all high energy time divisions

    high_fraction_hours = check_energy_concentration(high_fraction_hours, debug['variables'], debug['time_factor'],
                                                     insignificant_run_threshold, energy_fraction_threshold,
                                                     minimum_hour_gap)

    logger.debug('High energy hours after energy concentration check | {}'.format(high_fraction_hours))

    # If no high energy time divisions left, exit timed water heater module

    if len(high_fraction_hours) < 1:
        # Initialize final timed water heater data as zero

        box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        # Update boxes features

        boxes[:, Boxes.IS_VALID] = 0

        # Update the debug object with zero runs

        debug['num_runs'] = 0
        debug['timed_hld'] = 0

        logger.info('All runs eliminated in energy check, timed water heater detection changed to zero | ')

        return box_data, boxes, debug

    # Find the significant time divisions (close to timed water heater runs max fraction)

    significant_hours = find_significant_hours(high_fraction_hours, max_energy_fraction, edge_hod_count, wh_config,
                                               debug, logger_pass)

    logger.debug('Significant time divisions after significance check | {}'.format(significant_hours))

    # Iterate over each significant time divisions to subset the boxes

    for i in range(len(edge_hod_count)):
        if i in significant_hours:
            # Preserve the significant hours boxes

            continue
        else:
            boxes[(boxes[:, Boxes.TIME_DIVISION] == i) &
                  (boxes[:, Boxes.BOX_FRACTION] < max_energy_fraction), Boxes.IS_VALID] = 0

    box_data, boxes, debug = update_boxes_features(box_data, boxes, high_fraction_hours, debug, logger_pass)

    # Get the number of water heater runs for the user

    number_runs = get_runs_count(high_fraction_hours, debug['time_factor'])

    logger.info('Number of timed water heater runs | {}'.format(number_runs))

    # Saving the number of runs, high energy fraction time divisions to debug object

    debug['num_runs'] = number_runs
    debug['max_hours'] = high_fraction_hours

    return box_data, boxes, debug


def get_runs_count(high_fraction_hours, factor):
    """
    Finding the number of runs

    Parameters:
        high_fraction_hours         (np.ndarray)    : Hours with high energy fractions
        factor                      (int)           : Sampling rate factor

    Returns:
        count                       (int)           : Number of runs detected
    """

    # Find the difference between high energy fraction time divisions

    hour_diff = np.abs(np.diff(np.r_[high_fraction_hours[0], high_fraction_hours]))

    # Count the number of consecutive chunks far from each other

    count = np.sum((hour_diff > factor) & (hour_diff < ((Cgbdisagg.HRS_IN_DAY * factor) - 1))) + 1

    return count
