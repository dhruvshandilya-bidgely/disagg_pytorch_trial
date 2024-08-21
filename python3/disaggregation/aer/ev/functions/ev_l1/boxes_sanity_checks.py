"""
Author - Paras Tehria/ Sahana M
Date - 1-Feb-2022
Module to find high energy boxes in consumption data
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy


def boxes_sanity_checks(input_box_features, debug, min_energy, ev_config, logger_base):
    """
    This function is used to perform some sanity checks on the boxes
    Parameters:
        input_box_features          (np.ndarray)            : All the features of the boxes captured
        debug                       (dict)                  : Debug dictionary
        min_energy                  (float)                 : Minimum energy
        ev_config                   (dict)                  : EV configurations dictionary
        logger_base                 (Logger)                : Logger
    Returns:
        check_fail                  (Boolean)               : Boxes sanity check failure status
        debug                       (dict)                  : Debug dictionary

    """
    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('boxes_sanity_checks')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Reading config params

    sanity_check_config = ev_config['detection']['sanity_checks']

    duration_variation = sanity_check_config['duration_variation']
    box_energy_variation = sanity_check_config['box_energy_variation']
    within_box_energy_variation = sanity_check_config['within_box_energy_variation']

    box_features = deepcopy(input_box_features)

    # Overall variation derivation of duration, box energy and within box energy variation

    overall_duration_variation = np.mean(np.abs(box_features[:, 4] - np.mean(box_features[:, 4]))) / np.mean(box_features[:, 4])
    overall_box_energy_variation = np.std(box_features[:, 8]) / min_energy
    overall_within_box_energy_variation = np.mean(box_features[:, 6]) / min_energy

    logger.info('Duration variance | {}'.format(overall_duration_variation))
    logger.info('Box energy variance | {}'.format(overall_box_energy_variation))
    logger.info('Within box energy variance | {}'.format(overall_within_box_energy_variation))

    # Duration variance check

    if overall_duration_variation > duration_variation:
        duration_check_fail = True
    else:
        duration_check_fail = False

    # Boxes energy variation check

    if overall_box_energy_variation > box_energy_variation:
        box_energy_variation_check_fail = True
    else:
        box_energy_variation_check_fail = False

    # Within box energy variation check

    if overall_within_box_energy_variation > within_box_energy_variation:
        within_box_energy_variation_check_fail = True
    else:
        within_box_energy_variation_check_fail = False

    logger.info('L1 Duration variation check failed | {}'.format(duration_check_fail))
    logger.info('L1 Boxes energy variation check failed | {}'.format(box_energy_variation_check_fail))
    logger.info('L1 Within box energy variation check failed | {}'.format(within_box_energy_variation_check_fail))

    check_fail = duration_check_fail | box_energy_variation_check_fail | within_box_energy_variation_check_fail

    return check_fail, debug
