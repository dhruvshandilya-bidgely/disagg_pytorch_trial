"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to find high energy boxes in consumption data
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy


def boxes_sanity_checks(input_box_features, debug, min_energy, ev_config, logger_base):
    """
    Function to check the quality of boxes

        Parameters:
            input_box_features        (np.ndarray)        : Current Box Features
            debug                     (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
            min_energy                (float)              : Minimum energy of the boxes
            ev_config                  (dict)              : EV module config
            logger_base               (logger)            : Logging object to log important steps and values in the run

        Returns:
            check_fail                (bool)              : Boolean signifying whether boxes are acceptable
            debug                     (object)            : Object containing all important data/values as well as HSM
    """
    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('boxes_sanity_checks')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Reading config params

    sanity_check_config = ev_config.get('detection', {}).get('sanity_checks', {})

    duration_variation = sanity_check_config.get('duration_variation')
    box_energy_variation = sanity_check_config.get('box_energy_variation')
    within_box_energy_variation = sanity_check_config.get('within_box_energy_variation')

    box_features = deepcopy(input_box_features)

    # Getting column descriptions of box features matrix
    columns_dict = ev_config.get('box_features_dict')

    overall_duration_variation = np.mean(np.abs(box_features[:, columns_dict['boxes_duration_column']] - np.mean(
        box_features[:, columns_dict['boxes_duration_column']])))

    overall_box_energy_variation = np.std(box_features[:, columns_dict['boxes_median_energy']]) / min_energy
    overall_within_box_energy_variation = np.mean(box_features[:, columns_dict['boxes_energy_std_column']]) / min_energy

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

    logger.info('Duration variation check failed | {}'.format(duration_check_fail))
    logger.info('Boxes energy variation check failed | {}'.format(box_energy_variation_check_fail))
    logger.info('Within box energy variation check failed | {}'.format(within_box_energy_variation_check_fail))

    check_fail = duration_check_fail | box_energy_variation_check_fail | within_box_energy_variation_check_fail

    return check_fail, debug
