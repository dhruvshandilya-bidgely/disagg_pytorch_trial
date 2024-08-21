"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module checks if the non-timed water heater detection significant enough to override the
timed waterheater detection
"""

# Import python packages

import logging


def override_timed_wh_detection(debug, hld_probability, wh_config, logger_base):
    """
    Parameters:
        debug                   (dict)      : Output of algorithm intermediate steps
        hld_probability         (float)     : Probability of non-timed waterheater detection
        wh_config               (dict)      : Water heater config params
        logger_base             (dict)      : Logger object

    Returns:
        hld                     (int)       : Updated non-timed water heater detection
        debug                   (dict)      : Output of algorithm intermediate steps
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('timed_detection_override')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the relevant params from config
    # timed_bounds -> [0.75, 0.65, 0.50, 0.40, 0.30],
    # non_timed_bounds -> [0.95, 0.90, 0.85, 0.75, 0.65]

    timed_bounds = wh_config['thermostat_wh']['detection']['timed_detection_bounds']
    non_timed_bounds = wh_config['thermostat_wh']['detection']['non_timed_detection_bounds']

    # Get the timed water heater confidence

    timed_confidence = debug['timed_debug']['timed_confidence']

    # Conditions for overriding the timed water heater hld

    if hld_probability >= non_timed_bounds[0]:
        # Non-timed water heater preserved due to high confidence

        debug['timed_hld'] = 0
        hld = 1
    elif (hld_probability >= non_timed_bounds[1]) and timed_confidence < timed_bounds[0]:
        # Non-timed water heater preserved due to high confidence

        debug['timed_hld'] = 0
        hld = 1
    elif (hld_probability >= non_timed_bounds[2]) and timed_confidence < timed_bounds[1]:
        # Non-timed water heater preserved due to high confidence

        debug['timed_hld'] = 0
        hld = 1
    elif (hld_probability >= non_timed_bounds[3]) and timed_confidence < timed_bounds[2]:
        # Non-timed water heater preserved due to high confidence

        debug['timed_hld'] = 0
        hld = 1
    elif (hld_probability >= non_timed_bounds[4]) and timed_confidence < timed_bounds[3]:
        # Non-timed water heater preserved due to high confidence

        debug['timed_hld'] = 0
        hld = 1
    elif timed_confidence < timed_bounds[4]:
        # Non-timed water heater preserved due to high confidence

        debug['timed_hld'] = 0
        hld = 1
    else:
        hld = 0

        logger.info('Non-timed confidence not high enough to override timed | ')

    # Find the final hld of timed / non-timed and log the relevant valeus

    if hld == 1:
        logger.info('Timed water heater detection overridden, confidence (timed, non-timed) | ({}, {})'.
                    format(timed_confidence, hld_probability))
    else:
        logger.info('Non-timed water heater detection overridden, confidence (timed, non-timed) | ({}, {})'.
                    format(timed_confidence, hld_probability))

    return hld, debug
