"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to detect if a Timed water heater present with the user
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.timed.functions.filter_boxes import filter_boxes
from python3.disaggregation.aer.waterheater.functions.timed.functions.post_process import  fit_winter_energy
from python3.disaggregation.aer.waterheater.functions.timed.functions.consistency_check import consistency_check
from python3.disaggregation.aer.waterheater.functions.timed.functions.post_filtering_checks import post_filtering_checks
from python3.disaggregation.aer.waterheater.functions.timed.functions.get_timed_wh_estimation import get_timed_wh_estimation
from python3.disaggregation.aer.waterheater.functions.timed.functions.winter_consistency_check import winter_consistency_check
from python3.disaggregation.aer.waterheater.functions.timed.functions.timed_waterheater_boxes import get_timed_waterheater_boxes


def timed_waterheater_detection(in_data, features, wh_config, debug, logger_base):
    """
    Parameters:
        in_data         (np.ndarray)        : Input data (21-column matrix)
        features        (np.ndarray)        : Empty features data table to be appended in the algo
        wh_config       (dict)              : The configuration for Timed water heater
        debug           (dict)              : Contains output at each step of algorithm
        logger_base     (logger)            : The logger object to log values

    Returns:
        timed_wh_box    (np.ndarray)        : Epoch level estimate of Timed water heater
        debug           (dict)              : Contains output at each step of algorithm
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('timed_waterheater_detection')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking deepcopy of input data to keep local instances

    input_data = deepcopy(in_data)

    # Defining minimum and maximum energy per data point

    wh_min_amp = wh_config['min_amplitude'] * wh_config['sampling_rate'] / Cgbdisagg.SEC_IN_HOUR
    wh_max_amp = wh_config['max_amplitude'] * wh_config['sampling_rate'] / Cgbdisagg.SEC_IN_HOUR

    # Finding high energy boxes in the consumption data and save to debug object

    box_data = get_timed_waterheater_boxes(input_data, wh_config, wh_min_amp, logger_pass)
    debug['box_data'] = deepcopy(box_data)

    # Generate consistency metrics for initial detection of timed water heater

    features, debug = consistency_check(input_data, box_data, features, wh_config, debug, logger_pass)

    logger.info('Initial timed water heater detection | {}'.format(debug['timed_hld']))

    # If timed water detected, go for filtering and noise removal

    if debug['timed_hld'] == 1:
        # Remove noise and clean the timed runs

        timed_wh_box, debug = filter_boxes(box_data, wh_config, wh_max_amp, debug, logger_pass)

        # Check energy concentration to verify the consistency in output

        timed_wh_box, debug = post_filtering_checks(timed_wh_box, debug, wh_config, logger_pass)

    elif debug['timed_hld'] == 0:
        # If no timed water heater detected in complete data, go for detection in winter data

        # Check for different consistency metrics for initial detection in winter season

        features, debug, wtr_idx, wtr_data, wtr_box_data = winter_consistency_check(input_data, box_data, features,
                                                                                    wh_config, debug, logger_pass)

        # If timed water heater found in winter, go for filtering and noise removal

        if debug['timed_hld_wtr'] == 1:
            # Initialize input and box data from winter data

            box_data = deepcopy(wtr_box_data)

            logger.info('Winter timed water heater detection | {}'.format(debug['timed_hld']))

            # Remove noise and clean the timed runs

            timed_wh_box, debug = filter_boxes(box_data, wh_config, wh_max_amp, debug, logger_pass)

            # Check energy concentration to verify the consistency in output

            timed_wh_box, debug = post_filtering_checks(timed_wh_box, debug, wh_config, logger_pass)
        else:
            # Initializing default variables

            timed_wh_box = np.array([])

    else:
        # Initializing default variables

        timed_wh_box = np.array([])

    # Saving features to the debug object

    debug['features'] = features

    # Exit the module if no timed water heater detected in overall and winter data

    if debug['timed_hld'] == 0:
        # Return timed water heater consumption and amplitude as zero

        out_data = deepcopy(in_data)
        out_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        debug['timed_wh_amplitude'] = 0

        return out_data, debug

    # Join winter timed water heater consumption with rest of the data

    if debug.get('timed_hld_wtr') == 1:
        timed_wh_box = fit_winter_energy(timed_wh_box, wtr_idx, in_data)

    # If after filtering and noise removal zero runs left in the output, return zero output

    if debug['num_runs'] == 0:
        # Return timed water heater consumption and amplitude as zero

        out_data = deepcopy(in_data)
        out_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        debug['timed_wh_amplitude'] = 0

        return out_data, debug

    # Fix the estimation values to have energy consumption consistency

    timed_wh_box, mu = get_timed_wh_estimation(timed_wh_box, wh_config, logger_pass)

    # Save the timed water heater amplitude to debug object

    debug['timed_wh_amplitude'] = mu

    return timed_wh_box, debug
