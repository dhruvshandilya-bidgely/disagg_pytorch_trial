"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module for applying data check for ev module
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.ev.functions.estimation.filter_noise_boxes import filter_noise_boxes
from python3.disaggregation.aer.ev.functions.estimation.get_potential_ev_boxes import get_potential_ev_boxes
from python3.disaggregation.aer.ev.functions.detection.get_boxes_features import boxes_features
from python3.disaggregation.aer.ev.functions.estimation.remove_doubtful_boxes import remove_doubtful_boxes
from python3.disaggregation.aer.ev.functions.ev_l1.l1_estimation_post_processing import refine_l1_boxes


def estimate_ev_consumption(in_data, debug, ev_config, error_list, logger_pass):
    """
    Function for EV detection

    Parameters:
        in_data             (np.ndarray)            : Input matrix
        ev_config            (dict)                  : Configuration for the algorithm
        debug               (dict)                  : Dictionary containing output of each step
        error_list          (list)                  : The list of handled errors
        logger_pass         (logger)                : The logger object

    Returns:
        debug               (dict)                  : Output at each algo step
        error_list          (dict)                  : List of handled errors encountered
    """

    # Taking new logger base for this module

    logger_local = logger_pass.get('logger').getChild('ev_estimation')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_pass.get('logging_dict')}

    input_data = deepcopy(debug['input_after_baseload'])

    # Fill NaNs with zero

    energy = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    energy[np.isnan(energy)] = 0

    input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = energy

    if ev_config.get('disagg_mode') == 'mtd':
        hsm_in = debug.get('hsm_in')
    else:
        hsm_in = None

    # Get all the potentially EV boxes

    debug, updated_box_data = get_potential_ev_boxes(debug, ev_config, logger_pass, hsm_in)

    debug['new_box_data'] = deepcopy(updated_box_data)

    new_ev_output, debug = filter_noise_boxes(updated_box_data, debug, ev_config, logger_pass, input_data, hsm_in)

    box_features = boxes_features(new_ev_output, debug.get('factor'), ev_config)

    # Remove doubtful boxes if pilot id in the list of est_boxes_refine_pilots
    if str(ev_config.get('pilot_id')) in ev_config.get('est_boxes_refine_pilots'):
        new_ev_output, box_features = remove_doubtful_boxes(debug, ev_config, new_ev_output, box_features, logger_pass)

    # Remove noisy L1 boxes

    if debug.get('charger_type') == 'L1':
        new_ev_output, box_features = refine_l1_boxes(new_ev_output, box_features, ev_config, debug, logger_pass)

    debug['final_ev_signal'] = deepcopy(new_ev_output)
    debug['final_ev_box_features'] = deepcopy(box_features)

    # Get residual data

    residual_data = deepcopy(debug.get('original_input_data'))
    residual_data[np.isnan(residual_data)] = 0

    residual_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= new_ev_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    residual_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.fmax(0, residual_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    debug['residual_data'] = deepcopy(residual_data)

    return debug, error_list
