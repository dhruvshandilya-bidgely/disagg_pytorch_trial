"""
Author - Sahana M
Date - 14-November-2023
Module to perform box fitting
"""

# import python packages
import logging
from copy import deepcopy

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_ev_boxes
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import boxes_features
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_utils import boxes_sanity_checks
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_utils import pick_missing_l2_boxes
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_utils import remove_overlapping_boxes
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_utils import identify_all_ev_l2_partitions
from python3.disaggregation.aer.ev.functions.deep_learning.special_case_box_fitting import multimode_charging_detection


def box_fitting(final_boxes_detected, factor, dl_debug, logger_base):
    """
    This function is used to perform dynamic box fitting
    Parameters:
        final_boxes_detected            (np.ndarray)            : Final boxes detected
        factor                          (int)                   : Sampling rate factor
        dl_debug                        (dict)                  : Debug dictionary
        logger_base                     (logger)                : Logger passed

    Returns:
        box_data                        (np.ndarray)            : Fitted box data
        debug                           (dict)                  : Debug dictionary
    """

    # Initialise the logger
    logger_local = logger_base.get('logger').getChild('box_fitting')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))
    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}

    # Extract the required variables

    sampling_rate = int(Cgbdisagg.SEC_IN_HOUR/factor)
    hod_matrix = dl_debug.get('hod_matrix')
    hod_matrix = hod_matrix.flatten()
    ev_config = dl_debug.get('config')
    region = ev_config.get('region')
    minimum_duration = ev_config['minimum_duration'][sampling_rate]
    step_amplitude = ev_config['detection'][region + '_amplitude_step']
    start_box_amplitude = ev_config['detection'][region + '_start_box_amplitude']
    max_amplitude = ev_config.get('detection', {}).get(region + '_max_amplitude')
    input_data_original = deepcopy(final_boxes_detected)
    input_data_original = input_data_original.flatten()

    box_index = 1
    current_box_amplitude = deepcopy(start_box_amplitude)

    checks_fail = True

    # box_level_index: level 1 is for boxes of the previous iteration, level 2 is for boxes of the current iteration

    box_level_index = 1

    while checks_fail and current_box_amplitude <= max_amplitude:

        min_energy = current_box_amplitude / factor

        current_box_data = get_ev_boxes(input_data_original, minimum_duration, min_energy, factor)

        # Get box features

        current_box_features = boxes_features(current_box_data, hod_matrix, factor, ev_config)

        # Saving box data and features for previous iteration

        if box_index > box_level_index:
            dl_debug['box_data_' + str(box_level_index - 1)] = deepcopy(dl_debug.get('box_data_' + str(box_level_index)))
            dl_debug['box_features_' + str(box_level_index - 1)] = deepcopy(dl_debug.get('box_features_' + str(box_level_index)))
            dl_debug['amplitude_' + str(box_level_index - 1)] = deepcopy(dl_debug.get('amplitude_' + str(box_level_index)))

        # Saving box data and features for current iteration

        dl_debug['box_data_' + str(box_level_index)] = deepcopy(current_box_data)
        dl_debug['box_features_' + str(box_level_index)] = deepcopy(current_box_features)
        dl_debug['amplitude_' + str(box_level_index)] = current_box_amplitude

        if box_index > 1:
            current_box_data, current_box_features = \
                remove_overlapping_boxes(current_box_data, hod_matrix, current_box_features, box_level_index,
                                         min_energy, factor, dl_debug, ev_config)

        # Saving updated box data and features for previous iteration

        if box_index > box_level_index:
            dl_debug['updated_box_data_' + str(box_level_index - 1)] = deepcopy(dl_debug.get('updated_box_data_' + str(box_level_index)))
            dl_debug['updated_box_features_' + str(box_level_index - 1)] = deepcopy(
                dl_debug.get('updated_box_features_' + str(box_level_index)))

        # Saving updated box data and features for current iteration

        dl_debug['updated_box_data_' + str(box_level_index)] = deepcopy(current_box_data)
        dl_debug['updated_box_features_' + str(box_level_index)] = deepcopy(current_box_features)

        # Number of boxes left

        count_boxes = current_box_features.shape[0]

        if count_boxes <= ev_config['detection']['minimum_boxes_count']:
            box_level_index = 1
            break

        # Check box features

        checks_fail = boxes_sanity_checks(current_box_features, min_energy, ev_config)
        logger.info('DL L2 : Boxes sanity checks status for box index %s is | %s', str(box_index), checks_fail)

        # Update the params

        box_index += 1
        box_level_index = 2
        current_box_amplitude += step_amplitude

    # storing index of the last stable boxes

    final_box_index = min(box_level_index, max(1, box_index - 1))
    dl_debug['final_box_index'] = deepcopy(final_box_index)

    box_data = dl_debug.get('updated_box_data_' + str(final_box_index))

    # -------------------------------------------- Special cases handling --------------------------------------------

    # Multi-mode charging handling

    box_data, dl_debug = multimode_charging_detection(box_data, dl_debug, logger_pass)
    logger.info('DL L2 : Multi-mode charging detection complete | ')

    # Picking boxes from the predicted partitions

    box_data, dl_debug = pick_missing_l2_boxes(box_data, dl_debug)
    logger.info('DL L2 : Picking missing L2 boxes complete | ')

    # Identifying False Negative Partitions and picking them up

    box_data, dl_debug = identify_all_ev_l2_partitions(box_data, dl_debug)
    logger.info('DL L2 : Identifying all EV L2 partitions complete | ')

    rows = dl_debug.get('rows')
    cols = dl_debug.get('cols')
    box_data = box_data.reshape(rows, cols)

    return box_data, dl_debug
