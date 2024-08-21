"""
Author - Paras Tehria / Sahana M
Date - 1-Feb-2022
Module to find high energy boxes in consumption data
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.rolling_function import rolling_function
from python3.disaggregation.aer.ev.functions.get_ev_boxes import get_ev_boxes
from python3.disaggregation.aer.ev.functions.get_season_categorisation import add_season_month_tag
from python3.disaggregation.aer.ev.functions.ev_l1.get_boxes_features import boxes_features
from python3.disaggregation.aer.ev.functions.ev_l1.boxes_sanity_checks import boxes_sanity_checks
from python3.disaggregation.aer.ev.functions.ev_l1.overlapping_box_removal import overlapping_box_removal

from python3.disaggregation.aer.ev.functions.ev_l1.filter_ev_boxes import filter_ev_boxes


def dynamic_box_fitting(in_data, ev_config, debug, logger_base):
    """
    This function is used to obtain probable L1 boxes
    Parameters:
        in_data                 (np.ndarray)            : Input data
        ev_config               (dict)                  : EV configurations
        debug                   (dict)                  : Debug dictionary
        logger_base             (Logger)                : Logger object
    Returns:
        debug                   (dict)                  : Debug dictionary
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('ev_box_fitting')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking deepcopy of input data to avoid scoping issues

    input_data = deepcopy(in_data)
    min_max_signal = deepcopy(in_data)
    current_box_data = deepcopy(in_data)
    current_box_features = []

    # Extract required variables

    factor = debug['factor']
    region = ev_config['region']
    minimum_duration = ev_config['detection']['minimum_duration']
    max_box_amplitude = ev_config['detection']['max_box_amplitude']
    step_amplitude = ev_config['detection'][region + '_amplitude_step']
    min_amplitude = ev_config['detection'][region + '_start_box_amplitude']

    # Applying min max filter on the input data to remove noise

    window_size = ev_config['detection']['min_max_filter_duration'] * factor
    min_filtered = rolling_function(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], window_size, out_metric='min')
    max_filtered = rolling_function(min_filtered, window_size, out_metric='max')
    min_max_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = max_filtered

    box_index = 1
    checks_fail = True

    # getting month number and season epoch level

    input_data = add_season_month_tag(input_data)
    input_data_copy = deepcopy(input_data)

    debug['l1']['min_max_signal'] = min_max_signal
    input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = min_max_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Box fitting for different amplitudes

    current_box_amplitude = min_amplitude

    while checks_fail and current_box_amplitude <= max_box_amplitude:

        logger.info('Box fitting index in progress | {}'.format(box_index))

        min_energy = current_box_amplitude / factor

        debug['l1']['amplitude_' + str(box_index)] = current_box_amplitude

        logger.info('Current amplitude (in Watts) | {}'.format(current_box_amplitude))

        # Get EV boxes and filter them

        current_box_data = get_ev_boxes(input_data, minimum_duration, min_energy, factor, logger_pass)
        current_box_data = filter_ev_boxes(current_box_data, input_data_copy, factor, ev_config, logger_pass)
        current_box_data = get_ev_boxes(current_box_data, minimum_duration, min_energy, factor, logger_pass)

        # Get box features

        current_box_features = boxes_features(current_box_data, factor)

        # Saving box data and features for current iteration

        debug['l1']['box_data_' + str(box_index)] = deepcopy(current_box_data)
        debug['l1']['box_features_' + str(box_index)] = deepcopy(current_box_features)

        extra_boxes_idx = current_box_features[:, 7] >= max_box_amplitude

        # For all the extra boxes remove them by setting their consumption to 0

        for box in current_box_features[extra_boxes_idx]:
            start_idx, end_idx = box[:2].astype(int)
            current_box_data[start_idx: (end_idx + 1), 6] = 0

        current_box_features = current_box_features[np.logical_not(extra_boxes_idx)]

        # Remove the overlapped boxes from the new boxes

        if box_index > 1:
            current_box_data, current_box_features = overlapping_box_removal(current_box_data, current_box_features,
                                                                             box_index, min_energy, factor,
                                                                             debug, logger_pass)

        # Number of boxes left
        # Saving box data and features for current iteration

        debug['l1']['updated_box_data_' + str(box_index)] = deepcopy(current_box_data)
        debug['l1']['updated_box_features_' + str(box_index)] = deepcopy(current_box_features)

        count_boxes = current_box_features.shape[0]

        if count_boxes <= ev_config['detection']['minimum_boxes_count']:
            break

        logger.info('Number of boxes | {}'.format(count_boxes))

        # Check box features

        checks_fail, debug = boxes_sanity_checks(current_box_features, debug, min_energy, ev_config, logger_pass)

        box_index += 1
        current_box_amplitude += step_amplitude

    # Remove noisy boxes

    final_box_index = max(1, box_index - 1)

    # replacing min-max box data with original box data

    current_box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    for idx, box in enumerate(current_box_features):
        start_idx, end_idx = box[:2].astype(int)
        current_box_data[start_idx: end_idx + 1, Cgbdisagg.INPUT_CONSUMPTION_IDX] = \
            input_data_copy[start_idx: end_idx + 1, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # On the new box data get the end hour of the boxes

    col_indexes = ev_config.get('box_features_dict')

    current_box_features = boxes_features(current_box_data, factor)
    box_end_hod_arr = np.floor((current_box_features[:, col_indexes['boxes_start_hod']] +
                                current_box_features[:, col_indexes['boxes_duration_column']]) % 24)
    current_box_features = np.c_[current_box_features, box_end_hod_arr]

    night_time = np.arange(18, 32) % 24
    intersect_dur = []

    # Calculate boxes based on the day time or night time usage

    for idx, box in enumerate(current_box_features):
        box_hours = np.arange(box[col_indexes['boxes_start_hod']], box[col_indexes['boxes_duration_column']] +
                              box[col_indexes['boxes_start_hod']]) % 24
        intersect_dur.append(len(np.intersect1d(night_time, box_hours)))

    # Remove boxes based on duration

    current_box_features = np.c_[current_box_features, intersect_dur]
    boxes_idx_to_remove = current_box_features[:, 16] < minimum_duration

    for box in current_box_features[boxes_idx_to_remove]:
        start_idx, end_idx = box[:2].astype(int)
        current_box_data[start_idx: (end_idx + 1), 6] = 0

    current_box_features = current_box_features[np.logical_not(boxes_idx_to_remove)]

    # Calculate reliable and unreliable boxes

    reliable_boxes_idx = (current_box_features[:, 16] / current_box_features[:, 4]) >= 0.7
    reliable_boxes = current_box_features[reliable_boxes_idx]
    unreliable_boxes = current_box_features[np.logical_not(reliable_boxes_idx)]

    reliable_box_data = deepcopy(current_box_data)
    unreliable_box_data = deepcopy(current_box_data)

    for box in reliable_boxes:
        start_idx, end_idx = box[:2].astype(int)
        unreliable_box_data[start_idx: (end_idx + 1), 6] = 0

    for box in unreliable_boxes:
        start_idx, end_idx = box[:2].astype(int)
        reliable_box_data[start_idx: (end_idx + 1), 6] = 0

    min_energy_per_charge = ev_config['minimum_energy_per_charge'][region]

    # Store all the L1 boxes related features in the Debug dictionary

    debug['l1']['updated_box_data_' + str(final_box_index)] = deepcopy(current_box_data)
    debug['l1']['updated_box_features_' + str(final_box_index)] = deepcopy(current_box_features)

    debug['l1']['final_box_index'] = deepcopy(final_box_index)

    debug['l1']['reliable_box_data'] = deepcopy(reliable_box_data)
    debug['l1']['unreliable_box_data'] = deepcopy(unreliable_box_data)

    debug['l1']['reliable_box_features'] = deepcopy(reliable_boxes)
    debug['l1']['unreliable_box_features'] = deepcopy(unreliable_boxes)

    debug = remove_noise_detection_boxes(debug, min_energy_per_charge, final_box_index)

    return debug


def remove_noise_detection_boxes(debug, min_energy, index):
    """
    This function is used to removed noise boxes
    Parameters:
        debug               (dict)          : Debug dictionary
        min_energy          (float)         : Minimum energy
        index               (float)         : Index value
    Returns:
        debug               (dict)          : Debug dictionary
    """

    box_data = deepcopy(debug['l1']['updated_box_data_' + str(index)])

    box_features = deepcopy(debug['l1']['updated_box_features_' + str(index)])

    # Mark Invalid energy boxes based on minimum energy

    invalid_energy_boxes = box_features[box_features[:, 5] < min_energy]

    # Remove the invalid boxes

    for i, row in enumerate(invalid_energy_boxes):
        start_idx, end_idx = row[:2].astype(int)
        box_data[start_idx:(end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    box_features = box_features[box_features[:, 5] >= min_energy]

    debug['l1']['updated_box_data_' + str(index)] = deepcopy(box_data)
    debug['l1']['updated_box_features_' + str(index)] = deepcopy(box_features)

    return debug
