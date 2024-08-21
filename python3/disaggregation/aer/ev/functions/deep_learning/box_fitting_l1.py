"""
Author - Sahana M
Date - 14-Nov-2023
Module to get the potential EV boxes for L1
"""

# import python packages
import logging
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.rolling_function import rolling_function
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_ev_boxes
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_utils import filter_ev_boxes
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import boxes_features
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_utils import boxes_sanity_checks
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_utils import pick_missing_l1_boxes
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_utils import remove_overlapping_boxes
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_utils import identify_all_ev_l1_partitions


def box_fitting(final_boxes_detected, factor, dl_debug, logger_base):
    """
    This function is used to perform dynamic box fitting on L1 users
    Parameters:
        final_boxes_detected            (np.ndarray)            : Final boxes detected
        factor                          (int)                 : Sampling rate factor
        dl_debug                        (dict)                  : Debug dictionary
        logger_base                     (logger)                : Logger passed
    Returns:
        box_data                        (np.ndarray)            : Fitted box data
        debug                           (dict)                  : Debug dictionary
    """

    # Initialise the logger
    logger_local = logger_base.get('logger').getChild('box_fitting')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract the required variables

    hod_matrix = dl_debug.get('hod_matrix')
    hod_matrix = hod_matrix.flatten()
    dl_config = dl_debug.get('config')
    columns_dict = dl_config['features_dict']
    region = dl_debug.get('ev_config').get('region')
    max_amplitude = dl_config.get('max_amplitude_l1')
    minimum_duration = dl_config.get('min_duration_l1')
    input_data_original = deepcopy(final_boxes_detected)
    input_data_original = input_data_original.flatten()
    step_amplitude = dl_config.get(region + '_amplitude_step')
    start_box_amplitude = dl_config.get(region + '_start_box_amplitude')

    box_index = 1
    checks_fail = True
    current_box_amplitude = deepcopy(start_box_amplitude)
    input_data_copy = deepcopy(input_data_original)
    current_box_data = deepcopy(input_data_original)

    # Perform min-max rolling

    window_size = dl_config['min_max_filter_duration'] * factor
    min_filtered = rolling_function(current_box_data, window_size, out_metric='min')
    max_filtered = rolling_function(min_filtered, window_size, out_metric='max')
    min_max_signal = max_filtered

    input_data_original = min_max_signal
    current_box_features = []

    while checks_fail and current_box_amplitude <= max_amplitude:

        min_energy = current_box_amplitude / factor

        current_box_data = get_ev_boxes(input_data_original, minimum_duration, min_energy, factor)
        current_box_data = filter_ev_boxes(current_box_data, input_data_copy, factor, dl_config)
        current_box_data = get_ev_boxes(current_box_data, minimum_duration, min_energy, factor)
        current_box_features = boxes_features(current_box_data, hod_matrix, factor, dl_config, 'l1')

        extra_boxes_idx = current_box_features[:, columns_dict['boxes_energy_per_point_column']] >= max_amplitude

        # For all the extra boxes remove them by setting their consumption to 0

        for box in current_box_features[extra_boxes_idx]:
            start_idx, end_idx = box[: columns_dict['start_energy_column']].astype(int)
            current_box_data[start_idx: (end_idx + 1)] = 0

        current_box_features = current_box_features[np.logical_not(extra_boxes_idx)]

        dl_debug['box_data_' + str(box_index)] = deepcopy(current_box_data)
        dl_debug['box_features_' + str(box_index)] = deepcopy(current_box_features)

        if box_index > 1:
            current_box_data, current_box_features = \
                remove_overlapping_boxes(current_box_data, hod_matrix, current_box_features, box_index, min_energy,
                                         factor, dl_debug, dl_config, 'l1')

        count_boxes = current_box_features.shape[0]

        if count_boxes <= dl_config['minimum_boxes_count']:
            break

        # Check box features

        checks_fail = boxes_sanity_checks(current_box_features, min_energy, dl_config, 'l1')
        logger.info('DL L1 : Boxes sanity checks status for box index %s is | %s', str(box_index), checks_fail)

        # Update the params

        box_index += 1
        current_box_amplitude += step_amplitude

    # replacing min-max box data with original box data

    current_box_data[current_box_data > 0] = 0

    for idx, box in enumerate(current_box_features):
        start_idx, end_idx = box[: columns_dict['start_energy_column']].astype(int)
        current_box_data[start_idx: end_idx + 1] = input_data_copy[start_idx: end_idx + 1]

    # On the new box data get the end hour of the boxes

    col_indexes = dl_config.get('box_features_dict')

    current_box_features = boxes_features(current_box_data, hod_matrix, factor, dl_config, 'l1')
    box_end_hod_arr = np.floor((current_box_features[:, col_indexes['boxes_start_hod']] +
                                current_box_features[:, col_indexes['boxes_duration_column']]) % Cgbdisagg.HRS_IN_DAY)
    current_box_features = np.c_[current_box_features, box_end_hod_arr]

    night_time = np.arange(18, 32) % Cgbdisagg.HRS_IN_DAY
    intersect_dur = []

    # Calculate boxes based on the day time or night time usage

    for idx, box in enumerate(current_box_features):
        box_hours = np.arange(box[col_indexes['boxes_start_hod']], box[col_indexes['boxes_duration_column']] +
                              box[col_indexes['boxes_start_hod']]) % Cgbdisagg.HRS_IN_DAY
        intersect_dur.append(len(np.intersect1d(night_time, box_hours)))

    # Remove boxes based on duration

    current_box_features = np.c_[current_box_features, intersect_dur]
    boxes_idx_to_remove = current_box_features[:, col_indexes['weekend_boolean']] < minimum_duration

    for box in current_box_features[boxes_idx_to_remove]:
        start_idx, end_idx = box[: col_indexes['start_energy_column']].astype(int)
        current_box_data[start_idx: (end_idx + 1)] = 0

    current_box_features = current_box_features[np.logical_not(boxes_idx_to_remove)]

    # Calculate reliable and unreliable boxes

    reliable_boxes_idx = (current_box_features[:, col_indexes['weekend_boolean']] /
                          current_box_features[:, col_indexes['boxes_duration_column']]) >= 0.7
    reliable_boxes = current_box_features[reliable_boxes_idx]
    unreliable_boxes = current_box_features[np.logical_not(reliable_boxes_idx)]

    reliable_box_data = deepcopy(current_box_data)
    unreliable_box_data = deepcopy(current_box_data)

    for box in reliable_boxes:
        start_idx, end_idx = box[: col_indexes['start_energy_column']].astype(int)
        unreliable_box_data[start_idx: (end_idx + 1)] = 0

    for box in unreliable_boxes:
        start_idx, end_idx = box[: col_indexes['start_energy_column']].astype(int)
        reliable_box_data[start_idx: (end_idx + 1)] = 0

    # -------------------------------------------- Special cases handling --------------------------------------------

    rows = dl_debug.get('rows')
    cols = dl_debug.get('cols')
    box_data = current_box_data.reshape(rows, cols)

    # Picking boxes from the predicted partitions

    box_data, dl_debug = pick_missing_l1_boxes(box_data, dl_debug)
    logger.info('DL L1 : Missing L1 boxes picked | ')

    # Identifying False Negative Partitions and picking them up

    box_data, dl_debug = identify_all_ev_l1_partitions(box_data, dl_debug)
    logger.info('DL L1 : Identifying all EV partitions complete | ')

    rows = dl_debug.get('rows')
    cols = dl_debug.get('cols')
    box_data = box_data.reshape(rows, cols)

    return box_data, dl_debug
