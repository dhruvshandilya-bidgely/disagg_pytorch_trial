"""
Author - Paras Tehria/ Sahana M
Date - 10/10/2018
Module to find high energy boxes in consumption data
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.ev.functions.ev_l1.get_boxes_features import boxes_features


def overlapping_box_removal(input_box_data, input_box_features, box_index, box_min_energy, factor, debug, logger_base):
    """
    Function to remove overlapping boxes
    Parameters:
        input_box_data                  (np.ndarray)        : Input box data
        input_box_features              (np.ndarray)        : Input box features
        box_index                       (float)             : Box index
        box_min_energy                  (float)             : Minimum box energy
        factor                          (float)             : Sampling rate
        debug                           (Dict)              : Debug dictionary
        logger_base                     (Logger)            : Logger

    Returns:
        box_data                        (np.ndarray)        : Box data
        box_features                    (np.ndarray)        : Box features
    """
    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('overlapping_box_removal')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    box_data = deepcopy(input_box_data)
    box_features = deepcopy(input_box_features)

    # Previous box data and features

    previous_box_features = deepcopy(debug['box_features_' + str(box_index - 1)])

    previous_box_start_idx = previous_box_features[:, 0]
    previous_box_end_idx = previous_box_features[:, 1]

    # Subtracting the energy of overlapped boxes

    for i, box in enumerate(box_features):
        start_idx, end_idx = box[:2].astype(int)

        # Check if the current box overlaps completely with existing boxes

        overlap_box = np.where((previous_box_start_idx <= start_idx) & (previous_box_end_idx >= end_idx))[0]

        if len(overlap_box) > 0:
            # If box overlaps with existing one

            overlap_box_idx = overlap_box.astype(int)[0]

            start_idx_match = (start_idx == previous_box_start_idx[overlap_box_idx])
            end_idx_match = (end_idx == previous_box_end_idx[overlap_box_idx])
            unique_overlap_check = start_idx_match & end_idx_match

            if unique_overlap_check:
                previous_min_energy = 0
            else:
                previous_min_energy = previous_box_features[overlap_box_idx, 9]
        else:
            continue

        # Remove the minimum energy of the previous box as baseline for the current box

        box_max_energy = box[10]

        if (box_max_energy - previous_min_energy) >= box_min_energy:
            box_data[start_idx:(end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] -= previous_min_energy
        else:
            box_data[start_idx:(end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    box_features = boxes_features(box_data, factor)

    logger.info('Number of boxes reduced from {} to {} | '.format(input_box_features.shape[0], box_features.shape[0]))

    return box_data, box_features
