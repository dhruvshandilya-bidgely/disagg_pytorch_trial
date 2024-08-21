"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to remove overlapping boxes
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.ev.functions.detection.get_boxes_features import boxes_features


def remove_overlapping_boxes(input_box_data, input_box_features, box_index, box_min_energy, factor, debug, ev_config,
                             logger_base):
    """
    Function to remove overlapping boxes

        Parameters:
            input_box_data            (np.ndarray)        : Current box data
            input_box_features        (np.ndarray)        : Current Box Features
            debug                     (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
            factor                    (int)               : Number of data points in an hour
            box_min_energy            (float)              : Minimum energy of the boxes
            box_index                 (int)               : Index of the box data
            ev_config                  (dict)              : EV module config
            logger_base               (logger)            : Logging object to log important steps and values in the run

        Returns:
            box_data                  (np.ndarray)        : Updated box data
            box_features              (np.ndarray)        : Updated Box Features
    """
    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('overlapping_box_removal')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    box_data = deepcopy(input_box_data)
    box_features = deepcopy(input_box_features)

    # Previous box data and features

    previous_box_features = deepcopy(debug.get('box_features_' + str(box_index - 1)))

    previous_box_start_idx = previous_box_features[:, 0]
    previous_box_end_idx = previous_box_features[:, 1]

    # Subtracting the energy of overlapped boxes

    for i, box in enumerate(box_features):
        start_idx, end_idx = box[:ev_config['box_features_dict']['end_index_column']+1].astype(int)

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
                previous_min_energy = previous_box_features[
                    overlap_box_idx, ev_config['box_features_dict']['boxes_minimum_energy']]
        else:
            continue

        # Remove the minimum energy of the previous box as baseline for the current box

        box_max_energy = box[ev_config['box_features_dict']['boxes_maximum_energy']]

        if (box_max_energy - previous_min_energy) >= box_min_energy:
            box_data[start_idx:(end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] -= previous_min_energy
        else:
            box_data[start_idx:(end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    box_features = boxes_features(box_data, factor, ev_config)

    logger.info('Number of boxes reduced from {} to {} | '.format(input_box_features.shape[0], box_features.shape[0]))

    return box_data, box_features
