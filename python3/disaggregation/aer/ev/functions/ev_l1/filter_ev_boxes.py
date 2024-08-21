"""
Author - Paras Tehria / Sahana M
Date - 1-Feb-2022
Module to get potential EV boxes
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def filter_ev_boxes(current_box_data, in_data, factor, ev_config, logger_base):
    """
    This function is used to filter the EV boxes
    Parameters:
        current_box_data            (np.ndarray)        : Current box data
        in_data                     (np.ndarray)        : Input data
        factor                      (float)             : Sampling rate w.r.t 60 minutes
        ev_config                   (Dict)              : EV configurations
        logger_base                 (logger)            : logger pass
    Returns:
        current_box_data            (np.ndarray)        : Current box data
    """

    logger_local = logger_base.get('logger').getChild('filter_ev_boxes')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking local deepcopy of the ev boxes

    box_data = deepcopy(current_box_data)
    input_data = deepcopy(in_data)

    # Extraction energy data of boxes

    boxes_energy = deepcopy(box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    input_data_energy = deepcopy(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    logger.info("Total EV L1 consumption before filtering | {}".format(np.nansum(boxes_energy)))

    # Taking only positive box boundaries for edge related calculations

    box_energy_idx = (boxes_energy > 0)
    box_energy_idx_diff = np.diff(np.r_[0, box_energy_idx.astype(int), 0])

    # Find the start and end edges of the boxes

    box_start_boolean = (box_energy_idx_diff[:-1] > 0)
    box_end_boolean = (box_energy_idx_diff[1:] < 0)

    box_start_indices = np.where(box_start_boolean)[0]
    box_end_indices = np.where(box_end_boolean)[0]

    box_start_end_arr = np.c_[box_start_indices, box_end_indices]

    window_size = ev_config.get('minimum_duration', {}).get(ev_config.get('sampling_rate'))

    for i, row in enumerate(box_start_end_arr):
        box_start_idx = row[0]
        box_end_idx = row[1]

        box_left_idx = max(0, box_start_idx - int(window_size * factor))

        box_right_idx = min(box_end_idx + int(window_size * factor), len(box_data) - 1)

        left_min = 0
        right_min = 0

        num_pts_day = int(Cgbdisagg.HRS_IN_DAY * factor)

        # Identify the left and right box indexes

        last_day_left_idx = box_left_idx - num_pts_day
        last_day_start_idx = box_start_idx - num_pts_day
        last_day_right_idx = box_right_idx - num_pts_day

        if box_start_idx > 0:
            left_min = np.nanmin(input_data_energy[box_left_idx: box_start_idx])

        if last_day_left_idx >= 0:
            left_min = np.nanmin(np.r_[left_min, input_data_energy[last_day_left_idx: last_day_start_idx]])

        if box_end_idx < len(box_data) - 1:
            right_min = np.nanmin(input_data_energy[box_end_idx + 1: box_right_idx + 1])

        if last_day_right_idx <= len(box_data) - 1:
            right_min = np.nanmin(np.r_[right_min, input_data_energy[box_end_idx - num_pts_day + 1:last_day_right_idx + 1]])

        # Get the base energy by taking the maximum of left and right minimum

        base_energy = np.nanmax([left_min, right_min])

        boxes_energy[box_start_idx: (box_end_idx + 1)] -= base_energy

        current_box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = boxes_energy

    logger.info("Total EV L1 consumption after filtering | {}".format(np.nansum(boxes_energy)))

    return current_box_data
