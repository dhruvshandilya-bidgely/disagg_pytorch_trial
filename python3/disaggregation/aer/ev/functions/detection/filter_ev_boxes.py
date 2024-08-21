"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to get potential EV boxes
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.ev.functions.get_ev_boxes import get_ev_boxes


def filter_ev_boxes(current_box_data, in_data, factor, ev_config, min_energy, logger_pass):
    """
    This function is used to filter EV boxes
    Parameters:
        current_box_data           (np.ndarray)         : Current box data features
        in_data                    (np.ndarray)         : Array containing input data
        factor                     (int)                : Sampling rate division
        ev_config                  (dict)               : Ev configurations
        min_energy                 (float)              : Minimum energy
        logger_pass                (logger)             : Logging object to log important steps and values in the run

    Returns:
        current_box_data           (np.ndarray)         : Current box data features
    """

    # Taking local deepcopy of the ev boxes

    box_data = deepcopy(current_box_data)
    input_data = deepcopy(in_data)

    # Extraction energy data of boxes

    boxes_energy = deepcopy(box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    input_data_energy = deepcopy(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Taking only positive box boundaries for edge related calculations

    box_energy_idx = (boxes_energy > 0)
    box_energy_idx_diff = np.diff(np.r_[0, box_energy_idx.astype(int), 0])

    # Find the start and end edges of the boxes

    box_start_boolean = (box_energy_idx_diff[:-1] > 0)
    box_end_boolean = (box_energy_idx_diff[1:] < 0)

    box_start_indices = np.where(box_start_boolean)[0]
    box_end_indices = np.where(box_end_boolean)[0]

    box_start_end_arr = np.c_[box_start_indices, box_end_indices]

    minimum_duration = ev_config.get('minimum_duration', {}).get(ev_config.get('sampling_rate'))

    # Perform filtering for each EV box

    for i, row in enumerate(box_start_end_arr):
        box_start_idx = row[0]
        box_end_idx = row[1]

        # Get all the left and right indexes with a buffer

        box_left_idx = max(0, box_start_idx - int(minimum_duration * factor))
        box_right_idx = min(box_end_idx + int(minimum_duration * factor), len(box_data) - 1)

        left_min = 0
        right_min = 0

        num_pts_day = int(Cgbdisagg.HRS_IN_DAY * factor)

        # get the required variables for the last day

        last_day_left_idx = box_left_idx - num_pts_day
        last_day_start_idx = box_start_idx - num_pts_day
        last_day_right_idx = box_right_idx - num_pts_day

        # Extract the minimum energy from the left side of the box

        if box_start_idx > 0:
            left_min = np.nanmin(input_data_energy[box_left_idx: box_start_idx])

        if last_day_left_idx >= 0:
            left_min = np.nanmin(np.r_[left_min, input_data_energy[last_day_left_idx: last_day_start_idx]])

        # Extract the minimum energy from the right side of the box

        if box_end_idx < len(box_data) - 1:
            right_min = np.nanmin(input_data_energy[box_end_idx + 1: box_right_idx + 1])

        if last_day_right_idx <= len(box_data) - 1:
            right_min = np.nanmin(
                np.r_[right_min, input_data_energy[box_end_idx - num_pts_day + 1:last_day_right_idx + 1]])

        base_energy = np.nanmax([left_min, right_min])

        # Subtract the minimum energy from the EV box

        boxes_energy[box_start_idx: (box_end_idx + 1)] = np.fmax(
            boxes_energy[box_start_idx: (box_end_idx + 1)] - base_energy, 0)

    current_box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = boxes_energy

    region = ev_config.get('region')
    min_box_energy = ev_config['detection'][region + '_start_box_amplitude'] / factor

    current_box_data = get_ev_boxes(current_box_data, minimum_duration, max(min_energy * 0.75, min_box_energy), factor,
                                    logger_pass)

    return current_box_data
