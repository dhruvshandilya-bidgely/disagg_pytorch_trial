"""
Author - Paras Tehria / Sahana M
Date - 1-Feb-2022
Generic Module to get box features
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import packages from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def boxes_features(box_input, factor):
    """
    Function used to get the box features
    Parameters:
        box_input           (np.ndarray)        : EV L1 boxes captured
        factor              (float)             : Sampling rate w.r.t 60 minutes
    Returns:
        box_features            (np.ndarray)    : All the boxes identified and their features
    """

    # Taking local deepcopy of the ev boxes

    box_data = deepcopy(box_input)

    # Extraction energy data of boxes

    box_energy = deepcopy(box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Taking only positive box boundaries for edge related calculations

    box_energy_idx = (box_energy > 0)
    box_energy_idx_diff = np.diff(np.r_[0, box_energy_idx.astype(int), 0])

    # Find the start and end edges of the boxes

    box_start_boolean = (box_energy_idx_diff[:-1] > 0)
    box_end_boolean = (box_energy_idx_diff[1:] < 0)

    box_start_idx = np.where(box_start_boolean)[0]
    box_end_idx = np.where(box_end_boolean)[0]

    box_start_energy = box_energy[box_start_idx]
    box_end_energy = box_energy[box_end_idx]

    box_features = np.c_[box_start_idx, box_end_idx, box_start_energy, box_end_energy]

    # Duration of boxes in hours

    box_duration = (np.diff(box_features[:, :2], axis=1) + 1) / factor

    box_energy_std = np.array([])
    box_areas = np.array([])
    box_energy_per_point = np.array([])
    box_median_energy = np.array([])
    box_minimum_energy = np.array([])
    box_maximum_energy = np.array([])

    for i, row in enumerate(box_features):
        start_idx, end_idx = row[:2].astype(int)

        temp_energy = box_energy[start_idx: (end_idx + 1)]

        # Calculate the absolute deviation

        temp_energy_std = np.mean(np.abs(temp_energy - np.mean(temp_energy)))

        # Total energy of the box

        temp_area = np.sum(temp_energy)

        # Energy per hour in the box

        temp_energy_per_point = temp_area / box_duration[i]
        temp_box_median_energy = np.median(temp_energy)

        temp_box_minimum_energy = np.min(temp_energy)
        temp_box_maximum_energy = np.max(temp_energy)

        box_energy_std = np.append(box_energy_std, temp_energy_std)
        box_areas = np.append(box_areas, temp_area)
        box_energy_per_point = np.append(box_energy_per_point, temp_energy_per_point)
        box_median_energy = np.append(box_median_energy, temp_box_median_energy)
        box_minimum_energy = np.append(box_minimum_energy, temp_box_minimum_energy)
        box_maximum_energy = np.append(box_maximum_energy, temp_box_maximum_energy)

    box_features = np.c_[box_features, box_duration, box_areas, box_energy_std, box_energy_per_point,
                         box_median_energy, box_minimum_energy, box_maximum_energy]

    box_start_day = box_data[box_features[:, 0].astype(int), Cgbdisagg.INPUT_DAY_IDX]
    boxes_start_hod = box_data[box_features[:, 0].astype(int), Cgbdisagg.INPUT_HOD_IDX]
    boxes_start_month = box_data[box_features[:, 0].astype(int), Cgbdisagg.INPUT_DIMENSION]
    boxes_start_season = box_data[box_features[:, 0].astype(int), Cgbdisagg.INPUT_DIMENSION + 1]

    box_features = np.c_[box_features, boxes_start_hod, boxes_start_month, boxes_start_season, box_start_day]

    return box_features
