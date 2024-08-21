"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to calculate confidence of timed water heater
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def timed_confidence(debug, input_box_data, wh_config, edge_idx, wh_type):
    """
    Parameters:
        debug           (dict)          : Algorithm steps output
        input_box_data  (np.ndarray)    : Box data matrix
        wh_config       (dict)          : Config params for timed water heater
        edge_idx        (np.ndarray)    : Indices of box edges
        wh_type         (str)           : Water heater type based on edge (start/end)

    Returns:
        debug           (dict)          : Updated debug with confidence score
    """

    # Taking deepcopy of input box data

    box_data = deepcopy(input_box_data)

    # Extract time factor and number of time divisions for the data

    factor = debug['time_factor']
    n_divisions = Cgbdisagg.HRS_IN_DAY * factor

    # Retrieve the maximum raw and roll fractions

    max_raw = np.max(debug['max_count_raw'])
    max_roll = np.max(debug['max_count_roll'])

    valid_hours_range = wh_config['valid_hours_range']

    # Find the time division of max raw fraction

    if wh_type == 'start':
        max_time_division = np.argmax(debug['pos_count_raw'])
    else:
        max_time_division = np.argmax(debug['neg_count_raw'])

    # Define valid time divisions as -+ valid hours of the max fraction time division

    valid_time_divisions = np.arange(max_time_division - (valid_hours_range * factor),
                                     max_time_division + (valid_hours_range * factor))

    # Check if time divisions out ot valid range

    valid_time_divisions[valid_time_divisions >= n_divisions] -= n_divisions
    valid_time_divisions[valid_time_divisions < 0] += n_divisions

    # Get unique days data from debug and increment by 1 to get day number

    days, day_number = debug['box_days_ts'], debug['box_days_idx']
    day_number += 1

    # Extract the time division of all edge data points and the corresponding day number

    edge_hours = box_data[edge_idx, Cgbdisagg.INPUT_HOD_IDX]
    edge_day_number = day_number[edge_idx]

    # If no edge left, return with zero confidence

    if len(edge_hours) == 0:
        # If no edge left, return

        debug['timed_confidence'] = 0.0000

        return debug

    # Check which time divisions are valid and keep them

    edge_hours_in_valid = np.in1d(edge_hours, valid_time_divisions)

    # Retain the valid time divisions and the corresponding valid days

    valid_edge_hours = edge_hours[edge_hours_in_valid]
    valid_edge_days = edge_day_number[edge_hours_in_valid]

    # Check for repetitive days and remove them

    days_diff = np.r_[1, np.diff(valid_edge_days)]
    valid_days_idx = days_diff > 0

    valid_edge_hours = valid_edge_hours[valid_days_idx]

    # Find the chunks of equal time divisions in the edge data

    hours_diff = np.abs(np.r_[0, np.diff(valid_edge_hours)])
    hours_diff[hours_diff >= Cgbdisagg.HRS_IN_DAY] = n_divisions - hours_diff[hours_diff >= Cgbdisagg.HRS_IN_DAY]

    # Aggregate the continuous chunks and find fraction out of total days to get days based confidence score

    days_confidence = np.sum(hours_diff == 0) / len(hours_diff)

    # Take mean of max raw fraction and max roll fraction to get hour based confidence score

    hours_confidence = (max_raw + max_roll) / 2

    # Overall confidence score is mean of days and hours based confidence scores

    overall_confidence = (days_confidence + hours_confidence) / 2

    # Save confidence score to debug object

    debug['timed_confidence'] = np.round(overall_confidence, 4)

    return debug
