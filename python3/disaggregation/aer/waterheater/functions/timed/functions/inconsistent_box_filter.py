"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to find the inconsistent boxes based on across days usage / runs
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.disaggregation.aer.waterheater.functions.timed.functions.box_features import Boxes


def inconsistent_box_filter(box_info, days_limit, logger_base):
    """
    Remove the inconsistent boxes based on days adjacent to the current day

    Parameters:
        box_info            (np.ndarray)        : Boxes features
        days_limit          (int)               : Neighbour days limit
        logger_base         (dict)              : Logger object

    Returns:
        boxes               (np.ndarray)        : Updated boxes features
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('filter_boxes')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Take a local deepcopy for use within the function

    boxes = deepcopy(box_info)

    # Find the day number for boxes

    unq_days = np.unique(boxes[boxes[:, Boxes.IS_VALID] > 0, Boxes.DAY_NUMBER])

    # Get difference in days within boxes

    days_diff = np.diff(np.r_[unq_days[0] - 1, unq_days, unq_days[-1] + 1])

    # Vacant days before and after each run

    forward_diff = days_diff[:-1]
    backward_diff = days_diff[1:]

    # Get days with several vacant days before and after

    noise_days = unq_days[np.where((forward_diff >= days_limit) & (backward_diff >= days_limit))[0]]

    # Filter out the noise days

    for day in noise_days:
        boxes[boxes[:, Boxes.DAY_NUMBER] == day, Boxes.IS_VALID] = 0

    logger.info('Number of noise days removed | {}'.format(len(noise_days)))

    return boxes
