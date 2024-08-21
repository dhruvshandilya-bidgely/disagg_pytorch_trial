"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to get the noise boxes within a season
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.timed.functions.box_features import Boxes


def get_noise_boxes(box_info, max_hours, max_fraction, wh_config, debug, logger_base):
    """
    Removing the noise boxes

    Parameters:
        box_info        (np.ndarray)        : Boxes features
        max_hours       (np.ndarray)        : High energy fraction hours
        max_fraction    (float)             : Max energy fraction
        wh_config       (dict)              : Timed water heater params
        debug           (dict)              : Algorithm intermediate steps output
        logger_base     (dict)              : Logger object

    Returns:
        boxes           (np.ndarray)        : Updated boxes features (seasonal)
    """
    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_noise_boxes')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking deepcopy of boxes features to make local instances

    boxes = deepcopy(box_info)

    # Retrieve the required params from config

    elbow_threshold = wh_config['elbow_threshold']
    lower_duration_bound = wh_config['lower_duration_bound']
    upper_duration_bound = wh_config['upper_duration_bound']
    max_duration_bound = wh_config['max_duration_bound']

    # Retrieve the variables from debug object

    factor = debug['time_factor']
    n_divisions = Cgbdisagg.HRS_IN_DAY * factor

    wh_type = debug['wh_type']
    num_runs = debug['num_runs']

    # Saving original max hours

    org_max_hours = debug['max_hours']

    # #---------------------------- Chop over-extended boxes in non-significant hours ---------------------------------#

    # Initializing the validity bool array for all boxes

    org_max_boxes = np.array([True] * boxes.shape[0])

    # Iterate over max hours (high energy fraction time divisions)

    for hour in org_max_hours:
        # If the boxes edge match with significant hour, mask it valid

        org_max_boxes[(boxes[:, Boxes.TIME_DIVISION] == hour)] = False

    # Append the sampling rate factor to max hours

    max_hours = np.r_[(max_hours - factor), max_hours, (max_hours + factor)]

    # Adjust the edge time divisions to valid range

    max_hours[max_hours < 0] += n_divisions
    max_hours[max_hours > (n_divisions - 1)] -= n_divisions

    # Take unique max hours

    max_hours = np.unique(max_hours)

    # Initializing the validity bool array for all boxes

    invalid_boxes = np.array([False] * boxes.shape[0])
    max_boxes = np.array([True] * boxes.shape[0])

    # Iterate over the updated max hours

    for hour in max_hours:
        # If the boxes edge match with significant hour, mask it valid

        max_boxes[(boxes[:, Boxes.TIME_DIVISION] == hour)] = False

    # Calculate the run duration for each box (in time divisions)

    run_lengths = (boxes[:, Boxes.END_IDX] - boxes[:, Boxes.START_IDX]) + 1

    total_runs = len(run_lengths)

    # Calculate the median run duration

    median_run_len = np.median(run_lengths)

    # Chopping off abnormally short and long runs away from significant time divisions

    invalid_boxes[run_lengths < (lower_duration_bound * median_run_len)] = True
    invalid_boxes[run_lengths > (upper_duration_bound * median_run_len)] = True

    # Taking and of two invalid conditions

    final_invalid_boxes = invalid_boxes & max_boxes

    boxes[final_invalid_boxes, Boxes.IS_VALID] = 0

    logger.debug('Number of abnormal length boxes away from important hours: | {}'.format(np.sum(final_invalid_boxes)))

    # #---------------------------- Chop over-extended boxes in significant hours -------------------------------------#

    # Get abnormal boxes over or close to significant time divisions

    chop_boxes = (run_lengths > (upper_duration_bound * median_run_len)) & (~max_boxes)
    chop_boxes_idx = np.where(chop_boxes)[0]

    logger.debug('Number of abnormal length boxes close to important hours | {}'.format(len(chop_boxes_idx)))

    # Chop abnormal boxes overlapping with significant hours

    if wh_type == 'start':
        # If common start water heater, get the indices of the start edge

        start_hour = boxes[:, Boxes.TIME_DIVISION]

        # Calculate the end hour and make it valid

        end_hour = (boxes[:, Boxes.END_IDX] - boxes[:, Boxes.START_IDX]) + boxes[:, Boxes.TIME_DIVISION]
        end_hour[end_hour > (n_divisions - 1)] -= n_divisions
    else:
        # If common end water heater, get the indices of the end edge

        end_hour = boxes[:, Boxes.TIME_DIVISION]

        # Calculate the start hour and make it valid

        start_hour = boxes[:, Boxes.TIME_DIVISION] - (boxes[:, Boxes.END_IDX] - boxes[:, Boxes.START_IDX])
        start_hour[start_hour < 0] += n_divisions

    # Iterate over each box that is marked as over extended

    for i in chop_boxes_idx:
        # Get the start and end indices for current box

        start, end = start_hour[i], end_hour[i]

        # Get the run time divisions

        if start <= end:
            # If complete run within day

            run_hours = np.arange(start, end + 1)
        else:
            # If run is overnight

            run_hours = np.r_[np.arange(start, n_divisions), np.arange(0, end + 1)]

        # Check if the current box time divisions in high energy time divisions

        max_hour_check = np.isin(run_hours, max_hours)

        if np.sum(max_hour_check):
            # If current box has high energy time divisions, check the run duration of neighbor days

            neighbor_lengths = get_neighbor_run_len(run_lengths, total_runs, i)

            # Get the optimal run duration for current box

            current_length = np.min([run_lengths[i], upper_duration_bound * neighbor_lengths,
                                     max_duration_bound * median_run_len])

            # Adjust the edge index for the water heater

            if wh_type == 'start':
                # Adjust end index, if common start type

                boxes[i, Boxes.END_IDX] -= (len(max_hour_check) - current_length)
            else:
                # Adjust start index, if common end type

                boxes[i, Boxes.START_IDX] += (len(max_hour_check) - current_length)

            boxes[i, Boxes.IS_VALID] = 1

    # #------------------------------- Remove more than valid daily number of runs ------------------------------------#

    # Get unique days from boxes features

    unq_days, days_idx = np.unique(boxes[:, Boxes.DAY_NUMBER], return_inverse=True)

    # Get the number of runs for each day

    daily_run_count = np.bincount(days_idx, np.array([True] * boxes.shape[0]))

    # Mark days that have more than the optimal number of runs

    days_to_fix = unq_days[daily_run_count > num_runs]

    logger.debug('Days with for more boxes than number of runs: | {}'.format(len(days_to_fix)))

    # Define the error limit

    error_epsilon = 0.001

    # Iterate over each marked day boxes

    for day in days_to_fix:
        # Subset boxes of the current day

        temp_boxes = boxes[boxes[:, Boxes.DAY_NUMBER] == day]

        # Get the time divisions and edge fraction for current day

        fractions = temp_boxes[:, Boxes.BOX_FRACTION]
        time_divisions = temp_boxes[:, Boxes.TIME_DIVISION]

        # Mark the overlapped time divisions with significant hours as valid

        valid_hours = np.isin(time_divisions, max_hours)

        if (np.sum(valid_hours) > num_runs) and (np.min(fractions) < (elbow_threshold * max_fraction)):
            # If the valid hours have more than optimal runs, keep the highest fractions upto num_runs

            threshold = np.sort(fractions)[-num_runs] - error_epsilon
            valid_hours = (fractions > threshold)

        # Update the boxes features using valid hours info

        boxes[(boxes[:, Boxes.DAY_NUMBER] == day), Boxes.IS_VALID] = valid_hours.astype(int)

    return boxes


def get_neighbor_run_len(run_lens, total_runs, i):
    """
    Length of timed water heater run in the neighbour days

    Parameters:
        run_lens        (np.ndarray)    : Run duration for all runs
        total_runs      (int)           : Number of total runs
        i               (int)           : Current run index
    Returns:
        out_run         (int)           : Neighbour run duration
    """

    if i == 0:
        # If first run, then only one neighbour

        out_run = run_lens[i + 1]

    elif i == (total_runs - 1):
        # If last run, then only one neighbour

        out_run = run_lens[i - 1]
    else:
        # If run in between, then two neighbours

        out_run = (run_lens[i - 1] + run_lens[i + 1]) // 2

    return out_run
