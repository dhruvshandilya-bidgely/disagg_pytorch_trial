"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Updating the boxes based on the final significant time divisions
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.timed.functions.box_features import Boxes


def update_boxes_features(box_data, boxes, high_fraction_hours, debug, logger_base):
    """
    Removing the zero boxes data from box data

    Parameters:
        box_data                (np.ndarray)        : Input 21-column box data
        boxes                   (np.ndarray)        : Boxes features
        high_fraction_hours     (np.ndarray)        : High energy fraction hours
        debug                   (dict)              : Algorithm intermediate steps output
        logger_base             (dict)              : Logger object

    Returns:
        box_data                (np.ndarray)        : Updated box data
        boxes                   (np.ndarray)        : Updated boxes features
        debug                   (dict)              : Updated debug object
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('update_boxes_features')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the water heater type and factor info

    wh_type = debug['wh_type']

    factor = debug['time_factor']
    n_divisions = Cgbdisagg.HRS_IN_DAY * factor

    # Identify the boxes that are marked as invalid

    zero_boxes_idx = (boxes[:, Boxes.IS_VALID] == 0)

    # Based on water heater type, get the start / end time divisions of all boxes

    if wh_type == 'start':
        # Get the start idx of the boxes

        start_hour = boxes[:, Boxes.TIME_DIVISION]

        # Calculate the end idx of the boxes

        end_hour = (boxes[:, Boxes.END_IDX] - boxes[:, Boxes.START_IDX]) + boxes[:, Boxes.TIME_DIVISION]
        end_hour[end_hour > (n_divisions - 1)] -= n_divisions
    else:
        # Get the end idx of the boxes

        end_hour = boxes[:, Boxes.TIME_DIVISION]

        # Calculate the start idx of the boxes

        start_hour = boxes[:, Boxes.TIME_DIVISION] - (boxes[:, Boxes.END_IDX] - boxes[:, Boxes.START_IDX])
        start_hour[start_hour < 0] += n_divisions

    # Iterate over the boxes marked invalid

    for i in np.where(zero_boxes_idx)[0]:
        # Get the start and end idx of the current box

        start_idx, end_idx = boxes[i, :2].astype(int)

        # Get the start and end time division of the current box

        start, end = start_hour[i], end_hour[i]

        # Get run time divisions based on overnight run or within the day run

        if start <= end:
            # If the run is within the day

            run_hours = np.arange(start, end + 1)
        else:
            # If the run is overnight

            run_hours = np.r_[np.arange(start, n_divisions), np.arange(0, end + 1)]

        # Check if the current box time division have any significant hours

        max_hour_check = np.isin(run_hours, high_fraction_hours)

        # If a significant time division found in the box

        if np.sum(max_hour_check):
            # Significant time division found in the current box

            boxes[i, Boxes.IS_VALID] = 1

            # Redefine the edge to preserve the significant time divisions

            if wh_type == 'start':
                # If start type water heater, update the start index

                # Cut the time divisions before the first overlapped significant time division

                to_cut = np.where(max_hour_check)[0][0]

                # Update the box start index

                boxes[i, Boxes.START_IDX] += to_cut
                boxes[i, Boxes.TIME_DIVISION] += to_cut

                # Update the box data

                end_idx = (start_idx + to_cut)
                box_data[start_idx:end_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0
            else:
                # If end type water heater, update the end index

                # Cut the time divisions after the last overlapped significant time division

                to_cut = np.where(max_hour_check)[0][-1]

                # Update the box start index

                boxes[i, Boxes.END_IDX] -= (len(max_hour_check) - to_cut - 1)
                boxes[i, Boxes.TIME_DIVISION] -= (len(max_hour_check) - to_cut - 1)

                # Update the box data

                start_idx = end_idx - (len(max_hour_check) - to_cut - 1)
                box_data[(start_idx + 1):(end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

            boxes[i, Boxes.TIME_DIVISION] = mark_time_division_valid(boxes[i, Boxes.TIME_DIVISION], n_divisions)
        else:
            # No significant time division found in the current box, update box data

            box_data[start_idx:(end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    logger.info('Number of insignificant boxes removed: | {}'.format(boxes[boxes[:, Boxes.IS_VALID] == 0].shape[0]))

    # Updating the edge indices based on updated box data

    box_energy_idx = (box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0).astype(int)
    box_idx_diff = np.diff(np.r_[0, box_energy_idx, 0])

    # Recalculate the start / end edge indices based on water heater type

    if wh_type == 'start':
        box_edge_idx = (box_idx_diff[:-1] > 0)
    else:
        box_edge_idx = (box_idx_diff[1:] < 0)

    # Update the edge inidces in the debug object

    debug['variables']['box_' + wh_type] = box_edge_idx

    # Update the boxes features by removing the invalid marked boxes

    boxes = boxes[boxes[:, Boxes.IS_VALID] > 0]

    return box_data, boxes, debug


def mark_time_division_valid(hour, n_divisions):
    """
    Make invalid declared significant time divisions as valid

    Parameters:
        hour            (int)       : Time division
        n_divisions     (int)       : Number of time divisions

    Returns:
        hour            (int)       : Updated time division
    """

    # If the time division is beyond the accepted range

    if hour < 0:
        # If time division less than zero, add n_divisions

        hour += n_divisions

    elif hour >= n_divisions:
        # If time division more than or equal to n_divisions, subtract n_divisions

        hour -= n_divisions

    return hour
