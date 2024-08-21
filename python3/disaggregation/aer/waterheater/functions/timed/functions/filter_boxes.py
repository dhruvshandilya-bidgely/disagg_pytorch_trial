"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to detect if a Timed water heater present with the user
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.timed.functions.box_features import Boxes
from python3.disaggregation.aer.waterheater.functions.get_seasonal_segments import get_seasonal_segments
from python3.disaggregation.aer.waterheater.functions.timed.functions.seasonal_noise_filter import seasonal_noise_filter
from python3.disaggregation.aer.waterheater.functions.timed.functions.inconsistent_box_filter import inconsistent_box_filter
from python3.disaggregation.aer.waterheater.functions.timed.functions.inconsistent_runs_filter import inconsistent_runs_filter


def filter_boxes(input_box_data, wh_config, wh_max, debug, logger_base):
    """
    Parameters:
        input_box_data      (np.ndarray)    : Input box data matrix
        wh_config           (dict)          : Params for water heater module
        wh_max              (int)           : Max allowed energy per data point
        debug               (dict)          : Algorithm output at each step
        logger_base         (dict)          : Logger object

    Returns:
         box_data           (np.ndarray)    : Update box data
         debug              (dict)          : Update debug object
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('filter_boxes')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking deepcopy of input data to avoid scoping issues

    box_data = deepcopy(input_box_data)

    # Retrieve the boxes start and end edge boolean array

    start = debug['variables']['box_start']
    end = debug['variables']['box_end']

    # Retrieve info of the water heater type common start / end

    wh_type = debug['wh_type']

    # Get indices of the edge start and end

    start_idx = np.where(start)[0]
    end_idx = np.where(end)[0]

    # Retrieve info of the energy fraction

    edge_hod_count = debug['variables'][wh_type + '_count']

    # Append start and end edge boolean to box data

    box_data = np.hstack((box_data, start.reshape(-1, 1), end.reshape(-1, 1)))

    # Initialize features for each box in the box data with start and end indices

    boxes = np.hstack((start_idx.reshape(-1, 1), end_idx.reshape(-1, 1)))

    # Create blank columns for other box features

    boxes = np.hstack((boxes, np.ones(shape=(boxes.shape[0], Boxes.NUM_COLS))))

    logger.info('Number of boxes found | {}'.format(boxes.shape[0]))

    if wh_type == 'start':
        # If common start type timed water heater

        start_hour = box_data[start_idx, Cgbdisagg.INPUT_HOD_IDX].astype(int)
        boxes[:, Boxes.BOX_FRACTION] = [edge_hod_count[i] for i in start_hour]
        boxes[:, Boxes.TIME_DIVISION] = start_hour
    else:
        # If common end type timed water heater

        end_hour = box_data[end_idx, Cgbdisagg.INPUT_HOD_IDX].astype(int)
        boxes[:, Boxes.BOX_FRACTION] = [edge_hod_count[i] for i in end_hour]
        boxes[:, Boxes.TIME_DIVISION] = end_hour

    # #------------------------- FILTER-1: Remove insignificant / minor runs ------------------------------------------#

    box_data, boxes, debug = inconsistent_runs_filter(box_data, boxes, edge_hod_count, wh_config, debug, logger_pass)

    # Save the interim box data to debug object

    debug['interim_box'] = deepcopy(box_data)

    # If number of runs became zero, return module

    if debug['num_runs'] == 0:
        logger.info('All runs removed in the energy check |')

        return box_data, debug

    # #------------------------- FILTER-2: Seasonal noise removal filter ----------------------------------------------#

    # Water heater type mapping dictionary

    idx_col = {'start': Boxes.START_IDX, 'end': Boxes.END_IDX}

    # Separate data for each season

    season_temp, debug, boxes = get_seasonal_segments(box_data, [boxes, idx_col[wh_type]], debug, logger_pass,
                                                      wh_config)

    # Filtering boxes within each season

    boxes = seasonal_noise_filter(box_data, boxes, wh_config, debug, logger_pass)

    # #------------------------- FILTER-3: Inconsistent boxes removal -------------------------------------------------#

    # Remove boxes with no other boxes in certain vicinity

    boxes = inconsistent_box_filter(boxes, wh_config['vicinity_days_bound'], logger_pass)

    # Get energy diff to check if upscale and downscale is consistent for a box

    diff_box_energy = np.abs(np.diff(np.r_[0, box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], 0]))

    # Initialize bool array to mark final timed water heater indices

    final_box_idx = np.array([False] * box_data.shape[0])

    # #------------------------- FILTER-4: Final boxes validity check -------------------------------------------------#

    # Iterate over each box

    for i in range(boxes.shape[0]):
        # Get start and end of each box

        s, e = boxes[i, [Boxes.START_IDX, Boxes.END_IDX]].astype(int)

        # Energy diff with the edge neighbours

        start_energy = diff_box_energy[s]
        end_energy = diff_box_energy[e + 1]

        # Adding energy values to boxes features

        boxes[i, [Boxes.START_ENERGY, Boxes.END_ENERGY]] = start_energy, end_energy

        # Removing the boxes where diff value higher than max allowed

        if np.mean(boxes[i, [Boxes.START_ENERGY, Boxes.END_ENERGY]]) > wh_max:
            boxes[i, Boxes.IS_VALID] = 0

        # Updating the box data based on final validity flag of the current box

        if boxes[i, Boxes.IS_VALID] == 0:
            # Box is invalid, mask consumption values zero

            box_data[s:(e + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        elif boxes[i, Boxes.IS_VALID] == 1:
            # Box is valid, retain consumption values

            final_box_idx[s:(e + 1)] = True
            box_data[s:(e + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] -= 0

    # Mask all the non-box values as zero

    box_data[~final_box_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    return box_data, debug
