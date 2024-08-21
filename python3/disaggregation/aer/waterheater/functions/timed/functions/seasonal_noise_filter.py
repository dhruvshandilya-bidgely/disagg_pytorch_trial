"""
Author - Nikhil
Date - 10/10/2018
Module to filter the noisy boxes within each season
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.maths_utils import rotating_sum

from python3.disaggregation.aer.waterheater.functions.timed.functions.box_features import Boxes
from python3.disaggregation.aer.waterheater.functions.timed.functions.elbow_method import elbow_method
from python3.disaggregation.aer.waterheater.functions.timed.functions.get_noise_boxes import get_noise_boxes
from python3.disaggregation.aer.waterheater.functions.timed.functions.get_outier_boxes import get_outlier_boxes
from python3.disaggregation.aer.waterheater.functions.timed.functions.get_hourly_thresholds import get_hourly_thresholds


def seasonal_noise_filter(box_data, box_info, wh_config, debug, logger_base):
    """
    Filter the noise and outliers by each season

    Parameters:
        box_data        (np.ndarray)    : Input 21-column box data
        box_info        (np.ndarray)    : Boxes features
        wh_config       (dict)          : Config params
        debug           (dict)          : Algorithm intermediate steps output
        logger_base     (dict)          : Logger object
    Returns:
        boxes           (np.ndarray)    : Filtered boxes features
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('seasonal_noise_filter')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking deepcopy of boxes features to make local instances

    boxes = deepcopy(box_info)

    # Retrieve the variables from debug object

    wh_type = debug['wh_type']
    factor = debug['time_factor']
    timed_variables = debug['variables']

    # Get the box edge indices based on water heater type (start / end)

    all_edge_idx = timed_variables['box_' + wh_type]

    # Defining the boundaries for edge time divisions

    edges = np.arange(0, (Cgbdisagg.HRS_IN_DAY * factor) + 1) - 0.5

    # Get the season mapping from the config

    season_code = wh_config['season_code']

    # Get the elbow threshold from the config

    elbow_threshold = wh_config['elbow_threshold']

    # Iterate over each season

    for season in ['wtr', 'itr', 'smr']:
        # Check if season exists

        season_idx = timed_variables[season + '_idx']

        if np.sum(season_idx) > 0:
            logger.info('Cleaning for season: | {}'.format(season))

            # Filter edges for the current season

            season_edge_idx = season_idx & all_edge_idx

            # Taking deepcopy of season boxes features to make local instances

            season_boxes = deepcopy(boxes[(boxes[:, Boxes.SEASON] == season_code[season])])

            logger.info('Initial number of season boxes: | {}'.format(np.sum(season_boxes[:, Boxes.IS_VALID])))

            # Get count of days in the current season

            season_num_days = len(np.unique(box_data[season_idx, Cgbdisagg.INPUT_DAY_IDX]))

            # Finding energy fraction at each time division for the current season

            season_hod_count, _ = np.histogram(box_data[season_edge_idx, Cgbdisagg.INPUT_HOD_IDX], bins=edges)
            season_hod_count = season_hod_count / season_num_days

            # Finding roll fraction from raw fraction for the current season

            season_hod_count = rotating_sum(season_hod_count, factor)
            season_hod_count = np.fmin(season_hod_count, 1)

            max_fraction = np.max(season_hod_count)

            # Get the breakpoint fraction using the elbow method

            breakpoint_threshold = elbow_method(season_hod_count, elbow_threshold)

            logger.debug('The max fraction and border threshold are: | {}, {}'.format(max_fraction,
                                                                                      breakpoint_threshold))

            # Extract the edge time divisions for the current season

            edge_hour = box_data[season_edge_idx, Cgbdisagg.INPUT_HOD_IDX].astype(int)

            # Update seasonal fraction for each boxes' time division

            season_boxes[:, Boxes.SEASONAL_FRACTION] = [season_hod_count[i] for i in edge_hour]

            # Marking boxes above the breakpoint fraction as significant

            max_hours = np.where(season_hod_count >= breakpoint_threshold)[0]

            # Get fraction threshold for each time division

            thresholds = get_hourly_thresholds(season_hod_count, max_fraction, max_hours, breakpoint_threshold,
                                               wh_config, debug, logger_pass)

            # Iterate over each time division

            for i in range(len(season_hod_count)):
                # Compare seasonal faction for each time division and mark 0 / 1

                season_boxes[(season_boxes[:, Boxes.TIME_DIVISION] == i) &
                             (season_boxes[:, Boxes.SEASONAL_FRACTION] < thresholds[i]), Boxes.IS_VALID] = 0

            logger.info('Number of boxes filtered due to breakpoint threshold: | {}'.
                        format(season_boxes[season_boxes[:, Boxes.IS_VALID] == 0].shape[0]))

            # If season boxes left, check for noise and outlier boxes

            if season_boxes.shape[0] > 0:
                # Remove noise boxes

                season_boxes = get_noise_boxes(season_boxes, max_hours, max_fraction, wh_config, debug, logger_pass)

                # Remove outlier boxes

                season_boxes = get_outlier_boxes(season_boxes, max_hours, box_data, debug, logger_pass)

            # Replace season boxes back to original boxes features

            boxes[(boxes[:, Boxes.SEASON] == season_code[season])] = season_boxes

            logger.info('Final number of season boxes: | {}'.format(np.sum(season_boxes[:, Boxes.IS_VALID])))

    return boxes
