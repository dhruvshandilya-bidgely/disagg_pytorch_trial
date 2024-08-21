"""
Author - Paras Tehria
Date - 18-Feb-2020
Module for removing doubtful ev estimation boxes
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.maths_utils import moving_sum
from python3.disaggregation.aer.ev.functions.detection.get_boxes_features import boxes_features


def remove_doubtful_boxes(debug, ev_config, box_data, input_box_features, logger_base):
    """
    Function for removing doubtful ev estimation boxes

    Parameters:
        debug               (dict)                  : Dictionary containing output of each step
        ev_config            (dict)                  : Configuration for the algorithm
        box_data            (np.ndarray)            : EV box data
        input_box_features  (np.ndarray)            : EV boxes features
        logger_base         (logger)                : logger pass

    Returns:
        box_data            (np.ndarray)            : Updated ev estimation matrix
        box_features        (np.ndarray)            : Features of new boxes
    """
    features_column_dict = ev_config.get('box_features_dict')
    local_config = ev_config.get('est_boxes_refine_config')

    logger_local = logger_base.get('logger').getChild('remove_doubtful_boxes')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    box_data = deepcopy(box_data)
    box_features = deepcopy(input_box_features)

    amp_col = features_column_dict['boxes_energy_per_point_column']
    dur_col = features_column_dict.get('boxes_duration_column')

    clean_box_amp = debug.get('clean_box_amp')
    clean_box_dur = debug.get('clean_box_dur')

    low_amp_boxes = (box_features[:, amp_col] <= local_config.get('low_amp_ratio') * clean_box_amp)
    removal_boxes = box_features[low_amp_boxes, :]

    logger.info("Total number of low amp boxes in estimation: | {}".format(np.sum(low_amp_boxes)))

    # Removing info of erroneous boxes from box data
    for idx, row in enumerate(removal_boxes):
        start_idx, end_idx = row[:features_column_dict['end_index_column'] + 1].astype(int)
        box_data[start_idx: end_idx + 1, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    box_features = boxes_features(box_data, debug.get('factor'), ev_config)

    # Capping duration of high duration boxes
    max_allowed_dur = int(max(local_config.get('hi_dur_ratio') * clean_box_dur, local_config.get('hi_dur_hour_thresh')))
    high_dur_bool = (box_features[:, dur_col] > max_allowed_dur)
    high_dur_boxes = box_features[high_dur_bool, :]

    logger.info("Total number of high duration boxes in estimation: | {}".format(np.sum(high_dur_bool)))

    clean_box_energy = clean_box_amp / debug.get('factor')
    max_dur_smplng_pts = int(max_allowed_dur * debug.get('factor'))

    # For High duration, keeping the period with amplitude closest to clean amplitude
    for i, row in enumerate(high_dur_boxes):
        start_idx, end_idx = row[:features_column_dict['end_index_column'] + 1].astype(int)
        box_energy = box_data[start_idx:end_idx + 1, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # taking the moving average to find continuous time-period with amp closest to clean boxes amp
        rolling_amp = moving_sum(box_energy, max_dur_smplng_pts) / max_dur_smplng_pts
        closest_amp_idx = np.argmin(np.abs(rolling_amp-clean_box_energy))

        new_start_idx = max(start_idx, start_idx + closest_amp_idx - (max_dur_smplng_pts - 1))
        new_end_idx = min(new_start_idx + max_dur_smplng_pts, end_idx + 1)

        box_data[start_idx: new_start_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0
        box_data[new_end_idx: end_idx + 1, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    box_features = boxes_features(box_data, debug.get('factor'), ev_config)

    return box_data, box_features
