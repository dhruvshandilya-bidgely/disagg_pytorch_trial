"""
Date Created - 13 Nov 2018
Author name - Pratap
For Consolidating overlapping LAPs
"""

import logging
import numpy as np

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.refrigerator.functions.get_lap_lowpoint import get_lap_low_point


def get_consolidated_laps(ref_detection, config, logger_base):
    """
    Combines overlapping LAPs to get one LAP
    Separates truncated points from combined LAP which are corrected
    """
    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("get_consolidated_laps")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_base.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    input_data = ref_detection['input_data']

    # Finding the start and stop timestamps of each LAP
    low_limit = ref_detection['lapMidTimestamps'][:, 0] - ((config['LAPDetection']['lapHalfGap']) * Cgbdisagg.SEC_IN_HOUR)
    high_limit = ref_detection['lapMidTimestamps'][:, 0] + ((config['LAPDetection']['lapHalfGap']) * Cgbdisagg.SEC_IN_HOUR)
    low_limit = low_limit[:, np.newaxis]
    high_limit = high_limit[:, np.newaxis]

    # Find the low points within each LAP
    ref_detection['LAPs'] = np.hstack((low_limit, high_limit, ref_detection['lapMidTimestamps'][:, 1][:, np.newaxis]))
    lap_low_point, trnctd_laps = get_lap_low_point(ref_detection['input_data'], ref_detection['LAPs'], ref_detection,
                                                   config, logger_pass)
    data_lies_in_lap = lap_low_point[:, 3]

    # Following code is used for extracting start & end times for LAPs from
    # consolidated LAPs
    diff_x = np.diff(np.hstack((0, data_lies_in_lap, 0)))
    st_idx = np.where(diff_x > 0)
    end_idx = np.where(diff_x < 0)
    start = lap_low_point[st_idx[0], 1]
    endd = lap_low_point[end_idx[0] - 1, 1]
    laps = np.hstack((start[:, np.newaxis], endd[:, np.newaxis],
                      input_data[st_idx[0], Cgbdisagg.INPUT_BILL_CYCLE_IDX][:, np.newaxis]))

    logger.info('The number of consolidated LAPs is %d |', laps.shape[0])

    return laps, lap_low_point, trnctd_laps

