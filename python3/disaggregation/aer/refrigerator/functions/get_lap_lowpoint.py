"""
Date Created - 13 Nove 2018
Author name - Pratap
LAP low point calculated
"""

import logging
import numpy as np

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.refrigerator.functions.edge_correction_laps import edge_correction_laps


def get_lap_low_point(input_data, laps, ref_detection, config, logger_base):
    """
    Consolidated laps with edge correction
    inputs - Input Data, LAP information
    output - consolidated LAP with edge correction & truncated points
    """
    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("get_lap_lowpoint")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_base.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    data_lies_in_lap = np.full((input_data.shape[0], 1), False, dtype=bool)

    # Finding the timestamps that lie within LAPs
    for i in range(laps.shape[0]):
        chk1 = (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= laps[i, 0])[:, np.newaxis]
        chk2 = (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= laps[i, 1])[:, np.newaxis]
        data_lies_in_lap = np.logical_or(data_lies_in_lap, np.logical_and(chk1, chk2))

    logger.debug('Number of data points in LAPs is: {} |'.format(np.sum(data_lies_in_lap)))

    # Find the low points within LAP by taking diff forward and backward
    diff_data = np.diff(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    diff_data = diff_data[:, np.newaxis]
    forward_derivative = np.hstack((diff_data[:, 0], 0))[:, np.newaxis]
    back_derivative = np.hstack((0, diff_data[:, 0]))[:, np.newaxis]
    both_fwd_bckwd_derivative = np.hstack((back_derivative, forward_derivative))

    # If the forward is negative and backward is positive then it's a low point
    bool_low_point = np.logical_and(both_fwd_bckwd_derivative[:, 0] < 0, both_fwd_bckwd_derivative[:, 1] >= 0)
    bool_low_point = bool_low_point + 0

    # Applying edge correction post consolidation of laps
    data_lies_in_lap_2 = np.hstack((data_lies_in_lap, input_data[:, Cgbdisagg.INPUT_EPOCH_IDX][:, np.newaxis]))
    data_lies_in_lap_3, truncated_laps = edge_correction_laps(ref_detection, config, data_lies_in_lap_2, logger_pass)

    # Updating the low points of the LAPs
    lap_low_point = np.logical_and(bool_low_point, data_lies_in_lap_3[:, 0])
    lap_low_point = lap_low_point + 0

    output = np.hstack((input_data[:, [Cgbdisagg.INPUT_BILL_CYCLE_IDX, Cgbdisagg.INPUT_EPOCH_IDX, ]],
                        lap_low_point[:, np.newaxis], data_lies_in_lap_3[:, 0][:, np.newaxis]))

    return output, truncated_laps

