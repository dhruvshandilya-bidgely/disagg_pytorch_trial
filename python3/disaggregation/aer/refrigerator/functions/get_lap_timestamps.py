"""
Date Created - 13 Nove 2018
Author name - Pratap
LAP midtimestamps calculated
"""
import logging
import numpy as np
from scipy import signal

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.matlab_utils import percentile_1d


def get_lap_timestamps(input_data, config, logger_base):
    """
    getLAPtimestamps function does calculate mid timestamps based on
    convolving filter on energy data

    Parameters:
        input_data (np.ndarray):
        config (dict):
        logger_base (logger):

    Returns:
        lap_mid_timestamps (np.ndarray):
    """
    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("get_lap_timestamps")
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    lap_mid_timestamps = np.nan
    data_this_month = np.copy(input_data)

    # noinspection PyBroadException
    try:
        lap_mid_timestamps = get_monthly_lapmid_timestamps(data_this_month, config, logger)
        logger.info('Number of LAPs with mid-timestamps are: {} |'.format(lap_mid_timestamps.shape[0]))
    except Exception:
        logger.warning('No LAP Mid Timestamps |')

    return lap_mid_timestamps


def get_monthly_lapmid_timestamps(data_this_month, config, logger):
    """
    Monthly LAPtimestamps is actualy the function which calculates LAPs

    Parameters:
        data_this_month (np.ndarray):
        config (dict):
        logger (logger):

    Returns:
        lap_mid_timestamps_v2 (np.ndarray):

    """
    # Finding the window size of the 6-hr window
    fill_row = int(config['LAPDetection']['lapHalfGap'] * 2 * 3600 / config['samplingRate'])
    first_data = data_this_month[:, Cgbdisagg.INPUT_CONSUMPTION_IDX][:, np.newaxis]

    y_sum = signal.convolve2d(first_data, np.ones([fill_row, 1]), mode='same')
    y_sum = y_sum[1:, :]

    # Fixing the end values of the moving sum
    last_num = np.sum(data_this_month[(np.shape(data_this_month)[0] - int(fill_row / 2)):, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    y_sum = np.vstack((y_sum, last_num))

    second_data = data_this_month[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX][:, np.newaxis]
    third_data = data_this_month[:, Cgbdisagg.INPUT_EPOCH_IDX][:, np.newaxis]
    y_sum = np.hstack((y_sum, second_data, third_data))
    window_start = int(np.ceil((3600 / config['samplingRate']) * 3) + 1)
    window_end = int(np.ceil((3600 / config['samplingRate']) * 3))
    y_sum_inter = y_sum[(window_start - 1):(y_sum.shape[0] - window_end - 1), :]

    # Finding the 6th percentile threshold for LAPs
    y_low = percentile_1d(y_sum[y_sum[:, 0] > 0, 0], config['LAPDetection']['lowPercentileLimit'])
    logger.info('The 6th percentile energy threshold is: {} |'.format(y_low))

    y_sum2 = y_sum_inter[np.lexsort((y_sum_inter[:, 0], y_sum_inter[:, 1]))]
    unique_monthly_timestamp = np.unique(y_sum2[:, 1]).astype(int)
    append_monthly = np.empty((1, 3))

    # Sorting the LAPs by month and selecting top-30 from each
    for i in range(unique_monthly_timestamp.shape[0]):
        unique_month = y_sum2[y_sum2[:, 1] == unique_monthly_timestamp[i], :]
        unique_month2 = unique_month[unique_month[:, 0] < y_low, :]
        if unique_month2.shape[0] > 30:
            append_monthly = np.vstack((append_monthly, unique_month2))
        else:
            append_monthly = np.vstack((append_monthly, unique_month[:30, :]))

    lap_mid_timestamps = append_monthly[1:, 1:]
    lap_mid_timestamps_v2 = np.hstack((lap_mid_timestamps[:, 1][:, np.newaxis],
                                       lap_mid_timestamps[:, 0][:, np.newaxis]))

    return lap_mid_timestamps_v2
