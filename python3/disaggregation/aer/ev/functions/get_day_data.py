"""
Author - Mayank Sharan / Paras Tehria
Date - 11-Jan-2019
This converts 21 column data to day wise data matrix
"""

# Import python packages

import numpy as np
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_day_data(raw, in_data, detection, ev, residual, timezone, sampling_rate, padding_zero=False):
    """
    Parameters:
        raw                 (np.ndarray)        : Raw data
        in_data             (np.ndarray)        : Input data with timed appliances removed
        detection           (np.ndarray)        : Detection boxes data
        ev                  (np.ndarray)        : Estimation boxes data
        residual            (np.ndarray)        : Residual data
        timezone            (int)               : Timezone in hours w.r.t. to GMT
        sampling_rate       (int)               : Sampling rate of data
        padding_zero        (bool)              : Bill cycle boundary values

    Returns:
        final_output         (list)              : List containing 2d matrix with each row representing a day

    """

    # Take the deepcopy of the relevant variables for relevant use

    raw_data = deepcopy(raw)
    input_data = deepcopy(in_data)
    detection_data = deepcopy(detection)
    ev_data = deepcopy(ev)
    residual_data = deepcopy(residual)

    # Adjusting timestamps of data for timezone alignment

    raw_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
    input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
    detection_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
    ev_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
    residual_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)

    # Here we do not take DST into consideration while reshaping the data

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)

    # Figure out how many rows is the data going to have

    # Bug created here to make sure results match between old and new

    first_pt_date = datetime.utcfromtimestamp(input_data[0, Cgbdisagg.INPUT_EPOCH_IDX])
    first_pt_idx = int(((first_pt_date.hour + (first_pt_date.minute / 60)) * Cgbdisagg.SEC_IN_HOUR) / sampling_rate)

    num_data_pts = input_data.shape[0]

    first_day_pts = num_pd_in_day - first_pt_idx
    last_day_pts = (num_data_pts - first_day_pts) % num_pd_in_day

    # Bug handling in case we get exact days data

    if last_day_pts == 0:
        last_day_pts = num_pd_in_day

    num_rows = int(np.ceil((num_data_pts - first_day_pts) / num_pd_in_day)) + 1

    # Initialize 2d matrices

    month_ts = np.full(shape=(num_rows, num_pd_in_day), fill_value=np.nan)
    day_ts = np.full(shape=(num_rows, num_pd_in_day), fill_value=np.nan)
    epoch_ts = np.full(shape=(num_rows, num_pd_in_day), fill_value=np.nan)

    # Check if padding need to be done with zero or max values

    if padding_zero:
        raw_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        input_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        detection_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        ev_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        res_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
    else:
        raw_2d = np.full(shape=(num_rows, num_pd_in_day),
                         fill_value=np.max(raw_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        input_2d = np.full(shape=(num_rows, num_pd_in_day),
                           fill_value=np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        detection_2d = np.full(shape=(num_rows, num_pd_in_day),
                               fill_value=np.max(detection_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        ev_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=np.max(ev_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        res_2d = np.full(shape=(num_rows, num_pd_in_day),
                         fill_value=np.max(residual_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))

    # Fill the first row with the available data points of that day timestamps

    month_ts[0, first_pt_idx:] = raw_data[: first_day_pts, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    day_ts[0, first_pt_idx:] = raw_data[: first_day_pts, Cgbdisagg.INPUT_DAY_IDX]
    epoch_ts[0, first_pt_idx:] = raw_data[: first_day_pts, Cgbdisagg.INPUT_EPOCH_IDX]

    # Adjust the missing data points based on sampling rate

    if first_day_pts != num_pd_in_day:
        sampling_rate_arr = sampling_rate * np.arange(1, first_pt_idx + 1)
        fill_nan_arr = np.sort(raw_data[0, Cgbdisagg.INPUT_EPOCH_IDX] - sampling_rate_arr)
        epoch_ts[0, :first_pt_idx] = fill_nan_arr

    # Fill the first row for consumption data matrices

    raw_2d[0, first_pt_idx:] = raw_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    input_2d[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    detection_2d[0, first_pt_idx:] = detection_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    ev_2d[0, first_pt_idx:] = ev_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    res_2d[0, first_pt_idx:] = residual_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Fill the rows in between the bill cycles for demarcation

    month_ts[1: -1, :] = \
        np.reshape(a=input_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    day_ts[1: -1, :] = \
        np.reshape(a=input_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_DAY_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    epoch_ts[1: -1, :] = \
        np.reshape(a=input_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_EPOCH_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    raw_2d[1: -1, :] = \
        np.reshape(a=raw_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    input_2d[1: -1, :] = \
        np.reshape(a=input_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    detection_2d[1: -1, :] = \
        np.reshape(a=detection_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    ev_2d[1: -1, :] = \
        np.reshape(a=ev_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    res_2d[1: -1, :] = \
        np.reshape(a=residual_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    # Fill the last day data timestamps based on availability

    month_ts[-1, :last_day_pts] = raw_data[-last_day_pts:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    day_ts[-1, :last_day_pts] = raw_data[-last_day_pts:, Cgbdisagg.INPUT_DAY_IDX]
    epoch_ts[-1, :last_day_pts] = raw_data[-last_day_pts:, Cgbdisagg.INPUT_EPOCH_IDX]

    # Adjust for the incomplete last day data using NaNs

    if last_day_pts != num_pd_in_day:
        sampling_rate_arr = sampling_rate * np.arange(1, num_pd_in_day - last_day_pts + 1)
        fill_nan_arr = raw_data[-1, Cgbdisagg.INPUT_EPOCH_IDX] + sampling_rate_arr
        epoch_ts[-1, last_day_pts:] = fill_nan_arr

    # Fill the last row for consumption data matrices

    raw_2d[-1, :last_day_pts] = raw_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    input_2d[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    detection_2d[-1, :last_day_pts] = detection_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    ev_2d[-1, :last_day_pts] = ev_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    res_2d[-1, :last_day_pts] = residual_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Avoiding negative numbers

    raw_2d = np.fmax(raw_2d, 0)
    input_2d = np.fmax(input_2d, 0)
    detection_2d = np.fmax(detection_2d, 0)
    ev_2d = np.fmax(ev_2d, 0)
    res_2d = np.fmax(res_2d, 0)

    # Check for NaNs in the data once again due to first and last row

    if padding_zero:
        input_2d[np.isnan(input_2d)] = 0
        raw_2d[np.isnan(raw_2d)] = 0
        detection_2d[np.isnan(detection_2d)] = 0
        ev_2d[np.isnan(ev_2d)] = 0
        res_2d[np.isnan(ev_2d)] = 0
    else:
        raw_2d[np.isnan(raw_2d)] = np.max(raw_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        input_2d[np.isnan(input_2d)] = np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        detection_2d[np.isnan(detection_2d)] = np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        ev_2d[np.isnan(ev_2d)] = np.max(ev_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        res_2d[np.isnan(ev_2d)] = np.max(residual_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Stack all the output matrices to a list

    final_output = [month_ts, day_ts, epoch_ts, raw_2d, input_2d, detection_2d, ev_2d, res_2d]

    return final_output
