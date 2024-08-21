"""
Author - Mayank Sharan
Date - 11/1/19
This converts 21 column data to day wise data matrix
"""

# Import python packages

import numpy as np
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_day_data(raw, thin, fat, residual, timezone, sampling_rate, padding_zero=False):
    """
    Parameters:
        raw                 (np.ndarray)        : Input raw data
        thin                (np.ndarray)        : Thin pulse data
        fat                 (np.ndarray)        : Fat pulse data
        residual            (np.ndarray)        : Residual data
        timezone            (int)               : Timezone in hours w.r.t. to GMT
        sampling_rate       (int)               : Sampling rate of data
        padding_zero        (bool)              : Bill cycle boundary values

    Returns:
        day_data            (np.ndarray)        : 2d matrix with each row representing a day
        month_ts            (np.ndarray)        : 2d matrix with corresponding bill cycle timestamps
        daily_ts            (np.ndarray)        : 2d matrix with corresponding day timestamps
        epoch_ts            (np.ndarray)        : 2d matrix with corresponding epoch timestamps
    """

    # Take the deepcopy of the relevant variables for relevant use

    input_data = deepcopy(raw)
    thin_data = deepcopy(thin)
    fat_data = deepcopy(fat)
    residual_data = deepcopy(residual)

    wh_data = deepcopy(raw)
    wh_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = thin_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] + \
                                                  fat_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Adjusting timestamps of data for timezone alignment

    input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
    thin_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
    fat_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
    wh_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
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

    if padding_zero == True:
        raw_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        thin_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        fat_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        wh_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        res_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
    else:
        raw_2d = np.full(shape=(num_rows, num_pd_in_day),
                         fill_value=np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        thin_2d = np.full(shape=(num_rows, num_pd_in_day),
                          fill_value=np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        fat_2d = np.full(shape=(num_rows, num_pd_in_day),
                         fill_value=np.max(fat_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        wh_2d = np.full(shape=(num_rows, num_pd_in_day),
                        fill_value=np.max(wh_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        res_2d = np.full(shape=(num_rows, num_pd_in_day),
                         fill_value=np.max(residual_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))

    # Fill the first row with the available data points of that day timestamps

    month_ts[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    day_ts[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_DAY_IDX]
    epoch_ts[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_EPOCH_IDX]

    # Adjust the missing data points based on sampling rate

    if first_day_pts != num_pd_in_day:
        sampling_rate_arr = sampling_rate * np.arange(1, first_pt_idx + 1)
        fill_nan_arr = np.sort(input_data[0, Cgbdisagg.INPUT_EPOCH_IDX] - sampling_rate_arr)
        epoch_ts[0, :first_pt_idx] = fill_nan_arr

    # Fill the first row for consumption data matrices

    raw_2d[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    thin_2d[0, first_pt_idx:] = thin_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    fat_2d[0, first_pt_idx:] = fat_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    wh_2d[0, first_pt_idx:] = wh_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]
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
        np.reshape(a=input_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    thin_2d[1: -1, :] = \
        np.reshape(a=thin_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    fat_2d[1: -1, :] = \
        np.reshape(a=fat_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    wh_2d[1: -1, :] = \
        np.reshape(a=wh_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    res_2d[1: -1, :] = \
        np.reshape(a=residual_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    # Fill the last day data timestamps based on availability

    month_ts[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    day_ts[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_DAY_IDX]
    epoch_ts[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_EPOCH_IDX]

    # Adjust for the incomplete last day data using NaNs

    if last_day_pts != num_pd_in_day:
        sampling_rate_arr = sampling_rate * np.arange(1, num_pd_in_day - last_day_pts + 1)
        fill_nan_arr = input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX] + sampling_rate_arr
        epoch_ts[-1, last_day_pts:] = fill_nan_arr

    # Fill the last row for consumption data matrices

    raw_2d[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    thin_2d[-1, :last_day_pts] = thin_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    fat_2d[-1, :last_day_pts] = fat_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    wh_2d[-1, :last_day_pts] = wh_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    res_2d[-1, :last_day_pts] = residual_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Avoiding negative numbers

    raw_2d = np.fmax(raw_2d, 0)
    thin_2d = np.fmax(thin_2d, 0)
    fat_2d = np.fmax(fat_2d, 0)
    wh_2d = np.fmax(wh_2d, 0)
    res_2d = np.fmax(res_2d, 0)

    # Check for NaNs in the data once again due to first and last row

    if padding_zero == True:
        raw_2d[np.isnan(raw_2d)] = 0
        thin_2d[np.isnan(thin_2d)] = 0
        fat_2d[np.isnan(fat_2d)] = 0
        wh_2d[np.isnan(wh_2d)] = 0
        res_2d[np.isnan(fat_2d)] = 0
    else:
        raw_2d[np.isnan(raw_2d)] = np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        thin_2d[np.isnan(thin_2d)] = np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        fat_2d[np.isnan(fat_2d)] = np.max(fat_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        wh_2d[np.isnan(wh_2d)] = np.max(wh_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        res_2d[np.isnan(fat_2d)] = np.max(residual_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Stack all the output matrices to a list

    final_output = [month_ts, day_ts, epoch_ts, raw_2d, thin_2d, fat_2d, wh_2d, res_2d]

    return final_output


def get_day_data_twh(raw, temp, wh_output, residual, timezone, sampling_rate, padding_zero=False):
    """
    Parameters:
        raw                 (np.ndarray)        : Input raw data
        temp                (np.ndarray)        : Input temperature data
        wh_output           (np.ndarray)        : Timed wh output
        residual            (np.ndarray)        : Residual after timed wh detection
        timezone            (int)               : Timezone in hours w.r.t. to GMT
        sampling_rate       (int)               : Sampling rate of data
        padding_zero        (bool)              : Bill cycle boundary values
    Returns:
        day_data            (np.ndarray)        : 2d matrix with each row representing a day
        month_ts            (np.ndarray)        : 2d matrix with corresponding bill cycle timestamps
        daily_ts            (np.ndarray)        : 2d matrix with corresponding day timestamps
        epoch_ts            (np.ndarray)        : 2d matrix with corresponding epoch timestamps
    """

    # Take the deepcopy of the relevant variables for relevant use

    input_data = deepcopy(raw)
    temp_data = deepcopy(temp)
    residual_data = deepcopy(residual)

    wh_data = deepcopy(raw)
    wh_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = wh_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Adjusting timestamps of data for timezone alignment

    wh_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
    temp_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
    input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] += (Cgbdisagg.SEC_IN_HOUR * timezone)
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
        wh_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        raw_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        temp_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)
        residual_2d = np.full(shape=(num_rows, num_pd_in_day), fill_value=0)

    else:
        raw_2d = np.full(shape=(num_rows, num_pd_in_day),
                         fill_value=np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        wh_2d = np.full(shape=(num_rows, num_pd_in_day),
                        fill_value=np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        temp_2d = np.full(shape=(num_rows, num_pd_in_day),
                          fill_value=np.max(input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]))
        residual_2d = np.full(shape=(num_rows, num_pd_in_day),
                              fill_value=np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))

    # Fill the first row with the available data points of that day timestamps

    month_ts[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    day_ts[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_DAY_IDX]
    epoch_ts[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_EPOCH_IDX]

    # Adjust the missing data points based on sampling rate

    if first_day_pts != num_pd_in_day:
        sampling_rate_arr = sampling_rate * np.arange(1, first_pt_idx + 1)
        fill_nan_arr = np.sort(input_data[0, Cgbdisagg.INPUT_EPOCH_IDX] - sampling_rate_arr)
        epoch_ts[0, :first_pt_idx] = fill_nan_arr

    # Fill the first row for consumption data matrices

    wh_2d[0, first_pt_idx:] = wh_2d[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    raw_2d[0, first_pt_idx:] = input_data[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    temp_2d[0, first_pt_idx:] = residual_2d[: first_day_pts, Cgbdisagg.INPUT_TEMPERATURE_IDX]
    residual_2d[0, first_pt_idx:] = residual_2d[: first_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX]

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
        np.reshape(a=input_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    wh_2d[1: -1, :] = \
        np.reshape(a=wh_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    residual_2d[1: -1, :] = \
        np.reshape(a=residual_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    temp_2d[1: -1, :] = \
        np.reshape(a=temp_data[first_day_pts: num_data_pts - last_day_pts, Cgbdisagg.INPUT_TEMPERATURE_IDX],
                   newshape=(num_rows - 2, num_pd_in_day))

    # Fill the last day data timestamps based on availability

    month_ts[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    day_ts[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_DAY_IDX]
    epoch_ts[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_EPOCH_IDX]

    # Adjust for the incomplete last day data using NaNs

    if last_day_pts != num_pd_in_day:
        sampling_rate_arr = sampling_rate * np.arange(1, num_pd_in_day - last_day_pts + 1)
        fill_nan_arr = input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX] + sampling_rate_arr
        epoch_ts[-1, last_day_pts:] = fill_nan_arr

    # Fill the last row for consumption data matrices

    wh_2d[-1, :last_day_pts] = wh_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    raw_2d[-1, :last_day_pts] = input_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    temp_2d[-1, :last_day_pts] = temp_data[-last_day_pts:, Cgbdisagg.INPUT_TEMPERATURE_IDX]
    residual_2d[-1, :last_day_pts] = residual_data[-last_day_pts:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Avoiding negative numbers

    raw_2d = np.fmax(raw_2d, 0)
    wh_2d = np.fmax(wh_2d, 0)
    residual_2d = np.fmax(residual_2d, 0)
    temp_2d = np.fmax(temp_2d, 0)

    # Check for NaNs in the data once again due to first and last row

    if padding_zero == True:
        raw_2d[np.isnan(raw_2d)] = 0
        wh_2d[np.isnan(wh_2d)] = 0
        residual_2d[np.isnan(residual_2d)] = 0
        temp_2d[np.isnan(temp_2d)] = 0

    else:
        wh_2d[np.isnan(wh_2d)] = np.max(wh_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        raw_2d[np.isnan(raw_2d)] = np.max(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        temp_2d[np.isnan(temp_2d)] = np.max(temp_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX])
        residual_2d[np.isnan(residual_2d)] = np.max(residual_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Stack all the output matrices to a list

    final_output = [month_ts, day_ts, epoch_ts, raw_2d, temp_2d, wh_2d, residual_2d]

    return final_output
