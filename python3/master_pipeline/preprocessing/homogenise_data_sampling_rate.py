"""
Author - Mayank Sharan
Date - 25/09/18
Get all data to a single sampling rate, also can be used to just downsample data
"""

# Import python packages

import copy
import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg                                    # noqa
from python3.master_pipeline.preprocessing.downsample_data import downsample_data                       # noqa
from python3.master_pipeline.preprocessing.split_data_by_sample_rate import split_data_by_sample_rate   # noqa


def create_missing_data(chunk_start_end_data, sampling_rate):

    """
    Parameters:
        chunk_start_end_data(np.ndarray)        : 21 columns of the 2 points between which points will be inserted
        sampling_rate       (int)               : The sampling rate at which we will create points

    Returns:
        created_data        (np.ndarray)        : 21 column data complete with all created points
    """

    # Initialize created data matrix

    start_pt = chunk_start_end_data[0, :]
    end_pt = chunk_start_end_data[1, :]
    time_diff = end_pt[Cgbdisagg.INPUT_EPOCH_IDX] - start_pt[Cgbdisagg.INPUT_EPOCH_IDX]
    num_points_to_create = int(np.ceil(time_diff / sampling_rate)) - 1

    created_data = np.full(shape=(num_points_to_create, Cgbdisagg.INPUT_DIMENSION), fill_value=np.nan)

    # Initialize stuff you can directly initialize

    # Fill in the missing epoch timestamps, Fail case is if there is supposed to be DST shift in between

    missing_epoch_ts = np.arange(start=start_pt[Cgbdisagg.INPUT_EPOCH_IDX] + sampling_rate,
                                 stop=end_pt[Cgbdisagg.INPUT_EPOCH_IDX], step=sampling_rate)

    created_data[:, Cgbdisagg.INPUT_EPOCH_IDX] = missing_epoch_ts

    # Fill in missing bill cycle ts

    missing_bill_ts = copy.deepcopy(missing_epoch_ts)

    missing_bill_ts[missing_bill_ts >= end_pt[Cgbdisagg.INPUT_BILL_CYCLE_IDX]] = end_pt[Cgbdisagg.INPUT_BILL_CYCLE_IDX]

    start_month_ts_idx = np.logical_and(missing_bill_ts >= start_pt[Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                        missing_bill_ts < end_pt[Cgbdisagg.INPUT_BILL_CYCLE_IDX])
    missing_bill_ts[start_month_ts_idx] = start_pt[Cgbdisagg.INPUT_BILL_CYCLE_IDX]

    created_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] = missing_bill_ts

    # Fill in missing week ts

    week_ts_offset = start_pt[Cgbdisagg.INPUT_WEEK_IDX] % Cgbdisagg.SEC_IN_WEEK

    missing_week_ts = (np.divide(missing_epoch_ts - week_ts_offset, Cgbdisagg.SEC_IN_WEEK).astype(int)
                       * Cgbdisagg.SEC_IN_WEEK) + week_ts_offset

    created_data[:, Cgbdisagg.INPUT_WEEK_IDX] = missing_week_ts

    # Fill in missing day ts

    day_ts_offset = start_pt[Cgbdisagg.INPUT_DAY_IDX] % Cgbdisagg.SEC_IN_DAY

    missing_day_ts = (np.divide(missing_epoch_ts - day_ts_offset, Cgbdisagg.SEC_IN_DAY).astype(int)
                      * Cgbdisagg.SEC_IN_DAY) + day_ts_offset

    created_data[:, Cgbdisagg.INPUT_DAY_IDX] = missing_day_ts

    # Fill in missing dow idx

    dow_diff = np.divide(missing_day_ts - start_pt[Cgbdisagg.INPUT_DAY_IDX], Cgbdisagg.SEC_IN_DAY).astype(int)

    missing_dow = (dow_diff + start_pt[Cgbdisagg.INPUT_DOW_IDX]) % Cgbdisagg.DAYS_IN_WEEK
    missing_dow[missing_dow == 0] = Cgbdisagg.DAYS_IN_WEEK

    created_data[:, Cgbdisagg.INPUT_DOW_IDX] = missing_dow

    # Fill in missing hod idx

    missing_hod = np.divide(missing_epoch_ts - missing_day_ts, Cgbdisagg.SEC_IN_HOUR).astype(int)
    created_data[:, Cgbdisagg.INPUT_HOD_IDX] = missing_hod

    created_data = np.vstack((start_pt, created_data, end_pt))

    return created_data


def homogenise_data_sampling_rate(input_data, target_sample_rate=None):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column input data
        target_sample_rate  (int)               : The final sampling rate we want to achieve

    Return:
        down_sampled_data   (np.ndarray)        : 21 column data with down sampled values
    """

    # Split data by sampling rate

    sampling_rate_chunks = split_data_by_sample_rate(input_data)

    # Compute number of points present for each sampling rate

    unique_sampling_rates, sampling_rate_idx = np.unique(sampling_rate_chunks[:, 0], return_inverse=True)
    pts_by_sampling_rate = np.bincount(sampling_rate_idx, weights=sampling_rate_chunks[:, 3])
    pts_by_sampling_rate = np.c_[unique_sampling_rates, pts_by_sampling_rate]

    # Get share of each sampling rate as a percentage of points

    pts_by_sampling_rate[:, 1] *= 100 / input_data.shape[0]
    pts_by_sampling_rate = np.round(pts_by_sampling_rate, 3)

    # Select sampling rates making up more than 2% of data points translates to 8 days in 13 months data

    if np.sum(pts_by_sampling_rate[:, 1] > 2) > 0:
        meaningful_sampling_rates = pts_by_sampling_rate[pts_by_sampling_rate[:, 1] > 2, 0]
    else:
        meaningful_sampling_rates = pts_by_sampling_rate[:, 0]

    if target_sample_rate is None:
        target_sample_rate = np.max(meaningful_sampling_rates)

    min_sampling_rate = np.min(meaningful_sampling_rates)

    sampling_rate_ratios = np.r_[meaningful_sampling_rates, target_sample_rate] / float(min_sampling_rate)
    sampling_rate_ratios -= sampling_rate_ratios.astype(int)

    if np.sum(sampling_rate_ratios > 0):
        print('Data cannot be homogenised due to incompatible sampling rates')

    homogenised_data = np.empty(shape=(0, Cgbdisagg.INPUT_DIMENSION))

    # Process each of the sampling rate chunks downsample if needed

    for sampling_chunk in sampling_rate_chunks:

        chunk_sampling_rate = sampling_chunk[0]
        start_idx = int(sampling_chunk[1])
        end_idx = min(int(sampling_chunk[2]) + 1, input_data.shape[0])
        num_points_chunk = sampling_chunk[3]
        chunk_data = input_data[start_idx: end_idx, :]

        # Downsample

        if chunk_sampling_rate == target_sample_rate:
            homogenised_data = np.r_[homogenised_data, chunk_data]
        elif chunk_sampling_rate < target_sample_rate and \
                num_points_chunk >= 5 * target_sample_rate / chunk_sampling_rate:
            downsampled_data = downsample_data(chunk_data, target_sample_rate)
            homogenised_data = np.r_[homogenised_data, downsampled_data]

    homogenised_data_df = pd.DataFrame(data=homogenised_data)
    homogenised_data_df = homogenised_data_df.drop_duplicates(subset=[Cgbdisagg.INPUT_EPOCH_IDX], keep='first')

    homogenised_data = homogenised_data_df.values

    data_gap_chunks = split_data_by_sample_rate(homogenised_data)

    # Fill in gaps with nan valued data points

    continuous_data = np.empty(shape=(0, Cgbdisagg.INPUT_DIMENSION))

    for gap_chunk in data_gap_chunks:

        chunk_sampling_rate = gap_chunk[0]
        start_idx = int(gap_chunk[1])
        end_idx = int(gap_chunk[2]) + 1
        chunk_data = homogenised_data[start_idx: end_idx, :]

        if chunk_sampling_rate > target_sample_rate:
            chunk_start_end_data = chunk_data[[0, -1], :]
            missing_data = create_missing_data(chunk_start_end_data, target_sample_rate)
            continuous_data = np.r_[continuous_data, missing_data]
        else:
            continuous_data = np.r_[continuous_data, chunk_data]

    continuous_data_df = pd.DataFrame(data=continuous_data)
    continuous_data_df = continuous_data_df.drop_duplicates(subset=[Cgbdisagg.INPUT_EPOCH_IDX], keep='first')

    continuous_data = continuous_data_df.values

    # Quick fix for hod, This is a known bug that can happen when we create data for DST shift times

    continuous_data[continuous_data[:, Cgbdisagg.INPUT_HOD_IDX] > 23, Cgbdisagg.INPUT_HOD_IDX] = 23
    fix_idx = np.where(np.diff(continuous_data[:, Cgbdisagg.INPUT_HOD_IDX]) == -1)[0]

    if len(fix_idx) > 0:
        continuous_data[fix_idx, Cgbdisagg.INPUT_HOD_IDX] = continuous_data[fix_idx + 1, Cgbdisagg.INPUT_HOD_IDX]

    return target_sample_rate, continuous_data
