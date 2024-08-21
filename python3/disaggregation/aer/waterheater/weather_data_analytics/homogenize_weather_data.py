"""
Author: Mayank Sharan
Created: 12-Jul-2020
Homogenize the input data for a given sampling rate. Create missing chunks and down sample as needed
"""

# Import python packages

import copy
import logging
import numpy as np
import pandas as pd

# Import project functions and classes

from python3.disaggregation.aer.waterheater.weather_data_analytics.math_utils import find_seq

from python3.disaggregation.aer.waterheater.weather_data_analytics.nbi_constants import TimeConstants

from python3.disaggregation.aer.waterheater.weather_data_analytics.nbi_data_constants import SeqData
from python3.disaggregation.aer.waterheater.weather_data_analytics.nbi_data_constants import WeatherData


def downsample_data(input_data, sampling_rate, target_rate):

    """
    Downsample given weather data to target sampling rate
    Parameters:
        input_data          (np.ndarray)        : 13 column input data
        sampling_rate       (int)               : The sampling rate in seconds that the data is in
        target_rate         (int)               : The sampling rate in seconds that the data has to be down sampled to
    Returns:
        downsampled_data    (np.ndarray)        : 13 column array with down sampled data
    """

    # Check a few conditions on the sampling rates to determine feasibility of down sampling

    if sampling_rate == target_rate:

        return input_data

    else:

        rate_ratio = float(sampling_rate) / target_rate

        # Get indices and downsampled timestamps ready

        group_column = (np.ceil((input_data[:, WeatherData.epoch_ts_col] / target_rate) + rate_ratio) * target_rate)
        new_epoch_ts, new_epoch_idx, bin_idx = np.unique(group_column, return_index=True, return_inverse=True)

        # Fill values in the downsampled array

        bin_wise_count = np.bincount(bin_idx)
        downsampled_data = np.full(shape=(len(new_epoch_ts), WeatherData.num_cols), fill_value=np.nan)

        # Populate the timestamps

        downsampled_data[:, WeatherData.epoch_ts_col] = new_epoch_ts
        downsampled_data[:, WeatherData.day_ts_col] = input_data[new_epoch_idx, WeatherData.day_ts_col]

        # Populate the temperature and feels like with average

        downsampled_data[:, WeatherData.temp_col] = \
            np.bincount(bin_idx, weights=input_data[:, WeatherData.temp_col]) / bin_wise_count

        downsampled_data[:, WeatherData.feels_like_col] = \
            np.bincount(bin_idx, weights=input_data[:, WeatherData.feels_like_col]) / bin_wise_count

        # Populate the precipitation and snow with sum

        downsampled_data[:, WeatherData.prec_col] = np.bincount(bin_idx, weights=input_data[:, WeatherData.prec_col])
        downsampled_data[:, WeatherData.snow_col] = np.bincount(bin_idx, weights=input_data[:, WeatherData.snow_col])

        return downsampled_data


def create_missing_data(chunk_start_end_data, sampling_rate):

    """
    Provide missing timestamp rows in the data
    Parameters:
        chunk_start_end_data(np.ndarray)        : data of the 2 points between which points will be inserted
        sampling_rate       (int)               : The sampling rate at which we will create points
    Returns:
        created_data        (np.ndarray)        : data complete with all created points
    """

    # Initialize created data matrix

    start_pt = chunk_start_end_data[0, :]
    end_pt = chunk_start_end_data[1, :]

    time_diff = end_pt[WeatherData.epoch_ts_col] - start_pt[WeatherData.epoch_ts_col]
    num_points_to_create = int(np.ceil(time_diff / sampling_rate)) - 1

    created_data = np.full(shape=(num_points_to_create, WeatherData.num_cols), fill_value=np.nan)

    # Fill in the missing epoch timestamps, Fail case is if there is supposed to be DST shift in between

    missing_epoch_ts = np.arange(start=start_pt[WeatherData.epoch_ts_col] + sampling_rate,
                                 stop=end_pt[WeatherData.epoch_ts_col], step=sampling_rate)

    created_data[:, WeatherData.epoch_ts_col] = missing_epoch_ts

    # Fill in missing day ts

    day_ts_offset = start_pt[WeatherData.day_ts_col] % TimeConstants.sec_in_1_day

    missing_day_ts = (np.divide(missing_epoch_ts - day_ts_offset, TimeConstants.sec_in_1_day).astype(int)
                      * TimeConstants.sec_in_1_day) + day_ts_offset

    created_data[:, WeatherData.day_ts_col] = missing_day_ts

    created_data = np.vstack((start_pt, created_data, end_pt))

    return created_data


def homogenize_weather_data(weather_data, target_sample_rate, logger_pass):

    """
    Homogenize weather data
    Parameters:
        weather_data            (np.ndarray)    : Array containing weather data
        target_sample_rate      (int)           : The sampling rate in seconds the data needs to obey
        logger_pass             (dict)          : Dictionary containing objects needed for logging
    Returns:
        homo_weather_data       (np.ndarray)    : Array containing homogenized weather data
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('homogenize_weather_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Split data by sampling rate

    sampling_rate_chunks = find_seq(np.diff(weather_data[:, WeatherData.epoch_ts_col]), min_seq_length=1)
    sampling_rate_chunks[:, SeqData.seq_end_col] += 1

    # If the data is already continuous and at the needed sampling rate return

    if sampling_rate_chunks.shape[0] == 1 and sampling_rate_chunks[0, SeqData.seq_val_col] == target_sample_rate:
        logger.debug('No homogenization required |')
        return weather_data

    # Compute number of points present for each sampling rate

    unique_sampling_rates, sampling_rate_idx = np.unique(sampling_rate_chunks[:, SeqData.seq_val_col],
                                                         return_inverse=True)

    pts_by_sampling_rate = np.bincount(sampling_rate_idx, weights=sampling_rate_chunks[:, SeqData.seq_len_col])
    pts_by_sampling_rate = np.c_[unique_sampling_rates, pts_by_sampling_rate]

    # Get share of each sampling rate as a percentage of points

    pts_by_sampling_rate[:, 1] *= 100 / weather_data.shape[0]
    pts_by_sampling_rate = np.round(pts_by_sampling_rate, 3)

    # Select sampling rates making up more than 2% of data points translates to 8 days in 13 months data

    if np.sum(pts_by_sampling_rate[:, 1] > 2) > 0:
        meaningful_sampling_rates = pts_by_sampling_rate[pts_by_sampling_rate[:, 1] > 2, 0]
    else:
        meaningful_sampling_rates = pts_by_sampling_rate[:, 0]

    if target_sample_rate is None:
        target_sample_rate = np.max(meaningful_sampling_rates)

    sampling_rate_ratios = float(target_sample_rate) / np.r_[meaningful_sampling_rates, target_sample_rate]
    sampling_rate_ratios -= sampling_rate_ratios.astype(int)

    if np.sum(sampling_rate_ratios > 0):
        logger.warning('Weather data cannot be homogenised due to incompatible sampling rates |')

    homo_weather_data = np.empty(shape=(0, WeatherData.num_cols))

    # Process each of the sampling rate chunks down sample if needed

    for sampling_chunk in sampling_rate_chunks:

        chunk_sampling_rate = sampling_chunk[SeqData.seq_val_col]

        start_idx = int(sampling_chunk[SeqData.seq_start_col])
        end_idx = min(int(sampling_chunk[SeqData.seq_end_col]) + 1, weather_data.shape[0])

        num_points_chunk = sampling_chunk[SeqData.seq_len_col]

        chunk_data = weather_data[start_idx: end_idx, :]

        # Down sample if needed

        if chunk_sampling_rate == target_sample_rate:

            homo_weather_data = np.r_[homo_weather_data, chunk_data]

        elif chunk_sampling_rate < target_sample_rate and \
                num_points_chunk >= 5 * target_sample_rate / chunk_sampling_rate:

            downsampled_data = downsample_data(chunk_data, chunk_sampling_rate, target_sample_rate)
            homo_weather_data = np.r_[homo_weather_data, downsampled_data]

    # Remove any duplicates that might have been created while downsampling data

    homogenised_data_df = pd.DataFrame(data=homo_weather_data)
    homogenised_data_df = homogenised_data_df.drop_duplicates(subset=[WeatherData.epoch_ts_col], keep='first')

    homo_weather_data = homogenised_data_df.values

    # Check if there are any gaps

    data_gap_chunks = find_seq(np.diff(homo_weather_data[:, WeatherData.epoch_ts_col]), min_seq_length=1)
    data_gap_chunks[:, SeqData.seq_end_col] += 1

    # Fill in gaps with nan valued data points

    continuous_data = np.empty(shape=(0, WeatherData.num_cols))

    for gap_chunk in data_gap_chunks:

        chunk_sampling_rate = gap_chunk[SeqData.seq_val_col]

        start_idx = int(gap_chunk[SeqData.seq_start_col])
        end_idx = int(gap_chunk[SeqData.seq_end_col]) + 1

        chunk_data = homo_weather_data[start_idx: end_idx, :]

        if chunk_sampling_rate > target_sample_rate:

            chunk_start_end_data = chunk_data[[0, -1], :]
            missing_data = create_missing_data(chunk_start_end_data, target_sample_rate)
            continuous_data = np.r_[continuous_data, missing_data]

        else:

            continuous_data = np.r_[continuous_data, chunk_data]

    # Remove any duplicates that might have been created while creating missing points

    continuous_data_df = pd.DataFrame(data=continuous_data)
    continuous_data_df = continuous_data_df.drop_duplicates(subset=[WeatherData.epoch_ts_col], keep='first')

    continuous_data = continuous_data_df.values
    homo_weather_data = copy.deepcopy(continuous_data)

    return homo_weather_data
