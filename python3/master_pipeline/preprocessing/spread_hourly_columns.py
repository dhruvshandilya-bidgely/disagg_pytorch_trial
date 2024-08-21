"""
Author - Mayank Sharan
Date - 24/09/18
Pre-processes input data using different utility functions for disagg modules to use
"""

# Import python packages

import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg                                          # noqa
from python3.master_pipeline.preprocessing.split_data_by_sample_rate import split_data_by_sample_rate    # noqa


def spread_hourly_columns(input_data):
    """
    Parameters:
        input_data          (np.ndarray)        : 21 column data matrix

    Returns:
        input_data          (np.ndarray)        : 21 column data matrix
    """

    # Convert to data frame to make things easier

    input_df = pd.DataFrame(data=input_data)

    # Set columns to group by

    group_columns = [Cgbdisagg.INPUT_DAY_IDX, Cgbdisagg.INPUT_HOD_IDX]

    # Set columns to fill

    fill_columns = [Cgbdisagg.INPUT_SKYCOV_IDX, Cgbdisagg.INPUT_WIND_SPD_IDX, Cgbdisagg.INPUT_SUNRISE_IDX,
                    Cgbdisagg.INPUT_SUNSET_IDX, Cgbdisagg.INPUT_SL_PRESS_IDX, Cgbdisagg.INPUT_SPC_HUM_IDX,
                    Cgbdisagg.INPUT_REL_HUM_IDX, Cgbdisagg.INPUT_WIND_DIR_IDX, Cgbdisagg.INPUT_VISIBILITY_IDX,
                    Cgbdisagg.INPUT_COOLING_POTENTIAL_IDX, Cgbdisagg.INPUT_HEATING_POTENTIAL_IDX,
                    Cgbdisagg.INPUT_WH_POTENTIAL_IDX, Cgbdisagg.INPUT_COLD_EVENT_IDX, Cgbdisagg.INPUT_HOT_EVENT_IDX,
                    Cgbdisagg.INPUT_S_LABEL_IDX]

    # Set all values available hour wise to their mean

    input_df[fill_columns] = input_df.groupby(by=group_columns)[fill_columns].transform('mean')

    # Getting sampling rate >>

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

    min_sampling_rate = np.min(meaningful_sampling_rates)

    # Interpolate temperature values, 3 hours maximum in case of hourly data <<

    temp_interpolation_limit = 3 * int(3600 / min_sampling_rate)

    interpolation_columns = [Cgbdisagg.INPUT_TEMPERATURE_IDX, Cgbdisagg.INPUT_DEW_IDX, Cgbdisagg.INPUT_FEELS_LIKE_IDX,
                             Cgbdisagg.INPUT_WET_BULB_IDX, Cgbdisagg.INPUT_COOLING_POTENTIAL_IDX,
                             Cgbdisagg.INPUT_HEATING_POTENTIAL_IDX, Cgbdisagg.INPUT_WH_POTENTIAL_IDX]

    input_df[interpolation_columns] = input_df[interpolation_columns].interpolate(limit=temp_interpolation_limit,
                                                                                  limit_area='inside',
                                                                                  limit_direction='both')

    input_data = input_df.values

    return input_data
