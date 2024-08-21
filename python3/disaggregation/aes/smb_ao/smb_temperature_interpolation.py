"""
Author: Neelabh Goyal
Date:   14-June-2023
Called to interpolate missing temperature data
"""

# Import python packages

import pandas as pd
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def interpolate_temperature(sampling_rate, input_data, static_params):
    """
    Function interpolates temperature
    Parameters:
        sampling_rate       (int)          : sampling rate at which the data is given
        input_data          (np.array)     : 21 column data matrix
        static_params       (dict)         : Dictionary with SMB specific parameters

    Returns:
        input_data          (np.array)     : 21 column data matrix
    """

    # Set columns to group by

    group_columns = Cgbdisagg.INPUT_HOD_IDX

    # Set columns to fill

    interpolate_columns = Cgbdisagg.INPUT_FEELS_LIKE_IDX

    # Set number of points required

    params = static_params.get('temp_interpolation')

    num_pts_per_hr = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)
    window_size = params.get('interpolation_window_days') * num_pts_per_hr
    min_periods = params.get('required_data_days') * num_pts_per_hr

    idx_list = np.arange(0, np.shape(input_data)[0])

    # Interpolate across the days

    for hour_idx in range(Cgbdisagg.HRS_IN_DAY):
        hour_data_idx = idx_list[input_data[:, group_columns] == hour_idx]
        hour_temp_data = input_data[hour_data_idx, interpolate_columns]
        missing_value_idx = np.where(np.isnan(hour_temp_data))[0]

        # Convert to data frame to make things easier

        hour_temp_df = pd.DataFrame(data=np.copy(hour_temp_data))
        interp_values_df = hour_temp_df.rolling(window=window_size, min_periods=min_periods, center=True).mean()

        interp_temp_values = np.reshape(interp_values_df.values, newshape=(len(hour_temp_data, )))
        input_data[hour_data_idx[missing_value_idx], interpolate_columns] = interp_temp_values[missing_value_idx]

    # Interpolate across hours with a 15 hour limit

    input_df = pd.DataFrame(data=input_data)

    temp_interpolation_limit = params.get('max_hours_to_interpolate') * num_pts_per_hr
    interpolation_columns = Cgbdisagg.INPUT_FEELS_LIKE_IDX

    # Interpolate temperature across hours using data frame

    input_df[interpolation_columns] = input_df[interpolation_columns].interpolate(limit=temp_interpolation_limit,
                                                                                  limit_area='inside',
                                                                                  limit_direction='both')

    input_data = input_df.values

    return input_data
