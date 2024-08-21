"""
Author: Mayank Sharan
Created: 12-Jul-2020
Convert weather data to day wise 2d matrices
"""

# Import python packages

import copy
import numpy as np

# Import project functions and classes

from python3.disaggregation.aer.waterheater.weather_data_analytics.nbi_constants import TimeConstants

from python3.disaggregation.aer.waterheater.weather_data_analytics.nbi_data_constants import WeatherData


def get_day_wise_weather_data(weather_data):

    """
    Convert all columns of weather data to 2d day wise matrix
    Parameters:
        weather_data            (np.ndarray)    : Array containing weather data
    Returns:
        day_wise_data_dict      (dict)          : Dictionary containing all day wise data matrices
    """

    weather_data = copy.deepcopy(weather_data)

    # Get unique day timestamps and col idx to convert into 2d array

    day_ts, rev_idx = np.unique(weather_data[:, WeatherData.day_ts_col], return_inverse=True)

    col_idx = (weather_data[:, WeatherData.epoch_ts_col] - weather_data[:, WeatherData.day_ts_col]) \
        // TimeConstants.sec_in_1_hr

    # This will cause issue with DST days so limit max value to 23

    col_idx[col_idx > (TimeConstants.hr_in_1_day - 1)] = TimeConstants.hr_in_1_day - 1
    col_idx = col_idx.astype(int)

    # Initialize 2d arrays to be populated

    epoch_data = np.full(shape=(len(day_ts), TimeConstants.hr_in_1_day), fill_value=np.nan)
    day_temp_data = np.full(shape=(len(day_ts), TimeConstants.hr_in_1_day), fill_value=np.nan)
    day_fl_data = np.full(shape=(len(day_ts), TimeConstants.hr_in_1_day), fill_value=np.nan)
    day_prec_data = np.full(shape=(len(day_ts), TimeConstants.hr_in_1_day), fill_value=np.nan)
    day_snow_data = np.full(shape=(len(day_ts), TimeConstants.hr_in_1_day), fill_value=np.nan)

    # Populate 2d arrays with values

    epoch_data[rev_idx, col_idx] = weather_data[:, WeatherData.epoch_ts_col]
    day_temp_data[rev_idx, col_idx] = weather_data[:, WeatherData.temp_col]
    day_fl_data[rev_idx, col_idx] = weather_data[:, WeatherData.feels_like_col]
    day_prec_data[rev_idx, col_idx] = weather_data[:, WeatherData.prec_col]
    day_snow_data[rev_idx, col_idx] = weather_data[:, WeatherData.snow_col]

    # Populate and return dictionary with computed arrays2

    day_wise_data_dict = {
        'day_ts': day_ts,
        'epoch': epoch_data,
        'temp': day_temp_data,
        'fl': day_fl_data,
        'prec': day_prec_data,
        'snow': day_snow_data,
    }

    return day_wise_data_dict
