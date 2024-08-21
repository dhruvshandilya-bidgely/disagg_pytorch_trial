"""
Author: Mayank Sharan
Created: 12-Jul-2020
Interpolate weather data asa much as we can to ensure completeness
"""

# Import python packages

import logging
import numpy as np
import pandas as pd


def interpolate_weather_data(day_wise_data_dict, logger_pass):

    """
    Interpolate weather data that might be missing
    Parameters:
        day_wise_data_dict      (dict)          : Dictionary containing all day wise data matrices
        logger_pass             (dict)          : Dictionary containing objects needed for logging
    Returns:
        day_wise_data_dict      (dict)          : Dictionary containing all day wise data matrices
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('interpolate_weather_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Extract the different 2d matrices

    day_temp_data = day_wise_data_dict.get('temp')
    day_fl_data = day_wise_data_dict.get('fl')
    day_prec_data = day_wise_data_dict.get('prec')
    day_snow_data = day_wise_data_dict.get('snow')

    # Fill missing values in precipitation and snowfall data

    day_snow_data[np.isnan(day_snow_data)] = 0
    day_prec_data[np.isnan(day_prec_data)] = 0

    # Interpolate linearly to fill temperature and feelsLike values if missing more than 5%

    interpolation_window_days = 5
    interpolation_window_hours = 3

    temp_missing_perc = np.sum(np.isnan(day_temp_data)) * 100 / day_temp_data.size
    fl_missing_perc = np.sum(np.isnan(day_fl_data)) * 100 / day_fl_data.size

    if temp_missing_perc > 5:

        logger.info('Performing temperature interpolation. Missing percentage | %.2f', temp_missing_perc)

        day_temp_df = pd.DataFrame(np.copy(day_temp_data))

        day_temp_df = day_temp_df.interpolate(limit=interpolation_window_days, limit_area='inside',
                                              limit_direction='both', axis=0)

        day_temp_df = day_temp_df.interpolate(limit=interpolation_window_hours, limit_area='inside',
                                              limit_direction='both', axis=1)

        day_temp_data = day_temp_df.values

    else:
        logger.info('Temperature data needs no interpolation. Missing percentage | %.2f', temp_missing_perc)

    if fl_missing_perc > 5:

        logger.info('Performing feels like interpolation. Missing percentage | %.2f', fl_missing_perc)

        day_fl_df = pd.DataFrame(np.copy(day_fl_data))

        day_fl_df = day_fl_df.interpolate(limit=interpolation_window_days, limit_area='inside',
                                          limit_direction='both', axis=0)

        day_fl_df = day_fl_df.interpolate(limit=interpolation_window_hours, limit_area='inside',
                                          limit_direction='both', axis=1)

        day_fl_data = day_fl_df.values

    else:
        logger.info('Feels like data needs no interpolation. Missing percentage | %.2f', fl_missing_perc)

    # Populate the dictionary with interpolated values

    day_wise_data_dict['temp'] = day_temp_data
    day_wise_data_dict['fl'] = day_fl_data
    day_wise_data_dict['prec'] = day_prec_data
    day_wise_data_dict['snow'] = day_snow_data

    return day_wise_data_dict
