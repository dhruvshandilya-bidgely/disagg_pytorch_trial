"""
Author - Prasoon Patidar
Date - 18th June 2020
Lifestyle Submodule to calculate lowcooling constant for office goer users
lowcooling constant: Fraction of cooling which was supposed to be there,
but actually not there in summer and transition during day times, indicating
presence of office goer users.
"""

# import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.lifestyle.functions.lifestyle_utils import get_day_level_2d_matrix


def get_lowcooling_constant(lifestyle_input_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object(dict)               : dictionary containing inputs for lifestyle modules
        logger_pass(dict)                          : contains base logger and logging dictionary
    Returns:
        lowcooling_constant(float)                 : lowcooling constant for this user
        debug(dict)                                : step wise info for debugging and plotting purposes
    """

    t_low_cooling_module_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_lowcooling_constant')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    debug = dict()

    # Get input data, cooling estimate and config info from lifestyle input object

    cooling_epoch_estimate = lifestyle_input_object.get('cooling_epoch_estimate')

    # return zero if cooling_epoch_estimate is zero

    if np.sum(cooling_epoch_estimate) <= 0:
        logger.info("%s No cooling epoch estimate available", log_prefix('OfficeGoer'))

        low_cooling_constant = 0

        return low_cooling_constant, debug

    input_data = lifestyle_input_object.get('input_data')

    low_cooling_config = lifestyle_input_object.get('lowcooling_config')

    weather_config = lifestyle_input_object.get('weather_config')

    seasons = weather_config.get('season')

    season_idx = lifestyle_input_object.get('SEASON_IDX')

    # remove winter season from input data and cooling epoch estimate

    non_winter_input_data = input_data[~(input_data[:, season_idx] == seasons.winter.value), :]

    non_winter_cooling = cooling_epoch_estimate[~(input_data[:, season_idx] == seasons.winter.value)]

    low_cooling_constant_flag = True

    # if non winter input data is none, return

    if non_winter_input_data.shape[0] == 0:
        # return zero precooling constant

        logger.info("%s Non winter data not available for input, unable to evaluate lowcooling constant",
                    log_prefix('OfficeGoer'))

        low_cooling_constant = 0.

        low_cooling_constant_flag = False

    # do not calculate lowcooling constant if non winter cooling is not available

    if np.sum(non_winter_cooling) <= 0.:
        # return zero precooling constant

        logger.info("%s Non winter cooling is not non-zero, unable to evaluate lowcooling constant", log_prefix('OfficeGoer'))

        low_cooling_constant = 0.

        low_cooling_constant_flag = False

    if not low_cooling_constant_flag:
        return low_cooling_constant, debug

    # Get CDD Values based on temperature setpoint in weather config

    temp_setpoint = weather_config.get('temperature_setpoint')

    input_temp_vals = non_winter_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    if temp_setpoint is None:
        # If setpoint is not available in config, use default setpoint to be 65

        temp_setpoint = lifestyle_input_object.get("weather_config").get("default_setpoint")

        logger.info("%s Default temperature setpoint %s", log_prefix('OfficeGoer'), str(temp_setpoint))

    input_cdd_vals = np.maximum(0, input_temp_vals - temp_setpoint)

    # get cdd buckets based on weather data

    num_buckets = low_cooling_config.get('cdd_bincount')

    _, cdd_buckets = np.histogram(input_cdd_vals[input_cdd_vals > 0], bins=num_buckets)

    cdd_buckets = np.insert(cdd_buckets, 0, -1)

    # get cdd bucket for each CDD value in input data

    #TODO(Nisha) : Replace this by numpy digitize

    get_cdd_bucket = lambda x: np.where(cdd_buckets < x)[0][-1]

    fn_cdd_bucket = np.vectorize(get_cdd_bucket)

    input_cdd_buckets = fn_cdd_bucket(input_cdd_vals)

    # initialize input level lowcooling function

    input_low_cooling = np.zeros(non_winter_input_data.shape[0])

    # loop over all buckets to get lowcooling points

    base_cooling_percentile = low_cooling_config.get('bucket_normed_cons_percentile')

    for bucket_idx in range(num_buckets + 1):
        # get base cooling value for a given bucket

        bucket_non_winter_cooling = non_winter_cooling[input_cdd_buckets == bucket_idx]

        if bucket_non_winter_cooling.shape[0] > 0:

            base_cooling_consumption = np.percentile(bucket_non_winter_cooling,
                                                     base_cooling_percentile)

        else:

            base_cooling_consumption = 0.

        if base_cooling_consumption > 0:
            # fill lowcooling for all points in bucket where non-winter cooling is greater than lowcooling_percentile

            input_low_cooling[(input_cdd_buckets == bucket_idx) & (non_winter_cooling <= base_cooling_consumption)] = 1

    # get lowcooling start and end hour

    start_hour = low_cooling_config.get('start_hour')

    end_hour = low_cooling_config.get('end_hour')

    # insert lowcooling data inplace of consumption data and get day_val

    non_winter_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = input_low_cooling

    # Get day level data for weekdays lowcooling

    WEEKDAY_IDX = lifestyle_input_object.get('WEEKDAY_IDX')

    _, day_low_cooling_data = \
        get_day_level_2d_matrix(non_winter_input_data[non_winter_input_data[:, WEEKDAY_IDX] == True, :], logger_pass)

    # remove nan values from day level data

    day_low_cooling_data[np.isnan(day_low_cooling_data)] = 0.

    # Trim day_lowcooling data for allowed lowcooling hours

    day_low_cooling_data = day_low_cooling_data[:, start_hour:end_hour + 1]

    # get daily lowcooling fraction for lowcooling hour

    daily_low_cooling_fraction = np.sum(day_low_cooling_data, axis=1) / day_low_cooling_data.shape[1]

    # Get lowcooling constant

    weekday_low_cooling_fraction_agg_percentile = low_cooling_config.get('weekday_lowcooling_fraction_agg_percentile')

    if daily_low_cooling_fraction.shape[0] > 0:

        low_cooling_constant = np.percentile(daily_low_cooling_fraction, weekday_low_cooling_fraction_agg_percentile)

    else:

        low_cooling_constant = 0.

    t_low_cooling_module_end = datetime.now()

    logger.info("%s Running lowcooling module took | %.3f s", log_prefix('OfficeGoer'),
                get_time_diff(t_low_cooling_module_start, t_low_cooling_module_end))

    return low_cooling_constant, debug
