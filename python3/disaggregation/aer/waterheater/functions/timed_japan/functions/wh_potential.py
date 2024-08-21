"""
Author - Sahana M
Date - 23/06/2021
Calculate wh potential
"""

# Import python packages
import logging
import numpy as np
from datetime import datetime

# Import functions from within the project
from python3.utils.time.get_time_diff import get_time_diff
from python3.utils.maths_utils.maths_utils import rotating_avg
from python3.disaggregation.aer.waterheater.weather_data_analytics.math_utils import find_seq
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.get_weather_data_inputs import get_meta_data
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.get_weather_data_inputs import get_weather_data
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.process_weather_data import process_weather_data
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.get_alternate_weather_analytics import alternate_weather_data_output


def calculate_wh_potential(weather_data_output, wh_pot_config, logger_base):

    """
    Returns Water heater potential based on weather data analytics output
    Args:
        weather_data_output     (dict)      : Dictionary containing output from process weather data module
        wh_pot_config           (dict)      : Dictionary containing all needed configuration variables
        logger_base             (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        wh_potential            (np.array)  : Water heater potential array
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('calculate_wh_potential')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Initialise all the variables to be used

    min_seq_len = wh_pot_config.get('min_seq_length')
    s_label_weight = wh_pot_config.get('s_label_weight')
    pot_avg_window = wh_pot_config.get('pot_avg_window')
    type1_base_pot = wh_pot_config.get('type1_base_pot')
    type2_base_pot = wh_pot_config.get('type2_base_pot')
    scaling_threshold = wh_pot_config.get('scaling_threshold')
    max_temp_limit = wh_pot_config.get('max_temperature_limit')
    min_temp_limit = wh_pot_config.get('min_temperature_limit')
    type1_koppen_class = wh_pot_config.get('type1_koppen_class')
    feels_like = weather_data_output.get('weather').get('day_wise_data').get('fl')
    temperature = weather_data_output.get('weather').get('day_wise_data').get('temp')
    s_label = weather_data_output.get('weather').get('season_detection_dict').get('s_label')
    class_name = weather_data_output.get('weather').get('season_detection_dict').get('class_name')
    max_tr_temp = weather_data_output.get('weather').get('season_detection_dict').get('max_tr_temp')
    max_winter_temp = weather_data_output.get('weather').get('season_detection_dict').get('max_winter_temp')
    valid_season_bool = weather_data_output.get('weather').get('season_detection_dict').get('model_info_dict'). \
        get('valid_season_bool')

    # If winter present calculate wh_usage_threshold temperature with winter max temp as reference

    if valid_season_bool[0]:
        winter_offset = {
            'A': 0,
            'B': 0,
            'Bk': 0,
            'Bh': 2,
            'Ch': 3,
            'C': 5,
            'Ck': 3,
            'D': 5,
            'E': 7
        }

        # Handling different geographies temperature by customizing the wh_usage_threshold with an offset

        wh_usage_threshold = min(max_winter_temp + winter_offset.get(class_name), max_temp_limit)

        scaling_bool = True

    # If winter absent calculate wh_usage_threshold temperature with transition max temp as reference

    else:
        summer_offset = {
            'A': -7.2,
            'B': -7.2,
            'Bk': -7.2,
            'Bh': -9,
            'Ch': -5,
            'C': -5,
            'Ck': -5,
            'D': -5,
            'E': 7
        }

        # Handling different geographies temperature by customizing the wh_usage_threshold with an offset

        wh_usage_threshold = max(max_tr_temp + summer_offset.get(class_name), min_temp_limit)

        scaling_bool = False

    # Get the mean of day level Feels like temperature

    if np.sum(feels_like != 0) < feels_like.shape[0]*feels_like.shape[1]*0.8:
        feels_like_1d = np.nanmean(temperature, axis=1)

    else:
        feels_like_1d = np.nanmean(feels_like, axis=1)

    # Get the deviation of daily feels_like_1d with the wh_usage_threshold

    wh_potential = wh_usage_threshold - feels_like_1d

    # Perform scaling to enable estimation on transition days

    if scaling_bool:
        is_wh_day = wh_potential > scaling_threshold
    else:
        is_wh_day = wh_potential > 0

    # Perform scaling

    wh_potential = (wh_potential - 3) / 2

    # Perform tanh transformation to mimic Water Heater usage

    tanh_usage = np.tanh(wh_potential)

    # Include Season label as a bonus with 20% weightage

    wh_potential = s_label_weight * (1 - s_label) / 2 + (1-s_label_weight) * (tanh_usage + 1)/2

    wh_potential[wh_potential > 1] = 1

    # Make all the non wh days as having 0 potential

    wh_potential[~is_wh_day] = 0

    # Assign a base potential based on Koppen class

    if class_name in type1_koppen_class:
        base_pot = type1_base_pot
        min_base_pot = 0.2
    else:
        base_pot = type2_base_pot
        min_base_pot = 0.3

    # Calculate the new wh potential along with the base potential

    base_wh_pot = base_pot * (wh_potential > 0)
    wh_potential = base_wh_pot + min_base_pot + (1-base_pot) * wh_potential
    wh_potential[wh_potential > 1] = 1

    # Make sure that the wh potential has minimum 3 days continuity

    is_wh_day = wh_potential > 0
    wh_days_seq = find_seq(is_wh_day, min_seq_length=min_seq_len)
    wh_days_seq = wh_days_seq[wh_days_seq[:, 0] == 1, :]

    is_wh_day[:] = False

    for idx in range(wh_days_seq.shape[0]):
        curr_seq = wh_days_seq[idx, :]
        is_wh_day[int(curr_seq[1]): int(curr_seq[2]) + 1] = True

    # Removing all the less continuous days

    wh_potential[~is_wh_day] = 0

    # Get the final water heater potential by performing rotating average for smoothing

    wh_pot_indexes = wh_potential > 0
    wh_potential[wh_pot_indexes] = rotating_avg(wh_potential[wh_pot_indexes], pot_avg_window)

    logger.info('Number of days marked as WH potential | %d ', np.sum(wh_pot_indexes))

    return wh_potential


def get_wh_potential(input_data, wh_config, debug, logger_base):
    """
    Get wh potential function is used to identify the wh potential for every day of the data
    Parameters:
        input_data          (np.ndarray)        : Input 21 column array
        wh_config           (dict)              : WH configurations dictionary
        debug               (dict)              : Algorithm outputs
        logger_base         (logger)            : Logger passed
    Returns:
        wh_potential        (np.ndarray)        : WH potential ranging 0-1
        fl                  (np.ndarray)        : Feels like data for each hour of the day
        season_detection_dict (dict)            : Contains all the season related info
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('get_wh_potential')
    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Get weather data and meta data from input_data and wh_config

    weather_data = get_weather_data(input_data)
    weather_data = get_meta_data(weather_data, wh_config)

    # Pass the weather data dictionary to the weather data analytics module
    t1 = datetime.now()
    weather_data_output, exit_swh = process_weather_data(weather_data, logger_pass)
    t2 = datetime.now()
    wda_time = get_time_diff(t1, t2)
    debug['wda_time'] = wda_time

    # If weather data analytics module didn't run then go for alternate approach

    if exit_swh:
        logger.info('Running Alternate weather data analytics module | ')
        weather_data_output, exit_swh = alternate_weather_data_output(input_data, debug, wh_config, logger_pass)

    if exit_swh:
        logger.warning('Could not derive Weather Data Features | ')

    # Calculate the WH potential

    wh_potential = calculate_wh_potential(weather_data_output, wh_config['weather_data_configs'], logger_pass)

    fl = weather_data_output.get('weather').get('day_wise_data').get('fl')
    season_detection_dict = weather_data_output.get('weather').get('season_detection_dict')

    return wh_potential, fl, season_detection_dict, debug
