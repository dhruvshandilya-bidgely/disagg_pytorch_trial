"""
Author: Mayank Sharan
Created: 12-Jul-2020
Process weather data to detect seasons and calculate information to be used by the engine
"""

# Import python packages

import logging
from datetime import datetime

# Import project functions and classes

from python3.disaggregation.aer.waterheater.weather_data_analytics.time_utils import get_time_diff
from python3.disaggregation.aer.waterheater.weather_data_analytics.nbi_constants import TimeConstants

from python3.disaggregation.aer.waterheater.weather_data_analytics.detect_season import detect_season
from python3.disaggregation.aer.waterheater.weather_data_analytics.compute_hvac_potential import compute_hvac_potential
from python3.disaggregation.aer.waterheater.weather_data_analytics.homogenize_weather_data import homogenize_weather_data
from python3.disaggregation.aer.waterheater.weather_data_analytics.interpolate_weather_data import interpolate_weather_data
from python3.disaggregation.aer.waterheater.weather_data_analytics.get_day_wise_weather_data import get_day_wise_weather_data


def process_weather_data(nbi_input_data, logger_pass):

    """
    Process weather data for use by the engine
    Parameters:
        nbi_input_data          (dict)          : Dictionary containing data needed for the engine run
        logger_pass             (dict)          : Dictionary containing objects needed for logging
    Returns:
        nbi_input_data          (dict)          : Dictionary containing data needed for the engine run
        exit_swh                (Bool)          : Exit status
    """

    # Initialize the logger
    exit_swh = False

    logger_pass = logger_pass.copy()

    logger_base = logger_pass.get('logger').getChild('process_weather_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    logger_pass['logger_base'] = logger_base

    # Initialize variables to be use d

    weather_data = nbi_input_data.get('weather', {}).get('raw_weather', [])
    meta_data = nbi_input_data.get('meta_data', {})

    if len(weather_data) == 0:
        logger.warning('Weather data missing, skipping processing |')
        exit_swh = True
        return nbi_input_data, exit_swh

    # Make the weather data continuous and down sample to 1 hour rate
    try:
        t_before_homogenization = datetime.now()
        homo_weather_data = homogenize_weather_data(weather_data, TimeConstants.sec_in_1_hr, logger_pass)
        t_after_homogenization = datetime.now()

        logger.debug('Weather homogenization took | %.3f s', get_time_diff(t_before_homogenization, t_after_homogenization))

        # Convert weather data to day wise matrix

        t_before_day_wise = datetime.now()
        day_wise_data_dict = get_day_wise_weather_data(homo_weather_data)
        t_after_day_wise = datetime.now()

        logger.debug('Day wise data creation took | %.3f s', get_time_diff(t_before_day_wise, t_after_day_wise))

        # Interpolate weather data

        t_before_interpolation = datetime.now()
        day_wise_data_dict = interpolate_weather_data(day_wise_data_dict, logger_pass)
        t_after_interpolation = datetime.now()

        logger.debug('Weather interpolation took | %.3f s', get_time_diff(t_before_interpolation, t_after_interpolation))

        # Chunk data into year long blocks and mark seasons then unite chunks

        t_before_season_det = datetime.now()
        season_detection_dict = detect_season(day_wise_data_dict, meta_data, logger_pass)
        t_after_season_det = datetime.now()

        logger.debug('Season detection took | %.3f s', get_time_diff(t_before_season_det, t_after_season_det))

        if len(season_detection_dict) == 0:
            logger.warning('Insufficient weather data, skipping processing |')
            exit_swh = True
            return nbi_input_data, exit_swh

        # Compute hvac potential

        t_before_hvac_potential = datetime.now()
        hvac_potential_dict = compute_hvac_potential(day_wise_data_dict, season_detection_dict)
        t_after_hvac_potential = datetime.now()

        logger.debug('HVAC potential calculation took | %.3f s', get_time_diff(t_before_hvac_potential,
                                                                               t_after_hvac_potential))

        # Populate the information computed in the nbi input data dictionary

        nbi_input_data['weather']['pr_weather'] = homo_weather_data
        nbi_input_data['weather']['day_wise_data'] = day_wise_data_dict
        nbi_input_data['weather']['season_detection_dict'] = season_detection_dict
        nbi_input_data['weather']['hvac_potential_dict'] = hvac_potential_dict

    except Exception:
        exit_swh = True
        logger.warning('Exiting Weather analytics module due to unexpected issues | ')

    return nbi_input_data, exit_swh
