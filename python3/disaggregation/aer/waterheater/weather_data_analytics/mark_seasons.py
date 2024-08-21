"""
Author: Mayank Sharan
Created: 12-Jul-2020
Mark seasons at a day level
"""

# Import python packages

import logging
import numpy as np

# Import project functions and classes

from python3.disaggregation.aer.waterheater.weather_data_analytics.math_utils import find_seq

from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import fit_gmm
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import unite_season
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import get_avg_data
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import merge_seq_arr
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import ensure_tr_season
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import identify_seasons
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import modify_event_tags
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import mark_short_temp_events
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import make_seasons_continuous
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import mark_transition_seasons
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import mark_preliminary_seasons
from python3.disaggregation.aer.waterheater.weather_data_analytics.mark_season_utils import perform_hysteresis_smoothening

from python3.disaggregation.aer.waterheater.weather_data_analytics.init_mark_season_config import init_mark_season_config


def mark_seasons(day_wise_data_dict, chunk_info, class_name, prev_chunk_dict, logger_pass):

    """
    Mark seasons per day for the weather data corresponding to a chunk
    Parameters:
        day_wise_data_dict      (dict)          : Dictionary containing all day wise data matrices
        chunk_info              (np.ndarray)    : Information about the latest 1 year or lesser chunk to be used
        class_name              (str)           : The string representing the koppen class the user belongs too
        prev_chunk_dict         (dict)          : The dictionary containing information from previous chunk run
        logger_pass             (dict)          : Dictionary containing objects needed for logging
    Returns:
        chunk_season_dict       (dict)          : Dictionary containing chunk season detection data
    """

    # Initialize the logger

    logger_pass = logger_pass.copy()

    logger_base = logger_pass.get('logger_base').getChild('mark_season')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Extract data for the chunk

    start_idx = chunk_info[0]
    end_idx = chunk_info[1]
    num_days_data = chunk_info[2]
    chunk_type = chunk_info[3]

    mark_season_config = init_mark_season_config()

    # Compute day level average data

    daily_avg, daily_day_avg, daily_night_avg, smooth_daily_avg = get_avg_data(day_wise_data_dict, start_idx, end_idx,
                                                                               mark_season_config)

    # For chunk type 0 fit GMM and for type 1 use the info from last chunk

    if chunk_type == 0:

        logger.info('Full size chunk preparing to fit model |')

        # Initialize and fit gmm models on the data

        model, model_day, model_night = fit_gmm(daily_avg, daily_day_avg, daily_night_avg, class_name,
                                                mark_season_config)

        # Compute season boundaries

        min_x = int(np.floor(np.nanmin(daily_night_avg)))
        max_x = int(np.ceil(np.nanmax(daily_day_avg)))

        x = np.linspace(min_x, max_x, num=max_x - min_x + 1)
        y = model.predict(np.reshape(x, newshape=(len(x), 1)))
        y_d = model_day.predict(np.reshape(x, newshape=(len(x), 1)))
        y_n = model_night.predict(np.reshape(x, newshape=(len(x), 1)))

        season_switch_thr = x[np.r_[np.diff(y) >= 1, False]]
        season_switch_thr_d = x[np.r_[np.diff(y_d) >= 1, False]]
        season_switch_thr_n = x[np.r_[np.diff(y_n) >= 1, False]]

        model_info_dict = {
            'model': model,
            'model_day': model_day,
            'model_night': model_night,
            'model_means': (np.round(model.means_[:, 0])).astype(int),
            'model_weights': (np.round(model.weights_ * num_days_data)).astype(int),
            'model_means_d': (np.round(model_day.means_[:, 0])).astype(int),
            'model_weights_d': (np.round(model_day.weights_ * num_days_data)).astype(int),
            'model_means_n': (np.round(model_night.means_[:, 0])).astype(int),
            'model_weights_n': (np.round(model_night.weights_ * num_days_data)).astype(int),
            'season_switch_thr': season_switch_thr,
            'season_switch_thr_d': season_switch_thr_d,
            'season_switch_thr_n': season_switch_thr_n,
        }

        # Identify presence of seasons

        valid_season_bool, is_longer_summer, is_longer_winter = identify_seasons(model_info_dict, y, mark_season_config)

        logger.info('Day summer boolean | %s', str(is_longer_summer))
        logger.info('Night winter boolean | %s', str(is_longer_winter))

        model_info_dict['valid_season_bool'] = valid_season_bool
        model_info_dict['is_longer_summer'] = is_longer_summer
        model_info_dict['is_longer_winter'] = is_longer_winter

    else:

        logger.info('Small chunk loading model info from previous chunk |')

        model_info_dict = prev_chunk_dict.get('model_info_dict')

    # Initialize the season label array

    s_label = np.full(shape=(num_days_data,), fill_value=np.nan)

    s_label, max_winter_temp, max_tr_temp = mark_preliminary_seasons(s_label, daily_avg, daily_day_avg, daily_night_avg,
                                                                     class_name, model_info_dict, mark_season_config)

    logger.info('Max winter temperature | %s', str(max_winter_temp))
    logger.info('Max transition temperature | %s', str(max_tr_temp))

    # Mark transition summer and winter

    s_label = mark_transition_seasons(s_label, daily_avg, daily_day_avg, daily_night_avg, max_winter_temp, max_tr_temp,
                                      model_info_dict)

    # Perform hysteresis smoothening

    s_label = perform_hysteresis_smoothening(s_label, mark_season_config)

    # Mark short temperature events

    is_hot_event_bool, is_cold_event_bool = mark_short_temp_events(daily_avg, smooth_daily_avg, s_label,
                                                                   mark_season_config)

    # Remove sporadic season tags

    min_sl = 4

    seq_arr = find_seq(s_label, min_seq_length=min_sl)

    # In extremely small partial chunks we sometimes do not get sequences of length 4. This fixes that

    while seq_arr.shape[0] == 0:
        min_sl -= 1
        seq_arr = find_seq(s_label, min_seq_length=min_sl)

    seq_arr = merge_seq_arr(seq_arr, num_days_data)

    # Modify season tags for chunks with big parts marked as temperature events

    seq_arr = modify_event_tags(seq_arr, is_hot_event_bool, is_cold_event_bool, num_days_data, mark_season_config)

    # Unite the summer and winter seasons

    seq_arr = unite_season(seq_arr, smooth_daily_avg, max_tr_temp, max_winter_temp, num_days_data, mark_season_config,
                           season='summer')

    seq_arr = unite_season(seq_arr, smooth_daily_avg, max_tr_temp, max_winter_temp, num_days_data, mark_season_config,
                           season='winter')

    # Make seasons continuous

    seq_arr = make_seasons_continuous(seq_arr, smooth_daily_avg, max_tr_temp, max_winter_temp, num_days_data)

    # Make sure summer is cushioned with summer transitions

    seq_arr = ensure_tr_season(seq_arr, smooth_daily_avg, max_tr_temp, max_winter_temp, season='summer')

    # Make sure winter is cushioned with winter transitions

    seq_arr = ensure_tr_season(seq_arr, smooth_daily_avg, max_tr_temp, max_winter_temp, season='winter')

    # Update labels as per processing

    for idx in range(seq_arr.shape[0]):
        curr_seq = seq_arr[idx, :]
        s_label[int(curr_seq[1]): int(curr_seq[2]) + 1] = curr_seq[0]

    # Populate chunk season dictionary and return

    chunk_season_dict = {
        's_label': s_label,
        'seq_arr': seq_arr,
        'is_hot_event_bool': is_hot_event_bool,
        'is_cold_event_bool': is_cold_event_bool,
        'max_winter_temp': max_winter_temp,
        'max_tr_temp': max_tr_temp,
        'model_info_dict': model_info_dict,
    }

    return chunk_season_dict
