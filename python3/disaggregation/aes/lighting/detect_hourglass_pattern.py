"""
Author: Neelabh Goyal
Date:   14-June-2023
Called to detect external lighting (hourglass) pattern
"""

# Import python packages

import copy
import timeit
import logging
import scipy.stats
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def clean_data(in_data):
    """
    Function to handle nan values before the hourglass detection
    Parameters:
        in_data    (pd.DataFrame): DataFrame with input data at epoch level

    Returns:
        input_data (pd.DataFrame): Dataframe with cleaned and extracted input data
    """
    input_data = copy.deepcopy(in_data)
    columns = ['day', 'temperature', 'sunrise', 'sunset']
    # Imputing NaNs with values to ensure equal sizes for all pivoted dataframe
    smoothened_data = input_data[columns].ewm(span=int(input_data.shape[0]), axis=0).mean().values

    condition = np.isnan(input_data['day'])
    input_data['day'][condition] = smoothened_data[:, 0][condition].astype(int)

    condition = np.isnan(input_data['temperature'])
    input_data['temperature'][condition] = smoothened_data[:, 1][condition]

    condition = np.isnan(input_data['sunrise'])
    input_data['sunrise'][condition] = smoothened_data[:, 2][condition].astype(int)

    condition = np.isnan(input_data['sunset'])
    input_data['sunset'][condition] = smoothened_data[:, 3][condition].astype(int)

    del condition, smoothened_data, columns

    input_data['s_label'].ffill(limit=4, inplace=True)
    return input_data.fillna(0)


def get_hourglass_edge(raw_data, windows, window_size, sampling_rate_proxy, start_sample, evening=False):
    """
    Function to identify edges of the hourglass pattern from raw data
    Parameters:
        raw_data             (np.array): 2D array with epoch level raw consumption data
        windows              (np.range): Range to identify the count of windows to process
        window_size          (int)     : Size of each window
        sampling_rate_proxy  (int)     : Sampling rate of the user
        start_sample         (int)     : Index of the first sample under analysis
        evening              (bool)    : Bool to indicate if edges to be calculated are for evening time
    Returns:
        edge_               (np.array) : Array containing index of the identified edge samples
        median_             (np.array) : 2D array generated when calculating edges
    """

    medians_ = np.zeros((len(raw_data), len(windows)))
    for window in windows:
        medians_[:, window - start_sample] = np.nanmedian(raw_data[:, window - window_size: window], axis=1) - \
                                             np.nanmedian(raw_data[:, window: window + window_size], axis=1)

    if evening:
        edge = np.nanargmin(medians_, axis=1) + sampling_rate_proxy
    else:
        edge = np.nanargmax(medians_, axis=1) + sampling_rate_proxy

    edge[edge > np.nanpercentile(edge, 99)] = np.nanpercentile(edge, 99)
    edge_ = pd.DataFrame(edge + start_sample).ewm(span=30, axis=0, ignore_na=False).mean().values

    return edge_, medians_


def check_corr_less_days(morning_edge_, evening_edge_, morning_edge, sun_rise, evening_edge, sun_set, valid_days,
                         hourglass_pattern, params):
    """
    Function to check correlation when count of valid days is less than threshold
    Parameters:
        morning_edge_      (np.array): Array containing indexes denoting morning edge
        evening_edge_      (np.array): Array containing indexes denoting evening edge
        morning_edge       (np.array): Array containing indexes denoting processed morning edge
        evening_edge       (np.array): Array containing indexes denoting processed evening edge
        sun_rise           (np.array): Array containing indexes denoting sun_rise epochs
        sun_set            (np.array): Array containing indexes denoting sun_set epochs
        valid_days         (np.array): Boolean array indicating valid days
        hourglass_pattern  (dict)    : Dictionary containing hourglass related outputs
        params             (dict)    : Dictionary containing constants

    Returns:
        hourglass_pattern  (dict)    : Dictionary containing hourglass related outputs
    """
    corr_evening_morning_edge, _ = spearmanr(morning_edge_, evening_edge_ * -1)

    # Correlate the edges with corresponding sun-time (rise/set). Good correlation = presence of hourglass pattern.
    morning_corr, _ = spearmanr(morning_edge, sun_rise[valid_days])
    morning_corr = round(morning_corr, 3)
    evening_corr, _ = spearmanr(evening_edge, sun_set[valid_days])
    evening_corr = round(evening_corr, 3)

    if corr_evening_morning_edge > params.get('less_days').get('edge_corr') and len(morning_edge) > params.get('min_days'):

        hourglass_pattern['Bool'] = True
        hourglass_pattern['Reason'] = 'Correlation among edges higher than {}%'.format(params.get('less_days').get('edge_corr'))
        hourglass_pattern['Correlation'] = corr_evening_morning_edge

    elif np.maximum(morning_corr, evening_corr) > params.get('less_days').get('sun_edge_corr') and len(morning_edge) > params.get('min_days'):

        hourglass_pattern['Bool'] = True
        hourglass_pattern['Reason'] = 'One of Edge Correlation with sun higher than {}%'.format(
            params.get('less_days').get('sun_edge_corr'))
        hourglass_pattern['Correlation'] = np.maximum(morning_corr, evening_corr)

    else:

        hourglass_pattern['Bool'] = False
        hourglass_pattern['Reason'] = 'Correlation between edges for less count of days not enough'

    return hourglass_pattern


def check_other_corr(morning_edge_, evening_edge_, temperature, temperature_valid, valid_days, hourglass_pattern,
                     evening_corr, morning_corr, params):
    """
    Function to check edge correlation with sunrise/sunset and temperature
    Parameters:
        morning_edge_      (np.array): Array containing indexes denoting morning edge
        evening_edge_      (np.array): Array containing indexes denoting evening edge
        temperature        (np.array): Temperature values for each day
        temperature_valid  (np.array): Boolean to denote days with valid temperature data
        valid_days         (np.array): Boolean array indicating valid days
        hourglass_pattern  (dict)    : Dictionary containing hourglass related outputs
        evening_corr       (float)   : Evening edge's correlation with sunset
        morning_corr       (float)   : Morning edge's correlation with sunrise
        params             (dict)    : Dictionary containing constants

    Returns:
        hourglass_pattern  (dict)    : Dictionary containing hourglass related outputs
    """
    temperature_params = params.get('temperature_params')
    special_threshold = params.get('special_thresh')

    if np.maximum(evening_corr, morning_corr) >= special_threshold.get('sun_edge_corr') or (
            morning_corr + evening_corr) > special_threshold.get('sum_sun_edge_corr'):

        hourglass_pattern['Bool'] = True
        hourglass_pattern['Reason'] = 'One of the Edges has good correlation with sun rise/set'
        hourglass_pattern['correlation'] = np.maximum(evening_corr, morning_corr)

    # Modify further only if we still have at least 50 valid days.
    elif np.count_nonzero(np.logical_and(valid_days, temperature_valid)) > temperature_params.get('min_valid_days'):
        valid_days = np.logical_and(valid_days, temperature_valid)
        morning_edge = morning_edge_[valid_days]
        evening_edge = evening_edge_[valid_days]
        corr_evening_morning_edge, _ = spearmanr(morning_edge, evening_edge)
        corr_evening_morning_edge = round(corr_evening_morning_edge, 2)

        if corr_evening_morning_edge > temperature_params.get('min_edge_corr'):
            edge_distance = evening_edge - morning_edge
            temperature = temperature[valid_days]
            temp_and_edge_corr, _ = spearmanr(temperature, edge_distance.astype(float))
            temp_and_edge_corr = round(temp_and_edge_corr, 2)

            if temp_and_edge_corr > temperature_params.get('min_temp_edge_corr'):
                hourglass_pattern['Bool'] = True
                hourglass_pattern['Reason'] = 'Correlation of edge distance and temperature is good'
                hourglass_pattern['Correlation'] = temp_and_edge_corr

            elif np.isnan(temp_and_edge_corr):
                hourglass_pattern['Bool'] = False
                hourglass_pattern['Reason'] = 'Temperature Unavailable'

            else:
                hourglass_pattern['Bool'] = False
                hourglass_pattern['Reason'] = 'Temp-edge Correlation not strong enough'
                hourglass_pattern['Correlation'] = temp_and_edge_corr

    return hourglass_pattern


def get_hourglass_pattern(in_data, params, logger_pass):
    """
    Function to detect presence of hourglass pattern
    Parameters:
        in_data       (pd.DataFrame)   : Dataframe with raw data of a user
        params        (dict)           : Dictionary with hourglass specific constants
        logger_pass   (Logging object) : Logger to be used here

    Returns:
        hourglass_pattern (dict): Dictionary with info about detection of the hourglass pattern
        hourglass_data    (dict): Dictionary containing data associated with hourglass pattern detection
    """

    logger_local = logger_pass.get("logger").getChild("get_hourglass_pattern")
    logger_hourglass = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger_hourglass.debug(' Reading data in to local object for hourglass pattern detection. |')
    input_data = clean_data(in_data)

    raw_data = input_data.pivot_table(index='date', columns='time', values='consumption').fillna(0).values
    temperature = np.nanmean(input_data.pivot_table(index='date', columns='time', values='temperature'), axis=1)

    input_data['sunrise_'] = input_data['sunrise'] - input_data['day']
    input_data['sunset_'] = input_data['sunset'] - input_data['day']

    sun_rise = input_data.pivot_table(index='date', columns='time', values='sunrise_').fillna(0).values[:, 0]
    sun_set = input_data.pivot_table(index='date', columns='time', values='sunset_').fillna(0).values[:, 0]

    hourglass_data = {
        'temperature': temperature,
        'sunrise': sun_rise,
        'sunset': sun_set
    }

    logger_hourglass.info(' Initialised dictionary object for hourglass pattern detection information. |')
    s_label = np.nanmean(input_data.pivot_table(index='date', columns='time', values='s_label'), axis=1)
    temperature_valid = np.where(np.logical_and(-0.5 <= s_label, s_label <= 0.5), True, False)

    del input_data

    sampling_rate_proxy = int(raw_data.shape[1] / 24)
    window_size = params.get('window_size')

    hourglass_pattern = {
        'morning_hours': np.full(len(raw_data), fill_value=np.NaN),
        'evening_hours': np.full(len(raw_data), fill_value=np.NaN),
        'valid_days': [],
        'Bool': False,
        'Correlation': 0.0,
        'Reason': 'No correlation found'
    }

    morning_start_sample = params.get('morning_start_sample') * sampling_rate_proxy
    morning_end_sample = params.get('morning_end_sample') * sampling_rate_proxy
    start = timeit.default_timer()
    morning_windows = np.arange(morning_start_sample, morning_end_sample)

    logger_hourglass.info(' Identifying morning edges of the hourglass pattern. |')
    morning_edge_, morning_medians = get_hourglass_edge(raw_data, morning_windows, window_size, sampling_rate_proxy,
                                                        morning_start_sample)

    hourglass_pattern['morning_hours'] = np.round(morning_edge_, 0).astype(int)[:, 0]
    logger_hourglass.debug(' Morning edges identified for the hourglass pattern. |')

    cluster_diff_thresh = np.minimum(np.nanpercentile(raw_data, 2),
                                     params.get('cluster_diff_thresh_max') / sampling_rate_proxy)
    cluster_diff_thresh = np.maximum(cluster_diff_thresh, params.get('cluster_diff_thresh_min') / sampling_rate_proxy)
    morning_edge_valid = np.nanmax(morning_medians, axis=1) > cluster_diff_thresh

    evening_start_sample = params.get('evening_start_sample') * sampling_rate_proxy
    evening_end_sample = params.get('evening_end_sample') * sampling_rate_proxy
    evening_windows = np.arange(evening_start_sample, evening_end_sample)

    logger_hourglass.info(' Identifying evening edges of the hourglass pattern. |')
    evening_edge_, evening_medians = get_hourglass_edge(raw_data, evening_windows, window_size, sampling_rate_proxy,
                                                        evening_start_sample, evening=True)

    hourglass_pattern['evening_hours'] = np.round(evening_edge_, 0).astype(int)[:, 0]
    logger_hourglass.debug(' Evening edges identified for the hourglass pattern. |')

    cluster_diff_thresh = np.maximum(-1 * np.nanpercentile(raw_data, 2),
                                     -1 * params.get('cluster_diff_thresh_max') / sampling_rate_proxy)
    cluster_diff_thresh = np.minimum(cluster_diff_thresh,
                                     -1 * params.get('cluster_diff_thresh_min') / sampling_rate_proxy)

    evening_edge_valid = np.nanmin(evening_medians, axis=1) < cluster_diff_thresh

    valid_days = np.logical_and(morning_edge_valid, evening_edge_valid)
    evening_edge = np.round(evening_edge_[valid_days][:, 0], 3)
    morning_edge = np.round(morning_edge_[valid_days][:, 0], 3)

    hourglass_pattern['valid_days'] = valid_days
    del evening_windows, evening_medians

    end = timeit.default_timer()
    logger_hourglass.info(' Edge detection for hourglass pattern took {}s |'.format(round((end - start), 3)))

    # If the number of valid days is less than the 20% of total days, we exit
    if (len(morning_edge)/raw_data.shape[0]) < (params.get('min_valid_days_frac')):

        hourglass_pattern['Bool'] = False
        hourglass_pattern['Reason'] = 'Fraction of valid edges is less than {}'.format(params.get('min_valid_days_frac'))

        logger_hourglass.info(' {}. Hourglass pattern detected | {}'.format(hourglass_pattern.get('Reason'),
                                                                            hourglass_pattern.get('Bool')))

    # If we have edges for less than 90 days, decide hourglass on a higher threshold
    elif len(morning_edge) < params.get('less_days_thresh'):
        logger_hourglass.info('Count of days with valid edges for the user is less than {} | Applying higher thresholds'
                              .format(params.get('less_days_thresh')))

        hourglass_pattern = check_corr_less_days(morning_edge_, evening_edge_, morning_edge, sun_rise, evening_edge,
                                                 sun_set, valid_days, hourglass_pattern, params)

        logger_hourglass.info(' {}. Hourglass pattern detected | {}'.format(hourglass_pattern.get('Reason'),
                                                                            hourglass_pattern.get('Bool')))

    elif scipy.stats.mode(np.round(evening_edge, 0))[1][0] > len(evening_edge) * 0.5 \
            and scipy.stats.mode(np.round(morning_edge, 0))[1][0] > len(morning_edge) * 0.5:
        logger_hourglass.info(' Edges detected are very straight. Skipping further steps of hourglass detection |')
        hourglass_pattern['Bool'] = False
        hourglass_pattern['Reason'] = 'Edges detected are very straight'

    else:
        corr_evening_morning_edge, _ = spearmanr(morning_edge, evening_edge * -1)
        corr_evening_morning_edge = round(corr_evening_morning_edge, 2)

        # Correlate the edges with corresponding sun-time (rise/set). Good correlation = presence of hourglass pattern.
        morning_corr, _ = spearmanr(morning_edge, sun_rise[valid_days])
        morning_corr = round(morning_corr, 3)
        evening_corr, _ = spearmanr(evening_edge, sun_set[valid_days])
        evening_corr = round(evening_corr, 3)

        thresholds = params.get('generic_thresh')

        if corr_evening_morning_edge >= thresholds.get('edge_corr') and \
                np.maximum(evening_corr, morning_corr) >= thresholds.get('sun_edge_corr'):

            hourglass_pattern['Bool'] = True
            hourglass_pattern['Reason'] = 'Correlation between edges more than {} %'.format(thresholds.get('edge_corr'))
            hourglass_pattern['correlation'] = corr_evening_morning_edge

        elif corr_evening_morning_edge >= 0:

            hourglass_pattern = check_other_corr(morning_edge_, evening_edge_, temperature, temperature_valid,
                                                 valid_days, hourglass_pattern, evening_corr, morning_corr, params)

        else:

            hourglass_pattern['Bool'] = False
            hourglass_pattern['Reason'] = 'None of the correlations are strong enough'

        logger_hourglass.info(' {}. Hourglass pattern detected | {}'.format(hourglass_pattern.get('Reason'),
                                                                            hourglass_pattern.get('Bool')))

    return hourglass_pattern, hourglass_data
