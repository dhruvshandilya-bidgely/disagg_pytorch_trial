"""
Author - Mayank Sharan
Date - 26/11/2019
detect seasonal tries to detect seasonal consumption patterns in the data
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile


def detect_seasonal(data_fil, num_periods, pd_mult, config, logger_pass):

    """
    Parameters:
        data_fil            (np.ndarray)        : Day wise 2d matrix containing filtered data
        num_periods         (int)               : Number of sampling points in 1 day
        pd_mult             (int)               : Number of data points in 1 hour
        config              (dict)              : Dictionary containing all

    Returns:
        season_flag         (np.ndarray)        : Array containing values for true false of time slices being seasonal
        perc_season         (np.ndarray)        : Array containing values for percentage of seasonal load detected
        smoothing_noise_bound(float)            : The upper bound for smoothing noise to be removed
        debug               (dict)              : Contains all variables needed to debug the lighting algorithm
    """

    # Initialise data

    bin_size = config.get('HISTOGRAM_BIN_SIZE')

    data_season_detection = data_fil
    daily_cons = np.nansum(data_season_detection, axis=1)
    data_season_detection = data_season_detection[daily_cons > config.get('ZERO_DAY_CONSUMPTION'), :]

    num_non_zero_days = data_season_detection.shape[0]

    if num_non_zero_days == 0:
        num_non_zero_days = -1

    # Get smoothing noise bound

    kink_by_period = np.zeros(shape=[num_periods, 1])
    bin_data_total = np.ceil(data_season_detection / bin_size)

    # noinspection PyPep8
    for i in range(num_periods):

        binned_data = bin_data_total[:, i]
        data_values, _, accumidx = np.unique(binned_data, return_index=True, return_inverse=True)

        data_count = np.bincount(accumidx)

        calc_kink(i, data_count, data_values, bin_size, config, pd_mult, kink_by_period)

    smoothing_noise_bound = superfast_matlab_percentile(kink_by_period, config.get('SMOOTHING_BOUND_PERCENTILE'))[0]

    if smoothing_noise_bound > config.get('SMOOTHING_NOISE_UPPER_BOUND') / pd_mult:
        smoothing_noise_bound = config.get('SMOOTHING_NOISE_UPPER_BOUND') / pd_mult

    # Detect seasonal hours

    data_season_detection[data_season_detection < smoothing_noise_bound] = 0
    data_season_detection[data_season_detection > 0] = 1

    # Explanation for the following logic:
    # For each time-slice, get column of data with 0s and 1s. Then, calculate
    # length of continuous sequences of zero. For instance, say the series is 10010
    # Step 1: append 1 on both sides 1100101
    # Step 2: take diff on this 0 -1 0 1 -1 1
    # Step 3: get start indices (series < 0) 2 and 5 here
    # Step 4: get end indices + 1 (series > 0) 4 and 6 here
    # subtract these vectors to get the durations of consecutive zeros = 2
    # and 1 here.

    is_lighting = copy.deepcopy(data_season_detection)
    is_lighting[np.isnan(is_lighting)] = 0
    padding = np.ones(shape=(1, data_season_detection.shape[1]))
    ex_series_total = np.diff(np.r_[padding, is_lighting, padding], axis=0)

    total_zero_days = np.zeros(shape=(num_periods, 1))
    dur_max = np.zeros(shape=[num_periods, 1])

    for i in range(num_periods):

        ex_series = ex_series_total[:, i]
        start_idx = np.nonzero((ex_series < 0).astype(int))[0]
        end_idx = np.nonzero((ex_series > 0).astype(int))[0]

        duration = end_idx - start_idx

        if len(duration) > 0:
            total_zero_days[i, :] = np.sum(duration)
            dur_max[i, :] = np.max(duration)

    season_flag = dur_max > config.get('NUM_ZEROS_FOR_SEASONAL')
    season_perc = (num_non_zero_days - total_zero_days) * 100 / num_non_zero_days

    # Protect typical lighting hours from being removed as seasonal

    typical_lighting_hours = config.get('TYPICAL_LIGHTING_HOURS')
    typical_lighting = []

    for i in range(typical_lighting_hours.shape[0]):
        typical_lighting = np.r_[typical_lighting,
                                 typical_lighting_hours[i, 0] * pd_mult: (typical_lighting_hours[i, 1]) * pd_mult + 1]
    typical_lighting -= 1

    season_flag[typical_lighting.astype(int), :] = 0

    # %% Calculate percentage of season

    debug = {
        'season': {
            'catch': '|',
            'noise_cap_by_time': None,
            'smoothing_noise_bound': None,
            'season_percentage_by_time': None,
            'percentage_season': None,
            'consecutive_zeros_by_time': None,
            'seasonal_hours': None,
            'overall_season_percentage': None,
        },
        'data': None,
        'results': None,
        'lighting': None,
    }

    overall_season = superfast_matlab_percentile(season_perc[season_perc > 0], config.get('OVERALL_SEASON_PERCENTILE'))

    if sum(season_flag > 0):
        perc_season = superfast_matlab_percentile(season_perc[season_flag], config.get('SEASON_PERCENTAGE_PERCENTILE'))
    else:
        debug['season']['catch ']= '| No seasonal rejection %s' % debug['season']['catch']
        perc_season = config.get('DEFAULT_SEASON_PERCENTAGE')

    if overall_season - perc_season > config.get('SEASON_DISCREPANCY_THRESHOLD') or overall_season < perc_season:
        debug['season']['catch ']= '| Overall Season discrepancy switching %f to %f %s' % (
            perc_season, overall_season, debug['season']['catch'])
        perc_season = overall_season

    if overall_season > config.get('SEASON_PERCENTAGE_BOUND'):
        debug['season']['catch ']= '| Percentage Overall Season > 95 switching %f to 95 %s' % (
            perc_season, debug['season']['catch'])
        perc_season = config.get('SEASON_PERCENTAGE_BOUND')

    # %% Populate the debug object

    debug['season']['noise_cap_by_time'] = kink_by_period
    debug['season']['smoothing_noise_bound'] = smoothing_noise_bound
    debug['season']['season_percentage_by_time'] = season_perc
    debug['season']['percentage_season'] = perc_season
    debug['season']['consecutive_zeros_by_time'] = dur_max
    debug['season']['seasonal_hours'] = season_flag.astype(int)
    debug['season']['overall_season_percentage'] = overall_season

    return season_flag, perc_season, smoothing_noise_bound, debug


def calc_kink(i, data_count, data_values, bin_size, config, pd_mult, kink_by_period):

    """ Complexity fix 2 Computes the kink values for each time slice"""

    temp_j = 0

    for j in range(len(data_count) - 1):
        if (data_count[j] > data_count[j + 1] and
                data_values[j] * bin_size > config.get('SMOOTHING_NOISE_LOWER_BOUND') / pd_mult):
            temp_j = j
            break

    if temp_j < len(data_count) - 3:
        kink_by_period[i, :] = data_values[temp_j] * bin_size
    else:
        if len(data_values) != 0:
            kink_by_period[i, :] = data_values[0] * bin_size
        else:
            kink_by_period[i, :] = 0

