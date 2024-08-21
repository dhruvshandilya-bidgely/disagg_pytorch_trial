"""
Author - Sahana M
Date - 07/06/2021
Get all the features within an instance
"""

# Import python packages
import logging
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.derived_features import get_deviations
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.derived_features import get_tb_probability
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.derived_features import get_max_median_consistency
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.derived_features import get_auc_wh_pot_corr
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import chunk_indexes, seq_arr_idx
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.twh_maths_utils import get_start_end_idx
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.weather_data_features import get_double_dip_score
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.weather_data_features import get_reverse_seasonality_score
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.weather_data_features import get_one_sided_seasonality_score


def exp_moving_average(values, window):
    """
    Numpy implementation of EMA
    Parameters:
        values          (np.ndarray)    : array to perform exponential moving average on
        window          (int)           : Window to perform EMA
    Returns:
        a               (np.ndarray)    : EMA array
    """
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights)[:len(values)]
    a[:window] = a[window]
    return a


def penalty_func(x):
    """
    Penalty function increases exponentially till the value of 3 and later decreases exponentially
    but never becomes 0
    Parameters:
        x               (np.ndarray)    : Array to impose penalty on
    Returns:
        y               (float)         : Penalty
        """

    y = 0.1 + (1/(1+np.exp(-x+3)))*(1-(1/(1+np.exp(-x+3))))*4
    return y


def penalty_score(day_usage, vacation_day):
    """
    This function gives penalty incase of discontinuity in the day to day consumption
    Parameters:
        day_usage               (np.ndarray)    : Contains the every day consumption for the days in the season
        vacation_day            (np.ndarray)    : Boolean array to signify if vacation or not

    Returns:
        score                   (float)         : Final score obtained after penalty
        max_gap_days            (int)           : Maximum continuous discontinuous days found
    """

    # Note - The penalty increases exponentially and is the highest when there are 3 discontinuous days but after that
    # the penalty decreases, this is to make sure that we don't penalise long vacation days/ intended switching off of timed wh

    score = 0
    gap_day = 0
    max_gap_days = 0

    # For each day check for 0 consumption but ignore if it is a vacation day

    for j in range(len(day_usage) - 1):
        if not vacation_day[j]:

            # scenario 1 - day j = 0, j+1 > 0, OR day j, j+1 = 0, and j+1 is a vacation day, then we should make the gap day 0
            # to avoid unnecessary penalty

            if (day_usage[j] == 0 and day_usage[j + 1] > 0) or (day_usage[j] == 0 and day_usage[j + 1] == 0 and vacation_day[j + 1]):
                gap_day += 1
                score += penalty_func(gap_day)
                max_gap_days = max(max_gap_days, gap_day)
                gap_day = 0

            # scenario 2 - day j = 0, j+1 = 0, and j+1 is not a vacation day, then it is a legitimate gap day

            elif day_usage[j] == 0 and day_usage[j + 1] == 0 and not vacation_day[j + 1]:
                gap_day += 1
                score += penalty_func(gap_day)
                max_gap_days = max(max_gap_days, gap_day)

    return score, max_gap_days


def get_continuity_score(seq_arr, auc_not_smoothed, vacation_days_bool, wh_config):
    """
    This function is used to get the continuity score
    Parameters:
        seq_arr                     (np.ndarray)        : Contains info on each season detected
        auc_not_smoothed            (np.ndarray)        : AUC without Exponential moving average
        vacation_days_bool          (np.ndarray)        : Boolean vacation array
        wh_config                   (dict)              : WH configurations dictionary

    Returns:
        continuity_score            (float)             : Continuity score
        seq_arr                     (np.ndarray)        : Contains info on each season detected
    """

    # Note - A penalty is inflicted on discontinuity, this penalty value would change depending on the seasons,
    # Ex: Discontinuity in winter has higher penalty compared to the discontinuity in summer

    zero_col = np.zeros(shape=seq_arr.shape[0])
    seq_arr = np.c_[seq_arr, zero_col, zero_col, zero_col]
    season_penalty = wh_config.get('season_penalty')

    # For each season chunk in the seq arr identify penalty based on continuity

    for i in range(len(seq_arr)):

        # Make sure that the season in consideration has non zero consumption

        if seq_arr[i, seq_arr_idx['auc']] > 0:

            start = int(seq_arr[i, seq_arr_idx['start_day']])
            end = int(seq_arr[i, seq_arr_idx['end_day']])
            day_usage = auc_not_smoothed[start:end]
            vacation_day = vacation_days_bool[start:end]

            # Inflict penalty for discontinuity based on non vacation days

            score, max_gap_days = penalty_score(day_usage, vacation_day)
            seq_arr[i, seq_arr_idx['penalty']] = score / (penalty_func(3) * (seq_arr[i, seq_arr_idx['total_days']] / 3))
            seq_arr[i, seq_arr_idx['gap_days']] = min(max_gap_days / seq_arr[i, seq_arr_idx['total_days']],
                                                      0.4 * seq_arr[i, seq_arr_idx['total_days']])
            seq_arr[i, seq_arr_idx['penalty_config']] = season_penalty[seq_arr[i, seq_arr_idx['s_label']]]

        # For pilots like VSE, empty winter consumption is justified since they turn to gas heating in winters

        else:

            seq_arr[i, seq_arr_idx['penalty']] = 0
            seq_arr[i, seq_arr_idx['gap_days']] = 2
            seq_arr[i, seq_arr_idx['penalty_config']] = season_penalty[seq_arr[i, seq_arr_idx['s_label']]]

    # Get the weighted sum of the penalty weights & penalty + gap days summated score

    weights_sum = np.sum(seq_arr[:, seq_arr_idx['penalty_config']])
    summated_score = ((seq_arr[:, seq_arr_idx['penalty']] + seq_arr[:, seq_arr_idx['gap_days']])/2)

    # Get the continuity score where weights_sum are constants & summated_score are variables

    continuity_score = np.nansum(summated_score*seq_arr[:, seq_arr_idx['penalty_config']])/weights_sum
    continuity_score = 1-continuity_score

    return continuity_score, seq_arr


def get_instance_features(instances, weather_info, debug, wh_config, logger_base):
    """
    Get instance features function is used to identify various features corresponding to timed water heater
    Parameters:
        instances           (np.ndarray)        : Contains all the info about chunks identified
        weather_info        (dict)              : Contains all the weather related information
        debug               (dict)              : Algorithm outputs
        wh_config           (dict)              : WH configurations dictionary
        logger_base         (logger)            : Logger passed

    Returns:
        features            (dict)              : Features dictionary
        debug               (dict)              : Algorithm outputs
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('get_instance_features')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract necessary data

    rows = debug.get('rows')
    cols = debug.get('cols')
    days = wh_config.get('days')
    ema_days = wh_config.get('ema_days')
    start_idx_arr = debug.get('start_idx_arr')
    scored_matrix_rows = start_idx_arr.shape[0]
    in_box_fit_matrix = debug.get('box_fit_matrix')
    vacation_days_bool = debug.get('vacation_days_bool')
    in_masked_data_matrix = debug.get('masked_cleaned_data')

    # Initialise year round features

    duration_line = np.full(shape=rows, fill_value=0)
    amplitude_line = np.full(shape=rows, fill_value=0)
    amplitude_line_raw = np.full(shape=rows, fill_value=0)
    consistency_arr = np.full(shape=cols, fill_value=0.0)
    area_under_curve_line = np.full(shape=rows, fill_value=0)
    area_under_curve_line_raw = np.full(shape=rows, fill_value=0)
    box_fit_matrix = np.full_like(in_box_fit_matrix, fill_value=0.0)
    masked_data_matrix = np.full_like(in_masked_data_matrix, fill_value=0.0)

    max_consistency_score = 0
    total_consistent_check_days = 0

    # For each chunk extract its base features from the raw data & box fit data

    for i in range(instances.shape[0]):
        s_row = int(instances[i, chunk_indexes['window_start']])
        e_row = int(min((s_row + days), rows))
        s_col = int(instances[i, chunk_indexes['chunk_start']])
        e_col = int(instances[i, chunk_indexes['chunk_end']])

        # Extract the data from raw & box fit matrix corresponding to the chunk

        temp_1 = in_box_fit_matrix[s_row:e_row, s_col:e_col]
        temp_2 = in_masked_data_matrix[s_row:e_row, s_col:e_col]
        temp_1_raw = in_masked_data_matrix[s_row:e_row, s_col:e_col]

        # Assign the extracted data into a 2D matrix

        box_fit_matrix[s_row:e_row, s_col:e_col] = temp_1
        masked_data_matrix[s_row:e_row, s_col:e_col] = temp_2

        # Identify the area under the curve and amplitude

        temp_1_aur = np.sum(temp_1, axis=1)
        temp_1_aur_raw = np.sum(temp_1_raw, axis=1)
        temp_1_duration = np.sum(temp_1 > 0, axis=1)
        temp_1_amplitude = np.percentile(temp_1, q=90)
        temp_2_amplitude = np.percentile(temp_2, q=90)

        # In case of more than 1 chunks in a single 30 day division, make sure to add the rest chunks info

        duration_line[s_row:e_row] = duration_line[s_row:e_row] + temp_1_duration
        amplitude_line[s_row:e_row] = amplitude_line[s_row:e_row] + temp_1_amplitude
        amplitude_line_raw[s_row:e_row] = amplitude_line_raw[s_row:e_row] + temp_2_amplitude
        area_under_curve_line[s_row:e_row] = area_under_curve_line[s_row:e_row] + temp_1_aur
        area_under_curve_line_raw[s_row:e_row] = area_under_curve_line_raw[s_row:e_row] + temp_1_aur_raw

        # Calculate consistency features

        total_consistent_check_days += e_row - s_row
        consistency_arr[s_col:e_col] = \
            consistency_arr[s_col:e_col] + np.sum((masked_data_matrix[s_row:e_row, s_col:e_col] > 0).astype(int), axis=0)
        temp_3 = np.sum((masked_data_matrix[s_row:e_row, s_col:e_col] > 0).astype(int), axis=0)
        max_consistency_score += np.max(temp_3/(e_row - s_row))

    # Do exponential moving average on the extracted features

    auc_not_smoothed = deepcopy(area_under_curve_line)
    area_under_curve_line = exp_moving_average(area_under_curve_line, ema_days)
    duration_line = exp_moving_average(duration_line, ema_days)
    amplitude_line = exp_moving_average(amplitude_line, ema_days)
    amplitude_line_raw = exp_moving_average(amplitude_line_raw, ema_days)

    # Final consistency score

    consistency_arr /= total_consistent_check_days
    max_consistency_score = np.round(max_consistency_score/(instances.shape[0]), 2)

    # Get the start and end time of the instance

    wh_timings = np.sum(box_fit_matrix, axis=0)
    wh_timings = wh_timings > 0

    if np.sum(wh_timings.astype(int)) > 0:
        idx_diff = np.diff(np.r_[0, wh_timings.astype(int), 0])
        start_idx, end_idx = get_start_end_idx(idx_diff)
        start_time = int(start_idx[0])
        end_time = int(end_idx[-1])
    else:
        start_time = -1
        end_time = -1

    # For plotting purpose create a matrix which has chunks representing their scores

    plot_scored_matrix = np.full(shape=(scored_matrix_rows, cols), fill_value=0.0)
    for i in range(instances.shape[0]):
        start = int(instances[i, chunk_indexes['chunk_start']])
        end = int(instances[i, chunk_indexes['chunk_end']])
        window = int(instances[i, chunk_indexes['overall_index']])
        plot_scored_matrix[window, start:end] = instances[i, chunk_indexes['chunk_score']]

    # Get all the weather related info

    wh_potential = weather_info.get('wh_potential')
    season_detection_dict = weather_info.get('season_detection_dict')

    # Add consumption info (AUC, amplitude, duration) for each season in the seq_arr

    seq_arr = season_detection_dict.get('seq_arr')
    zero_col = np.zeros(seq_arr.shape[0])
    seq_arr = np.c_[seq_arr, zero_col, zero_col, zero_col]
    for i in range(len(seq_arr)):
        start = int(seq_arr[i, 1])
        end = int(seq_arr[i, 2] + 1)
        sn_auc = np.median(area_under_curve_line[start: end])
        sn_dur = np.median(duration_line[start: end])
        sn_amp = np.median(amplitude_line[start: end])
        seq_arr[i, 4] = sn_auc
        seq_arr[i, 5] = sn_dur
        seq_arr[i, 6] = sn_amp

    # Check for reverse seasonality

    reverse_seasonality_score = get_reverse_seasonality_score(seq_arr, wh_config)
    logger.info('Reverse seasonality score | {}'.format(reverse_seasonality_score))

    # check for one sided seasonality

    one_sided_seasonality_score = get_one_sided_seasonality_score(seq_arr, wh_config)
    logger.info('One sided seasonality score | {}'.format(one_sided_seasonality_score))

    # Check for double dip

    double_dip_score = get_double_dip_score(seq_arr, wh_config)
    logger.info('Double dip score | {}'.format(double_dip_score))

    # Continuity score

    continuity_score, seq_arr = get_continuity_score(seq_arr, auc_not_smoothed, vacation_days_bool, wh_config)
    logger.info('Continuity score | {}'.format(continuity_score))

    # get AUC & WH potential correlation

    auc_wh_pot_corr = get_auc_wh_pot_corr(wh_potential, area_under_curve_line, vacation_days_bool)
    logger.info('AUC WH potential score | {}'.format(auc_wh_pot_corr))

    # Get AUC, Duration & Amplitude deviations

    auc_std = get_deviations(area_under_curve_line)
    dur_std = get_deviations(duration_line)
    amp_std = get_deviations(amplitude_line)

    # Get the max median consistency value

    max_median_consistency = get_max_median_consistency(max_consistency_score, consistency_arr, start_time, end_time)
    logger.info('Max median consistency score | {}'.format(max_median_consistency))

    # Get the final Time band probability

    final_tb_prob = get_tb_probability(start_time, end_time, wh_config)
    logger.info('Final Time band probability | {}'.format(final_tb_prob))

    logger.info('Extracted all the Weather & Derived features for Timed WH |')

    features = {
        'auc_std': auc_std,
        'dur_std': dur_std,
        'amp_std': amp_std,
        'end_time': end_time,
        'start_time': start_time,
        'final_tb_prob': final_tb_prob,
        'duration_line': duration_line,
        'box_fit_matrix': box_fit_matrix,
        'amplitude_line': amplitude_line,
        'auc_wh_pot_corr': auc_wh_pot_corr,
        'double_dip_score': double_dip_score,
        'continuity_score': continuity_score,
        'masked_data_matrix': masked_data_matrix,
        'max_consistency': max_consistency_score,
        'amplitude_line_raw': amplitude_line_raw,
        'plot_scored_matrix': plot_scored_matrix,
        'area_under_curve_line': area_under_curve_line,
        'max_median_consistency': max_median_consistency,
        'reverse_seasonality_score': reverse_seasonality_score,
        'one_sided_seasonality_score': one_sided_seasonality_score,
        'season_label': season_detection_dict.get('s_label'),
        'wh_potential': wh_potential
    }

    return features, debug
