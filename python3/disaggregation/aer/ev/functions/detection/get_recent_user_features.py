"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to get recent detection features for EV detection
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.rolling_function import rolling_function

from python3.disaggregation.aer.ev.functions.detection.get_user_features import monthly_consistency_factor

from python3.disaggregation.aer.ev.functions.detection.get_user_features import remove_boxes_before_ev

from python3.disaggregation.aer.ev.functions.detection.get_user_features import prorate_weekly_count


def get_recent_user_features(debug, ev_config, logger_base, input_data):
    """
    Function to get features for EV detection in recent months

        Parameters:
            debug                     (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
            ev_config                  (dict)              : Module config dict
            logger_base               (logger)            : logger pass
            input_data                (np.ndarray)        : Input 21-column data

        Returns:
            debug                     (object)            : Object containing all important data/values as well as HSM

    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_recent_user_features')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the params from the config

    region = ev_config.get('region')

    sampling_rate = ev_config.get('sampling_rate')

    night_hours = ev_config.get('night_hours')

    weekend = ev_config.get('weekend_days')

    user_features = {
        'uuid': ev_config['uuid'],
        'region': ev_config['region'],
        'pilot_id': ev_config['pilot_id'],
        'sampling_rate': ev_config['sampling_rate']
    }

    columns_dict = ev_config['box_features_dict']

    # Find the serial number of final iteration of dynamic box fitting

    updated_box_keys = [key.split('_')[-1] for key in list(debug.keys()) if 'updated_box_data' in key]

    keys_int = list(map(int, updated_box_keys))

    max_key = str(np.max(keys_int))

    factor = debug.get('factor')

    # Get the final box data and box features

    box_data = debug.get('updated_box_data_' + max_key)

    box_features = debug.get('updated_box_features_' + max_key)

    recent_days = ev_config.get('recent_days')

    recent_days_points_count = recent_days * Cgbdisagg.HRS_IN_DAY * factor

    # Check if sufficient data available

    if recent_days_points_count > box_data.shape[0]:
        debug['user_recent_features'] = None

        # Updated features box data

        debug['recent_features_box_data'] = deepcopy(box_data)

        # Index for the recent data start

        debug['recent_data_cutoff_idx'] = 0

        return debug

    recent_days_cutoff_idx = box_data.shape[0] - int(recent_days_points_count)

    input_data_copy = deepcopy(input_data)
    box_data = box_data[recent_days_cutoff_idx:, :]
    input_data_copy = input_data_copy[recent_days_cutoff_idx:, :]

    start_index_column = columns_dict['start_index_column']
    end_index_column = columns_dict['end_index_column']

    box_features = box_features[box_features[:, start_index_column] >= recent_days_cutoff_idx]

    box_features[:, [start_index_column, end_index_column]] -= recent_days_cutoff_idx

    # Saving the index of the recent data start

    debug['recent_data_cutoff_idx'] = recent_days_cutoff_idx

    max_energy = debug.get('amplitude_' + max_key) / factor

    # Getting first EV months and removing boxes before that
    if len(box_features) > 0:
        box_data, box_features, input_data_copy = remove_boxes_before_ev(box_data, box_features, input_data_copy, debug,
                                                                         ev_config)

    debug['recent_detection_box_data'] = box_data
    debug['recent_detection_box_features'] = box_features

    # Removing extra columns not required here
    box_data = box_data[:, :Cgbdisagg.INPUT_DIMENSION]

    n_boxes = box_features.shape[0]
    n_data_points = box_data.shape[0]

    # Create the user features

    # Boxes energy / duration features

    duration_col = columns_dict['boxes_duration_column']
    boxes_areas_column = columns_dict['boxes_areas_column']

    start_index_column = columns_dict['start_index_column']

    boxes_median_energy = columns_dict['boxes_median_energy']
    boxes_energy_std_column = columns_dict['boxes_energy_std_column']

    average_duration = np.mean(box_features[:, columns_dict['boxes_duration_column']])

    duration_variation = np.mean(
        np.abs(box_features[:, duration_col] - np.mean(box_features[:, duration_col]))) / average_duration

    boxes_energy_variation = np.std(box_features[:, boxes_median_energy]) / max_energy

    within_boxes_variation = np.mean(box_features[:, boxes_energy_std_column]) / max_energy

    energy_per_hour = np.sum(box_features[:, boxes_areas_column]) / np.sum(
        box_features[:, columns_dict['boxes_duration_column']])

    energy_per_charge = np.mean(box_features[:, columns_dict['boxes_areas_column']])

    # Time of day features

    night_boolean = ((box_data[:, Cgbdisagg.INPUT_HOD_IDX] >= night_hours[0]) |
                     (box_data[:, Cgbdisagg.INPUT_HOD_IDX] <= night_hours[1])).astype(int)

    box_data = np.c_[box_data, night_boolean]

    # Boxes with mid-point during night hours

    box_features = np.c_[
        box_features, night_boolean[
            ((box_features[:, 0].astype(int) + box_features[:, 1].astype(int)) / 2).astype(int)]]

    night_count_fraction = np.sum(box_features[:, columns_dict['night_boolean']]) / n_boxes

    day_count_fraction = 1 - night_count_fraction

    start_tou_arr = box_data[box_features[:, start_index_column].astype(int), Cgbdisagg.INPUT_HOD_IDX]

    # Getting most common box start hour of the day

    if len(start_tou_arr) > 0:
        mode_tou = max(set(start_tou_arr), key=list(start_tou_arr).count)
    else:
        mode_tou = 0

    tou_variation = np.mean(get_tou_diff(start_tou_arr, mode_tou)) / Cgbdisagg.HRS_IN_DAY

    # Weekday / weekend based features

    weekend_boolean = ((box_data[:, Cgbdisagg.INPUT_DOW_IDX] == weekend[0]) |
                       (box_data[:, Cgbdisagg.INPUT_HOD_IDX] == weekend[1])).astype(int)

    box_data = np.c_[box_data, weekend_boolean]

    box_features = np.c_[box_features, weekend_boolean[box_features[:, 0].astype(int)]]

    weekend_boxes = box_features[box_features[:, columns_dict['weekend_boolean']] == 1]
    weekday_boxes = box_features[box_features[:, columns_dict['weekend_boolean']] == 0]

    weekend_night_fraction = np.sum(weekend_boxes[:, columns_dict['night_boolean']]) / weekend_boxes.shape[0]
    weekday_night_fraction = np.sum(weekday_boxes[:, columns_dict['night_boolean']]) / weekday_boxes.shape[0]

    weekend_day_fraction = 1 - weekend_night_fraction
    weekday_day_fraction = 1 - weekday_night_fraction

    # Seasonal features

    cutoff_temperature = ev_config['season_cutoff_temp']

    temp_arr = box_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]
    nan_temp = np.isnan(temp_arr)

    wtr_boolean = (temp_arr <= cutoff_temperature).astype(float)
    wtr_boolean[nan_temp] = np.nan

    wtr_idx = np.where(wtr_boolean == 1)[0]

    smr_boolean = (temp_arr > cutoff_temperature).astype(float)
    smr_boolean[nan_temp] = np.nan

    smr_idx = np.where(smr_boolean == 1)[0]

    box_data = np.c_[box_data, wtr_boolean, smr_boolean]

    box_features = np.c_[box_features, smr_boolean[box_features[:, start_index_column].astype(int)] + 1]
    box_features[np.isnan(box_features[:, columns_dict['season_id']]), columns_dict['season_id']] = 0

    # Count of boxes in different seasons

    wtr_boxes = box_features[box_features[:, columns_dict['season_id']] == ev_config['season_ids']['wtr']]
    smr_boxes = box_features[box_features[:, columns_dict['season_id']] == ev_config['season_ids']['smr']]

    wtr_count = wtr_boxes.shape[0]
    smr_count = smr_boxes.shape[0]

    # Updated number of boxes
    n_valid_temp_data_points = np.count_nonzero(~nan_temp)
    n_valid_temp_boxes = np.count_nonzero(box_features[:, columns_dict['season_id']] > 0)

    min_seasonal_boxes_count = ev_config['min_seasonal_boxes_count']

    # Fraction of seasons in input data

    wtr_count_fraction, smr_count_fraction, wtr_energy_fraction, smr_energy_fraction, seasonal_count_fraction_diff, \
    seasonal_energy_fraction_diff = get_seasonal_boxes_features(wtr_idx, smr_idx, wtr_count, smr_count,
                                                                n_valid_temp_boxes, n_valid_temp_data_points, box_data,
                                                                ev_config)

    # seasonal AUC diff

    if (wtr_count > min_seasonal_boxes_count) and (smr_count > min_seasonal_boxes_count):
        wtr_auc = np.mean(wtr_boxes[:, columns_dict['boxes_areas_column']])
        smr_auc = np.mean(smr_boxes[:, columns_dict['boxes_areas_column']])
        seasonal_auc_frac_diff = np.abs(wtr_auc - smr_auc) / max(wtr_auc, smr_auc)
    else:
        seasonal_auc_frac_diff = 0

    # seasonal amp diff

    if (wtr_count > min_seasonal_boxes_count) and (smr_count > min_seasonal_boxes_count):
        wtr_amp = np.mean(wtr_boxes[:, columns_dict['boxes_energy_per_point_column']])
        smr_amp = np.mean(smr_boxes[:, columns_dict['boxes_energy_per_point_column']])
        seasonal_amp_frac_diff = np.abs(wtr_amp - smr_amp) / max(wtr_amp, smr_amp)
    else:
        seasonal_amp_frac_diff = 0

    winter_id = ev_config.get('season_ids').get('wtr')
    summer_id = ev_config.get('season_ids').get('smr')

    # seasonal TOU diff
    if (wtr_count > min_seasonal_boxes_count) and (smr_count > min_seasonal_boxes_count):
        unq_smr_tou, count_smr_tou = np.unique(start_tou_arr[box_features[:, columns_dict['season_id']] == summer_id],
                                               return_counts=True)
        top_smr_tou = unq_smr_tou[count_smr_tou.argsort()[-2:]]

        unq_wtr_tou, count_wtr_tou = np.unique(start_tou_arr[box_features[:, columns_dict['season_id']] == winter_id],
                                               return_counts=True)
        top_wtr_tou = unq_wtr_tou[count_wtr_tou.argsort()[-2:]]

        combination_arr = np.array(np.meshgrid(top_smr_tou, top_wtr_tou)).T.reshape(-1, 2)
        seasonal_tou_diff = np.min(get_tou_diff(combination_arr[:, 0], combination_arr[:, 1])) / Cgbdisagg.HRS_IN_DAY
    else:
        seasonal_tou_diff = 0

    # Count charges per week

    number_of_weeks = n_data_points / (Cgbdisagg.DAYS_IN_WEEK * Cgbdisagg.HRS_IN_DAY * factor)

    weekly_count = n_boxes / number_of_weeks

    # Normalizing the weekly count
    weekly_count_pro = prorate_weekly_count(weekly_count, ev_config)

    # Correlation of boxes energy with temperature

    energy_idx = (box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0)

    box_energy = box_data[energy_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    box_temperature = np.abs(box_data[energy_idx, Cgbdisagg.INPUT_TEMPERATURE_IDX] - cutoff_temperature)

    correlation_factor = np.abs(
        np.corrcoef(box_energy[~np.isnan(box_temperature)], box_temperature[~np.isnan(box_temperature)])[0, 1])

    # Consistency factor

    day_consistency_score = consistency_factor(box_data, ev_config)

    monthly_consistency_score, monthly_energy_variation = monthly_consistency_factor(box_data, ev_config)

    n_boxes_pro, wtr_count_pro, smr_count_pro = prorate_count_features(wtr_count, smr_count, debug, ev_config)

    # Residual consumption features
    residual_consumption = input_data_copy[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    rolling_residual_window = ev_config.get('minimum_duration', {}).get(sampling_rate)
    rolling_residual = rolling_function(residual_consumption, window=rolling_residual_window, out_metric='min')
    residual_max_amp = np.percentile(rolling_residual, ev_config.get('rolling_percentile'))
    residual_max_amp = residual_max_amp * factor

    # Region categorical feature

    user_features['region_NA'] = int(region == 'NA')
    user_features['region_EU'] = int(region == 'EU')

    # Sampling rate categorical features

    user_features['sampling_900'] = int(sampling_rate == Cgbdisagg.SEC_IN_15_MIN)
    user_features['sampling_1800'] = int(sampling_rate == Cgbdisagg.SEC_IN_30_MIN)
    user_features['sampling_3600'] = int(sampling_rate == Cgbdisagg.SEC_IN_HOUR)

    ev_residual_amp_ratio = energy_per_hour / (residual_max_amp + 0.01)

    # Adding new features to user features dictionary

    user_features['total_count'] = n_boxes
    user_features['wtr_count'] = wtr_count
    user_features['smr_count'] = smr_count
    user_features['total_count_pro'] = n_boxes_pro
    user_features['wtr_count_pro'] = wtr_count_pro
    user_features['smr_count_pro'] = smr_count_pro
    user_features['wtr_count_fraction'] = np.round(wtr_count_fraction, 3)
    user_features['smr_count_fraction'] = np.round(smr_count_fraction, 3)
    user_features['seasonal_count_fraction_diff'] = np.round(seasonal_count_fraction_diff, 3)
    user_features['wtr_energy_fraction'] = np.round(wtr_energy_fraction, 3)
    user_features['smr_energy_fraction'] = np.round(smr_energy_fraction, 3)
    user_features['seasonal_energy_fraction_diff'] = np.round(seasonal_energy_fraction_diff, 3)
    user_features['average_duration'] = np.round(average_duration, 3)
    user_features['duration_variation'] = np.round(duration_variation, 3)
    user_features['boxes_energy_variation'] = np.round(boxes_energy_variation, 3)
    user_features['within_boxes_variation'] = np.round(within_boxes_variation, 3)
    user_features['energy_per_hour'] = np.round(energy_per_hour, 3)
    user_features['energy_per_charge'] = np.round(energy_per_charge, 3)
    user_features['night_count_fraction'] = np.round(night_count_fraction, 3)
    user_features['day_count_fraction'] = np.round(day_count_fraction, 3)
    user_features['weekend_night_fraction'] = np.round(weekend_night_fraction, 3)
    user_features['weekday_night_fraction'] = np.round(weekday_night_fraction, 3)
    user_features['weekend_day_fraction'] = np.round(weekend_day_fraction, 3)
    user_features['weekday_day_fraction'] = np.round(weekday_day_fraction, 3)
    user_features['weekly_count'] = np.round(weekly_count, 3)
    user_features['weekly_count_pro'] = np.round(weekly_count_pro, 3)
    user_features['day_consistency_score'] = np.round(day_consistency_score, 3)
    user_features['monthly_consistency_score'] = np.round(monthly_consistency_score, 3)
    user_features['monthly_energy_variation'] = np.round(monthly_energy_variation, 3)
    user_features['correlation_factor'] = np.round(correlation_factor, 3)
    user_features['tou_variation'] = np.round(tou_variation, 3)
    user_features['seasonal_auc_frac_diff'] = np.round(seasonal_auc_frac_diff, 3)
    user_features['seasonal_amp_frac_diff'] = np.round(seasonal_amp_frac_diff, 3)
    user_features['seasonal_tou_diff'] = np.round(seasonal_tou_diff, 3)
    user_features['ev_residual_amp_ratio'] = np.round(ev_residual_amp_ratio, 3)
    # Making reverse features

    # Taking reverse of the features that negatively impact presence of EV

    user_features['seasonal_count_fraction_diff'] = 1 / user_features['seasonal_count_fraction_diff']
    user_features['seasonal_energy_fraction_diff'] = 1 / user_features['seasonal_energy_fraction_diff']
    user_features['duration_variation'] = 1 / user_features['duration_variation']
    user_features['boxes_energy_variation'] = 1 / user_features['boxes_energy_variation']
    user_features['within_boxes_variation'] = 1 / user_features['within_boxes_variation']
    user_features['day_consistency_score'] = 1 / user_features['day_consistency_score']
    user_features['monthly_consistency_score'] = 1 / user_features['monthly_consistency_score']
    user_features['monthly_energy_variation'] = 1 / user_features['monthly_energy_variation']
    user_features['correlation_factor'] = 1 / user_features['correlation_factor']
    user_features['tou_variation'] = 1 / user_features['tou_variation']
    user_features['seasonal_auc_frac_diff'] = 1 / user_features['seasonal_auc_frac_diff']
    user_features['seasonal_amp_frac_diff'] = 1 / user_features['seasonal_amp_frac_diff']
    user_features['seasonal_tou_diff'] = 1 / user_features['seasonal_tou_diff']

    # Adding user features dictionary to debug object

    debug['user_recent_features'] = user_features

    # Updated features box data

    debug['recent_features_box_data'] = deepcopy(box_data)

    logger.info('Recent EV features calculation completed | ')

    return debug


def consistency_factor(input_box_data, ev_config):
    """
    Function to calculate consistency of boxes

        Parameters:
            input_box_data            (np.ndarray)        : Input box data
            ev_config                  (dict)              : Module config dict

        Returns:
            consistency_score         (float)              : Calculated consistency score

    """
    # Extract required params from config

    window_size = ev_config['detection']['consistency_window']

    box_data = deepcopy(input_box_data)

    energy_idx = (box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0)

    # Get the daily peaks count

    unq_days, days_idx = np.unique(box_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
    daily_peaks_count = np.bincount(days_idx, energy_idx)

    # Select days with non-zero peaks

    peak_days_idx = np.where(daily_peaks_count > 0)[0]

    if len(peak_days_idx) > 0:

        peak_days_boolean = (daily_peaks_count[peak_days_idx[0]:peak_days_idx[-1]] > 0).astype(int)

        window_count = rolling_function(peak_days_boolean, window_size, 'sum')

        consistency_score = np.std(window_count)
    else:
        consistency_score = 0

    return consistency_score


def prorate_count_features(wtr_count, smr_count, debug, ev_config):
    """
    Function to prorate count based features

        Parameters:
            wtr_count                  (int)              : Num winter boxes
            smr_count                  (int)              : Num summer boxes
            debug                      (dict)             : debug object
            ev_config                   (dict)             : Internal configs

        Returns:
            counts                     (np.array)         : Calculated counted scores

    """
    missing_data_info = debug.get('missing_data_info')

    wtr_missing_factor = missing_data_info['wtr_factor']
    smr_missing_factor = missing_data_info['smr_factor']

    # Scale up the seasonal count using the missing data factor

    wtr_count = int(wtr_count * wtr_missing_factor)
    smr_count = int(smr_count * smr_missing_factor)

    n_boxes = wtr_count + smr_count

    # Extract the config params
    boundaries = ev_config['detection']['boundaries']

    n_bounds = len(boundaries)

    min_bound = boundaries[1]
    max_bound = boundaries[-1]

    counts = np.array([n_boxes, wtr_count, smr_count])

    min_target_idx = np.where(counts <= min_bound)[0]

    counts[min_target_idx] = counts[min_target_idx] / min_bound

    max_target_idx = np.where(counts > max_bound)[0]

    counts[max_target_idx] = n_bounds - 1

    for i in range(1, n_bounds - 1):
        target_idx = np.where((counts > boundaries[i]) & (counts <= boundaries[i + 1]))[0]
        counts[target_idx] = i

    return counts


def get_tou_diff(time_arr_1, time_arr_2):
    """
    Function to calculate difference between circular times of use

        Parameters:
            time_arr_1                  (np.array)              : time of usage array
            time_arr_2                  (np.array)              : time of usage array

        Returns:
            tou_diff                    (np.array)              : array containing time difference
    """
    tou_diff = np.abs(time_arr_1 - time_arr_2)
    tou_diff[tou_diff > Cgbdisagg.HRS_IN_DAY / 2] = Cgbdisagg.HRS_IN_DAY - tou_diff[tou_diff > Cgbdisagg.HRS_IN_DAY / 2]
    tou_diff[np.isnan(tou_diff)] = 0
    return tou_diff


def get_seasonal_boxes_features(wtr_idx, smr_idx, wtr_count, smr_count, n_boxes, n_data_points, box_data, ev_config):
    """
    Function to get seasonal boxes features

        Parameters:
            wtr_idx                   (np.array)          : Indices of winter boxes
            smr_idx                   (np.array)          : Indices of summer boxes
            wtr_count                 (int)               : Number of boxes in winter season
            smr_count                 (int)               : Number of boxes in summer season
            n_boxes                   (int)               : Number of boxes detected
            n_data_points             (int)               : Number of data points in box_data
            box_data                  (np.ndarray)        : EV box data
            ev_config                  (dict)              : Module config dict

        Returns:
            wtr_count_fraction        (float)              : Fraction of boxes lying in winter season
            smr_count_fraction        (float)              : Fraction of boxes lying in summer season
            wtr_energy_fraction       (float)              : Fraction of ev energy lying in winter season
            smr_energy_fraction       (float)              : Fraction of ev energy lying in summer season
            seasonal_count_fraction_diff        (float)    : Difference between summer and winter count fraction
            seasonal_energy_fraction_diff       (float)    : Difference between summer and winter energy fraction
    """

    season_count_frac_thresh = ev_config['season_count_frac_thresh_recent']

    # Fraction of seasons in input data

    if n_data_points == 0:
        wtr_proportion = 0
    else:
        wtr_proportion = len(wtr_idx) / n_data_points
    smr_proportion = 1 - wtr_proportion

    if wtr_count > 0:
        wtr_count_fraction = wtr_count / (n_boxes * wtr_proportion)
    else:
        wtr_count_fraction = 0

    if smr_count > 0:
        smr_count_fraction = smr_count / (n_boxes * smr_proportion)
    else:
        smr_count_fraction = 0

    # Normalize the count fraction

    sum_fractions = wtr_count_fraction + smr_count_fraction

    if sum_fractions > 0:
        wtr_count_fraction /= sum_fractions
        smr_count_fraction /= sum_fractions
    else:
        wtr_count_fraction = 0
        smr_count_fraction = 0

    if (wtr_count_fraction >= season_count_frac_thresh) or (smr_count_fraction >= season_count_frac_thresh):
        seasonal_count_fraction_diff = 0
    else:
        seasonal_count_fraction_diff = np.abs(wtr_count_fraction - smr_count_fraction)

    # Proportion of energy in various seasons

    wtr_energy = np.sum(box_data[wtr_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    smr_energy = np.sum(box_data[smr_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    total_energy = wtr_energy + smr_energy

    if wtr_energy > 0:
        wtr_energy_fraction = wtr_energy / (total_energy * wtr_proportion)
    else:
        wtr_energy_fraction = 0

    if smr_energy > 0:
        smr_energy_fraction = smr_energy / (total_energy * smr_proportion)
    else:
        smr_energy_fraction = 0

    # Normalize the energy fraction

    sum_energy_fractions = wtr_energy_fraction + smr_energy_fraction

    if sum_energy_fractions > 0:
        wtr_energy_fraction /= sum_energy_fractions
        smr_energy_fraction /= sum_energy_fractions
    else:
        wtr_energy_fraction = 0
        smr_energy_fraction = 0

    seasonal_energy_fraction_diff = get_seasonal_energy_fraction_diff(wtr_energy_fraction, smr_energy_fraction,
                                                                      season_count_frac_thresh)

    return wtr_count_fraction, smr_count_fraction, wtr_energy_fraction, smr_energy_fraction, \
           seasonal_count_fraction_diff, seasonal_energy_fraction_diff


def get_seasonal_energy_fraction_diff(wtr_energy_fraction, smr_energy_fraction, season_count_frac_thresh):
    """
    This function is used to get the seasonal energy fraction difference
    Args:
        wtr_energy_fraction             (float)         : Winter energy fraction
        smr_energy_fraction             (float)         : Summer energy fraction
        season_count_frac_thresh        (float)         : Seasonal count fraction threshold
    Returns:
        seasonal_energy_fraction_diff   (float)         : Seasonal energy fraction difference
    """

    if (wtr_energy_fraction >= season_count_frac_thresh) or (smr_energy_fraction >= season_count_frac_thresh):
        seasonal_energy_fraction_diff = 0
    else:
        seasonal_energy_fraction_diff = np.abs(wtr_energy_fraction - smr_energy_fraction)

    return seasonal_energy_fraction_diff
