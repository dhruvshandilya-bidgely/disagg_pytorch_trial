"""
Author - Paras Tehria / Sahana M
Date - 2-Feb-2022
Module to find high energy boxes in consumption data
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.rolling_function import rolling_function
from python3.disaggregation.aer.ev.functions.ev_utils import bill_cycle_to_month


def get_user_features(debug, ev_config):
    """
    Function to extract the user features
    Parameters:
        debug                   (Dict)          : Debug dictionary
        ev_config               (Dict)          : EV configurations
    Returns:
        debug                   (Dict)          : Debug dictionary
    """
    # Retrieve the params from the config

    region = ev_config['region']

    sampling_rate = ev_config.get('sampling_rate')

    night_hours = ev_config.get('night_hours')

    weekend = ev_config.get('weekend_days')

    features_dict = ev_config.get('features_dict')

    user_features = {
        'uuid': ev_config['uuid'],
        'region': ev_config['region'],
        'pilot_id': ev_config['pilot_id'],
        'sampling_rate': ev_config['sampling_rate']
    }

    # Find the serial number of final iteration of dynamic box fitting

    updated_box_keys = [key.split('_')[-1] for key in list(debug['l1'].keys()) if 'updated_box_data' in key]

    keys_int = list(map(int, updated_box_keys))

    max_key = str(np.max(keys_int))

    factor = debug['factor']

    # Get the final box data and box features

    box_data_original = debug['l1']['updated_box_data_' + max_key]

    box_data = deepcopy(debug['l1']['reliable_box_data'])
    box_features = deepcopy(debug['l1']['reliable_box_features'])

    box_data_unrel = deepcopy(debug['l1']['unreliable_box_data'])
    box_features_unrel = deepcopy(debug['l1']['unreliable_box_features'])

    box_data = add_season_month_tag(box_data)

    box_start_day = box_data[box_features[:, 0].astype(int), Cgbdisagg.INPUT_DAY_IDX]
    boxes_start_hod = box_data[box_features[:, 0].astype(int), Cgbdisagg.INPUT_HOD_IDX]
    boxes_start_month = box_data[box_features[:, 0].astype(int), Cgbdisagg.INPUT_DIMENSION]

    box_features = np.c_[box_features, boxes_start_hod, boxes_start_month, box_start_day]

    if len(box_features) > 0:
        first_ev_month, idx_first_month = remove_boxes_before_ev(box_data, box_features, debug)
    else:
        first_ev_month, idx_first_month = 0, 0

    box_data = box_data[box_data[:, Cgbdisagg.INPUT_DIMENSION] >= first_ev_month, :]
    box_features = box_features[box_features[:, 12] >= first_ev_month, :]
    box_features[:, :2] = box_features[:, :2] - idx_first_month

    box_data_unrel = box_data_unrel[box_data_unrel[:, Cgbdisagg.INPUT_DIMENSION] >= first_ev_month, :]
    box_features_unrel = box_features_unrel[box_features_unrel[:, 12] >= first_ev_month, :]
    box_features_unrel[:, :2] = box_features_unrel[:, :2] - idx_first_month

    box_data = box_data[:, :Cgbdisagg.INPUT_DIMENSION]
    box_features = box_features[:, :11]
    max_energy = debug['l1']['amplitude_' + max_key] / factor

    debug['l1']['detection_box_data'] = box_data
    debug['l1']['detection_box_features'] = box_features

    box_data = box_data[:, :Cgbdisagg.INPUT_DIMENSION]
    box_features = box_features[:, :11]

    n_boxes = box_features.shape[0]
    n_data_points = box_data.shape[0]

    # Create the user features

    # Boxes energy / duration features

    average_duration = np.mean(box_features[:, features_dict['boxes_duration_column']])

    duration_variation = np.mean(np.abs(box_features[:, 4] - np.mean(box_features[:, 4]))) / average_duration

    boxes_energy_variation = np.std(box_features[:, 8]) / max_energy

    within_boxes_variation = np.mean(box_features[:, 6]) / max_energy

    energy_per_hour = np.sum(box_features[:, 5]) / np.sum(box_features[:, features_dict['boxes_duration_column']])

    energy_per_charge = np.mean(box_features[:, features_dict['boxes_areas_column']])

    # Time of day features

    night_boolean = ((box_data[:, Cgbdisagg.INPUT_HOD_IDX] >= night_hours[0]) |
                     (box_data[:, Cgbdisagg.INPUT_HOD_IDX] <= night_hours[1])).astype(int)

    box_data = np.c_[box_data, night_boolean]

    box_features = np.c_[
        box_features, night_boolean[
            ((box_features[:, 0].astype(int) + box_features[:, 1].astype(int)) / 2).astype(int)]]

    night_count_fraction = np.sum(box_features[:, features_dict['night_boolean']]) / n_boxes

    day_count_fraction = 1 - night_count_fraction

    start_tou_arr = box_data[box_features[:, 0].astype(int), Cgbdisagg.INPUT_HOD_IDX]

    if len(start_tou_arr) > 0:
        mode_tou = max(set(start_tou_arr), key=list(start_tou_arr).count)
    else:
        mode_tou = 0

    tou_variation = np.mean(get_tou_diff(start_tou_arr, mode_tou)) / 24

    # Weekday / weekend based features

    weekend_boolean = ((box_data[:, Cgbdisagg.INPUT_DOW_IDX] == weekend[0]) |
                       (box_data[:, Cgbdisagg.INPUT_HOD_IDX] == weekend[1])).astype(int)

    box_data = np.c_[box_data, weekend_boolean]

    box_features = np.c_[box_features, weekend_boolean[box_features[:, 0].astype(int)]]

    weekend_boxes = box_features[box_features[:, 12] == 1]
    weekday_boxes = box_features[box_features[:, 12] == 0]

    weekend_night_fraction = np.sum(weekend_boxes[:, features_dict['night_boolean']]) / weekend_boxes.shape[0]
    weekday_night_fraction = np.sum(weekday_boxes[:, features_dict['night_boolean']]) / weekday_boxes.shape[0]

    weekend_day_fraction = 1 - weekend_night_fraction
    weekday_day_fraction = 1 - weekday_night_fraction

    # Seasonal features

    cutoff_temperature = 65

    wtr_boolean = (box_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX] <= cutoff_temperature).astype(int)

    wtr_idx = np.where(wtr_boolean)[0]

    smr_boolean = (box_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX] > cutoff_temperature).astype(int)

    smr_idx = np.where(smr_boolean)[0]

    box_data = np.c_[box_data, wtr_boolean, smr_boolean]

    box_features = np.c_[box_features, smr_boolean[box_features[:, 0].astype(int)] + 1]

    # Count of boxes in different seasons

    wtr_boxes = box_features[box_features[:, 13] == 1]
    smr_boxes = box_features[box_features[:, 13] == 2]

    wtr_count = wtr_boxes.shape[0]
    smr_count = smr_boxes.shape[0]

    # Fraction of boxes for each season

    wtr_count_fraction, smr_count_fraction, seasonal_count_fraction_diff, wtr_proportion, smr_proportion = \
        get_seasonal_count_fraction(box_features, wtr_idx, n_data_points, n_boxes)

    # Proportion of energy in various seasons

    wtr_energy_fraction, smr_energy_fraction = get_seasonal_energy_fraction(box_data, wtr_idx, smr_idx,
                                                                            wtr_proportion, smr_proportion)

    # Get the seasonal features

    seasonal_energy_fraction_diff, seasonal_auc_frac_diff, seasonal_amp_frac_diff, seasonal_tou_diff = \
        get_seasonal_features(wtr_energy_fraction, smr_energy_fraction, wtr_count, smr_count, wtr_boxes, smr_boxes, start_tou_arr, box_features)

    # Count charges per week

    number_of_weeks = n_data_points / (Cgbdisagg.DAYS_IN_WEEK * Cgbdisagg.HRS_IN_DAY * factor)

    weekly_count = n_boxes / number_of_weeks

    # Normalizing the weekly count

    if weekly_count >= 7:
        weekly_count_pro = 3
    elif weekly_count >= 4:
        weekly_count_pro = 2
    elif weekly_count >= 2:
        weekly_count_pro = 1
    else:
        weekly_count_pro = weekly_count / 2

    # Correlation of boxes energy with temperature

    energy_idx = (box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0)

    box_energy = box_data[energy_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    box_temperature = np.abs(box_data[energy_idx, Cgbdisagg.INPUT_TEMPERATURE_IDX] - cutoff_temperature)

    correlation_factor = np.abs(np.corrcoef(box_energy, box_temperature)[0, 1])

    # Consistency factor

    day_consistency_score = consistency_factor(box_data, ev_config)

    monthly_consistency_score, monthly_energy_variation = monthly_consistency_factor(box_data)

    day_consistency_score_unrel = consistency_factor(box_data_unrel, ev_config)

    monthly_consistency_score_unrel, monthly_energy_variation_unrel = monthly_consistency_factor(box_data_unrel)

    n_boxes_pro, wtr_count_pro, smr_count_pro = prorate_count_features(wtr_count, smr_count, debug, ev_config)

    num_rel_box = len(box_features)

    if num_rel_box + len(box_features_unrel) > 0:
        fraction_rel = num_rel_box / (num_rel_box + len(box_features_unrel))
    else:
        fraction_rel = 0

    # Region categorical feature

    if region == 'NA':
        user_features['region_NA'] = 1
        user_features['region_EU'] = 0
    elif region == 'EU':
        user_features['region_NA'] = 0
        user_features['region_EU'] = 1
    else:
        user_features['region_NA'] = 0
        user_features['region_EU'] = 0

    # Sampling rate categorical features

    if sampling_rate == 900:
        user_features['sampling_900'] = 1
        user_features['sampling_1800'] = 0
        user_features['sampling_3600'] = 0
    elif sampling_rate == 1800:
        user_features['sampling_900'] = 0
        user_features['sampling_1800'] = 1
        user_features['sampling_3600'] = 0
    elif sampling_rate == 3600:
        user_features['sampling_900'] = 0
        user_features['sampling_1800'] = 0
        user_features['sampling_3600'] = 1
    else:
        user_features['sampling_900'] = 0
        user_features['sampling_1800'] = 0
        user_features['sampling_3600'] = 0

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
    user_features['num_rel_box'] = np.round(num_rel_box, 3)
    user_features['fraction_rel'] = np.round(fraction_rel, 3)
    user_features['day_consistency_score_unrel'] = np.round(day_consistency_score_unrel, 3)
    user_features['monthly_consistency_score_unrel'] = np.round(monthly_consistency_score_unrel, 3)
    user_features['monthly_energy_variation_unrel'] = np.round(monthly_energy_variation_unrel, 3)

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

    debug['l1']['user_features'] = user_features

    # Updated features box data

    debug['l1']['features_box_data'] = deepcopy(box_data_original)

    return debug


def consistency_factor(input_box_data, ev_config):
    """
    Function to calculate the consistency value for the EV boxes
    Parameters:
        input_box_data              (np.ndarray)            : Input box data
        ev_config                   (Dict)                  : EV configurations
    Returns:
        consistency_score           (float0                 : Consistency score
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


def monthly_consistency_factor(input_box_data):
    """
    Calculate the monthly consistency for the EV boxes
    Parameters:
        input_box_data          (np.ndarray)        : Input box data
    Returns:
        consistency_score       (float)             : Consistency score
        energy_variation        (float)             : Energy variation
    """

    # Extract required params from config

    window_size = 3

    box_data = deepcopy(input_box_data)

    energy_idx = (box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0)

    # Get the daily peaks count

    unq_bill_cycle, bill_cycle_idx = np.unique(box_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_inverse=True)
    monthly_peaks_count = np.bincount(bill_cycle_idx, energy_idx)

    # Select days with non-zero peaks

    peak_bill_cycle_idx = np.where(monthly_peaks_count > 0)[0]

    if len(peak_bill_cycle_idx) > 0:

        peak_days_boolean = (monthly_peaks_count > 0).astype(int)

        window_count = rolling_function(peak_days_boolean, window_size, 'sum')

        consistency_score = np.std(window_count)
    else:
        consistency_score = 0

    # Calculate the energy variation

    monthly_box_energy = np.bincount(bill_cycle_idx, box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    max_energy = np.nanmax(monthly_box_energy)

    energy_variation = np.nanstd(monthly_box_energy) / max_energy

    return consistency_score, energy_variation


def prorate_count_features(wtr_count, smr_count, debug, ev_config):
    """
    Calculate prorated count features for each season
    Parameters:
        wtr_count               (float)         : Winter boxes count
        smr_count               (float)         : Summer boxes count
        debug                   (Dict)          : Debug dictionary
        ev_config               (Dict)          : EV configurations dictionart
    Returns:
        counts                  (float)         : Box counts value
    """

    # Extract required variables

    missing_data_info = debug['missing_data_info']
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


def get_tou_diff(arr1, arr2):
    """
    This function is used to obtain the TOU difference
    Parameters:
        arr1                (np.ndarray)            : Array 1
        arr2                (np.ndarray)            : Array 2
    Returns:
        tou_diff            (np.ndarray)            : TOU difference
    """

    # Get the time of usage difference

    tou_diff = np.abs(arr1 - arr2)
    tou_diff[tou_diff > 12] = 24 - tou_diff[tou_diff > 12]
    tou_diff[np.isnan(tou_diff)] = 0
    return tou_diff


def remove_boxes_before_ev(box_data, box_features, debug):
    """
    Function used to remove boxes before EV
    Parameters:
        box_data                (np.ndarray)        : Input Box data
        box_features            (Dict)              : Box features
        debug                   (Dict)              : Debug dictionary
    Returns:
        first_ev_month          (Boolean)           : First EV month array
        idx_first_month         (float)             : First EV month index
    """

    # get the monthly box counts, unique months
    box_monthly_count = np.zeros((np.int(np.max(box_data[:, Cgbdisagg.INPUT_DIMENSION])) + 1,))
    uniq_box_months, month_counts = np.unique(box_features[:, 12], return_counts=True)

    month_num_days = np.bincount(box_data[:, Cgbdisagg.INPUT_DIMENSION].astype(int)) / (debug['factor'] * Cgbdisagg.HRS_IN_DAY)
    box_monthly_count[uniq_box_months.astype(int)] = month_counts

    # month with less than 7 days not included
    box_monthly_count[month_num_days < 7] = 0

    box_monthly_fractions = np.true_divide(box_monthly_count, month_num_days)
    mean_monthly_fractions = np.mean(box_monthly_fractions[box_monthly_fractions > 0])

    months_with_boxes = np.ones(len(box_monthly_fractions))
    months_with_boxes[(box_monthly_fractions <= 0.5 * mean_monthly_fractions) | np.isnan(box_monthly_fractions)] = 0

    # first month with boxes
    first_ev_month = np.where(months_with_boxes > 0)[0][0]
    idx_first_month = np.where(box_data[:, Cgbdisagg.INPUT_DIMENSION] == first_ev_month)[0][0]

    return first_ev_month, idx_first_month


def add_season_month_tag(input_data):
    """
    Function to tag season tag for each month
    Parameters:
        input_data              (np.ndarray)            : Input data
    Returns:
        input_data              (np.ndarray)            : Input data
    """
    # getting month number and season epoch level

    input_data_copy = bill_cycle_to_month(input_data)

    uniq_months, uniq_month_indices = np.unique(input_data_copy[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_inverse=True)
    input_data = np.c_[input_data, uniq_month_indices]

    return input_data


def get_seasonal_count_fraction(box_features, wtr_idx, n_data_points, n_boxes):
    """
    Function used to get the seasonal count fractions
    Parameters:
        box_features            (np.ndarray)            : Box features
        wtr_idx                 (float)                 : Winter index
        n_data_points           (float)                 : Number of data points in total
        n_boxes                 (float)                 : Number of boxes in total
    Returns:
        wtr_count_fraction      (float)                 : Winter boxes count fraction
        smr_count_fraction      (float)                 : Summer boxes count fraction
        seasonal_count_fraction_diff (float)            : Seasonal boxes count fraction
        wtr_proportion          (float)                 : Winter boxes proportion
        smr_proportion          (float)                 : Summer boxes proportion
    """

    # Count of boxes in different seasons

    wtr_boxes = box_features[box_features[:, 13] == 1]
    smr_boxes = box_features[box_features[:, 13] == 2]

    wtr_count = wtr_boxes.shape[0]
    smr_count = smr_boxes.shape[0]

    # Fraction of seasons in input data

    wtr_proportion = len(wtr_idx) / n_data_points
    smr_proportion = 1 - wtr_proportion

    if wtr_count > 0 and wtr_proportion > 0:
        wtr_count_fraction = wtr_count / (n_boxes * wtr_proportion)
    else:
        wtr_count_fraction = 0

    if smr_count > 0 and smr_proportion > 0:
        smr_count_fraction = smr_count / (n_boxes * smr_proportion)
    else:
        smr_count_fraction = 0

    # Normalize the count fraction

    sum_fractions = wtr_count_fraction + smr_count_fraction

    if sum_fractions > 0:
        wtr_count_fraction = wtr_count_fraction / sum_fractions
        smr_count_fraction = smr_count_fraction / sum_fractions
    else:
        wtr_count_fraction = 0
        smr_count_fraction = 0

    if (wtr_count_fraction >= 0.95) or (smr_count_fraction >= 0.95):
        seasonal_count_fraction_diff = 0
    else:
        seasonal_count_fraction_diff = np.abs(wtr_count_fraction - smr_count_fraction)

    return wtr_count_fraction, smr_count_fraction, seasonal_count_fraction_diff, wtr_proportion, smr_proportion


def get_seasonal_energy_fraction(box_data, wtr_idx, smr_idx, wtr_proportion, smr_proportion):
    """
    Function to get the seasonal energy differences in the boxes
    Parameters:
        box_data                (np.ndarray)                : Box data
        wtr_idx                 (float)                     : Winter indexes
        smr_idx                 (float)                     : Summer indexes
        wtr_proportion          (float)                     : Winter proportion
        smr_proportion          (float)                     : Summer proportion
    Returns:
        wtr_energy_fraction     (float)                     : Winter energy fraction
        smr_energy_fraction     (float)                     : Summer energy fraction
    """

    wtr_energy = np.sum(box_data[wtr_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    smr_energy = np.sum(box_data[smr_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    total_energy = wtr_energy + smr_energy

    if wtr_energy > 0 and wtr_proportion > 0:
        wtr_energy_fraction = wtr_energy / (total_energy * wtr_proportion)
    else:
        wtr_energy_fraction = 0

    if smr_energy > 0 and smr_proportion > 0:
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

    return wtr_energy_fraction, smr_energy_fraction


def get_seasonal_features(wtr_energy_fraction, smr_energy_fraction, wtr_count, smr_count, wtr_boxes, smr_boxes, start_tou_arr, box_features):
    """
    Calculate seasonal features
    Parameters:
        wtr_energy_fraction                 (float)         : Winter energy fraction
        smr_energy_fraction                 (float)         : Summer energy fraction
        wtr_count                           (float)         : Winter count
        smr_count                           (float)         : Summer count
        wtr_boxes                           (float)         : Winter boxes
        smr_boxes                           (float)         : Summer boxes
        start_tou_arr                       (np.ndarray)    : Start tou array
        box_features                        (np.ndarray)    : Box features
    Returns:
        seasonal_energy_fraction_diff       (float)         : Seasonal energy fraction difference
        seasonal_auc_frac_diff              (float)         : Seasonal AUC fraction difference
        seasonal_amp_frac_diff              (float)         : Seasonal amplituce fraction difference
        seasonal_tou_diff                   (float)         : Seasonal TOU difference
    """

    if (wtr_energy_fraction >= 0.95) or (smr_energy_fraction >= 0.95):
        seasonal_energy_fraction_diff = 0
    else:
        seasonal_energy_fraction_diff = np.abs(wtr_energy_fraction - smr_energy_fraction)

    # seasonal AUC diff

    if (wtr_count >= 2) and (smr_count > 2):
        wtr_auc = np.mean(wtr_boxes[:, 5])
        smr_auc = np.mean(smr_boxes[:, 5])
        seasonal_auc_frac_diff = np.abs(wtr_auc - smr_auc) / max(wtr_auc, smr_auc)
    else:
        seasonal_auc_frac_diff = 0

    # seasonal amp diff

    if (wtr_count > 2) and (smr_count > 2):
        wtr_amp = np.mean(wtr_boxes[:, 7])
        smr_amp = np.mean(smr_boxes[:, 7])
        seasonal_amp_frac_diff = np.abs(wtr_amp - smr_amp) / max(wtr_amp, smr_amp)
    else:
        seasonal_amp_frac_diff = 0

    # seasonal TOU diff
    if (wtr_count > 2) and (smr_count > 2):
        unq_smr_tou, count_smr_tou = np.unique(start_tou_arr[box_features[:, 13] == 2], return_counts=True)
        top_smr_tou = unq_smr_tou[count_smr_tou.argsort()[-2:]]

        unq_wtr_tou, count_wtr_tou = np.unique(start_tou_arr[box_features[:, 13] == 1], return_counts=True)
        top_wtr_tou = unq_wtr_tou[count_wtr_tou.argsort()[-2:]]

        combination_arr = np.array(np.meshgrid(top_smr_tou, top_wtr_tou)).T.reshape(-1, 2)
        seasonal_tou_diff = np.min(get_tou_diff(combination_arr[:, 0], combination_arr[:, 1])) / 24
    else:
        seasonal_tou_diff = 0

    return seasonal_energy_fraction_diff, seasonal_auc_frac_diff, seasonal_amp_frac_diff, seasonal_tou_diff
