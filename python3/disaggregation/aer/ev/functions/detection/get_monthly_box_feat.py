"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to monthly box features for postprocessing
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def remove_boxes_before_ev(box_data, box_features, debug, ev_config):
    """
    Function to calculate difference between circular times of use

        Parameters:
            box_data                  (np.ndarray)              : Box data
            box_features              (np.ndarray)              : Features of the boxes
            debug                     (dict)                    : debug object
            ev_config                  (dict)                    : Module config dict

        Returns:
            box_data                  (np.ndarray)              : Updated box data
            box_features              (np.ndarray)              : Updated features of the boxes
    """

    # Getting configs to be used in the function
    columns_dict = ev_config.get('box_features_dict')
    det_post_process_config = ev_config.get('detection_post_processing')

    if len(box_features) == 0:
        return box_data, box_features, np.max(box_data[:, ev_config['box_data_month_col']])

    # Computing unique months in box data
    box_monthly_count = np.zeros((np.int(np.max(box_data[:, ev_config.get('box_data_month_col')])) + 1,))
    uniq_box_months, month_counts = np.unique(box_features[:, columns_dict.get('boxes_start_month')],
                                              return_counts=True)

    # Computing unique months in which boxes are found
    month_num_days = np.bincount(box_data[:, ev_config['box_data_month_col']].astype(int)) / (
                debug['factor'] * Cgbdisagg.HRS_IN_DAY)
    box_monthly_count[uniq_box_months.astype(int)] = month_counts

    # Months with less than 7 days not included
    box_monthly_count[month_num_days < Cgbdisagg.DAYS_IN_WEEK] = 0

    box_monthly_fractions = np.true_divide(box_monthly_count, month_num_days)
    mean_monthly_fractions = np.mean(box_monthly_fractions[box_monthly_fractions > 0])

    months_with_boxes = np.ones(len(box_monthly_fractions))
    months_with_boxes[(box_monthly_fractions <= det_post_process_config[
        'start_month_count_frac_ratio'] * mean_monthly_fractions) | np.isnan(box_monthly_fractions)] = 0

    last_ev_month = np.where(months_with_boxes > 0)[0][-1]

    # first month with boxes
    first_ev_month = np.where(months_with_boxes > 0)[0][0]

    # Removing boxes before first ev month
    box_data = box_data[box_data[:, ev_config['box_data_month_col']] >= first_ev_month, :]
    box_features = box_features[box_features[:, columns_dict['boxes_start_month']] >= first_ev_month, :]

    return box_data, box_features, last_ev_month


def get_hod_diff(t1, t2):
    """
    Function to calculate difference between circular times of use

        Parameters:
            t1                  (int)              : time of usage
            t2                  (int)              : time of usage

        Returns:
            hod_diff            (int)              : time difference
    """
    # If dist between two hours of day is more than 12, we will 24 - dist to calculate difference
    if np.isnan(t1) or np.isnan(t2):
        hod_diff = 0
    elif abs(t1 - t2) > Cgbdisagg.HRS_IN_DAY / 2:
        hod_diff = Cgbdisagg.HRS_IN_DAY - abs(t1 - t2)
    else:
        hod_diff = abs(t1 - t2)

    return hod_diff


def get_seasonal_box_feat(box_data, box_features, debug, ev_config):
    """
    Function to calculate seasonal features of the boxes

        Parameters:
            box_data                      (np.ndarray)              : Box data
            box_features                  (np.ndarray)              : Features of the boxes
            debug                         (dict)                    : Debug object
            ev_config                      (dict)                    : Module config dict

        Returns:
            box_seasonal_count_var        (float)                    : Seasonal boxes count variation
            box_seasonal_fractions        (np.array)                : season-wise box fractions
    """

    # Getting configs to be used in the function
    columns_dict = ev_config.get('box_features_dict')
    det_post_process_config = ev_config.get('detection_post_processing')

    if len(box_features) == 0:
        return np.nan, [np.nan, np.nan, np.nan]

    # Computing unique season with boxes present
    box_seasonal_count = np.zeros((3,))
    uniq_box_seasons, seasonal_counts = np.unique(box_features[:, columns_dict['boxes_start_season']],
                                                  return_counts=True)

    # Computing unique season in the box data
    season_num_days = np.zeros(3)
    uniq_season, count_seasons = np.unique(box_data[:, ev_config.get('box_data_season_col')].astype(int),
                                           return_counts=True)

    # Removing season with less than 15 days
    season_num_days[uniq_season] = count_seasons / (debug['factor'] * Cgbdisagg.HRS_IN_DAY)
    box_seasonal_count[uniq_box_seasons.astype(int)] = seasonal_counts
    box_seasonal_count[season_num_days < det_post_process_config['min_days_in_month']] = np.nan

    box_seasonal_fractions = np.true_divide(box_seasonal_count, season_num_days)
    box_seasonal_fractions = box_seasonal_fractions / np.nansum(box_seasonal_fractions)
    mean_seas_frac = np.nanmean(box_seasonal_fractions)

    # Seasonal box fraction variation
    box_seasonal_count_var = np.nanstd(box_seasonal_fractions) / mean_seas_frac

    return box_seasonal_count_var, box_seasonal_fractions


def get_monthly_box_feat(box_data, box_features, debug, ev_config):
    """
    Function to calculate monthly features of the boxes

        Parameters:
            box_data                      (np.ndarray)              : Box data
            box_features                  (np.ndarray)              : Features of the boxes
            debug                         (dict)                    : Debug object
            ev_config                      (dict)                    : Module config dict

        Returns:
            box_monthly_count_var         (float)                    : Monthly boxes count variation
            box_monthly_presence_var      (float)                    : Monthly boxes presence variation
    """
    # Getting configs to be used in the function
    columns_dict = ev_config.get('box_features_dict')
    det_post_process_config = ev_config.get('detection_post_processing')

    if len(box_features) == 0:
        return np.nan, np.nan

    box_monthly_count = np.zeros((np.int(np.max(box_data[:, ev_config['box_data_month_col']])) + 1,))

    # Computing unique months with boxes present
    uniq_month_days = np.unique(box_features[:, [columns_dict['boxes_start_month'], columns_dict['box_start_day']]],
                                axis=0)
    uniq_box_months, month_counts = np.unique(uniq_month_days[:, 0], return_counts=True)

    # Computing unique months in the box data
    month_num_days = np.bincount(box_data[:, ev_config.get('box_data_month_col')].astype(int)) / (
                debug['factor'] * Cgbdisagg.HRS_IN_DAY)

    box_monthly_count[uniq_box_months.astype(int)] = month_counts

    # Removing months with less than one week of data
    box_monthly_count = box_monthly_count[month_num_days >= Cgbdisagg.DAYS_IN_WEEK]
    month_num_days = month_num_days[month_num_days >= Cgbdisagg.DAYS_IN_WEEK]

    # Calculating monthly boxes count and presence variation
    box_monthly_fractions = np.true_divide(box_monthly_count, month_num_days)
    mean_monthly_fractions = np.mean(box_monthly_fractions)

    months_with_boxes = np.ones(len(box_monthly_fractions))
    months_with_boxes[
        box_monthly_fractions < det_post_process_config['start_month_count_frac_ratio'] * mean_monthly_fractions] = 0

    box_monthly_count_var = np.std(box_monthly_fractions) / np.mean(box_monthly_fractions)
    box_monthly_presence_var = np.std(np.diff(months_with_boxes))

    return np.round(box_monthly_count_var, 2), np.round(box_monthly_presence_var, 2)


def get_seasonal_hod_diff(box_features, ev_config):
    """
    Function to calculate seasonal boxes hod

        Parameters:
            box_features       (np.ndarray)           : Features of the boxes
            ev_config           (dict)                 : Module config dict

        Returns:
            ac_day_bool        (bool)                 : Boolean signifying whether summer has prominently daytime boxes
            sh_night_bool      (bool)                 : Boolean signifying whether winter has prominently night boxes
            ac_sh_hod_dist     (int)                  : Distance between summer and winter boxes TOU
            prom_smr_hrs       (list)                 : Prominent summer time of use
            prom_wtr_hrs       (list)                 : Prominent winter time of use
    """
    # Getting configs to be used in the function
    columns_dict = ev_config.get('box_features_dict')
    det_post_process_config = ev_config.get('detection_post_processing')

    if len(box_features) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Extracting useful parameters from config
    day_time_start = det_post_process_config['day_time_start']
    day_time_end = det_post_process_config['day_time_end']
    night_time_start = det_post_process_config['night_time_start']
    night_time_end = det_post_process_config['night_time_end']

    # Defining day time and night time hours
    day_time = np.arange(day_time_start, day_time_end)
    night_time = np.arange(night_time_start, night_time_end + Cgbdisagg.HRS_IN_DAY) % Cgbdisagg.HRS_IN_DAY

    smr_subset = box_features[box_features[:, columns_dict['boxes_start_season']] == 2, :]
    wtr_subset = box_features[box_features[:, columns_dict['boxes_start_season']] == 0, :]

    # Calculating most prominent ev hours during summers and winters
    prom_smr_hrs = get_prominent_ev_hrs(smr_subset, columns_dict)
    prom_wtr_hrs = get_prominent_ev_hrs(wtr_subset, columns_dict)

    # Boolean signifying whether summer boxes are in daytime and winter boxes are in night
    ac_day_bool = np.count_nonzero(np.isin(prom_smr_hrs, day_time)) >= det_post_process_config['min_num_hvac_hrs']
    sh_night_bool = np.count_nonzero(np.isin(prom_wtr_hrs, night_time)) >= det_post_process_config['min_num_hvac_hrs']

    # Distance between prom summer and winter hour
    ac_sh_hod_dist = get_hod_diff(prom_smr_hrs[-1], prom_wtr_hrs[-1])

    return int(ac_day_bool), int(sh_night_bool), int(ac_sh_hod_dist), prom_smr_hrs, prom_wtr_hrs


def get_charging_frequency(box_features, last_ev_index, factor, ev_config):
    """
    Function to calculate seasonal boxes hod

        Parameters:
            box_features       (np.ndarray)           : Features of the boxes
            last_ev_index      (int)                  : Index of the last EV month
            factor             (int)                  : Number of data points in an hour
            ev_config           (dict)                 : Module config dict

        Returns:
            charging_freq       (float)                 : Charges per available day
            charges_per_day     (int)                  : charges per day on days with charging
            frac_multi_charge   (list)                 : Fraction of days with more than one charging instance
    """

    # Getting configs to be used in the function
    columns_dict = ev_config.get('box_features_dict')

    box_features_subset = box_features[box_features[:, columns_dict['boxes_start_month']] <= last_ev_index]

    if len(box_features_subset) == 0:
        return np.nan, np.nan, np.nan

    # Calculating number of unique charging days
    uniq_charge_days, counts = np.unique(box_features_subset[:, columns_dict['box_start_day']], return_counts=True)
    charges_per_day = np.mean(counts)

    # Calculating fraction of multi-charging days
    frac_multi_charge = np.count_nonzero(counts > 1) / len(counts)

    num_charge_days = len(uniq_charge_days)
    first_box_ts = box_features_subset[0, columns_dict['start_index_column']]
    last_box_ts = box_features_subset[-1, columns_dict['end_index_column']]

    # Total number of days between first and last box
    num_days = (last_box_ts - first_box_ts) / (factor * Cgbdisagg.HRS_IN_DAY)

    charging_freq = np.round(num_charge_days / num_days, 2)

    return charging_freq, np.round(charges_per_day, 2), np.round(frac_multi_charge, 2)


def get_prominent_ev_hrs(box_features, columns_dict):
    """
    Function to calculate seasonal boxes hod

        Parameters:
            box_features       (np.ndarray)           : Features of the boxes
            columns_dict       (dict)                 : Column definitions of box_features matrix

        Returns:
            prom_ev_hrs        (np.array)             : 3 most common ev hours
    """

    # Creating time buckets
    boxes_hod = np.zeros(Cgbdisagg.HRS_IN_DAY)

    # Calculating number of data points in each time bucket
    for idx, row in enumerate(box_features):
        start_hod = int(row[columns_dict['boxes_start_hod']])
        end_hod = int(np.ceil(row[columns_dict['boxes_start_hod']] + row[columns_dict['boxes_duration_column']]))
        if end_hod < Cgbdisagg.HRS_IN_DAY:
            boxes_hod[start_hod:end_hod] += 1
        else:
            boxes_hod[start_hod:] += 1
            boxes_hod[: end_hod % Cgbdisagg.HRS_IN_DAY] += 1

    # Extracting 3 most common EV hours
    prom_ev_hrs = np.argsort(boxes_hod)[-3:]
    if len(box_features) == 0:
        prom_ev_hrs = [np.nan, np.nan, np.nan]

    return prom_ev_hrs
