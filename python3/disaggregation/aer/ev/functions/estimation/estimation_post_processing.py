"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module with calls for EV estimation post processing functions
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def group_min(groups, data):
    """
    Returns min value in each np.unique bin

    Parameters:
        groups              (np.array)              :   Array containing information of bins
        data                (np.array)              :   Array containing data in each bin

    Returns:
        indices_to_keep     (np.array)              :   Indices of the cleanest boxes in multi-charging days
    """
    order = np.lexsort((data, groups))

    index = np.empty(len(groups), 'bool')
    index[0] = True
    index[1:] = groups[1:] != groups[:-1]

    return order[index]


def post_processing(debug, input_box_data, input_box_features, ev_config, logger):
    """
    Returns min value in each np.unique bin

    Parameters:
        debug                       (dict)                    :   Contains useful info of EV module
        input_box_data              (np.ndarray)              :   Box data
        input_box_features          (np.ndarray)              :   Box Features
        logger                      (logger)                  :   logger for this function
        ev_config                  (dict)                      : Module config dict

    Returns:
        box_data                    (np.ndarray)              :   Updated box data
        new_box_features            (np.ndarray)              :   Updated box Features
    """
    features_column_dict = ev_config.get('box_features_dict')

    box_data = deepcopy(input_box_data)
    box_features = deepcopy(input_box_features)

    # Detecting abnormal boxes using TOU, amp, auc as criteria
    if len(box_features) > 0:
        erroneous_bool = find_erroneous_boxes(box_features, debug, ev_config)
        new_box_features = box_features[~erroneous_bool, :]
        removal_boxes = box_features[erroneous_bool, :]

        logger.info("Total number of removed erroneous boxes from estimation: | {}".format(np.sum(erroneous_bool)))

        # Removing info of erroneous boxes from box data
        for idx, row in enumerate(removal_boxes):
            start_idx, end_idx = row[:features_column_dict['end_index_column'] + 1].astype(int)
            box_data[start_idx: end_idx + 1, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0
    else:
        new_box_features = box_features

    # Finding best boxes on multi-charging days
    if len(new_box_features) > 0:
        selected_boxes_bool = find_multicharge_boxes(new_box_features, debug, ev_config)
        removal_boxes = new_box_features[selected_boxes_bool == 0, :]
        new_box_features = new_box_features[selected_boxes_bool == 1, :]

        logger.info("Total number of removed multi-charge boxes from estimation |  {}".format(len(removal_boxes)))

        # Removing the erroneous boxes from box data
        for idx, row in enumerate(removal_boxes):
            start_idx, end_idx = row[:features_column_dict['end_index_column'] + 1].astype(int)
            box_data[start_idx: end_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    return box_data, new_box_features


def get_tou_diff(time_arr_1, time_arr_2):
    """
    Function to calculate difference between numpy arrays circular times of use

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


def find_erroneous_boxes(box_features, debug, ev_config):
    """
    Returns min value in each np.unique bin

    Parameters:
        debug                       (dict)                    :   Contains useful info of EV module
        box_features                (np.ndarray)              :   Box Features
        ev_config                    (dict)                    :   Module config dict

    Returns:
        final_bool                   (np.array)                :   Array containing info about boxes to remove
    """

    # Getting parameter dicts to be used in this function
    est_post_process_config = ev_config['estimation_post_processing']
    columns_dict = ev_config['box_features_dict']

    # Computing day-time and night-time boxes daytime
    day_time_start = est_post_process_config['day_time_start']
    day_time_end = est_post_process_config['day_time_end']
    night_time_start = est_post_process_config['night_time_start']
    night_time_end = est_post_process_config['night_time_end']

    day_time = np.arange(day_time_start, day_time_end)
    night_time = np.arange(night_time_start, night_time_end + Cgbdisagg.HRS_IN_DAY) % Cgbdisagg.HRS_IN_DAY

    # Definitions of the columns to be used
    auc_col = columns_dict['boxes_areas_column']
    amp_col = columns_dict['boxes_energy_per_point_column']
    season_col = columns_dict['boxes_start_season']
    hod_col = columns_dict['boxes_start_hod']

    # Getting box validity boolean based on probability of being auc, tou, hvac boxes
    tou_diff_thresh = est_post_process_config['tou_diff_thresh']
    lower_auc_ratio = est_post_process_config['lower_auc_ratio']
    lower_amp_ratio = est_post_process_config['lower_amp_ratio']
    auc_ratio = est_post_process_config['auc_ratio']
    amp_ratio = est_post_process_config['amp_ratio']

    smr_season_idx = est_post_process_config['smr']
    wtr_season_idx = est_post_process_config['wtr']

    # Getting box validity boolean based on probability of being auc, tou, hvac boxes
    auc_bool = (box_features[:, auc_col] <= lower_auc_ratio * debug['clean_box_auc']) & (box_features[:, amp_col] <= lower_amp_ratio * debug['clean_box_amp'])

    tou_diff = get_tou_diff(box_features[:, hod_col], debug['clean_box_tou'])

    # Boxes with unusual TOU as well as low AUC and Amplitude
    wrong_tou_bool = (tou_diff > tou_diff_thresh) & (box_features[:, auc_col] <= auc_ratio * debug['clean_box_auc']) & \
                     (box_features[:, amp_col] <= amp_ratio * debug['clean_box_amp'])

    # Boxes with unusual TOU in summer months as well as low AUC and Amplitude
    cooling_bool = (tou_diff >= tou_diff_thresh) & (box_features[:, season_col] == smr_season_idx) & (
        np.isin(box_features[:, hod_col], day_time)) & (box_features[:, auc_col] < auc_ratio * debug['clean_box_auc'])

    # Boxes with unusual TOU in winter months as well as low AUC and Amplitude
    heating_bool = (tou_diff >= tou_diff_thresh) & (box_features[:, season_col] == wtr_season_idx) & (
        np.isin(box_features[:, hod_col], night_time)) & (box_features[:, auc_col] < auc_ratio * debug['clean_box_auc'])

    final_bool = wrong_tou_bool | auc_bool | cooling_bool | heating_bool

    return final_bool


def find_multicharge_boxes(box_features, debug, ev_config):
    """
    Removes erroneous boxes on a multi-charge day

    Parameters:
        debug                       (dict)                    :   Contains useful info of EV module
        box_features                (np.ndarray)              :   Box Features
        ev_config                    (dict)                    :   Module config dict

    Returns:
        final_box_idx                (np.array)                :   Array containing info about boxes to remove
    """

    # Getting parameter dicts to be used in this function
    est_post_process_config = ev_config['estimation_post_processing']
    columns_dict = ev_config['box_features_dict']

    # Extracting useful parameters
    box_start_day_col = columns_dict['box_start_day']
    box_start_hod_col = columns_dict['boxes_start_hod']
    auc_col = columns_dict['boxes_areas_column']
    amp_col = columns_dict['boxes_energy_per_point_column']

    # Getting unique days
    uniq_days, uniq_days_idx, bin_idx = np.unique(box_features[:, box_start_day_col], return_index=True,
                                                  return_inverse=True)

    # TOU diff from clean boxes TOU
    tou_diff = get_tou_diff(box_features[:, box_start_hod_col], debug['clean_box_tou'])

    # Normalised TOU diff by dividing with 12 hours
    tou_diff_norm = tou_diff / (Cgbdisagg.HRS_IN_DAY / 2)

    # Normalised Area under the curve of each box
    auc_norm = box_features[:, auc_col] / np.max(box_features[:, auc_col])
    base_auc = debug['clean_box_auc'] / np.max(box_features[:, auc_col])

    # Normalised Amp of each box
    amp_norm = box_features[:, amp_col] / np.max(box_features[:, amp_col])
    base_amp = debug['clean_box_amp'] / np.max(box_features[:, amp_col])

    # Weighted euclidean distance of each box from the clean boxes
    # Getting weights of each distance from config

    tou_weight = est_post_process_config['tou_weight']
    auc_weight = est_post_process_config['auc_weight']
    amp_weight = est_post_process_config['amp_weight']

    dist_matrix = np.sqrt(tou_weight * np.square(tou_diff_norm) +
                          auc_weight * np.square(auc_norm - base_auc) + amp_weight * np.square(amp_norm - base_amp))

    final_box_idx = np.zeros(len(box_features))

    # Calculating best box on the days with multi-charge instances
    keep_index = group_min(bin_idx, dist_matrix)
    final_box_idx[keep_index] = 1

    removed_box_idx = final_box_idx == 0

    amp_diff_frac_allowed = est_post_process_config['amp_diff_frac_allowed']

    # Keeping boxes with amplitude very similar to that of clean boxes
    final_box_idx[removed_box_idx] = (np.abs((amp_norm[removed_box_idx] - base_amp) / base_amp) <= amp_diff_frac_allowed).astype(int)

    return final_box_idx
