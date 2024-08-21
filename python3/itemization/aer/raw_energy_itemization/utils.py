
"""
Author - Nisha Agarwal
Date - 10th Mar 2021
Utils functions for calculating 100% itemization module
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.itemization_utils import rolling_func

from python3.itemization.aer.functions.itemization_utils import get_index_array


def postprocess_conf_arr(app_confidence, app_potential, app_disagg, vacation):

    """
    Postprocess function for calculated potential and confidence array

    Parameters:
        app_confidence          (np.ndarray)        : appliance ts level confidence
        app_potential           (np.ndarray)        : appliance ts level potential
        app_disagg              (np.ndarray)        : appliance ts level disagg
        vacation                (np.ndarray)        : vacation data

    Returns:
        app_confidence          (np.ndarray)        : appliance ts level confidence
        app_potential           (np.ndarray)        : appliance ts level potential
    """

    # Sanity checks for app confidence

    app_confidence = np.nan_to_num(app_confidence)
    app_confidence = np.fmax(0, app_confidence)
    app_confidence = np.fmin(1, app_confidence)
    app_confidence[np.logical_and(app_disagg > 0, app_confidence == 0)] = 0.1

    # Sanity checks for app potential

    app_potential = np.nan_to_num(app_potential)
    app_potential = np.fmax(0, app_potential)
    app_potential = np.fmin(1, app_potential)
    app_potential[np.logical_and(app_disagg > 0, app_potential == 0)] = 0.1

    app_confidence[vacation] = 0
    app_potential[vacation] = 0

    return app_confidence, app_potential


def get_boxes(app_disagg, samples_per_hour):

    """
    calculate sequence of labels in an array, also calculates different attributes for individual sequence

    Parameters:
        app_disagg           (np.ndarray)        : appliance disagg output
        samples_per_hour     (int)               : samples in an hour

    Returns:
        box_features         (np.ndarray)        : box wise features for all activity boxes in ev/wh
    """

    app_tou = app_disagg > 0

    app_tou_copy = np.reshape(app_tou, app_tou.shape[0] * app_tou.shape[1])
    app_disagg_copy = np.reshape(app_disagg, app_tou.shape[0] * app_tou.shape[1])

    box_energy_idx_diff = np.diff(np.r_[0, app_tou_copy.astype(int), 0])

    # Find the start and end edges of the boxes

    box_start_boolean = (box_energy_idx_diff[:-1] > 0)
    box_end_boolean = (box_energy_idx_diff[1:] < 0)

    box_start_idx = np.where(box_start_boolean)[0]
    box_end_idx = np.where(box_end_boolean)[0]

    box_tou = (box_start_idx + box_end_idx) / 2

    length = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    row_start_idx = box_start_idx / length
    row_end_idx = box_start_idx / length
    col_start_idx = box_start_idx % length
    col_end_idx = box_end_idx % length
    box_tou = box_tou % length

    length = np.where(box_end_boolean)[0] - np.where(box_start_boolean)[0] + 1

    box_start_energy = app_disagg_copy[box_start_idx]
    box_end_energy = app_disagg_copy[box_end_idx]

    box_avg_energy = np.zeros(len(box_start_idx))

    for i in range(len(box_avg_energy)):
        box_avg_energy[i] = app_disagg_copy[int(box_start_idx[i]): int(box_end_idx[i] + 1)].mean()

    # Update features for each box

    box_features = np.c_[row_start_idx.astype(int), row_end_idx.astype(int), col_start_idx, col_end_idx,
                         length, box_start_energy, box_end_energy, box_start_idx, box_end_idx, box_tou, box_avg_energy]

    return box_features


def get_box_score(app_disagg, box_features):

    """
    Calculate rolling avg for the given window size

    Parameters:
        app_disagg                     (np.ndarray)        : app disagg
        box_features                   (int)               : box wise features for ev/wh

    Returns:
        consistent_usage_score_2d      (np.ndarray)        : ev/wh conf score based on consistency in usage
        tou_score_2d                   (np.ndarray)        : ev/wh conf score based on time of usage
    """

    # Calculate consistency, tou score, and valley score of calculating potential and estimation values

    app_tou = app_disagg > 0

    ev_tou_copy = np.reshape(app_tou, app_tou.shape[0] * app_tou.shape[1])
    ev_disagg_copy = np.reshape(app_disagg, app_tou.shape[0] * app_tou.shape[1])

    consistent_usage_score_copy = np.zeros(ev_tou_copy.shape)
    valley_score_copy = np.zeros(ev_tou_copy.shape)

    # Calculating ts level consistency score

    consistent_usage_score = 1 - ((box_features[:, -1] - np.mean(box_features[:, -1])) / np.mean(box_features[:, -1]))

    length = np.sum(np.sum(app_disagg, axis=1) > 0)

    # Calculating tou score

    tou_score = np.sum(app_disagg > 0, axis=0) / length + 0.3

    tou_score = np.fmin(tou_score, 1)
    tou_score_copy = np.zeros(app_disagg.shape)
    tou_score_copy[:, :] = tou_score[None, :]

    start = box_features[:, 7].astype(int)
    end = box_features[:, 8].astype(int)

    # Calculating ts level valley score

    diff = np.divide((ev_disagg_copy[(box_features[:, 8].astype(int) + 1) % len(ev_disagg_copy)] +
                      ev_disagg_copy[(box_features[:, 6].astype(int) - 1) % len(ev_disagg_copy)]) / 2 - box_features[:, -1], box_features[:, -1])

    diff = np.fmax(0, diff)

    valley_score = np.exp(-diff/0.5)

    # Assigning the score to individual boxes

    for i in range(len(box_features)):
        consistent_usage_score_copy[start[i]: end[i] + 1] = consistent_usage_score[i]
        valley_score_copy[start[i]: end[i] + 1] = valley_score[i]

    consistent_usage_score_copy = np.reshape(consistent_usage_score_copy, app_tou.shape)

    tou_score_2d = copy.deepcopy(tou_score_copy)

    samples = int(len(tou_score_copy[0]) / Cgbdisagg.HRS_IN_DAY)

    for i in range(len(tou_score)):
        tou_score_2d[:, i] = np.max(tou_score_copy[:, get_index_array(i-samples, i+samples, Cgbdisagg.HRS_IN_DAY*samples)])

    tou_score_2d = np.fmin(1, tou_score_2d+0.3)

    consistent_usage_score_2d = np.fmin(1, np.fmax(0, consistent_usage_score_copy))

    return consistent_usage_score_2d, tou_score_2d


def find_overlap_days(day, app_usage_unique_days, counts, overlap_count, samples_per_hour):

    """
    Find the overlapping days, for each pp usage pattern

    Parameters:
        day                     (np.ndarray)     : current pp usage pattern
        app_usage_unique_days   (np.ndarray)     : all the unique pp usages
        counts                  (np.ndarray)     : count of all unique pp usage
        overlap_count           (int)            : count of current pp usage pattern
        samples_per_hour        (int)            : samples in an hour

    Returns:
        overlap_day             (np.ndarray)     : pattern of overlapped pp day
        overlap_count           (int)            : count of overlapped pp pattern
    """

    count = samples_per_hour * Cgbdisagg.HRS_IN_DAY

    overlap_day = day

    for index, target_day in enumerate(app_usage_unique_days):

        if counts[index] > 3 and np.abs(target_day.astype(int) - day.astype(int)).sum() < count and np.abs(target_day.astype(int) - day.astype(int)).sum() > 0:
            overlap_day = target_day
            count = np.abs(target_day.astype(int) - day.astype(int)).sum()
            overlap_count = counts[index] + overlap_count

    return overlap_day, count < 2*samples_per_hour, overlap_count


def get_daily_profile(original_input_data, window_length):

    """
    Calculates cumulative energy profile that represents day wise consumption throughout the year

    Parameters:
        original_input_data        (np.ndarray)    : raw energy data in 2d
        window_length              (int)           : window length

    Returns:
        daily_energy_profile       (np.ndarray)    : cumulative energy profile
    """

    daily_energy_profile = np.zeros(original_input_data.shape)

    length = len(original_input_data)

    for window in range(0, length - window_length, 5):
        daily_energy_profile[window] = np.percentile(original_input_data[window: window + window_length], 60, axis=0)

    daily_energy_profile = daily_energy_profile[~np.all(daily_energy_profile == 0, axis=1)]

    # Valid number of days not found

    if not len(daily_energy_profile):
        return np.zeros(original_input_data.shape[1])

    daily_energy_profile = np.percentile(daily_energy_profile, 10, axis=0)

    return daily_energy_profile


def update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons, add_hvac=1):

    """
    update appliance ts level rages with new calculated values

    Parameters:
        app_index                   (int)           : mapping of the target appliance
        item_output_object          (dict)          : Dict containing all hybrid outputs
        mid_cons                    (np.ndarray)    : avg ts level consumption of the appliance
        max_cons                    (np.ndarray)    : avg ts level consumption of the appliance
        min_cons                    (np.ndarray)    : avg ts level consumption of the appliance

    Returns:
        item_output_object          (dict)          : Dict containing all hybrid outputs
    """

    if add_hvac:
        min_cons = np.minimum(min_cons, mid_cons)
        max_cons = np.maximum(max_cons, mid_cons)

    item_output_object["inference_engine_dict"]["appliance_mid_values"][app_index, :, :] = np.fmax(0, mid_cons)
    item_output_object["inference_engine_dict"]["appliance_max_values"][app_index, :, :] = np.fmax(0, max_cons)
    item_output_object["inference_engine_dict"]["appliance_min_values"][app_index, :, :] = np.fmax(0, min_cons)

    return item_output_object


def update_stat_app_tou(app_seq, app_tou, overest_tou, samples_per_hour, activity_curve, mid_cons, min_cons, max_cons):

    """
    Updating TOU of cook/ld, in case of overlap

    Parameters:
        app_seq                     (np.ndarray)    : sequences of appliance usage
        app_tou                     (np.ndarray)    : appliance tou
        overest_tou                 (np.ndarray)    : overestimation timstamps
        samples_per_hour            (int)           : samples in an hour
        activity_curve              (np.ndarray)    : actvity profile of the user
        min_cons                    (np.ndarray)    : ts level min app consumption
        mid_cons                    (np.ndarray)    : ts level avg app consumption
        max_cons                    (np.ndarray)    : ts level max app consumption

    Returns:
        min_cons                    (np.ndarray)    : ts level min app consumption
        mid_cons                    (np.ndarray)    : ts level avg app consumption
        max_cons                    (np.ndarray)    : ts level max app consumption
    """

    total_samples = int(samples_per_hour * Cgbdisagg.HRS_IN_DAY)
    samples_per_hour = int(samples_per_hour)

    app_seq = app_seq[app_seq[:, 0] == 1, :]

    for seq in range(len(app_seq)):

        # Updating TOU of individual usage segments, the final tou is decided using score array
        # This score array is assigned weightage using activity profile and appliance behaviour
        # This process is repeated for each tou segment

        if np.any(np.multiply(app_tou[get_index_array(app_seq[seq, 1], app_seq[seq, 2], Cgbdisagg.HRS_IN_DAY * samples_per_hour)],
                              overest_tou[get_index_array(app_seq[seq, 1], app_seq[seq, 2], Cgbdisagg.HRS_IN_DAY * samples_per_hour)])):
            length = app_seq[seq, 3]

            if length < 2:
                continue

            rolling_act_curve = rolling_func(activity_curve, length / 2)
            rolling_act_curve = rolling_act_curve[: total_samples]

            score = np.zeros(total_samples)
            score[get_index_array((app_seq[seq, 1] - 1 * samples_per_hour) % total_samples,
                                  (app_seq[seq, 2] + 1 * samples_per_hour) % total_samples, total_samples)] = 1
            score[app_tou.astype(bool)] = 1
            score[overest_tou.astype(bool)] = 0

            score = np.multiply(score, rolling_act_curve)

            if np.all(score == 0):
                continue

            score = rolling_func(score, length / 2)

            index = np.argmax(score)

            consumption = copy.deepcopy(mid_cons[:, get_index_array(app_seq[seq, 1], app_seq[seq, 2], total_samples)])

            # Updating min, max and mid app ranges

            mid_cons[:, get_index_array(app_seq[seq, 1], app_seq[seq, 2], total_samples)[: length]] = 0
            mid_cons[:, get_index_array((index - length / 2) % total_samples,
                                        (index + length / 2) % total_samples, total_samples)[: length]] = consumption

            min_cons[:, get_index_array(app_seq[seq, 1], app_seq[seq, 2], total_samples)] = 0
            min_cons[:, get_index_array((index - length / 2) % total_samples,
                                        (index + length / 2) % total_samples, total_samples)[: length]] = consumption

            max_cons[:, get_index_array(app_seq[seq, 1], app_seq[seq, 2], total_samples)] = 0
            max_cons[:, get_index_array((index - length / 2) % total_samples,
                                        (index + length / 2) % total_samples, total_samples)[: length]] = consumption

    return min_cons, mid_cons, max_cons
