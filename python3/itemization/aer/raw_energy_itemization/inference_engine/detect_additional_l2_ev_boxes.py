

"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update ev consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime
from numpy.random import RandomState

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.init_itemization_config import random_gen_config
from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val
from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val_for_valid_boxes
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.inference_engine.blocking_false_ev_detection_cases import block_user_with_feeble_ev_boxes
from python3.itemization.aer.raw_energy_itemization.inference_engine.blocking_false_ev_detection_cases import eliminate_seasonal_ev_cases

from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import get_recent_ev_flag
from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import get_freq
from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import check_seasonal_check_flag
from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import get_charger_type

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_ev_params_for_det


def detect_leftover_ev_boxes(item_input_object, l1_bool, ev_disagg, residual_cons, ev_absent_in_previous_run):
    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        l1_bool                   (bool)          : flag to determine the target charger type of current iteration
        ev_disagg                 (np.ndarray)    : Ev disagg output
        residual_cons             (np.ndarray)    : Residual data
        ev_absent_in_previous_run (bool)          : flag that represents whether l2 charger is present in previous disagg run
    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
        hybrid_ev_amp             (float)         : ev amplitude of ev added from hybrid module
    """

    seq_label = seq_config.SEQ_LABEL

    # removing baseload type consumption

    residual_cons = residual_cons - np.percentile(residual_cons, 30, axis=1)[:, None]

    # if ev charger type is l1 from true disagg, Ev l2 box detection wont run

    charger_type = get_charger_type(item_input_object)

    default_possible_ev_cons = np.zeros_like(ev_disagg)
    default_amp = 0

    l1_ev_type = charger_type == 1 or l1_bool

    samples = int(ev_disagg.shape[1] / Cgbdisagg.HRS_IN_DAY)

    params = get_ev_params_for_det(samples, ev_disagg, item_input_object)

    min_len = params.get('min_len')
    low_dur_box_thres = params.get('low_dur_box_thres')
    max_len = params.get('max_len')
    initial_amp = params.get('initial_amp')
    ev_amp = params.get('ev_amp')
    max_amp_for_disagg_users = params.get('max_amp_for_disagg_users')
    min_amp_for_disagg_users = params.get('min_amp_for_disagg_users')
    min_amp = params.get('min_amp')
    max_amp = params.get('max_amp')

    input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    ev_amp_cap = params.get('ev_amp_cap')
    high_freq_days_thres = params.get('high_freq_days_thres')
    high_cons_ev_thres = params.get('high_cons_ev_thres')

    min_ev_days = 5 * (not ev_absent_in_previous_run) + 15 * ev_absent_in_previous_run

    possible_ev_cons = residual_cons > initial_amp

    # checking points that can be potential ev boxes based on consumption level

    if np.sum(ev_disagg) > 0:
        possible_ev_cons = residual_cons > max(initial_amp, min_amp_for_disagg_users)
        possible_ev_cons = np.logical_and(possible_ev_cons, residual_cons < max_amp_for_disagg_users)

    elif ev_amp != -1:
        possible_ev_cons = residual_cons > max(initial_amp, min_amp)
        possible_ev_cons = np.logical_and(possible_ev_cons, residual_cons < max_amp)

    ev_1d = (possible_ev_cons > 0).flatten()
    ev_usage_seq = find_seq(ev_1d, np.zeros_like(ev_1d), np.zeros_like(ev_1d), overnight=0)

    possible_ev_cons[residual_cons > ev_amp_cap / samples] = 0
    possible_ev_cons = possible_ev_cons.flatten()

    residual_cons_1d = residual_cons.flatten()

    # Removing box consumption based on whether box length is in given range

    low_duration_tou = np.where(
        np.maximum(ev_disagg > 0, possible_ev_cons.reshape(ev_disagg.shape)).sum(axis=0) > Cgbdisagg.DAYS_IN_MONTH)[0]

    possible_ev_cons = remove_incorrect_ev_box_2(possible_ev_cons, ev_usage_seq, residual_cons_1d, low_duration_tou,
                                                 samples, min_len, max_len, low_dur_box_thres)

    possible_ev_cons = np.reshape(possible_ev_cons, ev_disagg.shape)

    # no further checks will be performed if ev boxes are not detected

    if (np.sum(possible_ev_cons) == 0) or l1_ev_type:
        return default_possible_ev_cons, default_possible_ev_cons, default_amp

    # checking possible ev consumption by removing low consumption boxes

    possible_ev_cons = residual_cons > np.percentile(residual_cons[possible_ev_cons > 0], 70) * 0.8

    # if ev disagg is non zero, including ev disagg boxes

    if np.sum(ev_disagg) > 0:
        possible_ev_cons = np.logical_and(possible_ev_cons, input_data < max_amp_for_disagg_users)

    ev_1d = (possible_ev_cons > 0).flatten()
    ev_usage_seq = find_seq(ev_1d, np.zeros_like(ev_1d), np.zeros_like(ev_1d))

    # Removing box consumption whether box length not in given range

    possible_ev_cons = possible_ev_cons.flatten()

    possible_ev_cons = remove_incorrect_ev_box_1(possible_ev_cons, ev_usage_seq, residual_cons_1d, low_duration_tou,
                                                 samples, min_len, max_len, low_dur_box_thres)

    possible_ev_cons = np.reshape(possible_ev_cons, ev_disagg.shape)

    # no further checks will be performed if ev very ev boxes are detected

    ev_usage_seq = find_seq(((possible_ev_cons + ev_disagg) > 0).flatten(), np.zeros_like(ev_1d), np.zeros_like(ev_1d))

    if (np.sum(possible_ev_cons) == 0) or (np.sum(ev_usage_seq[:, seq_label] > 0) < min_ev_days):
        return default_possible_ev_cons, default_possible_ev_cons, default_amp

    possible_ev_cons_copy = np.logical_or(possible_ev_cons > 0, ev_disagg > 0)

    data_missing_points = np.sum(item_input_object['item_input_params']['day_input_data'] > 0,
                                 axis=0) < Cgbdisagg.DAYS_IN_WEEK

    # checking frequency of boxes in a day
    # if day wise frequency of boxes is high, ev detection will be made 0

    possible_ev_cons_copy[:, data_missing_points] = 1

    ev_day_wise_freq = np.roll(possible_ev_cons_copy[:, np.logical_not(data_missing_points)], 1, axis=1).astype(int) - \
                       possible_ev_cons_copy[:, np.logical_not(data_missing_points)].astype(int)

    ev_day_wise_freq = np.sum(ev_day_wise_freq == -1, axis=1)

    ev_median_cons = np.median(residual_cons[possible_ev_cons_copy > 0])

    if (ev_median_cons > high_cons_ev_thres) or \
            item_input_object.get("config").get("pilot_id") in PilotConstants.EU_PILOTS:
        high_freq_days_thres = 0.3

    if samples == 1:
        ev_amp = ev_median_cons
        high_freq_days_thres = 0.15 * (ev_amp <= high_cons_ev_thres) + 0.25 * (ev_amp > high_cons_ev_thres)

    non_ev = (np.sum(ev_day_wise_freq > 1) / np.sum(ev_day_wise_freq > 0)) > high_freq_days_thres

    high_freq_ev = (non_ev > 0) and ev_absent_in_previous_run

    ev_days = np.sum(ev_disagg + possible_ev_cons, axis=1) > 0

    # remove ev in days where feeble ev boxes are present

    ev_days = block_user_with_feeble_ev_boxes(item_input_object, ev_days)

    ev_not_present = (np.sum(ev_days) < min_ev_days) or high_freq_ev

    ev_days = fill_arr_based_seq_val(ev_days, ev_days, 4, 0, 1)

    ev_days = fill_arr_based_seq_val(ev_days, ev_days, 4, 1, 0)

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    ev_days = ev_days[np.logical_not(vacation_days)]

    ev_usage_seq = find_seq(ev_days, np.zeros_like(ev_days), np.zeros_like(ev_days))

    # if high non ev days chunks are present, ev is not detected

    if ev_not_present or ((np.sum(ev_usage_seq[ev_usage_seq[:, seq_label] == 0, 3] > Cgbdisagg.DAYS_IN_MONTH * 2) > 2) and ev_absent_in_previous_run):
        possible_ev_cons = default_possible_ev_cons
        ev_amp = 0
    else:
        ev_amp = np.median(residual_cons[possible_ev_cons > 0])

    return possible_ev_cons, possible_ev_cons * ev_amp, ev_amp


def add_ev_boxes_to_maintain_cons_in_ev_tou(item_input_object, l1_bool, ev_disagg, valid_idx, residual_cons, median_amp,
                                            season, logger):
    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        l1_bool                   (bool)          : flag to determine the target charger type of current iteration
        ev_disagg                 (np.ndarray)    : Ev disagg output
        valid_idx                 (np.ndarray)    : initial ev detection points
        residual_cons             (np.ndarray)    : Residual data
        median_amp                (bool)          : ev amp
        season                    (np.ndarray)    : season data
        logger                    (logger)        : logger object

    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
        ts_level_ev_cons          (np.ndarray)    : ev estimation points
    """

    min_len, max_len, min_cons, max_cons, avg_dur, avg_freq, run_consitency_check, blocking_list, blocking_list2, median_amp = \
        prepare_ev_params_for_maintianing_consistency(item_input_object, l1_bool, ev_disagg, valid_idx, residual_cons,
                                                      median_amp, season, logger)

    if not run_consitency_check:
        return valid_idx, valid_idx * median_amp

    ev_tou = np.sum((valid_idx + ev_disagg) > 0, axis=0) > Cgbdisagg.DAYS_IN_MONTH

    samples = int(ev_disagg.shape[1] / Cgbdisagg.HRS_IN_DAY)

    ev_cons_1d = find_seq(((valid_idx + ev_disagg) > 0).flatten(), np.zeros_like((valid_idx > 0).flatten()),
                          np.zeros_like((valid_idx > 0).flatten()))

    min_dur = np.percentile(ev_cons_1d[ev_cons_1d[:, 0] > 0, 3], 10) - 3 * samples

    min_len = max(min_len, min_dur)

    new_cons = add_extra_ev_boxes_to_maintain_consistency(ev_tou, [min_len, max_len], min_cons, max_cons, avg_dur,
                                                          avg_freq,
                                                          blocking_list, item_input_object, valid_idx, residual_cons)

    possible_ev_cons = (new_cons + valid_idx)
    ts_level_ev_cons = (new_cons + valid_idx) * median_amp

    return possible_ev_cons, ts_level_ev_cons


def add_ev_boxes_to_maintain_con(item_input_object, l1_bool, ev_disagg, valid_idx, residual_cons, median_amp, season,
                                 logger):
    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        l1_bool                   (bool)          : flag to determine the target charger type of current iteration
        ev_disagg                 (np.ndarray)    : Ev disagg output
        valid_idx                 (np.ndarray)    : initial ev detection points
        residual_cons             (np.ndarray)    : Residual data
        median_amp                (bool)          : ev amp
        season                    (np.ndarray)    : season data
        logger                    (logger)        : logger object

    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
        ts_level_ev_cons          (np.ndarray)    : ev estimation points
    """

    min_len, max_len, min_cons, max_cons, avg_dur, avg_freq, run_consitency_check, blocking_list, blocking_list2, median_amp = \
        prepare_ev_params_for_maintianing_consistency(item_input_object, l1_bool, ev_disagg, valid_idx, residual_cons,
                                                      median_amp, season, logger)

    if not run_consitency_check:
        return valid_idx, valid_idx * median_amp

    tou = np.ones(len(valid_idx[0]))

    new_cons = add_extra_ev_boxes_to_maintain_consistency(tou, [min_len, max_len], min_cons, max_cons, avg_dur, avg_freq,
                                                          blocking_list, item_input_object, valid_idx, residual_cons)

    possible_ev_cons = (new_cons + valid_idx)
    ts_level_ev_cons = (new_cons + valid_idx) * median_amp

    return possible_ev_cons, ts_level_ev_cons


def add_ev_boxes_if_ev_detected_from_hybrid(ev_absent_in_prev_runs, item_input_object, l1_bool, ev_disagg, valid_idx,
                                            residual_cons, mode):
    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        l1_bool                   (bool)          : flag to determine the target charger type of current iteration
        ev_disagg                 (np.ndarray)    : Ev disagg output
        residual_cons             (np.ndarray)    : Residual data
        ev_absent_in_previous_run (bool)          : flag that represents whether l2 charger is present in previous disagg run
    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
        hybrid_ev_amp             (float)         : ev amplitude of ev added from hybrid module
    """

    # if ev charger type is l1 from true disagg, Ev l2 box detection wont run

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    default_possible_ev_cons = np.zeros_like(ev_disagg)
    default_amp = 0

    if l1_bool:
        return np.zeros_like(residual_cons), np.zeros_like(residual_cons), 0

    samples = int(ev_disagg.shape[1] / Cgbdisagg.HRS_IN_DAY)

    possible_ev_cons = copy.deepcopy(residual_cons)
    possible_ev_cons[valid_idx == 0] = 0
    ev_box_amp_list = np.zeros(valid_idx.size)

    # checking amplitude of all the ev boxes detected in previous step

    ev_1d = (possible_ev_cons > 0).flatten()
    possible_ev_cons_1d = (possible_ev_cons).flatten()
    ev_usage_seq = find_seq(ev_1d, np.zeros_like(ev_1d), np.zeros_like(ev_1d))

    m = 0
    for i in range(len(ev_usage_seq)):
        if ev_usage_seq[i, seq_label] == 1:
            ev_box_amp_list[m] = np.median(possible_ev_cons_1d[ev_usage_seq[i, seq_start]: ev_usage_seq[i, seq_end]])
            m = m + 1

    ev_box_amp_list = np.median(ev_box_amp_list[ev_box_amp_list > 0])

    # checking possible ev boxes based on ts level consumption and detected ev amplitude

    possible_ev_cons = residual_cons > ev_box_amp_list * 0.85
    possible_ev_cons = np.logical_and(possible_ev_cons, residual_cons < ev_box_amp_list * 1.3)

    params = get_ev_params_for_det(samples, np.sum(ev_disagg), item_input_object)

    min_len = params.get('min_len')
    max_len = params.get('max_len')
    ev_amp_cap = params.get('ev_amp_cap')

    possible_ev_cons[residual_cons > ev_amp_cap / samples] = 0

    ev_tou = np.sum(valid_idx, axis=0) > Cgbdisagg.DAYS_IN_MONTH * 0.5

    ev_tou = fill_neighbouring_values(ev_tou, 2 * samples, samples)

    # checking max len of ev boxes based on detected ev boxes

    if np.any(ev_usage_seq[:, seq_label] > 0):
        max_len = min(max_len, np.percentile(ev_usage_seq[ev_usage_seq[:, seq_label] > 0, 3], 85) + 2 * samples)

    possible_ev_cons = possible_ev_cons.flatten()
    ev_usage_seq = find_seq(possible_ev_cons, np.zeros_like(possible_ev_cons), np.zeros_like(possible_ev_cons))

    # Removing box consumption based on whether box length is in given range

    valid_boxes = np.logical_and(ev_usage_seq[:, seq_label] == 1, ev_usage_seq[:, seq_len] > max_len - 3 * samples)

    possible_ev_cons = fill_arr_based_seq_val_for_valid_boxes(ev_usage_seq, valid_boxes, possible_ev_cons, 1, 0)

    possible_ev_cons = np.reshape(possible_ev_cons, ev_disagg.shape)

    possible_ev_cons[:, np.logical_not(ev_tou)] = 0
    possible_ev_cons[valid_idx > 0] = 1

    ev_1d = (possible_ev_cons > 0).flatten()
    ev_usage_seq = find_seq(ev_1d, np.zeros_like(ev_1d), np.zeros_like(ev_1d), overnight=0)

    low_duration_tou = possible_ev_cons.sum(axis=0) > Cgbdisagg.DAYS_IN_MONTH
    low_duration_tou = np.where(low_duration_tou)[0]

    possible_ev_cons = possible_ev_cons.flatten()

    possible_ev_cons_copy = copy.deepcopy(possible_ev_cons)

    residual_cons_1d = residual_cons.flatten()

    eu_user = item_input_object.get("config").get("pilot_id") in PilotConstants.EU_PILOTS

    max_len = samples*12 * (eu_user) + samples*11 * (not eu_user)

    # Removing box consumption based on whether box length is in given range

    low_dur_box_thres = [min_len - 1]

    possible_ev_cons = remove_incorrect_ev_box_1(possible_ev_cons, ev_usage_seq, residual_cons_1d, low_duration_tou,
                                                 samples, min_len, max_len, low_dur_box_thres)

    possible_ev_cons = np.reshape(possible_ev_cons, ev_disagg.shape)

    low_duration_tou = possible_ev_cons.sum(axis=0) > 20

    low_duration_tou = fill_neighbouring_values(low_duration_tou, 1 * samples, samples)

    low_duration_tou = np.where(low_duration_tou)[0]

    possible_ev_cons = possible_ev_cons_copy

    ev_usage_seq = find_seq(possible_ev_cons, np.zeros_like(possible_ev_cons), np.zeros_like(possible_ev_cons))
    valid_boxes = np.logical_and(ev_usage_seq[:, seq_label] == 0, ev_usage_seq[:, seq_len] <= samples * 0.5)
    valid_boxes = valid_boxes * (samples > 1)
    possible_ev_cons = fill_arr_based_seq_val_for_valid_boxes(ev_usage_seq, valid_boxes, possible_ev_cons, 1, 1)

    low_ev_dur_thres = samples * 0.5

    if np.sum(possible_ev_cons.reshape(ev_disagg.shape).sum(axis=0) > 40) > 5 and samples == 1:
        low_ev_dur_thres = 1

    possible_ev_cons = remove_incorrect_ev_box(possible_ev_cons, ev_usage_seq, residual_cons_1d, low_duration_tou,
                                               samples, min_len, max_len, low_ev_dur_thres)

    possible_ev_cons = np.reshape(possible_ev_cons, ev_disagg.shape)
    possible_ev_cons[residual_cons > ev_amp_cap / samples] = 0

    amp = np.median(residual_cons[possible_ev_cons > 0])

    possible_ev_cons[residual_cons < amp * 0.7] = 0

    ev_cons_1d = possible_ev_cons.flatten()
    ev_cons_1d[np.logical_and(np.roll(ev_cons_1d, 1) == 0, np.roll(ev_cons_1d, -1) == 0)] = 0
    ev_cons_1d = np.reshape(ev_cons_1d, possible_ev_cons.shape)
    ev_cons_1d[:, low_duration_tou] = 1

    # checking frequency of ev boxes

    possible_ev_cons[ev_cons_1d == 0] = 0

    freq = get_freq(copy.deepcopy(valid_idx)) - 2

    temp_possible_ev_cons = np.sum(possible_ev_cons > 0, axis=1) > 0

    seq = find_seq(temp_possible_ev_cons, np.zeros_like(temp_possible_ev_cons), np.zeros_like(temp_possible_ev_cons))

    for i in range(1, len(seq)):

        if seq[i, seq_label] > 0 and seq[i - 1, 3] < freq:
            temp_possible_ev_cons[seq[i, seq_start]: seq[i, seq_end] + 1] = 0

    temp_possible_ev_cons[valid_idx.sum(axis=1) > 0] = 1

    possible_ev_cons[temp_possible_ev_cons == 0] = 0

    possible_ev_cons[np.sum(possible_ev_cons, axis=1) > (max_len + 3 * samples)] = 0

    ev_start_ts = np.roll(possible_ev_cons, 1, axis=1).astype(int) - possible_ev_cons.astype(int)

    ev_start_ts = np.sum(ev_start_ts == -1, axis=1)

    possible_ev_cons[ev_start_ts > 2] = 0

    ev_days = np.sum(possible_ev_cons, axis=1) > 0

    ev_days = block_user_with_feeble_ev_boxes(item_input_object, ev_days)

    possible_ev_cons[ev_days == 0] = 0

    # handling cases where sparse ev boxes is present

    days_seq = find_seq(ev_days, np.zeros_like(ev_days), np.zeros_like(ev_days))

    if len(days_seq) > 2:
        days_seq = days_seq[1: len(days_seq) - 1]

        non_ev_cons_seq = np.logical_and(days_seq[:, seq_label] == 0, days_seq[:, seq_len] > 100)

        if np.any(non_ev_cons_seq):
            val = np.where(non_ev_cons_seq)[0][-1]

            val = days_seq[val, 2]

            possible_ev_cons[:val] = 0

    thres = max(15, min(20, len(ev_days) * 0.05))

    if mode == 'mtd':
        thres = 0

    if (np.sum(possible_ev_cons) == 0 or np.sum(ev_days) < thres) and ev_absent_in_prev_runs:
        return default_possible_ev_cons, default_possible_ev_cons, default_amp

    amp = np.median(residual_cons[possible_ev_cons > 0])

    return possible_ev_cons, possible_ev_cons * amp, amp


def fill_neighbouring_values(ev_tou, duration, samples):
    """
    Extend EV tou in neighbouring points

    Parameters:
        ev_tou                  (np.ndarray)          : EV tou
        duration                (int)                 : window for ev extension
        samples                 (int)                 : samples in an hour
    Returns:
        ev_tou                  (np.ndarray)          : updated EV tou

    """

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    tou_seq = find_seq(ev_tou, np.zeros_like(ev_tou), np.zeros_like(ev_tou))
    for i in range(len(tou_seq)):
        if tou_seq[i, seq_label] > 0:
            ev_tou[get_index_array(tou_seq[i, seq_start] - duration, tou_seq[i, seq_start],
                                   Cgbdisagg.HRS_IN_DAY * samples)] = 1
            ev_tou[get_index_array(tou_seq[i, seq_end], tou_seq[i, seq_end] + duration,
                                   Cgbdisagg.HRS_IN_DAY * samples)] = 1

    return ev_tou


def remove_incorrect_ev_box(possible_ev_cons, ev_usage_seq, residual_cons_1d, low_duration_tou, samples, min_len,
                            max_len, low_ev_dur_thres):

    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        possible_ev_cons          (np.ndarray)    : ev detection points
        ev_usage_seq              (np.ndarray)    : ev box data
        residual_cons_1d          (np.ndarray)    : residual data
        low_duration_tou          (np.ndarray)    : time of data when low duration boxes can be added
        samples                   (int)           : samples in an hour
        min_len                   (int)           : min len of ev box
        max_len                   (int)           : max len of ev box
        low_dur_box_thres         (int)           : min len for low duration ev boxes
    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
    """

    seq_label = seq_config.SEQ_LABEL
    ev_usage_seq = ev_usage_seq[ev_usage_seq[:, seq_label] == 1]

    for i in range(len(ev_usage_seq)):

        if (ev_usage_seq[i, 3] < min_len or ev_usage_seq[i, 3] > max_len):

            if (ev_usage_seq[i, 3] > low_ev_dur_thres) and (ev_usage_seq[i, 3] < max_len) and \
                residual_cons_1d[ev_usage_seq[i, 1]: ev_usage_seq[i, 2] + 1].mean() > 3500/samples and \
                    ((ev_usage_seq[i, 1]%(samples * 24) in low_duration_tou) or (ev_usage_seq[i, 2]%(samples * 24)in low_duration_tou)):
                possible_ev_cons[ev_usage_seq[i, 1] : ev_usage_seq[i, 2] + 1] = 1

            else:
                possible_ev_cons[ev_usage_seq[i, 1]: ev_usage_seq[i, 2] + 1] = 0

        else:
            arr = np.arange(ev_usage_seq[i, 1], ev_usage_seq[i, 2] + 1)

            if len(arr) > 5:
                arr = arr[1:-1]

            if len(arr) > 0 and (np.percentile(residual_cons_1d[arr], 85) > 1.2*np.percentile(residual_cons_1d[arr], 10)):
                possible_ev_cons[ev_usage_seq[i, 1]: ev_usage_seq[i, 2] + 1] = 0

    return possible_ev_cons


def remove_incorrect_ev_box_1(possible_ev_cons, ev_usage_seq, residual_cons_1d, low_duration_tou, samples, min_len, max_len, low_dur_box_thres):

    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        possible_ev_cons          (np.ndarray)    : ev detection points
        ev_usage_seq              (np.ndarray)    : ev box data
        residual_cons_1d          (np.ndarray)    : residual data
        low_duration_tou          (np.ndarray)    : time of data when low duration boxes can be added
        samples                   (int)           : samples in an hour
        min_len                   (int)           : min len of ev box
        max_len                   (int)           : max len of ev box
        low_dur_box_thres         (int)           : min len for low duration ev boxes
    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
    """

    seq_label = seq_config.SEQ_LABEL

    ev_usage_seq = ev_usage_seq[ev_usage_seq[:, seq_label] == 1]

    for i in range(len(ev_usage_seq)):

        if (ev_usage_seq[i, 3] < min_len or ev_usage_seq[i, 3] > max_len):

            if ev_usage_seq[i, 3] in [low_dur_box_thres] and \
                    ((ev_usage_seq[i, 1] % (24 * samples) in low_duration_tou) or (
                            ev_usage_seq[i, 2] % (24 * samples) in low_duration_tou)):
                possible_ev_cons[ev_usage_seq[i, 1]: ev_usage_seq[i, 2] + 1] = 1
            else:
                possible_ev_cons[ev_usage_seq[i, 1]: ev_usage_seq[i, 2] + 1] = 0

        else:
            arr = np.arange(ev_usage_seq[i, 1], ev_usage_seq[i, 2] + 1)

            if len(arr) > 5:
                arr = arr[1:-1]

            if len(arr) > 0 and (
                    np.percentile(residual_cons_1d[arr], 85) > 1.2 * np.percentile(residual_cons_1d[arr], 10)):
                possible_ev_cons[ev_usage_seq[i, 1]: ev_usage_seq[i, 2] + 1] = 0

    return possible_ev_cons


def remove_incorrect_ev_box_2(possible_ev_cons, ev_usage_seq, residual_cons_1d, low_duration_tou, samples, min_len,
                              max_len, low_dur_box_thres):
    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        possible_ev_cons          (np.ndarray)    : ev detection points
        ev_usage_seq              (np.ndarray)    : ev box data
        residual_cons_1d          (np.ndarray)    : residual data
        low_duration_tou          (np.ndarray)    : time of data when low duration boxes can be added
        samples                   (int)           : samples in an hour
        min_len                   (int)           : min len of ev box
        max_len                   (int)           : max len of ev box
        low_dur_box_thres         (int)           : min len for low duration ev boxes
    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
    """

    seq_label = seq_config.SEQ_LABEL

    ev_usage_seq = ev_usage_seq[ev_usage_seq[:, seq_label] == 1]

    for idx in range(len(ev_usage_seq)):

        possible_ev_cons = \
            remove_incorrect_ev_box_for_each_seq(idx, possible_ev_cons, ev_usage_seq, residual_cons_1d, low_duration_tou,
                                                 samples, min_len, max_len, low_dur_box_thres)

    return possible_ev_cons


def remove_incorrect_ev_box_for_each_seq(seq_idx, possible_ev_cons, ev_usage_seq, residual_cons_1d, low_duration_tou, samples, min_len,
                                         max_len, low_dur_box_thres):
    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        seq_idx                   (int)           : index of target seq
        possible_ev_cons          (np.ndarray)    : ev detection points
        ev_usage_seq              (np.ndarray)    : ev box data
        residual_cons_1d          (np.ndarray)    : residual data
        low_duration_tou          (np.ndarray)    : time of data when low duration boxes can be added
        samples                   (int)           : samples in an hour
        min_len                   (int)           : min len of ev box
        max_len                   (int)           : max len of ev box
        low_dur_box_thres         (int)           : min len for low duration ev boxes
    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
    """

    arr = np.arange(ev_usage_seq[seq_idx, 1], ev_usage_seq[seq_idx, 2] + 1)

    if len(arr) > 5:
        arr = arr[1:-1]

    if (ev_usage_seq[seq_idx, 3] < min_len or ev_usage_seq[seq_idx, 3] > max_len):

        if ev_usage_seq[seq_idx, 3] in [low_dur_box_thres] and \
                ((ev_usage_seq[seq_idx, 1] % (24 * samples) in low_duration_tou) or
                 (ev_usage_seq[seq_idx, 2] % (24 * samples) in low_duration_tou)):

            if len(arr) > 2 and (np.percentile(residual_cons_1d[arr], 85) > 1.2 * np.percentile(residual_cons_1d[arr], 10)):
                possible_ev_cons[ev_usage_seq[seq_idx, 1]: ev_usage_seq[seq_idx, 2] + 1] = 0
            else:
                possible_ev_cons[ev_usage_seq[seq_idx, 1]: ev_usage_seq[seq_idx, 2] + 1] = 1
        else:
            possible_ev_cons[ev_usage_seq[seq_idx, 1]: ev_usage_seq[seq_idx, 2] + 1] = 0

    elif len(arr) > 0 and (np.percentile(residual_cons_1d[arr], 70) > 1.2*np.percentile(residual_cons_1d[arr], 10)):
        possible_ev_cons[ev_usage_seq[seq_idx, 1]: ev_usage_seq[seq_idx, 2] + 1] = 0

    return possible_ev_cons


def prepare_ev_params_for_maintianing_consistency(item_input_object, l1_bool, ev_disagg, valid_idx, residual_cons,
                                                  median_amp, season, logger):
    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        l1_bool                   (bool)          : flag to determine the target charger type of current iteration
        ev_disagg                 (np.ndarray)    : Ev disagg output
        valid_idx                 (np.ndarray)    : initial ev detection points
        residual_cons             (np.ndarray)    : Residual data
        median_amp                (bool)          : ev amp
        season                    (np.ndarray)    : season data
        logger                    (logger)        : logger object

    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
        hybrid_ev_amp             (float)         : ev amplitude of ev added from hybrid module
        low_duration_tou          (np.ndarray)    : time of data when low duration boxes can be added
        samples                   (int)           : samples in an hour
        min_len                   (int)           : min len of ev box
        max_len                   (int)           : max len of ev box
    """

    # initializing and fetching required inputs

    min_len = 0
    max_len = 0
    min_cons = 0
    max_cons = 0
    run_consitency_check = 0
    avg_dur = 0
    avg_freq = 0

    seq_label = seq_config.SEQ_LABEL
    seq_len = seq_config.SEQ_LEN

    blocking_list = np.zeros(2)
    blocking_list2 = np.zeros(2)

    if median_amp == 0 and np.any(ev_disagg > 0):
        median_amp = np.median(ev_disagg[ev_disagg > 0])

    samples = int(ev_disagg.shape[1] / Cgbdisagg.HRS_IN_DAY)

    run_seasonal_check = check_seasonal_check_flag(item_input_object, residual_cons)

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc = np.unique(bc_list)

    # check if detected ev is l1 or seasonal,
    # if yes, additional boxes are not added

    ev_residual = np.ones_like(ev_disagg)

    if run_seasonal_check and (np.sum(ev_disagg) == 0):
        ev_residual = eliminate_seasonal_ev_cases(item_input_object, vacation, valid_idx, samples,
                                                  copy.deepcopy(season), ev_disagg, valid_idx * median_amp, logger)

    ev_type = get_charger_type(item_input_object)

    if (ev_type == 1 or l1_bool) or (item_input_object.get("config").get('disagg_mode') == 'mtd') or (
            np.sum(ev_residual) == 0):
        return min_len, max_len, min_cons, max_cons, avg_dur, avg_freq, run_consitency_check, blocking_list, blocking_list2, median_amp

    # checking ev density in each of the billing cycles

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    arr = ((ev_disagg + valid_idx) > 0).flatten()

    seq = find_seq(arr, np.zeros_like(arr), np.zeros_like(arr))

    if np.sum(seq[:, seq_label] > 0) < 8:
        valid_idx[:, :] = 0

    if np.sum(valid_idx + ev_disagg) == 0:
        return min_len, max_len, min_cons, max_cons, avg_dur, avg_freq, run_consitency_check, blocking_list, blocking_list2, median_amp

    ev_cons = find_seq(((valid_idx + ev_disagg) > 0).flatten(), np.zeros_like((valid_idx > 0).flatten()),
                       np.zeros_like((valid_idx > 0).flatten()))

    avg_dur = np.mean(ev_cons[ev_cons[:, 0] > 0, 3])

    blocking_list = np.zeros_like(unique_bc)
    blocking_list2 = np.zeros_like(unique_bc)

    freq = np.zeros_like(unique_bc)

    # check whether feeble cons ev is present in a given billing cycle

    for i in range(len(unique_bc)):

        arr = ((valid_idx + ev_disagg)[bc_list == unique_bc[i]] > 0).flatten()

        if np.sum(arr) == 0:
            blocking_list[i] = 1
            blocking_list2[i] = 1
            continue

        factor = 1 - (vacation[bc_list == unique_bc[i]]).sum() / np.sum(bc_list == unique_bc[i])

        seq = find_seq(arr, np.zeros_like(arr), np.zeros_like(arr))

        if factor > 0:
            freq[i] = np.sum(seq[:, seq_label] > 0) / factor

        blocking_list[i] = bool(
            (np.sum(np.logical_and(seq[:, seq_label] > 0, seq[:, seq_len] > samples * 0.5)) < 6 * factor))

        blocking_list2[i] = bool(
            (np.sum(np.logical_and(seq[:, seq_label] > 0, seq[:, seq_len] > samples * 0.5)) < 2 * factor))

    seq = find_seq(blocking_list2, np.zeros_like(blocking_list), np.zeros_like(blocking_list), overnight=0).astype(int)

    recent_ev = seq[0, 0] == 1 and seq[0, 3] > 2

    min_cons = median_amp * 0.8
    max_cons = 20000 / samples

    # not adding additional ev boxes if all the billing cycle has feeble cons

    if not (np.all(blocking_list == 0) or np.all(blocking_list2 == 1)):
        avg_freq = np.mean(freq[blocking_list == 0])

        avg_freq = max(6, np.nan_to_num(avg_freq))

        params = get_ev_params_for_det(samples, ev_disagg, item_input_object)

        min_len = params.get('min_len_for_cons')
        max_len = params.get('max_len_for_cons')

        # not adding additional ev boxes if recent ev is detected

        recent_ev = get_recent_ev_flag(item_input_object, recent_ev)

        blocking_list = blocking_list * (1 - recent_ev)

        run_consitency_check = 1

    return min_len, max_len, min_cons, max_cons, avg_dur, avg_freq, run_consitency_check, blocking_list, blocking_list2, median_amp


def add_extra_ev_boxes_to_maintain_consistency(ev_tou, len_params, min_cons, max_cons, avg_dur, avg_freq,
                                               blocking_list, item_input_object, valid_idx, residual_cons):
    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        l1_bool                   (bool)          : flag to determine the target charger type of current iteration
        ev_disagg                 (np.ndarray)    : Ev disagg output
        residual_cons             (np.ndarray)    : Residual data
        ev_absent_in_previous_run (bool)          : flag that represents whether l2 charger is present in previous disagg run
    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
        hybrid_ev_amp             (float)         : ev amplitude of ev added from hybrid module
    """

    min_len = len_params[0]
    max_len = len_params[1]

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc = np.unique(bc_list)

    seq_label = seq_config.SEQ_LABEL

    new_cons = np.zeros_like(valid_idx)

    unique_bc = unique_bc[blocking_list > 0]
    blocking_list = blocking_list[blocking_list > 0]

    for idx in range(len(blocking_list)):

        valid_days = bc_list == unique_bc[idx]

        # Adding additional ev box to maintain billing cycle level consistency in ev output

        possible_cons = copy.deepcopy(residual_cons[valid_days])

        # adding boxes only with certain amplitude and which lies in ev tou window

        possible_cons = np.logical_and(possible_cons > min_cons, possible_cons < max_cons)

        possible_cons[vacation[valid_days] > 0] = 0

        possible_cons[:, ev_tou == 0] = 0

        possible_cons = possible_cons.flatten()

        ev_usage_seq = find_seq(possible_cons, np.zeros_like(possible_cons),
                                np.zeros_like(possible_cons), overnight=0)

        valid_boxes = np.logical_and(ev_usage_seq[:, 0] == 1,
                                     np.logical_or(ev_usage_seq[:, 3] < min_len, ev_usage_seq[:, 3] > max_len))

        possible_cons = fill_arr_based_seq_val_for_valid_boxes(ev_usage_seq, valid_boxes, possible_cons, 1, 0)

        possible_cons = possible_cons.reshape(residual_cons[valid_days].shape)

        for m in np.arange(len(possible_cons)):

            # if picked boxes is more than required, additional boxes are removed randomly

            ev_usage_seq = find_seq(possible_cons[m], np.zeros_like(possible_cons[m]),
                                    np.zeros_like(possible_cons[m]), overnight=0)

            ev_usage_seq = ev_usage_seq[ev_usage_seq[:, seq_label] > 0]

            seed = RandomState(random_gen_config.seed_value)

            remove_wh_frac = max(0, len(ev_usage_seq) - 2)

            if len(ev_usage_seq) == 0:
                continue

            remove_boxes = seed.choice(np.arange(len(ev_usage_seq)), remove_wh_frac, replace=False)

            if remove_wh_frac <= 0:
                continue

            # removing additional boxes

            possible_cons[m] = remove_extra_ev_boxes(remove_boxes, possible_cons[m], ev_usage_seq)

        possible_cons[valid_idx[valid_days] > 0] = 0

        possible_cons = possible_cons.flatten()

        ev_usage_seq = find_seq(possible_cons, np.zeros_like(possible_cons),
                                np.zeros_like(possible_cons), overnight=0)

        factor = 1 - (vacation[bc_list == unique_bc[idx]]).sum() / np.sum(bc_list == unique_bc[idx])

        # even if sufficient boxes are not available in original ev time of use, ev is added in other time of day

        freq = np.sum(ev_usage_seq[:, seq_label] > 0)

        if freq == 0 or factor == 0 or possible_cons.sum() == 0:
            continue

        updated_avg_dur_aft_box_add = np.mean(ev_usage_seq[ev_usage_seq[:, seq_label] > 0, 3])

        scaling_factor_based_on_ev_len = (max(0.2, min(5, (updated_avg_dur_aft_box_add / avg_dur))))

        if freq > avg_freq * factor / scaling_factor_based_on_ev_len:

            # if picked boxes is more than required, additional boxes are removed randomly

            ev_usage_seq = ev_usage_seq[ev_usage_seq[:, seq_label] > 0]

            seed = RandomState(random_gen_config.seed_value)

            remove_wh_frac = int(freq - (avg_freq * factor / scaling_factor_based_on_ev_len))

            remove_wh_frac = max(0, min(remove_wh_frac, len(ev_usage_seq)))

            remove_boxes = seed.choice(np.arange(len(ev_usage_seq)), remove_wh_frac, replace=False)

            if remove_wh_frac <= 0:
                continue

            possible_cons = remove_extra_ev_boxes(remove_boxes, possible_cons, ev_usage_seq)

        possible_cons = possible_cons.reshape(residual_cons[valid_days].shape)

        new_cons[valid_days] = possible_cons

    return new_cons


def remove_extra_ev_boxes(remove_boxes, possible_cons, ev_usage_seq):

    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        possible_ev_cons          (np.ndarray)    : ev detection points
        ev_usage_seq              (np.ndarray)    : ev box data
    Returns:
        possible_ev_cons          (np.ndarray)    : ev detection points
    """

    for k in range(len(remove_boxes)):
        possible_cons[ev_usage_seq[remove_boxes[k], 1]: ev_usage_seq[remove_boxes[k], 2] + 1] = 0

    return possible_cons
