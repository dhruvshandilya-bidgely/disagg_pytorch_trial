
"""
Author - Nisha Agarwal
Date - 4th April 2021
Detect box type consumption in residual data
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import seq_config

from python3.config.pilot_constants import PilotConstants

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.aer.raw_energy_itemization.residual_analysis.config.get_residual_config import get_residual_config


def box_detection(pilot, input_data, res_input_data, neg_res, min_amp, max_amp, min_len, max_len, detect_wh=0):

    """
    detect box type consumption in disagg residual data

    Parameters:
        pilot               (int)             : pilot id
        input_data          (np.ndarray)      : original input data
        res_input_data      (np.ndarray)      : disagg residual data
        neg_res             (np.ndarray)      : ts of negative residual
        min_amp             (int)             : box min amp
        max_amp             (int)             : box max amp
        min_len             (int)             : box min length
        max_len             (int)             : box max length

    Returns:
        updated_bool_arr    (np.ndarray)      : tou of detected boxes
        updated_cons_arr    (np.ndarray)      : estimation of detected boxes
        final_seq           (np.ndarray)      : contains params of all detected boxxes
    """

    config = get_residual_config().get('box_detection_config')

    min_len = int(min_len)
    max_len = int(max_len)

    total_samples = res_input_data.shape[1]

    length = res_input_data.shape[0]

    samples = int(total_samples/Cgbdisagg.HRS_IN_DAY)

    min_cons_for_twh_box_check = config.get('min_cons_for_twh_box_check')/samples

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    final_samples = total_samples + 2 * max_len

    final_bool_arr = np.zeros((length, final_samples))

    final_cons_arr = np.zeros((length, final_samples))

    # appending last and starting hours window to detect overnight boxes as well

    input_data_copy = np.hstack((res_input_data[:, total_samples - max_len:], res_input_data, res_input_data[:, :max_len]))
    neg_res_copy = np.hstack((neg_res[:, total_samples-max_len:], neg_res, neg_res[:, :max_len]))

    input_data_copy[:, :max_len] = np.roll(input_data_copy[:, :max_len], 1, axis=0)
    neg_res_copy[:, :max_len] = np.roll(neg_res_copy[:, :max_len], 1, axis=0)

    input_data_copy[:, max_len + total_samples:] = np.roll(input_data_copy[:, max_len + total_samples:], -1, axis=0)
    neg_res_copy[:, max_len + total_samples:] = np.roll(neg_res_copy[:, max_len + total_samples:], -1, axis=0)

    final_inc_arr = input_data_copy - np.roll(input_data_copy, 1, axis=1)
    final_dec_arr = input_data_copy - np.roll(input_data_copy, -1, axis=1)

    final_dec_arr = np.roll(final_dec_arr, 1, axis=1)

    roll_arr = np.roll(input_data_copy, 0, axis=0)

    length = len(res_input_data)

    # iterating over all combination of timestamp to find activity boxes of various amplitude and length

    for i in range(1, final_samples-min_len):

        for j in range(max_len-1, min_len-2, -1):

            start_idx = i
            end_idx = start_idx + j + 1

            if end_idx > final_samples-1:
                continue

            strt_arr = final_inc_arr[:, start_idx]
            end_arr = final_dec_arr[:, end_idx]

            bool_arr = np.zeros((length, j + 1))
            cons_arr = np.zeros((length, j + 1))

            tmp_bool_arr = find_valid_consistent_boxes(i, j, input_data_copy, samples, final_inc_arr, final_dec_arr, pilot, max_len, roll_arr, detect_wh)

            tmp_bool_arr = np.logical_and(tmp_bool_arr, strt_arr > min_amp)
            tmp_bool_arr = np.logical_and(tmp_bool_arr, end_arr > min_amp)

            tmp_bool_arr = np.logical_and(tmp_bool_arr, end_arr <= max_amp)
            tmp_bool_arr = np.logical_and(tmp_bool_arr, strt_arr <= max_amp)

            tmp_bool_arr = np.logical_and(tmp_bool_arr, final_bool_arr[:, start_idx-1] == 0)

            if detect_wh == 0:
                tmp_bool_arr[strt_arr < min_cons_for_twh_box_check] = \
                    np.logical_and(tmp_bool_arr, np.logical_not(neg_res_copy[:, start_idx-1]))[strt_arr < min_cons_for_twh_box_check]
                tmp_bool_arr[strt_arr < min_cons_for_twh_box_check] = \
                    np.logical_and(tmp_bool_arr, np.logical_not(neg_res_copy[:, end_idx]))[strt_arr < min_cons_for_twh_box_check]

            bool_arr[tmp_bool_arr] = 1
            cons_arr[tmp_bool_arr] = strt_arr[tmp_bool_arr][:, None]

            final_bool_arr[tmp_bool_arr, start_idx:end_idx] = 1
            final_cons_arr[tmp_bool_arr, start_idx:end_idx] = strt_arr[tmp_bool_arr][:, None]

    final_cons_arr, final_bool_arr = postprocess_box_cons(final_cons_arr, input_data_copy, final_bool_arr, detect_wh)

    updated_bool_arr = final_bool_arr[:, max_len:max_len+total_samples]
    updated_cons_arr = final_cons_arr[:, max_len:max_len+total_samples]

    updated_bool_arr[1:, :max_len] = updated_bool_arr[1:, :max_len] + final_bool_arr[:length-1, total_samples+max_len:]
    updated_cons_arr[1:, :max_len] = np.maximum(updated_cons_arr[1:, :max_len], final_cons_arr[:length-1, total_samples+max_len:])

    updated_bool_arr[:length-1, total_samples-max_len:] = updated_bool_arr[:length-1, total_samples-max_len:] + final_bool_arr[1:, :max_len]
    updated_cons_arr[:length-1, total_samples-max_len:] = np.maximum(updated_cons_arr[:length-1, total_samples-max_len:], final_cons_arr[1:, :max_len])

    updated_bool_arr = np.fmin(1, updated_bool_arr)

    final_bool_arr_1d = np.reshape(updated_bool_arr, updated_bool_arr.size)

    final_bool_arr_1d_seq = find_seq(final_bool_arr_1d, np.zeros(res_input_data.size), np.zeros(res_input_data.size))

    final_bool_arr_1d_seq[final_bool_arr_1d_seq[:, seq_len] < min_len, 0] = 1

    final_bool_arr_1d_seq = final_bool_arr_1d_seq.astype(int)

    for i in range(len(final_bool_arr_1d_seq)):
        if final_bool_arr_1d_seq[i, seq_label]:
            final_bool_arr_1d[final_bool_arr_1d_seq[i, seq_start]: final_bool_arr_1d_seq[i, seq_end] + 1] = 1

    final_bool_arr_1d_seq = find_seq(final_bool_arr_1d, np.zeros(res_input_data.size), np.zeros(res_input_data.size))

    final_bool_arr_1d_seq = final_bool_arr_1d_seq.astype(int)

    final_bool_arr_1d_seq[final_bool_arr_1d_seq[:, seq_len] > max_len, 0] = 0

    for i in range(len(final_bool_arr_1d_seq)):
        if not final_bool_arr_1d_seq[i, seq_label]:
            final_bool_arr_1d[final_bool_arr_1d_seq[i, seq_start]: final_bool_arr_1d_seq[i, seq_end] + 1] = 0

    updated_bool_arr = final_bool_arr_1d.reshape(res_input_data.shape)
    updated_cons_arr[updated_bool_arr == 0] = 0

    updated_cons_arr[updated_cons_arr < np.percentile(input_data, 10, axis=1)[:, None]] = 0
    updated_bool_arr[updated_cons_arr == 0] = 0

    final_seq = find_seq(final_bool_arr_1d, np.zeros(res_input_data.size), np.zeros(res_input_data.size), overnight=0)

    final_seq = np.hstack((final_seq, np.zeros((len(final_seq), 1))))

    final_cons = np.reshape(updated_cons_arr, res_input_data.size)

    for i in range(len(final_seq)):
        if final_seq[i, seq_label]:
            final_seq[i, 4] = np.mean(final_cons[int(final_seq[i, seq_start]): int(final_seq[i, seq_end] + 1)])

    return updated_bool_arr, updated_cons_arr, final_seq


def postprocess_box_cons(final_cons_arr, input_data_copy, final_bool_arr, detect_wh):

    """
    postprocessing for estimated box  type consumption

    Parameters:
        final_cons_arr              (np.ndarray)    : detected box cons
        input_data_copy             (np.ndarray)    : input data
        final_bool_arr              (np.ndarray)    : detected box tou
        detect_wh                   (int)           : bool to check whether boxes are estimated for wh conns

    Returns:
        final_cons_arr              (np.ndarray)    : detected box cons
        final_bool_arr              (np.ndarray)    : detected box tou
    """

    if detect_wh == 1:
        final_cons_arr[final_cons_arr > 0] = np.maximum(final_cons_arr[final_cons_arr > 0], input_data_copy[final_cons_arr > 0])
        final_cons_arr = np.minimum(input_data_copy, final_cons_arr)
        final_bool_arr[final_cons_arr == 0] = 0

    return final_cons_arr, final_bool_arr


def find_valid_consistent_boxes(i, j, input_data_copy, samples_per_hour, final_inc_arr, final_dec_arr, pilot, max_len, roll_arr, detect_wh):

    """
    check valid boxes

    Parameters:
        start_idx              (int)           : Dict containing all inputs
        end_idx                (int)           : tou of timed signature
        input_data_copy        (np.ndarray)    : input data
        samples_per_hour       (int)           : samples in an hour
        final_dec_arr          (np.ndarray)    : starting edge cons delta
        final_inc_arr          (np.ndarray)    : ending edge cons delta
        pilot                  (int)           : pilot id
        max_len                (int)           : max length of the boxes

    Returns:
        tmp_bool_arr           (np.ndarray)    : array containing, the days on which valid boxes were detected
    """

    samples = int(input_data_copy.shape[1] / Cgbdisagg.HRS_IN_DAY)

    samples = samples_per_hour * (detect_wh == 1) + samples * (detect_wh != 1)

    config = get_residual_config().get('box_detection_config')

    min_frac_cons_points_required = config.get('min_frac_cons_points_required')
    high_amp_box_thres = config.get('high_amp_box_thres')/samples

    start_idx = i
    end_idx = start_idx + j + 1

    strt_arr = final_inc_arr[:, start_idx]
    end_arr = final_dec_arr[:, end_idx]

    diff1 = get_delta_thres(i, samples, samples_per_hour, strt_arr, pilot, max_len, detect_wh)

    same_delta = np.abs(strt_arr - end_arr) <= diff1

    same_delta[strt_arr > high_amp_box_thres] = np.abs(strt_arr[strt_arr > high_amp_box_thres] - end_arr[strt_arr > high_amp_box_thres]) <= 3000 / samples
    same_delta[end_arr > high_amp_box_thres] = np.logical_or(same_delta[end_arr > high_amp_box_thres],
                                                             np.abs(strt_arr[end_arr > high_amp_box_thres] - end_arr[end_arr > high_amp_box_thres]) <= 3000 / samples)

    if j >= max(2, samples_per_hour):

        if detect_wh == 0 and j <= 3 * samples_per_hour:
            count_arr = np.sum(input_data_copy[:, start_idx:end_idx] >
                               0.7 * (input_data_copy[:, start_idx - 1] / 2 +
                                      input_data_copy[:, end_idx] / 2)[:, None], axis=1)

            consistent_arr = count_arr > j * min_frac_cons_points_required

            count_arr = np.sum(input_data_copy[:, start_idx:end_idx] >
                               0.7 * (input_data_copy[:, start_idx] / 2 +
                                      input_data_copy[:, end_idx - 1] / 2)[:, None], axis=1)

            consistent_arr3 = count_arr > j * min_frac_cons_points_required

            count_arr = np.sum(input_data_copy[:, start_idx:end_idx] <
                               3 * (input_data_copy[:, start_idx + 1] / 2 +
                                    input_data_copy[:, end_idx - 1] / 2)[:, None], axis=1)

            consistent_arr1 = count_arr > j * min_frac_cons_points_required

            count_arr = np.sum(np.abs(final_inc_arr[:, start_idx + 1:end_idx - 1]) < 0.5 * strt_arr[:, None], axis=1)

            if j > 4 * samples_per_hour:
                consistent_arr2 = np.fmax(1, count_arr > j * 0.85)
            else:
                consistent_arr2 = np.logical_or(count_arr > j * 0.85, np.abs(strt_arr) > 2800 / samples_per_hour)

                if (pilot in PilotConstants.TIMED_WH_PILOTS) and ((i - max_len) % (samples_per_hour * 24) < 4 * samples_per_hour):
                    consistent_arr2 = np.logical_or(count_arr > j * 0.85, np.abs(strt_arr) > 1200 / samples_per_hour)

            tmp_bool_arr = np.logical_and(np.logical_and(np.logical_and(consistent_arr2, consistent_arr1),
                                                         np.logical_and(consistent_arr3, consistent_arr)), same_delta)

        elif detect_wh > 0 and j <= 3 * samples_per_hour:
            count_arr = np.sum((input_data_copy - roll_arr)[:, start_idx:end_idx], axis=1) < 500 / samples_per_hour

            consistent_arr = np.any(input_data_copy[:, start_idx:end_idx], axis=1) > 0

            tmp_bool_arr = np.logical_and(count_arr, consistent_arr)

        else:
            consistent_arr = np.all(input_data_copy[:, start_idx:end_idx] >
                                    0.7 * (input_data_copy[:, start_idx - 1] / 2 +
                                           input_data_copy[:, end_idx] / 2)[:, None], axis=1)

            consistent_arr3 = np.all(input_data_copy[:, start_idx:end_idx] >
                                     0.7 * (input_data_copy[:, start_idx] / 2 +
                                            input_data_copy[:, end_idx - 1] / 2)[:, None], axis=1)

            consistent_arr2 = np.sum(input_data_copy[:, start_idx:end_idx] <
                                     3 * (input_data_copy[:, start_idx + 1] / 2 +
                                          input_data_copy[:, end_idx - 1] / 2)[:, None], axis=1)

            tmp_bool_arr = np.logical_and(np.logical_and(consistent_arr2,
                                                         np.logical_and(consistent_arr3, consistent_arr)), same_delta)

    else:
        tmp_bool_arr = same_delta

    return tmp_bool_arr


def get_delta_thres(index, samples, samples_per_hour, strt_arr, pilot, max_len, detect_wh):

    """
    estimate the  amount of wh diff allowed in the start and end index of a box type activity

    Parameters:
        index                  (int)           : index of the count box
        samples_per_hour       (int)           : samples in an hour
        strt_arr               (np.ndarray)    : list of starting index of the boxes
        pilot                  (int)           : pilot id
        max_len                (int)           : max length of the boxes
        detect_wh              (bool)          : true if box detection is being done for wh

    Returns:
        diff                  (np.ndarray)     : array containing diff allowed in the edges of the boes
    """

    diff = np.fmax((1000 / samples)  * (detect_wh == 0) + (4000/samples_per_hour) * (detect_wh > 0), strt_arr * 0.1)

    if (pilot in PilotConstants.TIMED_WH_PILOTS) and ((index - max_len) % (samples * Cgbdisagg.HRS_IN_DAY) < 4 * samples):

        diff = (1300/samples) * (detect_wh == 0) + (4000/samples_per_hour) * (detect_wh > 0)
        diff = np.fmax(diff, strt_arr * 0.1)

    return diff
