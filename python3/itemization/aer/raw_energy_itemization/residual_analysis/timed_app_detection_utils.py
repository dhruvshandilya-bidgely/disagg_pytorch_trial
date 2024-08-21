"""
Author - Nisha Agarwal
Date - 17th Feb 2021
Detect timed appliance in residual data
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.residual_analysis.config.get_residual_config import get_residual_config


def fetch_required_inputs(item_input_object, item_output_object):

    """
    fetching required inputs for timed signature detection

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs

    Returns:
        season_label              (np.ndarray)    : season label
        samples_per_hour          (int)           : samples in an hour
        twh_cons                  (np.ndarray)    : twh disagg output
        pp_cons                   (np.ndarray)    : pp disagg output
        positive_residual         (np.ndarray)    : disagg residual
    """

    season_label = item_output_object.get("season")
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")

    positive_residual =  item_output_object.get("positive_residual")

    twh_cons = np.zeros(positive_residual.shape)
    pp_cons = np.zeros(positive_residual.shape)
    output_data = item_input_object.get("item_input_params").get("output_data")

    app_list = np.array(item_input_object.get("item_input_params").get("app_list"))
    wh_index = np.where(app_list == 'wh')[0]
    pp_index = np.where(app_list == 'pp')[0]

    if len(wh_index) and item_input_object.get("item_input_params").get("timed_wh_user"):
        twh_cons = output_data[wh_index[0] + 1, :, :]
        twh_cons = np.nan_to_num(twh_cons)

    if len(pp_index):
        pp_cons = output_data[pp_index[0] + 1, :, :]
        pp_cons = np.nan_to_num(pp_cons)

    return season_label, samples_per_hour, twh_cons, pp_cons, positive_residual


def rolling_func(arr, window, perc=40):

    """

    calculated rolling percentile array

    Parameters:
        arr         (np.ndarray)          : target array
        window      (int)                 : target window

    Returns:
        arr2        (np.ndarray)          : calculated array
    """

    updates_arr = np.zeros_like(arr)
    flag = 1

    for i in range(window, len(arr) - window, 1):

        if len(arr[i - window:i + window]) == 0:
            return arr, 0

        updates_arr[i] = np.percentile(arr[i - window:i + window], perc)

    return updates_arr, flag


def eliminate_artifacts_fp_cases(input_data, day_start, day_end, samples_per_hour, valid_tou):

    """
    elimiate fp cases that might occur due to false artifacts in the input data

    Parameters:
        input_data                (np.ndarray)    : input data
        day_start                 (np.ndarray)    : start indexes
        day_end                   (np.ndarray)    : end indexes
        samples_per_hour          (int)           : samples in an hour
        valid_tou                 (np.ndarray)    : tou of timed signature

    Returns:
        day_start                 (np.ndarray)    : updated start indexes
        day_end                   (np.ndarray)    : updated end indexes
        valid_tou                 (np.ndarray)    : updated tou of timed signature
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    config = get_residual_config().get('timed_app_det_config')

    min_days_frac_to_remove_fp_cases = config.get('min_days_frac_to_remove_fp_cases')

    invalid_points = np.sum(input_data == 0, axis=0) > min_days_frac_to_remove_fp_cases*len(input_data)
    invalid_seq = find_seq(invalid_points, np.zeros_like(invalid_points), np.zeros_like(invalid_points))

    if np.any(invalid_seq[:, seq_start]):
        invalid_seq = invalid_seq[invalid_seq[:, seq_start] == 1]
        for i in range(len(invalid_seq)):

            if invalid_seq[i, seq_len] <= samples_per_hour:
                start_seq = invalid_seq[i, seq_start]
                day_end[day_start == start_seq+1] = 0
                day_start[day_start == start_seq+1] = 0

                end_seq = invalid_seq[i, seq_end]
                day_start[day_end == end_seq-1] = 0
                day_end[day_end == end_seq-1] = 0

        if np.all(day_start == 0) and np.all(day_end == 0):
            valid_tou[:, :] = 0

    return valid_tou, day_start, day_end


def extend_signature_for_low_duration_intervals(index, valid_tou, day_start, day_end, time_sig_days_seq, input_data):

    """
    if the non timed sig seq is found, and the duration is less than 10 days, it is considered for timed detected

    Parameters:
        index                     (int)           : index of thee current seq of timed sig
        valid_tou                 (np.ndarray)    : tou of timed signature
        day_start                 (np.ndarray)    : start indexes
        day_end                   (np.ndarray)    : end indexes
        time_sig_days_seq         (np.ndarray)    : all seq of timed sig
        input_data                (np.ndarray)    : input data

    Returns:
        day_start                 (np.ndarray)    : updated start indexes
        day_end                   (np.ndarray)    : updated end indexes
        valid_tou                 (np.ndarray)    : updated tou of timed signature
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    samples_per_hour = int(valid_tou.shape[1]/Cgbdisagg.HRS_IN_DAY)

    start = day_start[time_sig_days_seq[index, seq_start] - 1]
    end = day_end[time_sig_days_seq[index, seq_start] - 1] + 1

    idx_arr = get_index_array(start, end, len(input_data[0]))

    amp = int(np.mean(input_data[time_sig_days_seq[index - 1, 1]:time_sig_days_seq[index - 1, 2] + 1, idx_arr]))

    amp_range = np.arange(amp - 500 / samples_per_hour, amp + 500 / samples_per_hour)

    valid_days_tmp = np.isin(
        np.mean(input_data[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1, idx_arr], axis=1).astype(int),
        amp_range)

    valid_days_tmp[:] = 1

    valid_ts = np.zeros_like(input_data)
    valid_ts[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1, idx_arr] = 1

    valid_ts[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1][np.logical_not(valid_days_tmp)] = 0

    valid_tou[:, idx_arr] = valid_tou[:, idx_arr] + valid_ts[:, idx_arr]

    day_start[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1] = start
    day_end[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1] = end

    return valid_tou, valid_ts, day_start, day_end


def check_consistency_to_extend_signature(input_data, index, window, idx_arr, median_amp, day_cons_thres, cons_thres):

    """
    # Extended timed signature edges to the regions where edges are not clear,
    # but there are chances of timed appliance being present
    # this helps in providing pp output at the regions with heavy hvac

    Parameters:
        input_data                (np.ndarray)    : input data
        index                     (int)           : index of thee current seq of timed sig
        window                    (np.ndarray)    : length of window, for which consistency is being calculated
        idx_arr                   (np.ndarray)    : index array of detected timestamp

    Returns:
        extention_flag            (bool)          : input data
    """

    samples_per_hour = int(input_data.shape[1]/24)

    # consistency along the days

    consistent = np.percentile(input_data[index:index + window, idx_arr], 15, axis=1)

    if not np.any(consistent != 0):
        return 0

    consistent = np.percentile(consistent[consistent != 0], 25)

    # consistency along the hours of usage

    consistent2 = np.percentile(input_data[index:index + window, idx_arr], 15, axis=0)
    consistent2[consistent2 < median_amp * 0.4] = 0

    consistent2, flag = rolling_func(consistent2, int(len(idx_arr) * 0.8 / 2))

    if flag == 0:
        return 0

    consistent2 = int(np.any(consistent2 > median_amp * day_cons_thres))

    extension_flag = (consistent > median_amp * cons_thres)
    extension_flag = extension_flag and \
                     np.nan_to_num(np.sum(np.logical_and(input_data[index:index + window, idx_arr] < 250 / samples_per_hour,
                                                         input_data[index:index + window, idx_arr] > 0 / samples_per_hour))
                                   / np.size(input_data[index:index + window, idx_arr])) < 0.1
    extension_flag = extension_flag or consistent2

    return extension_flag


def extend_sigature_for_mid_duration_intervals(index, valid_tou, day_start, day_end, time_sig_days_seq, input_data):

    """
    if the non timed sig seq is found, and the duration is less than 40 days,
    it is considered for timed detected under certain conditions

    Parameters:
        index                     (int)           : index of thee current seq of timed sig
        valid_tou                 (np.ndarray)    : tou of timed signature
        day_start                 (np.ndarray)    : start indexes
        day_end                   (np.ndarray)    : end indexes
        time_sig_days_seq         (np.ndarray)    : all seq of timed sig
        input_data                (np.ndarray)    : input data

    Returns:
        day_start                 (np.ndarray)    : updated start indexes
        day_end                   (np.ndarray)    : updated end indexes
        valid_tou                 (np.ndarray)    : updated tou of timed signature
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    # checking non timed segment before timed signature segment

    samples_per_hour = int(valid_tou.shape[1]/Cgbdisagg.HRS_IN_DAY)

    start = day_start[time_sig_days_seq[index, seq_start] - 1]
    end = day_end[time_sig_days_seq[index, seq_start] - 1] + 1

    idx_arr = get_index_array(start, end, len(input_data[0]))

    amp = int(np.mean(input_data[time_sig_days_seq[index - 1, seq_start]:time_sig_days_seq[index - 1, seq_end] + 1, idx_arr]))

    amp_range = np.arange(amp - 500 / samples_per_hour, amp + 500 / samples_per_hour)

    valid_days_tmp = np.isin(
        np.mean(input_data[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1, idx_arr], axis=1).astype(int), amp_range)

    valid_ts = np.zeros_like(input_data)
    valid_ts[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1, idx_arr] = 1

    valid_ts[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1][np.logical_not(valid_days_tmp)] = 0

    valid_tou[:, idx_arr] = valid_tou[:, idx_arr] + valid_ts[:, idx_arr]

    # checking non timed segment after timed signature segment

    start = day_start[time_sig_days_seq[index, seq_end] + 1]
    end = day_end[time_sig_days_seq[index, seq_end] + 1] + 1

    amp = np.mean(input_data[time_sig_days_seq[index + 1, seq_start]:time_sig_days_seq[index + 1, seq_end] + 1, idx_arr])

    amp_range = np.arange(amp - 200 / samples_per_hour, amp + 200 / samples_per_hour)

    valid_days_tmp = np.isin(
        np.mean(input_data[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1, idx_arr], axis=1).astype(int), amp_range)

    valid_ts = np.zeros_like(input_data)
    valid_ts[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1, idx_arr] = 1

    valid_ts[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1][np.logical_not(valid_days_tmp)] = 0

    valid_tou[:, idx_arr] = valid_tou[:, idx_arr] + valid_ts[:, idx_arr]

    day_start[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1] = start
    day_end[time_sig_days_seq[index, seq_start]:time_sig_days_seq[index, seq_end] + 1] = end

    return valid_tou, valid_ts, day_start, day_end


def eliminate_nonconsistent_app(updated_valid_tou, samples_per_hour):

    """
    eliminate non consistent pp cases

    Parameters:
        updated_valid_tou         (np.ndarray)    : tou of timed signature
        samples_per_hour          (int)           : samples in an hour

    Returns:
        updated_valid_tou         (np.ndarray)    : tou of timed signature
        non_pp                    (int)           : if this flag is true, siganture is not alloted to pp
    """

    end = np.ones(len(updated_valid_tou)) * -1

    for i in range(len(updated_valid_tou)):
        if np.sum(updated_valid_tou[i]):
            if updated_valid_tou[i, 0] > 0:
                end[i] = np.where(updated_valid_tou[i] == 0)[0][0] - 1
            else:
                end[i] = np.where(updated_valid_tou[i] > 0)[0][-1]

    non_pp = 0

    if np.any(end > -1):

        frac = np.sum(np.abs(np.diff(end[end > -1])[1:]) > 3) / np.sum(end > -1)

        if frac > 0.7:
            updated_valid_tou[:, :] = 0

        frac = np.sum(np.abs(np.diff(end[end > -1])[1:]) >= 2) / np.sum(end > -1)

        if frac > 0.2:
            non_pp = 1

        if samples_per_hour == 1 and np.sum(updated_valid_tou.sum(axis=0) > 0) < 5:

            frac = np.sum(np.abs(np.diff(end[end > -1])[1:]) >= 1) / np.sum(end > -1)

            if frac > 0.2:
                non_pp = 1

    return updated_valid_tou, non_pp


def update_estimation_after_extension(index, estimation, schedule, valid_tou, day_index, time_sig_days_seq,
                                      tmp_schedule, input_data, median_amp, extension_flag):

    """
    # Extended timed signature edges to the regions where edges are not clear,
    # but there are chances of timed appliance being present
    # this helps in providing pp output at the regions with heavy hvac

    Parameters:
        index                     (int)           : index of thee current seq of timed sig
        estimation                (np.ndarray)    : estimation of timed signature
        schedule                  (np.ndarray)    : current timed schedule
        valid_tou                 (np.ndarray)    : tou of timed signature
        day_index                 (np.ndarray)    : overall tou
        time_sig_days_seq         (np.ndarray)    : all seq of timed sig
        tmp_schedule              (np.ndarray)    : start and end of current window
        input_data                (np.ndarray)    : input data
        median_amp                (int)           : median consumtion of the current schdule
        extension_flag            (bool)          : boolean for whether to extend timed signature

    Returns:
        estimation                (np.ndarray)    : updated estimation data
        day_start                 (np.ndarray)    : updated start indexes
        day_end                   (np.ndarray)    : updated end indexes
        valid_tou                 (np.ndarray)    : updated tou of timed signature
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    day_start = schedule[0]
    day_end = schedule[1]

    samples = int(estimation.shape[1]/Cgbdisagg.HRS_IN_DAY)

    if extension_flag:

        tmp_start = tmp_schedule[0]
        tmp_end = tmp_schedule[1]

        window = 15

        idx_arr = get_index_array(tmp_start, tmp_end, len(input_data[0]))

        valid_ts = np.zeros_like(valid_tou)
        valid_ts[day_index: day_index + window] = 1

        valid_tou[:, idx_arr] = valid_tou[:, idx_arr] + valid_ts[:, idx_arr]

        est_data = estimation[get_index_array(time_sig_days_seq[index, seq_start],
                                              time_sig_days_seq[index, seq_end], len(valid_tou))]

        if np.any(est_data > 300/samples):

            estimation[day_index: day_index + window] = np.mean(est_data[est_data > 300/samples])

            day_start[day_index:day_index + window] = tmp_start
            day_end[day_index:day_index + window] = tmp_end

            estimation[day_index: day_index + window][input_data[day_index: day_index + window] < 0.6 * median_amp] = 2

    return estimation, day_start, day_end, valid_tou


def get_valid_schedules(features, buffer, box_seq, valid_tou, valid_tou_tmp, box_cons, box_cons_tmp, valid_seq):

    """
    determine whether a given timed schedule should be considered as a timed signature

    Parameters:
        features                  (np.ndarray)    : timed features
        box_seq                   (np.ndarray)    : detected box type activity
        valid_tou                 (np.ndarray)    : timed sig tou
        valid_tou_tmp             (np.ndarray)    : timed sig tou of each seq
        box_cons                  (np.ndarray)    : timed sig cons
        box_cons_tmp              (np.ndarray)    : timed sig cons of each seq

    Returns:
        valid_tou                 (np.ndarray)    : overall tou
    """

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    samples_per_hour = int(box_cons.shape[1]/24)
    total_samples = box_cons.shape[1]

    start = np.where(features == np.max(features))[0][0]
    end = np.where(features == np.max(features))[1][0]

    if samples_per_hour > 1:
        valid_seq[np.logical_and(
            np.isin(box_seq[:, seq_start] % (total_samples), get_index_array(start - buffer, start + buffer, total_samples)),
            np.isin(box_seq[:, seq_len], get_index_array(end - buffer, end + buffer, total_samples)))] = 1
    else:
        valid_seq[np.logical_and(box_seq[:, seq_start] % (total_samples) == start, box_seq[:, seq_len] == end)] = 1

    for i in range(len(valid_seq)):
        if box_seq[i, seq_label] and valid_seq[i]:
            valid_tou_tmp[box_seq[i, seq_start]: box_seq[i, seq_end] + 1] = 1

    # checks whether the chunk of consumption can be a part of timed signature

    # variable end denotes the length of activity

    if end == 2 and samples_per_hour == 1 and np.sum(valid_tou_tmp) and \
            (np.percentile(box_cons_tmp[valid_tou_tmp > 0], 90) - np.percentile(box_cons_tmp[valid_tou_tmp > 0],
                                                                                15)) < 1000 / samples_per_hour:
        valid_tou = valid_tou + valid_tou_tmp

    quantile_dist = np.percentile(box_cons_tmp[valid_tou_tmp > 0], 80) - np.percentile(box_cons_tmp[valid_tou_tmp > 0], 20)

    if end > 2 and samples_per_hour == 1 and np.sum(valid_tou_tmp) and (quantile_dist) < 2000 / samples_per_hour:

        valid_tou_tmp = prepare_timed_sig_tou(valid_tou_tmp, valid_tou, box_cons)

        valid_tou = valid_tou + valid_tou_tmp

    if samples_per_hour > 1 and np.sum(valid_tou_tmp) and (quantile_dist) < 10000 / samples_per_hour:
        valid_tou_tmp = prepare_timed_sig_tou(valid_tou_tmp, valid_tou, box_cons)

        valid_tou = valid_tou + valid_tou_tmp

    return valid_tou


def prepare_timed_sig_tou(valid_tou_tmp, valid_tou, box_cons):

    """
    update timed signature tou based on new detected band

    Parameters:
        valid_tou_tmp             (np.ndarray)    : tou of new timed band
        valid_tou                 (np.ndarray)    : overall tou
        box_cons                  (np.ndaaray)    : consumption of detected activity

    Returns:
        valid_tou_tmp             (np.ndarray)    : tou of new timed band
    """

    tmp_seq = find_seq((valid_tou_tmp.reshape(box_cons.shape) > 0).sum(axis=1) > 0, np.zeros(len(box_cons)),
                       np.zeros(len(box_cons)))

    if np.any(np.logical_and(tmp_seq[:, 0] == 1, tmp_seq[:, 3] < 2)):
        tmp_seq = tmp_seq[np.logical_and(tmp_seq[:, 0] == 1, tmp_seq[:, 3] < 2)]

        valid_tou_tmp = valid_tou_tmp.reshape(box_cons.shape)

        for m in range(len(tmp_seq)):
            valid_tou_tmp[tmp_seq[m, 1]:tmp_seq[m, 2] + 1] = 0

        valid_tou_tmp = valid_tou_tmp.reshape(valid_tou.shape)

    return valid_tou_tmp


def post_process_for_estimation(timed_estimation, samples_per_hour, valid_tou, input_data):

    """
    Master function for detecting timed appliance in residual data

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        samples_per_hour          (int)           : samples in an hour
        valid_tou                 (np.ndarray)    : timed sig tou
        input_data                (np.ndarray)    : input data

    Returns:
        timed_estimation          (np.ndarray)    : timed estimation
    """

    # Cap high percentile points, remove low cons points

    # removing high high outlier points

    timed_estimation = np.fmin(timed_estimation, np.percentile(timed_estimation[timed_estimation > 0], 85))
    timed_estimation[np.logical_and(timed_estimation == 0, valid_tou > 0)] = np.percentile(timed_estimation[timed_estimation > 0], 50)
    timed_estimation[timed_estimation < min(np.percentile(timed_estimation[timed_estimation > 0], 5), 2000/samples_per_hour)] = 0
    timed_estimation[timed_estimation < 200 / samples_per_hour] = 0

    timed_estimation[timed_estimation == 0] = np.nan

    timed_estimation[np.nanmean(timed_estimation, axis=1) < np.percentile(timed_estimation[np.isnan(timed_estimation)], 25)] = 0

    timed_estimation = np.nan_to_num(timed_estimation)

    original_timed_estimation = copy.deepcopy(timed_estimation)

    timed_estimation_wh = copy.deepcopy(timed_estimation)

    if np.any(timed_estimation > 0):
        timed_estimation[timed_estimation > 0] = np.percentile(timed_estimation[timed_estimation > 0], 50)
        timed_estimation_wh[timed_estimation_wh > 0] = np.percentile(timed_estimation_wh[timed_estimation_wh> 0], 70)

    if np.any(original_timed_estimation > 0):
        original_timed_estimation[original_timed_estimation > 0] = np.percentile(original_timed_estimation[original_timed_estimation > 0], 80)

    if np.any(timed_estimation > 0):
        timed_estimation[timed_estimation > 0] = np.minimum(input_data[timed_estimation > 0], timed_estimation[timed_estimation > 0])

    if np.any(timed_estimation > 0):
        timed_estimation[timed_estimation < np.percentile(timed_estimation[timed_estimation > 0], 5)] = 0

    if np.any(timed_estimation_wh > 0):
        timed_estimation_wh[timed_estimation_wh > 0] = np.minimum(input_data[timed_estimation_wh > 0], timed_estimation_wh[timed_estimation_wh > 0])

    if np.any(timed_estimation_wh > 0):
        timed_estimation_wh[timed_estimation_wh < np.percentile(timed_estimation_wh[timed_estimation_wh > 0], 5)] = 0

    timed_estimation_wh[timed_estimation_wh < 400 / samples_per_hour] = 0

    return timed_estimation, original_timed_estimation, timed_estimation_wh
