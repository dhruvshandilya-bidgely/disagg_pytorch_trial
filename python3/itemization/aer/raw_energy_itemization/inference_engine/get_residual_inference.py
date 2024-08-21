
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Calculate weekend signature consumption from positive residual data
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import fill_array

from python3.itemization.aer.functions.itemization_utils import get_index_array


def weekend_act_detection(residual_data, input_data, dow):

    """
    Extract extra weekend consumption from disagg residual data

    Parameters:
        residual_data             (np.ndarray): disagg residual data
        input_data                (np.ndarray): input data
        dow                       (np.ndarray): dow tags

    Returns:
        detected_weekend          (np.ndarray): Detected weekend activity
    """

    residual = copy.deepcopy(residual_data)

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    samples_per_hour = int(residual_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    total_samples = residual_data.shape[1]

    residual = np.fmax(0, residual)

    residual = np.fmin(residual, np.percentile(residual[np.nonzero(residual)], 99))

    detected_weekend = np.zeros(residual_data.shape)

    weekend_days = np.logical_or(dow == 1, dow == 7)

    weekday = np.zeros(residual_data.shape)
    weekend = np.zeros(residual_data.shape)
    week = np.zeros(residual_data.shape)

    weekend_start = np.where(weekend_days)[0][0]
    weekend_days[: weekend_start] = 1
    weekday_start = np.where(np.logical_not(weekend_days))[0][0]
    weekday_start = np.arange(weekday_start, len(residual_data), Cgbdisagg.DAYS_IN_WEEK)
    weekend_start = np.arange(weekend_start, len(residual_data), Cgbdisagg.DAYS_IN_WEEK)

    # preparing weekly, weekday wise and weekend wise profile curve

    for i, start in enumerate(weekday_start):
        weekday[i] = np.mean(residual[start: start+5], axis=0)

    for i, start in enumerate(weekend_start):
        weekend[i] = np.mean(residual[start: start+2], axis=0)

    for i, start in enumerate(weekday_start):
        week[i] = np.mean(residual[start: start + 7], axis=0)

    weekday = weekday[~np.all(weekday == 0, axis=1)]
    week = week[~np.all(week == 0, axis=1)]

    weekend = weekend[~np.all(weekend == 0, axis=1)]

    if not len(weekend) or not len(weekday):
        return detected_weekend

    diff = max(np.max(weekend), np.max(weekday)) - min(np.min(weekend), np.min(weekday))
    diff = min(250, diff * 0.1)
    diff = max(diff, 50)

    detected_weekend_act = np.ones(weekend.shape)

    weekday_curve = np.percentile(weekday, 50, axis=0)

    temp_amp = np.zeros((2, len(residual_data[0])))

    min_samples_required_for_weekend_detection = 0.5 * samples_per_hour

    # from each chunk of week, extra weekend activity is extracted

    for i in range(min(len(week), len(weekend) - 1)):

        detected_weekend_act[i] = np.logical_and(detected_weekend_act[i], np.abs(weekend[i] - week[i]) > diff)

        detected_weekend_act[i] = detected_weekend_act[i].astype(bool)

        amplitude = weekend[i] - weekday_curve
        amplitude[amplitude < 0] = 0

        temp_amp[:, :] = amplitude

        detected_weekend_act[i][amplitude == 0] = 0

        seq = find_seq(detected_weekend_act[i], np.zeros(total_samples), np.zeros(total_samples))

        # filling small low duration sequence of 0 weekend activity

        seq[np.logical_and(seq[:, seq_label] == 0, seq[:, seq_len] <= min_samples_required_for_weekend_detection), 0] = 1

        for j in range(len(seq)):
            detected_weekend_act[i] = fill_array(detected_weekend_act[i], seq[j, seq_start], seq[j, seq_end], seq[j, seq_label])

        seq = find_seq(detected_weekend_act[i], np.zeros(total_samples), np.zeros(total_samples))

        # filling small low duration sequence of non-zero weekend activity

        seq[np.logical_and(seq[:, seq_label] == 1, seq[:, seq_len] <= min_samples_required_for_weekend_detection), 0] = 0

        for j in range(len(seq)):
            detected_weekend_act[i] = fill_array(detected_weekend_act[i], seq[j, seq_start], seq[j, seq_end], seq[j, seq_label])

        for j in range(len(seq)):
            if seq[j, seq_label] and seq[j, seq_len] > 3 * samples_per_hour:
                detected_weekend_act[i][get_index_array(seq[j, seq_start], seq[j, seq_start] + (seq[j, seq_len] - 3 * samples_per_hour), total_samples)] = 0

        detected_weekend[weekend_start[i]: weekend_start[i]+2, detected_weekend_act[i].astype(bool)] =\
            np.maximum(temp_amp[:, detected_weekend_act[i].astype(bool)], detected_weekend[weekend_start[i]: weekend_start[i]+2, detected_weekend_act[i].astype(bool)])

    weekend_cons = detected_weekend[weekend_days] > 0
    weekend_tou = np.sum(weekend_cons, axis=0) > (len(weekend_cons)*0.1)

    seq = find_seq(weekend_tou, np.zeros(total_samples), np.zeros(total_samples))

    seq[np.logical_and(seq[:, seq_label] == 0, seq[:, seq_len] <= 1.5 * samples_per_hour), 0] = 1

    for j in range(len(seq)):
        weekend_tou = fill_array(weekend_tou, seq[j, seq_start], seq[j, seq_end], seq[j, 0])

    seq = find_seq(weekend_tou, np.zeros(total_samples), np.zeros(total_samples))

    seq[np.logical_and(seq[:, seq_label] == 1, seq[:, seq_len] <= 0.25 * samples_per_hour), 0] = 0

    for j in range(len(seq)):
        weekend_tou = fill_array(weekend_tou, seq[j, seq_start], seq[j, seq_end], seq[j, seq_label])

    detected_weekend = np.minimum(detected_weekend, residual)

    detected_weekend = np.abs(detected_weekend)

    return detected_weekend


def get_res_inference(item_output_object):

    """
    Prepare hybrid input object

    Parameters:
        item_output_object        (dict)      : Dict containing all outputs

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]
    residual = copy.deepcopy(item_output_object.get("inference_engine_dict").get("residual_data"))

    weekend_act_det_not_required = np.all(residual <= 0) or len(input_data) < 10

    if weekend_act_det_not_required:
        item_output_object["inference_engine_dict"].update({
            "weekend_activity": np.zeros(residual.shape)
        })

    else:

        detected_weekend = \
            weekend_act_detection(residual, input_data, item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_DOW_IDX, :, 0])

        if np.any(detected_weekend > 0):
            item_output_object["residual_detection"][2] = 1

        item_output_object["inference_engine_dict"].update({
            "weekend_activity": detected_weekend
        })

    return item_output_object
