"""
Author - Sahana M
Date - 14-Nov-2023
Module containing the Box fitting utilities
"""

# import python packages
import numpy as np
from scipy import spatial
from copy import deepcopy

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.find_seq import find_seq
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import boxes_features
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_start_end_idx, get_wrong_indexes
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_partition_distribution, get_amp_distribution


def remove_overlapping_boxes(input_box_data, hod_matrix, input_box_features, box_index, box_min_energy, factor, dl_debug,
                             ev_config, ctype='l2'):
    """
    Function to remove overlapping boxes
        Parameters:
            input_box_data            (np.ndarray)        : Current box data
            hod_matrix                (np.ndarray)        : Hour of the day matrix
            input_box_features        (np.ndarray)        : Current Box Features
            box_index                 (int)               : Index of the box data
            box_min_energy            (float)             : Minimum energy of the boxes
            factor                    (int)               : Number of data points in an hour
            dl_debug                  (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
            ev_config                  (dict)             : EV module config
            ctype                      (string)           : Charger type
        Returns:
            box_data                  (np.ndarray)        : Updated box data
            box_features              (np.ndarray)        : Updated Box Features
    """

    box_data = deepcopy(input_box_data)
    box_features = deepcopy(input_box_features)

    # Previous box data and features
    previous_box_features = deepcopy(dl_debug.get('box_features_' + str(box_index - 1)))
    previous_box_start_idx = previous_box_features[:, 0]
    previous_box_end_idx = previous_box_features[:, 1]

    # Subtracting the energy of overlapped boxes

    for i, box in enumerate(box_features):
        start_idx, end_idx = box[:ev_config['box_features_dict']['end_index_column']+1].astype(int)

        # Check if the current box overlaps completely with existing boxes

        overlap_box = np.where((previous_box_start_idx <= start_idx) & (previous_box_end_idx >= end_idx))[0]

        if len(overlap_box) > 0:
            # If box overlaps with existing one

            overlap_box_idx = overlap_box.astype(int)[0]
            start_idx_match = (start_idx == previous_box_start_idx[overlap_box_idx])
            end_idx_match = (end_idx == previous_box_end_idx[overlap_box_idx])
            unique_overlap_check = start_idx_match & end_idx_match

            if unique_overlap_check:
                previous_min_energy = 0
            else:
                previous_min_energy = previous_box_features[
                    overlap_box_idx, ev_config['box_features_dict']['boxes_minimum_energy']]
        else:
            continue

        # Remove the minimum energy of the previous box as baseline for the current box

        box_max_energy = box[ev_config['box_features_dict']['boxes_maximum_energy']]

        if (box_max_energy - previous_min_energy) >= box_min_energy:
            box_data[start_idx:(end_idx + 1)] -= previous_min_energy
        else:
            box_data[start_idx:(end_idx + 1)] = 0

    box_features = boxes_features(box_data, hod_matrix, factor, ev_config, ctype)

    return box_data, box_features


def boxes_sanity_checks(input_box_features, min_energy, ev_config, ctype='l2'):
    """
    Function to check the quality of boxes
        Parameters:
            input_box_features        (np.ndarray)        : Current Box Features
            min_energy                (float)              : Minimum energy of the boxes
            ev_config                  (dict)              : EV module config
            ctype                      (string)            : Charger type

        Returns:
            check_fail                (bool)              : Boolean signifying whether boxes are acceptable
    """

    # Reading config params

    sanity_check_config = ev_config.get('detection', {}).get('sanity_checks', {})
    duration_variation = sanity_check_config.get('duration_variation')
    box_energy_variation = sanity_check_config.get('box_energy_variation')
    within_box_energy_variation = sanity_check_config.get('within_box_energy_variation')
    columns_dict = ev_config.get('box_features_dict')
    box_features = deepcopy(input_box_features)

    if ctype == 'l1':
        columns_dict = ev_config.get('features_dict')
        duration_variation = sanity_check_config.get('l1').get('duration_variation')
        box_energy_variation = sanity_check_config.get('l1').get('box_energy_variation')
        within_box_energy_variation = sanity_check_config.get('l1').get('within_box_energy_variation')

        overall_duration_variation = (np.mean(np.abs(box_features[:, columns_dict['boxes_duration_column']] -
                                                     np.mean(box_features[:, columns_dict['boxes_duration_column']])))) \
                                     / np.mean(box_features[:, columns_dict['boxes_duration_column']])

    else:
        # Getting column descriptions of box features matrix

        overall_duration_variation = np.mean(np.abs(box_features[:, columns_dict['boxes_duration_column']] - np.mean(
            box_features[:, columns_dict['boxes_duration_column']])))

    overall_box_energy_variation = np.std(box_features[:, columns_dict['boxes_median_energy']]) / min_energy
    overall_within_box_energy_variation = np.mean(box_features[:, columns_dict['boxes_energy_std_column']]) / min_energy

    # Duration variance check

    if overall_duration_variation > duration_variation:
        duration_check_fail = True
    else:
        duration_check_fail = False

    # Boxes energy variation check

    if overall_box_energy_variation > box_energy_variation:
        box_energy_variation_check_fail = True
    else:
        box_energy_variation_check_fail = False

    # Within box energy variation check

    if overall_within_box_energy_variation > within_box_energy_variation:
        within_box_energy_variation_check_fail = True
    else:
        within_box_energy_variation_check_fail = False

    check_fail = duration_check_fail | box_energy_variation_check_fail | within_box_energy_variation_check_fail

    return check_fail


def filter_ev_boxes(current_box_data, in_data, factor, ev_config):
    """
    This function is used to filter the EV boxes
    Parameters:
        current_box_data            (np.ndarray)        : Current box data
        in_data                     (np.ndarray)        : Input data
        factor                      (float)             : Sampling rate w.r.t 60 minutes
        ev_config                   (Dict)              : EV configurations
    Returns:
        current_box_data            (np.ndarray)        : Current box data
    """

    # Taking local deepcopy of the ev boxes
    box_data = deepcopy(current_box_data)
    input_data = deepcopy(in_data)

    # Extraction energy data of boxes
    boxes_energy = deepcopy(box_data)
    input_data_energy = deepcopy(input_data)

    # Taking only positive box boundaries for edge related calculations
    box_energy_idx = (boxes_energy > 0)
    box_energy_idx_diff = np.diff(np.r_[0, box_energy_idx.astype(int), 0])

    # Find the start and end edges of the boxes
    box_start_boolean = (box_energy_idx_diff[:-1] > 0)
    box_end_boolean = (box_energy_idx_diff[1:] < 0)
    box_start_indices = np.where(box_start_boolean)[0]
    box_end_indices = np.where(box_end_boolean)[0]
    box_start_end_arr = np.c_[box_start_indices, box_end_indices]
    window_size = ev_config.get('min_duration_l1')

    for i, row in enumerate(box_start_end_arr):
        box_start_idx = row[0]
        box_end_idx = row[1]

        box_left_idx = max(0, box_start_idx - int(window_size * factor))
        box_right_idx = min(box_end_idx + int(window_size * factor), len(box_data) - 1)

        left_min = 0
        right_min = 0
        num_pts_day = int(Cgbdisagg.HRS_IN_DAY * factor)

        # Identify the left and right box indexes

        last_day_left_idx = box_left_idx - num_pts_day
        last_day_start_idx = box_start_idx - num_pts_day
        last_day_right_idx = box_right_idx - num_pts_day

        if box_start_idx > 0:
            left_min = np.nanmin(input_data_energy[box_left_idx: box_start_idx])

        if last_day_left_idx >= 0:
            left_min = np.nanmin(np.r_[left_min, input_data_energy[last_day_left_idx: last_day_start_idx]])

        if box_end_idx < len(box_data) - 1:
            right_min = np.nanmin(input_data_energy[box_end_idx + 1: box_right_idx + 1])

        if last_day_right_idx <= len(box_data) - 1:
            right_min = np.nanmin(np.r_[right_min, input_data_energy[box_end_idx - num_pts_day + 1:last_day_right_idx + 1]])

        # Get the base energy by taking the maximum of left and right minimum

        base_energy = np.nanmax([left_min, right_min])
        boxes_energy[box_start_idx: (box_end_idx + 1)] -= base_energy
        current_box_data = boxes_energy

    return current_box_data


def get_amp_distribution_non_zero(ev_boxes_2d, non_zero_usage_days):
    """
    Function to get the amplitude distribution
    Parameters:
        ev_boxes_2d                       (np.ndarray)          : EV boxes array
        non_zero_usage_days               (np.ndarray)          : Boolean array for EV boxes
    Returns:
        ev_boxes_amp_distribution         (np.ndarray)          : EV boxes amplitude distribution
    """

    # Extract the required variables
    ev_boxes_amp_distribution = []
    non_zero_ev_boxes_2d = ev_boxes_2d[non_zero_usage_days]

    # Get the distribution on the EV
    for i in range(non_zero_ev_boxes_2d.shape[1]):
        ev_indexes = non_zero_ev_boxes_2d[:, i] > 0
        if np.sum(ev_indexes):
            amp = np.nanmean(non_zero_ev_boxes_2d[ev_indexes, i])
        else:
            amp = 0
        ev_boxes_amp_distribution.append(amp)

    ev_boxes_amp_distribution = np.asarray(ev_boxes_amp_distribution)
    return ev_boxes_amp_distribution


def get_ev_strike(curr_day, final_cur_day_bool, max_duration):
    """
    Function to extract the ev strikes
    Parameters:
        curr_day                    (np.ndarray)               : Current day data
        final_cur_day_bool          (np.ndarray)               : Boolean final current day indexes
        max_duration                (int)                      : Max duration
    Returns:
        curr_day_final              (np.ndarray)               : Current final day data
    """

    min_thr = 0.25
    max_thr = 1.75
    curr_day_final = deepcopy(curr_day)

    # get the boxes which are definitely EVs
    curr_day_final[~final_cur_day_bool] = 0

    # get the start and end indexes of the obvious strikes
    start_idx, end_idx = get_start_end_idx(curr_day_final)

    # Do left extension for each strike if the neighbouring amplitudes satisfy a certain minimum amplitude threshold
    for i in range(len(start_idx)):

        min_start_idx = max(int(start_idx[i] - (max_duration / 2)), 0)
        j = start_idx[i] - 1

        # perform left extension
        while j > 0 and curr_day[j] != 0 and j > min_start_idx:
            amplitude_condition = (curr_day[j] >= min_thr * curr_day[j+1]) & (curr_day[j] <= max_thr * curr_day[j+1])
            if amplitude_condition:
                curr_day_final[j] = curr_day[j]
            j = j-1

    # Do right extension for each strike if the neighbouring amplitudes satisfy a certain minimum amplitude threshold
    for i in range(len(end_idx)):

        max_end_idx = min(int(end_idx[i] + (max_duration / 2)), len(curr_day_final))
        j = end_idx[i] + 1
        # perform left extension
        while j < len(curr_day) and curr_day[j] != 0 and j < max_end_idx:
            amplitude_condition = (curr_day[j] >= min_thr * curr_day[j-1]) & (curr_day[j] <= max_thr * curr_day[j - 1])
            if amplitude_condition:
                curr_day_final[j] = curr_day[j]
            j = j + 1

    return curr_day_final


def get_missing_boxes(predictions_bool, prt_size, changes_indexes, boxes_data, raw_data):
    """
    This function is used to identify the missing EV boxes indexes
    Parameter:
        predictions_bool                (np.ndarray)        : Predictions boolean array
        prt_size                        (int)               : Partition size (in days)
        changes_indexes                 (np.ndaraay)        : Changes indexes
        boxes_data                      (np.ndarray)        : Boxes data
        raw_data                        (np.ndarray)        : Raw data
    Returns:
        ev_boxes_2d                     (np.ndarray)        : EV boxes data
        missing_boxes_2d                (np.ndarray)        : Missing boxes data
    """
    ev_boxes_2d = []
    missing_boxes_2d = []

    # Identify the missing EV boxes

    for i in range(len(predictions_bool)):
        if predictions_bool[i]:
            start_idx = int(i * prt_size)
            end_idx = int(start_idx + prt_size)
            changes_indexes[start_idx:end_idx] = True
            if len(ev_boxes_2d):
                ev_boxes_2d = np.r_[ev_boxes_2d, boxes_data[start_idx: end_idx, :]]
                missing_boxes_2d = np.r_[missing_boxes_2d, raw_data[start_idx: end_idx, :]]
            else:
                ev_boxes_2d = boxes_data[start_idx: end_idx, :]
                missing_boxes_2d = raw_data[start_idx: end_idx, :]

    return ev_boxes_2d, missing_boxes_2d


def pick_missing_l2_boxes(box_data, dl_debug, predictions_bool=None, raw_data=None):
    """
    Function to identify the missing boxes
    Parameters:
        box_data                    (np.ndarray)            : Box data
        dl_debug                    (Dict)                  : Debug dictionary
        predictions_bool            (np.ndarray)            : Predictions Booleanarray
        raw_data                    (np.ndarray)            : Raw data
    Returns:
        final_ev_boxes              (np.ndarray)            : Final EV boxes
        dl_debug                    (Dict)                  : Debug dictionary
    """

    # Extract the required variables
    boxes_added = 0
    boxes_data = deepcopy(box_data)
    partition_size = dl_debug.get('config').get('prt_size')
    box_columns = dl_debug.get('config').get('box_features_dict')
    overlap_percentage_1 = dl_debug.get('config').get('pick_missing_boxes').get('overlap_percentage_1')
    overlap_percentage_2 = dl_debug.get('config').get('pick_missing_boxes').get('overlap_percentage_2')
    low_amp_thr = dl_debug.get('config').get('pick_missing_boxes').get('low_amp_thr')
    high_amp_thr = dl_debug.get('config').get('pick_missing_boxes').get('high_amp_thr')

    if predictions_bool is None:
        raw_data = deepcopy(dl_debug.get('boxes_data'))
        predictions_bool = dl_debug.get('predictions_bool')

    final_box_index = dl_debug.get('final_box_index')
    box_features = dl_debug.get('updated_box_features_' + str(final_box_index))
    max_duration = (2 + np.ceil(np.mean(box_features[:, box_columns['boxes_duration_column']]))) * dl_debug.get('factor')

    # Get only the partitions with EV predictions

    changes_indexes = np.full(shape=(box_data.shape[0]), fill_value=False)
    ev_boxes_2d, missing_boxes_2d = get_missing_boxes(predictions_bool, partition_size, changes_indexes, boxes_data, raw_data)

    missing_boxes_2d = missing_boxes_2d - ev_boxes_2d

    # get the percentage distribution of EV usage in ev partitions
    final_ev_boxes = deepcopy(box_data)
    non_zero_usage_days = np.sum(ev_boxes_2d, axis=1) > 0

    if np.sum(non_zero_usage_days):
        hourly_ev_data = ev_boxes_2d[non_zero_usage_days] > 0
        ev_boxes_distribution = np.sum(hourly_ev_data, axis=0) / np.sum(np.sum(hourly_ev_data, axis=0)) * 100
        wrong_indexes = get_wrong_indexes(ev_boxes_distribution)
        ev_boxes_distribution[wrong_indexes] = 0

        # get the amplitude distribution of EV usage in ev partitions

        ev_boxes_amp_distribution = get_amp_distribution_non_zero(ev_boxes_2d, non_zero_usage_days)

        # go over each ev partition and fill in the EV boxes
        missing_boxes_2d_zero_days = missing_boxes_2d[~non_zero_usage_days]
        changes_indexes[changes_indexes > 0] = changes_indexes[changes_indexes > 0] & ~non_zero_usage_days
        new_boxes = np.zeros_like(missing_boxes_2d_zero_days)
        for i in range(len(missing_boxes_2d_zero_days)):
            curr_day = missing_boxes_2d_zero_days[i]

            # get the distribution overlap score
            curr_day_bool = curr_day > 0
            ev_boxes_distribution_bool = ev_boxes_distribution > 0
            percentage_overlap = np.sum(curr_day_bool[ev_boxes_distribution_bool])/len(curr_day_bool[ev_boxes_distribution_bool])
            final_cur_day_bool = curr_day_bool & ev_boxes_distribution_bool

            if percentage_overlap > overlap_percentage_1:
                ev_boxes_amplitude = ev_boxes_amp_distribution[final_cur_day_bool]
                cur_day_amplitude = curr_day[final_cur_day_bool]
                amplitude_bar_low = low_amp_thr * ev_boxes_amplitude
                amplitude_bar_high = high_amp_thr * ev_boxes_amplitude
                probable_boxes_indexes = (cur_day_amplitude > amplitude_bar_low) & (cur_day_amplitude < amplitude_bar_high)
                overlapping_percentage = np.sum(probable_boxes_indexes) / len(probable_boxes_indexes)
                if overlapping_percentage > overlap_percentage_2:
                    curr_day = get_ev_strike(curr_day, final_cur_day_bool, max_duration)
                    new_boxes[i] = curr_day
                    boxes_added += 1

        final_ev_boxes = deepcopy(box_data)
        final_ev_boxes[changes_indexes] = new_boxes
        dl_debug['boxes_added'] = boxes_added

    return final_ev_boxes, dl_debug


def get_partition_data(seq, data, idx):
    """
    Function to get the partitions
    Parameters:
        seq                     (np.ndarray)                : Sequence data
        data                    (np.ndarray)                : 2D matrix data
        idx                     (int)                       : Index value
    Return:
        partition_data          (np.ndarray)                : Partition data
    """

    partition_size = 14
    start_idx = seq[idx, 1]
    end_idx = seq[idx, 2]
    start_idx = int(start_idx * partition_size)
    end_idx = int(end_idx * partition_size + partition_size)
    partition_data = data[start_idx: end_idx, :]

    return partition_data


def identify_new_partition(prt_seq, i, ev_data, new_prt_predictions, new_prt_confidences, new_partitions_detected,
                           dl_debug, last_partition=False):
    """
    Function to identify if a partition has EV or not and update the confidences
    Parameters:
        prt_seq                     (np.ndarray)            : Partition sequences
        i                           (int)                   : Current partition index
        ev_data                     (np.ndarray)            : EV data array
        new_prt_predictions         (np.ndarray)            : Updated partition predictions
        new_prt_confidences         (np.ndarray)            : Updated partition confidences
        new_partitions_detected     (Boolean)               : Whether new partitions were detected or not
        dl_debug                    (Dict)                  : Debug dictionary
        last_partition              (Boolean)               : Check for last partition
    Returns:
        new_prt_predictions         (np.ndarray)            : Updated partition predictions
        new_prt_confidences         (np.ndarray)            : Updated partition confidences
        new_partitions_detected     (Boolean)               : Whether new partitions were detected or not
    """

    # Extract required variables

    input_data = deepcopy(dl_debug.get('hvac_removed_raw_data'))
    min_amp_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('min_amp_thr')
    max_amp_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('max_amp_thr')
    amp_curr_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('amp_curr_thr')
    similarity_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('similarity_thr')
    prev_overlap_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('prev_overlap_thr')
    overall_score_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('overall_score_thr')
    min_sequence_size = dl_debug.get('config').get('identifying_all_ev_partitions').get('min_sequence_size')
    amp_similarity_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('amp_similarity_thr')

    # If this is a last parition then don't create a next parition else create past, current and next partition
    if last_partition:
        prev_prt_detection = prt_seq[i - 1, 0]
        curr_prt_detection = prt_seq[i, 0]
        curr_prt_size = prt_seq[i, 3]
        prev_prt_data = get_partition_data(prt_seq, ev_data, i - 1)
        curr_prt_data = get_partition_data(prt_seq, input_data, i)
    else:
        prev_prt_detection = prt_seq[i - 1, 0]
        curr_prt_detection = prt_seq[i, 0]
        next_prt_detection = prt_seq[i + 1, 0]
        curr_prt_size = prt_seq[i, 3]
        prev_prt_data = get_partition_data(prt_seq, ev_data, i - 1)
        curr_prt_data = get_partition_data(prt_seq, input_data, i)

    # check if the current chunk of partition/s is not detected but its neighbouring partition chunks are detected

    if last_partition:
        squished_prt = ((curr_prt_detection == 0) & (prev_prt_detection == 1))
    else:
        squished_prt = ((curr_prt_detection == 0) & (prev_prt_detection == 1) & (next_prt_detection == 1))

    if squished_prt and curr_prt_size <= min_sequence_size:

        # get the amplitude distribution
        prev_amp_distribution = get_amp_distribution(prev_prt_data)

        if np.sum(prev_amp_distribution):
            min_amp = np.percentile(prev_amp_distribution[prev_amp_distribution > 0], q=50) * min_amp_thr
            curr_prt_data[curr_prt_data < min_amp] = 0

            # get the amplitude distribution
            curr_amp_distribution = get_amp_distribution(curr_prt_data)

            # get the normalised amp distribution
            prev_amp_norm_distribution = prev_amp_distribution / max(prev_amp_distribution)
            curr_amp_norm_distribution = curr_amp_distribution / max(curr_amp_distribution)

            # get the partition distribution
            prev_prt_distribution = get_partition_distribution(prev_prt_data)

            similarity_score = 1 - spatial.distance.cosine(prev_amp_norm_distribution, curr_amp_norm_distribution)
            prev_apm_norm_idx = prev_amp_norm_distribution > 0
            similarity_score_prev_overlap = 1 - spatial.distance.cosine(prev_amp_norm_distribution[prev_apm_norm_idx],
                                                                        curr_amp_norm_distribution[prev_apm_norm_idx])

            # get the amplitude similarity
            ev_hou = prev_prt_distribution > 0
            ev_amps = prev_amp_distribution[ev_hou]
            cur_amps = curr_amp_distribution[ev_hou]
            amp_similarity = (cur_amps > min_amp_thr * ev_amps) & (cur_amps < max_amp_thr * ev_amps)
            amp_similarity = np.sum(amp_similarity) / len(amp_similarity)

            ev_hou_curr_idx = cur_amps > 0
            amp_similarity_curr_idx = (cur_amps[ev_hou_curr_idx] > min_amp_thr * ev_amps[ev_hou_curr_idx]) & \
                                      (cur_amps[ev_hou_curr_idx] < max_amp_thr * ev_amps[ev_hou_curr_idx])
            amp_similarity_curr_idx = np.sum(amp_similarity_curr_idx) / len(amp_similarity_curr_idx)

            # Get the overall similarity score
            overall_score = similarity_thr * similarity_score + amp_similarity_thr * amp_similarity \
                            + prev_overlap_thr * similarity_score_prev_overlap + amp_curr_thr * amp_similarity_curr_idx

            if overall_score > overall_score_thr:
                start_idx = prt_seq[i, 1]
                end_idx = prt_seq[i, 2] + 1
                new_prt_predictions[start_idx: end_idx] = True
                new_prt_confidences[start_idx: end_idx] = overall_score
                new_partitions_detected = True

    return new_prt_predictions, new_prt_confidences, new_partitions_detected


def identify_all_ev_l2_partitions(box_data, dl_debug):
    """
    Function to identify all the partitions where there could be EV
    Parameters:
        box_data                    (np.ndarray)          : Box data
        dl_debug                    (Dict)                : Debug dictionary

    Returns:
        box_data                    (np.ndarray)          : Box data
        dl_debug                    (Dict)                : Debug dictionary
    """

    # Extract the required parameters
    ev_data = deepcopy(box_data)
    prt_predictions = dl_debug.get('predictions_bool')
    prt_confidences = dl_debug.get('prediction_confidences')
    new_partitions_detected = False
    prt_seq = find_seq(prt_predictions, min_seq_length=0)
    new_prt_predictions = deepcopy(prt_predictions)
    new_prt_confidences = deepcopy(prt_confidences)

    # look for the partitions that are not predicted but are squished between 2 detected partitions

    for i in range(1, len(prt_seq) - 1):
        new_prt_predictions, new_prt_confidences, new_partitions_detected = \
            identify_new_partition(prt_seq, i, ev_data, new_prt_predictions, new_prt_confidences, new_partitions_detected,
                                   dl_debug)

    # Check for the last partition
    i = len(prt_seq)-1

    # check if the current chunk of partition/s is not detected but its neighbouring partition chunks are detected

    new_prt_predictions, new_prt_confidences, new_partitions_detected = \
        identify_new_partition(prt_seq, i, ev_data, new_prt_predictions, new_prt_confidences, new_partitions_detected,
                               dl_debug, True)

    dl_debug['updated_partition_predictions'] = new_prt_predictions
    dl_debug['final_partition_predictions'] = new_prt_predictions
    dl_debug['updated_partition_confidences'] = new_prt_confidences

    # Picking boxes from the predicted partitions

    if new_partitions_detected:
        box_data, dl_debug = pick_missing_l2_boxes(box_data, dl_debug, new_prt_predictions,
                                                   deepcopy(dl_debug.get('hvac_removed_raw_data')))

    return box_data, dl_debug


def get_amp_distribution_arr(ev_boxes_2d, non_zero_usage_days):
    """
    Function to get the amplitude distribution for a partition
    Functions:
        ev_boxes_2d                 (np.ndarray)            : EV boxes data to get the distribution
        non_zero_usage_days         (boolean)               : Days when EV usage is present
    Returns:
        ev_boxes_amp_distribution   (np.ndarray)            : EV amplitude distribution array
    """

    ev_boxes_amp_distribution = []
    non_zero_ev_boxes_2d = ev_boxes_2d[non_zero_usage_days]

    # Get the distribution of EV presence for each time of the day

    for i in range(non_zero_ev_boxes_2d.shape[1]):
        ev_indexes = non_zero_ev_boxes_2d[:, i] > 0
        if np.sum(ev_indexes):
            amp = np.nanmean(non_zero_ev_boxes_2d[ev_indexes, i])
        else:
            amp = 0
        ev_boxes_amp_distribution.append(amp)
    ev_boxes_amp_distribution = np.asarray(ev_boxes_amp_distribution)

    return ev_boxes_amp_distribution


def get_ev_l1_strike(curr_day, final_cur_day_bool):
    """
    Function to get the refined EV strikes
    Parameters:
        curr_day                    (np.ndarray)    : Current partition data
        final_cur_day_bool          (np.ndarray)    : Current day indexes
    Returns:
        curr_day_final              (int)           : Final Current partition array
    """

    min_thr = 0.25
    max_thr = 1.75
    curr_day_final = deepcopy(curr_day)

    # get the boxes which are definitely EVs
    curr_day_final[~final_cur_day_bool] = 0

    # get the start and end indexes of the obvious strikes
    start_idx, _ = get_start_end_idx(curr_day_final)

    # Do left extension for each strike
    for i in range(len(start_idx)):

        j = start_idx[i] - 1
        # perform left extension
        while j > 0 and curr_day[j] != 0:
            amplitude_condition = (curr_day[j] >= min_thr * curr_day[j+1]) & (curr_day[j] <= max_thr * curr_day[j+1])
            if amplitude_condition:
                curr_day_final[j] = curr_day[j]
            j = j-1

    # Do right extension for each strike
    for i in range(len(start_idx)):

        j = start_idx[i] + 1
        # perform left extension
        while j < len(curr_day) and curr_day[j] != 0:
            amplitude_condition = (curr_day[j] >= min_thr * curr_day[j-1]) & (curr_day[j] <= max_thr * curr_day[j - 1])
            if amplitude_condition:
                curr_day_final[j] = curr_day[j]
            j = j + 1

    return curr_day_final


def pick_missing_l1_boxes(box_data, dl_debug, predictions_bool=None, raw_data=None):
    """
    Function to identify the missed L1 boxes while box fitting based on the similarity with the already captured L1
    boxes
    Parameters:
        box_data                    (np.ndarray)            : Box data
        dl_debug                    (Dict)                  : Debug dictionary
        predictions_bool            (np.ndarray)            : Predictions boolean
        raw_data                    (np.ndarray)            : Raw data
    Returns:
        final_ev_boxes              (np.ndarray)            : Final EV data
        debug                       (Dict)                  : Debug dictionary
    """

    # Extract the required variables

    prt_size = dl_debug.get('config').get('prt_size')
    overlap_percentage_1 = dl_debug.get('config').get('pick_missing_boxes').get('overlap_percentage_1')
    overlap_percentage_2 = dl_debug.get('config').get('pick_missing_boxes').get('overlap_percentage_2')
    low_amp_thr = dl_debug.get('config').get('pick_missing_boxes').get('low_amp_thr')
    high_amp_thr = dl_debug.get('config').get('pick_missing_boxes').get('high_amp_thr')

    boxes_added = 0
    boxes_data = deepcopy(box_data)

    if predictions_bool is None:
        raw_data = deepcopy(dl_debug.get('boxes_data_preserved'))
        predictions_bool = dl_debug.get('predictions_bool_l1')

    # get only the partitions with EV predictions
    changes_indexes = np.full(shape=(box_data.shape[0]), fill_value=False)
    ev_boxes_2d, missing_boxes_2d = get_missing_boxes(predictions_bool, prt_size, changes_indexes, boxes_data, raw_data)

    missing_boxes_2d = missing_boxes_2d - ev_boxes_2d

    # get the percentage distribution of EV usage in ev partitions
    non_zero_usage_days = np.sum(ev_boxes_2d, axis=1) > 0

    if np.sum(non_zero_usage_days):
        hourly_ev_data = ev_boxes_2d[non_zero_usage_days] > 0
        ev_boxes_distribution = np.sum(hourly_ev_data, axis=0) / np.sum(np.sum(hourly_ev_data, axis=0)) * 100
        wrong_indexes = get_wrong_indexes(ev_boxes_distribution, 'L1')
        ev_boxes_distribution[wrong_indexes] = 0

        # get the amplitude distribution of EV usage in ev partitions
        ev_boxes_amp_distribution = get_amp_distribution_arr(ev_boxes_2d, non_zero_usage_days)

        # go over each ev partition and fill in the EV boxes
        missing_boxes_2d_zero_days = missing_boxes_2d[~non_zero_usage_days]
        changes_indexes[changes_indexes > 0] = changes_indexes[changes_indexes > 0] & ~non_zero_usage_days
        new_boxes = np.zeros_like(missing_boxes_2d_zero_days)

        for i in range(len(missing_boxes_2d_zero_days)):
            curr_day = missing_boxes_2d_zero_days[i]

            # get the distribution overlap score
            curr_day_bool = curr_day > 0
            ev_boxes_distribution_bool = ev_boxes_distribution > 0
            percentage_overlap = np.sum(curr_day_bool[ev_boxes_distribution_bool])/len(curr_day_bool[ev_boxes_distribution_bool])
            final_cur_day_bool = curr_day_bool & ev_boxes_distribution_bool

            # identify the possible ev strike
            if percentage_overlap > overlap_percentage_1:
                ev_boxes_amplitude = ev_boxes_amp_distribution[final_cur_day_bool]
                cur_day_amplitude = curr_day[final_cur_day_bool]
                amplitude_bar_low = low_amp_thr * ev_boxes_amplitude
                amplitude_bar_high = high_amp_thr * ev_boxes_amplitude
                probable_boxes_indexes = (cur_day_amplitude > amplitude_bar_low) & (cur_day_amplitude < amplitude_bar_high)
                overlapping_percentage = np.sum(probable_boxes_indexes) /len(probable_boxes_indexes)
                if overlapping_percentage > overlap_percentage_2:
                    curr_day = get_ev_l1_strike(curr_day, final_cur_day_bool)
                    new_boxes[i] = curr_day
                    boxes_added += 1

        final_ev_boxes = deepcopy(box_data)
        final_ev_boxes[changes_indexes] = new_boxes
        dl_debug['boxes_added'] = boxes_added

    else:
        final_ev_boxes = deepcopy(box_data)

    return final_ev_boxes, dl_debug


def check_for_nan(feature):
    """Function for nan check
    Parameters:         feature            (np.ndarray)            : Feature array
    Returns             feature            (np.ndarray)            : Feature array"""

    if np.isnan(feature):
        feature = 0
    return feature


def identify_all_ev_l1_partitions(box_data, dl_debug):
    """
    Function to identify the EV L1 partitions
    Parameters:
        box_data                (np.ndarray)            : Box data
        dl_debug                   (Dict)                  : Debug dictionary
    Returns:
        box_data                (np.ndarray)            : Box data
        debug                   (Dict)                  : Debug dictionary
    """

    # Extract the required variables

    col_idx = {'value': 0,  'start_idx': 1,  'end_idx': 2, 'quantity': 3}
    confidence_threshold = dl_debug.get('config').get('confidence_threshold')
    min_amp_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('min_amp_thr')
    max_amp_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('max_amp_thr')
    amp_curr_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('amp_curr_thr')
    similarity_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('similarity_thr')
    prev_overlap_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('prev_overlap_thr')
    amp_similarity_thr = dl_debug.get('config').get('identifying_all_ev_partitions').get('amp_similarity_thr')
    min_sequence_size_l1 = dl_debug.get('config').get('identifying_all_ev_partitions').get('min_sequence_size_l1')

    # Extract the required variables
    ev_data = deepcopy(box_data)
    prt_predictions = dl_debug.get('predictions_bool_l1')
    input_data = deepcopy(dl_debug.get('boxes_data_preserved'))
    new_partitions_detected = False
    prt_seq = find_seq(prt_predictions, min_seq_length=0)
    new_prt_predictions = deepcopy(prt_predictions)

    # look for the partitions that are not predicted but are squished between 2 detected partitions

    for i in range(1, len(prt_seq) - 1):
        prev_prt_detection = prt_seq[i-1, col_idx['value']]
        curr_prt_detection = prt_seq[i, col_idx['value']]
        next_prt_detection = prt_seq[i+1, col_idx['value']]
        curr_prt_size = prt_seq[i, col_idx['quantity']]

        prev_prt_data = get_partition_data(prt_seq, ev_data, i-1)
        curr_prt_data = get_partition_data(prt_seq, input_data, i)

        # check if the current chunk of partition/s is not detected but its neighbouring partition chunks are detected

        squished_prt = (curr_prt_detection == 0) & (prev_prt_detection == 1) & (next_prt_detection == 1)

        if squished_prt and curr_prt_size <= min_sequence_size_l1:

            # get the amplitude distribution
            non_zero_usage_days = np.sum(prev_prt_data, axis=1) > 0
            prev_amp_distribution = get_amp_distribution_arr(prev_prt_data, non_zero_usage_days)
            # If there is an EV detected in the previous partitions then calculate a new score accordingly
            if np.sum(prev_amp_distribution):
                min_amp = np.percentile(prev_amp_distribution[prev_amp_distribution > 0], q=50) * 0.75
                curr_prt_data[curr_prt_data < min_amp] = 0

                # get the amplitude distribution
                non_zero_usage_days = np.sum(curr_prt_data, axis=1) > 0
                curr_amp_distribution = get_amp_distribution_arr(curr_prt_data, non_zero_usage_days)

                # get the normalised amp distribution
                prev_amp_norm_distribution = prev_amp_distribution/max(prev_amp_distribution)
                curr_amp_norm_distribution = curr_amp_distribution / max(curr_amp_distribution)

                # get the partition distribution
                prev_prt_distribution = get_partition_distribution(prev_prt_data, 'L1')

                similarity_score = 1 - spatial.distance.cosine(prev_amp_norm_distribution, curr_amp_norm_distribution)
                prev_apm_norm_idx = prev_amp_norm_distribution > 0
                similarity_score_prev_overlap = 1 - spatial.distance.cosine(prev_amp_norm_distribution[prev_apm_norm_idx],
                                                                            curr_amp_norm_distribution[prev_apm_norm_idx])

                similarity_score = check_for_nan(similarity_score)
                similarity_score_prev_overlap = check_for_nan(similarity_score_prev_overlap)

                # get the amplitude similarity
                ev_hou = prev_prt_distribution > 0
                ev_amps = prev_amp_distribution[ev_hou]
                cur_amps = curr_amp_distribution[ev_hou]
                amp_similarity = (cur_amps > min_amp_thr * ev_amps) & (cur_amps < max_amp_thr * ev_amps)
                amp_similarity = np.sum(amp_similarity) / len(amp_similarity)
                amp_similarity = check_for_nan(amp_similarity)

                ev_hou_curr_idx = cur_amps > 0
                amp_similarity_curr_idx = (cur_amps[ev_hou_curr_idx] > min_amp_thr * ev_amps[ev_hou_curr_idx]) & \
                                          (cur_amps[ev_hou_curr_idx] < max_amp_thr * ev_amps[ev_hou_curr_idx])
                amp_similarity_curr_idx = np.sum(amp_similarity_curr_idx) / len(amp_similarity_curr_idx)
                amp_similarity_curr_idx = check_for_nan(amp_similarity_curr_idx)

                overall_score = similarity_thr*similarity_score + amp_similarity_thr*amp_similarity \
                                + prev_overlap_thr*similarity_score_prev_overlap + amp_curr_thr*amp_similarity_curr_idx

                if overall_score > confidence_threshold:
                    start_idx = prt_seq[i, col_idx['start_idx']]
                    end_idx = prt_seq[i, col_idx['end_idx']] + 1
                    new_prt_predictions[start_idx: end_idx] = True
                    new_partitions_detected = True

    dl_debug['updated_partition_predictions_l1'] = new_prt_predictions
    dl_debug['final_partition_predictions_l1'] = new_prt_predictions

    # Picking boxes from the predicted partitions

    if new_partitions_detected:
        box_data, dl_debug = pick_missing_l1_boxes(box_data, dl_debug, new_prt_predictions, input_data)

    return box_data, dl_debug


def clean_boxes(invalid_boxes, box_data, box_features, columns_dict, dl_debug, ev_config, hod_matrix, ctype='l2'):
    """
    This function is used to remove the labelled invalid boxes from box data and features
    Parameters:
        invalid_boxes           (Boolean)               : Boolean invalid boxes
        box_data                (np.ndarray)            : EV boxes data
        box_features            (np.ndarray)            : EV boxes features
        columns_dict            (dict)                  : Column names and their indexes
        dl_debug                (dict)                  : Debug dictionary
        ev_config               (dict)                  : EV configurations
        hod_matrix              (np.ndarray)            : HOD matrix
        ctype                   (string)                : Charger type
    Returns:
        box_data                (np.ndarray)            : EV boxes data
        box_features            (np.ndarray)            : EV boxes features
    """

    for i in range(len(box_features)):
        if invalid_boxes[i]:
            start_idx = box_features[int(i), columns_dict['start_index_column']].astype(int)
            end_idx = (box_features[int(i), columns_dict['end_index_column']] + 1).astype(int)
            box_data[start_idx:end_idx] = 0

    box_features = boxes_features(box_data, hod_matrix, dl_debug.get('factor'), ev_config, ctype)

    return box_data, box_features


def refine_l1_boxes(boxes_data, dl_debug):
    """
    Function used to remove noise L1 boxes
    Parameters:
        boxes_data              (np.ndarray)            : EV boxes data
        dl_debug                   (dict)                  : Debug dictionary
    Returns:
        box_data                (np.ndarray)            : EV boxes data
    """

    # Extract required variables

    config = dl_debug.get('config')
    factor = dl_debug.get('factor')
    hod_matrix = dl_debug.get('hod_matrix')
    hod_matrix = hod_matrix.flatten()
    columns_dict = config.get('features_dict')
    charging_hours = config.get('charging_hours')
    invalid_box_thr = config.get('invalid_box_thr')
    dur_amp_threshold = config.get('dur_amp_threshold')
    minimum_duration_allowed = config.get('minimum_duration_allowed')
    maximum_duration_allowed = config.get('maximum_duration_allowed')
    box_data_flat = deepcopy(boxes_data)
    box_data_flat = box_data_flat.flatten()
    box_features = boxes_features(box_data_flat, hod_matrix, factor, config, 'l1')
    invalid_boxes = np.full(shape=(box_features.shape[0]), fill_value=False)

    # Remove boxes with very less duration (less than 3 hours)

    low_duration_boxes = box_features[:, columns_dict['boxes_duration_column']] < minimum_duration_allowed
    invalid_boxes = invalid_boxes | low_duration_boxes

    # Remove boxes with very high duration (greater than 16 hours)

    high_duration_boxes = box_features[:, columns_dict['boxes_duration_column']] > maximum_duration_allowed
    invalid_boxes = invalid_boxes | high_duration_boxes

    # Clean the boxes

    box_data, box_features = clean_boxes(invalid_boxes, box_data_flat, box_features, columns_dict, dl_debug, config,
                                         hod_matrix, 'l1')

    # Remove noise boxes occurring in either day/night

    n_boxes = box_features.shape[0]
    night_boolean = ((box_features[:, columns_dict['night_boolean']] >= charging_hours[0]) |
                     (box_features[:, columns_dict['night_boolean']] <= charging_hours[1])).astype(int)

    night_count_fraction = np.sum(night_boolean) / n_boxes
    day_count_fraction = 1 - night_count_fraction
    night_boolean = night_boolean == 1
    day_boolean = ~night_boolean

    # If the night count boxes are higher than calculate the features accordingly
    if night_count_fraction > day_count_fraction:
        median_night_duration = np.nanmedian(box_features[night_boolean, columns_dict['boxes_duration_column']])
        median_night_amplitude = np.nanmedian(box_features[night_boolean, columns_dict['boxes_median_energy']])
        median_night_dur_amp = median_night_duration*median_night_amplitude

        outlier_duration_boxes = \
            abs(median_night_duration - box_features[:, columns_dict['boxes_duration_column']]) >= \
            dur_amp_threshold*median_night_duration
        outlier_amp_boxes = \
            abs(median_night_amplitude - box_features[:, columns_dict['boxes_median_energy']]) >= \
            dur_amp_threshold*median_night_amplitude
        outlier_dur_amp_boxes = \
            abs(median_night_dur_amp -
                (box_features[:, columns_dict['boxes_duration_column']]*box_features[:, columns_dict['boxes_median_energy']])) \
            >= dur_amp_threshold*median_night_dur_amp

        invalid_boxes = (outlier_duration_boxes | outlier_amp_boxes | outlier_dur_amp_boxes) & day_boolean

    else:
        median_day_duration = np.nanmedian(box_features[day_boolean, columns_dict['boxes_duration_column']])
        median_day_amplitude = np.nanmedian(box_features[day_boolean, columns_dict['boxes_median_energy']])
        median_day_dur_amp = median_day_duration*median_day_amplitude

        outlier_duration_boxes = \
            abs(median_day_duration - box_features[:, columns_dict['boxes_duration_column']]) \
            >= dur_amp_threshold*median_day_duration
        outlier_amp_boxes = \
            abs(median_day_amplitude - box_features[:, columns_dict['boxes_median_energy']]) \
            >= dur_amp_threshold*median_day_amplitude
        outlier_dur_amp_boxes = \
            abs(median_day_dur_amp -
                (box_features[:, columns_dict['boxes_duration_column']]*box_features[:, columns_dict['boxes_median_energy']])) \
            >= dur_amp_threshold*median_day_dur_amp

        invalid_boxes = (outlier_duration_boxes | outlier_amp_boxes | outlier_dur_amp_boxes) & night_boolean

    # Clean the boxes
    box_data, box_features = clean_boxes(invalid_boxes, box_data, box_features, columns_dict, dl_debug, config,
                                         hod_matrix, 'l1')

    # Remove boxes with outlier amplitudes
    median_amplitude = np.nanmedian(box_features[:, columns_dict['boxes_energy_per_point_column']])
    invalid_boxes = abs(median_amplitude - box_features[:, columns_dict['boxes_energy_per_point_column']]) \
                    >= invalid_box_thr*median_amplitude

    # Clean the boxes
    box_data, box_features = clean_boxes(invalid_boxes, box_data, box_features, columns_dict, dl_debug, config,
                                         hod_matrix, 'l1')
    box_data = box_data.reshape(dl_debug.get('rows'), dl_debug.get('cols'))

    return box_data
