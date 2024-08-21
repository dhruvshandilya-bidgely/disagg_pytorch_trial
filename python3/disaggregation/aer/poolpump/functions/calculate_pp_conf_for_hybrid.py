
"""
Author - Nisha Agarwal
Date - 22/1/22
Utility functions to help compute the estimate
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.itemization.aer.functions.itemization_utils import find_seq


def calculate_pp_conf_for_hybrid(pairs, input_data, pp_config, samples, threshold, pp_run_days_arr):

    """
    Calculate pp score based on quality of edges for a given window

     Parameters:
        pairs                      (np.ndarray) : pp disagg schdules pair
        input_data                 (np.ndarray) : input data
        pp_config                  (dict)       : pp disagg cofig
        samples                    (int)        : samples in an hour
        threshold                  (int)        : threshold used to calculate score based on consistency
        pp_run_days_arr            (np.ndarray) : pp disagg runs info

     Returns:
         conf_val                   (float)     : confidence value
     """

    duration = 20

    value = np.fmax(0, pairs[:, 3] - pairs[:, 2])
    value = (value / 20) + 2
    value = value.sum()

    metrics = np.zeros((int(value) + 20) * 6)

    c = 0

    # calculate edge quality score for both side edges and their nearby points

    for m in np.arange(len(pairs)):

        if (pairs[m][3] - pairs[m, 2]) < 15:
            continue

        start = pairs[m, 2]
        end = pairs[m, 3]

        for n in range(start, end, duration):
            data = pp_config.get('input_data')[n:n + duration][:, get_index_array(pairs[m][0], pairs[m][1] - 1, 24 * samples)]

            data1 = input_data[n:n + duration][:, pairs[m][1]]

            calculate_hybrid_conf_score(metrics, c, samples, data, data1, threshold)

            c = c + 1

            data1 = input_data[n:n + duration][:, (pairs[m][1] - 1) % (samples * 24)]

            calculate_hybrid_conf_score(metrics, c, samples, data, data1, threshold)

            c = c + 1

            data1 = input_data[n:n + duration][:, (pairs[m][1] + 1) % (samples * 24)]

            calculate_hybrid_conf_score(metrics, c, samples, data, data1, threshold)

            c = c + 1

            data2 = input_data[n:n + duration][:, pairs[m][0]]

            calculate_hybrid_conf_score(metrics, c, samples, data, data2, threshold)

            c = c + 1

            data2 = input_data[n:n + duration][:, (pairs[m][0] - 1) % (samples * 24)]

            calculate_hybrid_conf_score(metrics, c, samples, data, data2, threshold)

            c = c + 1

            data2 = input_data[n:n + duration][:, (pairs[m][0] + 1) % (samples * 24)]

            calculate_hybrid_conf_score(metrics, c, samples, data, data2, threshold)

            c = c + 1

    metrics[metrics == 0] = -1

    if np.any(metrics > -1):
        conf_val = np.percentile(np.fmax(0, metrics[metrics > -1]), 95)
    else:
        conf_val = 0.65

    total_pp_run_days = np.count_nonzero(pp_run_days_arr)

    # Calculate number of poolpump usage days

    pp_arr = np.zeros_like(pp_run_days_arr)

    pp_seq = find_seq(pp_run_days_arr > 0, np.zeros_like(pp_run_days_arr), np.zeros_like(pp_run_days_arr))

    # filling pp days with small gaps

    bool_arr = np.logical_and(pp_seq[:, 0] == 0, pp_seq[:, 3] <= 5)

    if np.any(bool_arr):
        pp_seq[bool_arr, 0] = 1

    for i in range(len(pp_seq)):
        if pp_seq[i, 0]:
            pp_arr[pp_seq[i, 1]: pp_seq[i, 2]] = 1

    pp_seq = find_seq(pp_run_days_arr > 0, np.zeros_like(pp_run_days_arr), np.zeros_like(pp_run_days_arr))

    bool_arr = np.logical_and(pp_seq[:, 0] == 1, pp_seq[:, 3] <= 3)

    if np.any(bool_arr):
        pp_seq[bool_arr, 0] = 0

    for i in range(len(pp_seq)):

        if not pp_seq[i, 0]:
            pp_arr[pp_seq[i, 1]: pp_seq[i, 2]] = 0

    total_days = np.sum(np.sum(pp_config.get('input_data'), axis=1) > 0)
    days_score = total_pp_run_days / total_days

    days_score = np.fmin(1, 0.3 + days_score)

    # Calculating final confidence score for hybrid module

    conf_val = 0.85 * conf_val + 0.15 * days_score

    if pp_config.get('hybrid_conf_val') is None:
        conf_val = np.fmax(0.45, conf_val)
    else:
        conf_val = max(pp_config["hybrid_conf_val"], np.fmax(0.45, conf_val))

    return conf_val


def calculate_hybrid_conf_score(metrics, index, samples, pp_data, edge_data, threshold):

    """
    Calculate pp score based on quality of edges for a given window

    Parameters:
        metrics             (np.ndarray)      : array containing scores for diff windows
        index               (int)             : index for the given window

    Returns:
        metrics             (np.ndarray)      : updateed score
    """

    metrics[index] = 1 - ((np.percentile(edge_data, 75) - np.percentile(edge_data, 25)) / threshold)

    score1 = np.median(edge_data) * samples
    score1 = np.fmin(1, (0.6 * score1 - 100) / 200 - 0.1)
    score2 = np.sum(pp_data > np.percentile(edge_data, 35)) / np.size(pp_data)
    score2 = np.fmin(1, (score2 - 0.7) / 0.25)

    if score2 < 0.8:
        metrics[index] = metrics[index] * 0.5 + score1 * 0.25 + score2 * 0.25
    else:
        metrics[index] = metrics[index] * 0.6 + score1 * 0.25 + score2 * 0.15

    return metrics


def get_index_array(start, end, length):

    """
    calculate index array for given start and end index

    Parameters:
        start                    (int)          : Start index
        end                      (int)          : end index
        length                   (int)          : length of target array

    Returns:
        index_array              (np.ndarray)   : Final array of target index
    """

    start = start % length
    end = end % length

    if start <= end:
        index_array = np.arange(start, end + 1).astype(int)

    else:
        index_array = np.append(np.arange(start, length), np.arange(0, end + 1)).astype(int)

    index_array = (index_array.astype(int) % length).astype(int)

    return index_array
