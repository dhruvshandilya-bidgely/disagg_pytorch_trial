"""
Author - Nisha Agarwal
Date - 10/9/20
Utils functions required in various itemization modules
"""

# Import python packages

import copy
import numpy as np
from itertools import groupby

from numpy.random import RandomState

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.init_itemization_config import init_itemization_params

from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile


def fill_array(array, start, end, value):

    """
    fill values in a array for the given interval

    Parameters:
        array                    (np.ndarray)   : Target array
        start                    (int)          : Start index
        end                      (int)          : end index
        value                    (int)          : value to be filled

    Returns:
        array                    (np.ndarray)   : Target array with filled value
    """

    index_array = get_index_array(start, end, len(array))

    array[index_array] = value

    return array


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


def rolling_func(array, interval, avg=True):

    """
    Calculate rolling avg for the given window size

    Parameters:
        array                       (np.ndarray)        : target array
        interval                    (int)               : window size
        avg                         (bool)              : avg is calculated if this bool is True else sum

    Returns:
        out                         (np.ndarray)        : calculated rolling average array
    """

    # Taking deepcopy of input data

    data = copy.deepcopy(array)

    # Window size for the sum

    window = int(2*interval + 1)

    # Padding size for the array

    padding = int(2*interval // 2)

    factor = int(2*interval)

    # Pad the start and end of array with circular data

    data = np.r_[data[-factor:], data, data[:factor]]
    a = data.cumsum()
    a[window:] = a[window:] - a[:-window]

    # Subtract the padding array portion from output

    out = a[(factor + padding):-padding]

    if avg:
        out = out / (2*interval + 1)

    return out


def rolling_func_along_row(array, interval, avg=True):

    """
    Calculate rolling avg for the given window size

    Parameters:
        array                       (np.ndarray)        : target array
        interval                    (int)               : window size
        avg                         (bool)              : avg is calculated if this bool is True else sum

    Returns:
        rolling_avg                 (np.ndarray)        : calculated rolling average array
    """

    # Taking deepcopy of input data

    data = copy.deepcopy(array)

    # Window size for the sum

    window = int(2*interval + 1)

    # Padding size for the array

    padding = int(2*interval // 2)

    factor = int(2*interval)

    # Pad the start and end of array with circular data

    data = np.r_[data[-factor:, :], data, data[:factor, :]]
    a = data.cumsum(axis=0)
    a[window:, :] = a[window:, :] - a[:-window, :]

    # Subtract the padding array portion from output

    out = a[(factor + padding):-padding, :]

    if not(window % 2):
        out = out[:-1]

    if avg:
        out = out / (2*interval + 1)

    return out


def rolling_func_along_col(array, interval, avg=True):

    """
    Calculate rolling avg for the given window size

    Parameters:
        array                       (np.ndarray)        : target array
        interval                    (int)               : window size
        avg                         (bool)              : avg is calculated if this bool is True else sum

    Returns:
        rolling_avg                 (np.ndarray)        : calculated rolling average array
    """

    # Taking deepcopy of input data

    data = copy.deepcopy(array)

    # Window size for the sum

    window = int(2*interval + 1)

    # Padding size for the array

    padding = int(2*interval // 2)

    factor = int(2*interval)

    # Pad the start and end of array with circular data

    data = np.c_[data[:, -factor:], data, data[:, :factor]]
    a = data.cumsum(axis=1)
    a[:, window:] = a[:, window:] - a[:, :-window]

    # Subtract the padding array portion from output

    out = a[:, (factor + padding):-padding]

    if not(window % 2):
        out = out[:, :-1]

    if avg:
        out = out / (2*interval + 1)

    return out


def resample_day_data(data, total_samples):

    """
    This function resamples data, to the number of samples required,, eg 15min to 30 min user data conversion

    Parameters:
        data                       (np.ndarray)        : target array
        total_samples              (int)               : number of target samples in a day

    Returns:
        resampled_data             (np.ndarray)        : resampled array
    """

    total_samples = int(total_samples)

    samples_in_an_hour = len(data[0]) / (total_samples)

    # no sampling required

    if samples_in_an_hour == 1:

        return data

    # Downsample data

    elif samples_in_an_hour > 1:

        samples_in_an_hour = int(samples_in_an_hour)

        aggregated_data = np.zeros(data.shape)

        for sample in range(samples_in_an_hour):

            aggregated_data = aggregated_data + np.roll(data, sample)

        resampled_data = aggregated_data[:, np.arange(samples_in_an_hour-1, len(data[0]), samples_in_an_hour)]

    # Upsample data

    else:

        resampled_data = np.zeros((len(data), total_samples))

        for sample in range(int(1/samples_in_an_hour)):

            resampled_data[:, np.arange(sample, total_samples, int(1/samples_in_an_hour))] = data

        resampled_data = resampled_data * samples_in_an_hour

    return resampled_data


def get_2d_arr(tou_item, app_list, app_name):

    """
    utils function to fetch appliance consumption from a 3d array containing output of all the appliances

        Parameters:
            tou_item       (np.ndarray)        : 3d array containing output of all the appliances
            app_list       (np.ndarray)        : list of all appliances
            app_name       (str)               : Appliance name

    """

    if app_name not in app_list:
        return np.zeros_like(tou_item[0, :, :])

    idx = np.where(app_list == app_name)[0][0]

    return tou_item[idx, :, :]


def cap_app_consumption(total_consumption):

    """
    Cap the behavioral appliance consumption at 95th percentile

        Parameters:
            total_consumption       (np.ndarray)        : TS level output before capping

        Returns:
            total_consumption       (np.ndarray)        : TS level output after capping
    """

    total_consumption[total_consumption == 0] = np.nan
    energy_cap = superfast_matlab_percentile(total_consumption, 90, axis=0)
    energy_cap = np.nan_to_num(energy_cap)
    total_consumption = np.minimum(total_consumption, energy_cap[None, :])

    if np.any(total_consumption > 0):
        cap = np.percentile(total_consumption[total_consumption > 0], 95)
        total_consumption[total_consumption > cap] = cap

    total_consumption = np.nan_to_num(total_consumption)

    return total_consumption


def fill_arr_based_seq_val(initial_arr, target_arr, thres, check_val, update_val, overnight_tag=1):

    """
    update array based on seq value

    Parameters:
        initial_arr                (np.ndarray)        : initial array for which sequence is checked
        target_arr                 (np.ndarray)        : array to be updated based on seq value
        thres                      (int)               : threshold for length of sequence
        check_val                  (int)               : value to be checked for each sequence
        update_val                 (int)               : value to be updated for each sequence
        overnight_tag              (int)               : flag for considering seq array as a circular array

    Returns:
        target_arr                 (np.ndarray)        : array to be updated based on seq value
    """

    seq_label = 0
    seq_start = 1
    seq_end = 2
    seq_len = 3

    seq = find_seq(initial_arr, np.zeros_like(initial_arr), np.zeros_like(initial_arr), overnight=overnight_tag)

    for i in range(len(seq)):
        if seq[i, seq_label] == check_val and seq[i, seq_len] < thres:
            target_arr[seq[i, seq_start]:seq[i, seq_end] +1] = update_val

    return target_arr


def fill_arr_based_seq_val_for_valid_boxes(seq, valid_boxes, target_arr, check_val, update_val):

    """
    update array based on seq value

    Parameters:
        seq                        (np.ndarray)        : seq that needs to be checked before updating target array
        valid_boxes                (np.ndarray)        : flag array that is checked before updating target array
        target_arr                 (np.ndarray)        : array to be updated based on seq value
        check_val                  (int)               : value to be checked for each sequence
        update_val                 (int)               : value to be updated for each sequence

    Returns:
        target_arr                 (np.ndarray)        : array to be updated based on seq value
    """

    seq_start = 1
    seq_end = 2

    for i in range(len(seq)):
        if valid_boxes[i] == check_val:
            target_arr[seq[i, seq_start]:seq[i, seq_end] + 1] = update_val

    return target_arr


def fill_circular_arr_based_seq_val_for_valid_boxes(seq, valid_boxes, target_arr, length, check_val, update_val):

    """
    update array based on seq value

    Parameters:
        seq                        (np.ndarray)        : seq that needs to be checked before updating target array
        valid_boxes                (np.ndarray)        : flag array that is checked before updating target array
        target_arr                 (np.ndarray)        : array to be updated based on seq value
        check_val                  (int)               : value to be checked for each sequence
        update_val                 (int)               : value to be updated for each sequence
    Returns:
        target_arr                 (np.ndarray)        : array to be updated based on seq value
    """

    seq_start = 1
    seq_end = 2

    for i in range(len(seq)):
        if valid_boxes[i] == check_val:
            target_arr[get_index_array(seq[i, seq_start],seq[i, seq_end], length)] = update_val

    return target_arr


def add_noise_to_stat_app_est(consumption, start_time, end_time, usage_hours, total_samples, amplitude):

    """
    Add noise to tou level cooking estimate

       Parameters:
           consumption           (np.ndarray)         : tou level estimate before addition of noise
           start_time            (float)              : appliance usage start time
           end_time              (float)              : appliance usage end time
           usage_hours           (float)              : appliance usage duration
           total_samples         (int)                : total number of samples in a day
           amplitude             (float)              : Amplitude of appliance usage

       Returns:
           consumption           (np.ndarray)         : tou level estimate after addition of noise
       """

    # adding slight noise to ts level initialized stat app consumption to add some randomness
    # this also depends upon sampling rate of the user to adjust any sampling rate specific under/overest

    val = 0.4

    samples_in_hr = total_samples / Cgbdisagg.HRS_IN_DAY
    total_samples = int(total_samples)

    seed = RandomState(random_gen_config.seed_value)

    if samples_in_hr == 1:
        val = 0.9
    if samples_in_hr == 2:
        val = 0.7

    temp_consumption = np.zeros_like(consumption)

    # adding noise between start and end time of the stat app activity

    temp_consumption[:, np.arange(start_time, end_time).astype(int) % total_samples] = amplitude

    mid_interval = np.arange(start_time + usage_hours / 4, end_time - usage_hours / 4).astype(int) % total_samples
    randomness = seed.normal(0.5, 0.3, consumption.shape)
    randomness[:, mid_interval] = seed.normal(0.8, 0.1, randomness[:, mid_interval].shape)
    randomness[randomness < 0.7] = 0

    temp_consumption = np.multiply(temp_consumption, randomness)
    consumption = consumption + temp_consumption
    mean_consumption = np.mean(temp_consumption[np.nonzero(temp_consumption)])

    # adding noise slightly before start time of the stat app activity

    temp_consumption_copy = np.zeros_like(consumption)
    temp_consumption_copy[:, np.arange(start_time - usage_hours / 4, start_time + 1).astype(int) % total_samples] = \
        seed.normal(0.4, 0.3, temp_consumption_copy[:, np.arange(start_time - usage_hours / 4, start_time + 1).astype(int) % total_samples].shape) * mean_consumption
    temp_consumption_copy[temp_consumption_copy < val * mean_consumption] = 0

    consumption = consumption + temp_consumption_copy

    # adding noise slightly after end time of the stat app activity

    temp_consumption_copy = np.zeros(consumption.shape)
    temp_consumption_copy[:, np.arange(end_time, end_time + usage_hours / 4 + 1).astype(int) % total_samples] =\
        seed.normal(0.4, 0.3, temp_consumption_copy[:, np.arange(end_time, end_time + usage_hours / 4 + 1).astype(int) % total_samples].shape) * mean_consumption
    temp_consumption_copy[temp_consumption_copy < 0.9 * mean_consumption] = 0
    consumption = consumption + temp_consumption_copy

    consumption[consumption < 0.1] = 0
    consumption = np.nan_to_num(consumption)

    return consumption


def add_noise_in_laundry(consumption, start_time, end_time, usage_hours, total_samples, amplitude):

    """
        Add noise to tou level cooking estimate

        Parameters:
            consumption           (np.ndarray)         : tou level estimate before addition of noise
            start_time            (float)              : appliance usage start time
            end_time              (float)              : appliance usage end time
            usage_hours           (float)              : appliance usage duration
            total_samples         (int)                : total number of samples in a day
            amplitude             (float)              : Amplitude of appliance usage

        Returns:
            consumption           (np.ndarray)         : tou level estimate after addition of noise
    """

    # adding slight noise to ts level initialized stat app consumption to add some randomness
    # this also depends upon sampling rate of the user to adjust any sampling rate specific under/overest

    samples_per_hour = total_samples / Cgbdisagg.HRS_IN_DAY

    seed = RandomState(random_gen_config.seed_value)

    temp_consumption = np.zeros(consumption.shape)

    temp_consumption[:, np.arange(start_time, end_time).astype(int) % total_samples] = amplitude

    # adding noise between start and end time of the stat app activity

    mid_interval = np.arange(start_time, end_time).astype(int) % total_samples
    randomness = seed.normal(0.8, 0.1, consumption.shape)
    randomness[:, mid_interval] = seed.normal(0.9, 0.05, randomness[:, mid_interval].shape)
    randomness[randomness < 0.7] = 0
    temp_consumption = np.multiply(temp_consumption, randomness)
    consumption = consumption + temp_consumption

    # adding noise slightly before start time of the stat app activity

    temp_consumption2 = np.zeros(consumption.shape)
    temp_consumption2[:, np.arange(end_time, end_time + usage_hours / 4).astype(int) % total_samples] = \
        seed.normal(0.8, 0.05, temp_consumption2[:, np.arange(end_time, end_time + usage_hours / 4).astype(int) % total_samples].shape) * amplitude
    temp_consumption2[temp_consumption2 < 0.79 * amplitude] = 0
    consumption = consumption + temp_consumption2
    consumption[consumption < 0.1] = 0

    # adding noise slightly after end time of the stat app activity

    temp_consumption2 = np.zeros(consumption.shape)
    temp_consumption2[:, np.arange(start_time, start_time - usage_hours / 4).astype(int) % total_samples] = \
        seed.normal(0.8, 0.05, temp_consumption2[:, np.arange(start_time, start_time - usage_hours / 4).astype(int) % total_samples].shape) * amplitude
    temp_consumption2[temp_consumption2 < 0.82 * amplitude] = 0
    consumption = consumption + temp_consumption2
    consumption[consumption < 0.1] = 0

    consumption = np.nan_to_num(consumption)

    samples_per_hour = int(samples_per_hour)

    for i in range(len(consumption)):
        rolling_window = seed.choice(np.arange(-2 * samples_per_hour, 2 * samples_per_hour + 1))
        consumption[i] = np.roll(consumption[i], rolling_window)

    consumption = np.nan_to_num(consumption)

    return consumption


def get_idx(app_list, app):

    """
    Get app index
    """

    return np.where(app_list == app)[0][0]


def find_seq(labels, arr, derivative, overnight=True):

    """
    calculate sequence of labels in an array, also calculates different attributes for individual sequence
    Parameters:
        labels                (np.ndarray)        : array containing labels
        arr                   (np.ndarray)        : array containing values for all timestamp
        derivative            (np.ndarray)        : array containing derivative of values
        overnight             (int)               : true if array should be considered as cylindrical
    Returns:
        res                   (np.ndarray)        : final array containing start, end, and information for each seq
    """

    if np.max(labels) == np.min(labels) and np.all(arr == 0) and np.all(derivative == 0):
        res = np.array([[labels[0], 0, len(labels), len(labels)]])

    elif np.max(labels) == np.min(labels) and not (np.all(arr == 0) and np.all(derivative == 0)):
        max_derivative = np.max(derivative)
        low_perc = np.percentile(arr, 5)
        high_perc = np.percentile(arr, 95)
        mid_perc = np.percentile(arr, 50)
        derivative_strength = 0

        if labels[0] == 1:
            derivative_strength = np.sum(derivative > 0)
        elif labels[0] == -1:
            derivative_strength = np.sum(derivative < 0)

        res = np.array([[labels[0], 0, len(labels), len(labels), max_derivative, low_perc, high_perc, 0, derivative_strength, mid_perc]])

    else:

        # Initialise the result array

        get_seq_params = True

        if np.all(arr == 0) and np.all(derivative == 0):
            get_seq_params = False

        res = []
        start_idx = 0

        # Get groups

        group_list = groupby(labels)

        for seq_num, seq in group_list:

            # Get number of elements in the sequence

            seq_len = len(list(seq))

            if get_seq_params:

                params = [start_idx, seq_len, seq_num]

                temp_res = calculate_additional_seq_params(params, arr, labels, derivative)

            else:
                temp_res = [seq_num, start_idx, start_idx + seq_len - 1, seq_len]

            start_idx += seq_len
            res.append(temp_res)

        res = np.array(res)

        # Handle overnight cases

        res = handle_overnight_seq(arr, res, get_seq_params, labels, overnight)

    return res


def get_index(app_name):

    """
    fetch index of appliance from the list of all target appliances

    Parameters:
        app_name                    (str)           : target Appliance name

    """

    appliance_list = np.array(["ao", "ev", "cooling", "heating", 'li', "pp", "ref", "va1", "va2", "wh", "ld", "ent", "cook"])

    return int(np.where(appliance_list == app_name)[0][0])


def calculate_additional_seq_params(seq_params, arr, labels, derivative):

    """
    calculate extra parameters for a given sequence
    Parameters:
        seq_params            (list)              : list of start index, length of seq label of the given seq
        arr                   (np.ndarray)        : original array
        labels                (np.ndarray)        : array containing labels
        labels                (np.ndarray)        : derivative of the original array
    Returns:
        res                   (list)              : list of calculated params
    """

    start_idx = seq_params[0]
    seq_len = seq_params[1]
    seq_num = seq_params[2]

    index_array = np.arange(start_idx, start_idx + seq_len) if (start_idx + seq_len) <= len(labels) else \
        np.append(np.arange(start_idx, len(labels)), np.arange(0, start_idx + seq_len - len(labels)))

    index_array = index_array.astype(int)

    max_derivative = np.max(derivative[index_array])

    # percentile values in a given seq

    low_perc = np.percentile(arr[index_array], 5)
    high_perc = np.percentile(arr[index_array], 95)
    mid_perc = np.percentile(arr[index_array], 50)

    # Magnitude of derivative of a given seq

    net_derivative = arr[(start_idx + seq_len - 1) % len(arr)] - arr[(start_idx - 1) % len(arr)]

    derivative_strength = 0

    if seq_num == 1:
        derivative_strength = np.sum(derivative[index_array] > 0)
    elif seq_num == -1:
        derivative_strength = np.sum(derivative[index_array] < 0)

    res = [seq_num, start_idx, start_idx + seq_len - 1, seq_len, max_derivative, low_perc, high_perc,
           net_derivative, derivative_strength, mid_perc]

    return res


def handle_overnight_seq(arr, res, get_seq_params, labels, overnight):

    """
    submodule of calculating sequence of labels in an array, which handles combining of overnight sequences
    Parameters:
        arr                   (np.ndarray)        : array containing values for all timestamp
        res                   (np.ndarray)        : intermediate seq results
        get_seq_params        (bool)              : Bool to check whether we need to calculate all the seq params
        labels                (np.ndarray)        : array containing labels
        overnight             (int)               : true if array should be considered as cylindrical
    Returns:
        res                   (np.ndarray)        : final array containing start, end, and information for each seq
    """

    config = init_itemization_params().get('seq_config')

    if res[0, config.get('label')] == res[-1, config.get('label')] and overnight:

        # combining first and the last seq

        res[0, config.get('start')] = res[-1, config.get('start')]
        res[0, config.get('length')] = res[0, config.get('length')] + res[-1, config.get('length')]

        if get_seq_params:

            # combining first and the last seq, and also calculating other required sequence parameters

            res[0, config.get('deri_strength')] = res[0, config.get('deri_strength')] + res[-1, config.get('deri_strength')]
            res[0, config.get('max_deri')] = max(res[0, config.get('max_deri')], res[-1, config.get('max_deri')])

            start_idx = res[0, config.get('start')]
            seq_len = res[0, config.get('length')]

            index_array = get_index_array(start_idx, start_idx + seq_len, len(labels))

            res[0, config.get('low_perc')] = np.percentile(arr[index_array], 5)
            res[0, config.get('high_perc')] = np.percentile(arr[index_array], 95)
            res[0, config.get('mid_perc')] = np.percentile(arr[index_array], 50)
            res[0, config.get('net_deri')] = arr[int(start_idx + seq_len - 1) % len(arr)] - \
                        arr[int(start_idx - 1) % len(arr)]

        res = res[: -1]

    return res
