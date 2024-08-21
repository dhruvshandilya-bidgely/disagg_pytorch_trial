"""
Author - Sahana M
Date - 14-Nov-2023
Module containing Utilities
"""

# import python packages
import numpy as np
from copy import deepcopy

# import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.ev.functions.deep_learning.init_dl_config import FeatureCol

metrics = {
    'min': np.nanmin,
    'max': np.nanmax,
    'mean': np.nanmean,
    'sum': np.nansum,
    'std': np.nanstd,
    'median': np.nanmedian,
}


def min_amplitude_check(curr_partition,  min_amplitude, i, prediction_confidences, config, predictions_given=True):
    """
    Function to get the amplitude
    Parameters:
        curr_partition              (np.ndarray)            : Current partition data
        min_amplitude               (int)                   : Minimum amplitude
        i                           (int)                   : Index
        prediction_confidences      (list)                  : Prediction confidences
        config                      (Dict)                  : Configurations dictionary
        predictions_given           (Boolean)               : Predictions given status
    Returns:
        curr_partition              (np.ndarray)            : Current partition data
    """

    if predictions_given:
        if prediction_confidences[i] >= config.get('confidence_threshold'):
            curr_partition[curr_partition < min_amplitude] = 0
        else:
            curr_partition[curr_partition > 0] = 0
    else:
        curr_partition[curr_partition < min_amplitude] = 0
    return curr_partition


def min_duration_check(curr_partition, factor, min_duration):
    """
    Function to perform minimum duration check
    Parameters:
        curr_partition              (np.ndarray)               : Current partition data
        factor                      (int)                      : Sampling rate factor
        min_duration                (int)                      : Minimum duration threshold
    Returns:
        curr_partition              (np.ndarray)               : Current partition data
    """

    # Extract the required parameters

    row, col = curr_partition.shape
    flat_cur_partition = curr_partition.flatten()
    flat_cur_partition_bool = flat_cur_partition > 0
    box_energy_idx = np.diff(np.r_[0, flat_cur_partition_bool])
    box_start_idx = np.where(box_energy_idx == 1)[0]
    box_end_idx = np.where(box_energy_idx == -1)[0]

    # Identify the boxes duration
    if len(box_start_idx) > len(box_end_idx):
        box_start_idx = box_start_idx[:-1]
    box_durations = box_end_idx - box_start_idx
    box_durations = box_durations/factor

    # Get the boxes where duration is violated
    violating_boxes = np.where(box_durations <= min_duration)[0]

    if len(violating_boxes):
        for i in range(len(violating_boxes)):
            violating_index = int(violating_boxes[i])
            start_idx = int(box_start_idx[violating_index])
            end_idx = int(box_end_idx[violating_index])
            flat_cur_partition[start_idx: end_idx] = 0

    curr_partition = flat_cur_partition.reshape(row, col)
    return curr_partition


def get_curr_partitions(i, raw_data,  heat_pot_data, cool_pot_data, temp_data, s_label_data, dl_debug):
    """
    Identify the current partitions
    Parameters:
        i                           (int)                   : Current partition
        raw_data                    (np.ndarray)            : Raw data
        heat_pot_data               (np.ndarray)            : Heating potential data
        cool_pot_data               (np.ndarray)            : Cooling potential data
        temp_data                   (np.ndarray)            : Temperature data
        s_label_data                (np.ndarray)            : Season label data
        dl_debug                    (Dict)                  : Debug dictionary
    Returns:
        curr_partition_original     (np.ndarray)            : Current partition data
        heating_potential           (np.ndarray)            : Heating potential data
        cooling_potential           (np.ndarray)            : Cooling potential data
        temperature                 (np.ndarray)            : Temperature data
        s_label                     (np.ndarray)            : Season label data
    """

    # Extract required data
    total_days = dl_debug.get('total_days')
    initial_cols = dl_debug.get('initial_cols')
    prt_size = dl_debug.get('config').get('prt_size')

    padding = False
    start_day = i * prt_size
    end_day = start_day + prt_size

    padding_rows = []
    if end_day >= total_days:
        padding = True
        padding_days = end_day - total_days
        padding_rows = np.zeros((padding_days, initial_cols))

    # Perform padding for the data
    curr_partition_original = deepcopy(raw_data[start_day: end_day, :])
    heating_potential = deepcopy(heat_pot_data[start_day: end_day, :])
    cooling_potential = deepcopy(cool_pot_data[start_day: end_day, :])
    temperature = deepcopy(temp_data[start_day: end_day, :])
    s_label = deepcopy(s_label_data[start_day: end_day, :])

    if padding:
        curr_partition_original = np.r_[curr_partition_original, padding_rows]
        heating_potential = np.r_[heating_potential, padding_rows]
        cooling_potential = np.r_[cooling_potential, padding_rows]
        temperature = np.r_[temperature, padding_rows]
        s_label = np.r_[s_label, padding_rows]

    return curr_partition_original, heating_potential, cooling_potential, temperature, s_label


def get_curr_partitions_l1(i, raw_data, dl_debug):
    """
    Identify the current partitions
    Parameters:
        i                           (int)                   : Current partition
        raw_data                    (np.ndarray)            : Raw data
        dl_debug                    (Dict)                  : Debug dictionary
    Returns:
        curr_partition_original     (np.ndarray)            : Current partition data
    """

    # Extract required data
    total_days = dl_debug.get('total_days')
    initial_cols = dl_debug.get('initial_cols')
    prt_size = dl_debug.get('config').get('prt_size')

    padding = False
    start_day = i * prt_size
    end_day = start_day + prt_size

    padding_rows = []

    if end_day >= total_days:
        padding = True
        padding_days = end_day - total_days
        padding_rows = np.zeros((padding_days, initial_cols))

    # Perform padding for the data
    curr_partition_original = deepcopy(raw_data[start_day: end_day, :])

    if padding:
        curr_partition_original = np.r_[curr_partition_original, padding_rows]

    return curr_partition_original


def get_start_end_idx(arr):
    """
    Function to get the start and end index
    Parameters:
        arr                 (np.ndarray)            : Data array
    Returns:
        start_idx           (np.ndarray)            : Start indexes
        end_idx             (np.ndarray)            : End indexes
    """
    bool_idx = arr > 0
    start_end_idx = np.diff(np.r_[0, bool_idx, 0])
    start_idx = np.where(start_end_idx == 1)[0]
    end_idx = np.where(start_end_idx == -1)[0]

    return start_idx, end_idx


def get_wrong_indexes(identified_boxes_distribution, ctype='L2'):
    """
    Function to identify the wrong indexes which means the boxes where the duration is abnormal hence those strikes are
    marked as wrong indexes which can be rejected later.
    Parameters:
        identified_boxes_distribution           (np.ndarray)            : Identify the boxes distribution
        ctype                                   (string)                : Get the charger type
    Returns:
        wrong_indexes                           (np.ndarray)            : Wrong indexes
    """

    # Assign the threshold based on the charger type
    if ctype == 'L1':
        min_threshold = 1
    else:
        min_threshold = 3

    # Identify the strikes running for less than a few hours

    wrong_indexes = identified_boxes_distribution < min_threshold
    bool_idx = identified_boxes_distribution >= min_threshold
    start_end_idx = np.diff(np.r_[0, bool_idx, 0])
    start_idx = np.where(start_end_idx == 1)[0]
    end_idx = np.where(start_end_idx == -1)[0]
    hour_diff = end_idx - start_idx
    small_dur_hours = (hour_diff <= 2)

    # Mark the wrong indexes

    for i in range(len(small_dur_hours)):
        if small_dur_hours[i]:
            s_idx = int(start_idx[i])
            e_idx = int(end_idx[i])
            wrong_indexes[s_idx: e_idx] = True

    return wrong_indexes


def get_partition_distribution(data, ctype='L2'):
    """
    Function to get the partition distribution
    Parameters:
        data                (np.ndaraay)            : Data
        ctype               (str)                   : Charger type
    Returns:
        boxes_distribution  (np.ndarray)            : Boxes consumption distribution
    """

    # Get the distribution for the data
    if np.sum(data):
        non_zero_usage_days = np.sum(data, axis=1) > 0
        boxes_distribution = np.sum(data[non_zero_usage_days] > 0, axis=0) / \
                             np.sum(np.sum(data[non_zero_usage_days] > 0, axis=0)) * 100
        wrong_indexes = get_wrong_indexes(boxes_distribution, ctype)
        boxes_distribution[wrong_indexes] = 0
    else:
        boxes_distribution = np.zeros(data.shape[1])

    return boxes_distribution


def get_amp_distribution(data, ctype='L2'):
    """
    Function to get the amplitude distribution
    Parameters:
        data                    (np.ndarray)            : Data
        ctype                   (str)                   : Charger type
    Returns:
        amp_distribution        (np.ndarray)            : Amplitude distribution array
    """

    # Set threshold based on the charger type

    if ctype == 'L1':
        percentile_thr = 50
    else:
        percentile_thr = 99

    if np.sum(data):
        non_zero_usage_days = np.sum(data, axis=1) > 0
        amp_distribution = np.percentile(data[non_zero_usage_days], axis=0, q=percentile_thr)
    else:
        amp_distribution = np.zeros(data.shape[1])

    return amp_distribution


def get_prt_start_end_idx(prt_predictions):
    """
    Function to get the start and end indexes
    Parameters:
        prt_predictions                 (np.ndarray)            : Partition predictions
    Returns:
        start_idx                       (np.ndarray)            : Boolean array for start indexes
        end_idx                         (np.ndarray)            : Boolean array for end indexes
    """

    partition_size = 14
    start_idx = np.asarray([x * partition_size for x in range(len(prt_predictions))])
    end_idx = start_idx + partition_size

    return start_idx, end_idx


def get_day_data_2d(input_data, ev_config):

    """
    Covert 2D timestamp data to day level data

        Parameters:
            input_data                   (np.ndarray)         : Input data
            ev_config                    (Dict)               : EV configurations
        Returns:
            day_input_data               (np.ndarray)         : day level input data
            day_output_data              (np.ndarray)         : day level disagg output data
    """

    sampling_rate = ev_config.get('sampling_rate')

    num_pd_in_day = int(Cgbdisagg.SEC_IN_DAY / sampling_rate)
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    day_idx = Cgbdisagg.INPUT_DAY_IDX

    # Prepare day timestamp matrix and get size of all matrices

    day_ts, _, row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True, return_index=True)
    day_ts = np.tile(day_ts, reps=(num_pd_in_day, 1)).transpose()

    # Compute hour of day based indices to use

    col_idx = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - input_data[:, day_idx]
    col_idx = col_idx / Cgbdisagg.SEC_IN_HOUR
    col_idx = (pd_mult * (col_idx - col_idx.astype(int) + input_data[:, Cgbdisagg.INPUT_HOD_IDX])).astype(int)

    # Create day wise 2d arrays for each variable

    epochs_in_a_day = int(Cgbdisagg.HRS_IN_DAY * (Cgbdisagg.SEC_IN_HOUR / sampling_rate))
    no_of_days = len(day_ts)

    day_input_data = np.zeros((no_of_days, epochs_in_a_day, len(input_data[0])))

    day_input_data[row_idx, col_idx, :] = input_data

    day_input_data = np.swapaxes(day_input_data, 0, 1)
    day_input_data = np.swapaxes(day_input_data, 0, 2)

    return day_input_data, row_idx, col_idx


def rolling_function(in_data, window, out_metric='both'):
    """
        Parameters:
            in_data         (np.ndarray)        : Input 21-column matrix
            window          (int)               : Moving window size
            out_metric      (string)            : Type of metric

        Returns:
            output          (np.ndarray)        : Output array based on the asked metric
        """

    # Taking deep copy of the input data and replacing NaN with zeros

    data = deepcopy(in_data)
    data[np.isnan(data)] = 0

    # Making sure window is int
    window = int(window)

    # Creating an empty array to do row-wise calculations

    n_rows = data.shape[0]
    temp = np.zeros(shape=[n_rows, int(window)])

    # Treat odd / even sized windows separately

    if window % 2 == 0:
        # If the window size is n (even) data points
        # Filling columns before the middle column with backward shifted arrays

        for row, column in zip(range(-(window - 3) // 2, 0), range((window - 1) // 2)):
            temp[abs(row):, column] = data[:row]

        # Filling the middle column with original array

        temp[:, (window - 1) // 2] = data

        # Filling columns after the middle column with forward shifted arrays

        for row, column in zip(range(1, (window + 2) // 2), range(window // 2, int(window))):
            temp[:-row, column] = data[row:]
    else:
        # If the window size is n (odd) data points

        # Filling columns before the middle column with backward shifted arrays

        for row, column in zip(range(-(window - 1) // 2, 0), range((window - 1) // 2)):
            temp[abs(row):, column] = data[:row]

        # Filling the middle column with original array

        temp[:, (window - 1) // 2] = data

        # Filling columns after the middle column with forward shifted arrays

        for row, column in zip(range(1, (window + 1) // 2), range((window + 1) // 2, int(window))):
            temp[:-row, column] = data[row:]

        # Calculating the output metric on the data

    if out_metric == 'both':
        # If both min and max are required

        out1 = np.min(temp, axis=1)
        out2 = np.max(temp, axis=1)

        return out1, out2
    else:
        # If a particular output metric is required (from list of functions above)

        output = metrics[out_metric](temp, axis=1)

        return output


def get_static_values(raw_data):
    """
    Function to assign the static values
    Parameters:
        raw_data             (np.ndarray)            : Raw data matrix
    Returns:
        factor                  (int)                   : Factor
        sampling_rate           (int)                   : Sampling rate
    """

    epochs_in_a_day = raw_data.shape[1]
    if epochs_in_a_day == 24:
        sampling_rate = 3600
        factor = 1
    elif epochs_in_a_day == 48:
        sampling_rate = 1800
        factor = 2
    else:
        sampling_rate = 900
        factor = 4

    return factor, sampling_rate


def get_hod_matrix(raw_data, factor):
    """
    Function to get the HOD matrix
    Parameters:
        raw_data         (np.ndarray)        : Raw data matrix
        factor              (int)               : Sampling rate factor
    Returns:
        hod_matrix          (np.ndarray)        : HOD matrix
    """

    # Extract the required variables
    cols = int(raw_data.shape[1])
    hour = 0
    hod_matrix = np.zeros_like(raw_data)

    for i in range(0, cols, factor):
        hod_matrix[:, i:i + factor] = hour
        hour += 1

    return hod_matrix


def extract_box_features(final_boxes_detected, debug, ctype='L2'):
    """
    Function to extract the features of all the EV boxes
    Parameters:
        final_boxes_detected            (np.ndarray)            : Captured EV boxes detected
        debug                           (Dict)                  : Debug dictionary
        ctype                           (string)                : Charger type

    Returns:
        box_features                    (np.ndarray)            : Box features
    """

    # Extract the required variables as per L1 or L2 chargers

    if ctype == 'L1':
        prt_predictions = debug.get('final_partition_predictions_l1')
        prt_confidences = debug.get('prediction_confidences_l1')
    else:
        prt_predictions = debug.get('final_partition_predictions')
        prt_confidences = debug.get('updated_partition_confidences')
        if prt_confidences is None:
            prt_confidences = debug.get('prediction_confidences')

    # Extract required values

    s_label_data = debug.get('s_label_data')
    start_idx, end_idx = get_prt_start_end_idx(prt_predictions)

    box_features = np.c_[prt_predictions, prt_confidences, start_idx, end_idx]
    extra_features = 10
    zero_arr = np.zeros(shape=(box_features.shape[0], extra_features))
    box_features = np.c_[box_features, zero_arr]

    # Extract box features from each partition

    for i in range(len(box_features)):
        start_idx = box_features[i, FeatureCol.PRT_START_IDX].astype(int)
        end_idx = box_features[i, FeatureCol.PRT_END_IDX].astype(int)

        prt_data_2d = final_boxes_detected[start_idx: end_idx, :]
        prt_data_1d = deepcopy(prt_data_2d).flatten()
        prt_s_label_data_2d = s_label_data[start_idx: end_idx, :]
        unique_seasons = np.unique(prt_s_label_data_2d, return_counts=True)
        max_repeated_season = unique_seasons[0][np.argmax(unique_seasons[1])]
        box_features[i, FeatureCol.S_LABEL] = max_repeated_season
        box_features[i, FeatureCol.TOTAL_CONSUMPTION] = np.nansum(prt_data_2d)

        # Get the total number of strikes in this partition

        box_start_indexes, box_end_indexes = get_start_end_idx(prt_data_1d)
        total_boxes = len(box_start_indexes)

        if total_boxes > 0:
            single_box_features = []

            for j in range(total_boxes):
                strike_data = prt_data_1d[box_start_indexes[j]: box_end_indexes[j]]
                strike_data[np.isnan(strike_data)] = 0
                mean_strike_energy = np.mean(strike_data)
                min_strike_energy = np.min(strike_data)
                max_strike_energy = np.max(strike_data)
                strike_auc = np.sum(strike_data)
                strike_duration = len(strike_data)/debug.get('factor')
                temp = np.r_[mean_strike_energy, min_strike_energy, max_strike_energy,  strike_auc, strike_duration]
                single_box_features.append(temp)

            ev_days = np.sum(np.sum(prt_data_2d, axis=1) > 0)
            charging_freq = total_boxes/ev_days
            single_box_features = np.asarray(single_box_features)

            # After all the single boxes features in a partition are extracted, combine their info

            prt_boxes_features = np.mean(single_box_features, axis=0)
            prt_boxes_features = np.c_[prt_boxes_features.reshape(1, -1), total_boxes, charging_freq]
            box_features[i, FeatureCol.MEAN_BOX_ENERGY:FeatureCol.TOTAL_CONSUMPTION] = prt_boxes_features

    return box_features


def get_ev_boxes(in_data, minimum_duration, minimum_energy, factor):
    """
    Find the boxes in the consumption data

    Parameters:
        in_data             (np.ndarray)    : The input data (13-column matrix)
        minimum_duration    (float)         : Minimum allowed duration to qualify as an EV box
        minimum_energy      (np.ndarray)    : The minimum energy value per data point
        factor              (int)           : Number of data points in an hour

    Returns:
        input_data             (np.ndarray)    : The box data output at epoch level
    """

    # Taking deepcopy of input data to avoid scoping issues

    input_data = deepcopy(in_data)

    # Find the window size of the boxes required

    window_size = int(minimum_duration * factor)
    window_half = window_size // 2

    even_window = True if (window_size % 2) == 0 else False

    # Subset consumption data for finding boxes

    energy = input_data

    # Mark all the consumption points above minimum as valid

    valid_idx = np.array(energy >= minimum_energy)

    # Accumulate the valid count over the window

    moving_sum = rolling_function(valid_idx, window_size, 'sum')

    # Consolidate the chunks of valid high energy consumption points

    valid_sum_bool = (moving_sum >= window_size)

    sum_idx = np.where(valid_sum_bool)[0]
    sum_final = deepcopy(sum_idx)

    # Padding the boxes for the first and last window

    for i in range(1, window_half + 1):

        if (i == window_half) and even_window:
            sum_final = np.r_[sum_final, sum_idx + i]
        else:
            sum_final = np.r_[sum_final, sum_idx + i]
            sum_final = np.r_[sum_final, sum_idx - i]

        sum_final = np.sort(np.unique(sum_final))

    # Updating the valid sum bool

    valid_sum_bool[sum_final[sum_final < input_data.shape[0]]] = True

    # Make all invalid consumption points as zero in input data

    input_data[~valid_sum_bool] = 0

    return input_data


def boxes_features(box_input, hod_matrix, factor, ev_config, ctype='l2'):
    """
    Function to get features for EV detection

        Parameters:
            box_input                 (np.ndarray)              : Input box data
            hod_matrix                (np.ndarray)              : Hour of the day matrix
            factor                    (int)                     : Number of data points in an hour
            ev_config                  (dict)                    : Module config dict
            ctype                      (string)                  : Charger type

        Returns:
            box_features              (np.ndarray)              : Calculated box features

    """

    #  Getting parameter dicts to be used in this function

    if ctype == 'l2':
        columns_dict = ev_config['box_features_dict']
    else:
        columns_dict = ev_config['features_dict']

    # Taking local deepcopy of the ev boxes

    box_data = deepcopy(box_input)

    # Extraction energy data of boxes

    box_energy = deepcopy(box_data)

    # Taking only positive box boundaries for edge related calculations

    box_energy_idx = (box_energy > 0)
    box_energy_idx_diff = np.diff(np.r_[0, box_energy_idx.astype(int), 0])

    # Find the start and end edges of the boxes

    box_start_boolean = (box_energy_idx_diff[:-1] > 0)
    box_end_boolean = (box_energy_idx_diff[1:] < 0)

    box_start_idx = np.where(box_start_boolean)[0]
    box_end_idx = np.where(box_end_boolean)[0]

    box_start_energy = box_energy[box_start_idx]
    box_end_energy = box_energy[box_end_idx]

    box_features = np.c_[box_start_idx, box_end_idx, box_start_energy, box_end_energy]

    # Duration of boxes in hours

    box_duration = (np.diff(box_features[:, :2], axis=1) + 1) / factor

    box_energy_std = np.array([])
    box_areas = np.array([])
    box_energy_per_point = np.array([])
    box_median_energy = np.array([])
    box_minimum_energy = np.array([])
    box_maximum_energy = np.array([])

    # Looping over each identify box to extract various box level features

    for i, row in enumerate(box_features):
        start_idx, end_idx = row[:2].astype(int)

        temp_energy = box_energy[start_idx: (end_idx + 1)]

        # Calculate the absolute deviation

        temp_energy_std = np.mean(np.abs(temp_energy - np.mean(temp_energy)))

        # Total energy of the box

        temp_area = np.sum(temp_energy)

        # Energy per hour in the box

        temp_energy_per_point = temp_area / box_duration[i]
        temp_box_median_energy = np.median(temp_energy)

        temp_box_minimum_energy = np.min(temp_energy)
        temp_box_maximum_energy = np.max(temp_energy)

        box_energy_std = np.append(box_energy_std, temp_energy_std)
        box_areas = np.append(box_areas, temp_area)
        box_energy_per_point = np.append(box_energy_per_point, temp_energy_per_point)
        box_median_energy = np.append(box_median_energy, temp_box_median_energy)
        box_minimum_energy = np.append(box_minimum_energy, temp_box_minimum_energy)
        box_maximum_energy = np.append(box_maximum_energy, temp_box_maximum_energy)

    box_features = np.c_[box_features, box_duration, box_areas, box_energy_std, box_energy_per_point,
                         box_median_energy, box_minimum_energy, box_maximum_energy]

    boxes_start_idx_col = columns_dict['start_index_column']

    boxes_start_hod = hod_matrix[box_features[:, boxes_start_idx_col].astype(int)]
    boxes_start_month = np.zeros(box_features.shape[0])
    boxes_start_season = np.zeros(box_features.shape[0])
    box_start_day = np.zeros(box_features.shape[0])

    box_features = np.c_[box_features, boxes_start_hod, boxes_start_month, boxes_start_season, box_start_day]

    return box_features
