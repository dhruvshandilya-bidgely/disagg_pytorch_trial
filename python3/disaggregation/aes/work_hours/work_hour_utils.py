"""
Author: Neelabh Goyal
Date:   14 June 2023
Work hour utility functions
"""

# Import python packages

import copy
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.work_hours.get_work_hour_params import get_work_hour_params

day_number = {
    'SUNDAY': 1,
    'MONDAY': 2,
    'TUESDAY': 3,
    'WEDNESDAY': 4,
    'THURSDAY': 5,
    'FRIDAY': 6,
    'SATURDAY': 7}


def get_map_data(input_data_raw, logger_smb_month):
    """
    Function to get data maps for processing

    Parameters:
        input_data_raw      (pd.DataFrame)              : Contains consumption and time related information
        logger_smb_month    (logging object)            : Logs the code progress

    Returns:
        map_data            (tuple)                     : Contains tuple of data maps
    """

    # getting key data required to make smb estimates
    date_data = input_data_raw['date']
    time_data = input_data_raw['time']
    energy_data = input_data_raw['consumption']
    raw_ao_data = input_data_raw['raw-ao']
    dow_data = input_data_raw['day_of_week']
    diff_data_back = input_data_raw['difference_back']
    diff_data_front = input_data_raw['difference_front']
    month_data = input_data_raw['month']

    logger_smb_month.info("Generating individual data maps |")

    data_col = 2

    # getting energy map
    data = np.c_[date_data, time_data, energy_data]
    rows, row_pos = np.unique(data[:, 0], return_inverse=True)
    cols, col_pos = np.unique(data[:, 1], return_inverse=True)
    energy_map = np.zeros((len(rows), len(cols)), dtype=data.dtype)
    energy_map[row_pos, col_pos] = data[:, data_col]
    energy_map[~(energy_map >= 0)] = 0
    energy_map = np.nan_to_num(energy_map)
    energy_map = energy_map.astype(int)

    # getting raw map
    data = np.c_[date_data, time_data, raw_ao_data]
    raw_ao_map = np.zeros((len(rows), len(cols)), dtype=data.dtype)
    raw_ao_map[row_pos, col_pos] = data[:, data_col]
    raw_ao_map[~(raw_ao_map >= 0)] = 0
    raw_ao_map = np.nan_to_num(raw_ao_map)
    raw_ao_map = raw_ao_map.astype(int)

    # getting day of week map
    data = np.c_[date_data, time_data, dow_data]
    dow_map = np.empty((len(rows), len(cols)), dtype=data.dtype)
    dow_map[:] = np.nan
    dow_map[row_pos, col_pos] = data[:, data_col]
    dow_map = np.nanmean(dow_map, axis=1)

    # getting diff back map
    data = np.c_[date_data, time_data, diff_data_back]
    diff_map_back = np.zeros((len(rows), len(cols)), dtype=data.dtype)
    diff_map_back[row_pos, col_pos] = data[:, data_col]
    diff_map_back = np.nan_to_num(diff_map_back)
    diff_map_back[~((diff_map_back >= 0) | (diff_map_back < 0))] = 0
    diff_map_back = diff_map_back.astype(int)

    # getting diff front map
    data = np.c_[date_data, time_data, diff_data_front]
    diff_map_front = np.zeros((len(rows), len(cols)), dtype=data.dtype)
    diff_map_front[row_pos, col_pos] = data[:, data_col]
    diff_map_front = np.nan_to_num(diff_map_front)
    diff_map_front[~((diff_map_front >= 0) | (diff_map_front < 0))] = 0
    diff_map_front = diff_map_front.astype(int)

    # getting month data map
    data = np.c_[date_data, time_data, month_data]
    month_map = np.empty((len(rows), len(cols)), dtype=data.dtype)
    month_map[:] = np.nan
    month_map[row_pos, col_pos] = data[:, data_col]

    # getting overall map data
    map_data = (energy_map, raw_ao_map, dow_map, diff_map_back, diff_map_front, month_map)

    return map_data


def smoothen_raw_ao_consumption(input_data_raw, sampling_rate):
    """
    Smoothens raw data in case of low consumption smb, before computing work hours

    Parameters:
        input_data_raw      (pd.DataFrame)    : DataFrame containing 21 column  matrix
        sampling_rate       (int)             :

    Returns:
        input_data_raw      (np.ndarray)    : Array containing 21 column  matrix with consumption data smoothened
    """

    data = np.array(input_data_raw['raw-ao'])

    hour_factor = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    data_smooth = np.convolve(data, np.ones(hour_factor), 'same') / hour_factor

    input_data_raw['raw-ao'] = data_smooth

    return input_data_raw


def apply_continuity_filter(original_label, width=4):
    """
    Function to remove discontinuous/stray work hours from smb

    Parameters:
        original_label   (np.array)   : Array containing integer data
        width            (int)        : Carries window information for applying continuity

    Returns:
        label   (np.array)            : Array containing integer data
    """

    # appending zeros to maintain same length array
    append_zeros = np.zeros(width // 2)
    filter_ = np.ones(width)

    # At least half of valid epochs should contain non-zero values
    filter_threshold = (width // 2)
    # getting labels in required shape
    label = copy.deepcopy(original_label)
    n_label = label.shape[0]
    label = np.append(append_zeros, label)
    label = np.append(label, append_zeros)

    # getting 2d stack of labels, day-by-day
    a_hstack = np.hstack(label[i:i + width] for i in range(0, n_label))
    a_vstack = a_hstack.reshape(int(len(a_hstack) / width), width)

    # applying filter for consistency check
    filtered = a_vstack * filter_
    filtered_sum = filtered.sum(axis=1)

    # checking qualification of an epoch based on consistency score
    condition1 = filtered_sum >= filter_threshold
    # SMB v2.0 Improvement
    # condition 2 takes care of the 1st 'One (1)" in any series of 1s.
    filtered_sum = np.append(filtered_sum, 0)
    condition2 = np.logical_and(filtered_sum[1:] >= filter_threshold, filtered_sum[:-1] >= filter_threshold - 1)

    label = np.logical_and(np.logical_or(condition1, condition2), original_label)
    label = label.astype(int)

    return np.nan_to_num(label)


def sanctify_labels(in_data, kmeans_model, thresh):
    """
    Function to preserve the sanctity of labels obtained from kmeans clustering

    Parameters
    in_data         (np.array)      : Array with consumption information
    kmeans_model    (KMeans model)  : contains the kmean model
    thresh          (float64)       : threshold to check for clear separation in clusters

    returns
    labels           (np.array)     : contains labels
    """
    labels = kmeans_model.labels_

    if (not len(np.unique(labels)) == 1) and (kmeans_model.cluster_centers_[0] > kmeans_model.cluster_centers_[1]):
        labels = (~(labels.astype(bool))).astype(int)

    condition1 = (np.sum(labels == 1) > len(labels) * 0.8)
    condition2 = (0.85 <= kmeans_model.cluster_centers_[0] / kmeans_model.cluster_centers_[1] <= 1.15)[0]
    condition3 = np.nanmedian(in_data[labels == 1]) - np.nanpercentile(in_data[labels == 0], 75) < thresh
    condition4 = np.min(kmeans_model.cluster_centers_) > 1000
    condition4 = condition4 and (np.max(kmeans_model.cluster_centers_) / np.min(kmeans_model.cluster_centers_)) < 1.25

    if condition1 or condition2 or condition3 or condition4:
        labels = np.ones_like(labels)

    return labels


def get_binned_data(input_data):
    """
    Function to group data based on seasonal info
    Parameters:
        input_data  (pd.DataFrame) : Data frame with raw data

    Returns:
        binned_data (list)         : List of data bins
    """

    # Check if the season labels with 'NaN' value is less than 33% of days where cons is available
    if np.sum(np.isnan(input_data['s_label'])) < np.sum(~np.isnan(input_data['raw-ao'])) / 3:

        summer_epochs = input_data['s_label'] == 1
        winter_epochs = input_data['s_label'] == -1
        summer_transition_epochs = input_data['s_label'] == 0.5
        pure_transition_epochs = input_data['s_label'] == 0
        winter_transition_epochs = input_data['s_label'] == -0.5

    # Check if the feels_like temperature with 'NaN' value is less than 33% of days where cons is available
    elif np.sum(np.isnan(input_data['feels_like'])) < np.sum(~np.isnan(input_data['raw-ao'])) / 3:

        summer_epochs = input_data['feels_like'] > 75
        winter_epochs = input_data['feels_like'] < 55
        summer_transition_epochs = np.logical_and(input_data['feels_like'] > 68, input_data['feels_like'] <= 75)
        pure_transition_epochs = np.logical_and(input_data['feels_like'] >= 62, input_data['feels_like'] <= 68)
        winter_transition_epochs = np.logical_and(input_data['feels_like'] < 62, input_data['feels_like'] >= 55)

    # In absence of temperature or seasonal label, simply divide the epochs equally in 4 by months for clustering.
    else:

        summer_epochs = [True if x in [5, 6, 7] else False for x in input_data['month_']]
        winter_epochs = [True if x in [11, 12, 1] else False for x in input_data['month_']]
        summer_transition_epochs = [True if x in [4, 8] else False for x in input_data['month_']]
        pure_transition_epochs = [True if x in [3, 9] else False for x in input_data['month_']]
        winter_transition_epochs = [True if x in [2, 10] else False for x in input_data['month_']]

    return [summer_epochs, winter_epochs, summer_transition_epochs, winter_transition_epochs,
            pure_transition_epochs]


def get_input_data(disagg_input_object):
    """
    Function to create a dataframe from the input raw data
    Parameters:
        disagg_input_object    (dict): Contains all Input data

    Returns:
        input_data_raw (pd.DataFrame): Dataframe with extracted consumption raw data
    """
    # getting input data
    input_data_raw = copy.deepcopy(pd.DataFrame(disagg_input_object['input_data']))
    columns = Cgbdisagg.INPUT_COLUMN_NAMES
    input_data_raw.columns = columns

    # getting timestamps
    input_data_raw['timestamp'] = pd.to_datetime(input_data_raw['epoch'], unit='s')
    timezone = disagg_input_object.get('home_meta_data').get('timezone')

    # noinspection PyBroadException
    try:
        input_data_raw['timestamp'] = input_data_raw.timestamp.dt.tz_localize('UTC', ambiguous='infer').dt.tz_convert(
            timezone)
    except (IndexError, KeyError, TypeError):
        input_data_raw['timestamp'] = input_data_raw.timestamp.dt.tz_localize('UTC', ambiguous='NaT').dt.tz_convert(
            timezone)

    # getting time attributes info
    input_data_raw['date'] = input_data_raw['timestamp'].dt.date
    input_data_raw['time'] = input_data_raw['timestamp'].dt.time
    input_data_raw['year'] = input_data_raw['timestamp'].dt.year
    input_data_raw['month_'] = input_data_raw['timestamp'].dt.month

    return input_data_raw


def remove_external_lighting(input_data_raw, sampling, disagg_output_object):
    """
    Function to remove external lighting from the input raw data
    Parameters:
        input_data_raw        (pd.DataFrame): Dataframe with Raw Data
        sampling              (int)         : Integer denoting the count of samples per hour
        disagg_output_object  (dict)        : Object with pipeline level output object

    Returns:
        input_data_raw        (pd.DataFrame): Dataframe with Raw Data
        external_lighting     (np.array)    : Array with external lighting estimates
    """

    parameters = get_work_hour_params()
    ext_li_idx = disagg_output_object.get('output_write_idx_map').get('li_smb')
    external_lighting = np.nan_to_num(disagg_output_object['epoch_estimate'][:, ext_li_idx])
    input_data_raw['raw-ao'] = input_data_raw['raw-ao'] - external_lighting

    # Condition to identify epochs where hourglass is estimated
    hourglass_condition = external_lighting > 0

    # Condition to identify epochs where hourglass is estimated and the residue is less than 200Wh
    hourglass_condition = np.logical_and(hourglass_condition,
                                         input_data_raw['raw-ao'] < parameters.get('min_hourglass_residue') / sampling)

    # Suppress data points where hourglass was detected and the residue is <200Wh
    input_data_raw['raw-ao'][hourglass_condition] = 0

    return input_data_raw, external_lighting


def suppress_bad_work_hours(label_df):
    """
    Function to suppress work days where total number of work hours is statistically very low
    Parameters:
        label_df  (pd.DataFrame): DataFrame containing labels in a 2D array form

    Returns:
        frac_limit_for_day  (float)   : Minimum allowed fraction of work hours per day
        frac_of_work_in_day (np.array): Day wise fraction of work hours
    """

    frac_of_work_in_day = np.round(np.nansum(label_df, axis=1) / label_df.shape[1], 2)
    frac_limit_for_day = np.round(np.nanmedian(frac_of_work_in_day) - 3 * np.nanstd(frac_of_work_in_day), 2)
    frac_limit_for_day = np.minimum(frac_limit_for_day, np.round(np.nanmedian(frac_of_work_in_day) / 2, 2))

    # suppressing work hours in days where count of open hours is statistically very low
    if (np.nanmedian(frac_of_work_in_day) == 0) | (frac_limit_for_day < 0):

        try:
            frac_limit_for_day = np.min(frac_of_work_in_day[frac_of_work_in_day > 0])

        except (ValueError, IndexError, TypeError):
            frac_limit_for_day = 0

    return frac_of_work_in_day, frac_limit_for_day


def get_labels(cluster_centers, labels):
    """
    Function that returns processed k-means labels based on cluster count
    Parameters:
        cluster_centers (np.array) : Cluster centers as identified by the KMeans clustering
        labels          (np.array) : raw labels for each clustered data point

    Returns:
        labels (np.array): Processed labels for each data point
    """
    if len(cluster_centers) == 3:

        min_val_cluster_idx = np.argmin(cluster_centers)
        if not min_val_cluster_idx == 0:
            labels[labels == min_val_cluster_idx] = -1
            labels[labels == 0] = min_val_cluster_idx
            labels[labels == -1] = 0

        labels = labels.astype(bool) * 1

    elif len(cluster_centers) == 2 and cluster_centers[0] > cluster_centers[1]:

        labels = ~(labels.astype(bool)) * 1

    return labels


def data_clustering(clustering_data, cluster_count, clustering_threshold, daily_clustering_data=None):
    """
    Function to cluster the provided data using KMeans in the provided count of clusters
    Parameters:
        clustering_data       (np.array) : Numerical data that needs to be clustered
        cluster_count         (int)      : Integer denoting the desired number of clusters
        clustering_threshold  (float)    : Minimum consumption level for clustering
        daily_clustering_data (dict)     : Dictionary to hold clustering related information

    Returns:
        labels   (np.array) : Cluster labels for each of the data point as provided in the clustering data
    """
    labels = np.zeros_like(clustering_data)
    if len(clustering_data) > cluster_count:

        kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(clustering_data)

        if daily_clustering_data is not None:
            daily_clustering_data['kmeans_models'] = np.append(daily_clustering_data['kmeans_models'], kmeans)
            # Storing the cluster center separately to avoid a for loop in post processing.
            daily_clustering_data['kmeans_clusters'] = np.append(daily_clustering_data['kmeans_clusters'],
                                                                 kmeans.cluster_centers_[1])

        labels = get_labels(kmeans.cluster_centers_, kmeans.labels_)

        if np.max(kmeans.cluster_centers_) <= clustering_threshold:
            labels = np.zeros_like(kmeans.labels_)

    return labels


def get_contiguous_work_hours(labels_df, sampling, params):
    """
    Function to make work hour contiguous on a daily level
    Parameters:
        labels_df  (pd.DataFrame) : DataFrame with labels
        sampling   (int)          : Count of samples per hour
        params     (dict)         : Dictionary containing work hour specific constants

    Returns:
        labels_df  (pd.DataFrame) : DataFrame with labels
    """
    windows = np.arange(0, labels_df.shape[1], params.get('continuous_work_hour_window') * sampling)
    for start_window in windows:

        end_window = start_window + (2 * params.get('continuous_work_hour_window') * sampling)
        indexes_ = np.where(labels_df[:, int(start_window):int(end_window)] == 1)
        day_idx, start_hour_idx, end_hour_idx = np.unique(indexes_[0], return_index=True, return_counts=True)
        indexes = np.array(indexes_[1])
        indexes += int(start_window)
        end_hour_idx = start_hour_idx + end_hour_idx - 1

        # Make work hours contiguous
        for idx, day in enumerate(day_idx):
            labels_df[day.astype(int), indexes[start_hour_idx[idx]]:indexes[end_hour_idx[idx]]] = 1

        del (indexes_, day_idx, start_hour_idx, end_hour_idx)

    return labels_df


def check_work_hour_spread(labels, days_with_cons, hours_per_day, parameters):
    """
    Function to check if the detected work hours are spread across the year
    Parameters:
        labels           (pd.DataFrame) : Dataframe with sample level work hour labels
        days_with_cons   (np.array)     : Boolean array indicating presence of consumption for each day
        hours_per_day    (np.array)     : Integer array indicating sum of work hour samples for each day
        parameters       (dict)         : Dictionary with work hour specific parameters

    Returns:
        labels          (pd.DataFrame)  : DataFrame with processed work hour labels
    """
    # Check if the work hour is spread across the year
    count_of_months_with_work_hours = 0
    valid_iter_counter = 0
    for idx in range(0, labels.shape[0], Cgbdisagg.DAYS_IN_MONTH):
        # Only months with at least 15 non-zero days are valid for this calculation
        if np.sum(days_with_cons[idx:idx + Cgbdisagg.DAYS_IN_MONTH]) > 15:
            valid_iter_counter += 1
            if np.sum(hours_per_day[idx:idx + Cgbdisagg.DAYS_IN_MONTH]) > 1:
                count_of_months_with_work_hours += 1

    if count_of_months_with_work_hours > valid_iter_counter / 2:
        if np.sum(np.sort(labels.sum(axis=0))[-8 * parameters.get('sampling'):]) < np.sum(
                labels.sum(axis=0)) * 0.5:
            labels = np.zeros_like(labels)

    else:
        labels = np.zeros_like(labels)

    return labels


def get_alt_label(hsm_in):
    """
    Function to get the alternate label parameter from the HSM, if it exists
    Parameters:
        hsm_in     (dict) : Dictionary containing HSM object for the user

    Returns:
        alt_label  (int)  : Integer to identify if the user has run on the alternate work hour logic
    """
    try:
        alt_label = int(hsm_in.get('attributes').get('alt_label')[0])

    except(KeyError, TypeError, AttributeError):
        alt_label = 0

    return alt_label


def update_hsm(disagg_output_object, hsm_write):
    """
    Function to update the AO HSM with Work Hour HSM attributes
    Parameters:
        disagg_output_object   (dict)   : Dictionary containing pipeline level output
        hsm_write              (dict)   : Dictionary containing information to be written in HSM

    Returns:
        None
    """
    try:

        hsm_dic = disagg_output_object.get('created_hsm')
        hsm_out = hsm_dic.get('ao').get('attributes')

    except (KeyError, TypeError, AttributeError):
        # hsm for AO not created. Create new HSM dictionary to store work hour module details
        hsm_out = dict({'timestamp': hsm_write.get('last_timestamp')})
        work_hour_arr = hsm_write.get('work_arr')
        hsm_out['attributes'] = {}
        hsm_out['attributes']['user_work_hour_fraction'] = np.nansum(work_hour_arr) / len(work_hour_arr)
        hsm_out['attributes']['user_work_hour_arr'] = work_hour_arr
        hsm_out['attributes']['change_count'] = hsm_write.get('change_val')
        hsm_out['attributes']['alt_label'] = int(hsm_write.get('alt_label'))
        disagg_output_object['created_hsm']['ao'] = hsm_out

    else:
        # hsm for ao already created. Need to only add work hour attributes to the existing dictionary
        work_hour_arr = hsm_write.get('work_arr')
        hsm_out['user_work_hour_fraction'] = np.nansum(work_hour_arr) / len(work_hour_arr)
        hsm_out['user_work_hour_arr'] = work_hour_arr
        hsm_out['change_count'] = hsm_write.get('change_val')
        hsm_out['alt_label'] = int(hsm_write.get('alt_label'))
        disagg_output_object['created_hsm']['ao']['attributes'] = hsm_out


def align_work_hours_with_hsm(work_hours, parameters, all_labels, params, logger):
    """
    Function to align provided work hours with hsm
    Parameters:
        work_hours   (np.array)     : 1D Array containing epoch level work hour booleans as calculated by the algorithm
        parameters   (dict)         : Dictionary with work hour specific parameters
        all_labels   (np.array)     : 2D array with open close boolean on the representative raw data
        params       (dict)         : Dictionary with hsm specific constants
        logger       (logger object): Object to log important steps of the function

    Returns:
        work_hours    (np.array) : 1D Array containing work hour booleans after processing with HSM
        change_count  (int)      : HSM key to keep count of forced changes by HSM across runs
        hsm_used      (bool)     : Boolean flag to indicate if HSM was used to alter the work hours
    """

    hsm_used = False
    try:
        # Check if new HSM exists for work hour or not.
        change_count = parameters.get('hsm').get('change_count')[0]
        work_hour_frac = parameters.get('hsm').get('user_work_hour_fraction')[0]

    except (KeyError, TypeError, AttributeError):

        # If the new HSM with work hour specific attributes does not exist, simply exit this function
        logger.info(' Work Hour HSM not available for this run | Skipping HSM realignment')
        change_count = 0
        hsm_used = False
        return work_hours, change_count, hsm_used

    sampling = parameters.get('sampling')
    logger.info(' Valid HSM available for this run | Work Hour fraction of the user is {}'.format(work_hour_frac))

    # Condition to check if a user detected as 24x7 or 0x7 in this run is historically a defined work hour user
    if (0 < work_hour_frac < params.get('defined_work_hour_frac')) and not change_count > params.get('max_change') \
            and (np.sum(work_hours) == Cgbdisagg.HRS_IN_DAY * sampling or np.sum(work_hours) == 0):

        # Make the 24x7 user conditions harsher in case we are switching from defined work hours to 24x7
        if not (np.sum(all_labels.sum(axis=0) > params.get('max_24x7_days')) > (Cgbdisagg.HRS_IN_DAY - 1) * sampling
                or np.sum(all_labels.sum(axis=0) < params.get('max_0x7_days')) > (Cgbdisagg.HRS_IN_DAY - 2) * sampling):

            logger.info(' Replacing the detected 24x7 work hours with HSM defined work hours |')
            work_hours = np.array(parameters.get('hsm').get('user_work_hour_arr'))
            change_count += 1
            hsm_used = True

        else:

            change_count = 0

    # Condition to check if a user detected as a defined work hour user in this run is historically a 24x7 user
    elif work_hour_frac > params.get('defined_work_hour_frac') and not change_count > params.get('max_change')\
            and (0 < np.sum(work_hours) < Cgbdisagg.HRS_IN_DAY * sampling):

        # Relax the 24x7 condition
        if np.sum(all_labels.sum(axis=0) > params.get('min_24x7_days')) >= params.get('relaxed_24x7_cond') * sampling:

            logger.info(' Replacing the detected defined work hours with HSM 24x7 work hours |')
            work_hours = np.array(parameters.get('hsm').get('user_work_hour_arr'))
            change_count += 1
            hsm_used = True

        else:

            change_count = 0

    else:

        change_count = 0

    return work_hours, change_count, hsm_used


def align_epoch_work_hours_with_hsm(work_hours, parameters, all_labels, params, logger):
    """
    Function to align provided work hours with hsm
    Parameters:
        work_hours   (np.array)     : 2D Array containing epoch level work hour booleans as calculated by the algorithm
        parameters   (dict)         : Dictionary with work hour specific parameters
        all_labels   (np.array)     : 2D array with open close boolean on the representative raw data
        params       (dict)         : Dictionary with hsm specific constants
        logger       (logger object): Object to log important steps of the function

    Returns:
        work_hours   (np.array) : 2D Array containing work hour booleans after processing with HSM
        change_count (int)      : HSM key to keep count of forced changes by HSM across runs
        hsm_used     (bool)     : Boolean flag to indicate if HSM was used to alter the work hours
    """

    hsm_used = False
    try:
        # Check if new HSM exists for work hour or not.
        change_count = parameters.get('hsm').get('change_count')[0]
        work_hour_frac = parameters.get('hsm').get('user_work_hour_fraction')[0]

    except (KeyError, TypeError, AttributeError):

        # If the new HSM with work hour specific attributes does not exist, simply exit this function
        logger.info(' Work Hour HSM not available for this run | Skipping HSM realignment')
        change_count = 0
        hsm_used = False
        return work_hours, change_count, hsm_used

    # Condition to check if a user detected as 24x7 or 0x7 in this run is historically a defined work hour user
    sampling = parameters.get('sampling')
    work_hour_1d = (np.nansum(work_hours, axis=0) > params.get('min_work_hour_frac') * work_hours.shape[0]) * 1

    if (0 < work_hour_frac < params.get('defined_work_hour_frac')) and not change_count > params.get('max_change') \
            and (np.sum(work_hour_1d) > (Cgbdisagg.HRS_IN_DAY - 1) * sampling or np.sum(work_hour_1d) == 0):

        # Make the 24x7 & 0x7user conditions harsher in case we are switching from defined work hours to 24x7/0x7
        if not (np.sum(all_labels.sum(axis=0) > params.get('max_24x7_days')) > (Cgbdisagg.HRS_IN_DAY - 1) * sampling or
                np.sum(all_labels.sum(axis=0) < params.get('max_0x7_days')) > (Cgbdisagg.HRS_IN_DAY - 2) * sampling):

            work_hours = work_hours * np.array(parameters.get('hsm').get('user_work_hour_arr'))
            change_count += 1
            hsm_used = True

        else:

            change_count = 0

    # Condition to check if a user detected as a defined work hour user in this run is historically a 24x7 user
    elif (work_hour_frac > params.get('defined_work_hour_frac')) and not (change_count > params.get('max_change')) \
            and (0 < np.sum(work_hour_1d) < Cgbdisagg.HRS_IN_DAY * sampling):

        day_sum = np.nansum(work_hours, axis=1)
        constant_label_days = np.sum(np.logical_or(day_sum == 0, day_sum >= (Cgbdisagg.HRS_IN_DAY - 1) * sampling))

        # Relax the 24x7 condition
        if np.sum(all_labels.sum(axis=0) > params.get('min_24x7_days')) >= params.get('relaxed_24x7_cond') * sampling \
                or constant_label_days > (params.get('min_24x7_work_hour_frac') * work_hours.shape[0]):

            work_hours = np.tile(np.array(parameters.get('hsm').get('user_work_hour_arr')), (work_hours.shape[0], 1))
            change_count += 1
            hsm_used = True

        else:

            change_count = 0

    else:

        change_count = 0

    return work_hours, change_count, hsm_used


def get_survey_work_hours(survey_work_hour_dict):
    """
    Function to identify the most common work hours for the user as per the survey
    Parameters:
        survey_work_hour_dict  (dict) : Dictionary object containing the survey input w.r.t work hours

    Returns:
       survey_work_hours (dict): Dictionary containing work hours representative for the complete data of the user
    """

    start_min_arr = []
    end_min_arr = []
    off_days = []

    for day in survey_work_hour_dict.items():
        if day[1].get('startHour') is not None:
            start_min = day[1].get('startHour') * Cgbdisagg.SEC_IN_1_MIN + day[1].get('startMinute')
            end_min = day[1].get('endHour') * Cgbdisagg.SEC_IN_1_MIN + day[1].get('endMinute')

            # Check the validity of the i/p. start and end should be at least 60 minutes apart
            if end_min - start_min > Cgbdisagg.SEC_IN_1_MIN:
                start_min_arr.append(start_min)
                end_min_arr.append(end_min)

            else:
                off_days.append(day_number.get(str(day[0]).upper()))

        else:
            off_days.append(day_number.get(str(day[0]).upper()))

    if len(start_min_arr) == 0:
        survey_work_hours = {
            'median_start_min': 0,
            'median_end_min': 0,
            'off_days': off_days
        }

    else:
        survey_work_hours = {
            'median_start_min': np.nanmedian(start_min_arr),
            'median_end_min': np.nanmedian(end_min_arr),
            'off_days': off_days
        }

    return survey_work_hours


def compare_algo_and_survey_work_hours(survey_work_hour_dict, algo_work_hours, logger_pass):
    """
    Function compare work hours as calculated by algo against the survey input work hours
    Parameters:
        survey_work_hour_dict (dict)          : Dictionary object containing the survey input w.r.t work hours
        algo_work_hours       (np.array)      : 1D array containing epoch level work hour boolean
        logger_pass           (logging object): Logger for the function

    Returns:
        work_hours (np.array): 1D array containing epoch level work hour boolean
    """

    logger_base = logger_pass.get('logger').getChild('compare_algo_and_survey_work_hours')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    logger.info(' Comparing the algo work hours with the Survey input for work hours | ')
    survey_work_hours = np.zeros_like(algo_work_hours)
    samples_per_hour = len(algo_work_hours) / Cgbdisagg.HRS_IN_DAY
    survey_hours = get_survey_work_hours(survey_work_hour_dict)
    considering_survey = False

    if len(survey_hours.get('off_days')) >= 5:
        logger.info(' Survey likely not filled. | Skipping the comparison')
        return considering_survey

    start_idx = int(survey_hours.get('median_start_min')/Cgbdisagg.SEC_IN_1_MIN * samples_per_hour)
    end_idx = int(survey_hours.get('median_end_min')/Cgbdisagg.SEC_IN_1_MIN * samples_per_hour)
    survey_work_hours[start_idx: end_idx] = 1
    logger.info(' Survey work hour median array is calculated to be | {}'.format(survey_work_hours))

    algo_work_epochs = np.argwhere(algo_work_hours > 0)
    survey_work_epochs = np.argwhere(survey_work_hours > 0)
    if calculate_jaccard_similarity(algo_work_epochs, survey_work_epochs, threshold=0.75):
        considering_survey = True
        logger.info(' Survey and algo work hours are found to be similar | Considering the survey')

    else:
        considering_survey = False
        logger.info(' Survey work hours and algo work hours are divergent | Ignoring the survey')

    return considering_survey


def calculate_jaccard_similarity(np_arr1, np_arr2, threshold=0.5):
    """
    Function to calculate the Jaccard similarity of two numpy arrays
    Parameters:
        np_arr1     (np.array) : First numpy array
        np_arr2     (np.array) : Second numpy array
        threshold   (float)    : Minimum similarity score to call the two arrays similar

    Returns:
        is_similar  (bool)     : True/False signifying similarity between the two work hour arrays
    """
    intersection = np.intersect1d(np_arr1, np_arr2)
    union = np.union1d(np_arr1, np_arr2)
    similarity = len(intersection) / len(union)

    is_similar = similarity >= threshold

    return is_similar


def sanitize_epoch_with_survey(input_df, labels_epoch, survey_work_hour_dict, samples_per_hour):
    """
    Function to sanitize the epoch work hours based on survey work hour input
    Parameters:
        input_df              (pd.DataFrame): Contains 27 column raw data matrix
        labels_epoch          (pd.DataFrame): DataFrame with epoch level work hours
        survey_work_hour_dict (dict)        : Dictionary object containing the survey input w.r.t work hours
        samples_per_hour      (int)         : Integer to indicate count of samples per hour

    Returns:
        labels_epoch  (pd.DataFrame): DataFrame with epoch level work hours after processing
    """
    day_of_week_df = input_df.pivot_table(index='date', columns='time', values='day_of_week', aggfunc=np.min).values
    day_of_week = day_of_week_df[:, 0]

    for day in survey_work_hour_dict.items():
        survey_work_hour_arr = np.zeros_like(day_of_week_df[0, :])
        day_num = day_number.get(day[0])
        labels_slice = labels_epoch[day_of_week == day_num]

        if day[1].get('startHour') is not None:
            day_start_idx = int((day[1].get('startHour') + (day[1].get('startMinute') / Cgbdisagg.SEC_IN_1_MIN)) *
                                samples_per_hour)
            day_end_idx = int((day[1].get('endHour') + (day[1].get('endMinute') / Cgbdisagg.SEC_IN_1_MIN)) *
                              samples_per_hour)

            # Only days with more than 1 hour of work hour are valid days. Other days are treated as off days.
            if day_end_idx - day_start_idx > samples_per_hour:
                survey_work_hour_arr[day_start_idx: day_end_idx] = 1

        labels_epoch[day_of_week == day_num] = np.nan_to_num(labels_slice) * survey_work_hour_arr

    return labels_epoch
