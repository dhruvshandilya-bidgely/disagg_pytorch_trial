"""
Author: Neelabh Goyal
Date:   14-June-2023
Called to detect epoch level work hour
"""

# Import python packages
import copy
import timeit
import logging
import datetime
import numpy as np
import pandas as pd

# Import from within the project

from python3.disaggregation.aes.work_hours.work_hour_utils import sanitize_epoch_with_survey
from python3.disaggregation.aes.work_hours.work_hour_utils import calculate_jaccard_similarity
from python3.disaggregation.aes.work_hours.work_hour_utils import compare_algo_and_survey_work_hours
from python3.disaggregation.aes.work_hours.work_hour_utils import apply_continuity_filter, get_contiguous_work_hours
from python3.disaggregation.aes.work_hours.work_hour_utils import suppress_bad_work_hours, data_clustering, get_labels


def epoch_work_hour_preprocess(user_work_hours, static_params, sampling):
    """
    Function to pre process data before identifying epoch level open/close boolean
    Parameters:
        user_work_hours (np.array)  : Numpy array with user-level Boolean for open/close
        static_params   (dict)      : Dictionary containing work hour specific constants
        sampling        (int)       : Float indicating count of samples per hour

    Returns:
        yearly_work_hours (np.array): Processed numpy array with user-level Boolean for open/close
    """
    yearly_work_hours = copy.deepcopy(user_work_hours)

    yearly_work_hours_shifted = np.append(0, yearly_work_hours)
    diffs = yearly_work_hours_shifted[1:] - yearly_work_hours_shifted[:-1]

    # If theres are more than 2 bands, mark it as a 24x7 SMB and let the alternate work hours module take care
    if np.sum(diffs < 0) > static_params.get('max_work_bands'):
        yearly_work_hours = np.ones_like(yearly_work_hours)
        return yearly_work_hours

    # If 2 bands are detected, check if they are close, if yes, then merge them into 1 to form a single contiguous band.
    elif np.sum(diffs < 0) >= static_params.get('max_work_bands'):
        band_start_idx = np.argwhere(diffs > 0)
        bands_end_idx = np.argwhere(diffs < 0)
        gap_in_bands = band_start_idx[1][0] - bands_end_idx[0][0]
        if gap_in_bands <= static_params.get('max_allowed_gap_hours') * sampling:
            yearly_work_hours[bands_end_idx[0][0] + 1:band_start_idx[1][0] + 1] = 1

        # If one of the bands is overnight band starting before 10 pm and ending post 3 am, kill it
        elif bands_end_idx[-1][0] < band_start_idx[-1][0] < static_params.get('overnight_band_night_hour') * sampling \
                and bands_end_idx[0][0] > static_params.get('overnight_band_morning_hour') * sampling:
            # Everything post the last start index to be made zero
            yearly_work_hours[band_start_idx[-1][0]:] = 0
            # Everything before the first end index to be made zero
            yearly_work_hours[:bands_end_idx[0][0]] = 0

    if not yearly_work_hours.sum() >= static_params.get('max_work_hours') * sampling:
        # Identify the start and end of work hours and add 2 hours of buffer on either side to capture daily variations.
        start_index = int(np.where(yearly_work_hours == 1)[0][0])
        if start_index == 0:
            start_index = int(np.where(yearly_work_hours == 0)[0][-1]) + 1
            final_index = int(np.where(yearly_work_hours == 0)[0][0])
        else:
            final_index = int(np.where(yearly_work_hours == 1)[0][-1]) + 1

        ones_2_add = np.ones(2 * sampling)
        np.put(yearly_work_hours, np.arange(start_index - len(ones_2_add), start_index), ones_2_add, mode='clip')
        np.put(yearly_work_hours, np.arange(final_index, final_index + len(ones_2_add)), ones_2_add, mode='clip')

        # If the last sample is deep into the night, add 1.5 hours at the early morning stage
        if final_index >= len(yearly_work_hours):
            ones_to_add = np.ones(int(static_params.get('overnight_work_hour') * sampling))
            np.put(yearly_work_hours, np.arange(0, len(ones_to_add)), ones_to_add, mode='clip')

    return yearly_work_hours


def get_epoch_work_hours(parameters, in_data, disagg_output_object, static_params, logger_pass):
    """
    Function to identify work hour for each day of the user
    Parameters:
        parameters           (dict)        : Dictionary with necessary pre-calculated variables
        in_data              (pd.DataFrame): Dataframe with raw data
        disagg_output_object (dict)        : All pipeline level output data of the user.
        static_params        (dict)        : Dictionary containing work hour specific constants
        logger_pass          (logger)      : Logger to be used here

    Returns:
        labels_final_df       (np.array)  :  2D array contianing epoch level open / close boolean for the user
        daily_clustering_data (dict)      : Dictionary with clustering specific data
    """
    logger_base = logger_pass.get('logger').getChild('get_epoch_work_hours')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    daily_clustering_data = static_params.get('daily_clustering_dict')

    input_df = copy.deepcopy(in_data[['date', 'time', 'raw-ao', 'epoch']])
    epoch_df = input_df.pivot_table(index='date', columns='time', values='epoch', aggfunc=np.min).values
    epochs = epoch_df.flatten()

    input_df['labels'] = 0
    kmeans_data = copy.deepcopy(np.array(input_df['raw-ao']).reshape(-1, 1))
    total_days = np.unique(input_df['date']).shape[0]

    # Identify epochs where hourglass was estimated
    lighting_idx = disagg_output_object.get('output_write_idx_map').get('li_smb')
    hourglass_condition = (disagg_output_object['epoch_estimate'][:, lighting_idx] > 0).reshape(-1, 1)

    # Identify epochs where hourglass was estimated and the residue is below 200 Wh
    condition = np.logical_and(hourglass_condition,
                               kmeans_data < static_params.get('min_hourglass_residue') / parameters.get('sampling'))

    # Remove any such data point from our calculations
    kmeans_data[condition] = 0
    logger.info('Removed data points overlapping with external lighting and low residue consumption |')
    daily_clustering_data['kmeans_data'] = kmeans_data
    threshold_filter_min = np.minimum(
        np.nanpercentile(kmeans_data[kmeans_data > static_params.get('min_epoch_cons')], 15),
        static_params.get('min_hourglass_residue') / parameters.get('sampling'))

    labels_final = np.zeros_like(epoch_df)
    cons_level = parameters.get('cons_level')
    cluster_count = static_params.get('cluster_count').get(cons_level)

    if cons_level in ['low', 'v_low']:
        threshold_filter_max = np.nanmedian(kmeans_data) + 3 * np.nanstd(kmeans_data)
        days_to_add = np.minimum(total_days, static_params.get('max_days_for_non_high_cons_kmeans'))

    elif cons_level == 'high':
        threshold_filter_max = np.nanmedian(kmeans_data) + 2 * np.nanstd(kmeans_data)
        days_to_add = np.minimum(total_days, static_params.get('max_days_for_high_cons_kmeans'))

    else:
        threshold_filter_max = np.nanmedian(kmeans_data) + 2 * np.nanstd(kmeans_data)
        days_to_add = np.minimum(total_days, static_params.get('max_days_for_non_high_cons_kmeans'))

    daily_clustering_data['cluster_count'] = cluster_count
    start = timeit.default_timer()

    kmeans_data[kmeans_data > threshold_filter_max] = threshold_filter_max
    start_day = np.nanmin(input_df['date'])
    last_day = np.nanmax(input_df['date'])
    start_idx = 0

    while start_day <= last_day:

        daily_clustering_data['label_idx_start'] = np.append(daily_clustering_data['label_idx_start'], start_idx)

        if (start_day + datetime.timedelta(days=float(days_to_add))) >= (last_day - datetime.timedelta(days=5)):
            end_idx = len(input_df)
        else:
            end_idx = np.nanargmin(input_df['date'] <= start_day + datetime.timedelta(days=float(days_to_add)))

        daily_clustering_data['label_idx_end'] = np.append(daily_clustering_data['label_idx_end'], end_idx)

        clustering_data = np.nan_to_num(np.array(kmeans_data[start_idx:end_idx]))

        threshold_cluster = np.nanmedian(clustering_data) + (cluster_count * np.nanstd(clustering_data))

        clustering_data[clustering_data > threshold_cluster] = threshold_cluster

        labels = data_clustering(clustering_data, cluster_count, threshold_filter_min, daily_clustering_data)

        input_df['labels'].iloc[start_idx: end_idx] = apply_continuity_filter(np.array(labels),
                                                                              width=3 * parameters.get('sampling'))

        label_2d = input_df.iloc[start_idx: end_idx].pivot_table(index='date', columns='time', values='labels').values

        frac_of_work_in_day, frac_limit_for_day = suppress_bad_work_hours(label_2d)

        # updating valid days
        valid_days = np.array(frac_of_work_in_day >= frac_limit_for_day)
        label_2d[~valid_days, :] = 0

        work_hour_perc_cluster = np.nansum(input_df['labels'].iloc[start_idx: end_idx]) / (end_idx - start_idx)
        daily_clustering_data['work_hour_perc_cluster'] = np.append(daily_clustering_data['work_hour_perc_cluster'],
                                                                    work_hour_perc_cluster)

        if start_idx == 0:
            labels_final = label_2d
        else:
            labels_final = np.concatenate((labels_final, label_2d), axis=0)

        start_idx = end_idx

        if end_idx == len(input_df):
            break
        else:
            start_day = input_df['date'].iloc[end_idx]

    end = timeit.default_timer()
    logger.info(' Daily work hour loop took | {} seconds'.format(round(end - start, 3)))

    labels = labels_final.flatten()
    _, idx_mem_1, idx_mem_2 = np.intersect1d(epochs, input_df['epoch'], return_indices=True)
    input_df['labels'][idx_mem_2] = labels[idx_mem_1]
    daily_clustering_data['final_label_arr'] = input_df['labels']

    labels_final_df = input_df.pivot_table(index='date', columns='time', values='labels', aggfunc=np.min).values

    return np.nan_to_num(labels_final_df), daily_clustering_data


def check_work_hour_similarity(epoch_labels, yearly_label_1d, survey_work_hours, logger_root):
    """
    Function to calculate similarity between the epoch level and yearly work hours
    Args:
        epoch_labels        (np.array)      : 2D boolean array with epoch level work hour information
        yearly_label_1d     (np.array)      : Boolean array with user level work hour information
        survey_work_hours   (dict)          : Contains survey input w.r.t work hours
        logger_root         (logging object): Logger for the function

    Returns:
        return_val  (int): True/False signifying similarity between the two work hour arrays
    """
    logger_base = logger_root.get('logger').getChild('post_process_epoch_work_hours')
    logger_pass = {"logger": logger_base, "logging_dict": logger_root.get("logging_dict")}

    return_code = {
        'epoch_similar_to_survey': 1,
        'yearly_similar_to_epoch': 2,
        'no_similarity_found': 3
    }

    # Identify the common work hours detected at the user level
    yearly_label_hours = np.argwhere(yearly_label_1d > 0)

    # Get column wise sum of epoch level labels, this will help identify the most common work hours
    epoch_labels_sum = np.nansum(epoch_labels, axis=0)
    epoch_work_hour_arr_1d = data_clustering(epoch_labels_sum.reshape(-1, 1), 2, 0.1 * epoch_labels.shape[0])

    prominent_hours = np.argwhere(epoch_work_hour_arr_1d > 0)
    if compare_algo_and_survey_work_hours(survey_work_hours, epoch_work_hour_arr_1d, logger_pass):
        return_val = return_code.get('epoch_similar_to_survey')

    elif calculate_jaccard_similarity(prominent_hours, yearly_label_hours):
        return_val = return_code.get('yearly_similar_to_epoch')

    else:
        return_val = return_code.get('no_similarity_found')

    return return_val


def handle_long_hours(labels_final_df, yearly_label, ao_hvac_days, clustering_info, sampling, params):
    """
    Function to handle unusually long work hour days
    Parameters:
        labels_final_df (pd.DataFrame): 2D Matrix with epoch level work hour boolean
        yearly_label    (np.array)    : user level yearly label array
        ao_hvac_days    (np.array)    : Boolean array identifying days with AO HVAC
        clustering_info (dict)        : Dictionary with clustering information
        sampling        (int)         : Integer to identify the count of samples per hour
        params          (dict)        : Dictionary containing work hour specific constants

    Returns:
        labels_final_df (pd.DataFrame): 2D Matrix with epoch level work hour boolean
    """

    func_params = params.get('long_work_hours')
    hours_per_day = np.nansum(labels_final_df, axis=1)
    yearly_wh_sum = np.sum(yearly_label[0, :])

    if yearly_wh_sum < func_params.get('yearly_work_hour_thresh') * sampling:
        days_to_change_1 = np.logical_or(hours_per_day > func_params.get('max_work_hour_lim') * sampling,
                                         hours_per_day > np.maximum(func_params.get('min_long_work_hours') * sampling,
                                                                    func_params.get('max_variation') * yearly_wh_sum))
    else:
        days_to_change_1 = np.zeros_like(hours_per_day).astype(bool)

    hours_per_day = labels_final_df.sum(axis=1)
    hours_per_ao_hvac_day = hours_per_day[ao_hvac_days]
    median_hours_per_ao_hvac_day = np.nanmedian(hours_per_ao_hvac_day[hours_per_ao_hvac_day > 0])

    hours_per_non_ao_hvac_day = hours_per_day[~ao_hvac_days]
    median_hours_per_non_ao_hvac_day = np.nan_to_num(
        np.nanmedian(hours_per_non_ao_hvac_day[hours_per_non_ao_hvac_day > 0]))

    median_work_hours = labels_final_df[~ao_hvac_days][np.nanargmin(abs(hours_per_non_ao_hvac_day -
                                                                        median_hours_per_non_ao_hvac_day))]

    days_to_change = (ao_hvac_days & (hours_per_day > median_hours_per_non_ao_hvac_day)) | days_to_change_1

    if (median_hours_per_ao_hvac_day - np.nansum(clustering_info['yearly_work_hour']) > 2 * sampling) and \
            (median_hours_per_ao_hvac_day - median_hours_per_non_ao_hvac_day > 2 * sampling):
        labels_final_df[days_to_change, :] = median_work_hours

    return labels_final_df


def post_process_epoch_work_hours(labels_final_df, sampling, yearly_label, clustering_info, disagg_output_object,
                                  input_df, params, survey_work_hour_dict, logger_root):
    """
    Function to post process epoch level work hours. Handles FNs due to HVAC and makes work hour contiguous
    Parameters:
        labels_final_df       (pd.DataFrame) : Contains epoch level boolean for work hours
        sampling              (integer)      : Indicates the count of samples per hour
        yearly_label          (np.array)     : Array of 24 x sampling length indicating user level work hour boolean
        clustering_info       (dict)         : Dictionary with clustering information
        disagg_output_object  (dict)         : Dictionary with Pipeline level output
        input_df              (pd.DataFrame) : Dataframe with all input raw data
        params                (dict)         : Dictionary containing work hour specific constants
        survey_work_hour_dict (dict)         : Dictionary object containing the survey input w.r.t work hours
        logger_root           (logger)       : Logger to be used here

    Returns:
        labels_final_df    (np.array): Array with epoch level boolean for open/close status after postprocessing
    """

    start = timeit.default_timer()

    logger_base = logger_root.get('logger').getChild('post_process_epoch_work_hours')
    logger_pass = {"logger": logger_base, "logging_dict": logger_root.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_base, logger_root.get('logging_dict'))

    func_params = params.get('post_process_epoch_work_hours')
    # Identify sections of data with unusually high cluster centers
    cluster_center_limit = np.nanmedian(clustering_info['kmeans_clusters']) + 2 * np.std(
        clustering_info['kmeans_clusters'])

    cluster_center_limit = np.minimum(cluster_center_limit, np.nanmedian(clustering_info['kmeans_clusters']) * 2)
    logger.info(' Cluster center limit identified as | {}'.format(cluster_center_limit))

    cluster_center_idx = np.argmin(
        clustering_info['kmeans_clusters'] - np.nanmedian(clustering_info['kmeans_clusters']))
    valid_clusters = np.array(clustering_info['kmeans_clusters'] <= cluster_center_limit)

    work_hour_frac_75perc = np.nanpercentile(clustering_info['work_hour_perc_cluster'][valid_clusters], 75)

    logger.info(' The 75th percentile of work hour fraction for the user is | {}'.format(work_hour_frac_75perc))

    valid_clusters = np.logical_or(valid_clusters, clustering_info['work_hour_perc_cluster'] > work_hour_frac_75perc)
    input_df['ao_hvac'] = np.nan_to_num(disagg_output_object['ao_seasonality']['epoch_cooling']) + \
                          np.nan_to_num(disagg_output_object['ao_seasonality']['epoch_heating'])

    ao_hvac_df = input_df.pivot_table(index='date', columns='time', values='ao_hvac', aggfunc=np.sum)
    ao_hvac_days = np.nansum(ao_hvac_df, axis=1) > 0
    logger.info(' Count of days with AO HVAC estimation | {}'.format(np.nansum(ao_hvac_days)))

    day_sum = labels_final_df.sum(axis=1)
    # Identify hours of the day that are marked as operating hours for more than 75% of all days
    condition_1 = np.sum(labels_final_df.sum(axis=0) > (labels_final_df[day_sum > 0].shape[0] *
                                                        func_params.get('overall_day_thresh')))

    # Identify hours of the day that are marked as operating hours for more than 65% of all non-ao hvac days
    condition_2 = np.sum(labels_final_df[~ao_hvac_days].sum(axis=0) > (np.nansum(~ao_hvac_days & (day_sum > 0)) *
                                                                       func_params.get('non_hvac_day_thresh')))

    # Identify hours of the day that are marked as operating hours for more than 50% of all ao hvac days
    condition_3 = np.sum(labels_final_df[ao_hvac_days].sum(axis=0) > (np.nansum(ao_hvac_days & (day_sum > 0)) *
                                                                      func_params.get('hvac_day_thresh')))

    # Check if count of ao_hvac_days exceeds that of non-ao_hvac_days by a factor of 2.5
    condition_4 = np.nansum(ao_hvac_days & (day_sum > 0)) > (np.nansum(~ao_hvac_days & (day_sum > 0)) *
                                                             func_params.get('hvac_days_factor'))

    # Identify days where the sum of daily work hours is more than the user level identified work hours
    lwh_params = func_params.get('long_work_hours')
    condition_5 = np.sum(labels_final_df.sum(axis=1) > np.maximum(lwh_params.get('yearly_work_hour_thresh') * sampling,
                                                                  lwh_params.get('max_variation') * np.sum(
                                                                      yearly_label[0, :])))

    if (condition_1 > func_params.get('max_hrs_with_75perc_work_hours') * sampling) | \
            (condition_2 > func_params.get('max_non_hvac_hrs_with_65perc_work_hours') * sampling) | \
            ((condition_3 > func_params.get('max_hvac_hrs_with_50perc_work_hours') * sampling) & condition_4) | \
            condition_5 > np.count_nonzero(day_sum) / 2:

        logger.info(' Daily clusters identified resemble 24x7 behavior, marking the user 24x7 |')
        labels_final_df = np.ones_like(labels_final_df)
        cons = np.nansum(input_df.pivot_table(index='date', columns='time', values='consumption', aggfunc=np.sum).values, axis=1)
        labels_final_df[cons == 0] = 0
        return np.nan_to_num(labels_final_df)

    # Identify days that needs to be re clustered because of bad clustering earlier
    elif not np.sum(~valid_clusters) == 0:
        logger.info(' Identifying sections of data that might need re-clustering |')
        re_clustering_start_epochs = clustering_info['label_idx_start'][~valid_clusters]
        re_clustering_end_epochs = clustering_info['label_idx_end'][~valid_clusters]

        # Correct this for users with overnight data.
        re_clustering_epochs = np.hstack(
            np.arange(i, j) for i, j in zip(re_clustering_start_epochs, re_clustering_end_epochs))
        re_clustering_epochs = np.unique(re_clustering_epochs.astype(int))

        # Re-cluster affected data points based on the median cluster
        re_clustering_data = np.nan_to_num(np.take(clustering_info['kmeans_data'], re_clustering_epochs))
        kmeans_model = copy.deepcopy(clustering_info['kmeans_models'][cluster_center_idx])
        new_labels = kmeans_model.predict(re_clustering_data.reshape(-1, 1))

        new_labels = get_labels(kmeans_model.cluster_centers_, new_labels)

        re_clustered_labels = apply_continuity_filter(new_labels, width=3 * sampling)
        logger.info(' Re-clustered the identified data. Merging the new labels with existing label array |')

        np.put(np.asarray(clustering_info['final_label_arr']), re_clustering_epochs, re_clustered_labels)
        # The index should match since we do the matching in the previous function call.
        input_df['labels'] = clustering_info['final_label_arr']
        labels_final_df = input_df.pivot_table(index='date', columns='time', values='labels').values
        logger.info(' Successfully assimilated the new labels with existing labels for daily clusters |')

    pd.DataFrame(labels_final_df).replace(0, np.nan, inplace=True)

    logger.info(' Making the work hour labels contiguous |')

    labels_final_df = get_contiguous_work_hours(labels_final_df, sampling, func_params)

    similarity_code = check_work_hour_similarity(labels_final_df, yearly_label[0, :], survey_work_hour_dict, logger_pass)

    if similarity_code == 1:
        logger.info(' Jaccard similarity b/w the survey label and daily label is good | Sanitizing using survey labels')
        labels_final_df = sanitize_epoch_with_survey(input_df, labels_final_df, survey_work_hour_dict, sampling)

    elif similarity_code == 2:
        labels_final_df = np.nan_to_num(labels_final_df) * yearly_label
        logger.info(' Jaccard similarity b/w the yearly label and daily label is good | Sanitizing using yearly labels')

    else:
        labels_final_df = np.ones_like(labels_final_df)
        logger.info(' Jaccard similarity b/w the yearly & daily labels as well as Survey & daily labels is low | '
                    'Marking the user as 24x7')

    # Handle long work hours only if AO HVAC days are present
    if (np.nansum(ao_hvac_days) < ao_hvac_df.shape[0] * 0.8) and np.nansum(ao_hvac_days) > 1:

        logger.info(' Identifying days with unusually long work hours that needs to be handled |')
        labels_final_df = handle_long_hours(labels_final_df, yearly_label, ao_hvac_days, clustering_info, sampling,
                                            func_params)
        logger.info(' Successfully decreased work hours for days with unusually long work hours | ')

    hours_per_day = labels_final_df.sum(axis=1)
    labels_final_df[hours_per_day < 1 * sampling, :] = 0
    logger.info(' Removed stray work hours |')

    end = timeit.default_timer()
    logger.info('Daily work hour post processing took | {} s.'.format(round(end - start, 3)))

    return np.nan_to_num(labels_final_df)
