"""
Author: Neelabh Goyal
Date:   14-June-2023
Called to calculate overall user level work hours for the user
"""

# Import python packages
import timeit
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.work_hours.work_hour_utils import sanctify_labels, apply_continuity_filter


def user_work_hour_pre_process(input_df, cons_level, static_params, logger_pass):
    """
    Function to read and preprocess data for calculating user level work hours
    Parameters:
        input_df    (pd.DataFrame)  : Dictionary with input raw data
        cons_level       (string)   : Signifies the consumption level of the user
        static_params      (dict)   : Dictionary containing work hour specific constants
        logger_pass      (Logger)   : Logger object to be used in the function

    Returns:
        data_parameters          (dict) : Dictionary with all essential calculated parameters for clustering
        input_pivot_data (pd.DataFrame) : Raw data pivoted for easier processing
    """

    logger_base = logger_pass.get('logger').getChild('user_work_hour_pre_process')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    input_df['raw-ao'][input_df['raw-ao'] < 0] = 0
    input_df['raw-ao'] = np.nan_to_num(input_df['raw-ao'])
    input_df['s_label'] = np.nan_to_num(input_df['s_label'].ffill(limit=8))

    data_parameters = dict()
    data_parameters['daily_s_label'] = input_df.pivot_table(index='date', columns='time',
                                                            values='s_label').median(axis=1).values
    input_pivot_data = input_df.pivot_table(index='date', columns='time', values='raw-ao').values

    data_parameters['valid_days'] = np.sum(np.nan_to_num(input_pivot_data), axis=1) > 0
    sampling = int(input_pivot_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    if not cons_level == 'mid':
        cluster_diff = static_params.get('cluster_thresh').get(cons_level) / sampling
    else:
        cluster_diff = np.maximum(np.nanpercentile(input_pivot_data, 5), static_params.get('cluster_thresh').get('low') / sampling)
        cluster_diff = np.minimum(cluster_diff, static_params.get('cluster_thresh').get('high') / sampling)

    data_parameters['sampling'] = sampling
    data_parameters['cluster_diff'] = cluster_diff

    input_pivot_data_ao_day = np.nanmin(input_pivot_data, axis=1)

    input_pivot_data = input_pivot_data - np.tile(input_pivot_data_ao_day.reshape(-1, 1),
                                                  (1, input_pivot_data.shape[1]))
    logger.info(' Removed day level AO from the raw data |')

    input_pivot_data[input_pivot_data > np.nanpercentile(input_pivot_data, 99)] = np.nanpercentile(input_pivot_data, 99)
    logger.info(' Capped the raw data to 99th percentile value |')

    return data_parameters, input_pivot_data


def get_common_work_hours(input_pivot_data, parameters, static_params, logger_pass):
    """
    Function to identify user level work hours. These work hours act as a guidance when estimating epoch level work hour
    Parameters:
        input_pivot_data  (pd.DataFrame): DataFrame with pivoted raw consumption data
        parameters                (dict): Dictionary with essential pre-calculated variables
        static_params             (dict): Dictionary containing work hour specific constants
        logger_pass     (logging object): Logger to use here

    Returns:
        common_work_hours (np.array) : 1D Array (24 x sampling)with boolean values signifying user level open-close
        all_labels        (np.array) : 2D Array with work-hour boolean for representative raw-data of the user.
    """

    start = timeit.default_timer()

    logger_base = logger_pass.get('logger').getChild('get_common_work_hours')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    all_labels = np.zeros((8, input_pivot_data.shape[1]))

    valid_days = np.bitwise_and(np.nansum(input_pivot_data, axis=1) > 0,
                                np.logical_and(-0.5 <= parameters.get('daily_s_label'),
                                               parameters.get('daily_s_label') <= 0.5))

    if np.nansum(valid_days) < static_params.get('min_days_thresh'):
        logger.info(' Count of valid days after removing peak season days is less than 45 | Relaxing filter.')
        valid_days = np.nansum(input_pivot_data, axis=1) > 0

    if np.nansum(valid_days) < static_params.get('min_days_thresh'):
        logger.info(' Valid days count even after relaxing filter is less than {} | Not calculating work hours'.format(static_params.get("min_days_thresh")))
        common_work_hours = np.zeros(input_pivot_data.shape[1])
        return common_work_hours, all_labels

    data_range = static_params.get('data_range').get(parameters.get('cons_level'))

    logger.info(' Consumption percentile values to represent the data for the user is | {} '.format(data_range))

    for idx, perc in enumerate(data_range):

        in_data = np.array(np.nanpercentile(input_pivot_data[valid_days], perc, axis=0))
        in_data = np.nan_to_num(in_data)
        cluster_thresh = np.maximum(parameters.get('cluster_diff'), np.nanpercentile(in_data, 20))
        if parameters.get('cons_level') == 'high':
            cluster_thresh = np.minimum(cluster_thresh,
                                        parameters.get('cluster_diff') * static_params.get('high_cons_thresh_multiplier'))

        if in_data.sum() < static_params.get('min_daily_cons'):
            labels = np.zeros_like(in_data)
            logger.debug(' Total consumption for {} percentile value is less than {} Watts. Marking labels as 0 |'.format(perc, static_params.get("min_daily_cons")))

        else:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(in_data.reshape(-1, 1))
            labels = sanctify_labels(in_data, kmeans, cluster_thresh)
            logger.debug(' Identified labels for data {} percentile data is | {}'.format(perc, labels))

        all_labels[int(idx), :] = np.array(labels)

    common_work_hours = all_labels.sum(axis=0)

    end = timeit.default_timer()
    logger.info('Yearly work hour detection took | {} s '.format(round(end - start, 3)))

    return common_work_hours, all_labels


def user_work_hour_post_process(yearly_labels, all_labels, parameters, static_params, logger_pass):
    """
    Function to post process user level open-close boolean array
    Parameters:
        yearly_labels      (np.array): Array with user level open-close
        all_labels         (np.array): 2D array with open close boolean on the representative raw data
        parameters             (dict): Dictionary with essential pre-calculated variables
        static_params          (dict): Dictionary containing work hour specific constants
        logger_pass  (logging object): Logger to use here

    Returns:
        labels_2_year (np.array): Post processed user level open-close boolean array
    """

    logger_base = logger_pass.get('logger').getChild('user_work_hour_post_process')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    thresh = static_params.get('label_thresh').get(parameters.get('cons_level'))
    func_params = static_params.get('post_process_user_work_hours')

    if np.max(yearly_labels) <= func_params.get('high_percentile_clusters') and \
            np.sum(all_labels[-2, :]) >= func_params.get('high_percentile_clusters') * parameters.get('sampling'):
        logger.info(' Work hours identified for the highest consumption percentile values only. |')
        labels_2_year = np.where(np.sum(all_labels[-2:, :], axis=0) >= 1, 1, 0)
        labels_2_year = apply_continuity_filter(labels_2_year, width=func_params.get('high_percentile_clusters') * parameters.get('sampling'))
        logger.info(' Identified final work hours based on the higher percentile values only |')

    elif np.sum(yearly_labels > func_params.get('max_days_thresh')) > func_params.get('max_work_hour') * parameters.get('sampling'):
        labels_2_year = np.ones_like(yearly_labels)
        logger.info(' More than 60% of days have >20 hours marked as work hours. User identified as a 24x7 user |')

    # Check if more than 20 hours has less than 2 days (out of 8) marked as having work hour
    elif np.sum(yearly_labels <= func_params.get('min_days_thresh')) > func_params.get('max_work_hour') * parameters.get('sampling'):
        labels_2_year = np.ones_like(yearly_labels)
        logger.info(' More than 75% of days have <4 hours marked as work hours. User identified as a 24x7 user |')

    elif np.nanpercentile(yearly_labels, 95) - np.nanpercentile(yearly_labels, 5) >= thresh:
        logger.info(' Identifying common work hours |')

        kmeans_2 = KMeans(n_clusters=func_params.get('cluster_count'), random_state=0).fit(yearly_labels.reshape(-1, 1))
        labels_2_year = kmeans_2.labels_

        if kmeans_2.cluster_centers_[0] > kmeans_2.cluster_centers_[1]:
            labels_2_year = (~(labels_2_year.astype(bool))) * 1

        labels_2_year = apply_continuity_filter(labels_2_year, width=3 * parameters.get('sampling'))

    else:
        labels_2_year = np.ones(parameters.get('sampling') * Cgbdisagg.HRS_IN_DAY).astype(int)

    logger.info(' User level post processing of work hour label completed |')

    return labels_2_year
