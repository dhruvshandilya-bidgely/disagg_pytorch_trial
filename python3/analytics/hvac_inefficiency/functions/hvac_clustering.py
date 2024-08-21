"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains code for creating cluster to distinguish HVAC fan and compressor
"""

# Import python packages

import datetime
import logging

import copy
import numpy as np
from sklearn.cluster import KMeans

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile
from python3.analytics.hvac_inefficiency.configs.init_cycling_based_config import get_clustering_config


def cluster_hvac_consumption(input_hvac_inefficiency_object, output_hvac_inefficiency_object, logger_pass, device):
    """
        This function estimates solar generation for users with

        Parameters:
            input_hvac_inefficiency_object  (dict)              dictionary containing all input the information
            output_hvac_inefficiency_object (dict)              dictionary containing all output information
            logger_pass                     (object)            logger object
            device                          (string)            consumption device
        Returns:
            input_hvac_inefficiency_object  (dict)              dictionary containing all input the information
            output_hvac_inefficiency_object (dict)              dictionary containing all output information
    """

    # Taking new logger base for this module

    logger_local = logger_pass.get("logger").getChild("cluster_hvac_consumption")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}

    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    cluster_hvac_consumption_time = datetime.datetime.now()

    random_state = input_hvac_inefficiency_object.get('RANDOM_STATE')
    hvac_consumption_array = input_hvac_inefficiency_object.get('hvac_consumption_array')

    # Get config for clustering HVAC consumption

    config = get_clustering_config()

    # Adding fixed clustering method

    optimum_cluster_count = config.get('primary_optimum_cluster_count')

    # Cluster HVAC consumption into optimum number of clusters

    clustering_model = KMeans(n_clusters=optimum_cluster_count, random_state=random_state)
    clustering_out = clustering_model.fit_predict(hvac_consumption_array)

    logger.debug('Finished clustering HVAC consumption |')

    # Check cluster proportion to correct HVAC clusters

    clusters, cluster_proportion = np.unique(clustering_out, return_counts=True)
    cluster_proportion = cluster_proportion / cluster_proportion.sum()

    logger.debug('Finished checking clustering proportion | {} | {}'.format(device, cluster_proportion))

    # Find relative position of cluster centers

    smallest_cluster_idx = 0
    idx = np.argsort(clustering_model.cluster_centers_.ravel())
    smallest_cluster_id = idx[smallest_cluster_idx]

    lowest_cluster_cons_limit = config.get('lowest_cluster_cons_limit')

    lower_cluster_consumption = hvac_consumption_array[clustering_out == smallest_cluster_id]
    limit = super_percentile(lower_cluster_consumption[lower_cluster_consumption != 0], lowest_cluster_cons_limit * 100)

    # Get config for re-clustering condition

    max_lowest_cons_limit = config.get('max_lowest_cons_limit')
    high_cluster_portion_limit = config.get('high_cluster_portion')
    min_fraction_per_cluster = config.get('min_fraction_per_cluster')

    clusters_with_high_proportion = np.sum(cluster_proportion > high_cluster_portion_limit)

    # Check max consumption of the smallest cluster for re-clustering condition

    if (clusters_with_high_proportion == optimum_cluster_count) | (limit >= max_lowest_cons_limit):

        optimum_cluster_count = config.get('secondary_optimum_cluster_count')

        logger.debug('Condition failed, re clustering with secondary cluster count | {} |{}'.format(device, optimum_cluster_count))

        clustering_model = KMeans(n_clusters=optimum_cluster_count, random_state=random_state)
        clustering_out = clustering_model.fit_predict(hvac_consumption_array)

        clusters, cluster_proportion = np.unique(clustering_out, return_counts=True)
        cluster_proportion = cluster_proportion / cluster_proportion.sum()

        logger.debug('Finished checking clustering proportion, new cluster proportion | {} |{}'.format(device, cluster_proportion))

    if cluster_proportion.min() < min_fraction_per_cluster:

        optimum_cluster_count -= 1
        logger.debug('Min fraction failed, re-clustering with lesser cluster count | {} |{}'.format(device, optimum_cluster_count))

        clustering_model = KMeans(n_clusters=optimum_cluster_count, random_state=random_state)
        clustering_out = clustering_model.fit_predict(hvac_consumption_array)

    # Updating information to output dictionary

    initial_clustering_information = {'clustering_model': clustering_model,
                                      'clustering_out': clustering_out,
                                      'optimum_cluster_count': optimum_cluster_count,
                                      'hvac_consumption': hvac_consumption_array}

    output_hvac_inefficiency_object[device]['initial_clustering_information'] = initial_clustering_information

    time_taken = get_time_diff(cluster_hvac_consumption_time, datetime.datetime.now())
    logger.debug('Time taken for clustering | {} | {}'.format(device, time_taken))

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object


def correct_hvac_cluster(input_hvac_inefficiency_object, output_hvac_inefficiency_object, logger_pass, device):
    """
        Cluster correction based on consumption limits per data point

        Parameters:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
            logger_pass                         (object)        logger object
            device                              (str)           string indicating device, either AC or SH
        Returns:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
    """

    logger_local = logger_pass.get("logger").getChild("correct_hvac_cluster")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}

    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    correct_cluster_time = datetime.datetime.now()

    config = get_clustering_config()

    raw_input_data = input_hvac_inefficiency_object.get('raw_input_values')
    demand_hvac_col_idx = input_hvac_inefficiency_object.get('raw_input_idx').get('{}_demand'.format(device))
    ao_hvac_col_idx = input_hvac_inefficiency_object.get('raw_input_idx').get('{}_ao'.format(device))

    hvac_consumption =\
        copy.deepcopy(raw_input_data[:, demand_hvac_col_idx] + raw_input_data[:, ao_hvac_col_idx]).reshape(-1, 1)

    device_config = input_hvac_inefficiency_object.get(device)
    merge_limit = device_config.get('config').get('cluster_merge_limit')
    lower_center_limit = device_config.get('config').get('lower_center_limit')
    medium_center_limit = device_config.get('config').get('medium_center_limit')

    nan_idx = input_hvac_inefficiency_object.get('hvac_nan_idx')

    clustering_model =\
        output_hvac_inefficiency_object.get(device).get('initial_clustering_information').get('clustering_model')
    cluster_centers = clustering_model.cluster_centers_
    idx = np.argsort(cluster_centers.ravel())

    # Get cluster id for each data point

    predicted_clusters = clustering_model.predict(hvac_consumption)
    predicted_clusters = predicted_clusters.reshape(-1, 1)

    # Correct clusters only when there are more than 1 cluster centers

    if len(cluster_centers) > 1:

        logger.debug('More than 1 cluster, checking cluster correction criterion | {}'.format(device))

        # Find smallest, second smallest and largest centers

        smallest_cluster_idx = 0
        second_smallest_cluster_idx = 1
        largest_cluster_idx = -1

        smallest_cluster_id = idx[smallest_cluster_idx]
        largest_cluster_id = idx[largest_cluster_idx]
        second_smallest_cluster_id = idx[second_smallest_cluster_idx]

        # correcting cluster id based on consumption limits

        deviation_factor = config.get('deviation_factor')

        if clustering_model.cluster_centers_[second_smallest_cluster_id] <= merge_limit:

            logger.debug('Cluster center less than merge limit, merging clusters | {}'.format(device))

            # set second smallest cluster id to smallest cluster id

            predicted_clusters[predicted_clusters == second_smallest_cluster_id] = smallest_cluster_id

        elif clustering_model.cluster_centers_[second_smallest_cluster_id] <= lower_center_limit:

            logger.debug('Cluster center less than lower center limit | {}'.format(device))

            # set second smallest cluster id to smallest cluster id

            temp_array = hvac_consumption[predicted_clusters == second_smallest_cluster_id]
            std_deviation = np.nanstd(temp_array)
            median = np.nanmedian(temp_array)
            outlier_limit = median + (deviation_factor * std_deviation)
            temp_array = hvac_consumption < outlier_limit

            # Correcting low consumption points in second smallest cluster
            logger.debug('Correcting low consumption points in second smallest cluster | {}'.format(device))

            temp_array = temp_array & (predicted_clusters == second_smallest_cluster_id)
            predicted_clusters[temp_array] = smallest_cluster_id

        elif clustering_model.cluster_centers_[second_smallest_cluster_id] <= medium_center_limit:

            logger.debug('Cluster center less than medium center limit | {}'.format(device))

            # set second smallest cluster id to smallest cluster id

            temp_array = hvac_consumption[predicted_clusters == second_smallest_cluster_id]
            std_deviation = np.nanstd(temp_array)
            median = np.nanmedian(temp_array)
            outlier_limit = median - (deviation_factor * std_deviation)
            temp_array = hvac_consumption < outlier_limit

            # Correcting low consumption points in second smallest cluster

            logger.debug('Correcting low consumption points in second smallest cluster | {}'.format(device))

            temp_array = temp_array & (predicted_clusters == second_smallest_cluster_id)
            predicted_clusters[temp_array] = smallest_cluster_id
    else:

        logger.debug('Single cluster scenario, setting cluster id to 0 | {}'.format(device))

        single_cluster_id = 0
        smallest_cluster_id = idx[single_cluster_id]
        largest_cluster_id = idx[single_cluster_id]
        second_smallest_cluster_id = idx[single_cluster_id]

    # Adding changes to check condition on proportion

    cluster_information = []

    logger.debug('Computing cluster information | {}'.format(device))

    for id_ in np.unique(predicted_clusters):
        valid_idx = ((predicted_clusters == id_) & (~nan_idx))
        mean = np.nanmedian(hvac_consumption[valid_idx])
        count = np.sum(~np.isnan(hvac_consumption[valid_idx]))
        cluster_information.append([id_, mean, count])

    id_column = 0
    mean_column = 1
    count_column = 2

    cluster_information = np.array(cluster_information)
    total_points = np.sum(cluster_information[:, count_column])
    cluster_information[:, count_column] = cluster_information[:, count_column]/total_points
    cluster_information = cluster_information[cluster_information[:, mean_column].argsort()]

    logger.debug('Correcting clusters based on cluster fraction | {}'.format(device))

    for id_ in range(0, cluster_information.shape[0]):
        min_fraction_per_cluster = config.get('min_fraction_per_cluster')
        if cluster_information[id_, count_column] < min_fraction_per_cluster:
            old_cluster = cluster_information[id_, id_column]
            if id_ == 0:
                new_cluster_id = id_ + 1
                new_cluster = cluster_information[new_cluster_id, id_column]
            else:
                new_cluster = cluster_information[id_ - 1, id_column]

            predicted_clusters[predicted_clusters == old_cluster] = new_cluster

    # Updating cluster information after correction

    cluster_information = []

    logger.debug('Update cluster ids based on cluster fraction | {}'.format(device))

    for id_ in np.unique(predicted_clusters):
        valid_idx = ((predicted_clusters == id_) & (~nan_idx))
        mean = np.nanmedian(hvac_consumption[valid_idx])
        count = np.sum(~np.isnan(hvac_consumption[valid_idx]))
        cluster_information.append([id_, mean, count])

    mean_column = 1
    count_column = 2

    cluster_information = np.array(cluster_information)
    total_points = np.sum(cluster_information[:, count_column])
    cluster_information[:, count_column] = cluster_information[:, count_column] / total_points
    cluster_information = cluster_information[cluster_information[:, mean_column].argsort()]

    cluster_centers = cluster_information[:, mean_column]
    idx = np.argsort(cluster_centers.ravel())

    # Handling single cluster scenario

    if len(cluster_centers) == 1:

        logger.debug('Update cluster ids based on cluster fraction | {}'.format(device))

        single_cluster_id = 0
        smallest_cluster_id = idx[single_cluster_id]
        largest_cluster_id = idx[single_cluster_id]
        second_smallest_cluster_id = idx[single_cluster_id]

        predicted_clusters[:] = single_cluster_id

    updated_cluster_information = {'predicted_clusters': predicted_clusters,
                                   'cluster_information': cluster_information,
                                   'smallest_cluster_id': smallest_cluster_id,
                                   'largest_cluster_id': largest_cluster_id,
                                   'second_smallest_cluster_id': second_smallest_cluster_id,
                                   'cluster_count': len(cluster_centers)}

    output_hvac_inefficiency_object[device]['updated_cluster_information'] = updated_cluster_information

    time_taken = get_time_diff(correct_cluster_time, datetime.datetime.now())
    logger.debug('Time taken for correcting HVAC clusters | {} | {}'.format(device, time_taken))

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object
