"""
Author - Prasoon Patidar
Date - 08th June 2020
Get Cluster(Kmeans Model) Based information for user
"""

# import python packages

import copy
import logging
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cdist

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff


def get_daily_clusters(lifestyle_input_object, logger_pass):

    """
    Parameters:
        lifestyle_input_object (dict)              : Dictionary containing all inputs for lifestyle modules
        logger_pass(dict)                          : Contains base logger and logging dictionary

    Returns:
        day_cluster_arr(np.ndarray)                : cluster_ids for all day values
        input_day_profile_data                     : Profile Data(Min/Max normed) for all days
    """

    t_daily_clusters_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_daily_clusters')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: get Daily Clusters for all Days", log_prefix('DailyLoadType'))

    # Get day level data for user

    day_input_data = lifestyle_input_object.get('day_input_data')

    # copy input data for load profile for all days using min/max normalisation

    day_profile_data = copy.deepcopy(day_input_data)

    # get min and max values for all days

    day_data_min_vals = np.nanmin(day_profile_data, axis=1)

    day_data_max_vals = np.nanmax(day_profile_data, axis=1)

    # Normalize day_input_data to create load profile for days

    day_profile_data -= day_data_min_vals.reshape(-1, 1)

    day_profile_data /= (day_data_max_vals - day_data_min_vals).reshape(-1, 1)

    # Set NaN Values to be zero for profile data

    #TODO(Nisha): Obvious error in dev code: commenting this line to make results same.

    # Get cluster centers from kmeans daily model for user from input

    kmeans_daily_model = lifestyle_input_object.get('daily_profile_kmeans_model')
    cluster_centers = kmeans_daily_model.get('cluster_centers')

    # Get cluster ids(in int) for this specific model based on min distance

    day_cluster_distances = cdist(day_profile_data, cluster_centers, 'euclidean')

    day_cluster_position = np.argmin(day_cluster_distances, axis=1)

    # get cluster labels from model and enum for daily load types

    cluster_labels = kmeans_daily_model.get('cluster_labels')

    daily_load_types = lifestyle_input_object.get('daily_load_type')

    # get clusters based on cluster ids from kmeans cluster labels

    get_cluster_fn = np.vectorize(lambda x: daily_load_types[cluster_labels[x]].value)

    day_cluster_arr = get_cluster_fn(day_cluster_position)

    t_daily_clusters_end = datetime.now()

    logger.info("%s Getting clusters at day level took | %.3f s", log_prefix('DailyLoadType'),
                get_time_diff(t_daily_clusters_start, t_daily_clusters_end))

    return day_cluster_arr, day_profile_data


def get_cluster_fractions(input_data, day_clusters, day_cluster_idx, daily_load_type, logger_pass):

    """
    Parameters:
        input_data (np.ndarray)                    : Custom trimmed input data
        day_cluster(dict)                          : cluster values for complete raw data available
        day_cluster_idx(dict)                      : day value for given day_cluster
        daily_load_type(enum.Enum)                 : Enum for various daily load types
        logger_pass(dict)                          : Contains base logger and logging dictionary
    Returns:
        cluster_fractions(np.ndarray)              : cluster fractions indexed by daily load type enum
    """

    t_get_cluster_fraction_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_cluster_fractions')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # Get cluster value for all input_data_rows

    days_val_unique, day_val_row_idx = np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)

    # Get cluster value for each unique day_val

    day_cluster_vals = np.array([day_clusters[day_cluster_idx == day][0]
                                 for day in days_val_unique])

    # Get cluster_fraction value for each input data row

    input_data_cluster_vals = day_cluster_vals[day_val_row_idx]

    # Create new cluster fraction array based on total cluster counts

    cluster_fraction_array = np.zeros(daily_load_type.count.value)

    # return empty array if input data is empty

    if input_data.shape[0]==0:

        return cluster_fraction_array

    # Get bincount for all cluster values

    cluster_bincount = np.bincount(input_data_cluster_vals.astype(int))

    # Set cluster Fraction array using cluster bincounts

    cluster_fraction_array[:cluster_bincount.shape[0]] = cluster_bincount

    # Get fraction from total cluster fraction

    cluster_fraction_array /= np.sum(cluster_fraction_array)

    #TODO(Nisha): remove in final version if required

    cluster_fraction_array = np.round(cluster_fraction_array, 2)

    t_get_cluster_fraction_end = datetime.now()

    logger.debug("%s Getting cluster fractions took | %.3f s", log_prefix('DailyLoadType'),
                 get_time_diff(t_get_cluster_fraction_start, t_get_cluster_fraction_end))

    return cluster_fraction_array
