
"""
Author - Nisha Agarwal
Date - 3rd Jan 21
Identify clean days from the given days
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from numpy.random import RandomState

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import rolling_func_along_col

from python3.itemization.init_itemization_config import init_itemization_params

from python3.itemization.aer.behavioural_analysis.clean_day_score.get_day_score import get_clean_days_score
from python3.itemization.aer.behavioural_analysis.clean_day_score.config.init_clean_day_score_config import get_clean_day_score_config


def get_day_cleanliness_score(item_input_object, item_output_object, logger_pass):

    """
    Identify clean days from the given days and calculate cleanliness score for all days

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_day_cleanliness_score')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_activity_profile_start = datetime.now()

    clean_days_score_config = get_clean_day_score_config()
    seq_config = init_itemization_params().get('seq_config')

    # fetch required information from disagg pipeline

    vacation = item_input_object.get("item_input_params").get('vacation_data')
    input_data = item_input_object.get("item_input_params").get('day_input_data')
    temperature = item_input_object.get("item_input_params").get('temperature_data')
    samples_per_hour = item_input_object.get("item_input_params").get('samples_per_hour')

    vacation_days = np.sum(vacation, axis=1).astype(bool)

    total_days = np.sum(np.logical_not(vacation_days))

    # Initialize clean day score and bool value arrays

    clean_days_score = np.zeros(len(input_data))
    non_clean_day_bool_array = np.zeros(input_data.shape[0])

    # calculate rolling sum of input data

    window = clean_days_score_config.get("mask_clean_days_config").get('rolling_avg_window')*samples_per_hour

    t1 = datetime.now()

    kmeans, high_cons_bool_array, input_data_rolling_sum = prepare_high_cons_bool_array(input_data, vacation_days, window, clean_days_score_config)

    logger.info("Prepared high cons boolean array")

    t2 = datetime.now()

    logger.info("Clustering took | %.3f s",
                get_time_diff(t1, t2))
    logger.info("Cluster centers | %s", np.round(kmeans.cluster_centers_))

    # steps to calculate parameters to used further for calculating day cleanliness score

    # calculate 25th and 35th percentile score of day consumption

    yearly_consumption = np.sum(input_data, axis=1)
    yearly_consumption = yearly_consumption[np.logical_not(vacation_days)]

    t3 = datetime.now()

    if not np.all(vacation_days):

        dev_for_score = np.percentile(np.unique(yearly_consumption),
                                      clean_days_score_config.get("clean_day_score_config").get('perc_for_dev'))
        mean_for_score = np.percentile(np.unique(yearly_consumption),
                                       clean_days_score_config.get("clean_day_score_config").get('perc_for_mean'))
        threshold = np.percentile(np.unique(input_data),
                                  clean_days_score_config.get("clean_day_score_config").get('threshold'))

        input_data_copy = (input_data > threshold).astype(int)

        # mean temperature of each day
        temperature = np.mean(temperature, axis=1)

        logger.debug("Calculated information required for day scoring")

        logger.info("Prediction of high consumption | %.3f s",
                    get_time_diff(t2, t3))

        for day in range(len(input_data)):

            logger.debug("Calculating cleanliness score for day %d", day)

            # mark each timestamp as low or high consumption

            day_bool_array = high_cons_bool_array[day]

            # calculate the length of longest continuous high consumption points

            seq = find_seq(day_bool_array, derivative=np.zeros(len(day_bool_array)), arr=np.zeros(len(day_bool_array)))

            max_length = 0

            if np.any(seq[:, seq_config.get('label')] == 1):
                seq = seq[seq[:, seq_config.get('label')] == 1]
                max_length = seq[np.argmax(seq[:, seq_config.get('length')]), seq_config.get('length')]
                seq = seq[:, seq_config.get('length')]
            else:
                seq = []

            if max_length > clean_days_score_config.get('mask_clean_days_config').get('min_high_cons_limit')*samples_per_hour \
                    and np.max(input_data[day]) > clean_days_score_config.get('mask_clean_days_config').get('min_cons')/samples_per_hour:
                non_clean_day_bool_array[day] = 1

            bucket_count = np.sum(input_data_copy[day])

            # Calculate cleanliness score if its a non vacation day

            if (not vacation_days[day]) and (not np.all(input_data[day] == 0)):
                clean_days_score[day] = get_clean_days_score(max_length, samples_per_hour, input_data[day].sum(),
                                                             temperature[day], dev_for_score, mean_for_score, seq,
                                                             bucket_count, clean_days_score_config)

    t4 = datetime.now()

    logger.info("Masking and scoring for all days took | %.3f s",
                get_time_diff(t3, t4))

    non_clean_day_bool_array = non_clean_day_bool_array.astype(bool)

    final_mask_array = np.zeros(input_data.shape)

    final_mask_array[non_clean_day_bool_array, :] = -1

    # calculate fraction of clean days

    clean_day_fraction = 1 - np.round(np.sum(non_clean_day_bool_array)/total_days, 2)

    # pass the calculated parameters for further computations

    clean_day_score_object = {
        "clean_day_fraction": clean_day_fraction,
        "rolling_sum_input_data": input_data_rolling_sum,
        "clean_day_masked_array": final_mask_array,
        "clean_day_score": clean_days_score,
        "cluster_centers": kmeans.cluster_centers_
    }

    item_input_object.update({
        "clean_day_score_object": clean_day_score_object
    })

    item_output_object.get('debug').update({
        "clean_day_dict": item_input_object.get("clean_day_score_object")
    })

    t_activity_profile_end = datetime.now()

    logger.info("Clean days module took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return item_input_object, item_output_object


def prepare_high_cons_bool_array(input_data, vacation_days, window, clean_days_score_config):

    """
    Prepare bool array , where True value represents the timestamp lies in high consumption bucket

    Parameters:
        input_data                  (np.ndarray)       : original day input data
        vacation_days               (np.ndarray)       : vacation days bool array
        window                      (int)              : window size for calculating rolling sum
        clean_days_score_config     (dict)             : config dict for calculating bool array

    Returns:
        kmeans                      (dict)             : trained kmeans model
        high_cons_bool_array        (np.ndarray)       : prepared high cons boolean array
        input_data_rolling_sum      (np.ndarray)       : rolling sum input data ( along the col axis)

    """

    input_data_rolling_sum = rolling_func_along_col(input_data, (window)/2, 0)

    # cluster the rolling sum data points into two segments

    consumption_points = input_data_rolling_sum[np.logical_not(vacation_days)].flatten()
    consumption_points = consumption_points[consumption_points > -1].reshape(-1, 1)

    # Decreasing sample size of given data points , for decreasing runtime of clustering

    if np.all(vacation_days):
        consumption_points = input_data_rolling_sum[:].flatten()
        consumption_points = consumption_points[consumption_points > -1].reshape(-1, 1)
        min_val = np.min(consumption_points)
        max_val = np.max(consumption_points)
    else:
        min_val = np.min(consumption_points)
        max_val = np.max(consumption_points)

    if max_val == min_val:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(consumption_points.reshape(-1, 1))

    else:
        high_perc = np.percentile(consumption_points, clean_days_score_config.get("clean_day_score_config").get("perc_cap"))
        bucket_size = high_perc * clean_days_score_config.get('mask_clean_days_config').get('bucket_size_factor')
        bucket_size = max(clean_days_score_config.get('mask_clean_days_config').get('max_bucket_size'),
                          np.round(bucket_size, -1))

        temp_cons_points = np.ones(len(consumption_points)) * -1

        # Downsampling the points in order to reduce runtime of clustering

        bucket_labels = np.digitize(consumption_points, np.arange(min_val, max_val, bucket_size))

        unique_bucket_labels, bucket_count = np.unique(bucket_labels, return_counts=True)

        bucket_count = (bucket_count / clean_days_score_config.get('mask_clean_days_config').get(
            'clustering_fraction')).astype(int)

        count = 0

        for index, label in enumerate(unique_bucket_labels):
            temp_cons_points[count: count + bucket_count[index] + 1] = consumption_points[bucket_labels == label][:bucket_count[index] + 1]
            count = count + bucket_count[index]

        temp_cons_points = temp_cons_points[temp_cons_points > -1]

        if len(temp_cons_points) < clean_days_score_config.get('mask_clean_days_config').get('points_count_limit'):
            # Downsampling not required
            kmeans = KMeans(n_clusters=2, random_state=0).fit(consumption_points.reshape(-1, 1))
        else:
            # Downsampling required
            seed = RandomState(1234567890)
            seed.shuffle(temp_cons_points)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(temp_cons_points.reshape(-1, 1))

    consumption_level_boundry = np.mean(kmeans.cluster_centers_)

    high_cons_bool_array = np.zeros(input_data.shape)
    high_cons_bool_array[input_data_rolling_sum > consumption_level_boundry] = 1

    return kmeans, high_cons_bool_array, input_data_rolling_sum
