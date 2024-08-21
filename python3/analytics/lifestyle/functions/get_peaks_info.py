"""
Author - Prasoon Patidar
Date - 17th June 2020
Get consumption peak information for user
"""

# import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.lifestyle.functions.zscore_peak_detection import zscore_peak_detection


def get_peaks_info(day_input_data, day_input_idx, day_clusters, cluster_fractions, lifestyle_input_object, logger_pass):

    """
    Parameters:
        day_input_data (np.ndarray)                : custom trimmed input data formatted as 2-D daily matrix
        day_input_idx(np.ndarray)                  : day value for given day_cluster
        day_clusters(np.ndarray)                   : cluster values for given input days
        cluster_fractions(np.ndarray)              : cluster fractions for given input days
        lifestyle_input_object(dict)               : dictionary containing inputs for lifestyle modules
        logger_pass(dict)                          : contains base logger and logging dictionary
    Returns:
        peaks_info(list)                           : List of peaks for all cluster types
    """

    t_get_peaks_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('get_peaks_info')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    logger.debug("%s Start: Peak Detection for Billcycle Days", log_prefix('DailyLoadType'))

    # Get required input from lifestyle input object

    peak_detection_config = lifestyle_input_object.get('peak_detection_config')

    daily_load_types = lifestyle_input_object.get('daily_load_type')

    daily_kmeans_model = lifestyle_input_object.get('daily_profile_kmeans_model')

    # get cluster labels and cluster center for this kmeans model

    cluster_labels = daily_kmeans_model.get('cluster_labels')

    cluster_centers = daily_kmeans_model.get('cluster_centers')

    # Split peak detection config for convenience

    cluster_fraction_limit = peak_detection_config.get('allowed_cluster_fraction_limit')

    peak_duration_deviation_ratio = peak_detection_config.get('peak_duration_deviation_ratio')

    peak_duration_deviation_min = peak_detection_config.get('peak_duration_deviation_min')

    LAG_IDX, THRESHOLD_IDX, INFLUENCE_IDX = map(peak_detection_config.get,
                                                ['LAG_IDX', 'THRESHOLD_IDX', 'INFLUENCE_IDX'])

    reference_peak_params = peak_detection_config.get('REFERENCE_PEAKS')

    consumption_peak_params = peak_detection_config.get('CONSUMPTION_PEAKS')

    # Loop over clusters in descending order of cluster fraction till limit

    cluster_fraction_sum = 0

    peaks_info = []

    for cluster_id in np.argsort(cluster_fractions)[::-1]:

        logger.debug("%s Peak Detection for Cluster %s", log_prefix('DailyLoadType'), daily_load_types(cluster_id).name)

        # Get cluster input data and day idx

        cluster_input_data = day_input_data[day_clusters == cluster_id]

        cluster_input_idx = day_input_idx[day_clusters == cluster_id]

        # Get cluster center for given cluster_id

        cluster_name = daily_load_types(cluster_id).name

        cluster_center_idx = cluster_labels.index(cluster_name)

        cluster_center = cluster_centers[cluster_center_idx]

        # Get Peaks for this cluster center

        reference_peaks = zscore_peak_detection(cluster_center,
                                                reference_peak_params[LAG_IDX],
                                                reference_peak_params[THRESHOLD_IDX],
                                                reference_peak_params[INFLUENCE_IDX])

        # Get peaks for all days in input data

        get_consumption_peaks = lambda x: zscore_peak_detection(x, consumption_peak_params[LAG_IDX],
                                                                consumption_peak_params[THRESHOLD_IDX],
                                                                consumption_peak_params[INFLUENCE_IDX])
        cons_peaks = np.apply_along_axis(get_consumption_peaks,
                                         axis=1,
                                         arr=cluster_input_data)

        # Set id for each peak day as day epoch

        cons_peaks = np.array([cons_peaks[i].set_id(cluster_input_idx[i])
                               for i in range(len(cluster_input_idx))])

        # filter any zero peak days and merge other day peaks

        cons_peaks = np.array([day_peaks.merge_peaks() for day_peaks in cons_peaks
                               if (day_peaks.num_peaks > 0)])

        # Loop over reference peaks to get average peak times

        final_cluster_peaks = []

        for ref_peak_idx, ref_peak_median in enumerate(reference_peaks.medians):

            # Get maximum deviation allowed with peaks

            max_deviation_allowed = max(peak_duration_deviation_min,
                                        peak_duration_deviation_ratio * reference_peaks.durations[ref_peak_idx])

            # Get eligible peak start end times and days idx for all merged peaks

            start_times = np.full_like(cons_peaks, fill_value=np.nan, dtype=np.float64)

            end_times = np.full_like(cons_peaks, fill_value=np.nan, dtype=np.float64)

            peak_ids = np.array(list(map(lambda x: x.id, cons_peaks)))

            peak_strengths = np.full_like(cons_peaks, fill_value=np.nan, dtype=np.float64)

            for i, day_peaks in enumerate(cons_peaks):

                # Get eligible peak for this day

                closest_peak_dist = np.min(np.absolute(day_peaks.medians - ref_peak_median))

                if closest_peak_dist <= max_deviation_allowed:
                    closest_peak_idx = int(np.argmin(np.absolute(day_peaks.medians - ref_peak_median)))

                    # store start,end and strength of closest peak to ref peak on this day

                    start_times[i] = day_peaks.start_times[closest_peak_idx]

                    end_times[i] = day_peaks.end_times[closest_peak_idx]

                    peak_strengths[i] = \
                        np.median(cluster_input_data[cluster_input_idx == day_peaks.id, int(start_times[i]):int(end_times[i])])

            if np.all(np.isnan(start_times)):

                logger.debug("%s Not Peaks found for reference peak median %s ", log_prefix('DailyLoadType'),
                             str(ref_peak_median))

            else:

                final_cluster_peaks.append({
                    'start_time'       : np.nanmedian(start_times),
                    'end_time'         : np.nanmedian(end_times),
                    'start_time_std'   : np.nanstd(start_times),
                    'end_time_std'     : np.nanstd(end_times),
                    'observation_count': np.count_nonzero(~np.isnan(start_times)),
                    'strength'         : np.nanmedian(peak_strengths),
                    'dates'            : list(peak_ids[~np.isnan(start_times)])
                })

                logger.debug(
                    "%s Peak found for reference peak median %s: start: %.3f +/- %.3f, end: %.3f +/- %.3f, strength %.3f, days_count %.3f",
                    log_prefix('DailyLoadType'), str(ref_peak_median),
                    round(final_cluster_peaks[-1]['start_time'], 2),
                    round(final_cluster_peaks[-1]['start_time_std'], 2),
                    round(final_cluster_peaks[-1]['end_time'], 2),
                    round(final_cluster_peaks[-1]['end_time_std'], 2),
                    round(final_cluster_peaks[-1]['strength'], 2),
                    final_cluster_peaks[-1]['observation_count']
                )
        # append peaks for this cluster in peaks info

        peaks_info.append({
            'load_type'    : cluster_name,
            'load_fraction': cluster_fractions[cluster_id],
            'load_peaks'   : final_cluster_peaks
        })

        # add cluster fraction to cluster fraction sum
        cluster_fraction_sum += cluster_fractions[cluster_id]

        # Break if cluster fraction sum exceeds limit

        if cluster_fraction_sum > cluster_fraction_limit:
            break

    t_get_peaks_end = datetime.now()

    logger.info("%s Got peaks information in | %.3f s", log_prefix('DailyLoadType'),
                get_time_diff(t_get_peaks_start, t_get_peaks_end))

    return peaks_info
