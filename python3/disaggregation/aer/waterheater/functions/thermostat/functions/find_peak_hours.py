"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module finds the hours of active usage of non-timed water heater
"""

# Import python packages

import numpy as np
from scipy.signal import find_peaks


def find_peak_hours(max_amp_hours_counts, min_peak_distance, wh_config, logger):
    """
    Parameters:
        max_amp_hours_counts    (np.ndarray)    : Count of peaks in every hour of the day
        min_peak_distance       (int)           : Minimum distance between two peaks
        wh_config               (dict)          : Water heater params
        logger                  (logger)        : Logger object

    Returns:
        peaks                     (np.ndarray)    : Value of peaks selected
        locs                    (np.ndarray)    : index of the peaks selected
        widths                  (np.ndarray)    : width of the peaks selected
        proms                   (np.ndarray)    : prominence of the peaks selected
    """

    # Retrieve required params from the config

    estimation_config = wh_config['thermostat_wh']['estimation']

    num_peaks = estimation_config['allowed_peaks_count']

    peak_width = estimation_config['peak_width_bounds']

    # Finding all peaks irrespective of peak distances

    peak_info = find_peaks(max_amp_hours_counts, threshold=0, width=peak_width)

    # Meta data created for filtering peaks (each peak, peak's index)

    peak_info_tuple = [(max_amp_hours_counts[i], i) for i in peak_info[0]]

    # Determine the left and right positions of each peak

    positions = [(i, j) for i, j in zip(peak_info[0], range(len(peak_info[0])))]
    ind_left = np.array(peak_info[0])

    # Keeping peaks only with a gap of at least given hours (after sorting based on count)

    for i in np.argsort(-np.array([a[0] for a in peak_info_tuple])):
        if peak_info_tuple[i][1] in ind_left:
            ind_left = np.r_[ind_left[(ind_left < (peak_info_tuple[i][1] - min_peak_distance)) |
                                      (ind_left > (peak_info_tuple[i][1] + min_peak_distance))], peak_info_tuple[i][1]]

    # Select the top two peaks based on count

    ind_left = np.sort(ind_left)
    pos_left = [tup[1] for tup in positions if tup[0] in ind_left]
    pks_index = np.argsort(-max_amp_hours_counts[ind_left])[:num_peaks]

    # Get the corresponding peaks hour values

    peaks = np.sort(max_amp_hours_counts[ind_left])[::-1][:num_peaks]
    locs = []

    # Keeping only top-2 peaks (if present)

    for peak in peaks:
        temp_loc = np.intersect1d(np.where(max_amp_hours_counts == peak), ind_left)[0]
        locs = np.r_[locs, temp_loc]
        ind_left = ind_left[ind_left != temp_loc]

    # Calculate the width and prominence of the final peaks

    widths = np.take(peak_info[1]['widths'][pos_left], pks_index)
    proms = np.take(peak_info[1]['prominences'][pos_left], pks_index)

    logger.info('The fat pulse peak hours detected are | {}'.format(locs))

    return peaks, locs, widths, proms


def get_peak_range(edges, peak_hour, peak_size, hourly_count, peak_height, fat_duration_limit):
    """
    Parameters:
        edges                   (np.ndarray)        : Edges of energy histogram
        peak_hour               (int)               : Hour of the peak
        peak_size               (int)               : Size of the peak
        hourly_count            (np.ndarray)        : Count of pulses at each hour
        peak_height             (float)             : Fraction of accepted peak height
        fat_duration_limit      (int)               : Fat pulse duration limit

    Returns:
        left_edge               (int)               : Left hour of the peak
        right_edge              (int)               : Right hour of the peak
    """

    # Get the left and right portions of the peak

    left_hours = edges[edges < peak_hour]
    right_hours = edges[edges > peak_hour]

    # Find the left stopping index at less than half of peak

    left_edge_idx = np.where(hourly_count[left_hours] < (peak_height * peak_size))[0]

    if len(left_edge_idx) > 0:
        # If valid left edge found, return left side fat hours

        left_edge = np.fmax(left_hours[left_edge_idx[-1]], peak_hour - fat_duration_limit)
    else:
        # If no valid left edge, return default left side fat hours

        left_edge = peak_hour - fat_duration_limit

    # Find the right stopping index at less than half of peak

    right_edge_idx = np.where(hourly_count[right_hours] < (peak_height * peak_size))[0]

    if len(right_edge_idx) > 0:
        # If valid right edge found, return right side fat hours

        right_edge = np.fmin(right_hours[right_edge_idx[0]], peak_hour + fat_duration_limit)
    else:
        # If no valid right edge, return default right side fat hours

        right_edge = peak_hour + fat_duration_limit

    return left_edge, right_edge
