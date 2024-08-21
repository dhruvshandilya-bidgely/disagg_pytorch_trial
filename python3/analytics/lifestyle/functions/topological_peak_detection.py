"""
Author - Prasoon Patidar
Date - 22nd June 2020
Peak detection module using topology and river method
"""

# import python packages

import numpy as np


class TpPeak:

    """Class to represent each topological peak in signal"""

    def __init__(self, start_idx):

        #TODO(Nisha) : Add description of these variables

        # initialize peak from its start index

        self.born = self.left = self.right = start_idx

        # set died index to None

        self.died = None

    def get_persistence(self, signal):

        # get height of peak as on when it was born and when died

        if self.died is None:

            return np.inf

        else:

            return signal[self.born] - signal[self.died]

    def __str__(self):

        # string representation for debugging purposes

        peak_str = 'peak born:%d,left:%d,right:%d' % (self.born, self.left, self.right)

        return peak_str


def topological_peak_detection(signal, break_threshold, logger_pass):

    """
    Parameters:
        input_signal (np.ndarray)                : hour level 24 point array for day input data
        break_threshold(float)                   : threshold to stop going further down for more peaks
        logger_pass(dict)                        : contains base logger and logging dictionary
    Returns:
        peaks (np.ndarray)                       : list of objects storing all information regarding detected peaks
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('topological_peak_detection')
    logger_pass['logger_base'] = logger_base

    # initialize peak object array

    peaks = []

    # Maps indices to peaks

    idx_to_peak = [None for s in signal]

    # Sequence indices sorted by values

    indices = range(len(signal))
    indices = sorted(indices, key=lambda i: signal[i], reverse=True)

    # Process each sample in descending order

    for idx in indices:

        peaks, signal, idx_to_peak = process_sample(idx, signal, idx_to_peak, peaks)

        # if signal strength less than break threshold, exit the loop

        if (break_threshold is not None) and (signal[idx] < break_threshold):
            break

    return sorted(peaks, key=lambda p: p.get_persistence(signal), reverse=True)


def process_sample(idx, signal, idx_to_peak, peaks):

    """Process each sample in descending order"""

    # check if left index/ right index is processed or not

    lftdone = (idx > 0 and idx_to_peak[idx - 1] is not None)
    rgtdone = (idx < len(signal) - 1 and idx_to_peak[idx + 1] is not None)

    # Get peaks encompassing left and right side if processed

    il = idx_to_peak[idx - 1] if lftdone else None
    ir = idx_to_peak[idx + 1] if rgtdone else None

    # New peak born: neither left or right is part of a peak

    if not lftdone and not rgtdone:
        peaks.append(TpPeak(idx))
        idx_to_peak[idx] = len(peaks) - 1

    # Directly merge to next peak left: if left is a peak

    if lftdone and not rgtdone:
        peaks[il].right += 1
        idx_to_peak[idx] = il

    # Directly merge to next peak right: if right is a peak

    if not lftdone and rgtdone:
        peaks[ir].left -= 1
        idx_to_peak[idx] = ir

    # Merge left and right peaks: if both of them are peaks

    if lftdone and rgtdone:

        if signal[peaks[il].born] > signal[peaks[ir].born]:

            # Left was born earlier: merge right to left

            peaks[ir].died = idx
            peaks[il].right = peaks[ir].right
            idx_to_peak[peaks[il].right] = idx_to_peak[idx] = il

        else:

            # Right was born earlier: merge left to right

            peaks[il].died = idx
            peaks[ir].left = peaks[il].left
            idx_to_peak[peaks[ir].left] = idx_to_peak[idx] = ir

    return peaks, signal, idx_to_peak


def get_closest_peaks(peak_arr, ref_peak, distance_threshold, duration_threshold, distance_weight, duration_weight,
                      best_peak_start_limit, best_peak_end_limit, logger_pass):
    """
    Parameters:
        peak_arr (np.ndarray)                      : list of peaks for a given day
        ref_peak  (np.ndarray)                     : reference peak, the peak we are trying to get closest to
        distance_threshold(float)                  : distance limit between peaks birth time to qualify as close peak
        duration_threshold(float)                  : difference in peak duration limit to qualify as a close peak
        distance_weight(float)                     : weight assigned to difference in distance for scoring
        duration_weight(float)                     : weight assigned to difference in duration for scoring
        best_peak_start_limit(float)               : start hour for modifying close peaks
        best_peak_end_limit(float)                 : end hour for modifying close peaks
        logger_pass(dict)                          : contains base logger and logging dictionary
    Returns:
        peak(object: TP_Peak)                      : modified peak which is closest to reference peak
    """

    # get ref duration and function to qualify peak

    ref_duration = ref_peak.right - ref_peak.left + 1

    fn_qualified_peak = lambda x: (np.absolute(x.born - ref_peak.born) <= distance_threshold) \
                                  & (((x.right - x.left + 1) - ref_duration) < duration_threshold)

    fn_vec_qualified_peak = np.vectorize(fn_qualified_peak)

    # filter peaks based on duration(directionally more) and distance threshold

    is_qualified_peak = fn_vec_qualified_peak(peak_arr)

    filtered_peak_arr = peak_arr[is_qualified_peak]

    # return None if no peak is left in filtered peak

    if filtered_peak_arr.shape[0] == 0:
        return None

    fn_vec_peak_score = np.vectorize(lambda x:
                                     get_peak_score(x, ref_peak, distance_weight, duration_weight))

    filtered_peak_scores = fn_vec_peak_score(filtered_peak_arr)

    # get best peak before peak alterations

    best_peak_before_alteration = filtered_peak_arr[np.argmin(filtered_peak_scores)]

    # make changes based on hour limits

    # remove outside peaks: any peaks where left is greater than hour limit or right less than hour limit
    # remove complete overlapping peaks: any peak with left less than hour limit and right greater than hour limit

    fn_unqualified_filtered_peak = lambda x: ((x.left >= best_peak_end_limit) | (x.right <= best_peak_start_limit)) \
                                           | ((x.left < best_peak_start_limit) & (x.right > best_peak_end_limit))

    fn_vec_unqualified_filtered_peak = np.vectorize(fn_unqualified_filtered_peak)

    is_unqualified_filtered_peak = fn_vec_unqualified_filtered_peak(filtered_peak_arr)

    filtered_peak_arr = filtered_peak_arr[~is_unqualified_filtered_peak]

    # return best peak before alteration if no peak is left after filteration

    if len(filtered_peak_arr) == 0:
        return best_peak_before_alteration

    # make any peak whose right is more than hour limit to hour limit, and if left is less than hour limit to hour limit

    for i, peak in enumerate(filtered_peak_arr):

        if peak.right > best_peak_end_limit:
            filtered_peak_arr[i].right = best_peak_end_limit

        if peak.left < best_peak_start_limit:
            filtered_peak_arr[i].left = best_peak_start_limit

    altered_peak_scores = fn_vec_peak_score(filtered_peak_arr)

    best_peak_after_alteration = filtered_peak_arr[np.argmin(altered_peak_scores)]

    return best_peak_after_alteration


def get_peak_score(peak1, peak2, distance_weight, duration_weight):
    """
    Parameters:
        peak1  (np.ndarray)                        : peak for which closeness score is calculated
        peak2  (np.ndarray)                        : peak from which closeness score is calculated
        distance_weight(float)                     : weight assigned to difference in distance for scoring
        duration_weight(float)                     : weight assigned to difference in duration for scoring
    Returns:
        peak_score(float)                          : closeness score of peak1 from peak2
    """

    # get distance and duration weighted scores

    distance_score = (distance_weight * np.absolute(peak1.born - peak2.born))

    duration_score = (duration_weight * np.absolute((peak1.right - peak1.left) - (peak2.right - peak2.left)))

    return distance_score + duration_score
