"""
Author - Prasoon Patidar
Date - 17th June 2020
Peak detection module using zscore and moving average
"""

# import python packages

import numpy as np


class Peaks:

    """ Class for storing peaks information for a peak signal of a day"""

    def __init__(self, peak_signal, id=None):

        """
        Parameters:
            peak_signal  (np.ndarray)                  : hourly peak value
            id           (np.ndarray)                  : day value for given day_cluster
        """

        # Initialize peaks object based on 24 hour array

        self.signal = peak_signal

        self.id = id

        # get start and end times for peaks on this day

        start_times = np.where(peak_signal[:-1] < peak_signal[1:])[0]

        end_times = np.where(peak_signal[:-1] > peak_signal[1:])[0]

        # update start/end times if start/end hours included in peak

        if peak_signal[0] > 0:
            start_times = np.insert(start_times, 0, 0)

        if peak_signal[-1] > 0:
            end_times = np.insert(end_times, len(end_times), 23)

        self.start_times, self.end_times = start_times, end_times

        # get Duration and Median values and count of peaks in the data

        self.durations = self.end_times - self.start_times

        self.medians = (self.start_times + self.end_times) / 2

        self.num_peaks = self.medians.shape[0]

    def set_id(self, id):

        # Add id to this day, set of peaks

        self.id = id

        return self

    def merge_peaks(self):

        # Initialize merged signal

        merged_signal = self.signal

        # loop over to merge adjacent peaks

        for i in range(1, merged_signal.shape[0] - 1):

            if (merged_signal[i - 1] > 0) & (merged_signal[i + 1] > 0) & (merged_signal[i] == 0):

                merged_signal[i] = (merged_signal[i - 1] + merged_signal[i + 1]) / 2

        return Peaks(merged_signal, id=self.id)


def zscore_peak_detection(input_signal, lag, threshold, influence):

    """
    Parameters:
        input_signal (np.ndarray)                : hour level 24 point array for day input data
        lag(float)                               : lag duration, no. of lag hours to consider for peak identification
        threshold(float)                         : threshold to classify as peak/nonpeak shift
        influence(float)                         : influence of new hour on moving mean and std values
    Returns:
        peaks (object)                           : Object storing all information regarding detected peaks
    """

    # Add lag into input signal as a padding

    signal = np.zeros(input_signal.shape[0] + lag)

    signal[lag:] = input_signal

    signal[:lag] = input_signal[-lag:]

    # Initialize filter array(with lag) to keep note of moving mean, std

    filter_array = np.zeros_like(signal)

    filter_array[:lag] = signal[:lag]

    moving_mean = filter_array[:lag].mean()
    moving_std = filter_array[:lag].std()

    # Initialize peak signal

    peak_signal = np.zeros_like(signal)

    # loop over arr to find peaks in signal

    for i in range(int(lag), signal.shape[0]):

        if (signal[i] - moving_mean) > threshold * moving_std:

            # Peak detected

            peak_signal[i] += 1

            # update filter_array based on influence parameter

            filter_array[i] = influence * signal[i] + (1 - influence) * filter_array[i - 1]

        else:

            # Add current signal val to filter array

            filter_array[i] = signal[i]

        # update moving mean and std

        moving_mean = filter_array[i - lag:i].mean()
        moving_std = filter_array[i - lag:i].std()

    # get original size peak signal

    peak_signal = peak_signal[lag:]

    return Peaks(peak_signal)
