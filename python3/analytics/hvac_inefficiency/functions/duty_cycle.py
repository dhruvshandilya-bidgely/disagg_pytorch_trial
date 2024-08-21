"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains code for computing duty cycle
"""

# Import python packages

import time
import numpy as np


def compute_scaled_duty_cycle(unrolled_hvac_consumption, full_cycle_consumption, window_length = 12):
    """
        Compute duty cycle for each data point after scaling based on full cycle consumption

        Parameters:
            unrolled_hvac_consumption       (numpy.ndarray)     array of hvac compressor consumption
            full_cycle_consumption          (float)             representative size of the device
            window_length                   (int)               length of sliding window
        Returns:
            duty_cycle_array                (numpy.ndarray)     array containing duty cycle for each data point
    """

    # Calculating HVAC consumption duty cycle

    scaled_hvac_consumption = unrolled_hvac_consumption / full_cycle_consumption
    scaled_hvac_consumption = np.clip(scaled_hvac_consumption, 0, 1)
    scaled_hvac_consumption = scaled_hvac_consumption.ravel()

    mask = np.ones(window_length, dtype=np.float)
    mask = mask / window_length
    duty_cycle_array = np.convolve(scaled_hvac_consumption, mask, 'same')
    return duty_cycle_array


def compute_absolute_duty_cycle(unrolled_data_largest_cluster, window_length):
    """
            Compute duty cycle for each data point after scaling based on full cycle consumption

            Parameters:
                unrolled_data_largest_cluster       (numpy.ndarray)     array of hvac compressor consumption
                window_length                       (int)               length of sliding window
            Returns:
                duty_cycle_array_largest            (numpy.ndarray)     array containing duty cycle for each data point
    """

    # Calculating HVAC consumption duty cycle
    min_duty_cycle = 0
    max_duty_cycle = 1

    scaled_hvac_consumption = np.clip(unrolled_data_largest_cluster, min_duty_cycle, max_duty_cycle)
    scaled_hvac_consumption = scaled_hvac_consumption.ravel()

    mask = np.ones(window_length, dtype=np.float)
    mask = mask / window_length
    duty_cycle_array_largest = np.convolve(scaled_hvac_consumption, mask, 'same')

    return duty_cycle_array_largest
