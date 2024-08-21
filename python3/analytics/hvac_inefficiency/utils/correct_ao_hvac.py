"""
Author - Anand Kumar Singh
Date - 26th June 2021
HVAC inefficiency HSM related operations
"""

# Import python packages

import numpy as np


def correct_ao_hvac(hvac_demand, hvac_ao):
    """
            This function correct overestimated AO hvac
            Parameters:
                hvac_demand      (numpy.ndarray)          numpy array containing demand hvac
                hvac_ao          (numpy.ndarray)          numpy array containing ao hvac
            Returns:
                hvac_demand      (numpy.ndarray)          numpy array containing corrected demand hvac
                hvac_ao          (numpy.ndarray)          numpy array containing corrected ao hvac
    """

    # Find ao hvac distribution

    non_zero_ao_idx = hvac_ao != 0
    hvac_non_zero_ao = hvac_ao[non_zero_ao_idx]
    median_ao_hvac = np.median(hvac_non_zero_ao)
    std_ao_hvac = np.std(hvac_non_zero_ao)

    # Remove outlier-ish AO values
    ao_based_threshold = median_ao_hvac + (1.5 * std_ao_hvac)

    # Finding threshold based on this information

    high_ao_idx = hvac_ao > ao_based_threshold
    access_ao = np.zeros_like(hvac_ao, dtype=np.float)
    access_ao[high_ao_idx] = hvac_ao[high_ao_idx] - ao_based_threshold
    hvac_ao[high_ao_idx] = ao_based_threshold

    # Update demand hvac
    hvac_demand = hvac_demand + access_ao

    return hvac_demand, hvac_ao
