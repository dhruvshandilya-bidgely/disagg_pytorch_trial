"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to check the energy concentration across different runs of timed water heater
"""

# Import python packages

import numpy as np


def check_energy_concentration(high_fraction_hours, variables, factor, insignificant_run_threshold,
                               energy_fraction_threshold, minimum_hour_gap):
    """
    Parameters:
        high_fraction_hours         (np.ndarray)    : Array of time divisions with high fractions
        variables                   (dict)          : Variables of multiple runs and edges
        factor                      (int)           : Sampling rate factor
        insignificant_run_threshold (float)         : Minimum fraction for valid run
        energy_fraction_threshold   (float)         : Minimum energy fraction for valid run
        minimum_hour_gap            (int)           : Minimum gap between multiple runs (in hours)

    Returns:
        high_fraction_hours         (np.ndarray)    : Filtered time divisions with high fractions
    """

    # Retrieve the energy fractions

    hod_energy = variables['hod_count']

    # Initialize invalid time divisions array

    invalid_max_hours = np.array([])

    # Maximum energy fraction

    max_energy = np.max(hod_energy[high_fraction_hours])

    # Filter values where energy fraction above certain limit of max energy fraction

    energy_max_hours = np.where(hod_energy >= (energy_fraction_threshold * max_energy))[0]

    # Iterate at each time division to check if it lies close to a max fraction time division

    for hour in high_fraction_hours:
        # Check gap of current time division with max energy time division

        gap = np.min(np.abs(energy_max_hours - hour))

        # If time division has less energy and is far from max energy time division, it is invalid

        if (hod_energy[hour] < (insignificant_run_threshold * max_energy)) and (gap > (minimum_hour_gap * factor)):
            invalid_max_hours = np.r_[invalid_max_hours, hour]

    # Filter out the invalid time divisions from high energy time divisions

    high_fraction_hours = np.array([x for x in high_fraction_hours if x not in invalid_max_hours])

    high_fraction_hours = high_fraction_hours.astype(int)

    return high_fraction_hours
