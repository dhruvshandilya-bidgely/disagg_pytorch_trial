"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Class containing info of feature columns in thermstat water heater detection
"""


class WhFeatures:
    """
    Contains info of water heater detection features
    """

    # Number of base features
    n_base = 3

    # Number of derived features
    n_features = 18

    # Average energy per thin pulse
    ENERGY = 0

    # Standard deviation of thin pulses
    ENERGY_STD = 1

    # Count of thin pulses
    COUNT_PEAKS = 2

    # Count of peaks per LAP
    PEAKS_PER_LAP = 3

    # Count of thin pulses during night
    NIGHT_PEAKS = 4

    # Count of thin pulses during day
    DAY_PEAKS = 5

    # Average gap in hours between consecutive thin pulses
    PEAK_FACTOR = 6

    # Number of peaks per hour
    PEAKS_PER_HOUR = 7

    # Minimum time gap in hours between consecutive thin pulses
    MIN_TIME_DIFF = 8

    # Maximum time gap in hours between consecutive thin pulses
    MAX_TIME_DIFF = 9

    # Median time gap in hours between consecutive thin pulses
    MEDIAN_TIME_DIFF = 10

    # Consistency factor based on detection of thin pulses across days
    CONSISTENCY = 11

    # Fraction of days with thin pulses
    PEAK_DAYS_PER_MONTH = 12

    # Standard deviation of time gap between consecutive thin pulses
    PEAK_FACTOR_STD = 13

    # Number of LAP days with thin pulses
    VALID_LAP_DAYS = 14

    # Count of LAPs
    COUNT_LAPS = 15

    # Number of days in the given bill cycle
    NUM_DAYS = 16

    # Number of LAPs with 2 or more thin pulses
    TWO_PEAKS_LAP_COUNT = 17
