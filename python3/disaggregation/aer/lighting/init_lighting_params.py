"""
Author - Mayank Sharan
Date - 26/11/2019
initialisation lighting params initializes local parameters needed to run lighting
"""

# Import python packages

import numpy as np


def frange(x, y, jump):
    """ Float range generator"""
    while x < y:
        yield x
        x += jump


def setpoint_list(start, stop, step=1):
    """  Fill integer np.array  from start to stop by step """
    return np.array(list(range(int(start), int(stop + step), int(step))))


def setpoint_float_list(start, stop, step=1):
    """  Fill float np.array  from start to stop by step """
    return np.array(list(frange(float(start), float(stop + step), float(step))))


class Struct:
    """MATLAB structure"""
    def __init__(self, **entries):
        self.__dict__.update(entries)


def init_lighting_params():
    """ Fill config """

    # Initialize Lighting Parameters as needed

    config = {
        'UUID': '',
        'DURATION_VALID_DISAGG': 270,
        'HORIZONTAL_PERCENTILE': 30,
        'HORIZONTAL_WINDOW': 2,
        'VERTICAL_PERCENTILE': 30,
        'VERTICAL_WINDOW': 30,
        'HISTOGRAM_BIN_SIZE': 2,
        'ZERO_DAY_CONSUMPTION': 50,
        'SMOOTHING_NOISE_LOWER_BOUND': 20,
        'SMOOTHING_NOISE_UPPER_BOUND': 60,
        'SMOOTHING_BOUND_PERCENTILE': 60,
        'NUM_ZEROS_FOR_SEASONAL': 20,
        'TYPICAL_LIGHTING_HOURS': np.array([[6, 10], [16, 24]]),
        'OVERALL_SEASON_PERCENTILE': 60,
        'SEASON_PERCENTAGE_PERCENTILE': 70,
        'DEFAULT_SEASON_PERCENTAGE': 70,
        'SEASON_DISCREPANCY_THRESHOLD': 35,
        'SEASON_PERCENTAGE_BOUND': 95,
        'MIN_NOISE_RATIO': 6,
        'LOW_SEASON_THRESHOLD': 10,
        'TIME_SELECTION_THRESHOLD': 30,
        'LIGHTING_BAND_CAP': 12,
        'TIME_SELECTION_THRESHOLD_CAP': 96,
        'TIME_SELECTION_THRESHOLD_STEP': 2,
        'NON_TYPICAL_LIGHTING_1': np.array([1, 5]),
        'NON_TYPICAL_LIGHTING_2': np.array([12, 16]),
        'CAPACITY_PERCENTILE': 95,
        'CAPACITY_CORRECTION_STEP': 5,
        'LIGHTING_PERCENTILE_DAYS_FOR_CAPACITY_CALCULATION': 30,
        'ZERO_CAPACITY_FIX_STEP': 2,
        'DAY_MAX_PERCENTILE': 95,
        'MORNING_TIME': np.array([3, 15]),
        'CAPACITY_DIFF_THRESHOLD': 200,
        'CAPACITY_MIN_BOUND': 40,
        'CAPACITY_MAX_BOUND': 1200,
        'DEBUG': False,
        'DEBUG_DIR': 'lighting_results',
        'DEBUG_SAVEPLOTS': False,
        'DEBUG_SAVE_DAILYESTIMATE': False,
        'DEBUG_SAVE_DAILYPERCENTAGE': False,
        'DEBUG_SAVE_DAILYHOURS': False,
        'DEBUG_SAVE_LINEPLOTS': False,
        'DEBUG_SAVE_HEATMAP': False,
        'SEASONALITY': {
            'lightMonthScaling': 0.8,
            'minDays': 15,
            'lowerBoundLightMonthScaling': 0.75,
            'buffer': 10,
        }
    }
    return config
