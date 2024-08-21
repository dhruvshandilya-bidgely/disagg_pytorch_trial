"""
Author: Neelabh Goyal
Date:   14-June-2023
Called to fetch SMB Work hour related constant parameters
"""


def get_hourglass_params():
    """
    Function returns the static parameters used in hourglass only
    Returns:
        static_param (dict) : Dictionary containing all the static parameters
    """

    hourglass_params = {
        'morning_start_sample': 0,
        'morning_end_sample': 10,

        'evening_start_sample': 16,
        'evening_end_sample': 23,

        'cluster_diff_thresh_max': 200,
        'cluster_diff_thresh_min': 50,

        'less_days_thresh': 90,
        'less_days': {
            'edge_corr': 0.95,
            'sun_edge_corr': 0.8
        },

        'min_days': 20,
        'min_valid_days_frac': 0.2,
        'window_size': 3,

        'generic_thresh': {
            'edge_corr': 0.75,
            'sun_edge_corr': 0.65,
        },

        'special_thresh': {
            'sum_sun_edge_corr': 1.5,
            'sun_edge_corr': 0.85,
        },

        'temperature_params': {
            'min_valid_days': 50,
            'min_edge_corr': 0.5,
            'min_temp_edge_corr': 0.75
        },

        'lighting_estimation': {
            'daily_ao_hour': 3,
            'max_allowed_buffer': 4,
            'min_residue_frac': 0.1,
            'min_estimation_percentile': 65,
            'max_estimation_percentile': 97,
            'min_lighting_estimate': 120
        }

    }

    return hourglass_params
