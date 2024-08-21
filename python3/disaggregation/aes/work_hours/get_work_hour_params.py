"""
Author: Neelabh Goyal
Date:   14-June-2023
Called to fetch SMB Work hour related constant parameters
"""

# Import python packages
import numpy as np


def get_work_hour_params():
    """
    Function returns the static parameters used in work_hours only
    Returns:
        static_param (dict) : Dictionary containing all the static parameters
    """

    work_hour_params = {

        'cluster_count': {
            'high': 3,
            'mid': 2,
            'low': 2,
            'v_low': 2
        },

        'cluster_thresh': {
            'v_low': 60,
            'low': 120,
            'high': 401
        },

        'data_range': {
            'high': np.arange(15, 86, 10),
            'mid': np.arange(25, 96, 10),
            'low': np.arange(25, 96, 10),
            'v_low': np.arange(25, 96, 10)
        },

        'label_thresh': {
            'high': 2,
            'mid': 3,
            'low': 3,
            'v_low': 3
        },

        'post_process_user_work_hours': {
            'high_percentile_clusters': 3,
            'max_work_hour': 20,
            'max_days_thresh': 5,
            'min_days_thresh': 2,
            'cluster_count': 2
        },

        'daily_clustering_dict': {
            'cluster_count': 2,
            'kmeans_models': [],
            'kmeans_data': [],
            'kmeans_clusters': [],
            'label_idx_start': [],
            'label_idx_end': [],
            'work_hour_perc_cluster': [],
            'final_label_arr': []
        },

        'post_process_epoch_work_hours': {
            'overall_day_thresh': 0.75,
            'non_hvac_day_thresh': 0.65,
            'hvac_day_thresh': 0.5,
            'hvac_days_factor': 2.5,
            'max_hrs_with_75perc_work_hours': 20,
            'max_non_hvac_hrs_with_65perc_work_hours': 20,
            'max_hvac_hrs_with_50perc_work_hours': 18,
            'long_work_hours': {
                'yearly_work_hour_thresh': 18,
                'max_work_hour_lim': 23,
                'max_variation': 1.2,
                'min_long_work_hours': 6
            },
            'continuous_work_hour_window': 4
        },

        'alternate_work_hours': {
            'defined_user_work_band_thresh': 3,
            'defined_user_work_hour_thresh': 8,
            'defined_user_days_frac_thresh': 0.7,
            'max_allowed_hour_block': 12,
            'min_hour_block_for_24x7': 18,
            'max_work_hour_per_day': 8,
            'max_allowed_streaks_per_day': 5
        },

        'post_process_for_ext_light': {
            'morning_hour': 6,
            'evening_hour': 18,
            'min_night_work_hours': 3,
            'morning_hour_end': 12,
            'evening_hour_start': 13,
            'min_valid_days': 10,
            'min_valid_days_frac': 0.6,
            'corr_thresh': 0.7
        },

        'hsm_params': {
            'min_work_hour_frac': 0.25,
            'defined_work_hour_frac': 0.95,
            'max_change': 3,
            'relaxed_24x7_cond': 18,
            'max_0x7_days': 3,
            'max_24x7_days': 7,
            'min_24x7_days': 4,
            'min_24x7_work_hour_frac': 0.4
        },

        'high_cons_thresh_multiplier': 3,

        'min_non_zero_days_perc': 0.05,

        'min_days_thresh': 45,

        'min_cons_thresh_24x7_user': 50,

        'min_daily_cons': 50,

        'min_avg_monthly_cons': 5000,

        'min_work_label_perc': 0.005,

        'max_work_bands': 2,

        'min_hourglass_residue': 200,

        'min_epoch_cons': 5,

        'max_days_for_non_high_cons_kmeans': 15,

        'max_days_for_high_cons_kmeans': 30,

        'max_allowed_gap_hours': 2,

        'overnight_band_night_hour': 22,

        'overnight_band_morning_hour': 3,

        'max_work_hours': 22,

        'overnight_work_hour': 1.5
    }

    return work_hour_params
