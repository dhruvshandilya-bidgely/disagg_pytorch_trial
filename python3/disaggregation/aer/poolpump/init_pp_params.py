"""
Author - Mayank Sharan
Date - 11/1/19
Initialize the parameters for running vacation
"""

# Import python packages

import numpy as np


def init_pp_params():

    """
    Parameters:
    Returns:
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump
    """

    pp_config = dict({
        'min_days_to_run_pp': 150,
        'baseload_window': 7,
        'filtering_pad_rows': 11,
        'filtering_2_pad_rows': 5,
        'num_days_from_min_max_2': 35,
        'filtering_pad_cols_15_min': 5,
        'filtering_pad_cols_others': 3,
        'prewitt_operator': np.array([[1/3, -1/3], [1/3, -1/3], [1/3, -1/3]]),
        'zero_val_limit': 200,
        'zero_val_limit_raw': 400,
        'consistency_window_size_nms': 20,
        'smart_union_max_val': 5000,
        'smart_union_history_window': 70,
        'smart_union_window_size': 35,
        'smart_union_sliding_window': 10,
        'median_threshold_fraction': 0.3,
        'minimum_median_threshold': 0.6,
        'smart_union_passing_window_size': 30,
        'smart_union_first_loose_passing_window': 25,
        'smart_union_second_loose_passing_window': 20,
        'smart_union_lower_threshold_fraction': 0.3,
        'smart_union_upper_threshold_fraction': 0.6,
        'cleaning_minimum_sufficient_length': 25,
        'cleaning_minimum_distance_from_edge': 5,
        'cleaning_minimum_distance_for_consistency': 100,
        'cleaning_consistency_window_size': 40,
        'cleaning_invalid_distance': 500,
        'cleaning_small_edge_length': 7,
        'cleaning_consistency_margin': 10,
        'cleaning_amplitude_fraction': 0.9,
        'cleaning_best_time_div_num': 2,
        'cleaning_minimum_unmasked_length': 15,
        'cleaning_minimum_edge_length': 30,
        'minimum_day_signal_fraction': 0.8,
        'minimum_match_signal_fraction': 0.5,
        'cleaning_passing_number': 30,
        'window_size': 30,
        'window_step_size': 30,
        'weak_pair_length': 20,
        'strong_pair_length': 90,
        'probability_threshold': 0.45,
        'min_days_for_multiple_run': 70,
        'min_max_run_days': 30,
        'non_winter_run_threshold':0.7,
        'amplitude_margin': 0.9,
        'relaxed_amp_margin': 0.7,
        'moderate_amp_margin': 0.8,
        'minimum_duration_fraction': 0.5,
        'duration_one_hr': 3,
        'min_duration': 1,
        'max_duration': 18,
        'min_amp_ratio_higher': 0.7,
        'min_pair_length_lower': 10,
        'min_pair_length_higher': 30,
        'default_min_amp_ratio': 0.5,
        'min_amp_ratio_lower': 0.3645,
        'minimum_area_under_curve': 800,
        'amp_ratio_reduction_factor': 0.9,
    })

    return pp_config
