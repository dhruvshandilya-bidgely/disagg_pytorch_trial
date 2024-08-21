"""
Author - Nisha Agarwal
Date - 3rd Jan 21
This file contain config values for calculating activity sequences
"""


def init_activity_sequences_config():

    """
    Initialize config used for activity seq calculation

    Returns:
        config                   (dict)         : active hours config dict
    """

    config = dict()

    # alpha - used for exponential averaging of activity curve

    smooth_curve_config = {
        'alpha': 0.8
    }

    config.update({
        "smooth_curve_config": smooth_curve_config
    })

    # '30_min_weights': score weightages for sampling rate greater than 15 min,
    # 'non_30_min_weights': score weightages for sampling rate <= 15 min,
    # 'min_threshold': parameters used for calculation of threshold value for separating inc, dec, and constant seqs
    # 'threshold_multiplier':
    # 'higher_samples_per_hour_limit':
    # 'samples_per_hour_limit': post processing steps for sampling rate < 60 min,

    seq_config = {
        '30_min_weights': [0.7, 0.25, 0.05],
        'non_30_min_weights': [0.75, 0.24, 0.01],
        'min_threshold': 0.00015,
        'threshold_multiplier': 0.5,
        'samples_per_hour_limit': 1,
        'higher_samples_per_hour_limit': 4
    }

    config.update({
        "seq_config": seq_config
    })

    # 'derivative_threshold': activity curve derivative threshold ,
    # 'min_threshold': buffer allowed for increasing/decreasing labeling using derivative ,
    # 'max_min_diff_threshold': activity range threshold ,
    # 'max_min_diff_threshold_increament': activity range threshold increment if high activity points are present ,
    # 'diff_multiplier': activity curve range multiplier ,
    # 'high_cons_fraction', 'high_cons_diff_threshold': Thresholds used to identify whether high
    # activity points are present, this will further update threshold for calculating slow increasing activity

    slow_change_detection_config = {
        'derivative_threshold': 0.08,
        'min_threshold': 0.001,
        'max_min_diff_threshold': 0.02,
        'max_min_diff_threshold_increament': 0.01,
        'diff_multiplier': 0.5,
        'high_cons_fraction': 0.6,
        'high_cons_diff_threshold': 0.3
    }

    config.update({
        "slow_change_detection_config": slow_change_detection_config
    })

    merge_seq_config = {
        'threshold': 0.05
    }

    config.update({
        "merge_seq_config": merge_seq_config
    })

    # "activity_curve_diff_limit": minimum activity curve range,
    # "threshold": min score for classifying seq as non constant
    # "length_limit": max length for post processing sequences ,
    # "derivative_threshold": threshold for net derivative ,
    # "weightage": weightage for calculation of score

    remove_small_seq_config = {
        "activity_curve_diff_limit": 0.1,
        "threshold": [0.35, 0.47],
        "length_limit": 5,
        "derivative_threshold": 0.25,
        "weightage": [0.1, 0.3, 0.6]
    }

    config.update({
        "remove_small_seq_config": remove_small_seq_config
    })

    # 'threshold_array': minimum derivative threshold using different activity curve range,
    # 'window': window for neighbourhood seq comparison,
    # 'length_factor': max hour length for neighbourhood comparison ,
    # 'zigzag_pattern_limit': length threshold for zigzag pattern ,
    # 'zigzag_threshold_increament': threshold for calculating zigzag score

    neighbour_seq_config = {
        'threshold_array': [0.04, 0.06, 0.06, 0.06, 0.07, 0.09, 0.1, 0.11, 0.11, 0.12, 0.12, 0.12],
        'window': 3,
        'length_factor':0.5,
        'zigzag_pattern_limit': 2,
        'zigzag_threshold_increament': 0.02
    }

    config.update({
        "neighbour_seq_config": neighbour_seq_config
    })

    return config
