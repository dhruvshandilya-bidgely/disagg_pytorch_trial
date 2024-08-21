"""
Author: Mayank Sharan
Created: 12-Jul-2020
Initialize config to mark seasons
"""

# Import python packages

import numpy as np


def init_mark_season_config():

    """
    Initialize and return config used to mark season
    Returns:
        mark_season_config      (dict)          : Dictionary containing all config used to mark seasons
    """

    # Initialize the config

    mark_season_config = dict({})

    # Populate the config used for creation of average data
    # morn_day_bdry_hr      : The hour at which day starts in the morning (included)
    # eve_day_bdry_hr       : The hour at which day ends in the evening (excluded)
    # smooth_window_size    : Number of days over which rolling average is calculated

    mark_season_config['avg_data'] = {
        'morn_day_bdry_hr': 7,
        'eve_day_bdry_hr': 19,
        'smooth_window_size': 31,
    }

    # Populate the config used for fitting GMM model on the data
    # init_cluster_centres  : Mean temperature (in F) initialisations for GMM clusters
    # num_components        : Number of components in the GMM cluster
    # cov_type              : The covariance type of the GMM model
    # random_state          : The random state to ensure the results of the GMM are reproducible
    # day_night_offset      : Temperature offset used to shift means for day and night clusters

    mark_season_config['gmm_fit'] = {
        'init_cluster_centres': {
            'A': np.array([[50], [65], [80]]),
            'Bh': np.array([[50], [67], [83]]),
            'Bk': np.array([[40], [60], [83]]),
            'Ch': np.array([[60], [70], [82]]),
            'C': np.array([[53], [66], [77]]),
            'Ck': np.array([[45], [62], [72]]),
            'D': np.array([[38], [63], [70]]),
        },
        'num_components': 3,
        'cov_type': 'full',
        'random_state': 1,
        'day_night_offset': 3.6
    }

    # Populate the config used for fitting GMM model on the data
    # min_season_length     : Number of days needed in a cluster to qualify as a main season
    # max_ext_length        : Maximum length of season that can be made longer
    # longer_season_perc    : Extra percentage length a day summer or night winter needs to be added to be marked

    mark_season_config['identify_seasons'] = {
        'min_season_length': 25,
        'max_ext_length': 90,
        'longer_season_perc': 0.4,
    }

    # Populate the config used for marking preliminary season on the data
    # lim_winter_temp_dict  : The dictionary specifying limit to max winter temp
    # lim_tr_temp_dict      : The dictionary specifying limit to max transition temp
    # obv_winter_thr        : Threshold in F to mark winter
    # obv_summer_thr        : Threshold in F to mark summer
    # min_season_length     : Number of days needed in a cluster to qualify as a main season
    # bonus_base_days       : Number of days used as base to provide bonus
    # limit_base_days       : Number of days used as base to modify limit

    mark_season_config['mark_prelim_season'] = {
        'lim_winter_temp_dict': {
            'A': 64,
            'Bh': 64,
            'Bk': 60,
            'Ch': 64,
            'C': 62,
            'Ck': 60,
            'D': 58,
        },
        'lim_tr_temp_dict': {
            'A': 73,
            'Bh': 75,
            'Bk': 68,
            'Ch': 75,
            'C': 72,
            'Ck': 70,
            'D': 68,
        },
        'obv_winter_thr': 53.6,
        'obv_summer_thr': 80.6,
        'min_season_length': 25,
        'bonus_base_days': 30,
        'limit_base_days': 75,
    }

    # Populate the config used for hysteresis smoothening
    # past_period           : The number of days in past to use for smoothening
    # future_period         : The number of days in future to use for smoothening
    # past_weight           : The weight given to past label
    # future_weight         : The weight given to future label

    mark_season_config['hysteresis_smoothening'] = {
        'past_period': 14,
        'future_period': 7,
        'past_weight': 0.6,
        'future_weight': 0.4,
    }

    # Populate the config used for temperature event detection
    # event_window          : Days to sum score to find an event
    # min_ev_score          : Minimum score for qualification as an event

    mark_season_config['temp_event_det'] = {
        'event_window': 3,
        'min_ev_score': 3.5,
    }

    # Populate the config used for event tag modification
    # event_ratio_thr       : Minimum part of chunk to be composed of a type of event to modify label
    # label_offset          : Amount to shift the label by

    mark_season_config['modify_event_tag'] = {
        'event_ratio_thr': 0.3,
        'label_offset': 0.5,
    }

    # Populate the config used for uniting season
    # season_length_base    : Base to use for scoring length of chunk
    # season_temp_base      : Base to use for scoring temperature deviation of chunk
    # length_weight         : Weight given to length score
    # temp_weight           : Weight given to temperature score
    # max_gap_bridge        : Maximum gap between chunks allowed to be merged as a season

    mark_season_config['unite_season'] = {
        'season_length_base': 20,
        'season_temp_base': 14.4,
        'length_weight': 0.6,
        'temp_weight': 0.4,
        'max_gap_bridge': 90,
    }

    return mark_season_config
