"""
Author - Nikhil Singh Chauhan
Date - 15-May-2020
initialization ev params initializes local parameters needed to run ev
"""

from copy import deepcopy


def init_ev_l1_params(ev_config):
    """
    Initializing the parameters required for EV

    Parameters:
        ev_config           (dict)          : EV configuration params

    Returns:
        ev_l1_config        (dict)          : Initialized EV params
    """

    ev_l1_config = deepcopy(ev_config)

    ev_l1_config.update(
        {'night_hours': [21, 8],
         'weekend_days': [7, 1],

         'minimum_energy_per_charge': {
             'NA': 1000,
             'EU': 800
         },

         'detection': {
             'minimum_boxes_count': 25,
             'consistency_window': 14,
             'boundaries': [0, 20, 50, 80],

             'NA_start_box_amplitude': 1000,
             'NA_amplitude_step': 250,
             'EU_start_box_amplitude': 800,
             'EU_amplitude_step': 250,

             'max_box_amplitude': 3500,
             'minimum_duration': 3,
             'min_max_filter_duration': 4,

             'sanity_checks': {
                 'duration_variation': 0.8,
                 'box_energy_variation': 1.0,
                 'within_box_energy_variation': 1.0,
             }
         },

         'features_dict': {
             'start_index_column': 0,
             'end_index_column': 1,
             'start_energy_column': 2,
             'end_energy_column': 3,
             'boxes_duration_column': 4,
             'boxes_areas_column': 5,
             'boxes_energy_std_column': 6,
             'boxes_energy_per_point_column': 7,
             'boxes_median_energy': 8,
             'boxes_minimum_energy': 9,
             'boxes_maximum_energy': 10,
             'night_boolean': 11,
             'weekend_boolean': 12,
             'season_id': 13
         },

         'model_features': {
             'NA': ['sampling_900', 'sampling_1800', 'sampling_3600',
                    'tou_variation', 'seasonal_auc_frac_diff',
                    'seasonal_count_fraction_diff', 'seasonal_energy_fraction_diff',
                    'seasonal_amp_frac_diff', 'seasonal_tou_diff',
                    'average_duration', 'duration_variation', 'boxes_energy_variation',
                    'within_boxes_variation', 'energy_per_hour', 'energy_per_charge',
                    'night_count_fraction', 'day_count_fraction', 'weekend_night_fraction',
                    'weekday_night_fraction', 'weekend_day_fraction',
                    'weekday_day_fraction', 'weekly_count_pro', 'day_consistency_score',
                    'monthly_consistency_score', 'monthly_energy_variation',
                    'correlation_factor'],
             'EU': ['sampling_900', 'sampling_1800', 'sampling_3600',
                    'seasonal_count_fraction_diff', 'seasonal_energy_fraction_diff',
                    'seasonal_auc_frac_diff', 'seasonal_amp_frac_diff',
                    'seasonal_tou_diff', 'tou_variation',
                    'average_duration', 'duration_variation', 'boxes_energy_variation',
                    'within_boxes_variation', 'energy_per_hour', 'energy_per_charge',
                    'night_count_fraction', 'day_count_fraction', 'weekend_night_fraction',
                    'weekday_night_fraction', 'weekend_day_fraction',
                    'weekday_day_fraction', 'day_consistency_score',
                    'monthly_consistency_score', 'monthly_energy_variation',
                    'correlation_factor'],
             'l1': ['tou_variation',
                    'average_duration', 'duration_variation', 'boxes_energy_variation',
                    'within_boxes_variation', 'energy_per_charge',
                    'night_count_fraction',
                    'weekly_count_pro', 'day_consistency_score',
                    'monthly_consistency_score', 'monthly_energy_variation',
                    'correlation_factor', 'num_rel_box', 'fraction_rel', 'day_consistency_score_unrel',
                    'monthly_consistency_score_unrel', 'monthly_energy_variation_unrel']
        },

         'dur_amp_threshold': 0.6,
         'invalid_box_thr': 0.7,
         'minimum_duration_allowed': 3,
         'maximum_duration_allowed': 16,
         'charging_hours': [17, 8]
        },
    )

    return ev_l1_config
