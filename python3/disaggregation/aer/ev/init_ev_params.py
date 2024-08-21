"""
Author - Nikhil Singh Chauhan
Date - 15-May-2020
initialization ev params initializes local parameters needed to run ev
"""

# Import python packages

import pytz
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_timezone(timezone_string):
    """
    Parameters:
        timezone_string     (int)               : Timezone of the user

    Returns:
        timezone            (int)               : Timezone of the user in hours wrt GMT
    """

    # Get the timezone offset number from string

    timezone_offset = datetime.now(pytz.timezone(timezone_string))

    # Convert offset to number of hours

    timezone = timezone_offset.utcoffset().total_seconds() / Cgbdisagg.SEC_IN_HOUR

    return timezone


def init_ev_params(disagg_input_object, global_config):
    """
    Initializing the parameters required for EV

    Parameters:
        disagg_input_object     (dict)      : Dictionary containing all inputs
        global_config           (dict)      : Global configuration params

    Returns:
        config                  (dict)      : Initialized EV params
    """

    # Parameter definitions

    # Initialize EV Parameters as needed

    config = {
        # Generic parameters to be used throughout EV module
        'uuid': global_config.get('uuid'),
        'sampling_rate': global_config.get('sampling_rate'),
        'disagg_mode': global_config.get('disagg_mode'),
        'pilot_id': disagg_input_object['config']['pilot_id'],
        'time_zone': disagg_input_object.get('home_meta_data').get('timezone'),
        "timezone_hours": get_timezone(disagg_input_object.get('home_meta_data').get('timezone')),
        'country': disagg_input_object.get('home_meta_data').get('country'),
        'min_sampling_rate': 900,
        'min_num_days': 180,
        'recent_days': 120,

        # Pilots where estimation boxes have to be refined aggressively to remove doubtful boxes. Currently active only
        # for duke SC

        'est_boxes_refine_pilots': ['10013'],

        # Percentile used to cap input data
        'max_energy_percentile': 99.75,

        # Percentile used for numpy percentile function
        'rolling_percentile': 95,

        # Indices of 2 additional columns added to 21-column matrix
        'box_data_month_col': Cgbdisagg.INPUT_DIMENSION,
        'box_data_season_col': Cgbdisagg.INPUT_DIMENSION + 1,

        # Confidence threshold for removing pool pump algorithm output before EV module
        'pp_removal_conf': 0.7,

        # Column definition of features calculated for ev boxes
        'box_features_dict': {
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
            'boxes_start_hod': 11,

            # box start month is the index of the calendar month in which the start of the EV box lies
            'boxes_start_month': 12,

            # box start month is the index of the season in which the start of the EV box lies
            'boxes_start_season': 13,

            # box start day is the timestamp of the day in which the start of the EV box lies
            'box_start_day': 14,

            # temporarily added columns for feature calculation
            'night_boolean': 15,
            'weekend_boolean': 16,
            'season_id': 17,
        },

        #
        'start_month_count_frac_ratio': 0.5,

        # Temperature cutoff for dividing summer from winter
        'season_cutoff_temp': 65,
        'season_ids': {
            'wtr': 1,
            'smr': 2,
        },

        # Season_count_frac_thresh  is the maximum allowed fraction of boxes in dominant season
        'season_count_frac_thresh': 0.95,
        'season_count_frac_thresh_recent': 0.90,

        # Min number boxes in each season required to calculate seasonal features
        'min_seasonal_boxes_count': 2,

        # Parameters required for pro-rating weekly box count
        'weekly_count_boundaries': [2, 4, 7],
        'weekly_count_pro_values': [1, 2, 3],

        'minimum_duration': {
            3600: 2,
            1800: 1.5,
            900: 1.25,
        },

        # Window used for baseload removal
        'baseload_window': 24,

        'night_hours': [17, 7],
        'weekend_days': [7, 1],

        'minimum_energy_per_charge': {
            'NA': 8000,
            'EU': 6000
        },

        # Parameters used in EV detection module
        'detection': {
            'minimum_boxes_count': 20,
            'consistency_window': 14,
            'boundaries': [0, 20, 50, 80],

            'NA_start_box_amplitude': 3500,
            'NA_amplitude_step': 250,
            'NA_max_amplitude': 50000,
            'EU_start_box_amplitude': 2000,
            'EU_amplitude_step': 250,
            'EU_max_amplitude': 50000,

            'monthly_consistency_window': 3,
            'xgb_model_nan_replace_val': 10,

            'detection_conf_thresh': 0.5,
            'recent_detection_conf_thresh': 0.25,

            'sanity_checks': {
                'duration_variation': 3,
                'box_energy_variation': 0.25,
                'within_box_energy_variation': 0.15,
            }
        },

        # Features of models for EV detection
        'detection_model_features': {
            'xgb': {'NA': ['seasonal_count_fraction_diff', 'seasonal_energy_fraction_diff',
                           'duration_variation', 'boxes_energy_variation',
                           'within_boxes_variation', 'energy_per_charge', 'energy_per_hour',
                           'night_count_fraction', 'weekend_night_fraction',
                           'weekday_night_fraction', 'weekly_count_pro', 'day_consistency_score',
                           'monthly_consistency_score', 'monthly_energy_variation',
                           'correlation_factor'],
                    'EU': ['seasonal_count_fraction_diff', 'seasonal_energy_fraction_diff',
                           'duration_variation', 'boxes_energy_variation', 'average_duration',
                           'within_boxes_variation', 'energy_per_charge', 'energy_per_hour',
                           'night_count_fraction', 'weekend_night_fraction',
                           'weekday_night_fraction', 'weekly_count_pro', 'day_consistency_score',
                           'monthly_consistency_score', 'monthly_energy_variation',
                           'correlation_factor']},

            'rf': {'NA': ['seasonal_count_fraction_diff', 'seasonal_energy_fraction_diff',
                          'duration_variation', 'boxes_energy_variation',
                          'within_boxes_variation', 'energy_per_charge', 'energy_per_hour',
                          'night_count_fraction', 'weekend_night_fraction',
                          'weekday_night_fraction', 'weekly_count_pro', 'day_consistency_score',
                          'monthly_consistency_score', 'monthly_energy_variation',
                          'correlation_factor'],
                   'EU': ['seasonal_count_fraction_diff', 'seasonal_energy_fraction_diff',
                          'average_duration', 'duration_variation', 'boxes_energy_variation',
                          'within_boxes_variation', 'energy_per_hour', 'energy_per_charge',
                          'night_count_fraction', 'weekend_night_fraction',
                          'weekday_night_fraction', 'weekly_count_pro', 'day_consistency_score',
                          'monthly_consistency_score', 'monthly_energy_variation',
                          'correlation_factor']}
        },

        # Parameters required for detection post-processing
        'detection_post_processing': {
            'day_time_start': 10,
            'day_time_end': 22,
            'night_time_start': 22,
            'night_time_end': 10,
            'start_month_count_frac_ratio': 0.5,
            'min_days_in_month': 15,
            'min_num_hvac_hrs': 2,

            # post processing
            'monthly_count_var_thresh': 1.0,
            'max_probability_post_process': 0.75,
            'max_transition_frac_hvac': 0.2,
            'charging_freq_thresh': 0.95,
            'charges_per_day_thresh': 1.4,
            'min_charging_percent': 3,

            # timed activity detection
            'max_start_end_frac_diff': 0.15,
            'prevalence_factor_thresh': 0.65,

            # New probability
            'new_probability': 0.4,

            'wtr_idx': 0,
            'transition_idx': 1,
            'smr_idx': 2,
        },

        # Parameters used in EV estimation module
        'estimation': {
            'neighborhood_points_hrs': 1.5,
            'final_boxes_amp_ratio': 0.25,
            'energy_gmm_percentile': 90,
            'duration_gmm_percentile': 20,
            'clean_boxes_auc_percentile': 30,
            'clean_boxes_std_percentile': 30,
            'max_clean_boxes': 30,
            'min_clean_boxes': 3,
            'na_lower_amp_ratio': 0.7,
            'eu_lower_amp_ratio': 0.8,
            'edge_energy_ratio': 0.4
        },

        # Parameters required in estimation post-processing file
        'estimation_post_processing': {
            'day_time_start': 9,
            'day_time_end': 21,
            'night_time_start': 21,
            'night_time_end': 9,
            'lower_auc_ratio': 0.6,
            'lower_amp_ratio': 0.85,
            'auc_ratio': 0.8,
            'amp_ratio': 0.9,
            'tou_diff_thresh': 4,
            'smr': 2,
            'wtr': 0,

            # Multi-charge parameters
            'tou_weight': 1.5,
            'auc_weight': 0.75,
            'amp_weight': 0.75,
            'amp_diff_frac_allowed': 0.1,
        },

        # Run EU model if detection fails for Duke & Avista users
        'na_to_eu_model': {

            # Pilots which require EU model to be run after NA detection is 0
            'na_to_eu_model_pilots': [10010, 10011, 10012, 10013, 10014, 10015, 10027, 10046, 10048],

            # Minimum energy_per_charge*weekly_count_pro value to run EU model
            'energy_weekly_count_min': 3000,

            # Minimum energy_per_charge*weekly_count_pro value to run EU model
            'energy_weekly_count_max': 15000,

            # NA model probability weight
            'na_probability_weight': 0.3,

            # EU model probability weight
            'eu_probability_weight': 0.7,

            # Confidence threshold
            'na_confidence_thr': 0.35
        },

        # recent ev days proportion
        'recent_ev_days_proportion': 0.6,

        # threshold for seasonal variation along with multiple charging
        'seasonal_var_thr': 0.05,

        # threshold for seasonal EV proportion variation
        'seasonal_proportion_var_thr': 0.0025,

        # threshold for seasonal EV proportion variation for a single season dominant
        'seasonal_proportion_var_check_2_thr': 0.020,

        # confidence threshold for seasonality check
        'hld_check_confidence_thr': 0.55,

        # prior_recent_ev_proportion
        'prior_recent_ev_proportion': 0.01,

        # Year round ev presence threshold
        'year_round_ev_prob_thr': 0.9,

        # New probability value
        'new_seasonal_fp_probability': 0.3,

        # Minimum number of day in a season interms of percentage
        'min_season_per_thr': 0.07,

        # Previous HSM min probability required to update the new ev confidence
        'prev_prob_thr': 0.5,

        # seasonal_count_fraction_diff value for case 2
        'feature_1_case_2_value':  0.005,

        # seasonal_energy_fraction_diff value for case 2
        'feature_2_case_2_value': 0.005,

        # seasonal_count_fraction_diff value for case 1, 3
        'feature_1_case_1_3_value': 0.55,

        # seasonal_energy_fraction_diff value for case 1, 3
        'feature_2_case_1_3_value': 0.55,

        'data_sanity_configs': {

            # Max EV amplitude allowed
            'ev_max_amp_thr': 100000,

            # Max EV duration allowed
            'ev_max_duration_thr': 22,

            # Max EV AUC allowed
            'ev_max_auc_thr': 500000,

        },


        # HVAC removal using potentials parameters

        'hvac_removal_configs': {
            'l2': {
                'cooling_corr_thr': 0.7,
                'heavy_cooling_percentage_thr': 65,
                'heating_corr_thr': 0.7,
                'min_potential_value': 0.8,
                'heavy_heating_percentage_thr': 65,
            },
            'l1': {
                'cooling_corr_thr': 0.9,
                'heavy_cooling_percentage_thr': 75,
                'heating_corr_thr': 0.9,
                'min_potential_value': 0.8,
                'heavy_heating_percentage_thr': 75,
            },

            'min_sh_amp': 2000,
            'min_ac_amp': 1000,
            'min_duration': 1.5,

            'heating': {
                'total_hvac_days': 0.5,
                'hvac_duration': 24,
                'min_mean_hvac_cons': 0.8,
                'min_hvac_duration': 10,
                'min_base_hvac_cons_thr': 0.25,
            },

            'cooling': {
                'total_hvac_days': 0.5,
                'hvac_duration': 24,
                'min_mean_hvac_cons': 0.8,
                'min_hvac_duration': 12,
                'min_base_hvac_cons_thr': 0.25,
            },

            'min_hvac_days': 0.8,
            'cosine_similarity_thr': 0.70,
            'min_season_days': 0.7,
        },

        'nsp_winter_seasonality_configs': {
            'winter_proportion_thr': 0.95,
            'wtr_seasonal_pilots': [40003]

        }

    }

    pilot_string = str(global_config['pilot_id'])

    eu_pilots = ['20001', '20002', '20003', '20004', '20005', '20006', '20007', '20008', '20009', '20010', '20012',
                 '20013', '20015', '20017', '5056']

    fp_seasonal_pilots = ['10046', '10067']

    region = 'EU' if pilot_string in eu_pilots else 'NA'

    config['region'] = region

    # config changed for est_boxes_refine_pilots
    if pilot_string in config.get('est_boxes_refine_pilots'):
        config['estimation']['na_lower_amp_ratio'] = 0.8
        config['estimation']['eu_lower_amp_ratio'] = 0.8
        config['estimation_post_processing']['lower_auc_ratio'] = 0.7
        config['estimation_post_processing']['lower_amp_ratio'] = 0.8
        config['est_boxes_refine_config'] = {
            'low_amp_ratio': 0.8,
            'hi_dur_ratio': 1.75,
            'hi_dur_hour_thresh': 5
        }

    if pilot_string in fp_seasonal_pilots:
        config['seasonal_proportion_var_thr'] = 0.001

    return config
