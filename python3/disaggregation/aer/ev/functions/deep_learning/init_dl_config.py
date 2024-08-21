"""
Author - Sahana M
Date - 14-Nov-2023
Configurations file for Deep learning EV
"""


class FeatureCol:
    """Column names for box features"""
    PRT_PRED = 0
    PRT_CONFIDENCE = 1
    PRT_START_IDX = 2
    PRT_END_IDX = 3
    VACATION_COL = 4
    S_LABEL = 5
    MEAN_BOX_ENERGY = 6
    MIN_BOX_ENERGY = 7
    MAX_BOX_ENERGY = 8
    BOX_AUC = 9
    BOX_DURATION = 10
    TOTAL_EV_STRIKES = 11
    CHARGING_FREQ = 12
    TOTAL_CONSUMPTION = 13


def init_ev_params():
    """Initialising configurations dictionary for deep learning module"""
    config = {
        # min l2 amplitude
        'min_amplitude': 3500,

        # min l2 amplitude for EU region
        'min_amplitude_eu': 2000,

        # max l2 amplitude
        'max_amplitude': 50000,

        # min l1 start amplitude
        'min_amplitude_l1': 1000,

        # max l1 start amplitude
        'max_amplitude_l1': 3500,

        # min duration for l2
        'min_duration_l2': 0.5,

        # min duration for l1
        'min_duration_l1': 3,

        # NA region starting amplitude
        'NA_start_box_amplitude': 1000,

        # NA region step amplitude
        'NA_amplitude_step': 250,

        # EU region starting amplitude
        'EU_start_box_amplitude': 800,

        # EU region step amplitude
        'EU_amplitude_step': 250,

        # Minimum boxes needed
        'minimum_boxes_count': 25,

        # Min-max filter duration
        'min_max_filter_duration': 4,

        # EV charging hours
        'charging_hours': [17, 8],

        # Invalid box threshold
        'invalid_box_thr': 0.7,

        # Duration & Amplitude threshold
        'dur_amp_threshold': 0.6,

        # Minimum duration allowed for l1
        'minimum_duration_allowed': 3,

        # Maximum duration allowed for L1
        'maximum_duration_allowed': 16,

        # Partition detection threshold for L2
        'prt_conf_thr': 0.5,
        # Partition detection threshold for L1

        'prt_conf_thr_l1': 0.8,
        # Single partition size in days

        'prt_size': 14,

        # General Confidence threshold
        'confidence_threshold': 0.5,

        # Minimum partitions to qualify for recent EV
        'recent_ev_partitions': 8,
        # Minimum partitions to be detected for recent EV

        'recent_ev_partitions_thr': 3,


        # Penalty threshold for detection

        'penalty_thr_1': 0.2,
        'penalty_thr_2': 0.1,
        'penalty_thr_3': 0.1,
        'penalty_thr_4': 0.2,
        'penalty_thr_5': 0.2,
        'penalty_thr_6': 0.1,
        'penalty_thr_7': 0.1,
        'penalty_weight': 0.7,
        'penalty_weight_l1': 0.3,
        'seasonality_penalty_weight': 0.7,
        'seasonality_penalty_weight_l1': 0.3,
        'similarity_default': 0.3,

        # Confidence adjustment thresholds

        'r_value_1': 0.95,
        'r_value_2': 0.92,
        'a_value': 1,
        'final_weight_1': 0.8,
        'final_weight_2': 0.2,

        # Box features column names

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

        # Box features column names

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

        # Detection thresholds

        'detection': {
            'minimum_boxes_count': 20,
            'NA_start_box_amplitude': 3500,
            'NA_amplitude_step': 250,
            'NA_max_amplitude': 50000,
            'EU_start_box_amplitude': 2000,
            'EU_amplitude_step': 250,
            'EU_max_amplitude': 50000,

            # Sanity check thresholds

            'sanity_checks': {
                'duration_variation': 3,
                'box_energy_variation': 0.25,
                'within_box_energy_variation': 0.15,
                'l1': {
                    'duration_variation': 0.8,
                    'box_energy_variation': 1.0,
                    'within_box_energy_variation': 1.0,
                }
            }
        },

        # Duration thresholds

        'minimum_duration': {
            3600: 2,
            1800: 1.5,
            900: 1.25,
        },

        # Special case - multimode box fitting configurations

        'multimode_box_fitting': {
            'low_amp_thr': 0.35,
            'high_amp_thr': 0.65,
            'box_distribution_thr': 3,
            'similarity_score_thr': 0.8,
            'overlapping_thr': 0.75,
        },

        # Configurations to pick missing boxes

        'pick_missing_boxes': {
            'overlap_percentage_1': 0.3,
            'overlap_percentage_2': 0.6,
            'low_amp_thr': 0.6,
            'high_amp_thr': 1.4,
        },

        # Configurations to identify and update confidences of non-detected partitions

        'identifying_all_ev_partitions': {
            'min_sequence_size': 4,
            'min_sequence_size_l1': 12,
            'min_amp_thr': 0.75,
            'max_amp_thr': 1.35,
            'similarity_thr': 0.35,
            'amp_similarity_thr': 0.35,
            'prev_overlap_thr': 0.15,
            'amp_curr_thr': 0.15,
            'overall_score_thr': 0.6,
        },

        # Seasonality False positive check configs

        'seasonality_penalty': {
            'season_cols': {
                's_label_row': 0,
                'repetitions_row': 1,
                'avg_cons_row': 2,
                'total_cons_row': 3,
                'percentage_cons_row': 4,
                'season_ev_percentage_row': 5,
                'season_percentage_row': 6,
                'relative_cons_row': 7,
            },
            'ignore_season_percentage': 0.04,
            'ignore_season_percentage_l1': 0.05,
            's_labels': [-1, -0.5, 0, 0.5, 1],
            'seasonality_penalty_weight_l1': 0.5,
            'ev_var_thr': 0.2,
            'ev_prop_penalty': 0.3,
            'seasonality_penalty_score_default': 0.4,
        },

        # L2 charger detection configs

        'l2_detection_configs': {
            'ml_confidence_weight': 9,
            'dl_confidence_weight': 3.91,
            'seasonal_penalty_weight': 0.37,
            'constant': 4.41,
            'seasonal_fps_ml_confidences': [0.3, 0.4],
            'seasonal_fp_confidence': 0.4,
            'low_density_ml_weight': 0.7,
            'low_density_comb_weight': 0.3,
        },

        # L1 charger detection configs

        'l1_detection_configs': {
            'conf_thr_1': 0.3,
            'conf_thr_2': 0.4,
            'conf_thr_3': 0.5,
            'conf_thr_4': 0.6,
            'conf_thr_5': 0.7,
            'low_density_ml_weight': 0.7,
            'low_density_dl_weight': 0.3,
        }

    }
    return config
