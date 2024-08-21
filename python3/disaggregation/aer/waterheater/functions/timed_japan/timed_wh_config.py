"""
Author - Sahana M
Date - 20/07/2021
The file contains the configurations for timed wh
"""

# Import packages within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants

# Season information column mapping dictionary

avg_data_indexes = {
    'temperature': 0,
    'start_day': 1,
    'end_day': 2,
    'num_of_days': 3,
    's_label': 4
}

# Boxes information while dynamic box fitting

box_indexes = {
    'start_row': 0,
    'start_col': 1,
    'end_row': 2,
    'end_col': 3,
    'amplitude': 4,
    'window_size': 5
}


# Indexes mapping for the chunks

chunk_indexes = {
    'overall_index': 0,
    'window_start': 1,
    'chunk_id': 2,
    'chunk_start': 3,
    'chunk_end': 4,
    'consistency_idx': 5,
    'amplitude_idx': 6,
    'duration_idx': 7,
    'chunk_score': 8,
    'label': 9,
    'previous_chunk_idx': 10,
    'merged_seq': 11
}

# Sequence continuity types

seq_type = {
    'independent': 0,
    'continued': 1,
    'rescheduled': 2
}

# Sequence array indexes

seq_arr_idx = {
    's_label': 0,
    'start_day': 1,
    'end_day': 2,
    'total_days': 3,
    'auc': 4,
    'dur': 5,
    'amp': 6,
    'penalty': 7,
    'gap_days': 8,
    'penalty_config': 9
}

# Box fitting windows

windows = {
    1: [2, 3, 4],
    2: [2, 3, 4, 5, 6, 7, 8],
    4: [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
}


def init_twh_config(sampling_rate, debug):
    """
    This function is used to initialise timed wh configurations for Japan pilots
    Parameters:
        sampling_rate           (float)     : Sampling rate
        debug                   (dict)      : Debug dictionary
    Returns:
        twh_config              (dict)      : Timed wh configs
    """

    factor = Cgbdisagg.SEC_IN_HOUR/sampling_rate

    """
    min_amp                 : Minimum amplitude criteria (Wh)
    factor                  : Data granularity
    cols                    : Number of time divisions for a day
    min_amp_bar             : Minimum amplitude bar
    max_amp_bar             : Maximum amplitude bar
    min_consistent_thr      : Minimum consistency threshold
    base_consistent_thr     : Base consistency threshold
    amp_bar_filter          : Heating instance minimum amplitude bar
    amp_bar_twh             : Amplitude bar for timed wh
    heating_min_amp         : Minimum amplitude for a non heating instance
    heating_consistency_thr : Heating instance consistency threshold (since heating instance is generally timed)
    heating_percentage_days : Percentage of heating days
    heating_inst_twh_thr    : Heating instance consistency threshold for original matrix
    days                    : Number of input days
    sliding_window          : Sliding window to get chunks for every 15 days
    vertical_percentile     : Vertical percentile filter for cleaning the data
    horizontal_percentile   : Horizontal percentile filter for cleaning the data
    vertical_window_size    : Window size for vertical filtering
    horizontal_window_size  : Window size for horizontal filtering
    min_consistency         : Minimum consistency for chunk formation
    single_point_consistency: Minimum consistency to merge a single empty time frame into a chunk
    exploration_hour_cap    : Threshold for long duration heating instance after which no interpolation is needed
    min_amplitude_percentile: Minimum amplitude percentile
    max_amplitude_percentile: Maximum amplitude percentile
    base_num_chunks_allowed : Number of max chunks allowed, after which removal of unwanted chunks happens
    max_num_chunks_allowed  : Max number of chunks allowed, after which we keep top 5 chunks
    edge_amp                : Edge amplitude of all the chunks combined amplitude
    centered_amp            : Centered amplitude of all the chunks combined amplitude
    good_chunk_probability  : Ideal chunk probability
    default_independent_score : Default score for an independent chunk
    default_connectivity_score : Default score for connection between 2 chunks
    connectivity_score_weight  : Connectivity score weight
    amplitude_score_weight  : Amplitude score weight
    overall_score_weight    : Overall score weight
    td_score_weight         : Time deviation score weight
    previous_score_weight   : Previous chunk score weight
    current_score_weight_1  : Current chunk score weight 1
    current_score_weight_2  : Current chunk score weight 2
    consistency_score_weight: Consistency score weight
    base_reward             : Base reward for continuity
    propagating_reward      : Reward for propogation
    seq_centered_amp        : Centre amplitude for the sequence minimum threshold
    seq_centered_max_time   : Centre max time
    seq_overlap_thr         : Sequence overlap threshold
    seq_amp_weight          : Sequence amplitude weight
    seq_td_weight           : Sequence time deviation weight
    seq_dur_weight          : Sequence duration weight
    seq_min_weight          : Sequence minimum weight
    seq_max_weight          : Sequence maximum weight
    seq_overall_score_thr   : Sequence overall score threshold to assign it to a label
    seq_amp_score_thr       : Sequence amplitude score threshold
    seq_dur_score_thr       : Sequence duration score threshold
    seq_td_score_thr        : Sequence time deviation score threshold
    min_chunk_score         : Minimum chunk score
    chunk_gap_penalty       : Penalty for any gaps in the chunk continuity
    auc_min_amp_criteria    : AUC minimum criteria
    auc_max_amp_criteria    : AUC maximum criteria
    box_min_amp             : Box minimum amplitude threshold
    box_fit_amp_weight      : Box fitting amplitude weight
    box_fit_dur_weight      : Box fitting duration weight
    day_range               : Days to look before and after while box fitting
    neighbour_days_amp_thr  : Neighbouring days amplitude threshold
    max_duration            : Maximum duration allowed for timed wh otherwise penalty inflicted
    long_duration_penalty   : Long duration penalty weight
    abnormal_check_amp      : Amplitude to check for abnormal amplitude variation
    av_thr                  : Abnormal variation threshold
    diff_ratio_thr          : Difference ratio from original to timed wh estimation threshold
    av_penalty              : Abonormal variation penalty
    max_amplitude_variation : Consistency in the diferential amplitude required
    ema_days                : Days for exponential moving average
    default_rss             : Default reverse seasonality score
    default_oss             : Default One sided seasonality score
    default_dds             : Default double dip score
    hi_time                 : horizontal interpolation time
    re_population_base_amp  : Re-population base amplitude
    previous_confidence_weight: Weight for the previous confidence
    """

    twh_config = {
        'min_amp': 1000,
        'factor': factor,
        'cols': int(Cgbdisagg.HRS_IN_DAY*factor),
        'min_amp_bar': 0.8,
        'max_amp_bar': 1.2,
        'min_consistent_thr': 0.14,
        'base_consistent_thr': 0.1,
        'amp_bar_filter': 5000,
        'amp_bar_twh': 6000,
        'heating_min_amp': 700,
        'heating_consistency_thr': 0.6,
        'heating_percentage_days': 0.7,
        'heating_inst_twh_thr': 0.3,
        'days': 30,
        'sliding_window': 15,
        'vertical_percentile': 30,
        'horizontal_percentile': 10,
        'vertical_window_size': 7,
        'horizontal_window_size': 12,
        'min_consistency': 8,
        'single_point_consistency': 14,
        'exploration_hour_cap': 5,
        'min_amplitude_percentile': 90,
        'max_amplitude_percentile': 95,
        'base_num_chunks_allowed': 4,
        'max_num_chunks_allowed': 5,
        'edge_amp': 2450,
        'centered_amp': 1500,
        'good_chunk_probability': 0.2,
        'default_independent_score': 0.3,
        'default_connectivity_score': 0.5,
        'connectivity_score_weight': 0.4,
        'amplitude_score_weight': 0.6,
        'overall_score_weight': 0.6,
        'td_score_weight': 0.4,
        'previous_score_weight': 0.4,
        'current_score_weight_1': 0.6,
        'current_score_weight_2': 0.8,
        'consistency_score_weight': 0.2,
        'base_reward': 0.3,
        'propagating_reward': 0.75,
        'seq_centered_amp': 400,
        'seq_centered_max_time': 3,
        'seq_overlap_thr': 0.3,
        'seq_amp_weight': 0.5,
        'seq_td_weight': 0.4,
        'seq_dur_weight': 0.1,
        'seq_min_weight': 0.9,
        'seq_max_weight': 1.1,
        'seq_overall_score_thr': 0.4,
        'seq_amp_score_thr': 0.25,
        'seq_dur_score_thr': 0.1,
        'seq_td_score_thr': 0.1,
        'min_chunk_score': 0.35,
        'chunk_gap_penalty': 0.1,
        'auc_min_amp_criteria': 800,
        'auc_max_amp_criteria': 20000,
        'box_min_amp': 700,
        'box_fit_amp_weight': 0.5,
        'box_fit_dur_weight': 0.5,
        'day_range': 6,
        'neighbour_days_amp_thr': 1200,
        'max_duration': 8,
        'long_duration_penalty': 0.3,
        'abnormal_check_amp': 1600,
        'av_thr': 0.05,
        'diff_ratio_thr': 0.60,
        'av_penalty': 0.30,
        'max_amplitude_variation': 1400,
        'ema_days': 10,
        'default_rss': 0.25,
        'default_oss': 0.6,
        'default_dds': 0.2,
        'hi_time': 2,
        're_population_base_amp': 400,
        'previous_confidence_weight': 0.8,
    }

    twh_config['weather_data_configs'] = {
        'min_seq_length': 3,
        's_label_weight': 0.20,
        'pot_avg_window': 5,
        'type1_base_pot': 0.7,
        'type2_base_pot': 0.5,
        'scaling_threshold': -7.2,
        'max_temperature_limit': 64.4,
        'min_temperature_limit': 59.0,
        'type1_koppen_class': ['A', 'B', 'Ch'],
        'type_a_temp_thr': 66.2,
        'type_b_hot_thr': 64.4,
        'type_c_cold_temp_min': 32,
        'type_c_cold_temp_max': 66.2,
        'type_c_hot_temp_min': 50,
        'type_d_cold_temp_max': 32,
        'type_d_hot_temp_min': 50,
        'type_e_max_temp': 50,
        'c_sub_temp_thr': 65,
        'perc_thr': 0.7,
        'winter_set_point': 55.4,
        'summer_set_point': 69.8,
    }

    """
    Penalty for each season for seasonality score
    -1      => Winter
    -0.5    => Transition winter
    0       => Transition
    0.5     => Transition Summer
    1       => Summer
    """

    twh_config['season_penalty'] = {
        -1: -1,
        -0.5: -0.5,
        -0.0: -0.25,
        0: -0.25,
        0.5: -0.25,
        1: -0.35
    }

    """
    HLD check configurations
    """
    twh_config['hld_checks'] = {
        'max_hours_allowed': 8,
        'max_runs_allowed': 3,
        'duration_variation': 3.5,
        'amplitude_variation': 1200,
        'penalty_1': 0.3,
        'penalty_2': 0.4
    }

    """" Based on different sampling rates some configs are different"
    heating_inst_window         : Window for heating instance
    box_min_amp_bar             : Timed wh box minimum amplitude bar
    box_max_amp_bar             : Timed wh box maximum amplitude bar
    merge_seq_min_bar           : Merging sequence min bar
    merge_seq_max_bar           : Merging sequence max bar
    td_min_bar                  : Time deviation minimum bar
    td_max_bar                  : Time deviation maximum bar
    hi_min_bar                  : Heating instance minimum bar
    hi_max_bar                  : Heating instance maximum bar
    """
    if factor == 4:
        twh_config['heating_inst_window'] = 2
        twh_config['box_min_amp_bar'] = 0.8
        twh_config['box_max_amp_bar'] = 1.2
        twh_config['merge_seq_min_bar'] = 0.7
        twh_config['merge_seq_max_bar'] = 1.3
        twh_config['td_min_bar'] = 0.5
        twh_config['td_max_bar'] = 1.5
        twh_config['hi_min_bar'] = 0.8
        twh_config['hi_max_bar'] = 1.2

    elif factor == 2:
        twh_config['heating_inst_window'] = 1
        twh_config['box_min_amp_bar'] = 0.8
        twh_config['box_max_amp_bar'] = 1.2
        twh_config['merge_seq_min_bar'] = 0.7
        twh_config['merge_seq_max_bar'] = 1.3
        twh_config['td_min_bar'] = 0.5
        twh_config['td_max_bar'] = 1.5
        twh_config['hi_min_bar'] = 0.8
        twh_config['hi_max_bar'] = 1.2
    else:
        twh_config['heating_inst_window'] = 1
        twh_config['box_min_amp_bar'] = 0.4
        twh_config['box_max_amp_bar'] = 1.6
        twh_config['merge_seq_min_bar'] = 0.4
        twh_config['merge_seq_max_bar'] = 1.6
        twh_config['td_min_bar'] = 0.3
        twh_config['td_max_bar'] = 1.7
        twh_config['hi_min_bar'] = 0.5
        twh_config['hi_max_bar'] = 1.5

    # Time band probabilities for default and origin pilot

    twh_config['tb_probability_default'] = [0.85, 0.85, 0.9, 0.93, 0.92, 0.9, 0.73, 0.49, 0.41, 0.4, 0.39, 0.4, 0.75,
                                            0.93, 0.88, 0.80, 0.3, 0.25, 0.25, 0.24, 0.24, 0.25, 0.24, 0.25]

    twh_config['hvac_pilots'] = [30001]

    # Detection threshold

    twh_config['detection_thr'] = 0.5

    # User info for easy accessibility

    twh_config['uuid'] = debug.get('uuid')
    twh_config['sampling_rate'] = sampling_rate
    twh_config['pilot_id'] = debug.get('pilot_id')
    twh_config['home_meta_data'] = debug.get('home_meta_data')

    return twh_config
