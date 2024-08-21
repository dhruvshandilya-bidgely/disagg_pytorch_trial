"""
Author - Sahana M
Date - 2/3/2021
Initialise config to run seasonal waterheater module
"""


def init_seasonal_wh_config(pilot_id, sampling_rate, uuid):

    """
       Returns all the necessary configurations required for the disagg
       Parameters:
           pilot_id                (int)               : Id identifying the pilot user belongs to
           sampling_rate           (int)               : The frequency at which the data is sampled
           uuid                    (str)               : Unique id associated with the user
       Returns:
           seasonal_wh_config      (dict)              : Dictionary containing all configuration parameters for vacation
       """

    # Initialize the config dictionary

    seasonal_wh_config = dict()

    # Initialize user info in the config

    seasonal_wh_config['user_info'] = {
        'uuid': uuid,
        'pilot_id': pilot_id,
        'sampling_rate': sampling_rate
    }

    # Initialize computation config

    """
    'cleaning_window_size'              : Window size required to clean the input data
    'percentile_filter_value'           : Percentile value to remove noise from the input data while cleaning
    'max_temperature_limit'             : If winter present then the max temperature limit for water heater threshold
    'min_temperature_limit'             : If winter absent then the min temperature limit for water heater threshold
    'scaling_threshold'                 : Scaling threshold to enable estimation on transition days
    's_label_weight'                    : Weight given for seasons label as a bonus for wh potential
    'type1_koppen_class'                : type1_base_pot assigned for type1_koppen_class users
    'type1_base_pot'                    : Base potential percentage value for assigned for type1_koppen_class users
    'type2_base_pot'                    : Base potential percentage value for assigned for other users
    'min_seq_length'                    : Min number of days required to qualify as water heater potential days
    'pot_avg_window'                    : Window size for smoothing the wh potential array
    'min_buffer_days'                   : Minimum buffer days
    'min_amplitude'                     : Minimum amplitude required in an hour to qualify as water heater
    'max_amplitude'                     : Maximum amplitude required in an hour to qualify as water heater
    'min_base_amplitude'                : Minimum base amplitude required in an hour to qualify low consuming WHs
    'padding_days'                      : Padding days on either side of WH potential days to calculate correlation
    'min_wh_days'                       : Minimum WH potential days required to qualify as a potential time band
    'min_wh_percent'                    : Minimum percentage of WH potential days required to qualify as a potential time band
    'correlation_threshold'             : Band correlation threshold
    'combine_hours'                     : The common hours required to combine any 2 bands
    'buffer_amp_perc'                   : Get 80th percentile of energy consumed during buffer days
    'wh_amp_perc'                       : Get 80th percentile of energy consumed during wh potential days
    'energy_diff_perc'                  : Get the 90th percentile of energy difference of buffer & wh days after smoothing
    'amp_range_perc'                    : The low amplitude is 70th percentile of low_range values
    'max_base_energy'                   : Minimum amplitude required in an hour to qualify as water heater
    'detection_thr'                     : Detection threshold for the score obtained
    'sanity_fail_detection_thr'         : Default detection threshold when the sanity check fails
    'max_runs'                          : Max WH runs allowed in a day
    'wrong_days_perc_weight'            : Weight for wrong days feature
    'winter_consistency_weight'         : Weight for consistency in winter feature
    'tb_prob_weight'                    : Weight for Time band probability feature
    'std_diff_weight'                   : Weight for standard deviation feature
    'max_median_weight'                 : Weight for max median difference feature
    'c_weight'                          : Constant value for the scoring equation
    'buffer_amplitude'                  : Amplitude used as buffer while identifying the lower and upper bound amplitude
    'lower_amp_cap'                     : Lower Amplitude cap interms of percentile
    'upper_amp_cap'                     : Upper Amplitude cap interms of percentile
    """

    seasonal_wh_config['config'] = {
        'cleaning_window_size': 5,
        'percentile_filter_value': 90,
        'max_temperature_limit': 64.4,
        'min_temperature_limit': 59.0,
        'scaling_threshold': -7.2,
        's_label_weight': 0.20,
        'type1_koppen_class': ['A', 'B', 'Ch'],
        'type1_base_pot': 0.7,
        'type2_base_pot': 0.5,
        'min_seq_length': 3,
        'pot_avg_window': 5,
        'min_buffer_days': 8,
        'min_amplitude': 400,
        'max_amplitude': 5000,
        'min_base_amplitude': 200,
        'padding_days': 7,
        'min_wh_days': 15,
        'min_wh_percent': 0.05,
        'correlation_threshold': 0.33,
        'combine_hours': 4,
        'buffer_amp_perc': 80,
        'wh_amp_perc': 80,
        'energy_diff_perc': 90,
        'amp_range_perc': 70,
        'max_base_energy': 400,
        'detection_thr': 0.50,
        'sanity_fail_detection_thr': 0.30,
        'max_runs': 3,
        'max_time_div': 4,
        'wrong_days_perc_weight': 2.38,
        'winter_consistency_weight': 2.52,
        'tb_prob_weight': 2.41,
        'std_diff_weight': 0.58,
        'max_median_weight': 0.65,
        'c_weight': 3.71,
        'buffer_amplitude': 100,
        'lower_amp_cap': 20,
        'upper_amp_cap': 80
    }

    seasonal_wh_config['alternate_weather_data_config'] = {
        'days': 30,
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
    Time of interest to search for water heater
    '96'    - Sampling rate
    [16,72] - Corresponds to 4am to 6pm scaled for users with 15 minutes sampling rate
    """

    seasonal_wh_config['moving_index'] = {
        '96': [16, 72],
        '48': [8, 36],
        '24': [4, 18]
    }

    """
    Time bands of interest each of 3 hours to search for water heater
    0       - 0th time band
    [16,28] - Corresponds to 4am to 7am scaled for users with 15 minutes sampling rate covering 3 hours
    """

    seasonal_wh_config['time_zones_div'] = {
        0: [16, 28],
        1: [20, 32],
        2: [24, 36],
        3: [28, 40],
        4: [32, 44],
        5: [36, 48],
        6: [40, 52],
        7: [44, 56],
        8: [48, 60],
        9: [52, 64],
        10: [56, 68],
        11: [60, 72],
    }

    """
    Time bands of interest each of 3 hours to search for water heater
    0       - 0th time band
    [4,7] - Corresponds to 4am to 7am scaled for users with 60 minutes sampling rate covering 3 hours
    """

    seasonal_wh_config['time_zones_hour_div'] = {
        0: [4, 7],
        1: [5, 8],
        2: [6, 9],
        3: [7, 10],
        4: [8, 11],
        5: [9, 12],
        6: [10, 13],
        7: [11, 14],
        8: [12, 15],
        9: [13, 16],
        10: [14, 17],
        11: [15, 18],
    }

    """
    WH presence Probability assigned to each time band
    0       - 0th time band
    0.84    - Corresponds to the water heater presence probability assigned to the 0th time band based on exploration
    """

    seasonal_wh_config['tb_prob'] = {
        0: 0.84,
        1: 0.80,
        2: 0.78,
        3: 0.81,
        4: 0.72,
        5: 0.57,
        6: 0.38,
        7: 0.30,
        8: 0.32,
        9: 0.29,
        10: 0.24,
        11: 0.09,
    }

    return seasonal_wh_config
