"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains configs for cycling based inefficiencies
"""


def get_cycling_based_ineff_config(input_hvac_inefficiency_object, device):

    """
        This function estimates cycling based inefficiency

        Parameters:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            device                              (str)           string indicating device, either AC or SH
        Returns:
            config                              (dict)          dictionary containing configs

    """

    days_of_data = input_hvac_inefficiency_object.get('days_of_data')
    sampling_rate = input_hvac_inefficiency_object.get('sampling_rate')

    config = dict({})

    if sampling_rate == 900:
        if device == 'ac':
            largest_duty_cycle_window_length = 6
        else:
            largest_duty_cycle_window_length = 3

        config = {'cluster_merge_limit': 150,
                  'lower_center_limit': 300,
                  'medium_center_limit': 450,
                  'largest_duty_cycle_window_length': largest_duty_cycle_window_length,
                  'scaled_duty_cycle_window': 12,
                  'low_ac_temperature': 75,
                  'high_sh_temperature': 70,
                  'pre_sat_min_duty_cycle': 0.85,
                  'pre_sat_average_duty_cycle': 0.90,
                  'sat_min_duty_cycle': 0.85,
                  'sat_average_duty_cycle': 0.90,
                  'min_duty_cycle_short_cycling': 0.8,
                  'perc_min_data_for_relation_curve': 0.25,
                  'compressor_cut_off_percentile_3': 25,
                  'compressor_cut_off_percentile_2': 75,
                  'compressor_cut_off_percentile_no_comp': 90,
                  'upper_value_hvac_consumption_percentile':  95,
                  'days_of_data': days_of_data,
                  'max_continuous_off_time': 6,
                  'min_representing_percentile': 15}

    elif sampling_rate == 1800:
        largest_duty_cycle_window_length = 6
        config = {'cluster_merge_limit': 150,
                  'lower_center_limit': 300,
                  'medium_center_limit': 450,
                  'largest_duty_cycle_window_length': largest_duty_cycle_window_length,
                  'scaled_duty_cycle_window': 6,
                  'low_ac_temperature': 75,
                  'high_sh_temperature': 70,
                  'pre_sat_min_duty_cycle': 0.85,
                  'pre_sat_average_duty_cycle': 0.90,
                  'sat_min_duty_cycle': 0.85,
                  'sat_average_duty_cycle': 0.90,
                  'min_duty_cycle_short_cycling': 0.8,
                  'perc_min_data_for_relation_curve': 0.25,
                  'compressor_cut_off_percentile_3': 25,
                  'compressor_cut_off_percentile_2': 75,
                  'compressor_cut_off_percentile_no_comp': 90,
                  'upper_value_hvac_consumption_percentile': 95,
                  'days_of_data': days_of_data,
                  'max_continuous_off_time': 3,
                  'min_representing_percentile': 15}

    elif sampling_rate == 3600:
        largest_duty_cycle_window_length = 4
        config = {'cluster_merge_limit': 150,
                  'lower_center_limit': 300,
                  'medium_center_limit': 450,
                  'largest_duty_cycle_window_length': largest_duty_cycle_window_length,
                  'scaled_duty_cycle_window': 3,
                  'low_ac_temperature': 75,
                  'high_sh_temperature': 70,
                  'pre_sat_min_duty_cycle': 0.85,
                  'pre_sat_average_duty_cycle': 0.90,
                  'sat_min_duty_cycle': 0.85,
                  'sat_average_duty_cycle': 0.90,
                  'min_duty_cycle_short_cycling': 0.8,
                  'perc_min_data_for_relation_curve': 0.25,
                  'compressor_cut_off_percentile_3': 25,
                  'compressor_cut_off_percentile_2': 75,
                  'compressor_cut_off_percentile_no_comp': 90,
                  'upper_value_hvac_consumption_percentile': 95,
                  'days_of_data': days_of_data,
                  'max_continuous_off_time': 1,
                  'min_representing_percentile': 15}

    return config


def get_short_cycling_config(sampling_rate):

    """
        This function estimates config for short cycling

        Parameters:
            sampling_rate       (int)          sampling rate of the user
        Returns:
            config              (dict)         dictionary containing config

    """

    config = dict({})

    if sampling_rate == 900:
        config = {'short_cycling_duration_limit': 8,
                  'max_zeros_allowed': 2}
    elif (sampling_rate == 1800) | (sampling_rate == 3600):
        config = {'short_cycling_duration_limit': 4,
                  'max_zeros_allowed': 1}

    return config


def get_clustering_config():

    """
        This function estimates config for clustering module

        Parameters:
            None
        Returns:
            config              (dict)         dictionary containing config

    """
    config = {'primary_optimum_cluster_count': 2,
              'secondary_optimum_cluster_count': 3,
              'lowest_cluster_cons_limit': 0.90,
              'max_lowest_cons_limit': 380,
              'min_fraction_per_cluster': 0.05,
              'high_cluster_portion': 0.40,
              'deviation_factor': 1.5}

    return config
