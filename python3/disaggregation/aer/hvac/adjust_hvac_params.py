"""
Author - Mirambika
Date - 06/12/2023
Setting HVAC parameters based on consumption and temperature type
"""

# Import python packages
import copy
import numpy as np

# Import packages from within the pipeline
from scipy.stats.mstats import mquantiles
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile

static_params = hvac_static_params()


def get_user_cooling_flags(hvac_input_data, user_parameters, hvac_params, config):
    """
    Function to consolidate all primary and secondary user flags to enable/disable adjustment of parameters
    Args:
        hvac_input_data  (np.ndarray)    : 2D array of epoch level consumption and temperature data
        user_parameters  (dict)          : Dictionary containing all user extracted labels and features
        hvac_params      (dict)          : Dictionary containing hvac algo related initialized parameters
        config           (dict)          : Dictionary with fixed parameters to calculate user characteristics
    Returns:
        result           (dict)          : Dictionary containing all user pipeline related flags
    """

    # Initializing variables and making local copies of input consumption and temperature
    timezone = config['timezone']
    hvac_input_consumption = copy.deepcopy(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    hvac_input_temperature = copy.deepcopy(hvac_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX])

    adjust_midtemp_flag, adjust_ac_detection_range_flag = 0, 0
    adjust_ac_setpoint_flag , adjust_ac_detection_setpoint_flag = 0, 0
    adjust_ac_af_flag, adjust_ac_nclusters_flag, adjust_ac_min_amp_flag = 0, 0, 0
    user_temp_type = user_parameters.get('all_flags').get('hot_cold_normal_user_flag')
    invalid_idx = user_parameters.get('all_indices').get('invalid_idx')

    # Calculating mid-temperature scope index ahead of detection to check for conditions
    mid_temp_scope_idx = np.logical_and(~np.isnan(hvac_input_temperature),
                                        np.logical_and(hvac_input_temperature >=
                                                       hvac_params['detection']['MID_TEMPERATURE_RANGE'][0],
                                                       hvac_input_temperature <=
                                                       hvac_params['detection']['MID_TEMPERATURE_RANGE'][1]))
    mid_temp_scope = hvac_input_temperature[mid_temp_scope_idx]
    low_limit_mid_range = mquantiles(mid_temp_scope, hvac_params['detection']['MID_TEMPERATURE_QUANTILE'][0], alphap=0.5, betap=0.5)
    high_limit_mid_range = mquantiles(mid_temp_scope, hvac_params['detection']['MID_TEMPERATURE_QUANTILE'][1], alphap=0.5, betap=0.5)
    mid_temp_idx = (hvac_input_consumption > 0) & (hvac_input_temperature >= low_limit_mid_range) & \
                   (hvac_input_temperature <= high_limit_mid_range) & (~invalid_idx.astype(bool))

    # Updating secondary cooling detection/estimation flags based on primary user consumption/temperature features
    if user_parameters['all_flags']['is_night_ac']:
        adjust_ac_af_flag = 1
        adjust_ac_setpoint_flag = 1
        adjust_ac_detection_setpoint_flag = 1

    if user_parameters['all_flags']['is_disc_ac']:
        adjust_ac_af_flag = 1
        adjust_ac_nclusters_flag = 1

    if user_temp_type == config['user_temp_type_map']['hot'] or user_temp_type == config['user_temp_type_map']['cold'] or \
            np.sum(mid_temp_idx) < config['min_bincount_midtemp_hist'] or np.sum(mid_temp_idx)/len(mid_temp_idx) > config['max_saturation_baseline']:
        adjust_midtemp_flag = 1

    # Low consumption AC detection only enabled for indian pilots
    if (user_parameters['all_flags']['low_summer_consumption_user_flag']) and (timezone == 'Asia/Kolkata') \
            and (user_parameters['all_flags']['is_low_ac']):
        adjust_ac_detection_range_flag = 1
        adjust_ac_min_amp_flag = 1

    result = {
        'adjust_ac_af': adjust_ac_af_flag,
        'adjust_midtemp_flag': adjust_midtemp_flag,
        'adjust_ac_setpoint_flag': adjust_ac_setpoint_flag,
        'adjust_ac_detection_setpoint_flag': adjust_ac_detection_setpoint_flag,
        'adjust_ac_detection_range_flag': adjust_ac_detection_range_flag,
        'adjust_ac_nclusters_flag': adjust_ac_nclusters_flag,
        'is_night_ac': user_parameters['all_flags']['is_night_ac'],
        'is_day_ac': user_parameters['all_flags']['is_day_ac'],
        'is_disc_ac': user_parameters['all_flags']['is_disc_ac'],
        'is_not_ac': user_parameters['all_flags']['is_not_ac'],
        'hot_cold_normal_user_flag': user_parameters['all_flags']['hot_cold_normal_user_flag'],
        'low_summer_consumption_user_flag': adjust_ac_min_amp_flag,
        'low_winter_consumption_user_flag': 0,
        'min_days_condition_season_classn': user_parameters['all_flags']['min_days_condition_season_classn']
    }
    return result


def get_adjusted_hvac_parameters(disagg_input_object, user_flags, user_parameters, hvac_params, config, appliance):
    """
    Function to get adjusted hvac parameters
    Args:
        disagg_input_object (dict)  : Dictionary containing all disagg inputs
        user_flags          (dict)  : Dictionary containing all user pipeline related flags
        user_parameters     (dict)  : Dictionary containing all user extracted labels and features
        hvac_params         (dict)  : Dictionary containing hvac algo related initialized parameters
        config              (dict)  : Dictionary with fixed parameters to calculate user characteristics
        appliance           (str)   : String indicator for appliance type AC/SH
    Returns:
        disagg_input_object (dict)  : Updated Dictionary containing all disagg inputs
        hvac_params         (dict)  : Updated Dictionary containing hvac algo related initialized parameters
    """

    # Adjust the ac/sh parameters according to flags based on user consumption/temperature features
    if appliance == 'AC':
        # Check if setpoint range needs to be updated in the hvac_params object
        if user_flags['adjust_ac_setpoint_flag']:
            setpoints_list = hvac_params['setpoint'][appliance]['SETPOINTS']
            adjusted_setpoint_list = get_ac_adjusted_setpoint_list(setpoints_list, user_parameters, 'estimation')
            hvac_params['setpoint']['AC']['SETPOINTS'] = adjusted_setpoint_list

            setpoints_list = hvac_params['detection']['AC']['SETPOINTS']
            adjusted_setpoint_list = get_ac_adjusted_setpoint_list(setpoints_list, user_parameters, 'detection')
            hvac_params['detection']['AC']['SETPOINTS'] = adjusted_setpoint_list

        # Check if aggregation factor  needs to be updated to 24 in the disagg_input_object and hvac_params object
        if user_flags['adjust_ac_af']:
            disagg_input_object['switch']['hvac']['hour_aggregate_level_ac'] = config['correction_AF_AC']
            hvac_params['setpoint']['AC']['FALLBACK_HOUR_AGGREGATE'] = config['correction_fallback_AF_AC']

        # Check if detection / estimation / post processing lower limits of ac should be changed if it is a low consumption user
        if user_flags['low_summer_consumption_user_flag'] and user_flags['adjust_ac_detection_range_flag']:
            multiplier = Cgbdisagg.SEC_IN_HOUR / config['sampling_rate']
            hvac_params['detection']['AC']['MIN_AMPLITUDE'] = config['MINIMUM_AMPLITUDE']['low_consumption']['AC']
            hvac_params['estimation']['AC']['MIN_AMPLITUDE'] = config['MINIMUM_AMPLITUDE']['low_consumption']['AC']
            hvac_params['postprocess']['AC']['MIN_DAILY_KWH'] = (config['MINIMUM_AMPLITUDE']['low_consumption']['AC'] / Cgbdisagg.WH_IN_1_KWH) * 3 * multiplier

    # Default parameters for SH
    elif appliance == 'SH':
        disagg_input_object['switch']['hvac']['hour_aggregate_level_sh'] = config['correction_AF_SH']
        hvac_params['setpoint']['SH']['FALLBACK_HOUR_AGGREGATE'] = config['correction_fallback_AF_SH']

    return disagg_input_object, hvac_params


def adjust_ac_detection_range(amplitude_info, raw_hist_centers, hvac_input_consumption, pre_pipeline_params, hvac_params):
    """
    Function to adjust AC Detection Range lower cluster limits after AC detection pipeline
    Args:
        amplitude_info          (dict)           : Dictionary containing hvac detection related debugging information
        hvac_input_consumption  (np.ndarray)     : Array of epoch level consumption flowing into hvac module
        raw_hist_centers        (np.ndarray)     : Array of the consumption bin centers
        pre_pipeline_params     (dict)           : Dictionary containing all user extracted labels and features
        hvac_params             (dict)           : Dictionary containing hvac algo related initialized parameters
    Returns:
        amplitude_info          (dict)           : Dictionary containing hvac detection related debugging information
                                                   with updated amplitude range
    """
    # Unpack all input objects
    config = pre_pipeline_params['config']
    cluster_limits = amplitude_info['cluster_limits']
    adjust_ac_detection_range_flag = pre_pipeline_params['all_flags']['adjust_ac_detection_range_flag']
    mean_array = pre_pipeline_params['hvac']['consumption_parameters']['range_params']['summer']['mean_array']
    std_array = pre_pipeline_params['hvac']['consumption_parameters']['range_params']['summer']['std_array']
    bin_width = np.median(np.diff(raw_hist_centers))

    # Adjust lower mode range if the standard deviation of the range is zero
    mode_0_filtered = np.where((hvac_input_consumption >= cluster_limits[0][0]) & (hvac_input_consumption < cluster_limits[0][1]), hvac_input_consumption, 0)
    if np.sum(mode_0_filtered) > 0 and np.nanstd(mode_0_filtered[mode_0_filtered > 0]) == 0:
        amplitude_range = [cluster_limits[0][0] - bin_width, cluster_limits[0][1] + bin_width]
        cluster_limits = [amplitude_range , [cluster_limits[1][0], cluster_limits[1][1]]]

    # Adjust upper mode range if the standard deviation of the range is zero
    mode_1_filtered = np.where((hvac_input_consumption >= cluster_limits[1][0]) & (hvac_input_consumption < cluster_limits[1][1]), hvac_input_consumption, 0)
    if np.sum(mode_1_filtered) > 0 and np.nanstd(mode_1_filtered[mode_1_filtered > 0]) == 0:
        amplitude_range = [cluster_limits[1][0] - bin_width, cluster_limits[1][1] + bin_width]
        cluster_limits = [[cluster_limits[0][0], cluster_limits[0][1]], amplitude_range]

    # Adjust ac detection range in case of low consumption cooling user
    if adjust_ac_detection_range_flag:
        lower = mean_array - config['overlap_day_overnight']['arm_factor'] * std_array
        lower = max(hvac_params['estimation']['AC']['MIN_AMPLITUDE'], lower)
        lower = max(cluster_limits[0][0], lower)
        cluster_limits = ((lower, cluster_limits[0][1]), (cluster_limits[1][0], cluster_limits[1][1]))
    else:
        lower = max(hvac_params['estimation']['AC']['MIN_AMPLITUDE'], cluster_limits[0][0])
        cluster_limits = ((lower, cluster_limits[0][1]), (cluster_limits[1][0], cluster_limits[1][1]))

    amplitude_info['cluster_limits'] = cluster_limits
    return amplitude_info


def get_ac_adjusted_setpoint_list(setpoints_list, user_parameters, stage):
    """
    Function to get adjusted estimation / detection setpoint list
    Args:
        setpoints_list          (list)      : list of candidate setpoints to select cooling setpoint from
        user_parameters         (dict)      : Dictionary containing all user extracted labels and features
        stage                   (str)       : String to indicate detection/estimation setpoint range
    Returns:
        adjusted_setpoint_list  (list)      : Updated list of candidate setpoints to select cooling setpoint from
    """
    # Extract adjusted estimation/detection setpoint list
    if stage == 'estimation':
        valid_range = user_parameters['cooling']['cooling_temperature_params_candidates']['estimation_setpoint_valid']
    else:
        valid_range = user_parameters['cooling']['cooling_temperature_params_candidates']['detection_setpoint_valid']

    ac_low = int(np.min(setpoints_list))
    ac_high = int(np.max(setpoints_list))

    if stage == 'estimation':
        if valid_range[0] < ac_low or valid_range[1] < ac_high:
            ac_low = int(valid_range[0])
            ac_high = int(valid_range[1])
            setpoints_list = np.array(list(range(ac_high, ac_low-1, -1)))
    else:
        if valid_range[0] < ac_low or valid_range[1] < ac_high:
            ac_low = int(valid_range[0])
            ac_high = int(valid_range[1])
            setpoints_list = np.array(list(range(ac_high, ac_low-1, -5)))

    adjusted_setpoint_list = setpoints_list
    if len(adjusted_setpoint_list) == 0:
        adjusted_setpoint_list = setpoints_list

    return adjusted_setpoint_list


def get_ac_adjusted_detection_params(hvac_input_consumption, hvac_input_temperature, raw_hist_centers,
                                     pre_pipeline_params, debug_detection, hvac_params):
    """
    Function to get adjusted ac detection setpoint range and mid-temperature index before detection pipeline
    Args:
        hvac_input_consumption  (np.ndarray)   : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature  (np.ndarray)   : Array of epoch level temperature flowing into hvac module
        raw_hist_centers        (np.ndarray)   : Array containing all the bin centers
        pre_pipeline_params     (dict)         : Dictionary containing all calculated pre pipeline parameters
        debug_detection         (dict)         : Dictionary containing hvac detection related debugging information
        hvac_params             (dict)         : Dictionary containing hvac algo related initialized parameters
    Returns:
        baseline                (dict)         : Dictionary containing updated hvac detection related debugging information
        hvac_params             (dict)         : Dictionary containing updated hvac algo related initialized parameters
    """
    # Unpack all input objects
    config = pre_pipeline_params['config']
    user_temp_parameters = pre_pipeline_params['hvac']['cooling']['cooling_temperature_params_candidates']
    adjust_midtemp_flag = pre_pipeline_params['all_flags']['adjust_midtemp_flag']
    adjust_detection_setpoint_flag = pre_pipeline_params['all_flags']['adjust_ac_detection_setpoint_flag']
    adjust_ac_detection_range_flag = pre_pipeline_params['all_flags']['adjust_ac_detection_range_flag']
    mid_temp_idx = debug_detection['mid']['mid_temp_idx']
    midtemp_range_min = debug_detection['mid']['temp']
    hist_mid_temp = debug_detection['mid']['hist']
    quan_005 = np.around(super_percentile(hvac_input_consumption, 0.5), 5)
    quan_995 = np.around(super_percentile(hvac_input_consumption, 99.5), 5)
    qualified_epoch_idx = np.logical_and(hvac_input_consumption >= quan_005, hvac_input_consumption <= quan_995)
    consumption_array = hvac_input_consumption[qualified_epoch_idx]
    step_size = 5
    remove_amplitude = np.min(consumption_array) + hvac_params['detection']['NUM_BINS_TO_REMOVE'] * \
                       (np.max(consumption_array) - np.min(consumption_array)) / hvac_params['detection']['NUM_BINS']
    index_min_bin = np.argmin(np.abs(raw_hist_centers - remove_amplitude))
    num_bins_to_remove = int(np.max([hvac_params['detection']['NUM_BINS_TO_REMOVE'], index_min_bin]))
    detection_setpoints_list = hvac_params['detection']['AC']['SETPOINTS']

    # Initialise object for candidate mid temperature ranges
    hvac_params['detection']['NUM_BINS_TO_REMOVE'] = num_bins_to_remove
    baseline = {
        'epoch_level': {'mid_temp_idx': mid_temp_idx, 'hist_mid_temp': hist_mid_temp,
                        'midtemp_range': midtemp_range_min, 'valid': int(np.sum(mid_temp_idx) > config['min_bincount_midtemp_hist'])},
        'relative_winter': {'mid_temp_idx': mid_temp_idx, 'hist_mid_temp': hist_mid_temp,
                            'midtemp_range': midtemp_range_min, 'valid': 0},
    }

    # Adjust only detection setpoint if mid temp adjustment is disabled
    # Make sure that there are atleast 5 candidates in detection setpoint list > upper temperature of mid=temp range
    if (not adjust_midtemp_flag) and adjust_detection_setpoint_flag and (np.max(detection_setpoints_list) < midtemp_range_min[1]):
        start, end = int(baseline['epoch_level']['midtemp_range'][1]) + 1, int(
            baseline['epoch_level']['midtemp_range'][1]) + step_size - 1
        for candidate in np.arange(start, end):
            detection_setpoints_list = np.append(detection_setpoints_list, candidate)

    # Add another candidate for mid-temp range, which is only relative winter for cooling
    # Mark the validity of the mid-temp ranges by ensuring a minimum epochs present in that range
    if adjust_midtemp_flag and (np.sum(user_temp_parameters['mid_temp_idx']) >= config['min_bincount_midtemp_hist']):
        mid_temp_idx_winter = user_temp_parameters['mid_temp_idx']
        filtered_midtemp = hvac_input_temperature[mid_temp_idx_winter == 1]
        nan_indices = np.isnan(filtered_midtemp)
        midtemp_range = np.array([np.quantile(filtered_midtemp[(filtered_midtemp > 0) & ~nan_indices], 0.1),
                                  np.quantile(filtered_midtemp[~nan_indices], 0.8)])

        hist_mid_temp = get_hist_bincount(hvac_input_consumption[mid_temp_idx_winter == 1], raw_hist_centers,
                                          num_bins_to_remove)
        baseline['relative_winter']['mid_temp_idx'] = mid_temp_idx_winter
        baseline['relative_winter']['hist_mid_temp'] = hist_mid_temp
        baseline['relative_winter']['midtemp_range'] = midtemp_range
        baseline['relative_winter']['valid'] = int(np.sum(user_temp_parameters['mid_temp_idx']) > config['min_bincount_midtemp_hist'])

    # Adjust the limits in the hvac params for low consumption ac to avoid removal of low amplitude consumption
    if adjust_ac_detection_range_flag:
        hvac_params['estimation']['AC']['MIN_AMPLITUDE'] = pre_pipeline_params['config']['MINIMUM_AMPLITUDE']['low_consumption']['AC']
        hvac_params['detection']['AC']['MIN_AMPLITUDE'] = pre_pipeline_params['config']['MINIMUM_AMPLITUDE']['low_consumption']['AC']

    # Sort the detection setpoint in conservative way
    # Store it in the hvac params object
    detection_setpoints_list = detection_setpoints_list[np.argsort(detection_setpoints_list)[::-1]]
    hvac_params['detection']['AC']['SETPOINTS'] = detection_setpoints_list

    return baseline, hvac_params


def get_hist_bincount(array, raw_hist_centers, num_bins_to_remove):
    """
    Utility Function to return normalised histogram given input array and bin centers
    Args:
        array               (np.ndarray)    : Input array
        raw_hist_centers    (np.ndarray)    : Array containing bin centers
        num_bins_to_remove  (int)           : Number of bins from start to remove from the histogram
    Returns:
        hist_mid_remp       (np.ndarray)    : Normalised Histogram array
    """
    bin_counts, _ = np.histogram(array, bins=raw_hist_centers)
    bin_counts = np.r_[bin_counts, 0]
    bin_counts += 1
    hist_mid_temp = bin_counts.T / np.sum(bin_counts)
    hist_mid_temp[0:num_bins_to_remove] = 0
    return hist_mid_temp


