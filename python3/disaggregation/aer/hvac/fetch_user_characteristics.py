"""
Author - Mirambika Sikdar
Date - 06/12/2023
Calculating pre pipeline HVAC consumption / user temperature type parameters
"""

# Import python packages
import copy
import logging
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats.mstats import mquantiles
from sklearn.preprocessing import StandardScaler

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.hvac_utility import get_all_indices
from python3.disaggregation.aer.hvac.hvac_utility import softmax_custom
from python3.disaggregation.aer.hvac.get_seasonality_calc_funcxns import get_profiles_diff
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile
from python3.disaggregation.aer.hvac.get_seasonality_calc_funcxns import get_proportion_consumption
from python3.disaggregation.aer.hvac.get_seasonality_calc_funcxns import get_hvac_consumption_prob_params
from python3.disaggregation.aer.hvac.get_seasonality_calc_funcxns import get_seasonality_regression_params
from python3.disaggregation.aer.hvac.get_seasonality_calc_funcxns import get_similarity_consumption_profile_transition


def get_mean_hourly_profile(hvac_input_consumption, hvac_input_temperature, season_month_idx_map, all_indices, config,
                            season_label):
    """
    Function to calculate mean hourly vector for consumption and temperature over summer/winter days

    Args:
        hvac_input_consumption  (np.ndarray)    : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature  (np.ndarray)    : Array of epoch level temperature flowing into hvac module
        season_month_idx_map    (np.ndarray)    : Array of day level mapping to season labels
        all_indices             (dict)          : Dictionary containing all day/hour/epoch/month level indices
        config                  (dict)          : Dictionary with fixed parameters to calculate user characteristics
        season_label            (str)           : String identifying summer or winter
    Returns:
        mean_day_cons           (np.ndarray)    : Mean hourly consumption vector over all summer/winter days
        mean_day_temp           (np.ndarray)    : Mean hourly tempeature vector over all summer/winter days
    """
    #
    hour_idx = all_indices['hour_idx']
    invalid_idx = all_indices['invalid_idx']
    hottest_day_bool = all_indices['hottest_day_bool']
    coldest_day_bool = all_indices['coldest_day_bool']

    # Find the epoch indices associated with season=summer/winter days
    filtered_idx = season_month_idx_map == config['season_idx_map'][season_label]

    # If there are less than min season days, use epoch indices of 30 coldest/hottest days
    if np.sum(filtered_idx) <= config['min_season_days']:
        if season_label == 'summer':
            filtered_idx = hottest_day_bool
        else:
            filtered_idx = coldest_day_bool

    # Calculate the mean hourly consumption/temperature vector over all filtered days
    selected_idx = np.logical_and(~invalid_idx, filtered_idx == 1)
    valid_days_hourly_count = np.bincount(hour_idx[selected_idx > 0].astype(int))

    aggregate_day_cons = np.bincount(hour_idx[selected_idx > 0].astype(int),
                                     hvac_input_consumption[selected_idx > 0])
    mean_day_cons = aggregate_day_cons / valid_days_hourly_count

    aggregate_day_temp = np.bincount(hour_idx[selected_idx > 0].astype(int),
                                     hvac_input_temperature[selected_idx > 0])
    mean_day_temp = aggregate_day_temp / valid_days_hourly_count

    # Change the contribution of hour indices with nan or inf values to zero
    invalid_mean_idx = np.where(
        np.isnan(mean_day_cons) | np.isinf(mean_day_cons) | np.isnan(mean_day_temp) | np.isinf(mean_day_temp), 1, 0)
    mean_day_cons[invalid_mean_idx == 1] = 0
    mean_day_temp[invalid_mean_idx == 1] = 0

    return mean_day_cons, mean_day_temp


def classify_user_cons_profile(hvac_input_consumption, invalid_idx, season_month_idx_map, config):
    """
    Function to get user consumption related parameters for summer months separately

    Args:
        hvac_input_consumption              (np.ndarray)  : Array of epoch level consumption flowing into hvac module
        invalid_idx                         (np.ndarray)  : Array of invalid epochs based on consumption and temperature
        season_month_idx_map                (np.ndarray)  : Array of day mapping to a season label
        config                              (dict)        : Dictionary with fixed parameters to calculate user characteristics
    Returns:
        low_summer_consumption_user_flag    (int)         : Integer to flag low consumption cooling users
        consumption_range_params            (dict)        : Dictionary containing amplitude parameters to calculate updated min threshold

    """
    # Avoid making changes in the original array
    hvac_input_consumption = copy.deepcopy(hvac_input_consumption)

    # extract summer epoch indices
    selected_summer_idx = np.logical_and(~invalid_idx, season_month_idx_map == config['season_idx_map']['summer'])

    # initialise variables and result dictionary
    low_summer_consumption_user_flag = 0
    consumption_range_params = {
        'summer': {'mean_array': np.mean(np.zeros(hvac_input_consumption.shape)),
                   'std_array': np.std(np.zeros(hvac_input_consumption.shape)),
                   'percentage_below_min': 0},
    }

    # If summer days are present
    if np.sum(selected_summer_idx) > 0:
        summer_consumption = hvac_input_consumption[selected_summer_idx == 1]
        # Calculate percentage of consumption points with consumption less than 400 WH amongst summer days
        perc_above_lowest_amp_summer = sum(
            np.where(summer_consumption < config['MINIMUM_AMPLITUDE']['AC'], 1, 0)) / sum(selected_summer_idx)

        # Assign low consumption flag to users with high percentage of low amplitude present
        if perc_above_lowest_amp_summer >= config['Consumption_Type'].get('minimum_perc_low_consumption'):
            low_summer_consumption_user_flag = 1
        else:
            low_summer_consumption_user_flag = 0

        consumption_range_params['summer']['mean_array'] = np.nanmean(hvac_input_consumption[selected_summer_idx == 1])
        consumption_range_params['summer']['std_array'] = np.nanstd(hvac_input_consumption[selected_summer_idx == 1])
        consumption_range_params['summer']['percentage_below_min'] = perc_above_lowest_amp_summer

    return low_summer_consumption_user_flag, consumption_range_params


def get_summer_winter_start_end(season_month_idx_map, all_indices, extreme_temp_bool, config, complement_s_label):
    """
    Function to find the day index for start and end of peak summer / winter season
    Args:
        season_month_idx_map    (np.ndarray)  : Array of day mapping to a season label
        all_indices             (dict)        : Dictionary containing all day/hour/epoch/month level indices
        extreme_temp_bool       (np.ndarray)  : Array of boolean values marking hottest/coldest temperature days
        config                  (dict)        : Dictionary with fixed parameters to calculate user characteristics
        complement_s_label      (str)         : string identifier for complementary season to current season
    Returns:
        peak_season_start_end   (tuple)       : Tuple containing start and end day index of peak summer/winter season
    """
    # unpack input objects
    day_idx = all_indices['day_idx']
    unique_day_idx = all_indices['unique_day_idx']
    complementary_season_idx = config['season_idx_map'][complement_s_label]

    # Correct extreme temp bool indices to preserve only longest continous
    # days of hottest / coldest days
    extreme_temp_bool_daily = extreme_temp_bool[unique_day_idx]
    extreme_temp_bool_segments = np.split(extreme_temp_bool_daily, np.nonzero(np.diff(extreme_temp_bool_daily))[0] + 1)
    longest_continous_days_extreme_temp = int(
        np.argmax([len(x) if x[0] == 1 else 0 for x in extreme_temp_bool_segments]))
    cumulative_days = np.cumsum([len(x) for x in extreme_temp_bool_segments])
    if longest_continous_days_extreme_temp != 0:
        extreme_temp_bool_daily[0: cumulative_days[longest_continous_days_extreme_temp - 1]] = 0
    if longest_continous_days_extreme_temp != len(cumulative_days) - 1:
        extreme_temp_bool_daily[cumulative_days[longest_continous_days_extreme_temp + 1]:] = 0
    extreme_temp_bool = extreme_temp_bool_daily[day_idx]

    # Extract the start and end epoch index of the hottest/coldest segment of days
    if extreme_temp_bool[0]:
        start_day_idx = len(extreme_temp_bool) - np.nonzero(extreme_temp_bool[::-1])[0][0] - 1
        end_day_idx = np.nonzero(extreme_temp_bool)[0][0]
    else:
        start_day_idx = np.nonzero(extreme_temp_bool)[0][0]
        end_day_idx = len(extreme_temp_bool) - np.nonzero(extreme_temp_bool[::-1])[0][0] - 1
    start_end = (start_day_idx, end_day_idx)

    # Split the season index array into continous segments of summer/winter/transition
    season_month_idx_map_daily = season_month_idx_map[unique_day_idx]
    season_continous_segments = np.split(season_month_idx_map_daily,
                                         np.nonzero(np.diff(season_month_idx_map_daily))[0] + 1)

    # Store the length in days of the season segments and the cumulative sum array
    season_days_continous = np.array([len(x) for x in season_continous_segments])
    current_season_bool = [1 if x[0] != complementary_season_idx else 0 for x in season_continous_segments]
    cumulative_days = np.cumsum(season_days_continous)

    # Extract the season start and end day index for all continous segments of summer/winter/transition
    season_day_segments = np.array([[cumulative_days[i] - season_days_continous[i],
                                     cumulative_days[i] - 1] for i in range(len(season_days_continous))])

    # Calculate the overlap of each season segment with hottest/coldest segment
    segments_overlap_extreme_temp = np.array([max(0, min(season_day_segments[i][1], start_end[1]) -
                                                  max(season_day_segments[i][0], start_end[0])) for i in
                                              range(len(season_day_segments))])

    # Multiply the overlap array with factor proportional to length of continous season segment
    wt_segments_overlap_extreme_temp = segments_overlap_extreme_temp * \
                                       (season_days_continous / np.sum(season_days_continous)) \
                                       * current_season_bool

    # Extract start and end of the maximum overlap season segment
    # This will be defined as the peak season
    start_season_idx = season_day_segments[np.argmax(wt_segments_overlap_extreme_temp)][0]
    end_season_idx = season_day_segments[np.argmax(wt_segments_overlap_extreme_temp)][1]
    peak_season_start_end = (start_season_idx, end_season_idx)
    return peak_season_start_end


def get_season_start_end(season_month_idx_map, all_indices, season_label, hot_cold_normal_user_flag, config,
                         min_days_condition):
    """
    Function to get the summer/winter season start and end day index including offset transition days at the start and end
    Args:
        season_month_idx_map        (np.ndarray)  : Array of day mapping to a season label
        all_indices                 (dict)        : Dictionary containing all day/hour/epoch/month level indices
        season_label                (str)         : String identifier for summer/winter season
        hot_cold_normal_user_flag   (int)         : Identifier for user temp type
        config                      (dict)        : Dictionary with fixed parameters to calculate user characteristics
        min_days_condition          (bool)        : Flag to indicate if enough number of days are present for distinct
                                                    summer/winter/transition days
    Returns:
        season_start_end            (tuple)       : Tuple containing start and end day index of overall summer/winter season
    """
    # unpack input objects
    day_idx = all_indices['day_idx']
    unique_day_idx = all_indices['unique_day_idx']
    coldest_day_bool = all_indices['coldest_day_bool']
    hottest_day_bool = all_indices['hottest_day_bool']
    temp_type = hot_cold_normal_user_flag
    season_start_end = (day_idx[0], day_idx[-1])

    if not min_days_condition:
        return season_start_end

    # If user temp type is hot, consider all the months as summer + transition
    if hot_cold_normal_user_flag == config['user_temp_type_map']['hot'] and season_label == 'summer':
        season_start_end = (day_idx[0], day_idx[-1])

    # If user temp type is cold, consider all the months as winter + transition
    elif hot_cold_normal_user_flag == config['user_temp_type_map']['cold'] and season_label == 'winter':
        season_start_end = (day_idx[0], day_idx[-1])

    # If user temp type is normal, extract peak season start and end epoch level day index
    else:
        if season_label == 'winter':
            # Peak winter is extracted
            peak_season_start_end = get_summer_winter_start_end(season_month_idx_map, all_indices, coldest_day_bool,
                                                                config, 'summer')
        else:
            # Peak summer is extracted
            peak_season_start_end = get_summer_winter_start_end(season_month_idx_map, all_indices, hottest_day_bool,
                                                                config, 'winter')

        # Add adjacent transition days to peak season start, end indices to include onset and end season days
        buffer_transition_days_season = config['season_idx_map']['max_look_back_period'][temp_type]

        if peak_season_start_end[0] - buffer_transition_days_season < 0:
            remainder_days = buffer_transition_days_season - peak_season_start_end[0]
            season_onset_idx = max(0, day_idx[unique_day_idx[-1]] - remainder_days - 1)
        else:
            season_onset_idx = peak_season_start_end[0] - buffer_transition_days_season

        if peak_season_start_end[1] + buffer_transition_days_season > day_idx[unique_day_idx[-1]]:

            remainder_days = buffer_transition_days_season + peak_season_start_end[1] - \
                             day_idx[unique_day_idx[-1]]
            season_end_idx = remainder_days
        else:
            season_end_idx = min(day_idx[unique_day_idx[-1]], peak_season_start_end[1] + buffer_transition_days_season)
        season_start_end = (season_onset_idx, season_end_idx)

    return season_start_end


def handle_missing_days(daily_mean_temp, all_invalid_epochs_flag):
    """
    Handle the case where last days are invalid and the two arrays have different number of days

    Arguments:
        daily_mean_temp             (np.ndarray)   : Array containing mean temperature over valid epochs of every day
        all_invalid_epochs_flag     (np.ndarray)   : Array flagging all days with 0 valid epochs
    Returns:
        daily_mean_temp             (np.ndarray)   : Updated daily mean temperature array with the total number of days
    """
    # Check if error handling is required in the first place
    if len(daily_mean_temp) < len(all_invalid_epochs_flag):
        n_days_at_end_missing = len(all_invalid_epochs_flag) - len(daily_mean_temp)
        for _ in range(n_days_at_end_missing):
            daily_mean_temp = np.append(daily_mean_temp, np.nan)

    return daily_mean_temp


def get_relative_season_transitions(season_month_idx_map, config):
    """
    Function to calculate the number of summer to transition season label changes
    Arguments:
        season_month_idx_map        (np.ndarray) : Array of day mapping to a season label
        config                      (dict)       : Dictionary with fixed parameters to calculate user characteristics
    Return:
        cyclic_transition_summer    (int)        : Integer indicator of whether the temperature profile is cyclic
    """
    # If the user is a cold user, relative summer will be transition and transition (or baseline) will be winter
    if np.sum(season_month_idx_map == config['season_idx_map']['summer']) == 0:
        summer_mask = season_month_idx_map == config['season_idx_map']['transition']
        transition_mask = season_month_idx_map == config['season_idx_map']['winter']
    # Normal User
    else:
        summer_mask = season_month_idx_map == config['season_idx_map']['summer']
        transition_mask = season_month_idx_map == config['season_idx_map']['transition']

    cyclic_transition_summer = int(
        np.sum(summer_mask[:-1] & transition_mask[1:]) + np.sum(transition_mask[:-1] & summer_mask[1:]))
    return cyclic_transition_summer


def classify_user_temp_profile(hvac_input_temperature, all_indices, config):
    """
    Function to assign user temperature profile type - hot/cold/normal and assign day-level season labels
    Args:
        hvac_input_temperature  (np.ndarray)   : Array of epoch level temperature flowing into hvac module
        all_indices             (dict)         : Dictionary containing all day/hour/epoch/month level indices
        config                  (dict)         : Dictionary with fixed parameters to calculate user characteristics
    Returns:
        result_temp_profile     (dict)         : Dictionary containing user temperature and consumption type parameters
    """
    # Unpack all input objects
    hvac_input_temperature = copy.deepcopy(hvac_input_temperature)
    day_idx = all_indices['day_idx']
    unique_day_idx = all_indices['unique_day_idx']
    invalid_idx = all_indices['invalid_idx']
    sampling_rate = config['sampling_rate']
    epochs_in_hour = Cgbdisagg.SEC_IN_HOUR / sampling_rate

    # Get the lower quantile, median and upper quantile temperature
    lower_quantile_temp = np.nanquantile(hvac_input_temperature,
                                         config['user_temp_type_map']['classification_params']['lower_quantile'])
    upper_quantile_temp = np.nanquantile(hvac_input_temperature,
                                         config['user_temp_type_map']['classification_params']['upper_quantile'])
    median_daily_temp = np.nanmedian(hvac_input_temperature)
    lower_temp_threshold = config['user_temp_type_map']['classification_params']['lower_temp']
    upper_temp_threshold = config['user_temp_type_map']['classification_params']['upper_temp']
    median_temp_threshold = config['user_temp_type_map']['classification_params']['mid_temp']

    # Get the number of total days in data to check for min days condition
    n_days_data = (np.sum(~invalid_idx) / epochs_in_hour) / Cgbdisagg.HRS_IN_DAY
    min_days_condition = True
    invalid_idx = np.logical_or(invalid_idx, np.isnan(hvac_input_temperature))

    # If min days condition fails, assign user temp type as normal by default and number of different seasons as 2
    if n_days_data < config['season_idx_map']['min_months_req'] * Cgbdisagg.DAYS_IN_MONTH:
        flag = config['user_temp_type_map']['normal']
        min_days_condition = False
        n_clusters = 2
    else:
        # Based on conditions, assign user temp type
        # For hot and cold temp type, number of different seasons are 2 (summer and transition / winter and transition
        # For normal type, number of different seasons are 3 (summer and transition and winter
        # For hot temp type, assign invalid flag to epochs where temperature <= min_temp_check_non_cold_users

        # By default : Normal Temp Type
        flag = config['user_temp_type_map']['normal']
        n_clusters = 3

        if (lower_quantile_temp <= lower_temp_threshold) & (upper_quantile_temp <= upper_temp_threshold):
            flag = config['user_temp_type_map']['cold']
            n_clusters = 2
        elif lower_quantile_temp > lower_temp_threshold:
            flag = config['user_temp_type_map']['hot']
            invalid_idx_additional = np.logical_or(invalid_idx, hvac_input_temperature <
                                                   config['season_idx_map']['min_temp_check_non_cold_users'])
            invalid_idx = np.logical_or(invalid_idx, invalid_idx_additional)
            hvac_input_temperature[invalid_idx == 1] = np.nan
            n_clusters = 2

    # Calculate daily mean temperature excluding invalid epochs
    all_invalid_epochs_flag = np.where(np.bincount(day_idx, ~invalid_idx) == 0, 1, 0)
    season_labels_daily = np.zeros(unique_day_idx.shape)
    season_labels_daily[:] = -1

    daily_mean_temp = np.bincount(day_idx[invalid_idx == 0], hvac_input_temperature[invalid_idx == 0]) / \
                      np.bincount(day_idx[invalid_idx == 0])

    # If all the indices at the last days are invalid, daily_mean_temp will have lesser day indices than day_idx
    # Error correction : Append np.nan towards the end to represent missing values
    daily_mean_temp = handle_missing_days(daily_mean_temp, all_invalid_epochs_flag)

    daily_mean_temp[all_invalid_epochs_flag == 1] = np.nan

    # Identify valid days with atleast one epoch with real/non-nan temperature value
    invalid_days_idx = np.where(np.isinf(daily_mean_temp) | np.isnan(daily_mean_temp), 1, 0).astype(bool)
    if flag == config['user_temp_type_map']['hot']:
        # Hot Temperature Profile users should not have 0 F mean day temperature
        invalid_days_idx = np.logical_or(invalid_days_idx, (daily_mean_temp == 0))

    valid_days_idx = ~invalid_days_idx

    # Perform K-means clustering on scaled daily mean temperature of the valid days only
    daily_mean_temp = daily_mean_temp[valid_days_idx == 1]
    feature_vector = np.zeros((len(daily_mean_temp), 1))
    feature_vector[:, 0] = daily_mean_temp
    feature_vector_scaled = StandardScaler().fit_transform(feature_vector)
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(feature_vector_scaled)
    season_labels_daily[valid_days_idx == 1] = clusters
    season_labels_daily = season_labels_daily[day_idx]
    season_month_idx_map = copy.deepcopy(season_labels_daily)
    cluster_ids = np.unique(clusters)

    # Assignment of days associated with cluster ids will be into summer/winter/transition based on temperature of centroids
    if n_clusters == 3:
        # Sort cluster ids in ascending order wrt their centroid temperature value
        sorted_cluster_ids = np.argsort([np.nanmean(feature_vector[clusters == cluster_ids[0], 0]),
                                         np.nanmean(feature_vector[clusters == cluster_ids[1], 0]),
                                         np.nanmean(feature_vector[clusters == cluster_ids[2], 0])])
        season_month_idx_map[season_month_idx_map == cluster_ids[sorted_cluster_ids[0]]] = config['season_idx_map'][
            'winter']
        season_month_idx_map[season_month_idx_map == cluster_ids[sorted_cluster_ids[1]]] = config['season_idx_map'][
            'transition']
        season_month_idx_map[season_month_idx_map == cluster_ids[sorted_cluster_ids[2]]] = config['season_idx_map'][
            'summer']

    else:
        # Sort cluster ids in ascending order wrt their centroid temperature value
        sorted_cluster_ids = np.argsort([np.mean(feature_vector[clusters == cluster_ids[0], 0]),
                                         np.mean(feature_vector[clusters == cluster_ids[1], 0])])

        summer_days_condition = min_days_condition and (median_daily_temp >= median_temp_threshold)
        winter_days_condition = min_days_condition and (median_daily_temp < median_temp_threshold)

        if flag == config['user_temp_type_map']['hot'] or summer_days_condition:
            season_month_idx_map[season_month_idx_map == cluster_ids[sorted_cluster_ids[0]]] = config['season_idx_map'][
                'transition']
            season_month_idx_map[season_month_idx_map == cluster_ids[sorted_cluster_ids[1]]] = config['season_idx_map'][
                'summer']

        elif flag == config['user_temp_type_map']['cold'] or winter_days_condition:
            season_month_idx_map[season_month_idx_map == cluster_ids[sorted_cluster_ids[0]]] = config['season_idx_map'][
                'winter']
            season_month_idx_map[season_month_idx_map == cluster_ids[sorted_cluster_ids[1]]] = config['season_idx_map'][
                'transition']

    # Extract start and end day index of overall summer / winter season based on user temp type
    summer_start_end = get_season_start_end(season_month_idx_map, all_indices, 'summer', flag, config,
                                            min_days_condition)
    winter_start_end = get_season_start_end(season_month_idx_map, all_indices, 'winter', flag, config,
                                            min_days_condition)

    # Calculate number of summer<->transition conversions and identify cyclicity in temperature profile
    cyclic_transition_summer = get_relative_season_transitions(season_month_idx_map, config)

    cyclic_temp_profile = int(cyclic_transition_summer >= config['user_temp_type_map']['season_transition_cutoff'])

    # Final output object
    result_temp_profile = {
        'hot_cold_normal_user_flag': flag,
        'cyclic_temp_profile_flag': cyclic_temp_profile,
        'season_month_idx_map': season_month_idx_map,
        'summer_start_end': summer_start_end,
        'winter_start_end': winter_start_end,
        'season_transitions': cyclic_transition_summer,
        'min_days_condition': int(min_days_condition)
    }
    return result_temp_profile


def get_cooling_seasonality_params(hvac_input_consumption, hvac_input_temperature, hist_centers_consumption,
                                   all_indices, common_objects, config):
    """
    Function to calculate all features to check if cooling is present within overnight hours
    Args:
        hvac_input_consumption      (np.ndarray)   : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature      (np.ndarray)   : Array of epoch level temperature flowing into hvac module
        hist_centers_consumption    (np.ndarray)   : Array of histogram centers from overall consumption data
        all_indices                 (dict)         : Dictionary containing all day/epoch/month/hour level indices
        common_objects              (dict)         : Dictionary containing remaining required parameters
        config                      (dict)         : Dictionary with fixed parameters to calculate user characteristics
    Returns:
        result_seasonality_params   (dict)         : Dictionary storing all calculated cooling seasonality features
    """
    # Unpack input objects
    hvac_input_consumption = copy.deepcopy(hvac_input_consumption)
    hvac_input_temperature = copy.deepcopy(hvac_input_temperature)
    hour_idx = all_indices['hour_idx']
    invalid_idx = all_indices['invalid_idx']
    season_start_end = all_indices['summer_start_end']
    mean_day_cons = common_objects['mean_day_cons']
    day_hours_boundary = common_objects['day_hours']
    low_consumption_user_flag = common_objects['low_consumption_user_flag']
    hot_cold_normal_user_flag = common_objects['hot_cold_normal_user_flag']
    day_idx = all_indices['day_idx']

    # Get the index for morning hours, night hours and overnight hours and day hours
    valid_idx_overnight_1 = np.logical_and(~invalid_idx, hour_idx < day_hours_boundary[0])
    valid_idx_overnight_2 = np.logical_and(~invalid_idx, hour_idx > day_hours_boundary[1])
    valid_idx_overnight_3 = np.logical_or(valid_idx_overnight_1, valid_idx_overnight_2)
    valid_idx_day = np.logical_and(~invalid_idx,
                                   np.logical_and(hour_idx >= day_hours_boundary[0], hour_idx <= day_hours_boundary[1]))

    # Initialise variables with default values
    regression_result = {'coefficient': np.array([-10000]), 'r2': 0, 'corr': 0}
    regression_flag_overnight = False
    valid_range_morning, valid_range_night = np.array([np.nan, np.nan]), np.array([np.nan, np.nan])
    regression_result_low_consumption, regression_result_overnight_overnight = regression_result, regression_result
    regression_result_overnight_morning, regression_result_overnight_night, regression_result_day = regression_result, regression_result, regression_result
    cosine_similarity_day, cosine_similarity_overnight = 0, 0
    cosine_similarity_overnight_morning, cosine_similarity_overnight_night = 0, 0
    cooling_probability_low_consumption_candidates = []
    min_amplitude_hvac = config['MINIMUM_AMPLITUDE']['AC'] * (config['sampling_rate'] / Cgbdisagg.SEC_IN_HOUR)

    # Calculate proportion of consumption between day and overnight hours
    proportion_cons = get_proportion_consumption(hvac_input_consumption, mean_day_cons, day_hours_boundary, all_indices)

    # Calculate cooling histogram, minimum cooling amplitude and probability of cooling present
    # Do this independently for day, night, morning segments
    # Select the segment between night and morning with maximum probability of cooling being present
    cooling_probability_day = get_hvac_consumption_prob_params(hvac_input_consumption, hist_centers_consumption,
                                                               valid_idx_day, all_indices, config,
                                                               low_consumption_user_flag, hot_cold_normal_user_flag,
                                                               'summer')

    cooling_probability_overnight_morning = get_hvac_consumption_prob_params(hvac_input_consumption,
                                                                             hist_centers_consumption,
                                                                             valid_idx_overnight_1, all_indices, config,
                                                                             low_consumption_user_flag,
                                                                             hot_cold_normal_user_flag, 'summer')

    cooling_probability_overnight_night = get_hvac_consumption_prob_params(hvac_input_consumption,
                                                                           hist_centers_consumption,
                                                                           valid_idx_overnight_2, all_indices, config,
                                                                           low_consumption_user_flag,
                                                                           hot_cold_normal_user_flag, 'summer')

    cooling_probability_overnight_candidates = [cooling_probability_overnight_morning,
                                                cooling_probability_overnight_night]

    index = np.argmax(
        [cooling_probability_overnight_morning['hvac_prob'], cooling_probability_overnight_night['hvac_prob']])
    cooling_probability_overnight = cooling_probability_overnight_candidates[index]

    # If the cooling probability is > 0
    # Then seasonality measurement features are calculated
    # This is done independently for day, night, morning and overnight
    # Select candidate with the highest confidence in seasonality between night and morning
    # Store appropriate candidate for low consumption cooling seasonality detection
    if np.sum(cooling_probability_overnight_morning['max_diff_hist']) > 0:
        valid_range_morning = [
            np.min(hist_centers_consumption[cooling_probability_overnight_morning['max_diff_hist'] > 0]),
            np.max(hist_centers_consumption[cooling_probability_overnight_morning['max_diff_hist'] > 0])]
        regression_result_overnight_morning = get_seasonality_regression_params(hvac_input_consumption,
                                                                                hvac_input_temperature,
                                                                                valid_idx_overnight_1,
                                                                                all_indices, season_start_end, config,
                                                                                'summer', valid_range_morning)

        cosine_similarity_overnight_morning = get_profiles_diff(hvac_input_consumption, hvac_input_temperature,
                                                                valid_idx_overnight_1, all_indices, 'summer',
                                                                valid_range_morning, config)

        if valid_range_morning[1] <= min_amplitude_hvac:
            cooling_probability_low_consumption_candidates.append(regression_result_overnight_morning)
        regression_flag_overnight = True

    if np.sum(cooling_probability_overnight_night['max_diff_hist']) > 0:
        valid_range_night = [np.min(hist_centers_consumption[cooling_probability_overnight_night['max_diff_hist'] > 0]),
                             np.max(hist_centers_consumption[cooling_probability_overnight_night['max_diff_hist'] > 0])]
        regression_result_overnight_night = get_seasonality_regression_params(hvac_input_consumption,
                                                                              hvac_input_temperature,
                                                                              valid_idx_overnight_2,
                                                                              all_indices, season_start_end, config,
                                                                              'summer', valid_range_night)
        cosine_similarity_overnight_night = get_profiles_diff(hvac_input_consumption, hvac_input_temperature,
                                                              valid_idx_overnight_2, all_indices, 'summer',
                                                              valid_range_night,
                                                              config)

        if valid_range_night[1] <= min_amplitude_hvac:
            cooling_probability_low_consumption_candidates.append(regression_result_overnight_night)
        regression_flag_overnight = True

    if regression_flag_overnight:
        valid_range_overnight = [np.nanmin([valid_range_night[0], valid_range_morning[0]]),
                                 np.nanmax([valid_range_night[1], valid_range_morning[1]])]
        regression_result_overnight_overnight = get_seasonality_regression_params(hvac_input_consumption,
                                                                                  hvac_input_temperature,
                                                                                  valid_idx_overnight_3,
                                                                                  all_indices,
                                                                                  (day_idx[0], day_idx[-1]), config,
                                                                                  'summer', valid_range_overnight)
        if valid_range_overnight[1] <= min_amplitude_hvac:
            cooling_probability_low_consumption_candidates.append(regression_result_overnight_overnight)

    regression_result_overnight_candidates = [regression_result_overnight_morning, regression_result_overnight_night,
                                              regression_result_overnight_overnight]
    index = np.argmax(
        [regression_result_overnight_morning['coefficient'][0], regression_result_overnight_night['coefficient'][0],
         regression_result_overnight_overnight['coefficient'][0]])
    regression_result_overnight = regression_result_overnight_candidates[index]
    cosine_similarity_overnight_candidates = [cosine_similarity_overnight_morning, cosine_similarity_overnight_night]
    index = np.argmax(cosine_similarity_overnight_candidates)
    cosine_similarity_overnight = cosine_similarity_overnight_candidates[index]

    if np.sum(cooling_probability_day['max_diff_hist']) > 0:
        valid_range = [np.min(hist_centers_consumption[cooling_probability_day['max_diff_hist'] > 0]),
                       np.max(hist_centers_consumption[cooling_probability_day['max_diff_hist'] > 0])]

        regression_result_day = get_seasonality_regression_params(hvac_input_consumption,
                                                                  hvac_input_temperature,
                                                                  valid_idx_day,
                                                                  all_indices, season_start_end, config,
                                                                  'summer', valid_range)

        if valid_range[1] <= min_amplitude_hvac:
            cooling_probability_low_consumption_candidates.append(regression_result_day)

    if low_consumption_user_flag:
        valid_range = [0, min_amplitude_hvac]
        valid_idx = np.ones(hvac_input_consumption.shape)
        regression_result_low_consumption_filtered = get_seasonality_regression_params(hvac_input_consumption,
                                                                                       hvac_input_temperature,
                                                                                       valid_idx,
                                                                                       all_indices,
                                                                                       (day_idx[0], day_idx[-1]),
                                                                                       config,
                                                                                       'summer', valid_range)

        if valid_range[1] <= min_amplitude_hvac:
            cooling_probability_low_consumption_candidates.append(regression_result_low_consumption_filtered)

        index = np.argmax([x['coefficient'][0] for x in cooling_probability_low_consumption_candidates])
        regression_result_low_consumption = cooling_probability_low_consumption_candidates[index]

    # Calculate similarity in summer and transition consumption profiles
    consumption_profile_change = get_similarity_consumption_profile_transition(hvac_input_consumption, all_indices,
                                                                               hot_cold_normal_user_flag, 'AC', config)

    result_seasonality_params = {
        'Detection': {
            'consumption_profile_change': consumption_profile_change,
            'hvac_prob': {
                'day': cooling_probability_day,
                'overnight': cooling_probability_overnight,
            },
            'exist': False
        },
        'cosine_similarity': {
            'cosine_similarity_overnight': cosine_similarity_overnight,
            'cosine_similarity_day': cosine_similarity_day,
        },
        'proportion_cons': proportion_cons,

        'regression_result': {
            'day': regression_result_day,
            'overnight': regression_result_overnight,
            'low_consumption': regression_result_low_consumption
        },
    }
    return result_seasonality_params


def get_adjusted_cooling_temperature_range(hvac_input_temperature, hvac_input_consumption, seasonality_params,
                                           cooling_consumption_type, hot_cold_normal_user_flag, day_hours_boundary,
                                           all_indices, config, hvac_params):
    """
    Function to adjust cooling detection/estimation temperature ranges according to cooling consumption type
    Args:
        hvac_input_consumption          (np.ndarray)    : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature          (np.ndarray)    : Array of epoch level temperature flowing into hvac module
        seasonality_params              (dict)          : Dictionary storing all calculated cooling seasonality features
        cooling_consumption_type        (dict)          : Dictionary containing all user consumption type flags
        hot_cold_normal_user_flag       (int)           : Integer label to indicate user temperature profile type
        day_hours_boundary              (tuple)         : Tuple with start, end hour of a day
        all_indices                     (dict)          : Dictionary containing all day/epoch/month/hour level indices
        config                          (dict)          : Dictionary containing fixed parameters to calculate user characteristics
        hvac_params                     (dict)          : Dictionary containing hvac algo related initialized parameters
    Returns:
        result                          (dict)          : Dictionary containing adjusted setpoints and mid temp ranges
    """
    # Unpack input objects
    season_month_idx_map = all_indices['season_month_idx_map']
    invalid_idx = all_indices['invalid_idx']
    month_idx = all_indices['month_idx']
    hottest_day_bool = all_indices['hottest_day_bool']
    hour_idx = all_indices['hour_idx']
    diff_hist_night = seasonality_params['Detection']['hvac_prob']['overnight']['max_diff_hist']
    probable_ac_amp_overnight = seasonality_params['Detection']['hvac_prob']['overnight']['probable_hvac_amp']
    relative_summer_idx = season_month_idx_map == config['season_idx_map']['summer']
    # If no "summer" season label is present, consider the hottest days as relative summer
    if np.sum(relative_summer_idx) == 0:
        relative_summer_idx = hottest_day_bool
    correction = False

    # Initialise variables
    valid_setpoint_range_estm = [np.nanmax(hvac_input_temperature), np.nanmax(hvac_input_temperature)]
    valid_setpoint_range_detn = [np.nanmax(hvac_input_temperature), np.nanmax(hvac_input_temperature)]
    if hot_cold_normal_user_flag == config['user_temp_type_map']['cold']:
        valid_setpoint_range_detn[0] = config['lower_limit_AC_setpoint_cold_temp']
        valid_setpoint_range_estm[0] = config['lower_limit_AC_setpoint_cold_temp']
    else:
        valid_setpoint_range_detn[0] = config['lower_limit_AC_setpoint_normal_temp']
        valid_setpoint_range_estm[0] = config['lower_limit_AC_setpoint_normal_temp']
    correction_range = [np.nanmin(hvac_input_temperature), np.nanmax(hvac_input_temperature)]

    # Extract overnight index for summer days when consumption > probable lower limit hvac
    filter_idx = np.logical_and(~invalid_idx, relative_summer_idx)
    valid_idx_overnight = np.logical_and(~invalid_idx,
                                         np.logical_or(hour_idx < day_hours_boundary[0],
                                                       hour_idx > day_hours_boundary[1]))
    cooling_idx = np.logical_and(filter_idx, hvac_input_consumption >= probable_ac_amp_overnight)
    cooling_overnight_idx = np.logical_and(filter_idx, valid_idx_overnight)

    # Extract temperature values corresponding to overnight index
    filtered_temp_overnight = hvac_input_temperature[cooling_overnight_idx == 1]

    # Change setpoint range only if night cooling is detected
    if cooling_consumption_type['is_night_ac'] and (len(diff_hist_night[diff_hist_night > 0]) > 0) and not np.isnan(
            probable_ac_amp_overnight):
        correction_range = [np.nanquantile(filtered_temp_overnight[filtered_temp_overnight > 0], 0.1),
                            np.nanquantile(filtered_temp_overnight, 0.8)]
        correction_range_detn = [np.nanquantile(filtered_temp_overnight[filtered_temp_overnight > 0], 0.1),
                                 np.nanquantile(filtered_temp_overnight, 0.7)]
        correction = True
        valid_setpoint_range_detn[1] = max(correction_range_detn[1], valid_setpoint_range_detn[0] + 5)
        valid_setpoint_range_estm[1] = max(correction_range[1], valid_setpoint_range_detn[0] + 5)

    # Mid-temp range calculation is updated for hot temp type
    # Otherwise, default method is used
    if hot_cold_normal_user_flag == config['user_temp_type_map']['hot']:
        total_hrs_month = np.bincount(month_idx[~invalid_idx])
        monthly_avg_temp = np.bincount(month_idx[~invalid_idx], hvac_input_temperature[~invalid_idx]) / total_hrs_month
        coldest_mean_temp = np.ma.masked_array(monthly_avg_temp,
                                               mask=(monthly_avg_temp == 0) | (total_hrs_month == 0)).min()
        coldest_month = np.nonzero(monthly_avg_temp <= coldest_mean_temp)[0][0]
        filtered_temp = hvac_input_temperature[month_idx == coldest_month]
        low_limit_mid_range, high_limit_mid_range = np.nanquantile(filtered_temp, 0.1), np.nanmedian(filtered_temp)
        mid_temp_idx = (hvac_input_consumption > 0) & (hvac_input_temperature >= low_limit_mid_range) & \
                       (hvac_input_temperature <= high_limit_mid_range) & (~invalid_idx.astype(bool))
    else:
        mid_temp_scope_idx = np.logical_and(~np.isnan(hvac_input_temperature),
                                            np.logical_and(hvac_input_temperature >=
                                                           hvac_params['detection']['MID_TEMPERATURE_RANGE'][0],
                                                           hvac_input_temperature <=
                                                           hvac_params['detection']['MID_TEMPERATURE_RANGE'][1]))
        mid_temp_scope = hvac_input_temperature[mid_temp_scope_idx]
        low_limit_mid_range = mquantiles(mid_temp_scope, hvac_params['detection']['MID_TEMPERATURE_QUANTILE'][0],
                                         alphap=0.5, betap=0.5)
        high_limit_mid_range = mquantiles(mid_temp_scope, hvac_params['detection']['MID_TEMPERATURE_QUANTILE'][1],
                                          alphap=0.5, betap=0.5)
        mid_temp_idx = (hvac_input_consumption > 0) & (hvac_input_temperature >= low_limit_mid_range) & \
                       (hvac_input_temperature <= high_limit_mid_range) & (~invalid_idx.astype(bool))
        mid_temp_idx = np.logical_and(mid_temp_idx, ~relative_summer_idx)

    result_temp_params = {
        'mid_temp_idx': mid_temp_idx,
        'mid_temp_range': [low_limit_mid_range, high_limit_mid_range],
        'estimation_setpoint_valid': valid_setpoint_range_estm,
        'detection_setpoint_valid': valid_setpoint_range_detn,
        'probable_ac_idx': cooling_idx,
        'probable_ac_lower_amp_night': probable_ac_amp_overnight,
        'correction_range_nightAC': correction_range,
        'correction': correction,
    }
    return result_temp_params


def get_hvac_consumption_type(seasonality_params, season_classification, all_indices, day_hours_boundary, disagg_mode,
                              previous_hsm, config):
    """
    Main Function to assign cooling usage flags to the user
    Args:
        seasonality_params      (dict)    : Dictionary storing all calculated cooling seasonality features
        season_classification   (dict)    : Dictionary containing all user temperature profile features
        all_indices             (dict)    : Dictionary containing all day/epoch/month/hour level indices
        day_hours_boundary      (tuple)   : Tuple with start, end hour of a day
        disagg_mode             (str)     : Current disaggregation mode - historical/incremental/mtd
        previous_hsm            (dict)    : Latest HSM available
        config                  (dict)    : Dictionary containing fixed parameters to calculate user characteristics
    Returns:
        result_consumption_flags(dict)    : Dictionary containing all user consumption type flags
    """
    # Unpack all input objects
    detn_logit_thresholds = config['seasonality_detection']['logit_coefficients']
    detn_tn_thresholds = config['seasonality_detection']['true_negative_ac']
    invalid_idx = all_indices['invalid_idx']
    hour_idx = all_indices['hour_idx']
    hot_cold_normal_user_flag = season_classification['hot_cold_normal_user_flag']
    cooling_regression_night = seasonality_params['cooling']['regression_result']['overnight']['coefficient'][0]
    cooling_cosine_night = seasonality_params['cooling']['cosine_similarity']['cosine_similarity_overnight']
    cooling_cosine_day = seasonality_params['cooling']['cosine_similarity']['cosine_similarity_day']
    cooling_prob_night = seasonality_params['cooling']['Detection']['hvac_prob']['overnight']['hvac_prob']
    cooling_prob_day = seasonality_params['cooling']['Detection']['hvac_prob']['day']['hvac_prob']
    similarity_summer_transition = seasonality_params['cooling']['Detection']['consumption_profile_change']
    propn_ac = seasonality_params['cooling']['proportion_cons']
    corr_day_ac = seasonality_params['cooling']['regression_result']['day']['corr']
    corr_night_ac = seasonality_params['cooling']['regression_result']['overnight']['corr']
    corr_low_cons_ac = seasonality_params['cooling']['regression_result']['low_consumption']['corr']
    is_low_ac_flag = previous_hsm.get('ac_low_consumption_user', 0)
    is_night_ac_flag = previous_hsm.get('night_ac_user', 0)
    is_disc_ac = season_classification['cyclic_temp_profile_flag']

    # Lower the threshold for incremental mode if the previous low cooling detection is true
    if (disagg_mode == 'incremental') & (is_low_ac_flag == 1):
        threshold = 0.45
    else:
        threshold = 0.50

    # Lower the threshold for incremental mode if the previous night cooling detection is true
    if (disagg_mode == 'incremental') & (is_night_ac_flag == 1):
        threshold_nightac = 0.45
        threshold_dayac = 0.5
    else:
        threshold_nightac = 0.50
        threshold_dayac = 0.5

    # Initialise variables with default values
    is_night_ac, is_day_ac, is_not_ac, is_low_ac = False, False, False, False
    exclude_detection_idx = np.zeros(hour_idx.shape).astype(bool)

    # Assign low ac, night ac, day ac and discontinous (disc) ac flags based on
    # factors calculated from logistic regression for each user temp type
    if hot_cold_normal_user_flag == config['user_temp_type_map']['hot']:
        coefficients = detn_logit_thresholds['hot_temp_type']
        logit = np.dot(coefficients['night_ac'], [cooling_cosine_night, cooling_prob_night, 1])
        is_night_ac = softmax_custom(logit) >= threshold_nightac
        logit = np.dot(coefficients['day_ac'],
                       [corr_day_ac, cooling_cosine_day, propn_ac, 1])
        is_day_ac = softmax_custom(logit) >= threshold_dayac
        logit = np.dot(coefficients['low_ac'], [corr_low_cons_ac, 1])
        is_low_ac = softmax_custom(logit) >= threshold

    elif hot_cold_normal_user_flag == config['user_temp_type_map']['normal']:
        coefficients = detn_logit_thresholds['normal_temp_type']
        logit = np.dot(coefficients['night_ac'],
                       [cooling_cosine_night, corr_night_ac, cooling_regression_night, cooling_prob_night, 1])
        is_night_ac = softmax_custom(logit) >= threshold_nightac
        logit = np.dot(coefficients['day_ac'], [corr_day_ac, cooling_prob_day, 1])
        is_day_ac = softmax_custom(logit) >= threshold_dayac
        logit = np.dot(coefficients['low_ac'], [corr_low_cons_ac, 1])
        is_low_ac = softmax_custom(logit) >= threshold

    elif hot_cold_normal_user_flag == config['user_temp_type_map']['cold']:
        coefficients = detn_logit_thresholds['cold_temp_type']
        logit = np.dot(coefficients['night_ac'], [cooling_cosine_night, corr_night_ac, 1])
        is_night_ac = softmax_custom(logit) >= threshold_nightac
        logit = np.dot(coefficients['day_ac'], [corr_day_ac, cooling_prob_day, 1])
        is_day_ac = softmax_custom(logit) >= threshold

    # Assign not ac = 1 to suppress detection If night ac and day ac and low ac are all zero
    # and the similarity of consumption profile between summer/transition is high
    if (not is_night_ac) and (not is_day_ac) and (is_disc_ac == 0) and (is_low_ac == 0) and (
            similarity_summer_transition > 0):
        is_not_ac = (similarity_summer_transition > detn_tn_thresholds['cosine_similarity_thresh']) & \
                    (np.max([cooling_prob_day, cooling_prob_night]) < detn_tn_thresholds['max_prob_ac_thresh'])

    # For amplitude estimation, exclude day / night segments
    if is_night_ac and not is_day_ac:
        exclude_detection_idx = np.logical_and(~invalid_idx,
                                               np.logical_and(hour_idx >= day_hours_boundary[0],
                                                              hour_idx <= day_hours_boundary[1]))
    elif is_day_ac and not is_night_ac:
        exclude_detection_idx = np.logical_and(~invalid_idx,
                                               np.logical_or(hour_idx < day_hours_boundary[0],
                                                             hour_idx > day_hours_boundary[1]))
    result_consumption_flags = {
        'is_night_ac': is_night_ac,
        'is_day_ac': is_day_ac,
        'is_disc_ac': is_disc_ac,
        'is_not_ac': is_not_ac,
        'is_low_ac': is_low_ac,
        'exclude_detection_idx': exclude_detection_idx
    }
    return result_consumption_flags


def get_user_characteristic(hvac_input_data, invalid_idx, config, hvac_params, disagg_mode,
                            previous_hsm, logger_base):
    """
    Wrapper function to extract all user consumption and temperature features
    Args:
        hvac_input_data             (np.ndarray)    : 2D array of epoch level consumption and temperature data
        invalid_idx                 (np.ndarray)    : Array of invalid epochs based on consumption and temperature
        config                      (dict)          : Dictionary with fixed parameters to calculate user characteristics
        hvac_params                 (dict)          : Dictionary containing hvac algo related initialized parameters
        disagg_mode                 (str)           : String identifier for historical / incremental / mtd mode
        logger_base                 (logger)        : Writes logs during code flow
    Returns:
        result_pre_detection_params  (dict)         : Dictionary containing all user extracted labels and features before detection
    """
    # Intiialise logger object
    logger_local = logger_base.get("logger").getChild("fetch_pre_pipeline_user_characteristics")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # Make a copy of the main input consumption,temperature data and store all indices to be used
    hvac_input_data = copy.deepcopy(hvac_input_data)
    all_indices = get_all_indices(hvac_input_data, invalid_idx, config)
    invalid_idx = all_indices['invalid_idx']
    hvac_input_temperature = hvac_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX].squeeze()
    hvac_input_consumption = hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX].squeeze()
    hvac_input_temperature[invalid_idx] = np.nan
    hvac_input_consumption[invalid_idx] = 0

    # Analyse user temperature profile
    season_classification = classify_user_temp_profile(hvac_input_temperature, all_indices, config)

    # Get consumption histogram centers from hvac input consumption array
    quan_005 = np.around(super_percentile(hvac_input_consumption, 0.5), 5)
    quan_995 = np.around(super_percentile(hvac_input_consumption, 99.5), 5)
    qualified_epoch_idx = np.logical_and(hvac_input_consumption >= quan_005, hvac_input_consumption <= quan_995)
    _, raw_hist_edges = np.histogram(hvac_input_consumption[qualified_epoch_idx > 0].squeeze())
    raw_hist_centers = np.r_[0.5 * (raw_hist_edges[:-1] + raw_hist_edges[1:])]

    # Calculate Consumption and Temperature mean hourly aggregated profiles over summer and winter days
    season_month_idx_map = season_classification['season_month_idx_map']
    mean_day_cons_summer, mean_day_temp_summer = get_mean_hourly_profile(hvac_input_consumption, hvac_input_temperature,
                                                                         season_month_idx_map, all_indices, config,
                                                                         'summer')
    mean_day_cons_winter, mean_day_temp_winter = get_mean_hourly_profile(hvac_input_consumption, hvac_input_temperature,
                                                                         season_month_idx_map, all_indices, config,
                                                                         'winter')

    # Update objects in all_indices which is the main dictionary for all index maps
    all_indices.update({'season_month_idx_map': season_classification['season_month_idx_map']})
    all_indices.update({'winter_start_end': season_classification['winter_start_end']})
    all_indices.update({'summer_start_end': season_classification['summer_start_end']})
    all_indices.update({'summer_start_end': season_classification['summer_start_end']})

    # Get User Amplitude Consumption characteristics to identify low consumption users
    low_summer_consumption_user_flag, consumption_range_params = classify_user_cons_profile(
        hvac_input_consumption, invalid_idx,
        season_classification['season_month_idx_map'],
        config)
    logger_hvac.info('Season and consumption classification complete, user temperature type : | {} |'
                     .format(int(season_classification['hot_cold_normal_user_flag'])))

    # Store all common objects to be used in the functions below
    common_objects = {
        'summer': {
            'mean_day_cons': mean_day_cons_summer,
            'mean_day_temp': mean_day_temp_summer,
            'day_hours': config['day_hours_boundary'],
            'low_consumption_user_flag': low_summer_consumption_user_flag,
            'hot_cold_normal_user_flag': season_classification['hot_cold_normal_user_flag'],
        },
        'winter': {
            'mean_day_cons': mean_day_cons_winter,
            'mean_day_temp': mean_day_temp_winter,
        }
    }

    # Calculate cooling seasonality parameters
    seasonality_params_ac = get_cooling_seasonality_params(hvac_input_consumption, hvac_input_temperature,
                                                           raw_hist_centers,
                                                           all_indices, common_objects['summer'], config)

    seasonality_params_overall = {'cooling': seasonality_params_ac, 'heating': {}}

    # Assign flags for user consumption type
    cooling_consumption_type = get_hvac_consumption_type(seasonality_params_overall, season_classification, all_indices,
                                                         config['day_hours_boundary'], disagg_mode, previous_hsm,
                                                         config)

    logger_hvac.info('Seasonality classification complete | ')

    # Get adjusted setpoint, mid-temp and other temperature related parameters based on consumption type
    cooling_temperature_params_candidates = get_adjusted_cooling_temperature_range(hvac_input_temperature,
                                                                                   hvac_input_consumption,
                                                                                   seasonality_params_ac,
                                                                                   cooling_consumption_type,
                                                                                   season_classification[
                                                                                       'hot_cold_normal_user_flag'],
                                                                                   config['day_hours_boundary'],
                                                                                   all_indices,
                                                                                   config,
                                                                                   hvac_params)

    all_flags = {
        'hot_cold_normal_user_flag': season_classification['hot_cold_normal_user_flag'],
        'low_summer_consumption_user_flag': low_summer_consumption_user_flag,
        'is_night_ac': cooling_consumption_type['is_night_ac'],
        'is_day_ac': cooling_consumption_type['is_day_ac'],
        'is_disc_ac': cooling_consumption_type['is_disc_ac'],
        'is_not_ac': cooling_consumption_type['is_not_ac'],
        'is_low_ac': cooling_consumption_type['is_low_ac'],
        'min_days_condition_season_classn': season_classification['min_days_condition']
    }

    result_pre_detection_params = {
        'all_indices': all_indices,
        'all_flags': all_flags,
        'boundary': config['day_hours_boundary'],
        'consumption_parameters': {'raw_hist_centers': raw_hist_centers,
                                   'range_params': consumption_range_params},
        'cooling': {
            'seasonality_params': seasonality_params_ac,
            'cooling_temperature_params_candidates': cooling_temperature_params_candidates,
            'exclude_detection_idx': cooling_consumption_type['exclude_detection_idx']
        },
    }
    return result_pre_detection_params
