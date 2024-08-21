"""
Author - Mirambika Sikdar
Date - 06/12/2023
Utility functions required in seasonality calculation are defined here
"""

# Import python packages
import copy
import numpy as np
from sklearn.linear_model import LinearRegression

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg


def get_similarity_consumption_profile_transition(hvac_input_consumption, all_indices, hot_cold_normal_user_flag,
                                                  appliance, config):
    """
    Function to calculate probability of hvac usage in consumption over specified hours in summer/winter days
    Args:
        hvac_input_consumption      (np.ndarray)   : Array of epoch level consumption flowing into hvac module
        all_indices                 (dict)         : Dictionary containing all day/epoch/month/hour level indices
        hot_cold_normal_user_flag   (int)          : Integer label to indicate user temperature profile type
        appliance                   (str)          : String identifier for AC/SH
        config                      (dict)         : Dictionary containing fixed parameters to calculate user characteristics

    Returns:
        cosine_sim                  (float)        : cosine similarity of avg consumption profile of summer/winter
                                                     and transition days
    """
    # Unpack input objects
    hvac_input_consumption = copy.deepcopy(hvac_input_consumption)
    season_month_idx_map = all_indices['season_month_idx_map']
    hour_idx = all_indices['hour_idx']
    hvac_input_consumption_std = (hvac_input_consumption - np.nanmean(hvac_input_consumption)) / np.nanstd(
        hvac_input_consumption)
    transition = 'transition'

    # Check whether AC or SH to assign peak season label
    if appliance == 'AC':
        season_label = 'summer'
    else:
        season_label = 'winter'

    # Since there might be no winter day label in hot temp profile users
    # relative winter would be transition and baseline would be summer
    if (hot_cold_normal_user_flag == config['user_temp_type_map']['hot']) and (appliance == 'SH'):
        transition = 'summer'
        season_label = 'transition'

    # Since there might be no summer day label in cold temp profile users
    # relative summer would be transition and baseline would be winter
    if (hot_cold_normal_user_flag == config['user_temp_type_map']['cold']) and (appliance == 'AC'):
        transition = 'winter'
        season_label = 'transition'

    # Calculate 24-hour average vector of consumption over all baseline season days
    transition_cons_profile = np.bincount(hour_idx[season_month_idx_map == config['season_idx_map'][transition]],
                                          hvac_input_consumption_std[
                                              season_month_idx_map == config['season_idx_map'][transition]])
    # Initialise variable
    seasonal_cons_profile = np.zeros(transition_cons_profile.shape)

    # If season label days are present
    if (np.sum(season_month_idx_map == config['season_idx_map'][season_label]) > 0) and \
            (np.sum(season_month_idx_map == config['season_idx_map'][transition]) > 0):
        # Calculate 24-hour average vector of consumption over all relative summer/winter season days
        seasonal_cons_profile = np.bincount(hour_idx[season_month_idx_map == config['season_idx_map'][season_label]],
                                            hvac_input_consumption_std[
                                                season_month_idx_map == config['season_idx_map'][season_label]])

    # Calculate cosine similarity between baseline and season days
    cosine_sim = float(np.sum(np.multiply(seasonal_cons_profile, transition_cons_profile)) / (np.linalg.norm(seasonal_cons_profile) * np.linalg.norm(transition_cons_profile)))

    # Don't draw any conclusion in case of nan or inf cosine similarity
    if np.isnan(cosine_sim) | np.isinf(cosine_sim):
        cosine_sim = -1.0

    return cosine_sim


def get_hvac_consumption_prob_params(hvac_input_consumption, hist_centers_consumption, valid_idx, all_indices, config,
                                     low_consumption_user_flag, hot_cold_normal_user_flag, season_label):
    """
    Function to calculate probability of AC/SH usage in consumption over specified hours in summer/winter days
    Args:
        hvac_input_consumption      (np.ndarray)    : Array of epoch level consumption flowing into hvac module
        hist_centers_consumption    (np.ndarray)    : Array of histogram centers from overall consumption data
        valid_idx                   (np.ndarray)    : Array of boolean values to mark valid hours window/day segments
        all_indices                 (dict)          : Dictionary containing all day/epoch/month/hour level indices
        config                      (dict)          : Dictionary with fixed parameters to calculate user characteristics
        low_consumption_user_flag   (int)           : Integer to flag low cooling/heating consumption user
        hot_cold_normal_user_flag   (int)           : Integer label to indicate user temperature profile type
        season_label                (str)           : String identifier for summer/winter associated with AC/SH

    Returns:
        result                      (dict)          : Dictionary containing cooling/heating probabilities in the given hour segment
    """
    # Check current appliance
    appliance = 'AC'

    # Unpack input objects and initialise variables
    season_month_idx_map = all_indices['season_month_idx_map']
    summer_cons_hist, transition_cons_hist, winter_cons_hist = np.zeros(hist_centers_consumption.shape), np.zeros(
        hist_centers_consumption.shape), np.zeros(hist_centers_consumption.shape)
    diff_hist_1, diff_hist_2, diff_hist_3 = np.zeros(hist_centers_consumption.shape), np.zeros(
        hist_centers_consumption.shape), np.zeros(hist_centers_consumption.shape)

    # For probable low consumption ac users, adjust hvac min amplitude to 0
    if not low_consumption_user_flag:
        hvac_min_amplitude = config['MINIMUM_AMPLITUDE'][appliance]
    else:
        hvac_min_amplitude = 0

    # Minimum index above which we consider cooling to be present
    index_min = np.nonzero(hist_centers_consumption >= hvac_min_amplitude)[0]
    if len(index_min) > 0:
        index_min = min(index_min)
    else:
        index_min = 0

    # Get the epoch consumption values associated with valid summer season days
    selected_summer_idx = np.logical_and(valid_idx, season_month_idx_map == config['season_idx_map'][season_label])
    summer_cons = hvac_input_consumption[selected_summer_idx > 0]

    # Get the epoch consumption values associated with valid transition season days
    selected_transition_idx = np.logical_and(valid_idx, season_month_idx_map == config['season_idx_map']['transition'])
    transition_cons = hvac_input_consumption[selected_transition_idx > 0]

    # Get the epoch consumption values associated with valid winter season days
    selected_winter_idx = np.logical_and(valid_idx, season_month_idx_map == config['season_idx_map']['winter'])
    winter_cons = hvac_input_consumption[selected_winter_idx > 0]

    # Calculate the consumption probability histogram if the given season days are present
    # Do not consider probability below a given minimum amplitude
    if len(summer_cons) > 0:
        summer_cons_hist = get_hist_bincount(summer_cons, hist_centers_consumption)
        summer_cons_hist[:index_min + 1] = 0
    if len(transition_cons) > 0:
        transition_cons_hist = get_hist_bincount(transition_cons, hist_centers_consumption)
        transition_cons_hist[:index_min + 1] = 0
    if len(winter_cons) > 0:
        winter_cons_hist = get_hist_bincount(winter_cons, hist_centers_consumption)
        winter_cons_hist[:index_min + 1] = 0

    # If season days are present, calculate the difference in probability histograms
    # If summer label is not present, consider transition as relative summer and baseline
    # as winter for cooling probabilty calculation
    if np.sum(season_month_idx_map == config['season_idx_map']['summer']) == 0:
        diff_hist_2 = transition_cons_hist - winter_cons_hist
        diff_hist_2[diff_hist_2 < 0] = 0

    # If winter label is not present, consider transition as relative winter and baseline
    # as summer for heating probabilty calculation
    elif np.sum(season_month_idx_map == config['season_idx_map']['winter']) == 0:
        diff_hist_1 = summer_cons_hist - transition_cons_hist
        diff_hist_1[diff_hist_1 < 0] = 0

    # If all season day labels are present, calculate all candidate probability histograms
    else:
        diff_hist_1 = summer_cons_hist - transition_cons_hist
        diff_hist_1[diff_hist_1 < 0] = 0
        # Consider transition - winter for cooling detection for only cold temp types
        if hot_cold_normal_user_flag == config['user_temp_type_map']['cold']:
            diff_hist_2 = transition_cons_hist - winter_cons_hist
            diff_hist_2[diff_hist_2 < 0] = 0
        diff_hist_3 = summer_cons_hist - winter_cons_hist
        diff_hist_3[diff_hist_3 < 0] = 0

    # Identify the difference histogram that maximises cooling/heating consumption probability
    list_diff_hists = [diff_hist_1, diff_hist_2, diff_hist_3]
    hvac_prob_candidates = np.array([np.sum(diff_hist_1), np.sum(diff_hist_2), np.sum(diff_hist_3)])
    max_diff_hist = list_diff_hists[np.argmax(hvac_prob_candidates)]
    hvac_prob = hvac_prob_candidates[np.argmax(hvac_prob_candidates)]

    # Calculate probable hvac amplitude as an estimation for a lower threshold value
    if np.sum(max_diff_hist) > 0:
        max_diff_hist = max_diff_hist / np.sum(max_diff_hist)
        probable_hvac_amplitude = np.sum(hist_centers_consumption * max_diff_hist)
        probable_hvac_amplitude_idx = np.nonzero(hist_centers_consumption >= probable_hvac_amplitude)[0][0]
        probable_hvac_amplitude_wt = np.sum(max_diff_hist[probable_hvac_amplitude_idx:])
    else:
        probable_hvac_amplitude = np.nan
        probable_hvac_amplitude_wt = 0

    result = {
        'max_diff_hist': max_diff_hist,
        'hvac_prob': hvac_prob,
        'probable_hvac_amp': probable_hvac_amplitude,
        'probable_hvac_amplitude_wt': probable_hvac_amplitude_wt
    }
    return result


def get_seasonality_regression_params(hvac_input_consumption, hvac_input_temperature, valid_idx, all_indices,
                                      season_start_end, config, season_label, valid_range):
    """
    Function to calculate regression parameters to measure seasonality in consumption in peak summer/winter days
    Args:
        hvac_input_consumption (np.ndarray)    : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature (np.ndarray)    : Array of epoch level temperature flowing into hvac module
        valid_idx              (np.ndarray)    : Array of boolean values to mark valid hours window/day segment
        all_indices            (dict)          : Dictionary containing all day/epoch/month/hour level indices
        season_start_end       (tuple)         : Tuple with start and end day indices of peak summer/winter season
        config                 (dict)          : Dictionary containing fixed parameters to calculate user characteristics
        season_label           (str)           : String identifier for summer/winter associated with ac/sh
        valid_range            (list)          : Valid range of cooling/heating consumption to check for seasonality

    Returns:
        result                 (dict)          : Dictionary containing the coefficients, covariance and r2 parameters from regression
    """
    # Check current appliance
    if season_label == 'summer':
        appliance = 'AC'
    else:
        appliance = 'SH'

    # Unpack all input objects and initialise variables
    day_idx = all_indices['day_idx'][valid_idx == 1]
    hvac_input_consumption = copy.deepcopy(hvac_input_consumption)[valid_idx == 1]
    hvac_input_consumption = np.where(
        (hvac_input_consumption < valid_range[0]) | (hvac_input_consumption > valid_range[1]), 0,
        hvac_input_consumption)
    hvac_input_temperature = copy.deepcopy(hvac_input_temperature)[valid_idx == 1]
    setpoint = config['Detection_Setpoint'].get(appliance)
    coefficient, valid_days, corr = [-10000], 0, 0

    # Calculate cdd / hdd for regression
    if appliance == 'AC':
        hvac_input_temperature = hvac_input_temperature - setpoint
    else:
        hvac_input_temperature = setpoint - hvac_input_temperature
    hvac_input_temperature[hvac_input_temperature < 0] = 0

    # calculate mean daily consumption and cdd/hdd vector averaged over all hours in a day
    mean_cons_vector_all = np.bincount(day_idx, hvac_input_consumption) / np.bincount(day_idx)
    mean_temp_vector_all = np.bincount(day_idx, hvac_input_temperature) / np.bincount(day_idx,
                                                                                      hvac_input_temperature != 0)

    # Mask the days outside of season start and end so that they don't enter regression
    if season_start_end[0] <= season_start_end[1]:
        mask = np.zeros(mean_cons_vector_all.shape, bool)
        mask[season_start_end[0]:season_start_end[1] + 1] = True
        mean_cons_vector_all = mean_cons_vector_all[mask]
        mean_temp_vector_all = mean_temp_vector_all[mask]
    else:
        mask = np.ones(mean_cons_vector_all.shape, bool)
        mask[season_start_end[1]:season_start_end[0]] = False
        mean_cons_vector_all = mean_cons_vector_all[mask]
        mean_temp_vector_all = mean_temp_vector_all[mask]

    # Standardize averaged daily consumption and cdd/hdd vector
    std_dev_arm_valid_days = 2
    mean_cons_vector_all_std = (mean_cons_vector_all - np.nanmean(mean_cons_vector_all)) / np.nanstd(
        mean_cons_vector_all)
    mean_temp_vector_all_std = (mean_temp_vector_all - np.nanmean(mean_temp_vector_all)) / np.nanstd(
        mean_temp_vector_all)

    # Identify the valid days where consumption/temperature is not nan/inf after standardization
    # or where consumption/temperature is outside of 2 standard deviations
    valid_days = np.where(np.isnan(mean_cons_vector_all_std) | np.isinf(mean_cons_vector_all_std) |
                          np.isnan(mean_temp_vector_all_std) | np.isinf(mean_temp_vector_all_std) |
                          (mean_temp_vector_all <= 0) | (mean_cons_vector_all <= 0) |
                          (np.abs(mean_cons_vector_all - np.nanmean(mean_cons_vector_all)) > std_dev_arm_valid_days * np.nanstd(
                              mean_cons_vector_all)) &
                          (np.abs(mean_temp_vector_all - np.nanmean(mean_temp_vector_all)) > std_dev_arm_valid_days * np.nanstd(
                              mean_temp_vector_all)), 0, 1)

    # Reduce the weight of invalid days in regression to zero
    mean_cons_vector_all_std[valid_days == 0] = 0
    mean_temp_vector_all_std[valid_days == 0] = 0

    # If minimum number of days are present then implement regression
    if np.sum(valid_days) > config['seasonality_detection']['regression_seasonality']['min_points']:
        x_array = np.zeros((len(mean_temp_vector_all_std[valid_days == 1]), 1))
        x_array[:, 0] = mean_temp_vector_all_std[valid_days == 1]
        y_array = mean_cons_vector_all_std[valid_days == 1].reshape(-1, 1)

        regression = LinearRegression().fit(x_array, y_array)
        coefficient = regression.coef_.squeeze()
        corr = float(np.corrcoef(mean_temp_vector_all[valid_days == 1], mean_cons_vector_all[valid_days == 1])[0, 1])

    result = {
        'coefficient': np.array([coefficient]),
        'corr': np.array([corr]),
        'valid_days': np.sum(valid_days)
    }

    return result


def get_proportion_consumption(hvac_input_consumption, mean_day_cons, day_hours_boundary, all_indices):
    """
    Function to calculate proportion of mean consumption between overnight hours and day hours over summer/winter days
    Args:
        hvac_input_consumption      (np.ndarray) : Array of epoch level consumption flowing into hvac module
        mean_day_cons               (np.ndarray) : Array that is the mean hourly consumption vector over summer/winter days
        day_hours_boundary          (tuple)      : Tuple with start, end hour of a day
        all_indices                 (dict)       : Dictionary containing all day/epoch/month/hour level indices

    Returns:
        proportion_cons             (float)      : Ratio of consumption of overnight and day hours over summer/winter days

    """
    hour_idx = all_indices['hour_idx']
    invalid_idx = all_indices['invalid_idx']
    valid_idx_day = np.logical_and(~invalid_idx,
                                   np.logical_and(hour_idx >= day_hours_boundary[0], hour_idx <= day_hours_boundary[1]))
    valid_idx_overnight = np.logical_and(~invalid_idx,
                                         np.logical_or(hour_idx < day_hours_boundary[0],
                                                       hour_idx > day_hours_boundary[1]))

    # The mean hourly consumption vector should ideally be 24 hours in length
    if len(mean_day_cons) == Cgbdisagg.HRS_IN_DAY:
        night_index = np.r_[0: day_hours_boundary[0], day_hours_boundary[1] + 1: len(mean_day_cons)]
        proportion_cons = np.mean(mean_day_cons[night_index]) / np.mean(
            mean_day_cons[day_hours_boundary[0]:day_hours_boundary[1] + 1])
    else:
        proportion_cons = np.mean(hvac_input_consumption[valid_idx_overnight == 1]) / np.mean(
            hvac_input_consumption[valid_idx_day == 1])

    # Do not draw any conclusion if the output is an invalid value
    if np.isnan(proportion_cons):
        proportion_cons = 0

    return proportion_cons


def get_profiles_diff(hvac_input_consumption, hvac_input_temperature, selected_idx, all_indices, season, valid_range,
                      config):
    """
    Function to calculate cosine similarity between mean consumption and temperature vector over summer/winter days
    Args:
        hvac_input_consumption  (np.ndarray)     : Array of epoch level consumption flowing into hvac module
        hvac_input_temperature  (np.ndarray)     : Array of epoch level temperature flowing into hvac module
        selected_idx            (np.ndarray)     : Array of boolean values to mark valid hours window/day segment
        all_indices             (dict)           : Dictionary containing all day/epoch/month/hour level indices
        season                  (str)            : Identifier for summer / winter season
        valid_range             (list)           : Valid range of cooling/heating consumption to check for seasonality
        config                  (dict)           : Config parameters
    Returns:
        cosine_sim              (float)          : cosine similarity between consumption and cdd/hdd profiles
    """

    # Unpack all input objects
    day_idx = all_indices['day_idx'][selected_idx == 1]
    hvac_input_consumption = np.where(
        (hvac_input_consumption >= valid_range[0]) & (hvac_input_consumption <= valid_range[1]), hvac_input_consumption,
        0)
    hvac_input_consumption = hvac_input_consumption[selected_idx == 1]
    hvac_input_temperature = hvac_input_temperature[selected_idx == 1]
    if season == 'summer':
        detection_setpoint = config['Detection_Setpoint']['AC']
        hvac_input_temperature = hvac_input_temperature - detection_setpoint
    if season == 'winter':
        detection_setpoint = config['Detection_Setpoint']['SH']
        hvac_input_temperature = detection_setpoint - hvac_input_temperature
    hvac_input_temperature[hvac_input_temperature < 0] = 0

    mean_day_cons = np.bincount(day_idx, hvac_input_consumption) / np.bincount(day_idx)
    mean_day_temp = np.bincount(day_idx, hvac_input_temperature) / np.bincount(day_idx)
    mean_day_cons = (mean_day_cons - np.nanmean(mean_day_cons)) / (np.nanstd(mean_day_cons))
    mean_day_temp = (mean_day_temp - np.nanmean(mean_day_temp)) / (np.nanstd(mean_day_temp))
    remove_idx = np.where(np.isnan(mean_day_cons) | np.isinf(mean_day_cons) |
                          np.isnan(mean_day_temp) | np.isinf(mean_day_temp), 1, 0)
    mean_day_cons[remove_idx == 1] = 0
    mean_day_temp[remove_idx == 1] = 0

    cosine_sim = float(np.sum(np.multiply(mean_day_cons, mean_day_temp)) /
                       (np.linalg.norm(mean_day_cons) * np.linalg.norm(mean_day_temp)))

    if np.isnan(cosine_sim) | np.isinf(cosine_sim):
        cosine_sim = 0

    return cosine_sim


# General Utility functions
def get_hist_bincount(data, hist_centers):
    """
    General utility function to return histogram bin count
    Args:
        data         (np.ndarray)   : Array containing the values to be binned
        hist_centers (np.ndarray)   : Array containing values of bin centers for histogram

    Returns:
        bin_counts   (np.ndarray)   : Normalized histogram bin counts for input data
    """
    bin_counts, _ = np.histogram(data, bins=hist_centers)
    bin_counts = np.r_[bin_counts, 0]
    if np.sum(bin_counts) > 0:
        bin_counts = bin_counts / np.sum(bin_counts)
    return bin_counts
