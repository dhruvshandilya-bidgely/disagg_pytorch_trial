"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Aggregate monthly level features to user level features
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.disaggregation.aer.waterheater.functions.thermostat.functions.thermostat_features import WhFeatures
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.get_bill_cycle_features import season_columns


def get_user_features(debug, monthly_features, wh_config):
    """
    Parameters:
        debug               (dict)          : Algorithm intermediate steps output
        monthly_features    (np.ndarray)    : Monthly level features
        wh_config           (dict)          : Water heater params

    Returns:
        user_features       (dict)          : User level features
    """

    # Get the base number of features

    n = WhFeatures.n_base

    # Take deepcopy of features column index to use locally

    features = dict(WhFeatures.__dict__)

    # Updating the column indices used in the features array

    for key, value in features.items():
        if '__' not in key:
            features[key] = value + n

    # Retrieve season info from config

    season_info = wh_config['timed_wh']['season_code']

    # Get default feature value from config

    default_feature_value = wh_config['thermostat_wh']['estimation']['default_feature_value']

    # Initialize the user features dictionary

    user_features = {}

    ##-------------------------------------- Season dependent user features -------------------------------------------#

    # Get seasonal features for each season

    for season in season_info.keys():
        # Subset the monthly features of a particular season

        season_features = monthly_features[monthly_features[:, season_columns['season_id']] == season_info[season], :]

        # If monthly features present for the season, calculate aggregated features

        if season_features.shape[0] > 0:

            wh_laps = season_features[season_features[:, features['COUNT_PEAKS']] > 0, features['PEAKS_PER_LAP']]
            user_features[season + '_wh_laps'] = np.round(np.mean(wh_laps), 3)

            valid_lap_days = season_features[season_features[:, features['VALID_LAP_DAYS']] > 0,
                                             features['VALID_LAP_DAYS']]
            user_features[season + '_valid_lap_days'] = np.round(np.mean(valid_lap_days), 3)

            user_features[season + '_peaks_count'] = np.round(
                np.sum(season_features[:, features['COUNT_PEAKS']]) / season_features.shape[0], 2)
            user_features[season + '_peaks_count_night'] = np.round(
                (np.sum(season_features[:, features['NIGHT_PEAKS']]) / season_features.shape[0]) / \
                user_features[season + '_peaks_count'], 2)
            user_features[season + '_peaks_count_day'] = np.round(
                (np.sum(season_features[:, features['DAY_PEAKS']]) / season_features.shape[0]) / \
                user_features[season + '_peaks_count'], 2)
            user_features[season + '_two_peak_lap_count'] = np.mean(season_features[:, features['TWO_PEAKS_LAP_COUNT']])

            user_features[season + '_laps_count'] = np.round(
                np.sum(season_features[:, features['COUNT_LAPS']]) / season_features.shape[0], 2)

            user_features[season + '_peak_factor'] = np.round(np.mean(season_features[:, features['PEAK_FACTOR']]), 3)

            consistency = season_features[season_features[:, features['CONSISTENCY']] > 0, features['CONSISTENCY']]
            user_features[season + '_consistency'] = np.round(np.mean(consistency), 3)

            wh_days = season_features[season_features[:, features['PEAK_DAYS_PER_MONTH']] > 0,
                                      features['PEAK_DAYS_PER_MONTH']]
            user_features[season + '_wh_days'] = np.round(np.mean(wh_days), 3)

        else:
            # If monthly features doesn't exist for the season, assign default values

            user_features[season + '_wh_laps'] = default_feature_value

            user_features[season + '_valid_lap_days'] = default_feature_value

            user_features[season + '_peaks_count'] = default_feature_value
            user_features[season + '_peaks_count_night'] = default_feature_value
            user_features[season + '_peaks_count_day'] = default_feature_value
            user_features[season + '_two_peak_lap_count'] = default_feature_value

            user_features[season + '_laps_count'] = default_feature_value

            user_features[season + '_peak_factor'] = default_feature_value

            user_features[season + '_consistency'] = default_feature_value

            user_features[season + '_wh_days'] = default_feature_value

    ##-------------------------------------- Season independent user features -----------------------------------------#

    # Get the user level features

    user_features['energy_mean'] = np.round(
        np.mean(monthly_features[monthly_features[:, features['ENERGY']] > 0, features['ENERGY']]), 3)

    user_features['energy_std'] = np.round(
        np.mean(monthly_features[monthly_features[:, features['ENERGY_STD']] > 0, features['ENERGY_STD']]), 3)

    user_features['all_wh_laps'] = np.round(
        np.mean(monthly_features[monthly_features[:, features['COUNT_PEAKS']] > 0, features['PEAKS_PER_LAP']]), 3)

    user_features['std_percent'] = user_features['energy_std'] / user_features['energy_mean']
    user_features['std_percent'] = np.round(user_features['std_percent'], 3) if user_features['std_percent'] > 0 else 1

    peak_factor_std = monthly_features[monthly_features[:, features['PEAK_FACTOR_STD']] > 0,
                                       features['PEAK_FACTOR_STD']]

    user_features['all_peak_factor_std'] = np.round(np.mean(peak_factor_std), 3)
    if np.isnan(user_features['all_peak_factor_std']):
        user_features['all_peak_factor_std'] = 10

    monthly_valid_lap_days = monthly_features[monthly_features[:, features['VALID_LAP_DAYS']] > 0,
                                              features['VALID_LAP_DAYS']]
    user_features['all_valid_lap_days'] = np.round(np.mean(monthly_valid_lap_days), 3)

    # Handled during data filling

    if np.isnan(user_features['all_valid_lap_days']):
        user_features['all_valid_lap_days'] = default_feature_value

    user_features['max_valid_lap_days'] = np.nanmax([user_features['wtr_valid_lap_days'],
                                                     user_features['itr_valid_lap_days'],
                                                     user_features['smr_valid_lap_days']])

    user_features['all_peaks_count'] = np.round(
        np.sum(monthly_features[:, features['COUNT_PEAKS']]) / monthly_features.shape[0], 2)
    user_features['all_peaks_count_night'] = np.round((np.sum(monthly_features[:, features['NIGHT_PEAKS']]) /
                                                       monthly_features.shape[0]) / user_features['all_peaks_count'], 2)
    user_features['all_peaks_count_day'] = np.round((np.sum(monthly_features[:, features['DAY_PEAKS']]) /
                                                     monthly_features.shape[0]) / user_features['all_peaks_count'], 2)

    user_features['max_peaks_count'] = np.nanmax([user_features['wtr_peaks_count'],
                                                  user_features['itr_peaks_count'],
                                                  user_features['smr_peaks_count']])

    user_features['all_laps_count'] = np.round(
        np.sum(monthly_features[:, features['COUNT_LAPS']]) / monthly_features.shape[0], 2)

    user_features['max_laps_count'] = np.nanmax([user_features['wtr_laps_count'],
                                                 user_features['itr_laps_count'],
                                                 user_features['smr_laps_count']])

    user_features['all_peak_factor'] = np.round(np.mean(monthly_features[:, features['PEAK_FACTOR']]), 3)

    all_consistency = monthly_features[monthly_features[:, features['CONSISTENCY']] > 0, features['CONSISTENCY']]
    user_features['all_consistency'] = np.round(np.mean(all_consistency), 3)

    all_wh_days = monthly_features[monthly_features[:, features['PEAK_DAYS_PER_MONTH']] > 0,
                                   features['PEAK_DAYS_PER_MONTH']]
    user_features['all_wh_days'] = np.round(np.mean(all_wh_days), 3)

    user_features['max_wh_days'] = np.nanmax([user_features['wtr_wh_days'],
                                              user_features['itr_wh_days'],
                                              user_features['smr_wh_days']])

    debug['user_detection_features'] = user_features

    return user_features, debug
