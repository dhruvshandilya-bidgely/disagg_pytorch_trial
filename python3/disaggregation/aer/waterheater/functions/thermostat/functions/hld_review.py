"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Review the water heater detection if the probability is in the buffer zone
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def hld_review(features, hld_probability, wh_config, debug, logger):
    """
    Parameters:
        features                (dict)          : User features
        hld_probability         (float)         : Detection probability
        wh_config               (dict)          : Config params
        debug                   (dict)          : Algorithm intermediate steps output
        logger                  (logger)        : Logger object

    Returns:
        hld                     (int)           : Final detection
        debug                   (dict)          : Algorithm intermediate steps output
    """

    # Extract the lower and upper probability bounds for conflict zone

    zero_threshold = wh_config['thermostat_wh']['detection']['probability_thresholds'][0]
    one_threshold = wh_config['thermostat_wh']['detection']['probability_thresholds'][1]

    # Middle probability from the bounds of conflict zone

    border_threshold = np.round((zero_threshold + one_threshold) / 2, 3)

    logger.info('Home level detection probability thresholds: | [{}, {}]'.format(zero_threshold, one_threshold))

    # Give final home level detection

    if hld_probability >= one_threshold:
        # If probability more than upper threshold

        hld = 1
    elif hld_probability <= zero_threshold:
        # If probability less than lower threshold

        hld = 0
    else:
        # If probability in conflict zone, thin pulse check

        hld, debug = thin_pulse_hourly_check(features, debug, hld_probability, border_threshold, wh_config, logger)

    return hld, debug


def thin_pulse_hourly_check(features, debug, hld_probability, border_threshold, wh_config, logger):
    """
    Parameters:
        features                (dict)      : User features
        debug                   (dict)      : Algorithm intermediate steps output
        hld_probability         (float)     : Detection probability
        border_threshold        (float)     : Middle probability
        wh_config               (dict)      : Config params
        logger                  (logger)    : Logger object

    Returns:
        hld                     (int)       : Final detection
        debug                   (dict)      : Algorithm intermediate steps output
    """

    # Extract seasonal features from debug object

    season_features = debug['season_features']

    # Sampling rate factor

    factor = Cgbdisagg.SEC_IN_HOUR / wh_config['sampling_rate']

    # Getting config values required for checks

    peak_fraction = wh_config['thermostat_wh']['detection']['peak_fraction']
    peak_lap_ratio = wh_config['thermostat_wh']['detection']['peak_lap_ratio']

    # Initialize features to review with respect to detection

    review_features = {}

    # Iterate over each season

    for season in list(wh_config['season_code'].values()):

        # Get data of the current season

        season_data = season_features[season]['data']

        # Check if valid data available

        if len(season_data) > 0:
            # Get thin pulse indices of the current season

            single_peak_idx = season_features[season]['lap_peaks']
            lap_idx = (season_data[:, Cgbdisagg.INPUT_DIMENSION] > 0)
        else:
            # If no data available for the season

            single_peak_idx = np.array([])
            lap_idx = np.array([])

        # Get hourly count of peaks and laps

        peak_hourly_count = np.array([0] * Cgbdisagg.HRS_IN_DAY)
        lap_hourly_count = np.array([0] * Cgbdisagg.HRS_IN_DAY)

        # Calculate peaks to laps ratio and highest peak fraction at any hour

        peak_lap_ratio = 0
        highest_peak_fraction = 0

        if np.sum(single_peak_idx) > 0:
            # If thin pulses present in the data

            highest_peak_fraction, peak_lap_ratio = season_peaks_laps_ratio(season_data, peak_hourly_count,
                                                                            lap_hourly_count, single_peak_idx, lap_idx,
                                                                            factor, season, logger)
        else:
            # No thin pulses present in the data

            logger.info('No thin pulses found to check | ')

        # Save review features for the season

        features[season + '_peak_lap_ratio'] = peak_lap_ratio
        features[season + '_highest_peak_fraction'] = highest_peak_fraction

        # Save review features to the debug object

        review_features[season] = {
            'peak_lap_ratio': peak_lap_ratio,
            'lap_hourly_count': lap_hourly_count,
            'peak_hourly_count': peak_hourly_count,
            'highest_peak_fraction': highest_peak_fraction
        }

    debug['hld_review'] = review_features

    # Check the review features to conclude the detection

    # Extract peak fraction, peak/lap ratio and multiple peak laps count for winter

    wtr_peak_fraction = review_features['wtr']['highest_peak_fraction']
    wtr_peak_lap_ratio = review_features['wtr']['peak_lap_ratio']
    wtr_two_peak_count = features['wtr_two_peak_lap_count']

    # Extract peak fraction, peak/lap ratio and multiple peak laps count for intermediate

    itr_peak_fraction = review_features['itr']['highest_peak_fraction']
    itr_peak_lap_ratio = review_features['itr']['peak_lap_ratio']
    itr_two_peak_count = features['itr_two_peak_lap_count']

    # Check the winter review features

    wtr_peak_bool = (wtr_peak_fraction > peak_fraction)
    wtr_ratio_bool = (wtr_peak_lap_ratio > peak_lap_ratio)
    wtr_two_peak_bool = (wtr_two_peak_count == 0)

    # Check the intermediate review features

    itr_peak_bool = (itr_peak_fraction > peak_fraction)
    itr_ratio_bool = (itr_peak_lap_ratio > peak_lap_ratio)
    itr_two_peak_bool = (itr_two_peak_count == 0)

    # Generate boolean for winter and intermediate check

    wtr_check = (wtr_peak_bool or wtr_ratio_bool) and wtr_two_peak_bool
    itr_check = (itr_peak_bool or itr_ratio_bool) and itr_two_peak_bool

    # Give final detection

    if wtr_check and itr_check and (hld_probability > border_threshold):
        # If wtr and itr check fails and probability higher, give hld zero

        hld = 0

        logger.info('HLD zero due to wtr and itr check failure | ')

    elif (wtr_check or itr_check) and (hld_probability < border_threshold):
        # If wtr or itr check fails and probability lower, give hld zero

        hld = 0

        logger.info('HLD zero due to wtr or itr check failure | ')

    else:
        # If no check fails, give hld one

        hld = 1

        logger.info('HLD one due to wtr and itr check success | ')

    return hld, debug


def season_peaks_laps_ratio(season_data, peak_hourly_count, lap_hourly_count, single_peak_idx, lap_idx,
                            factor, season, logger):
    """
    Parameters:
        season_data                 (np.ndarray)    : Season's input data
        peak_hourly_count           (np.ndarray)    : Farction of peaks at each hour
        lap_hourly_count            (np.ndarray)    : Fraction of laps at each hour
        single_peak_idx             (np.ndarray)    : Indices of thin pulses
        lap_idx                     (np.ndarray)    : Indices of laps
        factor                      (int)           : Sampling rate factor
        season                      (str)           : Season string
        logger                      (logger)        : Logger object

    Returns:
        highest_peak_fraction       (float)         : Highest peak per hour fraction
        peak_lap_ratio              (float)         : Peaks to lap ratio at each hour
    """

    # If thin pulse present, get count at hourly level

    peak_hours, count = np.unique(season_data[single_peak_idx, Cgbdisagg.INPUT_HOD_IDX],
                                  return_counts=True)

    # Iterate over each available hour and populate count

    for i, hour in enumerate(peak_hours):
        peak_hourly_count[int(hour)] = count[i]

    # Get laps count at hourly level

    lap_hours, count = np.unique(season_data[lap_idx, Cgbdisagg.INPUT_HOD_IDX],
                                 return_counts=True)

    # Normalize for sampling rate

    count = count / factor

    # Iterate over each available hour and populate count

    for i, hour in enumerate(lap_hours):
        lap_hourly_count[int(hour)] = count[i]

    # Convert peak and lap counts to fractions

    peak_hourly_count = peak_hourly_count / np.sum(peak_hourly_count)
    lap_hourly_count = lap_hourly_count / np.sum(lap_hourly_count)

    # Calculate highest peak and lap fraction

    highest_peak_fraction = np.max(peak_hourly_count)
    highest_lap_fraction = lap_hourly_count[np.argmax(peak_hourly_count)]

    # Get peak / lap ratio

    peak_lap_ratio = np.round(highest_peak_fraction / highest_lap_fraction, 3)

    logger.info('Peak highest fraction for {} is | {}'.format(season, highest_peak_fraction))
    logger.info('Lap fraction of highest peak for {} is | {}'.format(season, highest_lap_fraction))

    return highest_peak_fraction, peak_lap_ratio
