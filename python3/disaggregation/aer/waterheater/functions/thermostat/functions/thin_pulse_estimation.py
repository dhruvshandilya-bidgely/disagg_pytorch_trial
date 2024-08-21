"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module estimates the thin pulse consumption of water heater
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.thermostat.functions.thermostat_features import WhFeatures
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.thin_pulse_filter import thin_pulse_filter


def thin_pulse_estimation(season_features, thin_peak_energy, thin_peak_energy_range, inter_pulse_gap, wh_config, season, logger_base):
    """
    Parameters:
        season_features         (dict)          : Seasonal features
        thin_peak_energy        (float)         : Thin pulse amplitude
        thin_peak_energy_range  (np.ndarray)    : Thin pulse energy range
        inter_pulse_gap         (flaot)         : Gap between thin pulses (in hours)
        wh_config               (dict)          : Config params
        season                  (str)           : Current season
        logger_base             (dict)          : Logger object

    Returns:
        thin_consumption        (np.ndarray)    : Thin pulse consumption
        max_thin_consumption    (float)         : Max thin pulse consumption
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('thin_pulse_estimation')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking a deepcopy of input data to keep local instances

    input_data = deepcopy(season_features['data'])
    thin_consumption = deepcopy(input_data[:, :Cgbdisagg.INPUT_DIMENSION])

    # Check if enough peaks available for estimation

    if len(season_features['lap_peaks']) > 0:
        # If valid number of thin pulses available, retrieve energy values

        actual_thin_pulse = input_data[season_features['lap_peaks'], Cgbdisagg.INPUT_CONSUMPTION_IDX]
    else:
        # If no thin pulses available, return blank

        actual_thin_pulse = np.array([])

    logger.info('(Season, count, thin_peak_energy, mu_std) | ({}, {}, {}, {})'.format(season, len(actual_thin_pulse),
                                                                                      np.nanmean(actual_thin_pulse),
                                                                                      np.nanstd(actual_thin_pulse)))

    # Estimate the maximum possible number of thin pulses using number of days and inter pulse gap

    num_thin_pulse = np.sum(season_features['features'][:, WhFeatures.n_base + WhFeatures.NUM_DAYS]) * \
                     (Cgbdisagg.HRS_IN_DAY / inter_pulse_gap)

    # Check if valid number of pulses

    if num_thin_pulse > 0:
        # If non-zero number of thin pulses, calculate maximum consumption

        max_thin_consumption = num_thin_pulse * thin_peak_energy
    else:
        # If no thin pulses available, default maximum consumption

        max_thin_consumption = np.sum(thin_consumption[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Calculate the window size using inter pulse gap

    thin_pulse_filter_size = int((inter_pulse_gap * Cgbdisagg.SEC_IN_HOUR) / wh_config['sampling_rate']) - 1

    # Use thin_pulse_filter_size to remove invalid pulses

    if (thin_consumption.shape[0] > 0) and (thin_peak_energy > 0):
        # If thin pulse data available and valid thin_peak_energy

        thin_consumption = thin_pulse_filter(thin_consumption, thin_peak_energy_range, wh_config,
                                             thin_pulse_filter_size, logger_pass)

        logger.info('Thin pulse consumption updated using thin pulse filter | ')

    elif thin_peak_energy == 0:
        # If no thin pulse data available, make consumption zero

        thin_consumption[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        logger.info('No thin pulses in the input data | ')

    else:
        # If input data is blank

        logger.info('No input data available | ')

        return np.array([], dtype=np.int64).reshape(0, Cgbdisagg.INPUT_DIMENSION), 0

    return thin_consumption, max_thin_consumption


def find_thin_peak_energy(season_features, wh_config, logger, log_it=True):
    """
    Parameters:
        season_features         (np.ndarray)    : Season features
        wh_config               (dict)          : Water heater params
        logger                  (logger)        : Logger object
        log_it                  (bool)          : Boolean to log values

    Returns:
        thin_peak_energy                     (dict)          : Thin pulse amplitude for current season
    """

    # Taking a deepcopy of input data to keep local instances

    features = deepcopy(season_features)

    # Retrieve minimum required peaks fraction value from config

    min_fraction = wh_config['thermostat_wh']['estimation']['min_peaks_fraction']

    # Get the feature column indices

    n_base = WhFeatures.n_base

    # Check if features are valid

    if features.shape[0] > 0:
        # If the features are non-zero

        # Sort bill cycles / months based on peaks count and get max peaks count

        features = features[features[:, n_base + WhFeatures.COUNT_PEAKS].argsort()[::-1]]
        max_count = features[0, n_base + WhFeatures.COUNT_PEAKS]

        # Keep bill cycles with at above 20 percent of max peaks count

        features = features[features[:, n_base + WhFeatures.COUNT_PEAKS] > (min_fraction * max_count)]

        # Extract the peaks count and the thin_peak_energy of each bill cycle / month

        peaks_count = features[:, n_base + WhFeatures.COUNT_PEAKS]
        mu_values = features[:, n_base + WhFeatures.ENERGY]

        # Take weighed mean of the mu_values

        thin_peak_energy = np.sum(mu_values * peaks_count) / np.sum(peaks_count)

        # Handle NaN thin_peak_energy

        if np.isnan(thin_peak_energy):
            thin_peak_energy = 0
    else:
        # if no features present

        thin_peak_energy = 0

    if log_it:
        logger.info('Thin pulse thin_peak_energy | {}'.format(thin_peak_energy))

    return thin_peak_energy


def get_inter_pulse_gap(season_features, wh_config, logger):
    """
    Parameters:
        season_features         (np.ndarray)    : Season features
        wh_config               (dict)          : Config params
        logger                  (logger)        : Logger object

    Returns:
        inter_pulse_gap         (float)         : Inter pulse gap for current season
    """

    # Extract peak factor values for the current season

    peak_factor_column_index = WhFeatures.n_base + WhFeatures.PEAK_FACTOR

    peak_factor = np.mean(season_features['features'][:, peak_factor_column_index])

    # Check if the peak factor is valid to find inter pulse gap

    if peak_factor > 0:
        # If non-zero peak factor, get inter pulse gap

        inter_pulse_gap = 1 / peak_factor
    else:
        # If peak factor zero (no thin pulses), use default inter pulse gap

        inter_pulse_gap = wh_config['thermostat_wh']['estimation']['min_inter_pulse_gap']

    logger.info('Inter thin pulse gap | {}'.format(inter_pulse_gap))

    return inter_pulse_gap
