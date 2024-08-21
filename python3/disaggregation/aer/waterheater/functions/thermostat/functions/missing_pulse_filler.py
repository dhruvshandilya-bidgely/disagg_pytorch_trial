"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module fixes the underestimation of thin pulses due to missed thin pulses over hvac or other appliances
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def thin_pulse_filler(debug, thin_signal, wh_config, logger):
    """
    Parameters:
        debug               (dict)          : Algorithm intermediate steps output
        thin_signal         (np.ndarray)    : Thin pulse consumption
        wh_config           (dict)          : Water heater params
        logger              (logger)        : Logger object

    Returns:
        debug               (dict)          : # Taking a deepcopy of input data to keep local instances
    """

    # Taking a deepcopy of thin data to keep local instances

    thin_output = deepcopy(thin_signal)

    # Get thin pulse indices

    peaks_idx = thin_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0

    # Count daily number of thin peaks

    unq_days, days_idx = np.unique(thin_output[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
    daily_peaks_count = np.bincount(days_idx, peaks_idx)

    # Check if any thin peak found

    if np.sum(daily_peaks_count) == 0:
        # If no thin peaks found, then deficit is zero

        debug['thin_deficit'] = 0

        debug['thin_deficit_perc'] = 0

        return debug

    # Extract required params from config

    peak_daily_proportion = wh_config['thermostat_wh']['estimation']['peak_daily_proportion']

    # Pick daily counts greater than zero

    daily_peaks_count = daily_peaks_count[daily_peaks_count > 0]

    bins = np.arange(0, np.max(daily_peaks_count) + 2) - 0.5

    # Histogram of daily peaks count

    peaks_count, edges = np.histogram(daily_peaks_count, bins=bins)
    edges = (edges[1:] + edges[:-1]) / 2

    # Select the count bin with highest frequency and the corresponding count

    ideal_edge = np.fmax(np.where(peaks_count >= peak_daily_proportion * np.max(peaks_count))[0][-1], 0)
    ideal_count = edges[ideal_edge]

    logger.debug('Ideal daily count | {}'.format(ideal_count))

    # Check if thin_peak_energy value available

    if debug['use_hsm']:
        # If MTD run mode, use thin_peak_energy values from hsm

        hsm_in = debug['hsm_in']

        thin_peak_energy = get_season_thin_peak_energy(hsm_in, debug['season'])
    else:
        # If historical / incremental run mode, get overall thin_peak_energy from debug

        thin_peak_energy = debug['thin_peak_energy']

    # Calculate the thin pulse energy deficit based on ideal daily count

    deficit = np.sum((ideal_count - edges[:ideal_edge]) * peaks_count[:ideal_edge]) * thin_peak_energy

    # Add deficit value to debug and log it

    debug['thin_deficit'] = deficit

    debug['thin_deficit_perc'] = np.round(100 * (deficit / np.sum(thin_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])), 3)

    logger.info('Thin pulse deficit | {}'.format(deficit))

    return debug


def get_season_thin_peak_energy(hsm_in, season):
    """
    Parameters:
        hsm_in          (dict)      : Input hsm
        season          (str)       : Season

    Returns:
        thin_peak_energy             (float)     : Thin pulse amplitude for the season
    """

    # Try to extract from array, if fails extract directly

    # noinspection PyBroadException
    try:
        thin_peak_energy = hsm_in[season + '_thin_peak_energy'][0]
    except IndexError:
        thin_peak_energy = hsm_in[season + '_thin_peak_energy']

    return thin_peak_energy


def deficit_distribution(debug, monthly_thin, days_per_bc, logger, minimum_days=5):
    """
    Parameters:
        debug               (dict)          : Algorithm intermediate steps output
        monthly_thin        (np.ndarray)    : Monthly thin pulse usage
        days_per_bc         (np.ndarray)    : Days per bill cycle
        logger              (logger)        : Logger object
        minimum_days        (int)           : Minimum days for bill cycle to be considered

    Returns:
        thin_monthly        (np.ndarray)    : Updated monthly thin pulse usage
    """

    # Extract deficit value from debug

    deficit = debug['thin_deficit']

    # If very small deficit value, skip distribution

    if deficit < 1:
        return monthly_thin

    # Bill cycles with less than min days

    invalid_bill_cycles = (days_per_bc <= minimum_days)

    # Taking a deepcopy of thin monthly data to keep local instances

    thin_monthly = deepcopy(monthly_thin)

    # Calculate the usage per day

    thin_monthly_per_day = thin_monthly / days_per_bc

    # Initialize a deficit array

    monthly_deficit = np.array([deficit] * len(thin_monthly_per_day))

    # Ratio of deficit is reciprocal of thin energy per day ratio

    thin_ratios = 1 / (thin_monthly_per_day / np.max(thin_monthly_per_day))

    # Replace any infinite to zero (zero thin pulse consumption months)

    thin_ratios[np.isinf(thin_ratios)] = 0

    # Prorate the ratio for number of days in bill cycle

    thin_ratios *= days_per_bc

    # Ignore ratios for bill cycles with low number of days

    thin_ratios[invalid_bill_cycles] = 0

    # Normalize ratio to sum = 1 and then multiply with deficit array

    monthly_deficit *= (thin_ratios) / np.sum(thin_ratios)

    monthly_deficit_tuple = [(ts, temp_deficit) for ts, temp_deficit in zip(debug['bill_cycle_ts'], monthly_deficit)]

    logger.info('Monthly thin pulse deficit | {}'.format(monthly_deficit_tuple))

    # Add deficit to thin pulse monthly array

    thin_monthly += monthly_deficit
    thin_monthly = np.round(thin_monthly, 2)

    return thin_monthly
