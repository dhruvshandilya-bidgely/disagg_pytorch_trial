"""
Author - Nikhil Singh Chauhan
Date - 16/10/18
This module fixes the missed thin or fat pulse at monthly level
"""

# Import python packages

import logging
import numpy as np
from math import modf

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.get_seasonal_segments import season_columns

from python3.disaggregation.aer.waterheater.functions.thermostat.functions.get_fat_pulse_range import get_fat_energy_range


def consumption_scaling(debug, wh_config, seasons, logger_base):
    """
    Parameters:
        debug               (dict)          : Algorithm intermediate steps output
        wh_config           (dict)          : Config params
        seasons             (np.ndarray)    : Bill cycle consumption info
        logger_base         (dict)          : Logger object

    Returns:
        seasons             (np.ndarray)    : Bill cycle consumption info
        debug               (dict)          : Algorithm intermediate steps output
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_scale_factor')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Get sampling rate in minutes

    sampling_rate_minutes = int(wh_config['sampling_rate'] / Cgbdisagg.SEC_IN_1_MIN)

    # Get the default scale factor values

    default_scale_factor = wh_config['thermostat_wh']['estimation']['default_scale_factor']

    # Check if hsm is to be used for scale factor

    if debug['use_hsm']:
        # Retrieve hsm from debug object

        hsm_in = debug['hsm_in']

        # Get the thin and fat scale factor from hsm

        thin_scale_factor = hsm_in.get('thin_scale_factor')

        fat_scale_factor = hsm_in.get('fat_scale_factor')

        # If the scale factor is None or invalid, give default scale factor

        if (thin_scale_factor is None) or (fat_scale_factor is None):
            thin_scale_factor, fat_scale_factor = default_scale_factor, default_scale_factor

    else:
        # Load duration model

        duration_model = debug['models']['thin_model']

        # Get the list of estimate for pulse duration and fat pulse energy bounds

        energy_ranges = get_fat_energy_range(debug['thin_peak_energy'], wh_config, duration_model, logger, False)

        # Retrieve the estimate bounds from energy ranges list

        dur, lower_fat_amp, fat_amp, _, _ = energy_ranges

        # Get minimum thin pulse energy bound from config

        min_thin_amp = wh_config['thermostat_wh']['estimation']['min_thin_pulse_amp']

        # Get max scale factor allowed for the different pulses (thin / fat)

        thin_scale_factor, fat_scale_factor, debug = get_scale_factor(seasons, sampling_rate_minutes, wh_config,
                                                                      min_thin_amp, dur, debug['thin_peak_energy'],
                                                                      lower_fat_amp, fat_amp, debug, logger_pass)

    # Update thin and fat monthly consumption

    seasons[:, season_columns['thin_monthly']] *= thin_scale_factor
    seasons[:, season_columns['fat_monthly']] *= fat_scale_factor

    # Saving total consumption for the user

    debug['new_thin_consumption'] = np.sum(seasons[:, season_columns['thin_monthly']])
    debug['new_fat_consumption'] = np.sum(seasons[:, season_columns['fat_monthly']])

    debug['new_wh_consumption'] = debug['new_thin_consumption'] + debug['new_fat_consumption']

    seasons = check_consumption_spillover(seasons, logger)

    return seasons, debug


def get_scale_factor(seasons, sampling_rate, wh_config, min_thin_amp, thin_duration, mu,
                     lower_fat_amp, fat_amp, debug, logger_base):
    """
    Parameters:
        seasons             (np.ndarray)        : Seasonal features information
        sampling_rate       (int)               : Sampling rate of the data
        wh_config           (dict)              : Water heater params
        logger_base         (dict)              : Logger object

    Returns:
        thin_scale_factor   (float)             : Thin consumption scale factor
        fat_scale_factor    (float)             : Fat consumption scale factor
        debug               (dict)              : Output from intermediate algo steps
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_scale_factor')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Get the default scale factor values

    default_scale_factor = wh_config['thermostat_wh']['estimation']['default_scale_factor']

    # Get total thin and fat consumption

    total_thin = np.sum(seasons[:, season_columns['thin_monthly']])
    total_fat = np.sum(seasons[:, season_columns['fat_monthly']])

    if (total_thin > 0) or (total_fat > 0):
        # Get thin to fat ratio

        thin_fat_ratio = total_thin / total_fat

        logger.info('Thin to fat ratio | {}'.format(thin_fat_ratio))

        # Get the thin scale factor

        thin_scale_factor = get_thin_scale_factor(thin_duration, min_thin_amp, mu, wh_config, sampling_rate)

        # Get the fat scale factor

        fat_scale_factor = get_fat_scale_factor(lower_fat_amp, fat_amp, wh_config, sampling_rate, debug, logger)

        # Calculate the overall scaling factor for water heater consumption

        overall_scale_factor = (thin_fat_ratio * thin_scale_factor + fat_scale_factor) / (thin_fat_ratio + 1)

    else:
        # If the thin / fat pulse consumption is zero, set default scaling

        logger.info('Thin to fat ratio is not defined | ')

        thin_fat_ratio = default_scale_factor - 1

        thin_scale_factor = default_scale_factor
        fat_scale_factor = default_scale_factor

        overall_scale_factor = default_scale_factor

    # Saving scale factors to hsm

    if np.sum(total_fat) > 0:
        debug['thin_fat_ratio'] = np.sum(total_thin) / np.sum(total_fat)
    else:
        debug['thin_fat_ratio'] = 0

    debug['thin_scale_factor'] = thin_scale_factor
    debug['fat_scale_factor'] = fat_scale_factor

    logger.info('Scale factor for thin pulse | {}'.format(thin_scale_factor))

    logger.info('Scale factor for fat pulse | {}'.format(fat_scale_factor))

    logger.info('Overall scale factor | {}'.format(overall_scale_factor))

    return thin_scale_factor, fat_scale_factor, debug


def get_thin_scale_factor(thin_duration, min_thin_amp, thin_peak_energy, wh_config, sampling_rate):
    """
    Parameters:
        thin_duration           (float)     : Duration of thin pulse
        min_thin_amp            (int)       : Minimum thin pulse energy per data point
        thin_peak_energy        (float)     : Thin pulse energy per data point
        wh_config               (dict)      : Water heater config params
        sampling_rate           (int)       : User sampling rate (in minutes)

    Returns:
        scale_factor            (float)     : Scaling factor for thin pulse
    """

    # Get the sampling rate dependent scaling factor from config

    thin_upscale_factor = wh_config['thermostat_wh']['estimation']['thin_upscale_factor']

    # Extract the minimum sampling rate required for scaling from config

    min_sampling_rate = int(wh_config['min_sampling_rate'] / Cgbdisagg.SEC_IN_1_MIN)

    # Calculate the scaling factor

    scale_factor = 1 / (1 - (min_thin_amp * thin_duration) / (thin_peak_energy * sampling_rate))

    # Get fraction part of scale factor

    scale_fraction, scale_integer = modf(scale_factor)

    # Account for detection probability in the pro-ration

    if sampling_rate > min_sampling_rate:
        # For higher sampling rate, upscale the factor

        scale_factor = scale_integer + (thin_upscale_factor * scale_fraction)

    # Cap the scale factor to max limit

    scale_factor = np.round(np.fmin(scale_factor, thin_upscale_factor), 4)

    return scale_factor


def get_fat_scale_factor(lower_fat_amp, fat_amp, wh_config, sampling_rate, debug, logger):
    """
    Parameters:
        lower_fat_amp       (float)     : Lower fat pulse energy bound
        fat_amp             (float)     : Optimum fat pulse energy
        wh_config           (dict)      : Water heater config params
        sampling_rate       (int)       : Sampling rate of the data
        debug               (dict)      : Algorithm intermediate steps output

    Returns:
        scale_factor        (float)     : Scaling factor for fat pulse
    """

    # Get the sampling rate dependent scaling factor from config

    fat_upscale_factor = wh_config['thermostat_wh']['estimation']['fat_upscale_factor']

    # Extract the minimum fat pulse duration

    min_fat_pulse_duration = wh_config['thermostat_wh']['estimation']['min_fat_pulse_duration']

    # Check if fat pulse pro-ration required for the pulse size

    if fat_amp < (fat_upscale_factor * lower_fat_amp):
        # If fat pulse is less than double of the minimum fat pulse size

        # Extract the fat consumption data

        fat_data = debug['final_fat_output']

        # Calculate number of days with fat pulse

        n_days = len(np.unique(fat_data[:, Cgbdisagg.INPUT_DAY_IDX]))

        # Get the hours of all fat pulse start

        fat_start_idx = (fat_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0)

        fat_pulse_hours = fat_data[fat_start_idx, Cgbdisagg.INPUT_HOD_IDX]

        # Define the bins for hourly count calculation

        bins = np.arange(-0.5, Cgbdisagg.HRS_IN_DAY + 0.5, 1)

        # Get the hourly count of fat pulse

        hourly_count, _ = np.histogram(fat_pulse_hours, bins=bins)

        # Convert hourly count to fractions

        hourly_fractions = np.fmin(hourly_count / n_days, 1)

        # Get the fraction standard deviation as a proportion of max hourly fraction

        fat_std_fraction = np.std(hourly_fractions) / np.max(hourly_fractions)

        # Save the fractions to debug object

        debug['fat_std_fraction'] = fat_std_fraction

        debug['max_fat_fraction'] = np.max(hourly_fractions)

        # Calculate the scale factor from fat pulse fractions

        fat_pulse_required = 1 - debug['max_fat_fraction']

        fraction_factor = 1 + fat_pulse_required

        logger.info('Fat fraction factor | {}'.format(fraction_factor))

        # Calculate the scale factor for missing fat pulse

        scale_factor = 1 + (min_fat_pulse_duration * (fat_upscale_factor * (lower_fat_amp / fat_amp) - 1) / sampling_rate)

        # Combining fractional scale factor and missing pulse scale factor

        scale_factor = np.mean([scale_factor, fraction_factor])

    else:
        # If fat pulse above the double of minimum fat pulse, give default scale factor

        scale_factor = 1

    # Cap the scale factor to max limit

    scale_factor = np.round(np.fmin(scale_factor, fat_upscale_factor), 4)

    return scale_factor


def check_consumption_spillover(seasons, logger):
    """
    Parameters:
        seasons         (np.ndarray)    : Bill cycle consumption info
        logger          (logger)        : Logger object

    Returns:
        seasons         (np.ndarray)    : Updated bill cycle consumption info
    """

    # Get total wh consumption at monthly level

    total_monthly_wh = np.sum(seasons[:, [season_columns['thin_monthly'], season_columns['fat_monthly']]], axis=1)

    # Get total raw consumption at monthly level

    total_monthly_input = seasons[:, season_columns['raw_monthly']]

    # Find highest spillover at any bill cycle

    spill_ratio = np.max(total_monthly_wh / total_monthly_input)

    # If spill over more than 1, prorate the water heater consumption

    if spill_ratio >= 1:
        # Get the scaling factor for over consumption

        proration_factor = 1 / spill_ratio

        seasons[:, [season_columns['thin_monthly'], season_columns['fat_monthly']]] *= proration_factor

        logger.info('Total water heater consumption more than total with ratio | {}'.format(spill_ratio))

    else:
        logger.info('Total water heater consumption within valid range | {}'.format(spill_ratio))

    return seasons
