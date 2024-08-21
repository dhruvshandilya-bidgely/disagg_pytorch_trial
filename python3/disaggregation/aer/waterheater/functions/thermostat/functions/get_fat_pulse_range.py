"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module calculates the fat pulse energy range (lower and upper bounds)
"""

# Import python packages

import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants


def get_fat_energy_range(thin_peak_energy, wh_config, duration_models, logger, log_it=True):
    """
    Parameters:
        thin_peak_energy        (float)         : Thin pulse amplitude
        wh_config               (dict)          : Config params
        duration_models         (dict)          : Models to predict thin pulse duration
        logger                  (logger)        : Logger object
        log_it                  (bool)          : Boolean to log values

    Returns:
        duration                (float)         : Duration of thin pulse
        lower_fat_amp           (float)         : Lower bound for fat pulse
        fat_amp                 (float)         : Optimum value for fat pulse
        upper_fat_amp           (float)         : Upper bound for fat pulse
        thin_peak_energy_range  (np.ndarray)    : Thin pulse energy range
    """

    # Taking a deepcopy of thin_peak_energy to keep local instances

    thin_peak_energy_local = deepcopy(thin_peak_energy)

    # Get the pilot id of the user

    pilot_id = wh_config['pilot_id']

    # Extract the estimation config params

    estimation_config = wh_config['thermostat_wh']['estimation']

    sampling_rate = wh_config['sampling_rate']
    min_sampling_rate = wh_config['min_sampling_rate']

    # Extract the thin pulse related config (min amplitude, duration deviation)

    min_amp = estimation_config['min_thin_pulse_amp']
    thin_pulse_dur_std = estimation_config['thin_pulse_dur_std']

    # Extract the minimum and maximum allowed thin pulse duration

    min_duration_factor = estimation_config['min_duration_factor']
    max_duration_factor = estimation_config['max_duration_factor']

    # Extract the minimum fat pulse duration

    min_fat_pulse_duration = estimation_config['min_fat_pulse_duration']

    # Decide which duration model to be used (NA / AU)

    if pilot_id in PilotConstants.NEW_ZEALAND_PILOTS:
        duration_model = duration_models['model_au']
    else:
        duration_model = duration_models['model_na']

    # Make prediction for the duration of thin pulse (in kW)

    duration = duration_model.predict(np.array([thin_peak_energy_local]).reshape(-1, 1))[0]

    # Get maximum and minimum thin pulse duration

    min_duration = duration * min_duration_factor
    max_duration = duration * max_duration_factor

    # Estimating the water heater power (size) rounded to 100 Watt

    wh_power = thin_peak_energy_local * Cgbdisagg.SEC_IN_1_MIN / duration
    wh_power = np.ceil(wh_power / 100) * 100

    # Fat pulse energy range considering the maximum / minimum thin pulse duration

    fat_amp = thin_peak_energy * (sampling_rate / Cgbdisagg.SEC_IN_1_MIN) / duration

    upper_fat_amp = thin_peak_energy * (sampling_rate / Cgbdisagg.SEC_IN_1_MIN) / min_duration
    lower_fat_amp = thin_peak_energy * (sampling_rate / Cgbdisagg.SEC_IN_1_MIN) / max_duration

    # Adjust the fat pulse energy bar for higher sampling rate

    if sampling_rate > min_sampling_rate:
        # Sampling rate above minimum (900), use min fat pulse duration

        fat_pulse_duration = min_fat_pulse_duration

        min_fat_amp = thin_peak_energy_local * fat_pulse_duration / duration

        # Keep the lower fat pulse bound considering min duration

        lower_fat_amp = np.fmin(min_fat_amp, lower_fat_amp)

    # Calculate the range of thin pulses, using min / max duration and deviation allowed

    thin_peak_energy_minimum = np.fmax(thin_peak_energy * (1 - thin_pulse_dur_std), min_amp)
    thin_peak_energy_maximum = thin_peak_energy * (1 + thin_pulse_dur_std)

    # Make a thin pulse range array

    thin_peak_energy_range = [thin_peak_energy_minimum, thin_peak_energy_maximum]

    energy_ranges = [duration, lower_fat_amp, fat_amp, upper_fat_amp, thin_peak_energy_range]

    # Log energy bounds

    if log_it:
        logger.info('Duration: | {}'.format(duration))
        logger.info('best_fat_amp: | {}'.format(fat_amp))
        logger.info('lower_fat_amp: | {}'.format(lower_fat_amp))
        logger.info('upper_fat_amp: | {}'.format(upper_fat_amp))
        logger.info('Waterheater power (in kW): | {}'.format(wh_power))
        logger.info('thin_peak_energy_range: | {}'.format(thin_peak_energy_range))

    return energy_ranges
