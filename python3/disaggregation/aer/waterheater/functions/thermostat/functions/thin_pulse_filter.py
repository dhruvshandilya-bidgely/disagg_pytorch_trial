"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to filter out the invalid thin pulses (based on proximity)
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.maths_utils import moving_sum
from python3.utils.maths_utils.maths_utils import convolve_function


def thin_pulse_filter(in_data, thin_peak_energy_range, wh_config, filter_window, logger_base):
    """
    Parameters:
        in_data                 (np.ndarray)        : Input 21-column matrix
        thin_peak_energy_range  (np.ndarray)        : Thin pulse energy range
        wh_config               (dict)              : Config params
        filter_window           (int)               : Thin pulse filter size
        logger_base             (dict)              : Logger object

    Returns:
        thin_consumption        (np.ndarray)        : Updated thin pulse consumption
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('thin_pulse_filter')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking a deepcopy of input data to keep local instances

    input_data = deepcopy(in_data)
    thin_consumption = deepcopy(in_data)

    # Extract the thin pulse energy values and mask it zero

    thin_energy = deepcopy(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    thin_energy[:] = 0

    # Extract the lower and upper bound for thin pulse energy

    mu_lower, mu_upper = thin_peak_energy_range

    # Get the allowed number of peaks value from config

    allowed_peaks_count = wh_config['thermostat_wh']['estimation']['allowed_peaks_count']

    # Get the thin pulse amplitude deviation threshold

    thin_amp_std = wh_config['thermostat_wh']['estimation']['thin_pulse_amp_std']

    # Extract the peak and diff filter to find sharp peaks

    peak_filter = wh_config['thermostat_wh']['estimation']['peak_filter']
    diff_filter = wh_config['thermostat_wh']['estimation']['diff_filter']

    # Convolve the peak and diff filter over the thin pulse energy values

    amp = convolve_function(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], peak_filter)
    der = convolve_function(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], diff_filter)

    # Pick the pulse which satisfy the energy bounds

    single_peak_idx = (amp > mu_lower) & (amp < mu_upper) & (der < thin_amp_std)

    logger.info('Number of peaks before filtering | {}'.format(np.sum(single_peak_idx)))

    # Update the thin pulse energy values

    thin_energy[single_peak_idx] = amp[single_peak_idx]

    logger.info('Inter pulse gap filter used | {}'.format(filter_window))

    # Filter the peaks based on peak density in the moving thin pulse filter window

    peaks_moving_count = moving_sum(single_peak_idx, filter_window)

    # Remove the regions which contain more than 2 peaks in the window

    wrong_peaks = (peaks_moving_count > allowed_peaks_count)
    thin_energy[wrong_peaks] = 0

    logger.info('Final number of peaks after filtering | {}'.format(len(np.where(thin_energy > 0)[0])))

    # Update the thin pulse consumption data with new energy values

    thin_consumption[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = thin_energy

    return thin_consumption
