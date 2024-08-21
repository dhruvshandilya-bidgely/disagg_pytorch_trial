"""
Author - Sahana M
Date - 20/07/2021
The module checks for heating instances
"""

# Import python packages
import logging
import numpy as np
from copy import deepcopy

# Import packages from withing the project

from python3.utils.maths_utils.rolling_function import rolling_function


def left_shifting(expansion_arr, common_time_bool, window):
    """
    This function is used for left shifting the array by the given window size
    Parameters:
        expansion_arr           (np.ndarray)    : Boolean array to shift
        common_time_bool        (np.ndarray)    : Boolean array containing the twh interval detected
        window                  (int)           : Window size
    Returns:
        expansion_arr           (np.ndarray)    : Boolean left shifted array
        """

    # Left shift and expand the duration
    for k in range(1, (window + 1)):
        expansion_arr[:-k] = expansion_arr[:-k] | common_time_bool[k:]

    return expansion_arr


def right_shifting(expansion_arr, common_time_bool, window):
    """
    This function is used for right shifting the array by the given window size
    Parameters:
        expansion_arr           (np.ndarray)    : Boolean array to shift
        common_time_bool        (np.ndarray)    : Boolean array containing the twh interval detected
        window                  (int)           : Window size
    Returns:
        expansion_arr           (np.ndarray)    : Boolean right shifted array
        """

    # Right shift and expand the duration
    for k in range(1, (window + 1)):
        expansion_arr[k:] = expansion_arr[k:] | common_time_bool[:-k]

    return expansion_arr


def long_extrapolation_check(common_time_bool, exploration_hour_cap, factor):
    """
    This function is used to check if the expansion has exceeded the maximum threshold
    Parameters:
        common_time_bool            (np.ndarray)    : Boolean array containing the twh interval detected
        exploration_hour_cap        (int)           : Threshold set for maximum expansion allowed
        factor                      (float)         : The number of units of data available in an hour of data
    Returns:
        common_time_bool            (np.ndarray)    : Boolean array containing the twh interval detected
    """

    # Identify the time interval detected for the twh

    time_idx_diff = np.diff(np.r_[0, common_time_bool.astype(int), 0])
    time_start_idx = np.where(time_idx_diff[:-1] > 0)[0]
    time_end_idx = np.where(time_idx_diff[1:] < 0)[0] + 1

    # If any extrapolation is more than 5 hours then don't extrapolate it

    time_diff = (time_end_idx - time_start_idx)
    long_dur_extrapolation = time_diff > exploration_hour_cap * factor
    for k in range(len(long_dur_extrapolation)):
        if long_dur_extrapolation[k]:
            start = time_start_idx[k]
            end = time_end_idx[k]
            common_time_bool[start:end] = False

    return common_time_bool


def check_box_timed_instance(filter_heating, filter_heating_days_bool, wh_config, amp_cap):
    """
    Check for timed heating instance
    Parameters:
        filter_heating                  (np.ndarray)     : 2D matrix containing the filtered data
        filter_heating_days_bool        (np.ndarray)     : Boolean array containing the probable heating days
        wh_config                       (dict)           : WH configuration dictionary
        amp_cap                         (float)          : Amplitude cap for heating instance capture

    Returns:
        filter_heating_days_bool        (np.ndarray)     : Boolean array containing the probable heating days
    """

    # If no heating days were identified then return

    if np.sum(filter_heating_days_bool) == 0:
        return filter_heating_days_bool

    # Get all the required data

    window_size = wh_config.get('factor')*1.5
    rows = filter_heating.shape[0]
    cols = filter_heating.shape[1]
    temp = deepcopy(filter_heating)
    temp = temp.flatten()

    # Check for timed heating instance by fitting boxes on a window of 1.5 hours

    valid_idx = np.array(temp >= amp_cap)
    moving_sum = rolling_function(valid_idx, window_size, 'sum')
    valid_sum_bool = (moving_sum >= window_size)
    valid_sum_bool = valid_sum_bool.reshape(rows, cols)
    valid_sum_bool = valid_sum_bool[filter_heating_days_bool]

    # All the days where timed heating boxes were found are legitimate heating days

    legitimate_heating_days = np.sum(valid_sum_bool, axis=1) > 0
    timed_heating_days = np.sum(legitimate_heating_days)
    percentage_timed_heating_days = timed_heating_days/len(valid_sum_bool)

    # If the timed heating is not consistent then probably not a heating instance

    if percentage_timed_heating_days < wh_config['heating_percentage_days']:
        filter_heating_days_bool[filter_heating_days_bool] = False

    return filter_heating_days_bool


def check_heating_instance(aoi_filtered_data, twh_data_matrix, debug, wh_config, logger_base):
    """
    This function is used to check for a heating instance and if present then interpolated a base twh pattern
    Parameters:
        aoi_filtered_data     (np.ndarray)    : 2D matrix containing the filtered data with non consistent time made 0
        twh_data_matrix       (np.ndarray)    : 2D matrix containing the original data with non consistent time made 0
        debug                 (dict)          : Dictionary containing algorithm outputs
        wh_config             (dict)          : WH configuration dictionary
        logger_base           (logger)        : Logger passed

    Returns:
        filter_matrix         (np.ndarray)    : 2D matrix containing the filtered data with heating instance, interpolated
        twh_matrix            (np.ndarray)    : 2D matrix containing the original data with heating instance, interpolated
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('check_heating_instance')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Initialise the necessary variables

    factor = debug.get('factor')
    amp_bar_twh = wh_config.get('amp_bar_twh')
    min_amp_bar = wh_config.get('min_amp_bar')
    max_amp_bar = wh_config.get('max_amp_bar')
    window = wh_config.get('heating_inst_window')
    amp_bar_filter = wh_config.get('amp_bar_filter')
    min_amp = int(wh_config.get('heating_min_amp')/factor)
    exploration_hour_cap = wh_config.get('exploration_hour_cap')
    heating_inst_twh_thr = wh_config.get('heating_inst_twh_thr')
    concentration_arr_bool = debug.get('concentration_arr_bool')
    heating_consistency_thr = wh_config.get('heating_consistency_thr')

    # Initialise the necessary data

    twh_matrix = deepcopy(twh_data_matrix)
    filter_matrix = deepcopy(aoi_filtered_data)

    # Get the individual groups of time bands from concentration_arr_bool

    band_idx_diff = np.diff(np.r_[0, concentration_arr_bool.astype(int), 0])
    band_start_idx = np.where(band_idx_diff[:-1] > 0)[0]
    band_end_idx = np.where(band_idx_diff[1:] < 0)[0]+1

    num_bands = 0
    debug['heating_instance_bands'] = {
        'user_heating_inst': False,
        'num_bands': num_bands
    }

    # For each time band check and interpolate for heating instance

    for i in range(len(band_start_idx)):

        # Identify the start and end time of the band and copy the data

        start = int(band_start_idx[i])
        end = int(band_end_idx[i])
        filter_heating = filter_matrix[:, start:end]
        twh_heating = twh_matrix[:, start:end]

        # Identify the high consumption days and get their consistency

        temp = np.percentile(filter_heating, axis=1, q=95)
        temp = temp[temp > min_amp]
        optimal_amp = np.median(temp)
        consistent = np.sum((temp > min_amp_bar * optimal_amp) & (temp < max_amp_bar * optimal_amp)) / len(temp)

        # Interpolate only if the consistency is less than the threshold

        if consistent < heating_consistency_thr:

            logger.info('Identifying heating instance days to interpolate | ')

            # For the filtered heating identify the heating days

            filter_days_amplitude = np.percentile(filter_heating, axis=1, q=95)
            filter_heating_days_bool = filter_days_amplitude > (amp_bar_filter/factor)
            filter_heating_days_bool = check_box_timed_instance(filter_heating, filter_heating_days_bool, wh_config,
                                                                (amp_bar_filter/factor))

            # For the twh heating (original data) identify the heating days

            twh_days_amplitude = np.percentile(twh_heating, axis=1, q=95)
            twh_heating_days_bool = twh_days_amplitude > (amp_bar_twh/factor)
            heating_instance_amp = np.median(twh_days_amplitude[twh_heating_days_bool])
            twh_heating_days_bool = check_box_timed_instance(twh_heating, twh_heating_days_bool, wh_config,
                                                             (amp_bar_twh/factor))

            # Get the common heating days

            common_heating_days_bool = filter_heating_days_bool | twh_heating_days_bool

            # Get the Probable Water heater days

            wh_days_bool = ~common_heating_days_bool
            filter_wh_days = filter_heating[wh_days_bool]
            twh_wh_days = twh_heating[wh_days_bool]

            # Calculate the probable timed wh amplitude

            filter_amplitude = np.median(filter_days_amplitude[~common_heating_days_bool])
            twh_amplitude = np.median(twh_days_amplitude[~common_heating_days_bool])

            # Identify the probable time of concentration of timed wh in both filtered and twh data

            filter_twh_time = (np.sum((filter_wh_days > min_amp_bar * filter_amplitude) &
                                      (filter_wh_days < max_amp_bar * filter_amplitude), axis=0)) / len(filter_wh_days)
            filter_twh_time_bool = filter_twh_time > heating_inst_twh_thr
            twh_time = (np.sum((twh_wh_days > min_amp_bar * twh_amplitude) &
                               (twh_wh_days < max_amp_bar * twh_amplitude), axis=0)) / len(twh_wh_days)
            twh_time_bool = twh_time > heating_inst_twh_thr

            # Get the common time of twh
            common_time_bool = filter_twh_time_bool | twh_time_bool

            # Check for longer duration expansions

            common_time_bool = long_extrapolation_check(common_time_bool, exploration_hour_cap, factor)

            expansion_arr = deepcopy(common_time_bool)

            # Left shift and expand the duration
            expansion_arr = left_shifting(expansion_arr, common_time_bool, window)

            # Right shift and expand the duration
            expansion_arr = right_shifting(expansion_arr, common_time_bool, window)

            # Expanded array start & end
            if np.sum(expansion_arr):
                expansion_idx_diff = np.diff(np.r_[0, expansion_arr.astype(int), 0])
                expansion_start_idx = np.where(expansion_idx_diff[:-1] > 0)[0]
                expansion_end_idx = np.where(expansion_idx_diff[1:] < 0)[0] + 1
                expansion_start = expansion_start_idx[0] + start
                expansion_end = expansion_end_idx[-1] + start
            else:
                expansion_start = -1
                expansion_end = -1

            # Replace the Filter data heating days with the interpolated amplitude

            replacing_days = filter_heating[common_heating_days_bool]
            replacing_days[:, expansion_arr] = filter_amplitude
            replacing_days[:, ~expansion_arr] = 0
            filter_heating[common_heating_days_bool] = replacing_days
            filter_matrix[:, start: end] = filter_heating

            # Replace the TWH data heating days with the interpolated amplitude

            replacing_days = twh_heating[common_heating_days_bool]
            replacing_days[:, expansion_arr] = twh_amplitude
            replacing_days[:, ~expansion_arr] = 0
            twh_heating[common_heating_days_bool] = replacing_days
            twh_matrix[:, start: end] = twh_heating

            # Store the necessary info in debug object

            if np.sum(common_heating_days_bool) > 0 and np.sum(expansion_arr) > 0:
                num_bands += 1
                interpolated_twh = True
                debug['heating_instance_bands']['num_bands'] = num_bands
                debug['heating_instance_bands']['user_heating_inst'] = True
                debug['heating_instance_bands']['band_idx_' + str(num_bands)] = {
                    'buffer': window*factor,
                    'twh_amp': twh_amplitude,
                    'end_time': expansion_end,
                    'start_time': expansion_start,
                    'expansion_arr': expansion_arr,
                    'heating_inst_amp': heating_instance_amp,
                    'has_heating_instance': interpolated_twh,
                    'twh_heating_days': common_heating_days_bool,
                }

                logger.info('Heating instance detected for this user | ')

    logger.info('Number of bands with heating instance detected and interpolated | {} '.format(num_bands))

    return filter_matrix, twh_matrix
