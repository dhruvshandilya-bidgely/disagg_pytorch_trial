"""
Author - Sahana M
Date - 2/3/2021
Detects seasonal wh based on time bands and gives the estimate of consumption in that time band
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy
from scipy.signal import savgol_filter

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.itemization.aer.water_heater.functions.get_duration import get_duration
from python3.itemization.aer.water_heater.functions.get_time_zones import get_time_zones
from python3.itemization.aer.water_heater.functions.get_valid_days import get_valid_days
from python3.itemization.aer.water_heater.functions.get_consistency import get_consistency
from python3.itemization.aer.water_heater.functions.get_buffer_days import get_buffer_days
from python3.itemization.aer.water_heater.functions.post_processing import post_processing
from python3.itemization.aer.water_heater.functions.init_data_matrix import init_data_matrix
from python3.itemization.aer.water_heater.functions.get_energy_range import get_energy_range
from python3.itemization.aer.water_heater.functions.data_sanity_checks import data_sanity_check
from python3.itemization.aer.water_heater.functions.math_utils import find_seq


def get_time_zones_mtd(hsm_in, logger):
    """
    This function is used to get the Time zones and Local maxima during the historical run to use it for MTD run
    Args:
        hsm_in          (dict)          : HSM dictionary
        logger          (Logger)        : Logger
    Returns:
        time_zones      (List)          : List of potential time zones identified
        local_max_idx   (List)          : Local maxima index identified
    """

    if hsm_in.get('swh_hld') == [1]:
        start_time_zones = hsm_in.get('swh_start_time_zones')
        end_time_zones = hsm_in.get('swh_end_time_zones')
        local_max_idx = hsm_in.get('local_max_idx')
        local_max_idx = tuple([local_max_idx])
        time_zones = []
        if len(start_time_zones):
            time_zones = start_time_zones
            time_zones = np.c_[time_zones, end_time_zones]
    else:
        time_zones = []
        local_max_idx = []
        logger.info('No Seasonal Water heater time zones detected |')

    return time_zones, local_max_idx


def get_score_for_mtd(debug, hsm_in, score, i, detection_thr):
    """
    This function is used to update the MTD run score based on Historical run
    Args:
        debug               (dict)          : Contains all variables required for debugging
        hsm_in              (dict)          : HSM dictionary
        score               (float)         : Score for the band
        i                   (int)           : Index in consideration
        detection_thr       (float)         : Detection threshold for the band
    Returns:
        score               (float)         : Score for the band
    """
    if debug.get('disagg_mode') == 'mtd':
        hsm_band_scores = hsm_in.get('swh_band_scores')

        if score < detection_thr:
            score = (score + hsm_band_scores[i]) / 2

        if (score >= detection_thr) and (hsm_band_scores[i] < detection_thr):
            score = 0

    return score


def store_in_debug(time_zones, local_max_idx, debug):
    """
    This function is used to store time zone and local maxima info in the debug object
    Args:
        time_zones      (List)          : List of potential time zones identified
        local_max_idx   (List)          : Local maxima index identified
        debug           (dict)          : Contains all variables required for debugging
    Returns:
        debug           (dict)          : Contains all variables required for debugging
    """
    if len(time_zones):
        debug['start_time_zones'] = np.asarray(time_zones)[:, 0]
        debug['end_time_zones'] = np.asarray(time_zones)[:, 1]
        debug['local_max_idx'] = np.asarray(local_max_idx)[0]
    else:
        debug['start_time_zones'] = []
        debug['end_time_zones'] = []
        debug['local_max_idx'] = []

    return debug


def update_detection_status(total_detections, debug, logger):
    """
    This function is used to update the detection status in the debug object
    Args:
        total_detections            (int)       : Total detected time bands
        debug                       (dict)      : Debug object
        logger                      (Logger)    : Logger
    Returns:
        debug                       (dict)      : Debug object
    """

    if total_detections >= 1:
        debug['swh_hld'] = 1
        logger.info('SWH detected | ')
    else:
        debug['swh_hld'] = 0
        logger.info('SWH not detected | ')

    return debug


def detect_time_band(in_data, seasonal_wh_config, debug, logger_pass):

    """
    This functions detects & estimates potential seasonal wh time band
    Args:
        in_data                 (np.ndarray)    : 21 column input data
        seasonal_wh_config      (dict)          : Dictionary containing all needed configuration variables
        debug                   (dict)          : Contains all variables required for debugging
        logger_pass             (dict)          : Contains the logger and the logging dictionary to be passed on
    Returns:
        debug                   (dict)          : Contains all variables required for debugging
    """

    # Initialize logger

    logger_base = logger_pass.get('base_logger').getChild('detect_time_band')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Initialise all the variables to be used

    swh_data = deepcopy(in_data)
    time_band_prob = seasonal_wh_config.get('tb_prob')
    wh_potential = deepcopy(debug.get('wh_potential'))
    debug['wh_potential'] = wh_potential
    moving_index = seasonal_wh_config.get('moving_index')
    time_zones_div = seasonal_wh_config.get('time_zones_div')
    min_wh_days = seasonal_wh_config.get('config').get('min_wh_days')
    padding_days = seasonal_wh_config.get('config').get('padding_days')
    min_amplitude = seasonal_wh_config.get('config').get('min_amplitude')
    max_amplitude = seasonal_wh_config.get('config').get('max_amplitude')
    min_wh_percent = seasonal_wh_config.get('config').get('min_wh_percent')
    sampling_rate = seasonal_wh_config.get('user_info').get('sampling_rate')
    correlation_thr = seasonal_wh_config.get('config').get('correlation_threshold')
    debug['fl'] = debug.get('weather_data_output').get('weather').get('day_wise_data').get('fl')
    cooling_potential = debug.get('weather_data_output').get('weather').get('hvac_potential_dict').get('cooling_pot')
    cooling_pot_thr = debug.get('weather_data_output').get('weather').get('hvac_potential_dict').get('cooling_min_thr')

    # Compute variables required for further operations

    hsm_in = debug.get('hsm_in').get('attributes')
    factor = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)
    time_division = str(factor * Cgbdisagg.HRS_IN_DAY)

    # ------------------------------------------ STAGE 1: INITIALISE DATA ---------------------------------------------

    # Get the (days x time) shaped consumption & vacation data matrix

    swh_data_matrix, vacation_data_matrix = init_data_matrix(swh_data, seasonal_wh_config, debug)

    # Perform Data sanity checks

    wh_potential, cooling_potential, debug = data_sanity_check(wh_potential, cooling_potential, swh_data_matrix, debug)

    # ------------------------------------------ STAGE 2: GET WH POTENTIAL DAYS ---------------------------------------

    # Filter out wh_potential days

    wh_present_indexes = wh_potential > 0 & (wh_potential != np.nan)

    # Get vacation day indexes

    vacation_day_indexes = np.sum(vacation_data_matrix[:, :], axis=1) > 0

    # Remove vacation days from wh_present_indexes

    wh_present_indexes = wh_present_indexes & ~vacation_day_indexes

    # Get no wh days

    seq_days = find_seq(wh_present_indexes)
    no_wh_seq = seq_days[seq_days[:, 0] == 0, :]

    # Get the buffer days for padding

    # ------------------------------------------ STAGE 3: GET BUFFER DAYS ---------------------------------------------

    padding_indexes = get_buffer_days(no_wh_seq, seq_days, wh_present_indexes, cooling_pot_thr, cooling_potential,
                                      seasonal_wh_config, logger_pass)

    # Extract all the necessary indexes

    start_idx, end_idx, trn_start_1, trn_end_1, trn_start_2, trn_end_2 = padding_indexes

    # Store all the indexes in debug dictionary

    debug['swh_wh_pot_start'] = start_idx
    debug['swh_wh_pot_end'] = end_idx
    debug['trn_start_1'] = trn_start_1
    debug['trn_end_1'] = trn_end_1
    debug['trn_start_2'] = trn_start_2
    debug['trn_end_2'] = trn_end_2
    debug['min_amplitude'] = min_amplitude
    debug['max_amplitude'] = max_amplitude
    debug['wh_potential'] = wh_potential
    debug['wh_present_idx'] = wh_present_indexes
    debug['swh_data_matrix'] = swh_data_matrix
    debug['factor'] = factor

    # ------------------------------------------ STAGE 4: GET TIME BAND CORRELATION -----------------------------------

    # Implement moving bands from 4 am to 6 pm

    moving_band_corr = []
    starting_time = moving_index[time_division][0]

    # Get correlation

    while (starting_time + 3 * factor) <= moving_index[time_division][1]:

        # get all days data in the corresponding time band

        band_data = swh_data_matrix[:, starting_time: (starting_time + 3 * factor)]

        # aggregate the time band energy

        band_energy = np.sum(band_data, axis=1)

        # Get the wh present days indexes (where wh potential > 0)

        valid_days_bool = deepcopy(wh_present_indexes)

        # Get the valid days bool with wh_present_index and buffer days padding

        valid_days_bool = get_valid_days(valid_days_bool, padding_days, debug)
        nan_indexes = np.isnan(wh_potential)
        valid_days_bool = valid_days_bool & ~nan_indexes

        # Get the final days to be considered for correlation (wh_present_days + buffer_days)

        num_valid_days = np.sum(valid_days_bool)

        # Ensure minimum days in num_valid_days (at least 15 days apart from padding days + at least 5% of days)

        if (num_valid_days <= (min_wh_days + 2 * padding_days)) or (
                ((num_valid_days - 2 * padding_days) / len(wh_present_indexes)) <= min_wh_percent):
            # if very less days then correlation is nan

            moving_band_corr.append(np.nan)
            starting_time += factor
            continue

        # Get energy of only wh present days

        band_energy = band_energy[valid_days_bool]

        # Smooth the energy band

        smooth_band_energy = savgol_filter(band_energy, 5, 1, mode='nearest')

        # Get the correlation of band energy and wh potential

        correlation = np.round(np.corrcoef(smooth_band_energy, wh_potential[valid_days_bool])[0, 1], 2)

        # append the correlation to moving_band_corr

        moving_band_corr.append(correlation)

        starting_time += factor

    logger.info('Max correlation of WH band obtained | %.1f ', np.nanmax(moving_band_corr))

    # ------------------------------------------ STAGE 5: ELIMINATE TIME BANDS ----------------------------------------

    if debug.get('disagg_mode') == 'mtd':
        # If mtd mode, then get the time zones detected in historical/incremental mode

        time_zones, local_max_idx = get_time_zones_mtd(hsm_in, logger)

    else:
        # Get only those time zones which satisfy min correlation threshold

        time_zones, local_max_idx = get_time_zones(moving_band_corr, correlation_thr, seasonal_wh_config)

    # Store time zones in debug

    debug['swh_runs'] = len(time_zones)
    debug = store_in_debug(time_zones, local_max_idx, debug)

    # ------------------------------------------ STAGE 6: TIME ZONES SCORING ------------------------------------------

    band_scores = []
    total_detections = 0

    if len(time_zones) > 0:

        # Initialise all the necessary variables
        combine_days = np.full(shape=swh_data_matrix.shape, fill_value=0.0)
        start_time_idx = []
        end_time_idx = []
        time_zone_band_scores = []
        max_time_div = seasonal_wh_config.get('config').get('max_time_div')

        for i in range(len(time_zones)):
            temp = np.full(shape=swh_data_matrix.shape, fill_value=0.0)
            start_time = int(time_zones_div[time_zones[i][0]][0] / (max_time_div / factor))
            end_time = int(time_zones_div[time_zones[i][1]][1] / (max_time_div / factor))
            temp[:, start_time: end_time] = deepcopy(swh_data_matrix[:, start_time: end_time])

            # ------------------------------------------ STEP 1: CONSISTENCY CHECK ------------------------------------

            # Get consistency values

            winter_consistency, start_hod_count, max_median = get_consistency(temp, start_time, end_time, debug,
                                                                              seasonal_wh_config)

            std_diff = min(np.std(abs(np.diff(np.r_[0, start_hod_count]))) * 10, 1)

            # ------------------------------------------ STEP 2: WRONG DAYS CHECK -------------------------------------

            # Calculate the day level consumption by aggregating at epoch level

            band_energy = np.sum(temp[:, start_time: end_time], axis=1)

            # Minimum consumption check

            check_1_bool = band_energy < min_amplitude

            # Maximum consumption check

            check_2_bool = band_energy - 1000 * (max(time_zones[i][1] - time_zones[i][0] - 1, 0)) > max_amplitude

            # Get the wrong days which are present in wh_present_indexes

            wrong_days_idx = (check_1_bool | check_2_bool) & wh_present_indexes

            # Calculate the wrong days percentage out of wh_present_indexes

            wrong_days_perc = np.round(np.sum(wrong_days_idx * 1) / np.sum(wh_present_indexes * 1), 2)

            # ------------------------------------------ STEP 3: BAND CORRELATION -------------------------------------

            # Get band correlation which is the correlation of local maxima of that time zone

            band_corr = np.max(moving_band_corr[int(time_zones[i][0]): max(int(time_zones[i][1]) + 1, len(moving_band_corr))])

            # ------------------------------------------ STEP 4: TIME BAND PROBABILITY --------------------------------

            tb_prob = time_band_prob[int(local_max_idx[0][i])]

            # ------------------------------------------ STEP 5: REMOVE BOXES BASED ON DURATION -----------------------

            cleaned_data, new_wh_present_idx = get_duration(swh_data_matrix, start_time, end_time, seasonal_wh_config,
                                                            debug)

            # ------------------------------------------ STEP 6: CALCULATE AMPLITUDE RANGE ----------------------------

            get_energy_range_output = get_energy_range(cleaned_data, start_time, end_time, new_wh_present_idx,
                                                       wrong_days_idx, debug, seasonal_wh_config)

            energy_diff, low_amp, high_amp, final_run_data, new_wh_present_idx = get_energy_range_output

            debug['estimation_wh_present_idx'] = deepcopy(new_wh_present_idx)

            # ------------------------------------------ STEP 7: SCORING ----------------------------------------------

            score = (seasonal_wh_config['config']['wrong_days_perc_weight'] * (1 - wrong_days_perc)
                     + seasonal_wh_config['config']['winter_consistency_weight'] * winter_consistency
                     + seasonal_wh_config['config']['tb_prob_weight'] * tb_prob
                     + seasonal_wh_config['config']['std_diff_weight'] * std_diff
                     + seasonal_wh_config['config']['max_median_weight'] * max_median
                     - seasonal_wh_config['config']['c_weight'])

            # Apply sigmoid function

            score = 1 / (1 + np.exp(-score))

            # ------------------------------------------ STEP 8: DETECTION --------------------------------------------

            detection_thr = seasonal_wh_config['config']['detection_thr']

            # For mtd mode check its detection correctness with historical/incremental mode band scores

            score = get_score_for_mtd(debug, hsm_in, score, i, detection_thr)

            # Do a sanity check for score

            if np.sum(final_run_data) <= 0:
                score = seasonal_wh_config['config']['sanity_fail_detection_thr']

            # If score is greater than the threshold it is a potential time band

            if score >= detection_thr:
                combine_days[:, start_time: end_time] = final_run_data[:, start_time: end_time]
                debug['swh_run' + str(total_detections) + '_estimation'] = final_run_data
                debug['swh_run' + str(total_detections) + '_detection'] = 1
                debug['swh_run' + str(total_detections) + '_band_corr'] = band_corr
                debug['swh_run' + str(total_detections) + '_wrong_days'] = wrong_days_perc
                debug['swh_run' + str(total_detections) + '_tb_prob'] = tb_prob
                debug['swh_run' + str(total_detections) + '_winter_consistency'] = np.round(winter_consistency, 2)
                debug['swh_run' + str(total_detections) + '_max_median'] = np.round(max_median, 2)
                debug['swh_run' + str(total_detections) + '_energy_diff'] = np.round(energy_diff, 2)
                debug['swh_run' + str(total_detections) + '_score'] = np.round(score, 2)
                debug['swh_run' + str(total_detections) + '_e_range'] = [np.round(low_amp, 2), np.round(high_amp, 2)]
                debug['swh_run' + str(total_detections) + '_start_time'] = start_time/factor
                debug['swh_run' + str(total_detections) + '_end_time'] = end_time/factor
                debug['swh_run' + str(total_detections) + '_consumption'] = np.round(np.sum(final_run_data), 2)
                start_time_idx.append(start_time/factor)
                end_time_idx.append(end_time/factor)
                debug['swh_start_time'] = start_time_idx
                debug['swh_end_time'] = end_time_idx

                band_scores.append(score)
                total_detections += 1

            time_zone_band_scores.append(score)

        debug['band_scores'] = band_scores
        debug['time_zone_band_scores'] = time_zone_band_scores
        logger.info('Total number of WH bands detected | %.1f ', total_detections)

        # Perform post processing

        final_low_amp = 0
        final_high_amp = 0
        final_swh_amplitude = 0
        swh_confidence = 0

        if total_detections >= 1:
            combine_days, final_low_amp, final_high_amp = post_processing(combine_days, debug, seasonal_wh_config, logger_pass)
            debug['final_start_time'] = start_time_idx[0]
            debug['final_end_time'] = end_time_idx[-1]
            final_swh_amplitude = np.round(np.median(combine_days[combine_days > 0]), 2)
            swh_confidence = np.round(np.mean(band_scores), 2)

        debug['final_estimation'] = combine_days
        debug['final_consumption'] = np.round(np.sum(combine_days), 2)
        debug['total_detections'] = total_detections
        debug['swh_correlation'] = moving_band_corr
        debug['band_scores'] = band_scores
        debug['final_low_amp'] = final_low_amp
        debug['final_high_amp'] = final_high_amp
        debug['final_e_range'] = [np.round(final_low_amp, 2), np.round(final_high_amp, 2)]
        debug['final_swh_amplitude'] = final_swh_amplitude
        debug['swh_confidence'] = swh_confidence

    else:
        combine_days = np.full(shape=swh_data_matrix.shape, fill_value=0.0)
        debug['final_estimation'] = combine_days
        debug['final_consumption'] = 0
        debug['total_detections'] = 0
        debug['swh_correlation'] = moving_band_corr
        debug['final_low_amp'] = 0
        debug['final_high_amp'] = 0
        debug['final_e_range'] = [0, 0]
        debug['band_scores'] = []
        debug['final_swh_amplitude'] = 0
        debug['time_zone_band_scores'] = []
        debug['swh_confidence'] = 0

    debug = update_detection_status(total_detections, debug, logger)

    logger.info('Total consumption for Seasonal WH | %.2f ', debug['final_consumption'])

    return debug
