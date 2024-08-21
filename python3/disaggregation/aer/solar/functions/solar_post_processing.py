"""
Author - Paras Tehria
Date - 12/11/19
This module contains functions to tackle obvious mis-classifications by cnn model
"""

# Import python packages

import logging
import datetime
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.utils.maths_utils.find_seq import find_seq
from python3.utils.time.get_time_diff import get_time_diff


def compute_num_disconnection(input_data, solar_config):
    """
    Utility to compute parameters regarding disconnections in data
        Parameters:
            input_data                     (np.ndarray)       : consumption 2d array
            solar_config                    (dict)             : config for solar detection
        Return:
            num_constant_cons              (int)              : Number of constant consumption hours
    """
    # TODO: (Paras) use standard method for disconnections as per DS standards, code present in check_data_quality file

    # number of constant consumption hours
    num_constant_cons = 0
    input_arr = deepcopy(input_data)

    # Converting to 1d array to find disconnections

    input_arr = input_arr[input_arr != -1].ravel()

    num_allowed_constant = solar_config.get('solar_post_processing').get('num_allowed_constant')
    split_cons_diff_data = np.diff(input_arr)
    diff_seq = find_seq(split_cons_diff_data, num_allowed_constant)

    if len(diff_seq > 0):
        diff_seq_constant = diff_seq[diff_seq[:, 0] == 0, :]

        num_constant_cons += np.sum(diff_seq_constant[:, 3])

    return num_constant_cons


def detect_misclassification(prob, idx, solar_config, zeros_daytime_ratio, zeros_night_ratio, edge_fluc, num_nzero_days,
                             logger):
    """
    This function detects obvious mis-classifications based on thresholds
    Parameters:
        prob                        (float)             : cnn output probability for the instance
        idx                         (int)              : index of the current instance
        solar_config                 (dict)             : config for solar detection
        zeros_daytime_ratio         (float)             : ratio of zeros during sunlight hours
        zeros_night_ratio           (float)             : ratio of zeros during non-sunlight hours
        edge_fluc                    (int)              : variation in time between sunrise and first zero consump value
        num_nzero_days              (int)              : number of days with no zeros during sunlight hours
        logger                      (dict)             : logger of this function
    Return:
        mask_disconn               (list)          :       containing chunks with disconnections
        probability_solar          (list)          :       detection probabilities before disconnections removal
        probability_solar_after_disconn (list)     :       detection probabilities after disconnections removal
    """

    # Getting post-processing parameters from config
    ns_zero_ratio_night = solar_config.get('solar_post_processing').get('ns_zero_ratio_night')
    s_zero_ratio_day = solar_config.get('solar_post_processing').get('s_zero_ratio_day')
    s_zero_ratio_night = solar_config.get('solar_post_processing').get('s_zero_ratio_night')
    s_edge_fluct = solar_config.get('solar_post_processing').get('s_edge_fluct')
    s_non_zero_days = solar_config.get('solar_post_processing').get('s_non_zero_days')
    ns_convert_prob = solar_config.get('solar_post_processing').get('ns_convert_prob')
    s_convert_prob = solar_config.get('solar_post_processing').get('s_convert_prob')

    if prob > solar_config.get('solar_disagg').get('detection_threshold'):
        # Basic checks to detect clear FP cases

        n_solar = False

        # if zeros values (in day level min-max normalised consumption) are more in night-time than day-time
        # or too many zeros in night-time, tag the current chunk as non-solar.

        if ((zeros_daytime_ratio < zeros_night_ratio) or (zeros_night_ratio > ns_zero_ratio_night)) and (
                zeros_daytime_ratio < 2 * zeros_night_ratio):
            n_solar = True

        if n_solar:
            logger.info('converting instance number {} to non-solar | '.format(idx))
            logger.info(
                'zeros_daytime_ratio, zeros_night_ratio:{},{} | '.format(zeros_daytime_ratio, zeros_night_ratio))
            prob = ns_convert_prob

    elif prob <= solar_config.get('solar_disagg').get('detection_threshold'):
        # Basic checks to detect clear FN cases

        solar = False

        # If high number of zeros (in day level min-max normalised consumption) during day-time and
        # a consistent edge is created by zero-value, tag the chunk as solar.
        if zeros_daytime_ratio > s_zero_ratio_day and zeros_night_ratio < s_zero_ratio_night \
                and edge_fluc < s_edge_fluct and num_nzero_days < s_non_zero_days:
            solar = True

        if solar:
            logger.info('converting instance number {} to solar | '.format(idx))
            prob = s_convert_prob

    return prob


def solar_post_process(cnn_detection_array, probability_solar, solar_config, logger_base):
    """
    This is the solar post processing function used to tackle obvious mis-classifications by cnn model
    Parameters:
        cnn_detection_array             (np.ndarray)       : detection 2d array
        probability_solar               (np.array)         : cnn output probability
        solar_config                    (dict)             : config for solar detection
        logger_base                     (dict)             : logger object to generate logs
    Return:
        mask_disconn                    (list)             :       containing chunks with disconnections
        probability_solar               (list)             :       detection probabilities before disconnections removal
        probability_solar_after_disconn (list)             :       detection probabilities after disconnections removal

    """

    logger_local = logger_base.get('logger').getChild('solar_post_process')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Getting post-processing parameters from config
    window = solar_config.get('solar_post_processing').get('window')
    lower_confl_prob = solar_config.get('solar_post_processing').get('lower_confl_prob')
    high_confl_prob = solar_config.get('solar_post_processing').get('high_confl_prob')
    constant_cons_perc_thresh = solar_config.get('solar_post_processing').get('constant_cons_perc_thresh')

    # checking disconnection for instance with solar detected

    disconn_idx = []

    benchmark_time = datetime.datetime.now()

    consumption_arr_idx = solar_config.get('consumption_arr_idx')
    sunlight_arr_idx = solar_config.get('sunlight_arr_idx')

    # Checking disconnections in chunks
    for idx in range(len(probability_solar)):

        cons_matrix = deepcopy(cnn_detection_array[idx, :][:, :, consumption_arr_idx])
        sunlight_matrix = deepcopy(cnn_detection_array[idx, :][:, :, sunlight_arr_idx])

        # masking sunlight hours consumption as
        cons_matrix[sunlight_matrix.astype(bool)] = -1

        # converting perfect 1 consumption (in min-max normalised data) to 0
        # because multiple 1's signifies same max value multiple number of times

        cons_matrix[cons_matrix == 1.0] = 0.0

        num_constant_cons = compute_num_disconnection(cons_matrix, solar_config)

        constant_cons_perc = num_constant_cons / np.count_nonzero(cons_matrix >= 0)

        if constant_cons_perc > constant_cons_perc_thresh:
            disconn_idx.append(idx)

    benchmark_time = get_time_diff(benchmark_time, datetime.datetime.now())
    logger.debug('Timing: Total time for finding disconnections is | %0.3f', benchmark_time)

    logger.info('Total number of disconnection instances:  | {}'.format(len(disconn_idx)))

    # Array where indices of disconnection chunks are marked as False
    mask_disconn = np.ones(len(probability_solar), dtype=bool)
    mask_disconn[disconn_idx] = False

    # writing probability as -2 for disconnections to right appropriately to the hsm
    probability_solar_after_disconn = deepcopy(probability_solar)
    probability_solar_after_disconn[disconn_idx] = -2
    probability_solar = probability_solar[mask_disconn]

    logger.debug('Total number of non-disconnection instances: | {}'.format(len(probability_solar)))

    # Checking boundary cases where detection probability is close to threshold
    conflict_neg_idx = \
        np.where(np.logical_and(probability_solar >= lower_confl_prob, probability_solar <= high_confl_prob))[0]

    cnn_detection_array = cnn_detection_array[mask_disconn]

    # Using loop as len of array will be small

    for idx in conflict_neg_idx:
        cons_matrix = deepcopy(cnn_detection_array[idx, :][:, :, 0])
        sunlight_matrix = deepcopy(cnn_detection_array[idx, :][:, :, 1])

        # very low scaled consumption capped to zero

        normalised_zero_thresh = solar_config.get('solar_post_processing').get('normalised_zero_thresh')
        cons_matrix[cons_matrix <= normalised_zero_thresh] = 0

        # ratio of zero during daytime (time with sunlight present) and nighttime (time with sunlight not present)

        zeros_daytime = np.logical_and(cons_matrix == 0, sunlight_matrix == 1)
        zeros_daytime_ratio = np.sum(zeros_daytime) / np.sum(sunlight_matrix == 1)

        zeros_night = np.logical_and(cons_matrix == 0, sunlight_matrix == 0)
        zeros_night_ratio = np.sum(zeros_night) / np.sum(sunlight_matrix == 0)

        # calculating time diff between first two successive zeros in consumption and sunrise
        # to capture if consumption dips to zero every day with very less deviation in time taken to reach zero

        # marking non-sunlight time consumption as negative 1
        cons_matrix[~sunlight_matrix.astype(bool)] = -1

        # window sum to detect two consecutive zeroes
        cumsum = (cons_matrix == 0).astype(int).cumsum(axis=1)
        cumsum[:, window:] = cumsum[:, window:] - cumsum[:, :-window]

        # distance between sunrise and two consecutive zeroes to capture curve
        first_idx_consec_zero = np.argmax(np.array(cumsum == window).astype(int), axis=1) - 1

        # index of sunrise
        sunrise_idx = np.argmax(sunlight_matrix, axis=1)

        # daily distance of non zero
        diff_zero_sunrise = first_idx_consec_zero - sunrise_idx
        pos_sunrise_diff = diff_zero_sunrise[diff_zero_sunrise >= 0]

        # edge fluctuation is calculated as l1 distance from median
        med_dist = np.median(pos_sunrise_diff)
        edge_fluc = np.sum(np.abs(pos_sunrise_diff - med_dist))

        num_days_insta = solar_config.get('solar_post_processing').get('num_days_insta')

        if pos_sunrise_diff.size > 0:
            edge_fluc *= num_days_insta / pos_sunrise_diff.size
        else:
            edge_fluc = np.inf

        # non-zero days that do not contain two consecutive zeroes or zeroes are very far from median dist
        non_zero_days = np.where(first_idx_consec_zero <= 0)[0]

        num_nzero_days = len(non_zero_days)

        # Update probabilities of wrongly-tagged instances
        probability_solar[idx] = detect_misclassification(probability_solar[idx], idx, solar_config,
                                                          zeros_daytime_ratio, zeros_night_ratio, edge_fluc,
                                                          num_nzero_days, logger)

    probability_solar_after_disconn[probability_solar_after_disconn != -2] = probability_solar

    return mask_disconn, probability_solar, probability_solar_after_disconn
