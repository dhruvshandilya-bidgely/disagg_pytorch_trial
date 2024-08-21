"""
Author - Mayank Sharan
Date - 15/9/19
Label type 1 vacation days
"""

# Import python packages

import logging
import numpy as np

# Import functions from within the project

from python3.disaggregation.aer.vacation.functions.compute_confidence import compute_power_chk_conf
from python3.disaggregation.aer.vacation.functions.compute_confidence import compute_final_chk_conf
from python3.disaggregation.aer.vacation.functions.get_probable_type_1_days import get_probable_type_1_days
from python3.disaggregation.aer.vacation.functions.check_moving_window_power import check_moving_window_power
from python3.disaggregation.aer.vacation.functions.get_confirmed_type_1_days import get_confirmed_type_1_days


def label_type_1(day_data, day_wise_power, day_wise_baseload, day_valid_mask_cons, vac_confidence, debug,
                 vacation_config, logger_pass):

    """
    Perform checks and identify days that qualify as type 1 vacation

    Parameters:
        day_data            (np.ndarray)        : Day wise data matrix
        day_wise_power      (np.ndarray)        : Power computed corresponding to each day
        day_wise_baseload   (np.ndarray)        : Baseload computed corresponding to each day
        day_valid_mask_cons (np.ndarray)        : Day wise boolean array denoting points masked as timed device usage
        vac_confidence      (np.ndarray)        : Confidence values for each day being vacation
        debug               (dict)              : Contains all variables needed for debugging
        vacation_config     (dict)              : Contains all configuration variables needed for vacation
        logger_pass         (dict)              : Contains the logger and the logging dictionary to be passed on

    Returns:
        type_1_bool         (np.ndarray)        : Day wise label with type 1 vacation days
        debug               (dict)              : Contains all variables needed for debugging
        vac_confidence      (np.ndarray)        : Confidence values for each day being vacation
    """

    # ------------------------------------------ STAGE 1: INITIALISATIONS ---------------------------------------------

    # Initialize logger

    logger_base = logger_pass.get('base_logger').getChild('label_type_1')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Prepare logger pass to pass to sub-modules

    logger_pass['base_logger'] = logger_base

    # Initialize variables to be used

    num_days = len(day_wise_baseload)
    confidence_config = vacation_config.get('static_confidence')

    # Debug variables capturing the power loop computation

    power_chk_thr = np.full(shape=(num_days,), fill_value=np.nan)
    loop_break_idx = np.full(shape=(num_days,), fill_value=np.nan)
    loop_break_power = np.full(shape=(num_days, 2), fill_value=np.nan)
    sliding_power = np.full(shape=(num_days, 12), fill_value=np.nan)

    # Debug variables filled for confirmation stage at the day level

    std_dev_all = np.full(shape=(num_days,), fill_value=np.nan)
    std_dev_thr = np.full(shape=(num_days,), fill_value=np.nan)
    perc_arr_all = np.full(shape=(num_days,), fill_value=np.nan)
    max_3_dev_all = np.full(shape=(num_days, 3), fill_value=np.nan)
    sliding_mean_thr = np.full(shape=(num_days,), fill_value=np.nan)
    sliding_power_mean_all = np.full(shape=(num_days,), fill_value=np.nan)

    # Boolean arrays for stages of selection

    power_check_passed_bool = np.full(shape=day_wise_baseload.shape, fill_value=False)
    confirmed_vac_bool = np.full(shape=day_wise_baseload.shape, fill_value=False)
    type_1_bool = np.full(shape=day_wise_baseload.shape, fill_value=False)

    # Populate the debug dictionary at this stage with initial values

    debug['labeling'] = {
        'std_dev_thr': std_dev_thr,
        'std_dev_all': std_dev_all,
        'perc_arr_all': perc_arr_all,
        'max_3_dev_all': max_3_dev_all,
        'sliding_power': sliding_power,
        'power_chk_thr': power_chk_thr,
        'loop_break_idx': loop_break_idx,
        'sliding_mean_thr': sliding_mean_thr,
        'loop_break_power': loop_break_power,
        'sliding_power_mean_all': sliding_power_mean_all,
        'power_check_passed_bool': power_check_passed_bool,
    }

    # ------------------------------------------ STAGE 2: PROBABLE VACATION -------------------------------------------

    # Get probable type 1 vacation days

    probable_vac_bool, probable_threshold = get_probable_type_1_days(day_wise_baseload, day_wise_power, vacation_config)

    # Mark confidence as 0 for days not even selected as probable vacation days

    vac_confidence[np.logical_not(probable_vac_bool)] = confidence_config.get('not_probable_conf')

    # Log number of days identified as probable vacation days

    num_probable_days = np.sum(probable_vac_bool)
    logger.info('Number of probable vacation days are | %d', num_probable_days)

    # Populate the debug dictionary with probable day detection related variables

    debug['labeling']['probable_vac_bool'] = probable_vac_bool
    debug['labeling']['probable_threshold'] = probable_threshold

    # If no days are probable vacation days return from here

    if num_probable_days == 0:
        return type_1_bool, debug, vac_confidence

    # ------------------------------------------ STAGE 3: WINDOW POWER CHECK ------------------------------------------

    # Run power check loop for each day and identify the ones that pass along with other associated values

    power_check_passed_bool, sliding_power, sliding_power_bkp, loop_break_idx, loop_break_power, power_chk_thr = \
        check_moving_window_power(day_data, day_wise_baseload, day_valid_mask_cons, probable_vac_bool, loop_break_idx,
                                  loop_break_power, vacation_config)

    # Identify the days rejected in the power check loop and compute confidence for rejected days

    rejected_in_power_chk = np.logical_and(probable_vac_bool, np.logical_not(power_check_passed_bool))

    vac_confidence = compute_power_chk_conf(rejected_in_power_chk, loop_break_power, power_chk_thr, vac_confidence,
                                            vacation_config)

    # Identify wild card days to be considered that just missed selection in power check. Only for Europe

    if vacation_config.get('user_info').get('is_europe'):

        # Days that pass the wild card threshold are marked as selected and get complete power array restored

        wild_card_days_bool = vac_confidence >= confidence_config.get('wild_card_conf_thr')

        power_check_passed_bool[wild_card_days_bool] = True
        sliding_power[wild_card_days_bool, :] = sliding_power_bkp[wild_card_days_bool, :]

        # Log the number of wild card days identified

        num_wild_card_days = np.sum(wild_card_days_bool)
        logger.info('Days selected as wild cards are | %d', num_wild_card_days)

    # Log the number of days continuing as candidates for type 1 vacation at this stage

    num_power_check_passed_days = np.sum(power_check_passed_bool)
    logger.info('Days in consideration for vacation after window power check | %d', num_power_check_passed_days)

    # Populate the debug dictionary with window power check related variables

    debug['labeling']['sliding_power'] = sliding_power
    debug['labeling']['power_chk_thr'] = power_chk_thr
    debug['labeling']['loop_break_idx'] = loop_break_idx
    debug['labeling']['loop_break_power'] = loop_break_power
    debug['labeling']['power_check_passed_bool'] = power_check_passed_bool

    # If no candidates for vacation remain return from here

    if num_power_check_passed_days == 0:
        return type_1_bool, debug, vac_confidence

    # ------------------------------------------ STAGE 4: VACATION CONFIRMATION ---------------------------------------

    # Perform final checks and identify days to be marked as confirmed vacation

    confirmed_vac_bool, sliding_power_mean, std_dev_arr, max_3_dev_arr, confidence_params = \
        get_confirmed_type_1_days(day_wise_baseload, power_check_passed_bool, sliding_power, confirmed_vac_bool,
                                  vacation_config)

    # Log number of days confirmed as vacation

    num_confirmed_days = np.sum(confirmed_vac_bool)
    logger.info('Number of days confirmed as type 1 vacation | %d', num_confirmed_days)

    # Populate confidence values for all days that were candidates

    vac_confidence, perc_arr = compute_final_chk_conf(power_check_passed_bool, confirmed_vac_bool, confidence_params,
                                                      vac_confidence, vacation_config)

    # Mark all days that have a confidence higher than the threshold as vacation

    vac_confidence = np.round(vac_confidence, 2)
    type_1_bool = vac_confidence >= confidence_config.get('vacation_selection_thr')

    # Log number of days identified as type 1 vacation

    num_type_1_days = np.sum(type_1_bool)
    logger.info('Number of days marked as type 1 vacation | %d', num_type_1_days)

    # Fill arrays for values that were calculated only for candidate days

    perc_arr_all[power_check_passed_bool] = perc_arr
    std_dev_all[power_check_passed_bool] = std_dev_arr
    max_3_dev_all[power_check_passed_bool, :] = max_3_dev_arr
    sliding_power_mean_all[power_check_passed_bool] = sliding_power_mean

    # Populate the debug dictionary with the new available and updated variables

    debug['labeling']['std_dev_all'] = std_dev_all
    debug['labeling']['perc_arr_all'] = perc_arr_all
    debug['labeling']['max_3_dev_all'] = max_3_dev_all
    debug['labeling']['sliding_power_mean_all'] = sliding_power_mean_all

    debug['labeling']['std_dev_thr'] = confidence_params.get('std_dev_thr')
    debug['labeling']['sliding_mean_thr'] = confidence_params.get('sliding_mean_thr')

    return type_1_bool, debug, vac_confidence
