"""
Author - Sahana M
Date - 19-Oct-2023
Module for performing data sanity checks
"""

import logging
import numpy as np
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.code_utils import get_2d_matrix
from python3.utils.maths_utils.find_seq import find_seq


def final_hld_sanity_checks(debug, negate_detection_status, ev_config, logger_base):
    """
    Function used to perform final data sanity checks for outlier usage FPs
    Parameters:
        debug                       (Dict)          : Debug dictionary
        negate_detection_status     (Boolean)       : Data Sanity issue boolean
        ev_config                   (Dict)          : EV configurations dictionary
        logger_base                 (Logger)        : Logger
    Returns:
        debug                       (Dict)          : Debug dictionary
        negate_detection_status     (Boolean)       : Data Sanity issue boolean
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('final_hld_sanity_checks')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))
    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}

    # ----------------------------------------- Outlier data users check ----------------------------------------------

    auc_violation, amp_violation, duration_violation = check_outlier_data(debug, ev_config, logger_pass)

    if amp_violation or duration_violation or auc_violation:
        negate_detection_status = True
        debug['ev_hld'] = 0
        debug['ev_probability'] = 0.2
        debug['charger_type'] = 'None'
        debug['ev_amplitude'] = 0
        debug['mean_duration'] = 0
        logger.info('Data Sanity Check : Amplitude/Duration/AUC criteria violated, turning the EV detection to 0 |')

    # ----------------------------------------- NSP Winter seasonality check ------------------------------------------

    if ev_config.get('pilot_id') in ev_config.get('nsp_winter_seasonality_configs').get('wtr_seasonal_pilots'):
        winter_device = winter_seasonality_check(debug.get('final_ev_signal'), ev_config)

        if winter_device:
            negate_detection_status = True
            debug['ev_hld'] = 0
            debug['ev_probability'] = 0.4
            debug['charger_type'] = 'None'
            debug['ev_amplitude'] = 0
            debug['mean_duration'] = 0
            logger.info('NSP Seasonality check : A highly seasonal winter device detected, hence turning EV detection '
                        'to 0 |')

    return debug, negate_detection_status


def check_outlier_data(debug, ev_config, logger_base):
    """
    Function to check for users where the data is outlierish
    Parameters :
        debug                       (Dict)          : Debug dictionary
        ev_config                   (Dict)          : EV configurations dictionary
        logger_base                 (Logger)        : Logger
    Returns:
        auc_violation               (Boolean)       : AUC violation flag
        amp_violation               (Boolean)       : Amplitude violation flag
        duration_violation          (Boolean)       : Duration violation flag
    """

    logger_local = logger_base.get('logger').getChild('check_outlier_data')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Initialise the required variables

    auc_violation = False
    amp_violation = False
    duration_violation = False
    ev_amplitude = debug.get('ev_amplitude')
    ev_duration = debug.get('mean_duration')
    ev_max_auc_thr = ev_config.get('data_sanity_configs').get('ev_max_auc_thr')
    ev_max_amp_thr = ev_config.get('data_sanity_configs').get('ev_max_amp_thr')
    ev_max_duration_thr = ev_config.get('data_sanity_configs').get('ev_max_duration_thr')

    # Check if this is just a Data multiplier/ false high outlier amplitude strikes captured as EV

    if ev_amplitude >= ev_max_amp_thr:
        amp_violation = True
        logger.info('Data sanity check : High amplitude FP user detected, hence turning EV detection to 0 | ')

    # Check if this is just a Data multiplier/ false high outlier duration strikes captured as EV

    if ev_duration >= ev_max_duration_thr:
        duration_violation = True
        logger.info('Data sanity check : High duration FP user detected, hence turning EV detection to 0 | ')

    # Check if this is just a Data multiplier/ false high outlier AUC strikes captured as EV

    auc = ev_amplitude * ev_duration
    if auc >= ev_max_auc_thr:
        auc_violation = True
        logger.info('Data sanity check : High AUC FP user detected, hence turning EV detection to 0 | ')

    return auc_violation, amp_violation, duration_violation


def winter_seasonality_check(box_data, ev_config):
    """
    Function to identify highly seasonal device popping up in winters only in NSP
    Parameters:
        box_data                (np.ndarray)            : EV box data identified
        ev_config               (Dict)                  : EV configurations
    Returns:
        winter_device           (Boolean)               : Winter device flag
        debug                   (Dict)                  : Debug dictionary
    """

    # Extract required variables
    min_winter_days = 7
    min_summer_days = 7
    min_winter_ev_strikes = 10
    max_summer_ev_strikes = 3
    winter_proportion_thr = ev_config.get('nsp_winter_seasonality_configs').get('winter_proportion_thr')

    # Get the 2D matrix of the box data

    sampling_rate = ev_config.get('sampling_rate')
    data_matrix, _, _ = get_2d_matrix(box_data, sampling_rate)

    # Get the EV data, Seasons label data

    ev_box_data = data_matrix[Cgbdisagg.INPUT_CONSUMPTION_IDX]
    s_label_data = data_matrix[Cgbdisagg.INPUT_S_LABEL_IDX]
    ev_present_days = (np.sum(ev_box_data, axis=1) > 0).astype(int)

    # Identify the season at day level

    day_season = []
    for i in range(len(s_label_data)):
        values, counts = np.unique(s_label_data[i, :], return_counts=True)
        day_season.append(values[np.argsort(-counts)][0])
    day_season = np.asarray(day_season)

    # Find the sequence of the seasons

    season_seq = find_seq(day_season, min_seq_length=7)

    # Identify the number of days where EV is present for a season

    ev_proportion_for_each_season = []
    for i in range(len(season_seq)):
        ev_proportion = np.sum(ev_present_days[int(season_seq[i, 1]): int(season_seq[i, 2])])
        ev_proportion_for_each_season.append(ev_proportion)
    ev_proportion_for_each_season = np.asarray(ev_proportion_for_each_season)

    season_seq = np.c_[season_seq, ev_proportion_for_each_season]

    # Check for the presence of both Winter & Summer season
    seasons = np.unique(season_seq[:, 0])
    winter_presence = any(val in [-1, -0.5] for val in seasons)
    smr_trans_presence = any(val in [1, 0.5, 0] for val in seasons)

    winter_device = False

    # Check for 2 winters and 1 summer

    if winter_presence and smr_trans_presence:

        # Identify winter indexes and its EV proportion
        wtr_indexes = np.isin(season_seq[:, 0], [-1, -0.5])
        wtr_consumption_percentage = np.sum(season_seq[wtr_indexes, 4])/np.sum(season_seq[:, 4])

        # Check for 2 times Heavy winter in the data & EV usage in both the winters
        heavy_winter_indexes = np.isin(season_seq[:, 0], [-1])
        heavy_winter_indexes = np.logical_and(heavy_winter_indexes, season_seq[:, 3] >= min_winter_days)
        heavy_winter_count = np.sum(heavy_winter_indexes)
        heavy_winter_ev_presence_count = np.sum(season_seq[heavy_winter_indexes, 4] >= min_winter_ev_strikes)

        # Check for Heavy winter usage along with more than 2 winters in the data
        if wtr_consumption_percentage >= winter_proportion_thr and heavy_winter_count >= 2 and \
                heavy_winter_count == heavy_winter_ev_presence_count:
            winter_device = True

    # Check for 2 summers and 1 winter

    if winter_presence and smr_trans_presence:

        # Identify winter indexes and its EV proportion
        wtr_indexes = np.isin(season_seq[:, 0], [-1, -0.5])
        wtr_consumption_percentage = np.sum(season_seq[wtr_indexes, 4]) / np.sum(season_seq[:, 4])

        # Identify the Winter indexes & the corresponding EV proportion
        heavy_winter_indexes = np.isin(season_seq[:, 0], [-1])
        heavy_winter_indexes = np.logical_and(heavy_winter_indexes, season_seq[:, 3] >= min_winter_days)
        heavy_winter_count = np.sum(heavy_winter_indexes)
        heavy_winter_ev_presence_count = np.sum(season_seq[heavy_winter_indexes, 4] >= min_winter_ev_strikes)

        # Identify the Summer indexes & the corresponding EV proportion
        heavy_smr_indexes = np.isin(season_seq[:, 0], [0, 0.5, 1])
        heavy_smr_indexes = np.logical_and(heavy_smr_indexes, season_seq[:, 3] >= min_summer_days)
        heavy_smr_count = np.sum(heavy_smr_indexes)
        heavy_smr_ev_presence_count = np.sum(season_seq[heavy_smr_indexes, 4] <= max_summer_ev_strikes)

        # Check for  1 heavy winter along with more than 2 summers
        wtr_dominance = wtr_consumption_percentage >= winter_proportion_thr and heavy_winter_count == 1 and \
                        heavy_winter_count == heavy_winter_ev_presence_count
        smr_nondominance = heavy_smr_count >= 2 and heavy_smr_ev_presence_count == heavy_smr_count
        wtr_end = ~heavy_winter_indexes[-1]

        if wtr_dominance and smr_nondominance and wtr_end:
            winter_device = True

    return winter_device
