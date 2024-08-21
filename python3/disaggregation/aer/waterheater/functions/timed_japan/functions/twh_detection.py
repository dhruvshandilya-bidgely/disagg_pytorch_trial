"""
Author - Sahana M
Date - 07/06/2021
Perform Timed water heater detection
"""

# Import python packages
import logging
import numpy as np

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import chunk_indexes
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.wh_potential import get_wh_potential
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.get_instance_features import \
    get_instance_features


def check_incremental_mtd(score, features, wh_config, debug):
    """
    This function is used to make use of the historical run information in mtd and incremental mode to update band scores
    Parameters:
        score               (float)             : Score of the time band
        features            (dict)              : Features dictionary for the time band
        wh_config           (dict)              : WH configurations dictionary
        debug               (dict)              : Contains algorithm output

    Returns:
        score               (float)             : Updated score of the time band
    """

    previous_confidence_weight = wh_config.get('previous_confidence_weight')

    # Check if the hsm exists
    if debug.get('hsm_in') is not None:

        # get all the hsm information

        historical_time_bands = debug.get('hsm_in').get('twh_time_bands')
        current_start_time = int(features['start_time'])
        current_end_time = int(features['end_time'])

        # Check if mode is incremental or mtd
        update_score_bool = False

        if (debug.get('disagg_mode') in ['incremental', 'mtd']) and \
                ((type(historical_time_bands) is np.ndarray) or
                 (type(historical_time_bands) is list)) and \
                (debug.get('hsm_in').get('timed_confidence_score') is not None):
            update_score_bool = True

        # If score is below threshold but was detected in the historical run, update the score

        if update_score_bool and np.sum(historical_time_bands[current_start_time:current_end_time]):
            previous_confidence = debug.get('hsm_in').get('timed_confidence_score')[0]
            if score < wh_config['detection_thr']:
                score = min((score + previous_confidence_weight * previous_confidence), 1.0)

        # Check if mode is mtd to avoid FP time bands

        update_score_mtd_bool = False

        if (debug.get('disagg_mode') in ['mtd']) and \
                ((type(historical_time_bands) is np.ndarray) or
                 (type(historical_time_bands) is list)):
            update_score_mtd_bool = True

        if update_score_mtd_bool and np.sum(historical_time_bands[current_start_time:current_end_time]) == 0:
            score = 0

    return score


def twh_detection(debug, wh_config, logger_base):
    """
    This function detects the presence of timed water heater
    Parameters:
        debug                   (dict)              : Contains algorithm output
        wh_config               (dict)              : WH configurations dictionary
        logger_base             (logger)            : Logger passed

    Returns:
        debug                   (dict)              : Contains algorithm output
    """

    # Initialise logger for this module

    logger_local = logger_base.get('logger').getChild('twh_detection')
    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract the necessary data

    vacation_matrix = debug.get('vacation_matrix')
    overall_chunk_data = debug.get('overall_chunk_data')
    in_masked_data_matrix = debug.get('masked_cleaned_data')
    final_detection_matrix = np.full_like(in_masked_data_matrix, fill_value=0.0)

    # Get the vacation days

    vacation_days = np.sum(vacation_matrix, axis=1)
    vacation_days_bool = vacation_days > 0
    debug['vacation_days_bool'] = vacation_days_bool

    # Initialise default values

    final_score = 0
    twh_bands = []
    debug['wda_time'] = 0
    debug['bands_info'] = {}
    twh_time_bands = np.full(shape=wh_config.get('cols'), fill_value=False)
    twh_time_band_scores = []

    # Get all the weather related info

    input_data = debug.get('input_data')
    wh_potential, fl, season_detection_dict, debug = get_wh_potential(input_data, wh_config, debug, logger_pass)
    weather_info = {
        'fl': fl,
        'wh_potential': wh_potential,
        'season_detection_dict': season_detection_dict
    }

    # For each instance sequence get the corresponding features

    num_of_merged_instance = np.unique(overall_chunk_data[:, chunk_indexes['merged_seq']])

    logger.info('Number of bands before detection | {} '.format(len(num_of_merged_instance)))

    features = {}

    for idx in num_of_merged_instance:

        instances = overall_chunk_data[overall_chunk_data[:, chunk_indexes['merged_seq']] == idx]
        features, debug = get_instance_features(instances, weather_info, debug, wh_config, logger_pass)

        # CALCULATE SCORE

        score = (0.46 * features.get('auc_wh_pot_corr')
                 + 0.88 * features.get('one_sided_seasonality_score')
                 + 0.98 * features.get('reverse_seasonality_score')
                 + 1.63 * features.get('double_dip_score')
                 + 3.55 * features.get('continuity_score')
                 + 0.95 * features.get('auc_std')
                 + 0.91 * features.get('dur_std')
                 + 0.55 * features.get('amp_std')
                 + 3.02 * features.get('final_tb_prob')
                 + 2.09 * features.get('max_consistency')
                 + 0.65 * features.get('max_median_consistency')) \
                - 11.79

        score = np.round(1 / (1 + np.exp(-score)), 2)

        # Check for incremental mtd

        score = check_incremental_mtd(score, features, wh_config, debug)

        logger.info('Score for the band idx {} is | {} '.format(idx, score))

        final_score = max(score, final_score)

        # check if the band satisfies the detection threshold

        if score >= wh_config.get('detection_thr'):
            final_detection_matrix = np.maximum(final_detection_matrix, features.get('masked_data_matrix'))
            twh_bands.append(idx)
            debug['num_runs'] += 1
            twh_time_bands[features.get('start_time'): features.get('end_time')] = True
            twh_time_band_scores.append(score)

        # Store the band level infos in debug

        debug['bands_info']['band_' + str(idx)] = {
            'score_' + str(idx): score,
            'end_time_' + str(idx): features.get('end_time'),
            'start_time_' + str(idx): features.get('start_time'),
            'duration_line_' + str(idx): features.get('duration_line'),
            'auc_line_' + str(idx): features.get('area_under_curve_line'),
            'amplitude_line_box_' + str(idx): features.get('amplitude_line'),
            'amplitude_line_raw_' + str(idx): features.get('amplitude_line_raw'),
            'masked_data_matrix_' + str(idx): features.get('masked_data_matrix'),
            'features_' + str(idx): features
        }

    if final_score >= wh_config.get('detection_thr'):
        debug['timed_hld'] = 1

    logger.info('Total number of bands detected before HLD checks | {} '.format(debug.get('num_runs')))

    # Store all the info in debug

    debug['twh_bands'] = twh_bands
    debug['timed_confidence'] = final_score
    debug['twh_time_bands'] = twh_time_bands
    debug['season_label'] = features.get('season_label')
    debug['wh_potential'] = features.get('wh_potential')
    debug['vacation_days_bool'] = vacation_days_bool
    debug['final_twh_matrix'] = final_detection_matrix
    debug['twh_time_band_scores'] = twh_time_band_scores

    logger.info('Timed Water Heater Confidence before HLD checks | {} '.format(debug.get('timed_confidence')))

    return debug
