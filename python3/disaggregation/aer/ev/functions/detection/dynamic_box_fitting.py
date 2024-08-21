"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to find high energy boxes in consumption data
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project


from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.ev.functions.get_ev_boxes import get_ev_boxes
from python3.disaggregation.aer.ev.functions.detection.get_boxes_features import boxes_features
from python3.disaggregation.aer.ev.functions.get_season_categorisation import add_season_month_tag
from python3.disaggregation.aer.ev.functions.detection.boxes_sanity_checks import boxes_sanity_checks
from python3.disaggregation.aer.ev.functions.detection.filter_ev_boxes import filter_ev_boxes
from python3.disaggregation.aer.ev.functions.detection.overlapping_box_removal import remove_overlapping_boxes


def dynamic_box_fitting(in_data, ev_config, debug, logger_base):

    """
    Function to find high energy boxes in consumption data

    Parameters:
        in_data                   (np.ndarray)        : Current box data
        debug                     (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
        ev_config                  (dict)              : EV module config
        logger_base               (logger)            : Logging object to log important steps and values in the run

    Returns:
        debug                     (dict)              : Object containing all important data/values as well as HSM
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('dynamic_box_fitting')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger for this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking deepcopy of input data to avoid scoping issues

    input_data_original = deepcopy(in_data)

    sampling_rate = ev_config.get('sampling_rate')

    # Getting config params

    region = ev_config.get('region')

    factor = debug.get('factor')

    minimum_duration = ev_config['minimum_duration'][sampling_rate]

    start_box_amplitude = ev_config['detection'][region + '_start_box_amplitude']
    step_amplitude = ev_config['detection'][region + '_amplitude_step']
    max_amplitude = ev_config.get('detection', {}).get(region + '_max_amplitude')

    box_index = 1
    current_box_amplitude = deepcopy(start_box_amplitude)

    checks_fail = True

    # box_level_index: level 1 is for boxes of the previous iteration, level 2 is for boxes of the current iteration

    box_level_index = 1

    # Getting epoch level calendar month number and season

    input_data_original = add_season_month_tag(input_data_original)

    while checks_fail and current_box_amplitude <= max_amplitude:
        logger.info('Box fitting index in progress | {}'.format(box_index))

        min_energy = current_box_amplitude / factor

        logger.info('Current amplitude (in Watts) | {}'.format(current_box_amplitude))

        current_box_data = get_ev_boxes(input_data_original, minimum_duration, min_energy, factor, logger_pass)
        # Get box features

        current_box_features = boxes_features(current_box_data, factor, ev_config)

        # Saving box data and features for previous iteration

        if box_index > box_level_index:
            debug['box_data_' + str(box_level_index - 1)] = deepcopy(debug.get('box_data_' + str(box_level_index)))
            debug['box_features_' + str(box_level_index - 1)] = deepcopy(debug.get('box_features_' + str(box_level_index)))
            debug['amplitude_' + str(box_level_index - 1)] = deepcopy(debug.get('amplitude_' + str(box_level_index)))

        # Saving box data and features for current iteration

        debug['box_data_' + str(box_level_index)] = deepcopy(current_box_data)
        debug['box_features_' + str(box_level_index)] = deepcopy(current_box_features)
        debug['amplitude_' + str(box_level_index)] = current_box_amplitude

        if box_index > 1:
            current_box_data, current_box_features = remove_overlapping_boxes(current_box_data, current_box_features,
                                                                              box_level_index, min_energy, factor,
                                                                              debug, ev_config, logger_pass)
        # Saving updated box data and features for previous iteration
        if box_index > box_level_index:
            debug['updated_box_data_' + str(box_level_index - 1)] = deepcopy(debug.get('updated_box_data_' + str(box_level_index)))
            debug['updated_box_features_' + str(box_level_index - 1)] = deepcopy(
                debug.get('updated_box_features_' + str(box_level_index)))

        # Saving updated box data and features for current iteration

        debug['updated_box_data_' + str(box_level_index)] = deepcopy(current_box_data)
        debug['updated_box_features_' + str(box_level_index)] = deepcopy(current_box_features)

        # Number of boxes left

        count_boxes = current_box_features.shape[0]

        logger.info('Number of boxes | {}'.format(count_boxes))

        if count_boxes <= ev_config['detection']['minimum_boxes_count']:
            box_level_index = 1
            break

        # Check box features

        checks_fail, debug = boxes_sanity_checks(current_box_features, debug, min_energy, ev_config, logger_pass)

        # Update the params

        box_index += 1
        box_level_index = 2
        current_box_amplitude += step_amplitude

    # Remove noisy boxes

    min_energy_per_charge = ev_config.get('minimum_energy_per_charge', {}).get(region)

    # storing index of the last stable boxes
    final_box_index = min(box_level_index, max(1, box_index - 1))
    debug['final_box_index'] = deepcopy(final_box_index)

    box_data = debug.get('updated_box_data_' + str(final_box_index))
    min_energy = debug.get('amplitude_' + str(final_box_index)) / factor

    box_data = remove_timed_block_boxes(box_data, debug, ev_config, logger_base)
    box_data = filter_ev_boxes(box_data, input_data_original, factor, ev_config, min_energy, logger_pass)
    box_features = boxes_features(box_data, factor, ev_config)

    debug['updated_box_data_' + str(final_box_index)] = deepcopy(box_data)
    debug['updated_box_features_' + str(final_box_index)] = deepcopy(box_features)

    debug = remove_noise_detection_boxes(debug, min_energy_per_charge, final_box_index, ev_config)

    return debug


def remove_timed_block_boxes(box_data, debug, ev_config, logger_base):
    """
    Function to remove erroneous boxes

    Parameters:
        box_data                  (np.ndarray)          : Array containing all the box and their features
        debug                     (dict)                : Dict containing hsm_in, bypass_hsm, make_hsm
        ev_config                 (dict)                : EV config
        logger_base               (logger)              : Logging object to log important steps and values in the run
    Returns:
        box_data                  (np.ndarray)          : Array containing all the box and their features
    """

    factor = int(debug.get('factor'))
    sampling_rate = ev_config.get('sampling_rate')

    cons_arr = deepcopy(box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    cons_bool = (cons_arr > 0).astype(int)

    day_ts, row_idx = np.unique(box_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)

    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)
    col_idx = np.floor((box_data[:, Cgbdisagg.INPUT_EPOCH_IDX] - box_data[:, Cgbdisagg.INPUT_DAY_IDX]) / ev_config.get('sampling_rate'))
    col_idx = col_idx / Cgbdisagg.SEC_IN_HOUR
    col_idx = (pd_mult * (col_idx - col_idx.astype(int) + box_data[:, Cgbdisagg.INPUT_HOD_IDX])).astype(int)

    # Initialize and populate the day wise data array

    num_days = len(day_ts)
    num_pd = int(Cgbdisagg.HRS_IN_DAY * factor)

    # Convert 1D consumption to 2D consumption matrix

    day_cons_bool = np.full(shape=(num_days, num_pd), fill_value=np.nan)
    day_cons_bool[row_idx, col_idx] = cons_bool

    # Nans to Zero

    nan_idx = np.isnan(day_cons_bool)
    day_cons_bool[nan_idx] = 0

    # Get the start and end index of EV boxes

    padding_arr = np.full((1, day_cons_bool.shape[1]), fill_value=0)
    padded_day_cons = np.r_[padding_arr, deepcopy(day_cons_bool), padding_arr]
    diff_arr = np.diff(padded_day_cons, axis=0)

    start_idx = np.argwhere(diff_arr == 1)
    end_idx = np.argwhere(diff_arr == -1)

    # Sort the start and end index

    start_idx = start_idx[np.lexsort((start_idx[:, 0], start_idx[:, 1]))]
    end_idx = end_idx[np.lexsort((end_idx[:, 0], end_idx[:, 1]))]

    # Get the duration of each box and subset the box with >= 7 days of continuous consumption

    combined_arr = np.c_[start_idx[:, 1], start_idx[:, 0], end_idx[:, 0], end_idx[:, 0] - start_idx[:, 0]]
    combined_arr = combined_arr[combined_arr[:, 3] >= 7]

    # Remove the timed boxes

    for i, row in enumerate(combined_arr):
        day_cons_bool[row[1]: row[2], row[0]] = 0

    # Convert from 2D to 1D

    day_cons_bool = day_cons_bool[row_idx, col_idx].flatten()
    nan_idx_flattened = nan_idx[row_idx, col_idx].flatten()
    day_cons_bool = day_cons_bool[np.logical_not(nan_idx_flattened)]

    day_cons_flatten = day_cons_bool

    cons_arr = np.multiply(cons_arr, day_cons_flatten)

    box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = cons_arr
    box_data = get_ev_boxes(box_data, ev_config['minimum_duration'][sampling_rate], 0, factor, logger_base)

    return box_data


def remove_noise_detection_boxes(debug, min_energy, index, ev_config):

    """
    Function to remove erroneous boxes

    Parameters:
        debug                     (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
        min_energy                (float)              : Minimum allowed EV amplitude
        index                     (logger)            : Index of the fbox data
        ev_config                  (dict)              : EV config

    Returns:
        debug                     (object)            : Object containing all important data/values as well as HSM
    """

    box_data = deepcopy(debug.get('updated_box_data_' + str(index)))

    box_features = deepcopy(debug.get('updated_box_features_' + str(index)))

    columns_dict = ev_config.get('box_features_dict')

    auc_column = columns_dict.get('boxes_areas_column')
    end_index_col = columns_dict.get('end_index_column')

    # Removing boxes with lower than allowed energy per charge

    invalid_energy_boxes = box_features[box_features[:, auc_column] < min_energy]

    for i, row in enumerate(invalid_energy_boxes):
        start_idx, end_idx = row[:end_index_col + 1].astype(int)

        box_data[start_idx:(end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    box_features = box_features[box_features[:, auc_column] >= min_energy]

    debug['updated_box_data_' + str(index)] = deepcopy(box_data)
    debug['updated_box_features_' + str(index)] = deepcopy(box_features)

    return debug
