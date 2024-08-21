"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Module to detect energy amplitude (mean and std dev) for a Timed Water Heater and also thin_peak_energy values
that are specific to high and low temperature days
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.maths_utils import rotating_sum

from python3.disaggregation.aer.waterheater.functions.timed.functions.get_winter_data import get_winter_data
from python3.disaggregation.aer.waterheater.functions.timed.functions.timed_confidence import timed_confidence


def winter_consistency_check(input_data, input_box_data, features, wh_config, debug, logger_base):
    """
    Check if consistency of box present in winter season

    Parameters:
        input_data          (np.ndarray)    : Input 21-column matrix
        input_box_data      (np.ndarray)    : Box data
        features            (np.ndarray)    : Information of hourly fractions
        wh_config           (dict)          : Configuration of the algorithm
        debug               (dict)          : Output for each step of the algorithm
        logger_base         (logger)        : The logger object

    Returns:
        features            (np.ndarray)    : Information of hourly fractions
        debug               (dict)          : Output for each step of the algorithm
        wtr_idx             (np.ndarray)    : Index of winter season data
        wtr_data            (np.ndarray)    : Winter season data
        winter_box_data     (np.ndarray)    : Winter box data
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('winter_consistency_check')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the winter data and season info

    wtr_idx, wtr_data, winter_box_data = get_winter_data(input_data, input_box_data, logger)

    # Check if valid winter data present

    if wtr_data.shape[0] == 0:
        # If no winter data found, return with default values

        debug['timed_hld_wtr'] = 0

        return [], debug, wtr_idx, wtr_data, winter_box_data

    # Taking deepcopy of input data to keep local instances

    raw_data = deepcopy(wtr_data)
    box_data = deepcopy(winter_box_data)

    # Loading all the relevant params for the algorithm from config

    factor = debug['time_factor']

    std_thres = wh_config['std_thres']
    detection_threshold = wh_config['wtr_detection_threshold']
    raw_roll_threshold_wtr = wh_config['raw_roll_threshold_wtr']

    max_count_bar = wh_config['max_count_bar']
    minimum_fraction_idx = wh_config['minimum_fraction_idx']
    raw_count_threshold = wh_config['raw_count_threshold_wtr']

    start_mean_threshold_major = wh_config['start_mean_threshold_major']
    start_mean_threshold_minor = wh_config['start_mean_threshold_minor']

    # Allowed maximum fraction at any hour of day is 1

    max_concentration = 1

    # Finding the number of days in the data with boxes

    num_box_days = len(np.unique(box_data[:, Cgbdisagg.INPUT_DAY_IDX]))

    # Finding the highest time division (referred as hour of day in future usage)

    max_hod = int(factor * Cgbdisagg.HRS_IN_DAY) - 1
    debug['max_hod'] = max_hod

    # Extraction energy data of boxes

    box_energy = deepcopy(box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    # Taking only box boundaries for edge related calculations

    box_energy_idx = (box_energy > 0).astype(int)
    box_idx_diff = np.diff(np.r_[0, box_energy_idx, 0])

    # Find the start and end edges of the boxes

    box_start_idx = (box_idx_diff[:-1] > 0)
    box_end_idx = (box_idx_diff[1:] < 0)

    # Checking for the hourly concentration throughout the data

    # Defining time division brackets for edges aggregation

    edges = np.arange(0, max_hod + 2) - 0.5

    # Finding the fraction of days with start edges

    start_hod_count, _ = np.histogram(box_data[box_start_idx, Cgbdisagg.INPUT_HOD_IDX], bins=edges)
    start_hod_count = start_hod_count / num_box_days

    # Save start edge fraction to timed debug

    debug['pos_count_raw'] = start_hod_count
    debug['pos_count_raw_max'] = np.max(np.fmin(start_hod_count, max_concentration))

    # Finding the rolling fraction of days with start edges

    start_hod_count = rotating_sum(start_hod_count, factor)
    start_hod_count = np.fmin(start_hod_count, max_concentration)

    # Finding the fraction of days with end edges
    end_hod_count, _ = np.histogram(box_data[box_end_idx, Cgbdisagg.INPUT_HOD_IDX], bins=edges)
    end_hod_count = end_hod_count / num_box_days

    # Save end edge fraction to timed debug

    debug['neg_count_raw'] = end_hod_count
    debug['neg_count_raw_max'] = np.max(np.fmin(end_hod_count, max_concentration))

    # Finding the rolling fraction of days with end edges

    end_hod_count = rotating_sum(end_hod_count, factor)
    end_hod_count = np.fmin(end_hod_count, max_concentration)

    # Find the maximum rolling fraction among start and end edges

    max_count_roll = np.r_[np.max(start_hod_count), np.max(end_hod_count)]

    logger.info('Winter max count of Start and End | {}, {}'.format(max_count_roll[0], max_count_roll[1]))

    # Stack the start and end fraction values to features

    features = np.vstack((features, start_hod_count))
    features = np.vstack((features, end_hod_count))

    # Finding fraction of valid box energy values at each hour

    hod_count, _ = np.histogram(box_data[box_energy_idx, Cgbdisagg.INPUT_HOD_IDX], bins=edges)
    hod_count = hod_count / num_box_days
    hod_count = np.fmin(hod_count, max_concentration)

    # Append the energy fraction values to the features

    features = np.vstack((features, hod_count))

    # Check if the water heater is common start / end time based

    if np.max(start_hod_count) >= (wh_config['start_end_ratio'] * np.max(end_hod_count)):
        # If start fraction more than end fraction

        wh_type = 'start'

        # Find the standard deviation and fourth highest values of start fractions

        fraction_std = np.std(start_hod_count)
        fourth_highest = np.sort(start_hod_count)[-minimum_fraction_idx]

        # Find the proportion of hours above a certain threshold

        count_above_thres = len(np.where(start_hod_count > detection_threshold)[0]) / factor
    else:
        # If start fraction less than end fraction

        wh_type = 'end'

        # Find the standard deviation and fourth highest values of start fractions

        fraction_std = np.std(end_hod_count)
        fourth_highest = np.sort(end_hod_count)[-minimum_fraction_idx]

        # Find the proportion of hours above a certain threshold

        count_above_thres = len(np.where(end_hod_count > detection_threshold)[0]) / factor

    # Find the difference between max fraction with fourth highest fraction

    max_fourth_diff = (np.max(max_count_roll) - fourth_highest) / np.max(max_count_roll)

    logger.info('Winter std deviation of fraction | {}'.format(fraction_std))

    debug['max_count_raw'] = np.r_[debug['pos_count_raw_max'], debug['neg_count_raw_max']]
    debug['max_count_roll'] = max_count_roll

    # Calculate raw / roll fraction

    raw_roll = np.max(debug['max_count_raw']) / np.max(debug['max_count_roll'])

    logger.info('Winter raw by roll fraction | {}'.format(raw_roll))

    # Save all the relevant variables to debug

    debug['wh_type'] = wh_type
    debug['raw_roll'] = raw_roll
    debug['timed_std'] = fraction_std

    debug['fourth_high'] = fourth_highest
    debug['max_fourth_diff'] = max_fourth_diff
    debug['count_above_thres'] = count_above_thres

    # Finding the average daily number of runs

    # Get unique days timestamp of box data and save to debug

    unq_days, days_idx = np.unique(box_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)

    debug['box_days_ts'] = unq_days
    debug['box_days_idx'] = days_idx

    # Get daily runs count for start as well as edge (ideally same)

    daily_start_count = np.bincount(days_idx, box_start_idx)
    daily_end_count = np.bincount(days_idx, box_end_idx)

    # Add runs count to debug

    debug['timed_counts'] = {
        'start': {'mean': np.mean(daily_start_count)},
        'end': {'mean': np.mean(daily_end_count)}
    }

    logger.info('Average count of boxes per day | {}'.format(debug['timed_counts'][wh_type]['mean']))

    # Checking amplitude consistency using edge energy values

    box_energy_diff = np.diff(np.r_[0, raw_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]])

    # Find edge energy values based on common start / end type

    if wh_type == 'start':
        edge_energy = box_energy_diff[box_start_idx]
    else:
        edge_energy = box_energy_diff[box_end_idx]

    # Calculate the standard deviation of the edge energy values

    debug['timed_edge_std'] = np.std(edge_energy)

    # #--------------------------------------- Detection Phase-1 ------------------------------------------------------#

    # Detection using the hourly consistency and daily count features

    # Filtering based on max roll fraction, fraction deviation and raw / roll fraction

    if (np.max(max_count_roll) >= detection_threshold) and (fraction_std >= std_thres) and \
            (raw_roll >= raw_roll_threshold_wtr):
        detection = 1
    else:
        detection = 0

    # #--------------------------------------- Detection Phase-2 ------------------------------------------------------#

    # Filtering based on daily runs count

    # If very high daily runs count (possibly noise)

    if debug['timed_counts'][wh_type]['mean'] >= start_mean_threshold_major:
        detection = 0

    # If high daily runs count with less max_count_roll (possibly noise)

    if (debug['timed_counts'][wh_type]['mean'] >= start_mean_threshold_minor) and \
            (np.max(max_count_roll) < max_count_bar):
        detection = 0

    # If product of raw / roll fraction and max roll fraction below a certain number (hazy edge)

    if (raw_roll * np.max(max_count_roll)) < raw_count_threshold:
        detection = 0

    # Save the initial timed water heater detection to the debug

    debug['timed_hld'] = detection
    debug['timed_hld_wtr'] = detection

    # Saving the important variable for next steps in timed water heater (if detected)

    if detection == 1:
        debug['variables'] = {
            'hod_count': hod_count,
            'start_count': start_hod_count,
            'end_count': end_hod_count,
            'box_start': box_start_idx,
            'box_end': box_end_idx,
            'num_box_days': num_box_days,
        }

    # Subset relevant edge indices to be used for confidence score calculation

    edge_idx = box_start_idx if wh_type == 'start' else box_end_idx

    # Calculate the confidence score for timed water heater

    debug = timed_confidence(debug, box_data, wh_config, edge_idx, wh_type)

    return features[1:, :], debug, wtr_idx, wtr_data, winter_box_data
