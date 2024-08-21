"""
Author - Sahana M
Date - 21-Feb-2022
Module for performing post processing on L1 boxes
"""

# Import python packages
import logging
import numpy as np

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.ev.functions.detection.get_boxes_features import boxes_features


def clean_boxes(invalid_boxes, box_data, box_features, columns_dict, debug, ev_config):
    """
    This function is used to remove the labelled invalid boxes from box data and features
    Parameters:
        invalid_boxes           (Boolean)               : Boolean invalid boxes
        box_data                (np.ndarray)            : EV boxes data
        box_features            (np.ndarray)            : EV boxes features
        columns_dict            (dict)                  : Column names and their indexes
        debug                   (dict)                  : Debug dictionary
        ev_config               (dict)                  : EV configurations
    Returns:
        box_data                (np.ndarray)            : EV boxes data
        box_features            (np.ndarray)            : EV boxes features
    """
    for i in range(len(box_features)):
        if invalid_boxes[i]:
            start_idx = box_features[int(i), columns_dict['start_index_column']].astype(int)
            end_idx = (box_features[int(i), columns_dict['end_index_column']] + 1).astype(int)
            box_data[start_idx:end_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    box_features = boxes_features(box_data, debug.get('factor'), ev_config)

    return box_data, box_features


def refine_l1_boxes(box_data, box_features, ev_config, debug, logger_base):
    """
    Function used to remove noise L1 boxes
    Parameters:
        box_data                (np.ndarray)            : EV boxes data
        box_features            (np.ndarray)            : EV boxes features
        ev_config               (dict)                  : EV configurations
        debug                   (dict)                  : Debug dictionary
        logger_base             (logger)                : logger pass
    Returns:
        box_data                (np.ndarray)            : EV boxes data
        box_features            (np.ndarray)            : EV boxes features
    """

    logger_local = logger_base.get('logger').getChild('refine_l1_boxes')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Extract required variables

    columns_dict = debug.get('l1_config').get('features_dict')
    charging_hours = debug.get('l1_config').get('charging_hours')
    invalid_box_thr = debug.get('l1_config').get('invalid_box_thr')
    dur_amp_threshold = debug.get('l1_config').get('dur_amp_threshold')
    minimum_duration_allowed = debug.get('l1_config').get('minimum_duration_allowed')
    maximum_duration_allowed = debug.get('l1_config').get('maximum_duration_allowed')

    invalid_boxes = np.full(shape=(box_features.shape[0]), fill_value=False)

    # Remove boxes with very less duration (less than 3 hours)

    low_duration_boxes = box_features[:, columns_dict['boxes_duration_column']] < minimum_duration_allowed
    invalid_boxes = invalid_boxes | low_duration_boxes

    # Remove boxes with very high duration (greater than 16 hours)

    high_duration_boxes = box_features[:, columns_dict['boxes_duration_column']] > maximum_duration_allowed
    invalid_boxes = invalid_boxes | high_duration_boxes

    logger.info("Number of Invalid L1 boxes removed based on Min and Max duration | {}".format(np.sum(invalid_boxes)))

    # Clean the boxes

    box_data, box_features = clean_boxes(invalid_boxes, box_data, box_features, columns_dict, debug, ev_config)

    # Remove noise boxes occurring in either day/night

    n_boxes = box_features.shape[0]
    night_boolean = ((box_features[:, columns_dict['night_boolean']] >= charging_hours[0]) |
                     (box_features[:, columns_dict['night_boolean']] <= charging_hours[1])).astype(int)

    night_count_fraction = np.sum(night_boolean) / n_boxes
    day_count_fraction = 1 - night_count_fraction

    night_boolean = night_boolean == 1
    day_boolean = ~night_boolean

    if night_count_fraction > day_count_fraction:
        median_night_duration = np.nanmedian(box_features[night_boolean, columns_dict['boxes_duration_column']])
        median_night_amplitude = np.nanmedian(box_features[night_boolean, columns_dict['boxes_median_energy']])
        median_night_dur_amp = median_night_duration*median_night_amplitude

        outlier_duration_boxes = \
            abs(median_night_duration - box_features[:, columns_dict['boxes_duration_column']]) >= \
            dur_amp_threshold*median_night_duration
        outlier_amp_boxes = \
            abs(median_night_amplitude - box_features[:, columns_dict['boxes_median_energy']]) >= \
            dur_amp_threshold*median_night_amplitude
        outlier_dur_amp_boxes = \
            abs(median_night_dur_amp -
                (box_features[:, columns_dict['boxes_duration_column']]*box_features[:, columns_dict['boxes_median_energy']])) \
            >= dur_amp_threshold*median_night_dur_amp

        invalid_boxes = (outlier_duration_boxes | outlier_amp_boxes | outlier_dur_amp_boxes) & day_boolean

    else:
        median_day_duration = np.nanmedian(box_features[day_boolean, columns_dict['boxes_duration_column']])
        median_day_amplitude = np.nanmedian(box_features[day_boolean, columns_dict['boxes_median_energy']])
        median_day_dur_amp = median_day_duration*median_day_amplitude

        outlier_duration_boxes = \
            abs(median_day_duration - box_features[:, columns_dict['boxes_duration_column']]) \
            >= dur_amp_threshold*median_day_duration
        outlier_amp_boxes = \
            abs(median_day_amplitude - box_features[:, columns_dict['boxes_median_energy']]) \
            >= dur_amp_threshold*median_day_amplitude
        outlier_dur_amp_boxes = \
            abs(median_day_dur_amp -
                (box_features[:, columns_dict['boxes_duration_column']]*box_features[:, columns_dict['boxes_median_energy']])) \
            >= dur_amp_threshold*median_day_dur_amp

        invalid_boxes = (outlier_duration_boxes | outlier_amp_boxes | outlier_dur_amp_boxes) & night_boolean

    logger.info("Number of Invalid L1 boxes removed based on Day/Night count fraction | {}".format(np.sum(invalid_boxes)))

    # Clean the boxes

    box_data, box_features = clean_boxes(invalid_boxes, box_data, box_features, columns_dict, debug, ev_config)

    # Remove boxes with outlier amplitudes

    median_amplitude = np.nanmedian(box_features[:, columns_dict['boxes_energy_per_point_column']])
    invalid_boxes = abs(median_amplitude - box_features[:, columns_dict['boxes_energy_per_point_column']]) \
                    >= invalid_box_thr*median_amplitude

    logger.info("Number of Invalid L1 boxes removed based on Outlier Amplitudes | {}".format(np.sum(invalid_boxes)))

    # Clean the boxes

    box_data, box_features = clean_boxes(invalid_boxes, box_data, box_features, columns_dict, debug, ev_config)

    return box_data, box_features
