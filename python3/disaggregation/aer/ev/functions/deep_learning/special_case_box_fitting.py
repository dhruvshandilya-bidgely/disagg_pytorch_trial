"""
Author - Sahana M
Date - 14-Nov-2023
Module for Special case box fitting
"""

# import python packages
import logging
import numpy as np
from copy import deepcopy
from scipy import spatial

# Import packages from within the project
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_wrong_indexes
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_ev_boxes


def multimode_charging_detection(box_data, dl_debug, logger_base):
    """
    Function to detect multimode charging
    Parameters:
        box_data                (np.ndarray)            : Box data
        dl_debug                (Dict)                  : Debug dictionary
        logger_base             (logger)                : Logger passed

    Returns:
        identified_boxes_2d     (np.ndarray)            : Identified boxes 2D
        dl_debug                (Dict)                  : Debug dictionary
    """

    # Initialise the logger
    logger_local = logger_base.get('logger').getChild('multimode_charging_detection')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # extract the required variables

    mutlimode_detection = False
    rows, cols = dl_debug.get('rows'), dl_debug.get('cols')
    low_amp_thr = dl_debug.get('config').get('multimode_box_fitting').get('low_amp_thr')
    high_amp_thr = dl_debug.get('config').get('multimode_box_fitting').get('high_amp_thr')
    overlapping_thr = dl_debug.get('config').get('multimode_box_fitting').get('overlapping_thr')
    box_distribution_thr = dl_debug.get('config').get('multimode_box_fitting').get('box_distribution_thr')
    similarity_score_thr = dl_debug.get('config').get('multimode_box_fitting').get('similarity_score_thr')

    identified_boxes_2d = deepcopy(box_data)
    identified_boxes_2d = identified_boxes_2d.reshape(rows, cols)

    # subtract the identified boxes from the raw data

    original_box_data = deepcopy(dl_debug.get('boxes_data'))
    unidentified_boxes_2d = original_box_data - identified_boxes_2d

    if np.sum(unidentified_boxes_2d) != 0:
        unidentified_boxes_1d = unidentified_boxes_2d.flatten()

        # get the non-zero consumption days
        identified_boxes_2d_non_zero_days = np.sum(identified_boxes_2d, axis=1) > 0
        unidentified_boxes_2d_non_zero_days = np.sum(unidentified_boxes_2d, axis=1) > 0

        if np.sum(identified_boxes_2d_non_zero_days) and np.sum(unidentified_boxes_2d_non_zero_days):
            # get the distribution of the identified boxes

            identified_boxes_distribution = np.sum(identified_boxes_2d[identified_boxes_2d_non_zero_days] > 0, axis=0) /\
                                            np.sum(np.sum(identified_boxes_2d[identified_boxes_2d_non_zero_days] > 0, axis=0))*100

            wrong_indexes = get_wrong_indexes(identified_boxes_distribution)

            identified_boxes_distribution[wrong_indexes] = 0

            # get the amplitude of the identified boxes 2d
            identified_amp_arr = np.percentile(identified_boxes_2d[identified_boxes_2d_non_zero_days], axis=0, q=99)
            unidentified_amp_arr = np.percentile(unidentified_boxes_2d[unidentified_boxes_2d_non_zero_days], axis=0, q=99)
            identified_amp_arr[wrong_indexes] = 0
            unidentified_amp_arr[wrong_indexes] = 0

            # get the minimum amplitude expected for multimode charging
            amplitude_bar_low = low_amp_thr*identified_amp_arr
            amplitude_bar_high = high_amp_thr*identified_amp_arr
            probable_boxes_indexes = (unidentified_amp_arr > amplitude_bar_low) & (unidentified_amp_arr < amplitude_bar_high)
            overlapping_percentage = np.sum(identified_amp_arr[probable_boxes_indexes > 0] > 0) / np.sum(identified_amp_arr > 0)

            # get the distribution of the unidentified boxes
            average_amplitude = np.mean(amplitude_bar_low[amplitude_bar_low > 0])
            unidentified_boxes_2d[unidentified_boxes_2d < average_amplitude] = 0
            unidentified_boxes_distribution = np.sum(unidentified_boxes_2d[unidentified_boxes_2d_non_zero_days] > 0, axis=0) /\
                                              np.sum(np.sum(unidentified_boxes_2d[unidentified_boxes_2d_non_zero_days] > 0, axis=0))*100
            unidentified_boxes_distribution[unidentified_boxes_distribution < box_distribution_thr] = 0

            # cosine similarity
            similarity_score = 1 - spatial.distance.cosine(identified_boxes_distribution, unidentified_boxes_distribution)

            # if the similarity score is greater than 0.7 then the user has multimode charging

            # get the amplitude of the unidentified boxes
            min_energy = np.median(amplitude_bar_low[amplitude_bar_low > 0])

            if similarity_score >= similarity_score_thr and overlapping_percentage >= overlapping_thr:
                new_boxes = get_ev_boxes(unidentified_boxes_1d, 2, min_energy, dl_debug.get('factor'))
                mutlimode_detection = True
            else:
                new_boxes = np.zeros_like(identified_boxes_2d)

            # convert the new boxes to 2d
            new_boxes = new_boxes.reshape(rows, cols)

            identified_boxes_2d = identified_boxes_2d + new_boxes

    dl_debug['mutlimode_detection'] = mutlimode_detection
    logger.info('DL L2 : Multi-mode charger detected | %s ', mutlimode_detection)

    return identified_boxes_2d, dl_debug
