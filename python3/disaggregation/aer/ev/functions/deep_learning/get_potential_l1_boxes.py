"""
Author - Sahana M
Date - 14-Nov-2023
Module to get the potential EV L1 boxes
"""

# import python packages
import logging
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_l1 import box_fitting
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_utils import refine_l1_boxes
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import min_duration_check
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import min_amplitude_check
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_curr_partitions_l1


def get_potential_l1_boxes(raw_data, dl_debug, logger_base):
    """
    Function to get the potential EV boxes
    Parameters:
        raw_data                    (np.ndarray)        : Raw data
        dl_debug                    (Dict)              : Debug dictionary
        logger_base                (logger)                : Logger passed

    Returns:
        boxes_data                  (np.ndarray)        : Box data
        debug                       (Dict)              : Debug dictionary
    """

    # Initialise the logger
    logger_local = logger_base.get('logger').getChild('get_potential_l1_boxes')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))
    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}

    # Extract the required variables
    config = dl_debug.get('config')
    factor = dl_debug.get('factor')
    total_partitions = dl_debug.get('total_partitions')
    min_amplitude = config.get('min_amplitude_l1')
    min_duration = config.get('min_duration_l1')

    min_amplitude = min_amplitude/factor
    prediction_confidences = dl_debug.get('prediction_confidences_l1')

    boxes_data = []
    final_hvac_removed_data = []

    for i in range(total_partitions):

        # get the partition data, heating and cooling potential

        curr_partition_original = get_curr_partitions_l1(i, raw_data, dl_debug)

        # remove the obvious values

        curr_partition = deepcopy(curr_partition_original)
        curr_partition_all_partitions = deepcopy(curr_partition_original)

        # remove any boxes that are running for 15 min or half an hour
        curr_partition = min_amplitude_check(curr_partition, min_amplitude, i, prediction_confidences, config)
        curr_partition = min_duration_check(curr_partition, factor, min_duration)

        # remove any boxes that are running for 15 min or half an hour
        curr_partition_all_partitions = min_amplitude_check(curr_partition_all_partitions, min_amplitude, i, [], config,
                                                            False)
        curr_partition_all_partitions = min_duration_check(curr_partition_all_partitions, factor, min_duration)

        if i == 0:
            boxes_data = curr_partition
            final_hvac_removed_data = curr_partition_all_partitions
        else:
            boxes_data = np.r_[boxes_data, curr_partition]
            final_hvac_removed_data = np.r_[final_hvac_removed_data, curr_partition_all_partitions]

    dl_debug['rows'] = boxes_data.shape[0]
    dl_debug['cols'] = boxes_data.shape[1]
    dl_debug['boxes_data_l1'] = deepcopy(boxes_data)
    dl_debug['boxes_data_preserved'] = deepcopy(final_hvac_removed_data)

    # use box fitting to get the boxes
    if np.sum(dl_debug.get('predictions_bool_l1')):

        boxes_data, dl_debug = box_fitting(boxes_data, factor, dl_debug, logger_pass)
        logger.info('DL L1 : Box fitting complete | ')

        # Remove possible False Positive boxes

        boxes_data = refine_l1_boxes(boxes_data, dl_debug)
        logger.info('DL L1 : Potential False Positive boxes removal complete | ')

    return boxes_data, dl_debug
