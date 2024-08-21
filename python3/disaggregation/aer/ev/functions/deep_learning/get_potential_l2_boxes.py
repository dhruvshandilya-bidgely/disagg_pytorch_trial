"""
Author - Sahana M
Date - 14-Nov-2023
Module to get the potential EV boxes
"""

# import python packages
import logging
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.disaggregation.aer.ev.functions.deep_learning.box_fitting_l2 import box_fitting
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import min_duration_check
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import min_amplitude_check
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import get_curr_partitions


def get_potential_l2_boxes(dl_debug, logger_base):
    """
    Function to get the potential EV boxes
    Parameters:
        dl_debug                    (Dict)                  : Debug dictionary
        logger_base                (logger)                : Logger passed
    Returns:
        box_data                    (np.ndarray)            : Box data array
        debug                       (Dict)                  : Debug dictionary
    """

    # Initialise the logger
    logger_local = logger_base.get('logger').getChild('get_potential_l2_boxes')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))
    logger_pass = {'logger': logger_local, 'logging_dict': logger_base.get('logging_dict')}

    # Extract the required variables

    raw_data = dl_debug.get('raw_data')
    config = deepcopy(dl_debug.get('config'))
    factor = deepcopy(dl_debug.get('factor'))
    temp_data = deepcopy(dl_debug.get('temperature'))
    heat_pot_data = deepcopy(dl_debug.get('heat_pot'))
    cool_pot_data = deepcopy(dl_debug.get('cool_pot'))
    s_label_data = deepcopy(dl_debug.get('s_label_data'))
    total_partitions = deepcopy(dl_debug.get('total_partitions'))

    min_amplitude = config.get('min_amplitude')
    if dl_debug.get('region') == 'EU':
        min_amplitude = config.get('min_amplitude_eu')

    min_duration = config.get('min_duration_l2')

    min_amplitude = min_amplitude/factor
    prediction_confidences = dl_debug.get('prediction_confidences')

    # Initialising the arrays

    boxes_data = []
    final_heat_pot = []
    final_cool_pot = []
    final_temperature = []
    final_s_label_data = []
    final_hvac_removed_data = []

    for i in range(total_partitions):

        # get the partition data, heating and cooling potential

        curr_partition_original, heat_potential, cool_potential, temperature, curr_s_label = \
            get_curr_partitions(i, raw_data, heat_pot_data, cool_pot_data, temp_data, s_label_data, dl_debug)

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
            final_heat_pot = heat_potential
            final_cool_pot = cool_potential
            final_temperature = temperature
            final_hvac_removed_data = curr_partition_all_partitions
            final_s_label_data = curr_s_label
        else:
            boxes_data = np.r_[boxes_data, curr_partition]
            final_hvac_removed_data = np.r_[final_hvac_removed_data, curr_partition_all_partitions]
            final_heat_pot = np.r_[final_heat_pot, heat_potential]
            final_cool_pot = np.r_[final_cool_pot, cool_potential]
            final_temperature = np.r_[final_temperature, temperature]
            final_s_label_data = np.r_[final_s_label_data, curr_s_label]

    all_weather_data = dict()
    all_weather_data[0] = final_heat_pot
    all_weather_data[1] = final_cool_pot
    all_weather_data[2] = final_temperature

    dl_debug['rows'] = boxes_data.shape[0]
    dl_debug['cols'] = boxes_data.shape[1]
    dl_debug['boxes_data'] = deepcopy(boxes_data)
    dl_debug['s_label_data'] = deepcopy(final_s_label_data)
    dl_debug['all_weather_data'] = deepcopy(all_weather_data)
    dl_debug['hvac_removed_raw_data'] = deepcopy(final_hvac_removed_data)

    # use box fitting to get the boxes
    if np.sum(dl_debug.get('predictions_bool')):

        boxes_data, dl_debug = box_fitting(boxes_data, factor, dl_debug, logger_pass)
        logger.info('DL L2 : Box fitting complete')

    return boxes_data, dl_debug
