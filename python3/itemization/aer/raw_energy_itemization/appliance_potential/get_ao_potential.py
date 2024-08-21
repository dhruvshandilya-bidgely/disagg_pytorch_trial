"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update AO confidence and potential values
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff


def get_ao_potential(app_index, item_input_object, item_output_object, sampling_rate, logger_pass):

    """
    Calculate AO confidence and potential values

    Parameters:
        app_index                   (int)           : Index of app in the appliance list
        item_input_object         (dict)            : Dict containing all hybrid inputs
        item_output_object        (dict)            : Dict containing all hybrid outputs
        sampling_rate               (int)           : sampling rate
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_ao_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetching required inputs

    season = item_output_object.get("season")
    ao_disagg = item_output_object.get("updated_output_data")[app_index, :, :]
    original_input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    ao_confidence = np.zeros(ao_disagg.shape)

    # Calculating potential and confidence values using assumption of having similar AO in a particular season

    for season_val in [-1, -0.5, 0, 0.5, 1]:

        if np.any(season == season_val):
            ao_confidence[season == season_val] = 1 - (ao_disagg[season == season_val] - np.mean(ao_disagg[season == season_val]))\
                                                  / np.mean(ao_disagg[season == season_val])

    ao_potential = ao_disagg / np.max(ao_disagg)

    # Sanity checks

    ao_confidence[original_input_data == 0] = 0
    ao_potential[original_input_data == 0] = 0

    ao_potential = np.fmax(0, ao_potential)
    ao_potential = np.fmin(1, ao_potential)

    ao_confidence = np.fmax(0, ao_confidence)
    ao_confidence = np.fmin(1, ao_confidence)

    # Updating hybrid output object

    item_output_object["app_confidence"][app_index, :, :] = ao_confidence
    item_output_object["app_potential"][app_index, :, :] = ao_potential

    t_end = datetime.now()

    logger.debug("AO potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object
