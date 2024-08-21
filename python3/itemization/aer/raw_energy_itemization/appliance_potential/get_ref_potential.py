
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update ref confidence and potential values

"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff


def get_ref_potential(app_index, item_input_object, item_output_object, sampling_rate, logger_pass):

    """
    Calculate ref confidence and potential values

    Parameters:
        app_index                   (int)           : Index of app in the appliance list
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        sampling_rate               (int)           : sampling rate
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_ref_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    ref_disagg = item_output_object.get("updated_output_data")[app_index, :, :]

    ref_potential = np.ones(ref_disagg.shape)

    ref_potential = ref_potential / np.max(ref_potential)

    ref_confidence = 1 - np.divide(np.abs(ref_disagg - ref_potential*np.mean(ref_disagg)), ref_disagg)

    # Sanity checks

    ref_confidence = (ref_confidence) + 0.3

    ref_potential = np.fmax(0, ref_potential)
    ref_potential = np.fmin(1, ref_potential)

    ref_confidence = np.fmax(0, ref_confidence)
    ref_confidence = np.fmin(1, ref_confidence)

    # Dumping appliance confidence and potential values

    item_output_object["app_confidence"][app_index, :, :] = ref_confidence
    item_output_object["app_potential"][app_index, :, :] = ref_potential

    t_end = datetime.now()

    logger.debug("Ref potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object
