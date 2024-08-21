"""
Author - Paras Tehria
Date - 12/11/19

This module computes solar detection confidence
"""

# Import python packages

import logging
import numpy as np
from copy import deepcopy

# Import functions from within the project
from python3.disaggregation.aer.solar.functions.get_detection_probability import get_detection_probability

from python3.disaggregation.aer.solar.functions.solar_post_processing import solar_post_process


def get_prob_solar_detection(detection_arr_original, disagg_input_object, logger_base, solar_config):
    """
    This function gives the probability of solar detection using cnn model

    Parameters:
        detection_arr_original     (np.ndarray)    :       input array containing instances for detection
        disagg_input_object        (dict)          :       disagg input object
        logger_base                (dict)          :       base logger
        solar_config               (dict)           :       config file

    Return:
        mask_disconn               (list)          :       containing chunks with disconnections
        probability_solar          (list)          :       detection probabilities before disconnections removal
        probability_solar_after_disconn (list)     :       detection probabilities after disconnections removal
    """

    cnn_detection_array = deepcopy(detection_arr_original)

    logger_local = logger_base.get('logger').getChild('get_prob_solar_detection')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Reading solar detection model

    model = disagg_input_object.get('loaded_files', {}).get('solar_files', {}).get('detection_model')

    if model:

        logger.info('Solar model read successfully inside the detection module | ')

        # Getting probability output from cnn model

        output = get_detection_probability(cnn_detection_array, model, solar_config=solar_config)

        # Converting cnn output to float probability

        probability_solar = np.array([np.round(float(x[1]), 2) for x in output]).reshape(-1, )

        prob_string = ', '.join(map(str, probability_solar))
        logger.info("CNN Model probability output | {} ".format(prob_string))
        # postprocessing

        logger.debug("Applying post processing on cnn output")
        mask_disconn, probability_solar, probability_solar_after_disconn = solar_post_process(cnn_detection_array,
                                                                                              probability_solar,
                                                                                              solar_config, logger_base)

    else:
        # Passing zero probability if model not found

        mask_disconn = np.ones(len(cnn_detection_array), dtype=bool)
        probability_solar = np.zeros((len(cnn_detection_array),))
        probability_solar_after_disconn = probability_solar
        logger.warning('Solar model not found | ')

    return mask_disconn, probability_solar, probability_solar_after_disconn
