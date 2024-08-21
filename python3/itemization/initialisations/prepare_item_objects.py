"""
Author - Sahana M
Date - 20-Sep-2021
This function is used to prepare itemisation input and output objects
"""
# Import python packages

import logging
import traceback
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff
from python3.itemization.initialisations.prepare_data import prepare_data


def prepare_item_object(item_input_object, item_output_object, logger_pass):

    """
    Prepare hybrid input object

    Parameters:
        item_input_object             (dict)      : Dict containing all hybrid inputs
        item_output_object            (dict)      : Dict containing all hybrid outputs
        logger_pass                   (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object             (dict)      : Dict containing all hybrid inputs
        item_output_object            (dict)      : Dict containing all hybrid outputs
        faulty_input_data             (int)       : 1 if the input data is invalid, and pipeline will not run further
    """

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('prepare_item_object')
    logger = logging.LoggerAdapter(logger_base, item_input_object.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    faulty_input_data = 0
    error_code = 1

    # Not running pipeline for faulty input data

    input_data = item_input_object.get('input_data')[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    sampling_rate = item_input_object.get("config").get("sampling_rate")
    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    if np.all(input_data == 0):
        logger.warning("Not running itemization pipeline since all values are zero")
        faulty_input_data = 1

    elif np.all(input_data < 0):
        logger.warning("Not running itemization pipeline since all values are negative")
        faulty_input_data = 1

    elif np.all(np.isnan(input_data)):
        logger.warning("Not running itemization pipeline since all values are NAN")
        faulty_input_data = 1

    elif len(input_data) <= samples_per_hour*24:
        logger.warning("Not running itemization pipeline since only 1 day of data is available")
        faulty_input_data = 1

    if faulty_input_data:
        item_input_object['item_input_params'] = dict()
        item_input_object['item_input_params']['run_hybrid_v2_flag'] = 0
        return item_input_object, item_output_object, faulty_input_data, -2

    # prepare itemization input data

    t_start = datetime.now()

    try:

        item_input_object, item_output_object, error_code = prepare_data(item_input_object, item_output_object, logger_pass)

        if item_input_object["pilot_level_config_present"] <= 0:
            logger.warning("Not running itemization pipeline since pilot model file is absent")

            t_end = datetime.now()
            item_output_object['itemization_metrics']['itemization_pipeline'] = {
                'time': get_time_diff(t_start, t_end),
                'exit_status': {
                    'exit_code': error_code,
                    'error_list': [''],
                    'itemization_pipeline_status': True
                }
            }

    # General exception

    except Exception:
        error_code = -1
        t_end = datetime.now()
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in data preparation, not running itemization pipeline | %s', error_str)
        faulty_input_data = 1

        item_input_object['item_input_params'] = dict()
        item_input_object['item_input_params']['run_hybrid_v2_flag'] = False

        item_output_object['itemization_metrics']['itemization_pipeline'] = {
            'time': get_time_diff(t_start, t_end),
            'exit_status': {
                'exit_code': -1,
                'error_list': [error_str],
                'itemization_pipeline_status': False
            }
        }

    return item_input_object, item_output_object, faulty_input_data, error_code
