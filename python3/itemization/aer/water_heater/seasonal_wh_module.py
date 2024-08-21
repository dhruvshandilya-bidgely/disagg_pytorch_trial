"""
Author - Sahana M
Date - 2/3/2021
Run different components of seasonal waterheater disaggregation
"""

# Import python packages

import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff
from python3.itemization.aer.water_heater.functions.dump_results import dump_results
from python3.itemization.aer.water_heater.functions.swh_hsm_utils import make_hsm_from_debug
from python3.itemization.aer.water_heater.functions.preprocess_data import preprocess_data
from python3.itemization.aer.water_heater.functions.get_wh_potential import get_wh_potential
from python3.itemization.aer.water_heater.functions.detect_time_band import detect_time_band
from python3.itemization.aer.water_heater.init_seasonal_wh_config import init_seasonal_wh_config
from python3.itemization.aer.water_heater.functions.get_weather_features import get_weather_features


def seasonal_wh_module(in_data, debug, global_config, logger_pass):
    """
    Run the vacation disaggregation module
    Parameters:
        in_data             (np.ndarray)        : 21 column input data
        debug               (dict)              : Contains all variables required for debugging from the main disagg
        global_config       (dict)              : Dictionary containing all needed configuration variables
        logger_pass         (dict)              : Contains the logger and the logging dictionary to be passed on
    Returns:
        debug               (dict)              : Contains all variables required for debugging from the main disagg
        hsm_in              (dict)              : Contains all the attributes required for hsm
    """

    t_swh_start = datetime.now()

    # ------------------------------------------ STAGE 1: INITIALISATIONS ---------------------------------------------

    t_init_start = datetime.now()

    # Initialize seasonal wh logger

    logger_base = logger_pass.get('logger').getChild('seasonal_wh_disagg')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Prepare logger pass to pass to sub-modules

    logger_pass['base_logger'] = logger_base

    # Initialise config dictionary with required parameters

    seasonal_wh_config = init_seasonal_wh_config(global_config.get('pilot_id'), global_config.get('sampling_rate'),
                                                 global_config.get('uuid'))

    # Initialise input data

    input_data = deepcopy(in_data)
    debug['input_data'] = deepcopy(input_data)

    t_init_end = datetime.now()

    logger.info('Itemization WH Initialization took | %.3f s ', get_time_diff(t_init_start, t_init_end))

    # ------------------------------------------ STAGE 2: GET WEATHER DATA ANALYTICS ----------------------------------

    t_weather_data_start = datetime.now()

    weather_data_output, exit_swh = get_weather_features(input_data, seasonal_wh_config, logger_pass)

    t_weather_data_end = datetime.now()

    logger.info('Obtained weather features took | %.3f s ', get_time_diff(t_weather_data_start, t_weather_data_end))

    debug['weather_data_output'] = weather_data_output

    # ------------------------------------------ STAGE 3: PREPROCESS INPUT DATA ---------------------------------------

    # If Weather data output obtained then run the detection module

    if not exit_swh:
        t_preprocess_start = datetime.now()

        # Preprocess input data

        input_data, debug = preprocess_data(input_data, seasonal_wh_config, debug, logger_pass)

        debug['swh_cleaned_data'] = deepcopy(input_data)

        t_preprocess_end = datetime.now()

        logger.info('Preprocessing Itemization WH Input data took | %.3f s ', get_time_diff(t_preprocess_start, t_preprocess_end))

        # ------------------------------------------ STAGE 4: COMPUTE WH POTENTIAL ------------------------------------

        t_potential_start = datetime.now()

        # Get Water heater potential

        wh_potential = get_wh_potential(in_data, logger_pass)

        debug['wh_potential'] = wh_potential

        t_potential_end = datetime.now()

        logger.info('WH Potential calculation took | %.3f s ', get_time_diff(t_potential_start, t_potential_end))

        # ------------------------------------------ STAGE 5: TIME BAND DETECTION & ESTIMATION ------------------------

        t_detection_start = datetime.now()

        # Perform detection and estimation of seasonal water heater

        debug = detect_time_band(input_data, seasonal_wh_config, debug, logger_pass)

        t_detection_end = datetime.now()

        logger.info('WH Detection & Estimation took | %.3f s ', get_time_diff(t_detection_start, t_detection_end))

        # Total time taken

        t_swh_end = datetime.now()

        logger.info('SWH module took | %.3f s ', get_time_diff(t_swh_start, t_swh_end))

        # ------------------------------------------ STAGE 6: PLOTTING THE DATA ---------------------------------------

        # Saving results

        dump_results(debug, seasonal_wh_config, global_config)

    else:
        logger.info('Not running Seasonal WH module due to Insufficient Weather Data |')
        debug['swh_hld'] = 0

    # Create HSM

    debug = make_hsm_from_debug(debug, logger)

    return debug, exit_swh
