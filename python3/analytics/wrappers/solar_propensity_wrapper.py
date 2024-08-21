"""
Author - Paras Tehria
Date - 1-Dec-2020
Call the Solar propensity module and update in user profile
"""

# Import python packages

import copy
import time
import logging

# Import functions from within the project

from python3.analytics.solar_propensity.initialize_solar_propensity_params import init_solar_propensity_config

from python3.analytics.solar_propensity.functions.compute_solar_propensity import compute_solar_propensity

from python3.analytics.solar_propensity.functions.get_solar_propensity_user_profile import get_solar_propensity_user_profile


def solar_propensity_wrapper(analytics_input_object, analytics_output_object):
    """
    This is the wrapper used for running solar propensity module
    Parameters:
        analytics_input_object (dict)              : Dictionary containing all inputs
        analytics_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the Solar disaggregation module module
    logger_solar_base = analytics_input_object.get("logger").getChild("solar_propensity_wrapper")
    logger_solar_pass = {"logger_base": logger_solar_base,
                         "logging_dict": analytics_input_object.get("logging_dict")}
    logger_solar = logging.LoggerAdapter(logger_solar_base, analytics_input_object.get("logging_dict"))

    # Starting the algorithm time counter
    time_solar_start = time.time()

    # List to store the error points throughout the algo run
    error_list = []

    # Start detection hsm
    input_data = copy.deepcopy(analytics_input_object.get('input_data'))

    detection_hsm = analytics_output_object.get("special_outputs", {}).get("solar", {})

    # Reading global configuration from disagg_input_object
    global_config = analytics_input_object.get("config")

    # Reading solar flag from detection hsm

    if detection_hsm is not None:
        # Reading solar flag from detection hsm
        solar_detection_flag = detection_hsm.get('attributes', {}).get('solar_present')
    else:
        solar_detection_flag = None

    propensity_models = analytics_input_object.get('loaded_files', {}).get('solar_files', {}).get('propensity_model')
    solar_propensity_config = init_solar_propensity_config(global_config=global_config,
                                                           analytics_input_object=analytics_input_object)

    disagg_mode = global_config.get('disagg_mode')

    debug = {'input_data': input_data}

    # Calculate solar propensity
    if propensity_models is None or len(propensity_models) == 0:
        logger_solar.warning("Solar propensity models not found, skipping propensity module | ")
        error_list.append("propensity_models_not_present")

    elif disagg_mode == 'mtd':
        logger_solar.warning("Solar propensity does not run for mtd mode | ")

    elif solar_detection_flag == 1:
        logger_solar.info("Solar panel detected for the user, skipping propensity module | ")

    else:

        debug = compute_solar_propensity(solar_propensity_config, debug,
                                         analytics_input_object, logger_solar_pass)

    logger_solar.info("Time taken for solar propensity | {}".format(time.time() - time_solar_start))

    if not (disagg_mode == 'mtd'):

        analytics_output_object = \
            get_solar_propensity_user_profile(analytics_input_object, analytics_output_object, logger_solar_pass, debug)

    return analytics_input_object, analytics_output_object
