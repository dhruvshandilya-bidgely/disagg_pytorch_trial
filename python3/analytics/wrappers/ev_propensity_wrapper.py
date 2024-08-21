"""
Author - Paras Tehria
Date - 27 May 2021
EV propensity module wrapper
"""

# import python packages

import time
import logging

# import functions from within the project

from python3.analytics.ev_propensity.init_ev_propensity_config import init_ev_propensity_config
from python3.analytics.ev_propensity.functions.compute_ev_propensity import compute_ev_propensity
from python3.analytics.ev_propensity.functions.get_ev_app_profile import get_ev_app_profile
from python3.analytics.ev_propensity.create_debug_object import create_debug_object
from python3.analytics.ev_propensity.populate_ev_propensity_user_profile import populate_ev_propensity_user_profile


def ev_propensity_wrapper(analytics_input_object, analytics_output_object):

    """
    EV propensity module wrapper

    Parameters:
        analytics_input_object (dict)              : Dictionary containing all inputs
        analytics_output_object(dict)              : Dictionary containing all outputs

    Returns:
        analytics_output_object(dict)              : Dictionary containing all outputs
    """

    # initiate logger for EV propensity module

    logger_ev_prop_base = analytics_input_object.get('logger').getChild('ev_propensity_wrapper')
    logger_ev_prop = logging.LoggerAdapter(logger_ev_prop_base, analytics_input_object.get('logging_dict'))
    logger_ev_prop_pass = {
        'logger_base': logger_ev_prop_base,
        'logging_dict': analytics_input_object.get('logging_dict'),
    }

    t_start = time.time()

    # List to store the error points throughout the algo run
    error_list = []

    # Start detection hsm
    ev_hsm = analytics_output_object.get("created_hsm", {}).get("ev", {})

    # Reading global configuration from disagg_input_object
    global_config = analytics_input_object.get("config")

    # Reading ev flag from detection hsm
    ev_detection_flag = ev_hsm.get('attributes', {}).get('ev_hld') if isinstance(ev_hsm, dict) else None

    ev_propensity_config = init_ev_propensity_config(global_config, analytics_input_object)

    # Reading appliance profile of the user for EV present

    ev_present, ev_app_profile_yes = get_ev_app_profile(analytics_input_object, logger_ev_prop)

    disagg_mode = global_config.get('disagg_mode')

    # Create debug object for EV propensity
    debug = create_debug_object(analytics_input_object, analytics_output_object)

    propensity_model = debug.get("propensity_model")

    # Calculate ev propensity
    if propensity_model is None or len(propensity_model) == 0:
        logger_ev_prop.warning("EV propensity models not found, skipping propensity module | ")
        error_list.append("propensity_models_not_present")

    elif disagg_mode == 'mtd':
        logger_ev_prop.warning("EV propensity does not run for mtd mode | ")

    elif ev_app_profile_yes:
        logger_ev_prop.warning("User has said yes to EV in app profile, not running EV propensity |")

    elif ev_detection_flag == 1:
        logger_ev_prop.info("EV detected for the user, skipping propensity module | ")

    else:
        debug = compute_ev_propensity(ev_propensity_config, debug, logger_ev_prop_pass)

    logger_ev_prop.info("Time taken to run EV propensity model | {}".format(time.time() - t_start))

    t_before_profile_update = time.time()
    if not (disagg_mode == 'mtd'):
        analytics_output_object = populate_ev_propensity_user_profile(analytics_input_object, analytics_output_object,
                                                                      logger_ev_prop_pass, debug)

    logger_ev_prop.info("Time taken to fill EV propensity profile | {}".format(time.time() - t_before_profile_update))

    return analytics_output_object
