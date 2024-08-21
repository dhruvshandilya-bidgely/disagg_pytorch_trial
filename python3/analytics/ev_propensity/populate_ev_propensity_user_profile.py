"""
Author - Paras Tehria
Date - 27-May-2020
Updates the ev user profile with ev propensity score
"""

# Import python packages

import logging
import traceback
from copy import deepcopy

# Import functions from within the project

from python3.config.mappings.get_app_id import get_app_id

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def populate_ev_propensity_user_profile(analytics_input_object, analytics_output_object, logger_base, debug):
    """
    This function populates ev propensity user profile
    Parameters:
        analytics_input_object     (dict)      : Dict containing all the inputs to the pipeline
        analytics_output_object    (dict)      : Dict containing all the outputs to the pipeline
        logger_base                (logger)    : Logger object
    Returns:
        user_profile_object     (dict)      : User profile from all appliances
    """
    # Taking logger base for this function

    logger_ev_base = logger_base.get("logger_base").getChild("populate_ev_propensity_user_profile")
    logger_pass = {"logger": logger_ev_base,
                   "logging_dict": analytics_input_object.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_ev_base, logger_base.get("logging_dict"))

    ev_app_id = get_app_id('ev')

    # Initialize variables needed to populate bc level time of usage

    out_bill_cycles = analytics_input_object.get('out_bill_cycles')

    for idx, row in enumerate(out_bill_cycles):
        bill_cycle_start, bill_cycle_end = row[:2]

        # noinspection PyBroadException
        try:
            ev_user_profile_object = analytics_output_object.get('appliance_profile', {}).get(bill_cycle_start, {}).get('profileList')[0].get(str(ev_app_id))

            if ev_user_profile_object is None or len(ev_user_profile_object) == 0:
                ev_user_profile_object = [default_user_profile(int(bill_cycle_start), int(bill_cycle_end))]

            if debug is None or len(debug) == 0 or debug.get('ev_propensity_score') is None:
                logger.info('Writing default values to user propensity since debug object is None | ')
            else:
                logger.info('Writing user profile values for bill cycle | {}'.format(bill_cycle_start))

                ev_user_profile_object[0]['attributes']['evPropensity'] = debug.get('ev_propensity_score')

            # Populate appliance profile for the given bill cycle

            analytics_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str(ev_app_id)] = \
                deepcopy(ev_user_profile_object)

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.error('EV Propensity Profile Fill for billcycle %d failed | %s' % (int(bill_cycle_start), error_str))
            logger.info('EV Propensity Profile Empty for billcycle | %d ', int(bill_cycle_start))

        # Schema Validation for filled appliance profile

        validate_appliance_profile_schema_for_billcycle(analytics_output_object, bill_cycle_start, logger_pass)

        logger.info('EV propensity user profile complete for bill cycle | {}'.format(bill_cycle_start))

    return analytics_output_object


def default_user_profile(bc_start, bc_end):
    """
    This function initialises default ev user profile
    Parameters:
        bc_start     (int)      : Bill cycle start timestamp
        bc_end       (int)      : Bill cycle end timestamp
    Returns:
        profile      (dict)      : Default ev profile
    """

    profile = {
        "validity": {
            "start": bc_start,
            "end": bc_end
        },
        "isPresent": False,
        "detectionConfidence": None,
        "count": None,
        "attributes": {
            "evPropensity": None,
            "evConsumption": None,
            "chargerType": None,
            "amplitude": None,
            "chargingInstanceCount": None,
            "averageChargingDuration": None,
            "timeOfUsage": None
        },
        "debugAttributes": {}
    }

    return profile
