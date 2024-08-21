"""
Author - Paras Tehria
Date - 08-Dec-2020
Updates the solar propensity user profile
"""

# Import python packages

import logging
import traceback
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.mappings.get_app_id import get_app_id
from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def get_solar_propensity_user_profile(analytics_input_object, analytics_output_object, logger_base, debug):
    """
    This function populates solar propensity user profile

    Parameters:
        analytics_input_object     (dict)      : Dict containing all the inputs to the pipeline
        analytics_output_object    (dict)      : Dict containing all the outputs to the pipeline
        logger_base                (logger)    : Logger object
    Returns:
        analytics_output_object    (dict)      : All th outputs of analytics pipeline
    """
    # Taking logger base for this function

    logger_solar_base = logger_base.get("logger_base").getChild("get_solar_propensity_user_profile")
    logger_pass = {"logger": logger_solar_base,
                   "logging_dict": analytics_input_object.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_solar_base, logger_base.get("logging_dict"))

    solar_app_id = get_app_id('solar')

    # Initialize variables needed to populate bc level time of usage

    out_bill_cycles = analytics_input_object.get('out_bill_cycles')

    for idx, row in enumerate(out_bill_cycles):
        bill_cycle_start, bill_cycle_end = row[:2]

        # noinspection PyBroadException
        try:
            solar_user_profile_object = analytics_output_object.get('appliance_profile', {}).get(bill_cycle_start, {}).get('profileList')[0].get(str(solar_app_id))

            if solar_user_profile_object is None or len(solar_user_profile_object) == 0:
                solar_user_profile_object = [default_user_profile(int(bill_cycle_start), int(bill_cycle_end))]

            if debug is None or len(debug) == 0:
                logger.info('Writing default values to user propensity since debug object is None | ')
            else:
                logger.info('Writing user profile values for bill cycle | {}'.format(bill_cycle_start))

                solar_user_profile_object[0]['attributes']['solarPropensity'] = debug.get('propensity_score_model_2')
                solar_user_profile_object[0]['attributes']['requiredPanelCapacity'] = debug.get('panel_capacity')
                solar_user_profile_object[0]['attributes']['breakEvenPeriod'] = debug.get('break_even_period')

            # Populate appliance profile for the given bill cycle

            analytics_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str(solar_app_id)] = \
                deepcopy(solar_user_profile_object)

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.error('Solar Propensity Profile Fill for billcycle %d failed | %s' % (int(bill_cycle_start), error_str))
            logger.info('Solar Propensity Profile Empty for billcycle | %d ', int(bill_cycle_start))

        # Schema Validation for filled appliance profile

        validate_appliance_profile_schema_for_billcycle(analytics_output_object, bill_cycle_start, logger_pass)

        logger.info('User profile complete for bill cycle | {}'.format(bill_cycle_start))

    return analytics_output_object


def default_user_profile(bc_start, bc_end):
    """
    This function initialises default solar user profile

    Parameters:
        bc_start     (int)      : Bill cycle start timestamp
        bc_end       (int)      : Bill cycle end timestamp

    Returns:
        profile      (dict)      : Default solar profile
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
            "solarPropensity": None,
            "requiredPanelCapacity": None,
            "breakEvenPeriod": None,
            "solarGeneration": None,
            "chunkStart": None,
            "chunksEnd": None,
            "chunksConfidence": None,
            "excessGeneration": None,
            "solarCapacity": None,
            "solarCapacityConfidence": None,
            "timeOfUsage": None
        },
        "debugAttributes": {}
    }

    return profile
