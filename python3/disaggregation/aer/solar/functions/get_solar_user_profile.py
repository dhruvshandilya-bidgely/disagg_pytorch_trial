"""
Author - Paras Tehria
Date - 08-Dec-2020
Updates the solar user profile
"""

# Import python packages

import logging
import traceback
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.mappings.get_app_id import get_app_id

from python3.utils.maths_utils.maths_utils import create_pivot_table

from python3.master_pipeline.preprocessing import downsample_data
from python3.utils.prepare_bc_tou_for_profile import prepare_bc_tou_for_profile
from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def get_solar_user_profile(disagg_input_object, disagg_output_object, logger_base):
    """
    This function populates solar user profile

    Parameters:
        disagg_input_object     (dict)      : Dict containing all the inputs to the pipeline
        disagg_output_object    (dict)      : Dict containing all the outputs to the pipeline
        logger_base             (logger)    : Logger object
    Returns:
        user_profile_object     (dict)      : User profile from all appliances
    """
    # Taking logger base for this function

    logger_solar_base = logger_base.get("logger").getChild("get_solar_user_profile")
    logger_pass = {"logger": logger_solar_base,
                   "logging_dict": disagg_input_object.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_solar_base, logger_base.get("logging_dict"))

    solar_app_id = get_app_id('solar')

    # Initialize variables needed to populate bc level time of usage

    out_bill_cycles = disagg_input_object.get('out_bill_cycles')

    # Get Time of Usage

    solar_out_idx = disagg_output_object.get('output_write_idx_map').get('solar')
    solar_estimate = disagg_output_object.get('epoch_estimate')[:, solar_out_idx]
    input_data = disagg_input_object.get('input_data')

    tou_arr = prepare_bc_tou_for_profile(input_data, solar_estimate, out_bill_cycles)

    solar_detection_hsm = disagg_output_object.get('created_hsm', {}).get('solar')

    for idx, row in enumerate(out_bill_cycles):
        bill_cycle_start, bill_cycle_end = row[:2]

        # noinspection PyBroadException
        try:
            user_profile_object = default_user_profile(int(bill_cycle_start), int(bill_cycle_end))

            if solar_detection_hsm is None or len(solar_detection_hsm) == 0:
                logger.info('Writing default values to user profile since solar hsm is None | ')
            else:
                logger.info('Writing user profile values for bill cycle | {}'.format(bill_cycle_start))

                # Populate presence of the solar

                solar_present = solar_detection_hsm.get('attributes', {}).get('solar_present')
                user_profile_object['isPresent'] = bool(solar_present)
                user_profile_object['attributes']['timeOfUsage'] = list(tou_arr[bill_cycle_start])
                user_profile_object['count'] = 1

                # Populate remaining solar appliance profile attributes for the given bill cycle

                user_profile_object = \
                    update_solar_appliance_profile_attributes(disagg_input_object, disagg_output_object,
                                                              solar_detection_hsm, user_profile_object, idx, solar_out_idx, logger)

            # Populate appliance profile for the given bill cycle

            disagg_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str(solar_app_id)] = \
                [deepcopy(user_profile_object)]

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.error('Solar Profile Fill for billcycle %d failed | %s' % (int(bill_cycle_start), error_str))
            logger.info('Solar Profile Empty for billcycle | %d ', int(bill_cycle_start))

        # Schema Validation for filled appliance profile

        validate_appliance_profile_schema_for_billcycle(disagg_output_object, bill_cycle_start, logger_pass)

        logger.info('User profile complete for bill cycle | {}'.format(bill_cycle_start))

    return disagg_output_object


def update_solar_appliance_profile_attributes(disagg_input_object, disagg_output_object, solar_detection_hsm,
                                              user_profile_object, idx, solar_out_idx, logger):

    """
    This function populates solar profile for a given billing cycle

    Parameters:
        disagg_input_object     (dict)      : Dict containing all the inputs to the pipeline
        disagg_output_object    (dict)      : Dict containing all the outputs to the pipeline
        solar_detection_hsm     (dict)      : Solar hsm
        user_profile_object     (dict)      : User profile from all appliances
        idx                     (int)       : index of current billing cuycle
        solar_out_idx           (int)       : index of solar
        logger                  (logger)    : solar Logger object
    Returns:
        user_profile_object     (dict)      : User profile from all appliances
    """

    user_type = solar_detection_hsm.get('attributes', {}).get('kind', {})

    is_new_user = True if user_type == 1 else False
    is_old_user = True if user_type == 2 else False

    samples_in_an_hour = int(Cgbdisagg.SEC_IN_HOUR / disagg_input_object.get('config').get('sampling_rate'))

    if (user_profile_object['isPresent']) & ('confidence' in solar_detection_hsm['attributes'].keys()):
        user_profile_object['detectionConfidence'] = float(solar_detection_hsm.get('attributes', {}).get('confidence'))

    if (solar_detection_hsm.get('attributes', {}).get('remove_solar_gen') is not None) and \
            (solar_detection_hsm.get('attributes', {}).get('remove_solar_gen') > 0):
        user_profile_object['isPresent'] = False
        user_profile_object['detectionConfidence'] = 0.0
        user_profile_object['count'] = 0

    if (user_profile_object['isPresent']) & ('r_squared_threshold' in solar_detection_hsm['attributes'].keys()):
        user_profile_object['detectionConfidence'] = float( solar_detection_hsm.get('attributes', {}).get('detection_confidence'))
        user_profile_object['attributes']['chunkStart'] = np.array(solar_detection_hsm.get('attributes', {}).get('chunk_start')).tolist()
        user_profile_object['attributes']['chunksEnd'] = np.array(solar_detection_hsm.get('attributes', {}).get('chunk_end')).tolist()
        user_profile_object['attributes']['solarCapacityConfidence'] = round(solar_detection_hsm.get('attributes', {}).get('r_squared_threshold'), 2)

        if solar_detection_hsm.get('attributes', {}).get('capacity') is not None:
            user_profile_object['attributes']['solarCapacity'] = solar_detection_hsm.get('attributes', {}).get('capacity') * samples_in_an_hour
        else:
            logger.warning('Solar Capacity is None | ')

        user_profile_object['attributes']["solarGeneration"] = disagg_output_object.get('bill_cycle_estimate', {})[idx, solar_out_idx]
        user_profile_object['attributes']['chunksConfidence'] = solar_detection_hsm.get('attributes', {}).get('instance_probabilities')
        user_profile_object['attributes']['startDate'] = solar_detection_hsm.get('attributes', {}).get('start_date')
        user_profile_object['attributes']['endDate'] = solar_detection_hsm.get('attributes', {}).get('end_date')
        user_profile_object['attributes']['isNewUser'] = is_new_user
        user_profile_object['attributes']['isOldUser'] = is_old_user

    return user_profile_object


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
            "timeOfUsage": None,
            "startDate": None,
            "isNewUser": False,
            "endDate": None,
            "isOldUser": False
        },
        "debugAttributes": {}
    }

    return profile
