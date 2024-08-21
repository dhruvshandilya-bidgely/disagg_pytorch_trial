"""
Author - Nikhil Singh Chauhan
Date - 28/MAY/2020
Call the refrigerator disaggregation module and get results
"""

# Import python packages

import logging
import traceback
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.mappings.get_app_id import get_app_id

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def get_ref_profile(disagg_input_object, disagg_output_object, logger_base, debug=None):

    """
    Parameters:
        user_profile_object     (dict)      : User profile from all appliances
        logger_base             (logger)    : Logger object
        debug                   (dict)      : Output of all algorithm steps
    Returns:
        user_profile_object     (dict)      : User profile from all appliances (ref added)
    """
    # Taking logger base for this function

    logger_ref_base = logger_base.get("logger").getChild("get_ref_profile")
    logger_pass = {"logger": logger_ref_base,
                   "logging_dict": disagg_input_object.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_ref_base, logger_base.get("logging_dict"))

    ref_app_id = get_app_id('ref')

    profile = {
        "validity": {
            "start": 0,
            "end": 0
        },
        "isPresent": False,
        "detectionConfidence": 0.0,
        "count": 0,
        "attributes": {
            "refConsumption": 0.0,
            "amplitude": 0.0,
            "multipleRef": False,
            "summerAmplitude": 0.0,
            "winterAmplitude": 0.0,
            "transitionAmplitude": 0.0
        },
        "debugAttributes": {}
    }

    # Initialize variables needed to populate bc level time of usage

    out_bill_cycles = disagg_input_object.get('out_bill_cycles')

    for row in out_bill_cycles:

        bill_cycle_start, bill_cycle_end = row[:2]

        try:

            if debug is None:

                logger.info('Writing default values to user profile since debug is None | ')
            else:
                ref_hsm = debug.get('hsm')

                if ref_hsm is not None:
                    logger.info('Writing user profile values for bill cycle | {}'.format(bill_cycle_start))

                    profile['validity'] = {
                        'start': int(bill_cycle_start),
                        'end': int(bill_cycle_end)
                    }

                    profile['isPresent'] = True

                    profile['detectionConfidence'] = 1.0

                    profile['count'] = 1

                    debug['sampling_rate'] = disagg_input_object.get('config').get('sampling_rate')

                    profile['attributes'] = get_seasonal_estimate(debug, bill_cycle_start, bill_cycle_end, ref_hsm, logger)

                    logger.info('Valid ref energy per data point from hsm | ')
                else:

                    logger.info('Invalid ref hsm, writing default default values to user profile | ')

            # Populate appliance profile for the given bill cycle

            disagg_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str(ref_app_id)] = \
                [deepcopy(profile)]

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.error('Ref Profile Fill for billcycle %d failed | %s' % (int(bill_cycle_start), error_str))
            logger.info('Ref Profile Empty for billcycle | %d ', int(bill_cycle_start))

        # Schema Validation for filled appliance profile

        validate_appliance_profile_schema_for_billcycle(disagg_output_object, bill_cycle_start, logger_pass)

        logger.info('User profile complete for bill cycle | {}'.format(bill_cycle_start))

    return disagg_output_object


def get_seasonal_estimate(debug, bc_start, bc_end, ref_hsm, logger):

    """
    Parameters:
        debug       (dict)      : Output of all module steps
        profile     (dict)      : User profile w.r.t. ref
        logger      (logger)    : Logger object to log values
    Returns:
        profile     (dict)      : Updated user profile
    """

    # Initialize ref attributes

    attributes = {}

    ref_output = debug['refHourlyOutput']

    bc_ref_output = ref_output[(ref_output[:, 0] >= bc_start) & (ref_output[:, 0] < bc_end)]

    attributes['refConsumption'] = np.nansum(bc_ref_output[:, 2])

    scaling_factor = Cgbdisagg.SEC_IN_HOUR / debug.get('sampling_rate')

    attributes['amplitude'] = ref_hsm.get('attributes').get('Ref_Energy_Per_DataPoint') * scaling_factor

    attributes['multipleRef'] = False

    # Check for winter

    if debug.get('wtrEstimatedRef') is not None:

        attributes['winterAmplitude'] = debug['wtrEstimatedRef'] * scaling_factor

        if not np.isnan(attributes['winterAmplitude']):
            logger.info('Winter ref value found | ')
        else:
            attributes['winterAmplitude'] = -1.
            logger.info('Winter ref value not found | ')

    # Check for intermediate
    if debug.get('itrEstimatedRef') is not None:

        attributes['transitionAmplitude'] = debug['itrEstimatedRef'] * scaling_factor

        if not np.isnan(attributes['transitionAmplitude']):
            logger.info('Intermediate ref value found | ')
        else:
            attributes['transitionAmplitude'] = -1.
            logger.info('Intermediate ref value not found | ')

    # Check for summer
    if debug.get('smrEstimatedRef') is not None:

        attributes['summerAmplitude'] = debug['smrEstimatedRef'] * scaling_factor

        if not np.isnan(attributes['summerAmplitude']):
            logger.info('Summer ref value found | ')
        else:
            attributes['summerAmplitude'] = -1.
            logger.info('Summer ref value not found | ')

    return attributes
