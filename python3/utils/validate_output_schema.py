"""
Author - Prasoon Patidar
Date - 28/04/2020
Validate schema of various profiles based on WriteAPI Format
"""

# Import python packages

import logging
import traceback
from schema import Schema

# Import functions from within the project

from python3.initialisation.object_initialisations.appliance_profile_schema import appliance_profile_schema
from python3.initialisation.object_initialisations.lifestyle_profile_schema import lifestyle_profile_schema


def validate_appliance_profile_schema_for_billcycle(disagg_output_object, billcycle_start, logger_pass):

    """
    Validates schema of Appliance Profile based on writeAPI schema

    Parameters:
        disagg_output_object(dict)              : Dictionary containing all outputs
        billcycle_start     (int)               : Start Time(in Epoch) for Bill Cycle that is being validated
        logger_pass         (dict)              : Contains base logger and logging dictionary

    Returns:
        None
    """

    logger_base = None

    if 'logger' in logger_pass.keys():

        logger_base = logger_pass.get('logger').getChild('appliance_profile_schema_validation')

    elif 'logger_base' in logger_pass.keys():

        logger_base = logger_pass.get('logger_base').getChild('appliance_profile_schema_validation')

    logger_app_profile_validation = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    exit_status = False

    # noinspection PyBroadException
    try:

        appliance_profile = disagg_output_object['appliance_profile'][billcycle_start]
        Schema(appliance_profile_schema).validate(appliance_profile)
        logger_app_profile_validation.debug('Appliance profile schema validation passed for bill cycle start | %d',
                                            billcycle_start)

        exit_status = True

    except Exception:

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_app_profile_validation.error('Appliance profile schema validation failed for bill cycle start | %d - %s',
                                            billcycle_start, error_str)

    return exit_status


def validate_lifestyle_profile_schema_for_billcycle(disagg_output_object, billcycle_start, logger_pass):

    """
    Validates schema of Lifestyle Profile based on writeAPI schema

    Parameters:
        disagg_output_object(dict)              : Dictionary containing all outputs
        billcycle_start     (int)               : Start Time(in Epoch) for Bill Cycle that is being validated
        logger_pass         (dict)              : Contains base logger and logging dictionary

    Returns:
        None
    """

    logger_base = None

    if 'logger' in logger_pass.keys():

        logger_base = logger_pass.get('logger').getChild('lifestyle_profile_schema_validation')

    elif 'logger_base' in logger_pass.keys():

        logger_base = logger_pass.get('logger_base').getChild('lifestyle_profile_schema_validation')

    logger_app_profile_validation = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    exit_status = False

    # noinspection PyBroadException
    try:

        lifestyle_profile = disagg_output_object['lifestyle_profile'][billcycle_start]
        Schema(lifestyle_profile_schema).validate(lifestyle_profile)
        logger_app_profile_validation.debug('Lifestyle profile schema validation passed for bill cycle start | %d',
                                            billcycle_start)

        exit_status = True

    except Exception:

        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_app_profile_validation.error('Lifestyle profile schema validation failed for bill cycle start | %d - %s',
                                            billcycle_start, error_str)

    return exit_status
