"""
Author - Paras Tehria
Date - 01-Dec-2020
This module disables solar disagg module
"""

# Import functions from within the project

from python3.config.pilot_constants import PilotConstants


def disable_solar(disagg_input_object, logger):
    """
    This function disables solar disagg module

    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        logger              (logger)            : Logger object to write logging statements

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Extract parameters used for decision-making

    pilot_id = disagg_input_object.get('config').get('pilot_id')

    module_seq = disagg_input_object.get('config').get('module_seq')

    # Based on the conditions decide if solar should be disabled

    if 'solar' in module_seq and (
            pilot_id not in PilotConstants.SOLAR_DETECTION_ENABLED_PILOTS and pilot_id not in PilotConstants.SOLAR_ESTIMATION_ENABLED_PILOTS):

        module_seq.remove('solar')
        logger.info('Solar module disabled | pilot id : %s', str(pilot_id))

    # Write modified module sequence

    disagg_input_object['config']['module_seq'] = module_seq

    return disagg_input_object
