"""
Author - Paras Tehria
Date - 01-Dec-2020
This module disables solar propensity module
"""

# Import functions from within the project

from python3.config.pilot_constants import PilotConstants


def disable_solar_propensity(analytics_input_object, logger):
    """
    This function disables solar propensity module

    Parameters:
        analytics_input_object (dict)              : Contains all inputs required to run the pipeline
        logger              (logger)            : Logger object to write logging statements

    Returns:
        analytics_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Extract parameters used for decision-making

    pilot_id = analytics_input_object.get('config').get('pilot_id')

    module_seq = analytics_input_object.get('config').get('module_seq')

    # Based on the conditions decide if solar propensity should be disabled

    if 'so_propensity' in module_seq and (pilot_id not in PilotConstants.SOLAR_PROPENSITY_ENABLED_PILOTS):
        module_seq.remove('so_propensity')
        logger.info('Solar propensity disabled | pilot id : %s', str(pilot_id))

    # Write modified module sequence

    analytics_input_object['config']['module_seq'] = module_seq

    return analytics_input_object
