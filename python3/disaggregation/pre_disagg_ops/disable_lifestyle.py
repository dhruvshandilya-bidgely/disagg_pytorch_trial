"""
Author - Nisha Agarwal
Date - 15th Oct
This function disables lifestyle based on conditions
"""

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants


def disable_lifestyle(disagg_input_object, logger):

    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        logger              (logger)            : Logger object to write logging statements

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Extract variables to make the decision on

    pilot_id = disagg_input_object.get('config').get('pilot_id')
    user_profile_module_seq = disagg_input_object.get('config').get('user_profile_module_seq')

    # Based on the conditions decide if water heater should be disabled

    if 'life' in user_profile_module_seq and (pilot_id not in PilotConstants.LIFESTYLE_ENABLED_PILOTS):
        user_profile_module_seq.remove('life')
        logger.info('Lifestyle disabled | pilot id : %s', str(pilot_id))

    # Write modified module sequence

    disagg_input_object['config']['user_profile_module_seq'] = user_profile_module_seq

    return disagg_input_object
