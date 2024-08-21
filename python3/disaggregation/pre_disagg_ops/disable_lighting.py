"""
Author - Mayank Sharan
Date - 27/01/19
This function disables lighting based on conditions
"""

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants


def disable_lighting(disagg_input_object, logger):

    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        logger              (logger)            : Logger object to write logging statements

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Extract variables to make the decision on

    pilot_id = disagg_input_object.get('config').get('pilot_id')
    module_seq = disagg_input_object.get('config').get('module_seq')
    sampling_rate = disagg_input_object.get('config').get('sampling_rate')

    # Based on the conditions decide if lighting should be disabled

    if 'li' in module_seq and \
            (sampling_rate == Cgbdisagg.SEC_IN_HOUR or pilot_id in PilotConstants.LIGHTING_DISABLED_PILOTS):
        module_seq.remove('li')
        logger.info('Lighting disabled | sampling rate : %.1f, pilot id : %s', sampling_rate, str(pilot_id))

    # Write modified module sequence

    disagg_input_object['config']['module_seq'] = module_seq

    return disagg_input_object
