"""
Author - Mayank Sharan
Date - 09/03/2019
This function disables all modules for japan. To be renamed to a more suitable name and modified later on
"""

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants


def disable_pipeline(disagg_input_object, logger):

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
    disagg_mode = disagg_input_object.get('config').get('disagg_mode')

    # Based on the pilot_id and disagg mode decide if pipeline should run or not

    if disagg_mode in ['incremental', 'mtd'] and pilot_id in PilotConstants.PIPELINE_DISABLED_FOR_INCREMENTAL_MTD:
        module_seq = []
        logger.info('All modules disabled | disagg mode : %s, pilot id : %s', disagg_mode, str(pilot_id))

    # Write modified module sequence

    disagg_input_object['config']['module_seq'] = module_seq

    return disagg_input_object
