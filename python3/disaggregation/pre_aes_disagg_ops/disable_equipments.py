"""
Author: Neelabh Goyal
Date:   14 June 2023
This function disables equipment modules for SMB pipeline based on conditions
"""

# Import functions from within the project

from python3.config.pilot_constants import PilotConstants
from python3.config.smb_type_constants import SMBTypeConstants


def disable_equipments(disagg_input_object, logger):
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
    smb_type = disagg_input_object.get('home_meta_data').get('smb_type')
    module_seq = disagg_input_object.get('config').get('module_seq')

    # Based on the conditions decide if equipments should be disabled

    if 'equipments_smb' in module_seq and \
            (smb_type.upper() not in SMBTypeConstants.EQUIPMENTS_ENABLED_SMB_TYPES or
             pilot_id in PilotConstants.SMB_EQUIPMENTS_DISABLED_PILOTS):

        module_seq.remove('equipments_smb')
        logger.info('Equipment module disabled | pilot id : %s, smb type: %s', str(pilot_id), str(smb_type))

    # Write modified module sequence

    disagg_input_object['config']['module_seq'] = module_seq

    return disagg_input_object
