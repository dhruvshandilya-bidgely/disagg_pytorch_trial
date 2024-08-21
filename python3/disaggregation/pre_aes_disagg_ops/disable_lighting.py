"""
Author: Neelabh Goyal
Date:   14 June 2023
This function disables lighting modules for SMB pipeline based on conditions
"""

# Import functions from within the project

from python3.config.smb_type_constants import SMBTypeConstants
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
    smb_type = disagg_input_object.get('home_meta_data').get('smb_type')
    sampling_rate = disagg_input_object.get('config').get('sampling_rate')

    # Based on the conditions decide if lighting should be disabled

    if 'li_smb' in module_seq and \
            (smb_type.upper() in SMBTypeConstants.LIGHTING_DISABLED_SMB_TYPES or
             pilot_id in PilotConstants.SMB_LIGHTING_DISABLED_PILOTS):

        module_seq.remove('li_smb')
        logger.info('Lighting disabled | sampling rate : %.1f, pilot id : %s, smb type : %s',
                    sampling_rate, str(pilot_id), str(smb_type))

    if 'statistical_li_smb' in module_seq and (smb_type.upper() in SMBTypeConstants.LIGHTING_DISABLED_SMB_TYPES or
                                               pilot_id in PilotConstants.LIGHTING_DISABLED_PILOTS):
        module_seq.remove('statistical_li_smb')
        logger.info('Statistical Lighting disabled | sampling rate : %.1f, pilot id : %s, smb type : %s',
                    sampling_rate, str(pilot_id), str(smb_type))

    # Write modified module sequence
    disagg_input_object['config']['module_seq'] = module_seq

    return disagg_input_object
