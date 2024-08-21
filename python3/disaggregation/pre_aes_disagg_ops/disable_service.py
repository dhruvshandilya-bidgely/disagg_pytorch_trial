"""
Author: Neelabh Goyal
Date:   14 June 2023
This function disables church-service modules for SMB pipeline based on conditions
"""

# Import functions from within the project

from python3.config.smb_type_constants import SMBTypeConstants


def disable_service(disagg_input_object, logger):

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

    if 'service_smb' in module_seq and smb_type not in SMBTypeConstants.SERVICE_ENABLED_SMB_TYPES:

        module_seq.remove('service_smb')
        logger.info('Service module disabled | pilot id : %s, smb type: %s', str(pilot_id), str(smb_type))

    # Write modified module sequence

    disagg_input_object['config']['module_seq'] = module_seq

    return disagg_input_object
