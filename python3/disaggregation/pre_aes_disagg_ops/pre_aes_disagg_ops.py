"""
Author: Neelabh Goyal
Date:   14-June-2023
This function calls all operations we need to do before we run the SMB disagg
"""

# Import python packages

import logging

# Import functions from within the project

from python3.disaggregation.pre_aes_disagg_ops.disable_ref import disable_ref
from python3.disaggregation.pre_aes_disagg_ops.disable_service import disable_service
from python3.disaggregation.pre_aes_disagg_ops.disable_cooking import disable_cooking
from python3.disaggregation.pre_aes_disagg_ops.disable_pipeline import disable_pipeline
from python3.disaggregation.pre_aes_disagg_ops.disable_lighting import disable_lighting
from python3.disaggregation.pre_aes_disagg_ops.disable_equipments import disable_equipments
from python3.disaggregation.pre_aes_disagg_ops.disable_water_heater import disable_water_heater
from python3.disaggregation.pre_aes_disagg_ops.reconstruct_rounded_signal import reconstruct_rounded_signal


def pre_aes_disagg_ops(disagg_input_object):

    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Initialize the logger

    logger_base = disagg_input_object.get('logger').getChild('pre_aes_disagg_ops')
    logger = logging.LoggerAdapter(logger_base, disagg_input_object.get('logging_dict'))

    # Disable appliances for specific pilots and sampling rates
    if disagg_input_object.get('home_meta_data').get('smb_type') is None:
        disagg_input_object['home_meta_data']['smb_type'] = 'all'

    disagg_input_object = disable_lighting(disagg_input_object, logger)
    disagg_input_object = disable_cooking(disagg_input_object, logger)
    disagg_input_object = disable_water_heater(disagg_input_object, logger)
    disagg_input_object = disable_ref(disagg_input_object, logger)
    disagg_input_object = disable_equipments(disagg_input_object, logger)
    disagg_input_object = disable_service(disagg_input_object, logger)
    disagg_input_object = disable_pipeline(disagg_input_object, logger)

    # Reconstruct progressive signal

    disagg_input_object = reconstruct_rounded_signal(disagg_input_object, logger)

    return disagg_input_object
