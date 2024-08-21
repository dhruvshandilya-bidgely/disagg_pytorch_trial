"""
Author - Mayank Sharan
Date - 27/01/19
This function calls all operations we need to do before we run the disagg
"""

# Import python packages

import logging

# Import functions from within the project


from python3.disaggregation.pre_disagg_ops.disable_ref import disable_ref
from python3.disaggregation.pre_disagg_ops.disable_solar import disable_solar
from python3.disaggregation.pre_disagg_ops.disable_lighting import disable_lighting
from python3.disaggregation.pre_disagg_ops.disable_pool_pump import disable_pool_pump
from python3.disaggregation.pre_disagg_ops.disable_pipeline import disable_pipeline
from python3.disaggregation.pre_disagg_ops.disable_lifestyle import disable_lifestyle
from python3.disaggregation.pre_disagg_ops.disable_water_heater import disable_water_heater
from python3.disaggregation.pre_disagg_ops.reconstruct_rounded_signal import reconstruct_rounded_signal


def pre_aer_disagg_ops(disagg_input_object):

    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Initialize the logger

    logger_base = disagg_input_object.get('logger').getChild('pre_aer_disagg_ops')
    logger = logging.LoggerAdapter(logger_base, disagg_input_object.get('logging_dict'))

    # Disable appliances for specific pilots and sampling rates

    disagg_input_object = disable_lighting(disagg_input_object, logger)
    disagg_input_object = disable_pool_pump(disagg_input_object, logger)
    disagg_input_object = disable_water_heater(disagg_input_object, logger)
    disagg_input_object = disable_ref(disagg_input_object, logger)
    disagg_input_object = disable_solar(disagg_input_object, logger)
    disagg_input_object = disable_pipeline(disagg_input_object, logger)

    # Reconstruct progressive signal

    disagg_input_object = reconstruct_rounded_signal(disagg_input_object, logger)

    return disagg_input_object
