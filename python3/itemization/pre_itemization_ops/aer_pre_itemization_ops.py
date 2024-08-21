"""
Author - Sahana M
Date - 1/09/2021
This function calls all operations we need to do before we run the Itemization
"""

# Import python packages

import logging

# Import functions from within the project

from python3.itemization.pre_itemization_ops.disable_water_heater import disable_water_heater

from python3.itemization.pre_itemization_ops.disable_hybrid_hvac import disable_hybrid_hvac
from python3.itemization.pre_itemization_ops.disable_poolpump import disable_hybrid_poolpump
from python3.itemization.pre_itemization_ops.disable_hybrid_water_heater import disable_hybrid_water_heater
from python3.itemization.pre_itemization_ops.disable_electric_vehicle import disable_hybrid_electric_vehicle

from python3.disaggregation.pre_disagg_ops.reconstruct_rounded_signal import reconstruct_rounded_signal


def aer_pre_itemization_ops(item_input_object):

    """
    Parameters:
        item_input_object (dict)              : Contains all inputs required to run the pipeline

    Returns:
        item_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Initialize the logger

    logger_base = item_input_object.get('logger').getChild('aer_pre_itemization_ops')
    logger = logging.LoggerAdapter(logger_base, item_input_object.get('logging_dict'))

    # Disable appliances for specific pilots and sampling rates

    item_input_object = disable_water_heater(item_input_object, logger)

    item_input_object = disable_hybrid_hvac(item_input_object, logger)
    item_input_object = disable_hybrid_poolpump(item_input_object, logger)
    item_input_object = disable_hybrid_water_heater(item_input_object, logger)
    item_input_object = disable_hybrid_electric_vehicle(item_input_object, logger)

    item_input_object = reconstruct_rounded_signal(item_input_object, logger)

    return item_input_object
