"""
Author - Sahana M
Date - 1/09/2021
This function disables water heater in Itemization based on conditions
"""
import copy

# Import functions from within the project

from python3.config.pilot_constants import PilotConstants


def disable_water_heater(item_input_object, logger):

    """
    Parameters:
        item_input_object (dict)              : Contains all inputs required to run the pipeline
        logger              (logger)            : Logger object to write logging statements

    Returns:
        item_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Extract variables to make the decision on

    pilot_id = item_input_object.get('config').get('pilot_id')
    sampling_rate = item_input_object.get('config').get('sampling_rate')
    module_seq = copy.deepcopy(item_input_object.get('config').get('itemization_module_seq'))

    # Based on the conditions decide if water heater should be disabled

    if 'wh' in module_seq and (pilot_id not in PilotConstants.SEASONAL_WH_ENABLED_PILOTS):
        module_seq.remove('wh')
        logger.info('Water Heater disabled | sampling rate : %.1f, pilot id : %s', sampling_rate, str(pilot_id))

    # Write modified module sequence

    item_input_object['config']['itemization_module_seq'] = module_seq

    return item_input_object
