"""
Author - Nisha
Date - 1/05/2022
This function disables Pool pump in Itemization based on conditions
"""

# Import functions from within the project

from python3.config.pilot_constants import PilotConstants


def disable_hybrid_poolpump(item_input_object, logger):

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
    module_seq = item_input_object.get('config').get('itemization_to_disagg')

    # Based on the conditions decide if water heater should be disabled

    if 'pp' in module_seq and (pilot_id in PilotConstants.HYBRID_PP_DISABLED_PILOTS):
        module_seq.remove('pp')
        logger.info('Hybrid Poolpump disabled | sampling rate : %.1f, pilot id : %s', sampling_rate, str(pilot_id))

    # Write modified module sequence

    item_input_object['config']['itemization_to_disagg'] = module_seq

    return item_input_object
