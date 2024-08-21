"""
Author - Nikhil Singh Chauhan
Date - 16/10/19
Module to retrieve water heater app profile from disagg_input_object
"""


def get_wh_app_profile(disagg_input_object, wh_config, logger):
    """
    Parameters:
        disagg_input_object     (dict)      : Dictionary containing all inputs
        wh_config               (dict)      : WH configurations
        logger                  (logger)    : Logger object to logs values

    Returns:
        wh_present              (bool)      : Boolean to mark presence/absence of water heater
    """

    # Retrieve the water heater profile from disagg input object

    wh_app_profile = disagg_input_object.get('app_profile').get('wh')
    block_wh_types = wh_config.get('block_wh_types')

    # If no status info present in meta data then run module assuming present

    if wh_app_profile is not None:
        # wh_present is False if profile says zero water heater present

        wh_number = wh_app_profile.get('number')
        wh_type = wh_app_profile.get('type')

        wh_present_number = False if wh_number == 0 else True
        wh_present_type = False if (wh_type in block_wh_types) else True

        wh_present = wh_present_number and wh_present_type

        logger.info('Water heater present Status | %s', wh_present)
        logger.info('Water heater type | %s', wh_type)

    else:
        # wh_present is True if profile has no specific answer

        wh_present = True
        logger.info('Water heater present Status | Not Sure')

    return wh_present
