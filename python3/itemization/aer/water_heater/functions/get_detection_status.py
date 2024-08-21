"""
Author - Sahana M
Date - 20/4/2021
Function to identify the detection status of WH from Disagg run
"""


def get_detection_status(global_config, item_input_object, hsm_fail, hsm_in, logger):
    """
    Function to identify the detection status of Water heater from the Disaggregation run
    Args:
        global_config                  (dict)       : Dictionary containing all the global configurations
        item_input_object              (dict)       : Dictionary containing all disagg outputs
        hsm_fail                       (bool)       : Boolean denoting HSM fetch status
        hsm_in                         (dict)       : Dictionary containing HSM attributes
        logger                         (Logger)     : Logging object
    Returns:
        disable_run                    (bool)       : Boolean for wh detection status
    """

    disagg_mode = global_config.get('disagg_mode')

    # Disable run based on app profile information

    disable_run = check_app_profile(item_input_object, logger)

    if not disable_run:

        # If run mode historical or incremental

        if (disagg_mode == 'historical') or (disagg_mode == 'incremental'):

            # If water heater already detected from the Disagg module then don't run this module

            if not (item_input_object.get('created_hsm').get('wh') is None or
                    (item_input_object.get('created_hsm').get('wh').get('attributes').get('thermostat_hld') == 0) and
                    (item_input_object.get('created_hsm').get('wh').get('attributes').get('timed_hld') == 0)):

                disable_run = True

        # If run mode is mtd then use available hsm

        elif disagg_mode == 'mtd' and not hsm_fail:

            # Run this module only if mtd is not run in the Disagg run

            if not ((hsm_in.get('attributes').get('timed_hld') == [0] and hsm_in.get('attributes').get('thermostat_hld') == [0]) or (
                    hsm_in.get('attributes').get('timed_hld') is None and hsm_in.get('attributes').get('thermostat_hld') is None)):

                disable_run = True

    return disable_run


def check_app_profile(item_input_object, logger):
    """
    Disable run based on the app profile information
    Parameters:
        item_input_object              (np.ndarray) :  Dictionary containing all disagg outputs
        logger                         (Logger)     : Logging object
    Returns:
        disable_run                    (bool)       : Boolean for swh run status
    """

    disable_run = True

    # Disable based on App profile information

    wh_app_profile = item_input_object.get('app_profile').get('wh')
    block_wh_types = ['GAS', 'Gas', 'gas', 'PROPANE', 'Propane', 'propane']

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

    if wh_present:
        disable_run = False

    return disable_run
