"""
Author - Paras Tehria
Date - 28-May-2021
This function disables ev propensity module
"""

# Import functions from within the project

from python3.config.pilot_constants import PilotConstants


def disable_ev_propensity(analytics_input_object, logger):

    """
    Function to disable ev propensity

    Parameters:
        analytics_input_object (dict)              : Contains all inputs required to run the pipeline
        logger              (logger)            : Logger object to write logging statements

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Extract variables to make the decision on

    pilot_id = analytics_input_object.get('config').get('pilot_id')
    user_profile_module_seq = analytics_input_object.get('config').get('module_seq')

    # Based on the conditions decide if ev propensity should be disabled

    if 'ev_propensity' in user_profile_module_seq and (pilot_id not in PilotConstants.EV_PROPENSITY_ENABLED_PILOTS):
        user_profile_module_seq.remove('ev_propensity')
        logger.info('EV propensity is disabled pilot id : | %s', str(pilot_id))

    # EV propensity module will only work if lifestyle module has run

    if 'ev_propensity' in user_profile_module_seq and ('life' not in user_profile_module_seq):
        user_profile_module_seq.remove('ev_propensity')
        logger.info('EV propensity is disabled because lifestyle module is disabled | ')

    # Write modified module sequence

    analytics_input_object['config']['module_seq'] = user_profile_module_seq

    return analytics_input_object
