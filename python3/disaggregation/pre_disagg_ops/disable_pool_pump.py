"""
Author - Arpan Agrawal
Date - 26/04/19
This function disables pool pump based on conditions
"""

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants


def disable_pool_pump(disagg_input_object, logger):

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
    pp_app_profile = disagg_input_object.get('app_profile').get('pp')

    # Based on the conditions decide if pool pump should be disabled

    pp_app_profile_bool = False
    pp_app_profile_no = False

    if pp_app_profile is not None:
        pp_number = pp_app_profile.get('number')
        if pp_number > 0:
            pp_app_profile_bool = True
        elif pp_number <= 0:
            pp_app_profile_no = True

    run_for_all_but_no_bool = pilot_id in PilotConstants.PILOTS_TO_RUN_PP_FOR_ALL_USERS and not pp_app_profile_no

    if 'pp' in module_seq and (not(pp_app_profile_bool or pilot_id in PilotConstants.AUSTRALIA_PILOTS or
                                   run_for_all_but_no_bool)
                               or pilot_id in PilotConstants.HVAC_JAPAN_PILOTS):

        module_seq.remove('pp')
        logger.info('Pool Pump disabled | PP Present Status : %s, pilot id : %s',
                    str(pp_app_profile_bool), str(pilot_id))

    # Write modified module sequence

    disagg_input_object['config']['module_seq'] = module_seq

    return disagg_input_object
