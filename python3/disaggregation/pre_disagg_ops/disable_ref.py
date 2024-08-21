"""
Author - Mayank Sharan
Date - 30/01/19
This function disables refrigerator based on conditions
"""

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants


def disable_ref(disagg_input_object, logger):

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
    sampling_rate = disagg_input_object.get('config').get('sampling_rate')
    app_profile_ref = disagg_input_object.get('app_profile').get('ref')

    if app_profile_ref is None:
        ref_count = -1
    else:
        ref_count = int(app_profile_ref.get('number'))

    # Based on the conditions decide if ref should be disabled

    if 'ref' in module_seq and \
            (sampling_rate == Cgbdisagg.SEC_IN_HOUR or ref_count == 0 or pilot_id in PilotConstants.REF_DISABLED_PILOTS):
        module_seq.remove('ref')
        logger.info('Ref disabled | sampling rate : %.1f, ref count : %d, pilot id : %s', sampling_rate, ref_count,
                    str(pilot_id))

    # Write modified module sequence

    disagg_input_object['config']['module_seq'] = module_seq

    return disagg_input_object
