"""
Author - Mayank Sharan
Date - 27/01/19
This function calls scales the ref output for multi ref scenarios
"""

# Import python modules

from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants


def adjust_for_multi_ref(disagg_input_object, disagg_output_object, logger):

    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        logger              (logger)            : Logger object to write logging statements

    Returns:
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Extract ref count from app profile

    pilot_id = disagg_input_object.get('config').get('pilot_id')
    sampling_rate = disagg_input_object.get('config').get('sampling_rate')

    ref_app_profile = disagg_input_object.get('app_profile').get('ref')
    ref_count = -1

    if ref_app_profile is not None:
        ref_count = int(ref_app_profile.get('number'))

    logger.info('Ref count from app profile is | %d', ref_count)

    ref_est_col = disagg_output_object.get('output_write_idx_map').get('ref')

    if ref_count == 0 or ref_est_col is None:
        return disagg_output_object

    # Based on the count, sampling rate and pilot id decide scaling factor

    additional_multi_ref_factor = 0

    # Compensating for all pilots with 30 min sampling rate

    if sampling_rate == Cgbdisagg.SEC_IN_30_MIN:
        additional_multi_ref_factor = 0.1

    # Compensating for ovo in terms of 50 % multi ref users

    if ref_count < 0 and pilot_id in PilotConstants.MULTI_REF_PILOTS:
        additional_multi_ref_factor = 0.1
    elif ref_count > 1:
        additional_multi_ref_factor = (ref_count - 1) * 0.3

    logger.info('Adjustment factor for multi ref is | %.1f', 1 + additional_multi_ref_factor)

    # Modify the ref output

    ref_est = disagg_output_object.get('bill_cycle_estimate')[:, ref_est_col]
    disagg_output_object['bill_cycle_estimate'][:, ref_est_col] = ref_est * (1 + additional_multi_ref_factor)

    # Log the final ref consumption

    monthly_output_log = \
        [(datetime.utcfromtimestamp(disagg_output_object.get('bill_cycle_estimate')[i, 0]).strftime('%b-%Y'),
          disagg_output_object.get('bill_cycle_estimate')[i, ref_est_col]) for i in range(ref_est.shape[0])]

    logger.info('The monthly ref consumption after adjustment (in Wh) is : | %s',
                str(monthly_output_log).replace('\n', ' '))


    return disagg_output_object


def adjust_for_multi_ref_epoch(disagg_input_object, disagg_output_object, logger):

    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        logger              (logger)            : Logger object to write logging statements

    Returns:
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Extract ref count from app profile

    pilot_id = disagg_input_object.get('config').get('pilot_id')
    sampling_rate = disagg_input_object.get('config').get('sampling_rate')

    ref_app_profile = disagg_input_object.get('app_profile').get('ref')
    ref_count = -1

    if (ref_app_profile is not None) and (ref_app_profile.get('number') is not None):
        ref_count = int(ref_app_profile.get('number'))

    logger.info('Ref count from app profile is | %d', ref_count)

    ref_est_col = disagg_output_object.get('output_write_idx_map').get('ref')

    if ref_count == 0 or ref_est_col is None:
        return disagg_output_object

    # Based on the count, sampling rate and pilot id decide scaling factor

    additional_multi_ref_factor = 0

    # Compensating for all pilots with 30 min sampling rate

    if sampling_rate == Cgbdisagg.SEC_IN_30_MIN:
        additional_multi_ref_factor = 0.1

    # Compensating for ovo in terms of 50 % multi ref users

    if ref_count < 0 and pilot_id in PilotConstants.MULTI_REF_PILOTS:
        additional_multi_ref_factor = 0.1
    elif ref_count > 1:
        additional_multi_ref_factor = (ref_count - 1) * 0.3

    logger.info('Adjustment factor for multi ref is | %.1f', 1 + additional_multi_ref_factor)

    # Modify the ref output

    ref_est = disagg_output_object.get('epoch_estimate')[:, ref_est_col]
    disagg_output_object['epoch_estimate'][:, ref_est_col] = ref_est * (1 + additional_multi_ref_factor)

    return disagg_output_object
