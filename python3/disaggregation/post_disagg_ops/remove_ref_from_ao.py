"""
Author - Mayank Sharan
Date - 27/01/19
This function removes "ref" from AO estimates
"""

# Import python packages

import copy
import numpy as np
from datetime import datetime
from numpy.random import RandomState

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants


def init_ref_removal_from_ao_params(disagg_input_object):

    """Utility to initialize parameters for ref removal from AO"""

    sampling_rate = disagg_input_object.get('config').get('sampling_rate')
    pilot_id = disagg_input_object.get('config').get('pilot_id')

    # Initialize parameters based on pilot id and sampling rate

    if pilot_id in PilotConstants.HVAC_JAPAN_PILOTS:

        adj_params = {
            'alpha': 0.3,
            'max_ao_adjust': 75000,
            'min_ao_for_adjust': 25000,
        }

        if sampling_rate <= Cgbdisagg.SEC_IN_15_MIN:
            adj_params['alpha'] = 0
        elif sampling_rate == Cgbdisagg.SEC_IN_30_MIN:
            adj_params['alpha'] = 0.25
    else:

        adj_params = {
            'alpha': 0.2,
            'max_ao_adjust': 45000,
            'min_ao_for_adjust': 25000,
        }

        if sampling_rate <= Cgbdisagg.SEC_IN_15_MIN:
            adj_params['alpha'] = 0
        elif sampling_rate == Cgbdisagg.SEC_IN_30_MIN:
            adj_params['alpha'] = 0.15

    return adj_params


def remove_ref_from_ao(disagg_input_object, disagg_output_object, logger):
    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        disagg_output_object(dict)              : Dictionary containing all outputs
        logger              (logger)            : Logger object to write logging statements

    Returns:
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Extract sampling rate

    sampling_rate = disagg_input_object.get('config').get('sampling_rate')
    pilot_id = disagg_input_object.get('config').get('pilot_id')

    # Extract ao estimate and modify it

    ao_est_col = disagg_output_object.get('output_write_idx_map').get('ao')

    # If ao is not being run for the user no modification

    if ao_est_col is None:
        return disagg_output_object

    # Adjust the AO estimate in case the user is from Repsol or has 60 minute sampling rate

    if (pilot_id in PilotConstants.REPSOL_REF_REMOVAL_METHOD_PILOTS) and (sampling_rate == Cgbdisagg.SEC_IN_HOUR):

        # Initial declarations

        bill_cycle_start_col = 0
        values_col = 1

        # Initialise parameters for AO adjustment

        ref_removal_config = PilotConstants.REPSOL_REF_REMOVAL_CONFIG[pilot_id]
        slope = ref_removal_config.get('slope')
        intercept = ref_removal_config.get('intercept')
        min_allowed_ao = ref_removal_config.get('min_allowed_ao')
        ao_est = disagg_output_object.get('bill_cycle_estimate')[:, [bill_cycle_start_col, ao_est_col]]

        pro_rate_factor = np.ones_like(ao_est)
        pro_rate_factor[:, bill_cycle_start_col] = ao_est[:, bill_cycle_start_col]
        raw_data = disagg_input_object['input_data']

        # Compute number of days present in each bill cycle

        for bill_cycle in pro_rate_factor[:, bill_cycle_start_col]:

            bill_cycle_index = raw_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == bill_cycle

            pro_rate_factor[pro_rate_factor[:, bill_cycle_start_col] == bill_cycle, values_col] = \
                (len(np.unique(raw_data[bill_cycle_index, Cgbdisagg.INPUT_DAY_IDX])) /
                 ref_removal_config.get('default_days'))

        # Computing multipliers for ref

        ref_app_profile = disagg_input_object.get('app_profile').get('ref')
        ref_count = 0

        if ref_app_profile is not None:
            ref_count = int(ref_app_profile.get('number')) - 1

        ref_count_multiplier = 1 + ref_count * ref_removal_config['additional_ref_multiplier']

        # Code to reduce ref consumption

        intercept = intercept * pro_rate_factor[:, values_col]
        min_allowed_ao = min_allowed_ao * pro_rate_factor[:, values_col]
        reduce_by = np.zeros_like(ao_est)
        reduce_by[:, bill_cycle_start_col] = ao_est[:, bill_cycle_start_col]
        eligible_idx = ao_est[:, values_col] > min_allowed_ao
        reduce_by[eligible_idx, values_col] = (ao_est[eligible_idx, values_col] * slope) + intercept[eligible_idx]
        reduce_by[:, values_col] = reduce_by[:, values_col] * ref_count_multiplier
        changed_ao = np.zeros_like(ao_est)
        changed_ao[:, bill_cycle_start_col] = ao_est[:, bill_cycle_start_col]
        changed_ao[:, values_col] = ao_est[:, values_col] - reduce_by[:, values_col]

        eligible_idx = (ao_est[:, values_col] > min_allowed_ao) & (changed_ao[:, values_col] < min_allowed_ao)

        # Generating a random number between -1 and 1
        seed = RandomState(12345)
        random_number = seed.rand() * 2 - 1

        changed_ao[eligible_idx, values_col] = changed_ao[eligible_idx, values_col] + intercept[
            eligible_idx] * random_number
        disagg_output_object['bill_cycle_estimate'][:, ao_est_col] = changed_ao[:, values_col]

        monthly_ao_output = [(datetime.utcfromtimestamp(changed_ao[i, bill_cycle_start_col]).strftime('%b-%Y'),
                              changed_ao[i, values_col]) for i in range(changed_ao.shape[0])]
        logger.info("The monthly always on consumption (in remove_ref_from_ao) after slope based removal is : | %s",
                    str(monthly_ao_output).replace('\n', ' '))

        pro_rate_factor[:, values_col] = pro_rate_factor[:, values_col] * ref_removal_config.get('default_days')

        monthly_prf = [(datetime.utcfromtimestamp(pro_rate_factor[i, bill_cycle_start_col]).strftime('%b-%Y'),
                        pro_rate_factor[i, values_col]) for i in range(pro_rate_factor.shape[0])]

        logger.info("The number of days (in remove_ref_from_ao) after slope based removal is : | %s",
                    str(monthly_prf).replace('\n', ' '))

    else:

        # AO adjustment for users not from Repsol or without 60-minute

        ao_est = disagg_output_object.get('bill_cycle_estimate')[:, ao_est_col]
        reduce_by = np.zeros_like(ao_est)

        # Set parameters as per pilot id and sampling rate

        adj_params = init_ref_removal_from_ao_params(disagg_input_object)

        logger.info('Alpha for ao adjustment set to | %.2f', adj_params.get('alpha'))

        eligible_idx = ao_est > adj_params.get('min_ao_for_adjust')

        reduce_by[eligible_idx] = (ao_est[eligible_idx] - adj_params.get('min_ao_for_adjust')) * adj_params.get('alpha')
        reduce_by = np.minimum(reduce_by, adj_params.get('max_ao_adjust'))

        logger.info('Reduction in ao as ref adjustment | %s', str(reduce_by).replace('\n', ' '))

        # Write the modified ao estimate

        mod_ao_est = ao_est - reduce_by
        disagg_output_object['bill_cycle_estimate'][:, ao_est_col] = mod_ao_est

    # Log the final ao consumption

    monthly_output_log = \
        [(datetime.utcfromtimestamp(disagg_output_object.get('bill_cycle_estimate')[i, 0]).strftime('%b-%Y'),
          disagg_output_object.get('bill_cycle_estimate')[i, ao_est_col]) for i in range(ao_est.shape[0])]

    logger.info('The monthly ao consumption after adjustment (in Wh) is : | %s',
                str(monthly_output_log).replace('\n', ' '))

    ao_bc_level = disagg_output_object['bill_cycle_estimate'][:, ao_est_col]
    bc_list_in_bc_level_est = disagg_output_object['bill_cycle_estimate'][:, 0]
    bc_list = disagg_input_object.get("input_data")[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]
    ao_est = disagg_output_object.get("epoch_estimate")[:, ao_est_col]

    unique_bc, bc_count, bc_indices = np.unique(bc_list, return_counts=True, return_index=True)

    # Extending the AO adjustment to epoch level

    if disagg_input_object.get("config").get("disagg_mode") in ["historical", "incremental"]:
        for bc in unique_bc:
            current_billing_cycle_days = bc_list_in_bc_level_est == bc
            if np.any(current_billing_cycle_days):
                ao_indices = np.logical_and(bc_list == bc, ao_est > 0)
                ao_est[ao_indices] = ao_bc_level[current_billing_cycle_days][0] / np.sum(ao_indices)
    else:
        ao_est[:] = (ao_bc_level).sum() / len(bc_list)

    ao_est = np.minimum(ao_est, disagg_input_object.get("input_data")[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    disagg_output_object.get("epoch_estimate")[:, ao_est_col] = ao_est

    return disagg_output_object
