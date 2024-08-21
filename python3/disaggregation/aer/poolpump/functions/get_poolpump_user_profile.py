"""
Author - Arpan Agrawal
Date - 28th May 2020
Populate attributes in the user profile dictionary for Poolpump
"""

# Import python packages

import copy
import logging
import traceback
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.mappings.get_app_id import get_app_id

from python3.utils.prepare_bc_tou_for_profile import prepare_bc_tou_for_profile

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def get_poolpump_user_profile(disagg_input_object, disagg_output_object, pp_info, logger_pass):

    """
    Function to bill cycle level user profile for pool pump

    Parameters:
        disagg_input_object     (dict)              : Dictionary containing all inputs
        disagg_output_object    (dict)              : Dictionary containing all outputs
        pp_info                 (dict)              : Dictionary containing pool pump info
        logger_pass             (dict)              : Contains base logger and logging dictionary

    Returns:
        disagg_output_object    (dict)              : Dictionary containing all outputs after updating user profile
    """

    # Initialize the logger

    logger_pp_base = logger_pass.get('logger_base').getChild('get_poolpump_user_profile')
    logger = logging.LoggerAdapter(logger_pp_base, disagg_input_object.get('logging_dict'))

    logger_pass = {
        'logger_base': logger_pp_base,
        'logging_dict': disagg_input_object.get('logging_dict'),
    }

    pp_app_id = get_app_id('pp')
    out_bill_cycles = disagg_input_object.get('out_bill_cycles')
    pp_epoch = pp_info['pp_epoch']

    pp_amplitude = 0
    pp_cons = pp_epoch[:, 1]

    pp_ts_estimate = disagg_output_object.get("special_outputs").get("pp_consumption")

    input_data = copy.deepcopy(disagg_input_object.get('input_data'))
    pp_input_aligned = np.zeros(shape=(input_data.shape[0], 2))

    pp_input_aligned[:, 0] = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    _, idx_1, idx_2 = np.intersect1d(pp_input_aligned[:, 0], pp_ts_estimate[:, 0], return_indices=True)

    pp_input_aligned[idx_1, 1] = pp_ts_estimate[idx_2, 1]

    # dump_output(item_input_object, item_output_object)

    tou_dict = prepare_bc_tou_for_profile(input_data, pp_input_aligned[:, 1], out_bill_cycles)

    # List of amplitude column indexes in schedule attribute
    amplitude_idx_list = [8, 9, 10]

    for row in out_bill_cycles:
        bill_cycle_start, bill_cycle_end = row[:2]

        logger.info('Writing pool pump user profile for bill cycle | {}'.format(bill_cycle_start))

        try:

            # Initialize user profile for a bill cycle

            user_profile_object = dict(
                {
                    "validity": {
                        "start": int(bill_cycle_start),
                        "end": int(bill_cycle_end),
                    },
                    "isPresent": None,
                    "detectionConfidence": None,
                    "count": None,
                    "attributes": {
                        "ppConsumption": None,
                        "appType": None,
                        "fuelType": None,
                        "numberOfRuns": None,
                        "amplitude": None,
                        "schedule": None,
                        "timeOfUsage": None,
                    },
                    "debugAttributes": {}
                })

            # Populate presence and count of the pool pump

            user_profile_object['isPresent'] = False
            user_profile_object['count'] = 0

            if np.sum(pp_cons) > 0:
                user_profile_object['isPresent'] = True
                user_profile_object['count'] = 1

            # Populate the confidence of pool pump detection

            user_profile_object['detectionConfidence'] = np.round(float(pp_info['confidence_value'])/100, 2)

            # Populate the pool pump consumption for the bill cycle

            pp_epoch_bc_idx = np.where((pp_epoch[:, 0] <= bill_cycle_end) & (pp_epoch[:, 0] >= bill_cycle_start))[0]
            pp_epoch_bc_cons = float(np.sum(pp_epoch[pp_epoch_bc_idx][:, 1]))
            user_profile_object['attributes']['ppConsumption'] = pp_epoch_bc_cons

            # Populate information associated with the type of pool pump

            pp_run_type_mapping = {
                0: 'NoRun',
                1: 'Single',
                2: 'Multiple',
                3: 'Variable'
            }

            user_profile_object['attributes']['appType'] = pp_run_type_mapping[pp_info['run_type_code']]
            user_profile_object['attributes']['fuelType'] = 'Electric'
            user_profile_object['attributes']['numberOfRuns'] = [float(pp_info['pp_runs'])]

            user_profile_object['attributes']["timeOfUsage"] = list(tou_dict[bill_cycle_start])

            # Populate the run schedule information of the pool pump

            schedules = pp_info['schedules']

            if schedules.shape[0] > 0:
                schedules_bc_idx = np.where((schedules[:, 0] <= bill_cycle_end) & (schedules[:, 3] >= bill_cycle_start))
                schedules_bc = schedules[schedules_bc_idx]
                user_profile_object['attributes']['schedule'] = np.array(schedules_bc, dtype=float).tolist()

                pp_amplitude = np.unique(schedules_bc[:, amplitude_idx_list]).astype(float)

                pp_amplitude = pp_amplitude[pp_amplitude > 0]

                user_profile_object['attributes']['amplitude'] = list(pp_amplitude)

            user_profile_list = [copy.deepcopy(user_profile_object)]

            # Populate appliance profile for the given bill cycle

            disagg_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str(pp_app_id)] = \
                user_profile_list

        except Exception:
            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.error('Poolpump Profile Fill for billcycle %d failed | %s' % (int(bill_cycle_start), error_str))
            logger.info('Poolpump Profile Empty for billcycle | %d ', int(bill_cycle_start))

        # Schema Validation for filled appliance profile

        validate_appliance_profile_schema_for_billcycle(disagg_output_object, bill_cycle_start, logger_pass)

        logger.debug('Completed pool pump user profile for bill cycle | %d', bill_cycle_start)

    return disagg_output_object
