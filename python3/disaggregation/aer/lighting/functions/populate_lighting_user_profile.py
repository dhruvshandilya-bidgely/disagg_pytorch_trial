"""
Author - Mayank Sharan
Date - 06th Nov 2019
Populate attributes in the user profile dictionary for lighting
"""

# Import python packages

import copy
import traceback
import logging
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.maths_utils.maths_utils import moving_sum

from python3.config.mappings.get_app_id import get_app_id

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def populate_lighting_user_profile(disagg_input_object, disagg_output_object, debug, logger_pass):

    """
    Populate the lighting user profile object by bill cycle

    Parameters:
        disagg_input_object     (dict)              : Dictionary containing all inputs
        disagg_output_object    (dict)              : Dictionary containing all outputs
        debug                   (dict)              : Dictionary containing all intermediate debug variables
        logger_pass             (dict)              : Contains base logger and logging dictionary

    Returns:
        disagg_output_object    (dict)              : Dictionary containing all outputs
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('populate_lighting_user_profile')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # Initialize constants to be used

    bc_start_col = 0
    li_cons_col = 1

    li_app_id = get_app_id('li')

    # Initialize variables needed to populate bc level time of usage

    out_bill_cycles = disagg_input_object.get('out_bill_cycles')
    sampling_rate = disagg_input_object.get('config').get('sampling_rate')
    pd_mult = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    # Extract variables from debug object needed to populate TOU

    month_ts = debug.get('data').get('month_ts')
    lighting_data = debug.get('data').get('lighting')
    monthly_lighting = debug.get('results').get('monthly_lighting')

    # Prepare 1d data for TOU calculation

    epoch_ts = debug.get('data').get('ts')
    ts_1d = np.reshape(epoch_ts, newshape=(len(epoch_ts) * len(epoch_ts[0]),))
    lighting_1d = np.reshape(lighting_data, newshape=(len(lighting_data) * len(lighting_data[0]),))
    lighting_epoch = np.c_[ts_1d, lighting_1d]

    input_data = copy.deepcopy(disagg_input_object.get('input_data'))
    lighting_input_aligned = np.zeros(shape=(input_data.shape[0], 2))

    lighting_input_aligned[:, 0] = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    _, idx_1, idx_2 = np.intersect1d(lighting_input_aligned[:, 0], lighting_epoch[:, 0], return_indices=True)

    lighting_input_aligned[idx_1, 1] = lighting_epoch[idx_2, 1]

    for bill_cycle_idx in range(out_bill_cycles.shape[0]):

        # Extract the bill cycle to populate the profile for

        bill_cycle_start = out_bill_cycles[bill_cycle_idx, bc_start_col]

        logger.debug('Lighting appliance profile population started for | %d', bill_cycle_start)

        try:

            # Initialize the dictionary containing the user profile attributes

            user_profile_object = dict(
                {
                    "validity": None,
                    "isPresent": True,
                    "detectionConfidence": 1.0,
                    "count": None,
                    "attributes": {
                        "lightingConsumption": None,
                        "morningCapacity": None,
                        "eveningCapacity": None,
                        "timeOfUsage": None,
                    },
                    "debugAttributes": {}
                }
            )

            # Fill morning and evening capacity values

            user_profile_object['attributes']['morningCapacity'] = \
                float(np.round(debug.get('params').get('morning_capacity'), 2))

            user_profile_object['attributes']['eveningCapacity'] = \
                float(np.round(debug.get('params').get('evening_capacity'), 2))

            # For each bill cycle compute values 0 to 1 reflecting TOU and convert it to the hourly level

            days_in_bc_bool = np.sum(month_ts == bill_cycle_start, axis=1) > 0

            bc_lighting_data = lighting_data[days_in_bc_bool, :]
            bc_lighting_data_bool = bc_lighting_data > 0

            bc_tou = np.sum(bc_lighting_data_bool, axis=0) / np.sum(days_in_bc_bool)

            band_moving_sum = moving_sum(bc_tou, pd_mult)
            moving_idx = np.arange(pd_mult - 1, len(bc_tou), pd_mult)

            mod_band = np.round(band_moving_sum[moving_idx] / pd_mult, 5)

            user_profile_object['attributes']['timeOfUsage'] = [float(x) for x in mod_band]

            logger.info('Lighting TOU for bc | %d - %s', bill_cycle_start, ','.join(mod_band.astype(str)))

            # Populate consumption for the bill cycle

            bc_row_bool = monthly_lighting[:, bc_start_col] == bill_cycle_start
            bc_lighting_cons = monthly_lighting[bc_row_bool, li_cons_col]

            if len(bc_lighting_cons) == 0:
                bc_lighting_cons = 0

            user_profile_object['attributes']['lightingConsumption'] = float(np.round(bc_lighting_cons, 2))

            user_profile_list = [copy.deepcopy(user_profile_object)]

            # Populate appliance profile for the given bill cycle

            disagg_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str(li_app_id)] = \
                user_profile_list

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.error('Lighting Profile Fill for billcycle %d failed | %s' % (int(bill_cycle_start), error_str))
            logger.info('Lighting Profile Empty for billcycle | %d ', int(bill_cycle_start))

        # Schema Validation for filled appliance profile

        validate_appliance_profile_schema_for_billcycle(disagg_output_object, bill_cycle_start, logger_pass)

        logger.debug('Lighting appliance profile population completed for | %d', bill_cycle_start)

    return disagg_output_object
