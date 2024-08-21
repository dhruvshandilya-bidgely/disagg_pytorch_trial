"""
Author - Sahana M
Date - 21/08/2021
Fill the appliance profile for wh
"""

# Import python packages

import logging
import traceback
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.mappings.get_app_id import get_app_id
from python3.utils.prepare_bc_tou_for_profile import prepare_bc_tou_for_profile
from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def get_waterheater_profile(item_input_object, item_output_object, logger_base, debug=None):

    """
    Parameters:
        item_input_object               (dict)      : Dictionary containing all item inputs
        item_output_object              (dict)      : Dictionary containing all item outputs
        logger_base             (logger)    : Logger object
        debug                   (dict)      : Output of all algorithm steps
    Returns:
        user_profile_object     (dict)      : User profile from all appliances (ref added)
    """
    # Taking logger base for this function

    logger_wh_base = logger_base.get("logger").getChild("get_waterheater_profile")
    logger_pass = {"logger": logger_wh_base,
                   "logging_dict": item_input_object.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_wh_base, logger_base.get("logging_dict"))

    wh_app_id = get_app_id('wh')

    scale_factor = Cgbdisagg.SEC_IN_HOUR / item_input_object['config']['sampling_rate']

    # Initialize variables needed to populate bc level time of usage

    out_bill_cycles = item_input_object.get('out_bill_cycles')

    # Get Time of Usage

    water_heater_out_idx = item_output_object.get('output_write_idx_map').get('wh')
    wh_estimate = item_output_object.get('epoch_estimate')[:, water_heater_out_idx]
    input_data = item_input_object.get('input_data')

    tou_arr = prepare_bc_tou_for_profile(input_data, wh_estimate, out_bill_cycles)

    for idx, row in enumerate(out_bill_cycles):
        bill_cycle_start, bill_cycle_end = row[:2]

        try:
            profile = default_user_profile(bill_cycle_start, bill_cycle_end)

            if debug is None:

                logger.info('Writing default values to user profile since debug is None | ')
            else:
                logger.info('Writing user profile values for bill cycle | {}'.format(bill_cycle_start))

                profile, filled_timed = seasonal_wh_profile(debug, profile, bill_cycle_start, scale_factor)
                profile["isPresent"] = bool(debug.get('swh_hld'))
                profile["attributes"]["timeOfUsage"] = list(tou_arr[bill_cycle_start])

            # Populate appliance profile for the given bill cycle

            item_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str(wh_app_id)] = \
                [deepcopy(profile)]

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.error('Water Heater Profile Fill for billcycle %d failed | %s' % (int(bill_cycle_start), error_str))
            logger.info('Water Heater Profile Empty for billcycle | %d ', int(bill_cycle_start))

        # Schema Validation for filled appliance profile

        validate_appliance_profile_schema_for_billcycle(item_output_object, bill_cycle_start, logger_pass)

        logger.info('User profile complete for bill cycle | {}'.format(bill_cycle_start))

    return item_output_object


def default_user_profile(bc_start, bc_end):

    """
    Returns:
         profile        (dict)      : Updated profile with water heater profile
    """

    profile = {
        "validity"           : {
            "start": int(bc_start),
            "end"  : int(bc_end)
        },
        "isPresent"          : False,
        "detectionConfidence": 0.0,
        "count"              : 0,
        "attributes"         : {
            "whConsumption"       : 0.0,
            "appType"             : None,
            "fuelType"            : None,
            "runsCount"           : 0,
            "amplitude"           : 0.0,
            "amplitudeConfidence" : 0.0,
            "dailyThinPulseCount" : 0,
            "passiveUsageFraction": 0.0,
            "activeUsageFraction" : 0.0,
            "timeOfUsage"         : None
        },
        "debugAttributes"    : {}
    }

    return profile


def seasonal_wh_profile(debug, profile, bc_start, factor):

    """
        Parameters:
            debug           (dict)      : Output of all algorithm steps
            profile         (dict)      : User profile for water heater
            bc_start        (int)       : Bill cycle start timestamp
            factor          (int)       : Sampling rate division
        Returns:
            profile         (dict)      : Updated user profile for water heater
            filled          (boolean)   : Whether this water heater present
        """

    swh_hld = debug.get('swh_hld')

    if (swh_hld is None) or (swh_hld == 0):
        filled = False
    else:
        filled = True

        swh_output = debug['final_wh_signal']

        bc_swh_output = swh_output[swh_output[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == bc_start]

        consumption_sum = np.nansum(bc_swh_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        if consumption_sum > 0:
            profile['isPresent'] = True
            profile['detectionConfidence'] = float(debug.get('swh_confidence'))
            profile['count'] = 1

            profile['attributes']['whConsumption'] = float(consumption_sum)
            profile['attributes']['appType'] = 'seasonal'

            profile['attributes']['fuelType'] = 'Electric'

            profile['attributes']['runsCount'] = int(1)

            profile['attributes']['amplitude'] = float(debug['final_swh_amplitude'] * factor)
            profile['attributes']['amplitudeConfidence'] = debug.get('swh_confidence')

    return profile, filled
