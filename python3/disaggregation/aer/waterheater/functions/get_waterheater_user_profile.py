"""
Author - Nikhil Singh Chauhan
Date - 28/May/2020
Call the refrigerator disaggregation module and get results
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


def get_waterheater_profile(disagg_input_object, disagg_output_object, logger_base, debug=None):

    """
    Parameters:
        user_profile_object     (dict)      : User profile from all appliances
        logger_base             (logger)    : Logger object
        debug                   (dict)      : Output of all algorithm steps
    Returns:
        user_profile_object     (dict)      : User profile from all appliances (ref added)
    """
    # Taking logger base for this function

    logger_wh_base = logger_base.get("logger").getChild("get_waterheater_profile")
    logger_pass = {"logger"      : logger_wh_base,
                   "logging_dict": disagg_input_object.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_wh_base, logger_base.get("logging_dict"))

    wh_app_id = get_app_id('wh')

    scale_factor = Cgbdisagg.SEC_IN_HOUR / disagg_input_object['config']['sampling_rate']

    # Initialize variables needed to populate bc level time of usage

    out_bill_cycles = disagg_input_object.get('out_bill_cycles')

    # Get Time of Usage

    water_heater_out_idx = disagg_output_object.get('output_write_idx_map').get('wh')
    wh_estimate = disagg_output_object.get('epoch_estimate')[:, water_heater_out_idx]
    input_data = disagg_input_object.get('input_data')

    tou_arr = prepare_bc_tou_for_profile(input_data, wh_estimate, out_bill_cycles)

    for row in out_bill_cycles:
        bill_cycle_start, bill_cycle_end = row[:2]

        try:
            profile = default_user_profile(bill_cycle_start, bill_cycle_end)

            if debug is None:

                logger.info('Writing default values to user profile since debug is None | ')
            else:
                logger.info('Writing user profile values for bill cycle | {}'.format(bill_cycle_start))

                profile, filled_timed = timed_profile(debug, profile, bill_cycle_start, scale_factor, logger)
                profile, filled_thermostat = thermostat_profile(debug, profile, bill_cycle_start, scale_factor, logger)
                profile["isPresent"] = bool(debug.get('timed_hld')) or bool(debug.get('thermostat_hld'))
                profile["attributes"]["timeOfUsage"] = list(tou_arr[bill_cycle_start])

            # Populate appliance profile for the given bill cycle

            disagg_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str(wh_app_id)] = \
                [deepcopy(profile)]

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.error('Water Heater Profile Fill for billcycle %d failed | %s' % (int(bill_cycle_start), error_str))
            logger.info('Water Heater Profile Empty for billcycle | %d ', int(bill_cycle_start))

        # Schema Validation for filled appliance profile

        validate_appliance_profile_schema_for_billcycle(disagg_output_object, bill_cycle_start, logger_pass)

        logger.info('User profile complete for bill cycle | {}'.format(bill_cycle_start))

    return disagg_output_object


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


def timed_profile(debug, profile, bc_start, factor, logger):

    """
        Parameters:
            debug           (dict)      : Output of all algorithm steps
            profile         (dict)      : User profile for water heater
            logger          (logger)    : Logger object for logs
        Returns:
            profile         (dict)      : Updated user profile for water heater
            filled          (boolean)   : Whether this water heater present
        """

    timed_hld = debug.get('timed_hld')

    timed_debug = debug.get('timed_debug')

    if (timed_hld is None) or (timed_hld == 0):
        filled = False
    else:
        filled = True

        timed_output = debug['timed_wh_signal']

        bc_timed_output = timed_output[timed_output[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == bc_start]

        consumption_sum = np.nansum(bc_timed_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

        if consumption_sum > 0:
            profile['isPresent'] = True
            profile['detectionConfidence'] = float(timed_debug['timed_confidence'])
            profile['count'] = 1

            profile['attributes']['whConsumption'] = float(consumption_sum)
            profile['attributes']['appType'] = 'timed'

            profile['attributes']['fuelType'] = 'Electric'

            profile['attributes']['runsCount'] = int(timed_debug['num_runs'])

            profile['attributes']['amplitude'] = float(debug['timed_wh_amplitude'] * factor)
            profile['attributes']['amplitudeConfidence'] = timed_debug['timed_confidence']

    return profile, filled


def thermostat_profile(debug, profile, bc_start, factor, logger):

    """
    Parameters:
        debug           (dict)      : Output of all algorithm steps
        profile         (dict)      : User profile for water heater
        logger          (logger)    : Logger object for logs
    Returns:
        profile         (dict)      : Updated user profile for water heater
        filled          (boolean)   : Whether this water heater present
    """
    thermostat_hld = debug.get('thermostat_hld')
    timed_hld = debug.get('timed_hld')

    if (thermostat_hld == 1) and ((timed_hld is None) or (timed_hld == 0)):
        filled = True

        thermostat_output = debug['final_wh_signal']

        non_zero_wh_output = thermostat_output[thermostat_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0]

        max_energy_point = np.percentile(non_zero_wh_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX], 95)

        wh_amplitude = float(max_energy_point * factor)

        final_estimate = debug['final_estimate']

        month_estimate = final_estimate[final_estimate[:, 0] == bc_start]

        if len(month_estimate) > 0:
            thin_consumption = month_estimate[0, 3]
            fat_consumption = month_estimate[0, 4]
        else:
            thin_consumption = 0
            fat_consumption = 0

        consumption_sum = thin_consumption + fat_consumption

        daily_thin_pulse_count = get_daily_thin_pulse(debug, bc_start)

        if consumption_sum > 0:
            profile['isPresent'] = True
            profile['detectionConfidence'] = float(debug['thermostat_hld_prob'])
            profile['count'] = 1

            profile['attributes']['whConsumption'] = float(consumption_sum)
            profile['attributes']['appType'] = 'thermostat'

            profile['attributes']['fuelType'] = 'Electric'

            profile['attributes']['runsCount'] = 1

            profile['attributes']['amplitude'] = wh_amplitude
            profile['attributes']['amplitudeConfidence'] = float(debug['thermostat_hld_prob'])

            profile['attributes']['dailyThinPulseCount'] = int(daily_thin_pulse_count)
            profile['attributes']['passiveUsageFraction'] = float(thin_consumption / consumption_sum)
            profile['attributes']['activeUsageFraction'] = float(fat_consumption / consumption_sum)

    else:
        filled = False

    return profile, filled


def get_daily_thin_pulse(debug, bc_start):
    """
    This function is used to calculate the daily thin pulse count for a billing cycle
    Parameters:
        debug                   (dict)          : Debug dictionary
        bc_start                (float)         : Bill cycle
    Returns:
        daily_thin_pulse_count  (int)           : Daily average thin pulses in the billing cycle
    """

    # Extract the required variables

    bill_cycle_idx = debug['bill_cycle_idx']
    final_estimate = deepcopy(debug['final_estimate'])
    thin_pulse_signal = deepcopy(debug['final_thin_output'])

    # Get the billing cycle number and the corresponding indices

    bill_cycle_idx_number = np.where(bc_start == final_estimate[:, 0])[0]

    if len(bill_cycle_idx_number):
        bill_cycle_idx_bool = bill_cycle_idx == bill_cycle_idx_number
        thin_pulse_signal = thin_pulse_signal[bill_cycle_idx_bool, :]

        # Get the start indexes of all the thin pulses

        thin_pulse_signal_bool = (thin_pulse_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0)
        thin_pulse_signal_idx_diff = np.diff(np.r_[0, thin_pulse_signal_bool.astype(int)])
        thin_pulse_signal_idx_diff[thin_pulse_signal_idx_diff < 0] = 0
        thin_pulse_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = thin_pulse_signal_idx_diff

        # For each unique day calculate the total thin pulses and get the average for a bill cycle

        unq, idx, cnt = np.unique(thin_pulse_signal[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True, return_counts=True)
        thin_pulses = np.bincount(idx, weights=thin_pulse_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
        thin_pulses_bool = thin_pulses > 0

        if np.nansum(thin_pulses_bool):
            daily_thin_pulse_count = int(np.nanmean(thin_pulses[thin_pulses_bool]))
        else:
            daily_thin_pulse_count = 0

    else:
        daily_thin_pulse_count = 0

    return daily_thin_pulse_count
