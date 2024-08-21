"""
Author - Paras Tehria
Date - 08-Dec-2020
Updates the EV user profile
"""

# Import python packages

import logging
import traceback
import numpy as np
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.mappings.get_app_id import get_app_id

from python3.utils.maths_utils.maths_utils import create_pivot_table

from python3.utils.prepare_bc_tou_for_profile import prepare_bc_tou_for_profile

from python3.master_pipeline.preprocessing.downsample_data import downsample_data

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def populate_bc_ev_profile(debug, bill_cycle_start, bill_cycle_end, tou_dict, logger):

    """
    Populates EV profile for a given bill cycle

    Parameters:
        debug                   (dict)      : Output of all algorithm steps
        bill_cycle_start        (int)       : Timestamp marking the start of the billcycle
        bill_cycle_end          (int)       : Timestamp marking the end of the billcycle
        tou_dict                (dict)      : Dictionary containing TOU of each billing cycle
        logger                  (logger)    : Logging object to add log statements

    Returns:
        user_profile_object     (dict)      : Appliance profile for Ev for the given bill cycle
    """

    # Initialize user profile object

    user_profile_object = default_user_profile(int(bill_cycle_start), int(bill_cycle_end))

    if debug is None or len(debug) == 0 or debug.get('ev_probability') is None:
        logger.info('Writing default values to user profile for bill cycle | {}'.format(bill_cycle_start))
    else:
        logger.info('Writing user profile values for bill cycle | {}'.format(bill_cycle_start))

        # Populate presence of the EV

        ev_hld = debug.get('ev_hld')
        user_profile_object['isPresent'] = bool(debug.get('ev_hld'))
        user_profile_object['count'] = 1

        # Populate the confidence of EV detection

        user_profile_object['detectionConfidence'] = float(debug.get('ev_probability'))

        if ev_hld:

            # Populate the EV attributes for the current bill cycle
            user_profile_object["attributes"]['chargerType'] = debug.get("charger_type")

            user_profile_object["attributes"]['amplitude'] = float(debug.get('ev_amplitude'))
            user_profile_object["attributes"]['averageChargingDuration'] = float(debug.get('mean_duration'))

            # Populate the EV consumption for the bill cycle
            ev_signal = debug.get('final_ev_signal')

            ev_epoch_bc_idx = np.where((ev_signal[:, Cgbdisagg.INPUT_EPOCH_IDX] <= bill_cycle_end) & (
                ev_signal[:, Cgbdisagg.INPUT_EPOCH_IDX] >= bill_cycle_start))[0]
            ev_epoch_bc_cons = float(np.sum(ev_signal[ev_epoch_bc_idx][:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
            user_profile_object['attributes']['evConsumption'] = ev_epoch_bc_cons

            ev_signal_subset = ev_signal[ev_epoch_bc_idx]
            if len(ev_signal_subset) == 0:
                logger.info("faulty bill cycle | {}".format(bill_cycle_start))
                return {}

            instance_count = get_ev_instance_count(ev_signal_subset)

            user_profile_object["attributes"]["chargingInstanceCount"] = instance_count
            user_profile_object['attributes']['timeOfUsage'] = tou_dict[bill_cycle_start].tolist()

        else:
            user_profile_object["attributes"]['chargerType'] = None
            user_profile_object["attributes"]['amplitude'] = 0.0
            user_profile_object["attributes"]['averageChargingDuration'] = 0.0
            user_profile_object["attributes"]['chargingInstanceCount'] = 0
            user_profile_object['attributes']['evConsumption'] = 0.0
            user_profile_object['attributes']['timeOfUsage'] = None

    return user_profile_object


def get_ev_profile(disagg_input_object, disagg_output_object, logger_base, debug=None):

    """
    This function populates EV user profile

    Parameters:
        disagg_input_object     (dict)      : Dict containing all the inputs to the pipeline
        disagg_output_object    (dict)      : Dict containing all the outputs of the pipeline
        logger_base             (logger)    : Logger object
        debug                   (dict)      : Output of all algorithm steps

    Returns:
        disagg_output_object    (dict)      : Dict containing all the outputs of the pipeline
    """

    # Taking logger base for this function

    logger_ev_base = logger_base.get("logger").getChild("get_ev_profile")
    logger_pass = {"logger": logger_ev_base,
                   "logging_dict": disagg_input_object.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_ev_base, logger_base.get("logging_dict"))

    ev_app_id = get_app_id('ev')

    # Initialize variables needed to populate bc level time of usage

    out_bill_cycles = disagg_input_object.get('out_bill_cycles')

    # Prepare 1d data for TOU calculation

    ev_out_idx = disagg_output_object.get('output_write_idx_map').get('ev')
    ev_ts_estimate = disagg_output_object.get('epoch_estimate')[:, ev_out_idx]
    input_data = disagg_input_object.get('input_data')

    # dump_output(item_input_object, item_output_object)

    tou_dict = prepare_bc_tou_for_profile(input_data, ev_ts_estimate, out_bill_cycles)

    for row in out_bill_cycles:
        bill_cycle_start, bill_cycle_end = row[:2]

        # noinspection PyBroadException
        try:
            user_profile_object = populate_bc_ev_profile(debug, bill_cycle_start, bill_cycle_end, tou_dict, logger)

            # Populate appliance profile for the given bill cycle

            if len(user_profile_object) > 0:
                disagg_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str(ev_app_id)] = \
                    [deepcopy(user_profile_object)]

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.error('EV Profile Fill for billcycle %d failed | %s' % (int(bill_cycle_start), error_str))
            logger.info('EV Profile Empty for billcycle | %d ', int(bill_cycle_start))

        # Schema Validation for filled appliance profile

        validate_appliance_profile_schema_for_billcycle(disagg_output_object, bill_cycle_start, logger_pass)

        logger.info('User profile complete for bill cycle | {}'.format(bill_cycle_start))

    return disagg_output_object


def get_ev_instance_count(ev_signal):
    """
    This function calculates ev instances in a billing cycle

    Parameters:
        ev_signal             (np.ndarray)      : EV consumption matrix

    Returns:
        ev_instance_count     (int)             : Number of EV instances in the bill cycle
    """
    ev_signal_idx = (ev_signal[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] > 0).astype(int)

    ev_box_start_idx = np.where(np.diff(np.r_[0, ev_signal_idx, 0]) == 1)[0]

    ev_instance_count = len(ev_box_start_idx)

    return ev_instance_count


def get_ev_tou_arr(ev_signal):
    """
    This function calculates ev tou array in a bill cycle

    Parameters:
        ev_signal             (np.ndarray)      : EV consumption matrix

    Returns:
        tou_arr               (np.array)        : array of length 24 representing fraction of days with ev charging at that particular hour of the day
    """

    ev_signal_copy = deepcopy(ev_signal)

    ev_signal_downsampled = downsample_data(ev_signal_copy, Cgbdisagg.SEC_IN_HOUR)

    consumption_pivot, _, _ = create_pivot_table(data=ev_signal_downsampled, index=Cgbdisagg.INPUT_DAY_IDX,
                                                 columns=Cgbdisagg.INPUT_HOD_IDX,
                                                 values=Cgbdisagg.INPUT_CONSUMPTION_IDX)

    consumption_pivot[consumption_pivot > 0] = 1
    tou_arr = np.nansum(consumption_pivot, axis=0) / len(consumption_pivot)
    return tou_arr


def default_user_profile(bc_start, bc_end):
    """
    This function initialises default EV user profile

    Parameters:
        bc_start     (int)      : Bill cycle start timestamp
        bc_end       (int)      : Bill cycle end timestamp

    Returns:
        profile      (dict)      : Default EV profile
    """

    profile = {
        "validity": {
            "start": bc_start,
            "end": bc_end
        },
        "isPresent": False,
        "detectionConfidence": None,
        "count": None,
        "attributes": {
            "evPropensity": None,
            "evConsumption": None,
            "chargerType": None,
            "amplitude": None,
            "chargingInstanceCount": None,
            "averageChargingDuration": None,
            "timeOfUsage": None
        },
        "debugAttributes": {}
    }

    return profile
