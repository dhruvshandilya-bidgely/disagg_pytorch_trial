
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Ref master file
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.ref.add_seasonality import add_seasonality
from python3.itemization.aer.ref.init_ref_config import init_ref_config
from python3.itemization.aer.ref.get_ref_estimate import get_ref_estimate
from python3.itemization.aer.ref.check_ref_estimate import check_ref_estimate
from python3.itemization.aer.ref.update_ref_estimate import update_ref_estimate


def ref_hybrid_module(item_input_object, item_output_object, logger_ref_pass):

    """
    Master function for ref modules

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        logger_ref_pass             (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        ref_estimate_ts             (numpy.ndarray) : ref consumption
        hybrid_ref                  (bool)          : bool to check whether ref is calculated from hybrid module
    """
    # Initialize ref logger

    logger_base = logger_ref_pass.get('logger_base').getChild('run_ref_module')
    logger = logging.LoggerAdapter(logger_base, logger_ref_pass.get('logging_dict'))
    logger_ref_pass['logger_base'] = logger_base

    t_ref_module_start = datetime.now()

    logger.debug("Start of ref hybrid module")

    # Get ref epoch level estimate from true disagg

    ref_config = init_ref_config()

    ref_column = item_input_object.get("disagg_output_write_idx_map").get("ref")

    ref_estimate = item_input_object["disagg_epoch_estimate"][:, ref_column]

    hybrid_ref = 0

    ref_estimate_disagg_bc_estimate = item_input_object["disagg_bill_cycle_estimate"][:, ref_column]

    factor = 1

    hsm_posting_flag = (item_input_object.get('config').get('disagg_mode') == 'historical') or \
                       (item_input_object.get('config').get('disagg_mode') == 'incremental' and
                        len(item_input_object.get('item_input_params').get('day_input_data')) >= 70)

    ref_hsm_present_flag = item_input_object.get("item_input_params").get('ref_hsm') is not None and \
                item_input_object.get("item_input_params").get('ref_hsm').get('hybrid_ref') is not None

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    if hybrid_config.get('geography') == 'eu':
        factor = factor * 1.3

    # If true disagg output is nan (or may be 60 min), calculate epoch estimate using hybrid module

    if np.all(np.logical_or(ref_estimate_disagg_bc_estimate == 0, np.isnan(ref_estimate_disagg_bc_estimate))) or \
            (np.all(np.logical_or(ref_estimate == 0, np.isnan(ref_estimate)))):

        hybrid_ref = 1

        ref_estimate, day_level_estimate, item_output_object = \
            prepare_ref_estimate_from_hybrid(item_input_object, item_output_object, ref_config, factor, logger)

    else:

        if not np.any(ref_estimate > 0):
            ref_estimate = int(np.median(np.nan_to_num(ref_estimate))*item_input_object.get("item_input_params").get("samples_per_hour")*24)
        else:
            ref_estimate = int(np.median(np.nan_to_num(ref_estimate)[np.nan_to_num(ref_estimate) > 0])*item_input_object.get("item_input_params").get("samples_per_hour")*24)

        ref_app_count = get_ref_app_ids_counts(item_input_object, ref_config)
        ref_estimate = update_ref_estimate(ref_estimate, ref_app_count, ref_config, logger)

        if hsm_posting_flag and \
                (item_output_object.get('created_hsm').get('ref') is None):
            item_output_object['created_hsm']['ref'] = {
                'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
                'attributes': dict()
            }

        ref_estimate = update_ref_estimate_based_on_hsm_for_disagg_users(ref_estimate, item_input_object, ref_hsm_present_flag)

        created_hsm = dict({
            'hybrid_ref': ref_estimate,
            'disagg_ref_bool': 1
        })

        if hsm_posting_flag and \
                (item_output_object.get('created_hsm') is not None and
                 item_output_object.get('created_hsm').get('ref') is not None) and \
                (item_output_object.get('created_hsm').get('ref').get('attributes') is not None):
            item_output_object['created_hsm']['ref']['attributes'].update(created_hsm)

        ref_estimate, day_level_estimate = add_seasonality(ref_estimate, item_input_object)

    ts_list = item_input_object.get("input_data")[:, Cgbdisagg.INPUT_EPOCH_IDX]

    ref_estimate_ts = np.vstack((ts_list, ref_estimate)).T

    t_ref_module_end = datetime.now()

    logger.debug("End of ref hybrid module")

    logger.info("Running main ref module took | %.3f s", get_time_diff(t_ref_module_start, t_ref_module_end))

    return item_input_object, item_output_object, ref_estimate_ts, hybrid_ref, day_level_estimate


def get_ref_app_ids_counts(item_input_object, ref_config):

    """
    Calculate ref app id count and type

    Parameters:
        item_input_object     (dict)             : Dict containing all hybrid inputs
        ref_config              (dict)             : Dictionary containing all information

    Returns:
        ref_app_count           (numpy.ndarray)    : Calculated ref app count
    """

    # Fetch user appliance profile information

    appliance_profile_list = list(item_input_object["app_profile"])

    # Initialize appliance counts

    ref_app_count = np.ones(ref_config.get("app_profile").get("ref_app_ids_count")) * -1

    ref_app_ids = ref_config.get("app_profile").get("ref_app_ids")

    # Calculate number of each category refrigerator appliances

    for appliance in appliance_profile_list:

        app_id = item_input_object["app_profile"][appliance]["appID"]

        if app_id in ref_app_ids:

            count = item_input_object.get("app_profile")[appliance]["number"]

            app_ids_column = ref_config.get("app_profile").get("ref_ids_columns")[ref_app_ids.index(app_id)]

            ref_app_count[app_ids_column] = 0 if count <= 0 else count

    return ref_app_count


def update_ref_estimate_based_on_hsm_for_disagg_users(ref_estimate, item_input_object, ref_hsm_present_flag):

    """
    update ref value based on hsm data for disagg users

    Parameters:
        ref_estimate              (float)         : ref consumption
        item_input_object         (dict)          : Dict containing all hybrid inputs
        ref_hsm_present_flag      (bool)          : flag to represent whether ref hsm is present

    Returns:
        ref_estimate              (float)         : ref consumption
    """

    valid_ref_hsm = item_input_object.get("item_input_params").get('valid_ref_hsm')

    disagg_ref_bool = -1

    if not (valid_ref_hsm and ref_hsm_present_flag):
        return ref_estimate

    weights_for_disagg_users = [0.75, 0.25]
    weights_for_non_disagg_users = [0.9, 0.1]

    ref_hsm = item_input_object.get("item_input_params").get('ref_hsm').get('hybrid_ref')

    # if previous run ref output source was true disagg, higher confidence is given to hsm ref value

    if item_input_object.get("item_input_params").get('ref_hsm').get('disagg_ref_bool') is not None and \
            (len(item_input_object.get('item_input_params').get('day_input_data')) >= 100):

        disagg_ref_bool = item_input_object.get("item_input_params").get('ref_hsm').get('disagg_ref_bool', 0)

        if isinstance(disagg_ref_bool, list):
            disagg_ref_bool = disagg_ref_bool[0]

    if disagg_ref_bool > 0:
        if isinstance(ref_hsm, list):
            ref_estimate = ref_estimate * weights_for_disagg_users[0] + ref_hsm[0] * weights_for_disagg_users[1]

        else:
            ref_estimate = ref_estimate * weights_for_disagg_users[0] + ref_hsm * weights_for_disagg_users[1]

    elif disagg_ref_bool <= 0:
        if isinstance(ref_hsm, list):
            ref_estimate = ref_estimate * weights_for_non_disagg_users[0] + ref_hsm[0] * weights_for_non_disagg_users[1]

        else:
            ref_estimate = ref_estimate * weights_for_non_disagg_users[0] + ref_hsm * weights_for_non_disagg_users[1]

    return ref_estimate


def update_ref_estimate_based_on_hsm(ref_estimate, item_input_object, ref_hsm_present_flag):

    """
    update ref value based on hsm data for non-disagg users

    Parameters:
        ref_estimate              (float)         : ref consumption
        item_input_object         (dict)          : Dict containing all hybrid inputs
        ref_hsm_present_flag      (bool)          : flag to represent whether ref hsm is present

    Returns:
        ref_estimate              (float)         : ref consumption
    """

    valid_ref_hsm = item_input_object.get("item_input_params").get('valid_ref_hsm')

    disagg_ref_bool = -1

    weights_for_nonzero_disagg_users = [0.1, 0.9]
    weights_for_zero_disagg_users = [0.35, 0.65]
    weight_with_disagg_info_absent = [0.2, 0.8]

    if not (valid_ref_hsm and ref_hsm_present_flag):
        return ref_estimate

    ref_hsm = item_input_object.get("item_input_params").get('ref_hsm').get('hybrid_ref')

    # if previous run ref output source was true disagg, higher confidence is given to hsm ref value

    if item_input_object.get("item_input_params").get('ref_hsm').get('disagg_ref_bool') is not None and \
            (len(item_input_object.get('item_input_params').get('day_input_data')) >= 100):

        disagg_ref_bool = item_input_object.get("item_input_params").get('ref_hsm').get('disagg_ref_bool', 0)

        if isinstance(disagg_ref_bool, list):
            disagg_ref_bool = disagg_ref_bool[0]

    if disagg_ref_bool > 0:
        if isinstance(ref_hsm, list) :
            ref_estimate = ref_estimate * weights_for_nonzero_disagg_users[0] + ref_hsm[0] * weights_for_nonzero_disagg_users[1]

        else:
            ref_estimate = ref_estimate * weights_for_nonzero_disagg_users[0] + ref_hsm * weights_for_nonzero_disagg_users[1]

    elif disagg_ref_bool == 0:
        if isinstance(ref_hsm, list):
            ref_estimate = ref_estimate * weights_for_zero_disagg_users[0] + ref_hsm[0] * weights_for_zero_disagg_users[1]

        else:
            ref_estimate = ref_estimate * weights_for_zero_disagg_users[0] + ref_hsm * weights_for_zero_disagg_users[1]

    else:
        if isinstance(ref_hsm, list):
            ref_estimate = ref_estimate * weight_with_disagg_info_absent[0] + ref_hsm[0] * weight_with_disagg_info_absent[1]

        else:
            ref_estimate = ref_estimate * weight_with_disagg_info_absent[0] + ref_hsm * weight_with_disagg_info_absent[1]

    return ref_estimate


def prepare_ref_estimate_from_hybrid(item_input_object, item_output_object, ref_config, factor, logger):

    """
    Function estimates ref for users with 0 disagg

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        ref_config                (dict)          : config dictionary
        logger                    (logger)        : logger object

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
        ref_estimate              (float)         : ref consumption
        day_level_estimate        (numpy.ndarray) : day wise ref consumption
    """

    hsm_posting_flag = (item_input_object.get('config').get('disagg_mode') == 'historical') or \
                       (item_input_object.get('config').get('disagg_mode') == 'incremental' and
                        len(item_input_object.get('item_input_params').get('day_input_data')) >= 70)

    ref_hsm_present_flag = item_input_object.get("item_input_params").get('ref_hsm') is not None and \
                item_input_object.get("item_input_params").get('ref_hsm').get('hybrid_ref') is not None

    logger.info("Calculating ref consumption using hybrid module")

    ref_estimate, features = get_ref_estimate(item_input_object, ref_config, logger)

    logger.debug("Calculated ref consumption")

    # Count number of all kind of refrigerator appliances provided by a user

    ref_app_count = get_ref_app_ids_counts(item_input_object, ref_config)

    logger.debug("Calculated ref app profile information | %s", ref_app_count)

    # Update ref estimate if appliance profile is given

    ref_estimate = update_ref_estimate(ref_estimate, ref_app_count, ref_config, logger)

    # Safety check rules for ref output

    ref_estimate = check_ref_estimate(ref_estimate, item_input_object, ref_config)

    ref_estimate = ref_estimate * factor

    # Safety checks based on pilot level config

    monthly_cons_max_limit = item_input_object.get("pilot_level_config").get('ref_config').get('bounds').get('max_cons')
    monthly_cons_min_limit = item_input_object.get("pilot_level_config").get('ref_config').get('bounds').get('min_cons')

    if monthly_cons_max_limit > 0 and ref_estimate > 0:
        ref_estimate = min(ref_estimate, monthly_cons_max_limit * Cgbdisagg.WH_IN_1_KWH / Cgbdisagg.DAYS_IN_MONTH)
    if monthly_cons_min_limit > 0 and ref_estimate > 0:
        ref_estimate = max(ref_estimate, monthly_cons_min_limit * Cgbdisagg.WH_IN_1_KWH / Cgbdisagg.DAYS_IN_MONTH)

    # Reduce ref estimation, incase AO component level is less

    input_data = copy.deepcopy(item_input_object.get("item_input_params").get("day_input_data"))
    base_cons = np.nanpercentile(input_data, 1, axis=1)
    base_cons = np.nanpercentile(base_cons, 5)
    base_cons = np.nan_to_num(base_cons)

    if len(input_data[0]) == Cgbdisagg.HRS_IN_DAY and base_cons > 20:
        ref_estimate = np.fmin(ref_estimate, base_cons * Cgbdisagg.HRS_IN_DAY)

    # Updated Ref HSM
    # If mode is MTD, fetch ref estimate using HSM data

    if hsm_posting_flag and \
            (item_output_object.get('created_hsm').get('ref') is None):
        item_output_object['created_hsm']['ref'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    ref_estimate = update_ref_estimate_based_on_hsm(ref_estimate, item_input_object, ref_hsm_present_flag)

    item_output_object.update({
        "ref_estimate": ref_estimate,
        "ref_features": features
    })

    created_hsm = dict({
        'hybrid_ref': int(ref_estimate),
        'disagg_ref_bool': 0
    })

    if hsm_posting_flag and \
            (item_output_object.get('created_hsm') is not None and
             item_output_object.get('created_hsm').get('ref') is not None) and \
            (item_output_object.get('created_hsm').get('ref').get('attributes') is not None):
        item_output_object['created_hsm']['ref']['attributes'].update(created_hsm)

    logger.debug("Post processing of ref consumption done")

    # Add seasonality in ref epoch level output

    ref_estimate, day_level_estimate = add_seasonality(ref_estimate, item_input_object)

    return ref_estimate, day_level_estimate, item_output_object
