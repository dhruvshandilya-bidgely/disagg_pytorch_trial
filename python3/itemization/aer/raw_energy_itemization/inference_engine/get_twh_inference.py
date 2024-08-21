"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update twh consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def get_twh_inference(app_index, item_input_object, item_output_object, logger_pass):

    """
    Update twh consumption ranges using inference rules
    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on
    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_twh_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    app_pot = item_output_object.get("inference_engine_dict").get("appliance_pot")[app_index]
    app_conf = item_output_object.get("inference_engine_dict").get("appliance_conf")[app_index]
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    max_cons = item_output_object.get("inference_engine_dict").get("appliance_max_values")[app_index, :, :]
    min_cons = item_output_object.get("inference_engine_dict").get("appliance_min_values")[app_index, :, :]

    wh_app_profile = item_input_object.get("app_profile").get('wh')

    # output of timed wh signature found in residual data
    timed_output = item_output_object.get("timed_app_dict").get("twh")

    residual = copy.deepcopy(item_output_object.get("inference_engine_dict").get("residual_data"))
    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    max_cons = np.maximum(max_cons, disagg_cons)
    mid_cons = np.maximum(mid_cons, np.multiply(disagg_cons, app_conf + 0.1))

    max_cons = np.minimum(max_cons, disagg_cons)
    mid_cons = np.minimum(mid_cons, disagg_cons)
    min_cons = np.minimum(min_cons, disagg_cons)

    pilot = item_input_object.get("config").get("pilot_id")

    no_twh_sig_present_in_disagg_and_hybrid = np.all(disagg_cons == 0) and not np.sum(timed_output)

    if no_twh_sig_present_in_disagg_and_hybrid:
        return item_output_object

    config = get_inf_config().get("wh")

    ###################### RULE 1 - Updating TWH output baed on app profle information  ###########################

    add_twh, wh_feul_type = fetch_app_profile_info(pilot, item_output_object, wh_app_profile, disagg_cons, logger)

    min_days_required_for_hsm_posting = config.get('min_days_required_for_hsm_posting')

    # remove wh if app profile is 0

    zero_twh_flag = (np.all(disagg_cons == 0) and (not add_twh)) or (np.all(disagg_cons == 0) and (wh_feul_type in config.get("non_electric_wh_types")))

    if zero_twh_flag:
        logger.info("non timed WH is 0 and app profile is also 0")

        created_hsm = initialize_wh_hsm(disagg_cons)

        created_hsm['item_tou'] = np.sum(disagg_cons > 0, axis=0)  > min_days_required_for_hsm_posting

        item_output_object = update_hsm(item_input_object, item_output_object, mid_cons, disagg_cons, config)

        return item_output_object

    # remove wh if app profile is gas

    if wh_feul_type in config.get("non_electric_wh_types"):
        logger.info("WH is not electric type")
        created_hsm = initialize_wh_hsm(disagg_cons)

        created_hsm['item_tou'] = np.sum(disagg_cons > 0, axis=0)  > min_days_required_for_hsm_posting

        item_output_object = update_hsm(item_input_object, item_output_object, mid_cons, disagg_cons, config)

        min_cons = np.zeros(min_cons.shape)
        mid_cons = np.zeros(min_cons.shape)
        max_cons = np.zeros(min_cons.shape)

    # decrease wh if app profile is heatpump type

    elif wh_feul_type in ["HEATPUMP"] and np.sum(disagg_cons) > 0:
        logger.debug("WH is heat_pump type, thus reducing the consumption")
        min_cons = min_cons * config.get("heatpump_factor")
        mid_cons = mid_cons * config.get("heatpump_factor")

    else:

        ########################### RULE 2 - Decreasing consumtion in overestimation scenario #############################

        min_cons, max_cons = update_range_for_overest_cases(app_pot, app_conf, disagg_cons, min_cons, max_cons, input_data, residual, config)

        ################# RULE 3 - Updating ranges with calculated timed app leftover in residual data  ###############

        mid_cons = mid_cons + timed_output
        max_cons = max_cons + timed_output
        min_cons = min_cons + timed_output

        season = item_output_object.get("season")

        opp_seasonality_trend = \
            np.all(disagg_cons == 0) and len(input_data) > config.get('min_days_to_check_seasonality') and\
            np.any(season >= 0) and np.any(season < 0) and \
            np.sum(mid_cons[season >= 0]) > 0 and np.sum(mid_cons[season < 0]) == 0

        if opp_seasonality_trend:
            logger.debug("Opposite seasonality in wh consumption")
            min_cons = np.zeros(min_cons.shape)
            mid_cons = np.zeros(min_cons.shape)
            max_cons = np.zeros(min_cons.shape)

    min_cons = np.minimum(min_cons, mid_cons)
    max_cons = np.maximum(max_cons, mid_cons)

    max_cons = np.minimum(max_cons, input_data)
    min_cons = np.minimum(min_cons, input_data)
    mid_cons = np.minimum(mid_cons, input_data)

    if np.sum(disagg_cons) == 0 and np.sum(mid_cons) > 0:
        logger.info("type of WH added in postprocessing | timed")
        item_input_object['item_input_params']['wh_added_type'] = 'timed'

    # prepare hybrid hsm for non-zero wh cases

    item_output_object = update_hsm(item_input_object, item_output_object, mid_cons, disagg_cons, config)

    # Updating the values in the original dictionary

    twh_disagg_ts = np.logical_and(disagg_cons > 0, timed_output == 0)

    if np.any(twh_disagg_ts):
        mid_cons[twh_disagg_ts] = np.minimum(disagg_cons[twh_disagg_ts], mid_cons[twh_disagg_ts])
        max_cons[twh_disagg_ts] = np.maximum(disagg_cons[twh_disagg_ts], max_cons[twh_disagg_ts])

    item_output_object["inference_engine_dict"]["appliance_conf"][app_index][timed_output > 0] = 1

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons)

    item_output_object["inference_engine_dict"]["output_data"][app_index, :, :] = disagg_cons

    t_end = datetime.now()

    logger.debug("Twh inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def update_hsm(item_input_object, item_output_object, mid_cons, disagg_cons, config):

    """
    Update wh hsm with hybrid wh attributes
    Parameters:
        item_input_object           (dict)             : Dict containing all hybrid inputs
        item_output_object          (dict)             : Dict containing all hybrid outputs
        mid_cons                    (np.ndarray)       : wh itemization output
        disagg_cons                 (np.ndarray)       : wh disagg output
        config                      (dict)             : WH config
    Returns:
        item_output_object          (dict)             : Dict containing all hybrid outputs
    """

    disagg_confidence = 0

    min_days_required_for_hsm_posting = config.get('min_days_required_for_hsm_posting')

    valid_twh_conf_present_flag = item_input_object.get("disagg_special_outputs") is not None and \
                                 item_input_object.get("disagg_special_outputs").get("timed_wh_confidence") is not None

    if valid_twh_conf_present_flag:
        disagg_confidence = copy.deepcopy(item_input_object.get("disagg_special_outputs").get("timed_wh_confidence"))

    created_hsm = initialize_wh_hsm(disagg_cons)

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    # adding additional attributes into WH HSM based on hybrid TWH output

    if np.sum(mid_cons) > 0:
        created_hsm['item_tou'] = np.sum(mid_cons > 0, axis=0) > min_days_required_for_hsm_posting
        created_hsm['item_amp'] = np.percentile(mid_cons[mid_cons > 0], 75) * samples_per_hour
        created_hsm['item_hld'] = 1
        created_hsm['item_type'] = 1
        created_hsm['item_conf'] = disagg_confidence
    else:
        created_hsm['item_tou'] = np.sum(mid_cons > 0, axis=0) > min_days_required_for_hsm_posting

    post_hsm_flag = item_input_object.get('item_input_params').get('post_hsm_flag')

    if post_hsm_flag and (item_output_object.get('created_hsm').get('wh') is None):
        item_output_object['created_hsm']['wh'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    wh_hsm_key_is_present = post_hsm_flag and (item_output_object.get('created_hsm').get('wh') is not None) and \
                            (item_output_object.get('created_hsm').get('wh').get('attributes') is not None)

    if wh_hsm_key_is_present:
        item_output_object['created_hsm']['wh']['attributes'].update(created_hsm)

    return item_output_object


def update_range_for_overest_cases(app_pot, app_conf, disagg_cons, min_cons, max_cons, input_data, residual, config):

    """
    adjust TWH ts level min/max consumption incase of overestimation of TWH
    which is causing overshoot of total disagg compared to input data

    Parameters:
        app_pot                     (np.ndarray)    : TS level PP usage potential
        app_conf                    (np.ndarray)    : TS level PP confidence
        disagg_cons                 (np.ndarray)    : pp disagg output
        min_cons                    (np.ndarray)    : ts level min PP consumption
        max_cons                    (np.ndarray)    : ts level max PP consumption
        input_data                  (np.ndarray)    : user input data
        residual                    (np.ndarray)    : disagg residual data
        config                      (dict)          : twh config

    Returns:
        min_cons                    (np.ndarray)    : ts level min PP consumption
        max_cons                    (np.ndarray)    : ts level max PP consumption

    """

    perc_cap_for_app_pot = config.get('perc_cap_for_app_pot')

    if not np.all(disagg_cons == 0):
        if np.any(max_cons > 0):
            max_cons = np.percentile(max_cons[max_cons > 0], perc_cap_for_app_pot) * app_pot

        if np.any(min_cons > 0):
            min_cons = np.percentile(min_cons[min_cons > 0], perc_cap_for_app_pot) * app_pot

        # updating min WH estimates for timestamps where overestimation is happening and TWH is being used at thoes points

        overest_ts = residual < 0.02 * input_data

        modified_cons = copy.deepcopy(min_cons)

        modified_cons[overest_ts] = np.multiply(min_cons[overest_ts], app_conf[overest_ts])

        if np.any(min_cons > 0):
            min_cons = app_pot * np.median(modified_cons[modified_cons > 0])

    return min_cons, max_cons


def initialize_wh_hsm(disagg_cons):

    """
    initialize wh hybrid hsm with default values
    Parameters:
        disagg_cons        (np.ndarray))      : wh disagg output
    Returns:
        created_hsm        (dict)             : default wh hsm
    """

    created_hsm = dict({
        'item_tou': np.zeros(len(disagg_cons[0])),
        'item_hld': 0,
        'item_type': 0,
        'item_amp': 0
    })

    return created_hsm


def fetch_app_profile_info(pilot, item_output_object, wh_app_profile, disagg_cons, logger):

    """
    fetch app profile info
    Parameters:
        pilot                       (int)       : pilot info
        item_output_object          (dict)      : Dict containing all hybrid outputs
        wh_app_profile              (dict)      : app profile info
        disagg_cons                 (np.ndarray): disagg output
        logger                      (logger)    : logger object
    Returns:
        add_twh_bool                (int)       : true if twh can be added
        wh_type                     (str)       : wh feul type given in app profile
    """

    wh_type = "ELECTRIC"

    config = get_inf_config().get("wh")

    if wh_app_profile is not None:
        wh_type = wh_app_profile.get("type", 'ELECTRIC')
        attributes = wh_app_profile.get("attributes", '')
        add_twh_bool = wh_app_profile.get("number", 0)

        if (attributes is not None) and ('heatpump' in attributes) and wh_type not in config.get("non_electric_wh_types"):
            wh_type = "HEATPUMP"
    else:
        add_twh_bool = 0

        logger.info('WH app profile info not present |')

    # Add TWH output if user belongs to TWH pilots

    user_is_from_twh_pilots = (pilot in PilotConstants.TIMED_WH_PILOTS) and \
                              item_output_object.get("timed_app_dict").get("twh").sum() and (wh_app_profile is None)

    if user_is_from_twh_pilots:
        add_twh_bool = 1
        logger.info('adding twh since the pilot belongs to twh pilot list |')

    if wh_type in config.get("other_type_list"):
        wh_type = 'other'

        if np.sum(disagg_cons) == 0:
            add_twh_bool = 0

            logger.info('Not adding twh since type is others |')

    return add_twh_bool, wh_type
