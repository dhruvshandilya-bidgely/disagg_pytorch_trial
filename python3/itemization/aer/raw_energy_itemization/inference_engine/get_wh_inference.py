"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update wh consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
import pandas as pd

from numpy.random import RandomState

from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants

from python3.itemization.init_itemization_config import random_gen_config

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val_for_valid_boxes

from python3.itemization.aer.functions.hsm_utils import check_validity_of_hsm
from python3.itemization.init_itemization_config import init_itemization_params

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.aer.raw_energy_itemization.inference_engine.postprocess_detect_wh_signature import check_m2m
from python3.itemization.aer.raw_energy_itemization.inference_engine.postprocess_detect_wh_signature import check_min_cons
from python3.itemization.aer.raw_energy_itemization.inference_engine.postprocess_detect_wh_signature import check_seasonality_and_m2m
from python3.itemization.aer.raw_energy_itemization.inference_engine.postprocess_detect_wh_signature import postprocess_wh_output

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def get_wh_inference(app_index, item_input_object, item_output_object, date_list, logger_pass):

    """
    Update wh consumption ranges using inference rules

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        date_list                   (np.ndarray): list of target dates for heatmap dumping
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    swh_output = 0

    logger_base = logger_pass.get('logger_base').getChild('get_wh_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    pilot = item_input_object.get("config").get("pilot_id")
    app_profile = item_input_object.get("app_profile").get('wh')
    app_conf = item_output_object.get("inference_engine_dict").get("appliance_conf")[app_index, :, :]
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    max_cons = item_output_object.get("inference_engine_dict").get("appliance_max_values")[app_index, :, :]
    min_cons = item_output_object.get("inference_engine_dict").get("appliance_min_values")[app_index, :, :]
    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    config = get_inf_config(int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)).get("wh")

    item_input_object['item_input_params']['swh_hld'] = 0

    # checking if SWH information is present in disagg output
    # and fetching whether the user has SWH

    swh_tag_present_flag = \
        (item_output_object.get("created_hsm") is not None and
         item_output_object.get("created_hsm").get('wh') is not None and
         item_output_object.get("created_hsm").get('wh').get("attributes") is not None and
         item_output_object.get("created_hsm").get('wh').get("attributes").get('swh_hld') is not None)

    if swh_tag_present_flag:
        item_input_object['item_input_params']['swh_hld'] = item_output_object.get("created_hsm").get('wh').get("attributes").get('swh_hld')

    # initializing min/mid/max ts level WH estimates

    min_cons = np.fmax(0.5 * mid_cons, min_cons)
    max_cons = np.maximum(max_cons, disagg_cons)
    mid_cons = np.maximum(mid_cons, np.multiply(disagg_cons, app_conf))

    output_data = item_output_object.get("hybrid_input_data").get("output_data")
    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")

    # calculating consumption value that wont be the part of extra WH consumption to be added from hybrid (residual_wh)

    cons_excluded_from_wh = output_data[np.where(np.array(appliance_list) == 'pp')[0][0] + 1] + \
                            output_data[np.where(np.array(appliance_list) == 'wh')[0][0] + 1] + \
                            output_data[np.where(np.array(appliance_list) == 'ao')[0][0] + 1] + \
                            output_data[np.where(np.array(appliance_list) == 'ref')[0][0] + 1] + \
                            output_data[np.where(np.array(appliance_list) == 'ev')[0][0] + 1] + \
                            output_data[np.where(np.array(appliance_list) == 'li')[0][0] + 1]

    ##################### RULE 1 - Modify consumption based on app profile information  #############################

    flow_wh = 0

    # fetching WH app profile information

    app_profile, wh_prof_type = fetch_app_profile_info(pilot, app_profile, item_input_object, item_output_object, disagg_cons)

    if np.all(disagg_cons == 0) and (not app_profile):
        logger.debug("non timed WH is 0 and app profile is also 0")

        created_hsm = initialize_wh_hsm(disagg_cons)

        created_hsm['item_tou'] = np.sum(disagg_cons, axis=0) > 0

        item_output_object = update_hsm(item_input_object, item_output_object, created_hsm, disagg_cons)

        return item_input_object, item_output_object

    # if wh fuel type is non electric, WH estimates are made 0

    if wh_prof_type in config.get("non_electric_wh_types"):
        logger.debug("WH is not electric type, given type is | %s", wh_prof_type)
        min_cons = np.zeros(min_cons.shape)
        mid_cons = np.zeros(min_cons.shape)
        max_cons = np.zeros(min_cons.shape)
        created_hsm = initialize_wh_hsm(disagg_cons)

        created_hsm['item_tou'] = np.sum(disagg_cons, axis=0) > 0

        item_output_object = update_hsm(item_input_object, item_output_object, created_hsm, disagg_cons)

    # else if wh type is heatpump, WH estimates are reduced

    elif wh_prof_type in ["HEATPUMP"] and np.sum(disagg_cons)>0:
        logger.debug("WH is heat_pump type, thus reducing the consumption")
        min_cons = min_cons * config.get("heatpump_factor")
        mid_cons = mid_cons * config.get("heatpump_factor")

    # else, WH estimates are updated based on the expected seasonality and residual WH is added from disagg residual to hybrid WH estimates
    else:
        min_cons, mid_cons, max_cons, flow_wh = \
            update_wh_range(app_index, swh_output, logger, cons_excluded_from_wh, item_input_object, item_output_object,
                            min_cons, mid_cons, max_cons)

    # A max ts level consumption cap is applied on WH ts level estimates

    max_cons_thres = config.get('wh_max_cons') * (not flow_wh) + config.get('flow_max_cons') * (flow_wh)

    samples = int(mid_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)

    min_cons, mid_cons, max_cons = block_low_cons_wh(item_input_object, min_cons, mid_cons, max_cons, disagg_cons, samples)

    if np.sum(disagg_cons) == 0:
        samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')
        mid_cons = np.fmin(max_cons_thres/samples_per_hour, mid_cons)
        min_cons = np.fmin(max_cons_thres/samples_per_hour, min_cons)
        max_cons = np.fmin(max_cons_thres/samples_per_hour, max_cons)

    # low consumption WH estimates are blocked for tankless WH (based on app profile information)


    min_cons = np.minimum(min_cons, mid_cons)
    max_cons = np.maximum(max_cons, mid_cons)

    max_cons = np.minimum(max_cons, input_data)
    min_cons = np.minimum(min_cons, input_data)
    mid_cons = np.minimum(mid_cons, input_data)

    # maintaining a minimum consumption required at ts level

    min_disagg_frac_required = 0.7

    if np.sum(mid_cons) > 0:
        min_cons = np.maximum(disagg_cons*min_disagg_frac_required, min_cons)
        max_cons = np.maximum(disagg_cons*min_disagg_frac_required, max_cons)
        mid_cons = np.maximum(disagg_cons*min_disagg_frac_required, mid_cons)

    # Blocking WH on inactive days or summer days(incase of SWH users)

    min_cons, mid_cons, max_cons = \
        block_summer_cons_for_swh(vacation_days, config, disagg_cons, item_input_object, date_list, pilot, min_cons, mid_cons, max_cons)

    # udpating debug dictionary

    item_input_object, item_output_object = \
        update_wh_type(pilot, app_index, disagg_cons, mid_cons, config, item_input_object, item_output_object, logger)

    # updating WH HSM with additional hybrid attributes

    created_hsm = initialize_wh_hsm(disagg_cons)

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    created_hsm = prepare_wh_hsm(created_hsm, mid_cons, samples_per_hour)

    item_output_object = update_hsm(item_input_object, item_output_object, created_hsm, disagg_cons)

    # Updating the values in the original dictionary

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons)

    item_output_object["inference_engine_dict"]["output_data"][app_index, :, :] = disagg_cons

    t_end = datetime.now()

    logger.info("WH inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_input_object, item_output_object


def update_wh_type(pilot, app_index, disagg_cons, mid_cons, config, item_input_object, item_output_object, logger):

    """
    Update debug output with new detected WH type

    Parameters:
        pilot                     (int)       : pilot id of the user
        app_index                 (int)       : Index of app in the appliance list
        disagg_cons               (np.ndarray): wh disagg output
        mid_cons                  (np.ndarray): app mid cons
        config                    (dict)      : WH config
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
        logger                    (logger)      : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        item_output_object        (dict)      : Dict containing all hybrid outputs
    """

    conf_of_new_detected_boxes = 1.1

    if np.any(np.logical_and(disagg_cons == 0, mid_cons > 0)) > 0:
        item_output_object["inference_engine_dict"]["appliance_conf"][app_index][np.logical_and(disagg_cons == 0, mid_cons > 0)] = conf_of_new_detected_boxes

    swh_user_added = (pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS) and (np.sum(disagg_cons) == 0 and np.sum(mid_cons) > 0)
    storage_wh_user_added = np.sum(disagg_cons) == 0 and np.sum(mid_cons) > 0

    if swh_user_added:
        logger.info("type of WH added in postprocessing | seasonal")
        item_input_object['item_input_params']['wh_added_type'] = 'seasonal'
    elif storage_wh_user_added:
        logger.info("type of WH added in postprocessing | storage")
        item_input_object['item_input_params']['wh_added_type'] = 'thermostat'

    return item_input_object, item_output_object


def prepare_possible_others(input_data, disagg_cons, residual_cons, cons_excluded_from_wh, item_output_object):

    """
    Calculates ts level consumption that can be included in WH hybrid estimates

    Parameters:
        input_data                (np.ndarray): user input data
        disagg_cons               (np.ndarray): wh disagg output
        residual_cons             (np.ndarray): WH boxes detected in disagg residual
        cons_excluded_from_wh     (int)       : consumption that will not be included in WH hybrid estimates
        item_output_object        (dict)      : Dict containing all hybrid outputs

    Returns:
        possible_others           (np.ndarray): consumption that can be included in WH hybrid estimates
    """

    output_data = item_output_object.get("hybrid_input_data").get("output_data")
    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")

    possible_others = input_data - cons_excluded_from_wh - residual_cons

    # removing EV boxes from consumption that can be included in WH hybrid estimates

    if item_output_object.get("ev_residual") is not None:
        possible_others = possible_others - item_output_object.get("ev_residual")

    timed_signature = copy.deepcopy(item_output_object.get("timed_app_dict").get("timed_output"))

    # removing timed signature from consumption that can be included in WH hybrid estimates

    if timed_signature is not None:
        possible_others = possible_others - timed_signature

    possible_others = np.fmax(0, possible_others)

    # removing HVAC disagg from consumption that can be included in WH hybrid estimates

    if np.sum(disagg_cons) > 0:
        possible_others = possible_others - \
                          output_data[np.where(np.array(appliance_list) == 'cooling')[0][0] + 1] - \
                          output_data[np.where(np.array(appliance_list) == 'heating')[0][0] + 1]

    return possible_others


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


def prepare_wh_hsm(created_hsm, mid_cons, samples_per_hour):

    """
    initialize wh hybrid hsm with default values

    Parameters:
        disagg_cons        (np.ndarray))      : wh disagg output

    Returns:
        created_hsm        (dict)             : default wh hsm
    """

    max_wh_amp_thres = 6000
    wh_hsm_amp_perc_thres = 95

    if np.sum(mid_cons) > 0:

        # saving wh tou in hsm
        created_hsm['item_tou'] = np.sum(mid_cons, axis=0) > 0

        # saving wh amplitude in wh hsm
        created_hsm['item_amp'] = min(max_wh_amp_thres, np.percentile(mid_cons[mid_cons > 0], wh_hsm_amp_perc_thres) * samples_per_hour)

        # 1 means wh is detected
        created_hsm['item_hld'] = 1

        # type is thermostat
        created_hsm['item_type'] = 2
    else:
        created_hsm['item_tou'] = np.sum(mid_cons, axis=0) > 0

    return created_hsm


def fetch_app_profile_info(pilot, app_profile, item_input_object, item_output_object, disagg_cons):
    """
    fetch app profile info

    Parameters:
        pilot                       (int)       : pilot info
        app_profile                 (dict)      : wh app profile
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        disagg_cons                 (np.ndarray): disagg output

    Returns:
        app_profile                 (int)       : wh app profile
        type                        (str)       : type of wh in app profile
    """

    attributes = ""
    app_prof_wh_type = "ELECTRIC"

    config = get_inf_config(int(disagg_cons.shape[1] / 24)).get("wh")

    if app_profile is not None:
        app_prof_wh_type = app_profile.get("type", 'ELECTRIC')
        attributes = app_profile.get("attributes", '')
        app_profile = app_profile.get("number", 0)
    else:
        app_profile = 0

    if (attributes is not None) and ('heatpump' in attributes) and app_prof_wh_type not in config.get("non_electric_wh_types"):
        app_prof_wh_type = "HEATPUMP"

    # adding wh if user belong to swh pilot

    if pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS and item_output_object.get("hvac_dict").get("wh").sum() and \
            (item_input_object.get("app_profile").get('wh') is None):
        app_profile = 1

    # not adding wh if user belongs to high wh coverage pilot

    if (item_output_object.get('box_dict').get('hybrid_wh') == 1) and (item_input_object.get("app_profile").get('wh') is None):
        app_profile = 1

    # not adding wh if type is others
    if app_prof_wh_type in config.get("other_type_list"):
        app_prof_wh_type = 'other'

        if np.sum(disagg_cons) == 0:
            app_profile = 0

    return app_profile, app_prof_wh_type


def pick_wh_type_boxes_detected_in_disagg_residual(item_output_object, residual):

    """
    Calculate non timed WH consumption in the leftover residual data

    Parameters:
        item_output_object        (dict)        : Dict containing all hybrid outputs
        residual                  (np.ndarray)  : TS level residual data
    Returns:
        item_output_object        (dict)        : Dict containing all hybrid outputs
        valid_idx                 (dict)        : updated Dict containing all hybrid outputs
    """

    box_seq = item_output_object.get("box_dict").get("box_seq_wh")

    # Calculating WH boxes amplitude and length range

    box_seq = box_seq.astype(int)

    valid_idx = np.zeros(residual.size)

    seq_config = init_itemization_params().get("seq_config")

    # Filtering WH boxes in inactive hours

    item_output_object["box_dict"]["box_seq_wh"] = box_seq

    boxes_score = item_output_object.get("box_score_wh")
    wh_boxes = boxes_score[:, 0] == np.max(boxes_score, axis=1)
    wh_boxes[boxes_score[:, 0] == 0] = 0

    for i in range(len(wh_boxes)):
        if wh_boxes[i]:
            valid_idx[box_seq[i, seq_config.get("start")]: box_seq[i, seq_config.get("end")] + 1] = 1

    valid_idx = np.reshape(valid_idx, residual.shape)

    return item_output_object, valid_idx


def update_hsm(item_input_object, item_output_object, created_hsm, disagg_cons):

    """
    Update wh hsm with hybrid wh attributes

    Parameters:
        item_input_object           (dict)             : Dict containing all hybrid inputs
        item_output_object          (dict)             : Dict containing all hybrid outputs
        created_hsm                 (dict)             : default wh hsm
        disagg_cons                 (np.ndarray)       : wh disagg output

    Returns:
        item_output_object          (dict)             : Dict containing all hybrid outputs
    """

    # determines whether we need to update wh hsm with new attributes

    post_hsm_flag = item_input_object.get('item_input_params').get('post_hsm_flag')

    if post_hsm_flag and (item_output_object.get('created_hsm').get('wh') is None):
        item_output_object['created_hsm']['wh'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    if post_hsm_flag and (item_output_object.get('created_hsm').get('wh') is not None) and \
            (item_output_object.get('created_hsm').get('wh').get('attributes') is not None):
        item_output_object['created_hsm']['wh']['attributes'].update(created_hsm)

    return item_output_object


def update_wh_range(app_index, swh_output, logger, cons_excluded_from_wh, item_input_object, item_output_object, min_cons, mid_cons, max_cons):

    """
    In this function , WH estimates are updated based on the expected seasonality
    and residual WH is added from disagg residual to hybrid WH estimates

    Parameters:
        app_index                   (int)              : Index of app in the appliance list
        swh_output                  (np.ndarray)       : wh disagg output
        logger                      (logger)           : logger object
        cons_excluded_from_wh       (int)              : consumption that will not be included in WH hybrid estimates
        item_input_object           (dict)             : Dict containing all hybrid inputs
        item_output_object          (dict)             : Dict containing all hybrid outputs
        min_cons                    (np.ndarray)       : app min cons
        mid_cons                    (np.ndarray)       : app mid cons
        max_cons                    (np.ndarray)       : app max cons
    Returns:
        min_cons                    (np.ndarray)       : app min cons
        mid_cons                    (np.ndarray)       : app mid cons
        max_cons                    (np.ndarray)       : app max cons
    """

    pilot = item_input_object.get("config").get("pilot_id")
    residual = copy.deepcopy(item_output_object.get("inference_engine_dict").get("residual_data"))
    app_conf = item_output_object.get("inference_engine_dict").get("appliance_conf")[app_index, :, :]
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    if np.all(app_conf == 0):
        app_conf = np.ones(app_conf.shape)

    config = get_inf_config(int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)).get("wh")

    ############# RULE 1 - Fetching the boxes similar to wh capacity in the disagg residual ############

    if np.sum(disagg_cons) == 0:
        residual = np.fmax(0, copy.deepcopy(input_data - cons_excluded_from_wh))

    residual_copy = copy.deepcopy(residual)
    residual_copy = np.fmax(residual_copy, 0)
    item_output_object, valid_idx = pick_wh_type_boxes_detected_in_disagg_residual(item_output_object, residual_copy)

    item_output_object["inference_engine_dict"]["appliance_conf"][app_index][valid_idx > 0] = 1.1

    # removing baseload consumption from WH boxes

    residual_cons = copy.deepcopy(residual_copy) - np.min(residual_copy, axis=1)[:, None]
    residual_cons[np.logical_not(valid_idx)] = 0

    samples_per_hour = int(len(disagg_cons[0]) / Cgbdisagg.HRS_IN_DAY)

    # capping the ts level consumption of WH boxes

    if np.sum(residual_cons > 0):
        max_cons_val = np.percentile(residual_cons[residual_cons > 0], 90)
        min_cons_val = max(500 / samples_per_hour, np.percentile(residual_cons[residual_cons > 0], 10))

        residual_cons = np.fmin(residual_cons, max_cons_val)
        residual_cons[residual_cons < min_cons_val] = 0

    residual_cons = np.minimum(residual_cons * 1.5, np.fmax(0, residual))

    samples_per_hour = int(len(disagg_cons[0]) / Cgbdisagg.HRS_IN_DAY)

    ############# RULE 2 - Fetching the WH type seasonal signature from seasonal signature detection output  ############

    residual_wh = copy.deepcopy(item_output_object.get("hvac_dict").get("wh"))

    min_cons_thres = config.get('swh_amp_thres') * (pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS) + \
                     config.get('wh_amp_thres') * (pilot not in PilotConstants.SEASONAL_WH_ENABLED_PILOTS)

    # capping the ts level consumption of WH boxes

    residual_wh[residual_wh < min_cons_thres / samples_per_hour] = 0
    residual_wh[residual_wh > config.get('max_seasonal_cons') / samples_per_hour] = 0

    min_cons = np.multiply(min_cons, app_conf / np.max(app_conf))

    flow_wh = 0

    # limiting the number of WH boxes in a day

    residual_cons = check_freq(residual_cons)

    # preparing the amount of consumption that can be added in WH estimates

    possible_others = \
        prepare_possible_others(input_data, disagg_cons, residual_cons, cons_excluded_from_wh, item_output_object)

    if np.any(residual_wh > 0):
        flow_wh = (np.percentile(residual_wh[residual_wh > 0], 75) >
                   config.get('flow_thres') / samples_per_hour) * (np.sum(residual_wh)) + 0 * (not np.sum(residual_wh))

    ################################### RULE 3 - Postprocessing of detected WH signature ##############################

    # adding seasonal WH into WH estimates

    min_cons, mid_cons, max_cons, residual_cons = \
        update_wh_output_with_seasonal_sig(min_cons, mid_cons, max_cons, disagg_cons, residual_cons, pilot,
                                           residual_wh, item_input_object, item_output_object)

    # Removing WH residual during inactive hours

    non_wh_hours = config.get('non_wh_hours')

    if item_input_object.get('item_input_params').get('swh_hld') > 0 or pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS:
        non_wh_hours = config.get('non_swh_hours')

    residual_cons[:, non_wh_hours] = 0
    residual_cons = np.fmax(0, residual_cons)

    # maintain seasonality and m2m consistency in box type WH residual that will be added into WH consumption

    residual_cons = check_seasonality_and_m2m(item_output_object.get("wh_pot")[:, 0], disagg_cons,
                                              item_input_object, item_output_object, residual_cons, possible_others,
                                              non_wh_hours)

    residual_cons = check_m2m(item_output_object.get("wh_pot")[:, 0], disagg_cons, item_input_object,
                              item_output_object, residual_cons, possible_others, non_wh_hours)

    if item_input_object.get('item_input_params').get('swh_hld') > 0:
        mid_cons = check_m2m(item_output_object.get("wh_pot")[:, 0], np.zeros_like(disagg_cons), item_input_object,
                             item_output_object, mid_cons, possible_others, non_wh_hours)

    min_cons, mid_cons, max_cons, residual_cons = \
        postprocess_wh_output(disagg_cons, item_output_object, item_input_object, residual, residual_cons, residual_wh,
                              [min_cons, mid_cons, max_cons], app_index, swh_output, flow_wh)

    # updating WH residual based on previous run information

    mid_cons = prepare_wh_cons_based_on_hsm(possible_others, config, disagg_cons, item_input_object, item_output_object,
                                            pilot, mid_cons, non_wh_hours)

    return min_cons, mid_cons, max_cons, flow_wh


def prepare_wh_cons_based_on_hsm(possible_others, config, disagg_cons, item_input_object, item_output_object, pilot, mid_cons, non_wh_hours):

    """
    In this function , WH estimates are updated based on the expected seasonality
    and residual WH is added from disagg residual to hybrid WH estimates

    Parameters:
        possible_others             (np.ndarray)       : disagg residual that can be added to wh
        config                      (dict)             : wh config
        disagg_cons                 (np.ndarray)       : WH disagg output
        item_input_object           (dict)             : Dict containing all hybrid inputs
        item_output_object          (dict)             : Dict containing all hybrid outputs
        pilot                       (int)              : pilot id
        mid_cons                    (np.ndarray)       : app mid cons
        non_wh_hours                (np.ndarray)       : inactive hours of the user
    Returns:
        mid_cons                    (np.ndarray)       : app mid cons
    """

    valid_wh_hsm = item_input_object.get("item_input_params").get('valid_wh_hsm')

    valid_hsm_flag = check_validity_of_hsm(valid_wh_hsm, item_input_object.get("item_input_params").get('wh_hsm'),
                                           'wh_backup_cons')

    # killing itemization wh output in mtd, if HSM wh output is 0
    # or maintaining a minimum consumption in current disagg run based on previous run info

    if valid_hsm_flag:

        hsm_wh = item_input_object.get("item_input_params").get('wh_hsm').get('wh_backup_cons')

        if hsm_wh is not None and isinstance(hsm_wh, list):
            hsm_wh = hsm_wh[0]

        hsm_wh = hsm_wh * 0.5

        min_cons_not_required = pilot not in PilotConstants.SEASONAL_WH_ENABLED_PILOTS

        if min_cons_not_required:
            mid_cons = check_min_cons(item_output_object.get("wh_pot")[:, 0], disagg_cons, item_input_object,
                                      item_output_object, mid_cons, possible_others, non_wh_hours, hsm_wh)

    return mid_cons


def block_summer_cons_for_swh(vacation_days, config, disagg_cons, item_input_object, date_list, pilot, min_cons, mid_cons, max_cons):


    """
    In this function , WH estimates are updated based on the expected seasonality

    Parameters:
        vacation_days               (np.ndarray)       : vacation output
        config                      (dict)             : wh config
        disagg_cons                 (np.ndarray)       : WH disagg output
        item_input_object           (dict)             : Dict containing all hybrid inputs
        item_output_object          (dict)             : Dict containing all hybrid outputs
        pilot                       (int)              : pilot id
        min_cons                    (np.ndarray)       : app min cons
        mid_cons                    (np.ndarray)       : app mid cons
        max_cons                    (np.ndarray)       : app max cons
    Returns:
        min_cons                    (np.ndarray)       : app min cons
        mid_cons                    (np.ndarray)       : app mid cons
        max_cons                    (np.ndarray)       : app max cons
    """

    # blocking WH during vacation days

    valid_days = np.logical_and(vacation_days, np.sum(disagg_cons, axis=1) == 0)

    mid_cons[valid_days] = 0
    max_cons[valid_days] = 0
    min_cons[valid_days] = 0

    # removing WH during summer months for SWH type user

    if (item_input_object.get('item_input_params').get('swh_hld') > 0) and (np.sum(disagg_cons) > 0) and \
            (pilot not in config.get("all_year_wh")):
        month_list = pd.DatetimeIndex(date_list).month.values

        non_wh_months = config.get('non_swh_months')

        min_cons[np.isin(month_list, non_wh_months)] = 0
        mid_cons[np.isin(month_list, non_wh_months)] = 0
        max_cons[np.isin(month_list, non_wh_months)] = 0

    return min_cons, mid_cons, max_cons


def update_wh_output_with_seasonal_sig(min_cons, mid_cons, max_cons, disagg_cons, residual_cons, pilot, residual_wh,
                                       item_input_object, item_output_object):

    """
    Add seasonal wh into wh output

    Parameters:
        min_cons                    (np.ndarray)       : app min cons
        mid_cons                    (np.ndarray)       : app mid cons
        max_cons                    (np.ndarray)       : app max cons
        disagg_cons                 (np.ndarray)       : app disagg cons
        residual_cons               (np.ndarray)       : residual box type wh
        pilot                       (int)              : pilot id
        residual_wh                 (np.ndarray)       : residual seasonal wh
        item_input_object           (dict)             : Dict containing all hybrid inputs
        item_output_object          (dict)             : Dict containing all hybrid outputs

    Returns:
        min_cons                    (np.ndarray)       : app min cons
        mid_cons                    (np.ndarray)       : app mid cons
        max_cons                    (np.ndarray)       : app max cons
        residual_cons               (np.ndarray)       : residual box type wh
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_label = seq_config.SEQ_LABEL

    samples_per_hour = int(mid_cons.shape[1]/Cgbdisagg.HRS_IN_DAY)

    config = get_inf_config(samples_per_hour).get("wh")

    swh_pilots = PilotConstants.SEASONAL_WH_ENABLED_PILOTS

    residual_wh = np.fmax(0, np.minimum(residual_wh, item_output_object.get("hybrid_input_data").get("true_disagg_res")))

    # removing SWH WH signatures if user doesnot belong to SWH pilots

    if pilot in config.get("japan_pilots"):
        residual_wh[:, :] = 0

    if np.sum(disagg_cons) > 0 and (pilot not in swh_pilots):
        residual_wh[:, :] = 0

    # remove seasonal component from late night hours and days hours

    swh_start_hour = config.get('swh_start_hour')

    if item_input_object.get('item_input_params').get('swh_hld') > 0 or pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS:
        residual_wh[:, config.get('swh_end_hour'):] = 0
        residual_wh[:, :swh_start_hour] = 0
    else:
        residual_wh[:, config.get('wh_end_hour'):] = 0
        residual_wh[:, :swh_start_hour] = 0

    # adding SWH signature into WH estimates

    mid_cons = mid_cons + residual_wh
    max_cons = max_cons + residual_wh
    min_cons = min_cons + residual_wh

    season_potential = item_output_object.get("wh_pot")[:, 0]

    # determing the days where wh fat pulse can be added based on wh potential informationn
    # These potential days can extend into summer months based on consumtion behavior of the user

    if (pilot not in config.get("all_year_wh")) and (np.sum(season_potential) > 0):

        wh_pot_buffer_days = config.get('wh_pot_buffer_days')

        high_wh_potential_days = (season_potential > config.get('pot_thres')).astype(int)

        # extending the wh usage days to add some buffer days around high wh potential days chunk

        seq = find_seq(high_wh_potential_days, np.zeros_like(high_wh_potential_days), np.zeros_like(high_wh_potential_days))

        for i in range(len(seq)):
            if seq[i, seq_label]:
                high_wh_potential_days[max(0, seq[i, seq_start]-wh_pot_buffer_days): seq[i, seq_start]] = 1
                high_wh_potential_days[seq[i, seq_end]: seq[i, seq_end]+wh_pot_buffer_days] = 1

        residual_cons[np.logical_not(high_wh_potential_days)] = \
            np.minimum(residual_cons[np.logical_not(high_wh_potential_days)],
                       item_output_object.get("hybrid_input_data").get("true_disagg_res")[np.logical_not(high_wh_potential_days)])

    return min_cons, mid_cons, max_cons, residual_cons


def block_low_cons_wh(item_input_object, min_cons, mid_cons, max_cons, disagg_cons, samples):

    """
    Modify appliance mid/min/max ranges in cases where low cons wh has been added for tankless WH

    Parameters:
        item_input_object         (dict)      : Dict containing all hybrid inputs
        min_cons                  (np.ndarray)       : app min cons
        mid_cons                  (np.ndarray)       : app mid cons
        max_cons                  (np.ndarray)       : app max cons
        disagg_cons               (np.ndarray)       : disagg wh
        samples                   (int)              : samples in an hour

    Returns:
        min_cons                  (np.ndarray)       : app min cons
        mid_cons                  (np.ndarray)       : app mid cons
        max_cons                  (np.ndarray)       : app max cons
    """

    # remove low cons points for tankless wh (given in app profile)

    config = get_inf_config(int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)).get("wh")

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    if item_input_object.get('item_input_params').get('tankless_wh') > 0 and np.sum(disagg_cons) == 0:
        min_cons_thres_for_tankless_wh = config.get('tankless_min_ts_cons')
        mid_cons[mid_cons < min_cons_thres_for_tankless_wh/samples_per_hour] = 0
        min_cons[mid_cons < min_cons_thres_for_tankless_wh/samples_per_hour] = 0
        max_cons[mid_cons < min_cons_thres_for_tankless_wh/samples_per_hour] = 0

    return min_cons, mid_cons, max_cons


def check_freq(residual_cons):

    """
    In this function , WH estimates are updated based on the maximum number of fat pulses in a day

    Parameters:
        residual_cons                   (np.ndarray)       : Additional box type signature to be added to WH estimates
    Returns:
        residual_cons                   (np.ndarray)       : Additional box type signature to be added to WH estimates
    """

    max_freq = 3

    samples = residual_cons.shape[1]

    residual_cons2 = copy.deepcopy(residual_cons)
    residual_cons2[:, 1*samples:5*samples+1] = 0

    seed = RandomState(random_gen_config.seed_value)

    for i in range(len(residual_cons)):

        if np.sum(residual_cons2[i]) <= 0:
            continue

        seq = residual_cons2[i] > 0

        seq = find_seq(seq, np.zeros_like(seq), np.zeros_like(seq))

        seq = seq[seq[:, 0] > 0]

        if np.sum(seq[:, 0]) > max_freq:

            remove_boxes = seed.choice(np.arange(len(seq)), np.sum(seq[:, 0])-max_freq, replace=False).astype(int)

            seq[remove_boxes, 0] = 0

            for k in range(len(seq)):
                if seq[k, 0] == 0:
                    residual_cons[i][seq[k, 1]: seq[k, 2] + 1] = 0

    return residual_cons
