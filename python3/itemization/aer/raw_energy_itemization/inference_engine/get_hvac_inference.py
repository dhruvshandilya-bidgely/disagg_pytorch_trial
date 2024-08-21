"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update cooling and heating consumption ranges using inference rules
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

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def get_cool_inference(app_index, item_input_object, item_output_object, logger_pass):

    """
    Update cooling consumption ranges using inference rules

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object          (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_cooling_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    season = item_output_object.get("season")
    pilot = item_input_object.get("config").get("pilot_id")

    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    app_conf = item_output_object.get("inference_engine_dict").get("appliance_conf")[app_index, :, :]
    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    max_cons = item_output_object.get("inference_engine_dict").get("appliance_max_values")[app_index, :, :]
    min_cons = item_output_object.get("inference_engine_dict").get("appliance_min_values")[app_index, :, :]

    cool_ao = item_input_object.get("item_input_params").get("ao_cool")

    config = get_inf_config().get("cool")

    if cool_ao is None:
        cool_ao = np.zeros(disagg_cons.shape)

    conf_thres = config.get('conf_thres')
    min_disagg_frac_required = config.get('min_disagg_frac_required')
    heavy_hvac_pilot = config.get('heavy_hvac_pilot')

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    # adding cooling component found in residual data

    max_cons = np.maximum(max_cons, disagg_cons)
    mid_cons[app_conf > conf_thres] = np.maximum(mid_cons, disagg_cons)[app_conf > conf_thres]
    min_cons[app_conf > conf_thres] = np.maximum(min_cons, disagg_cons)[app_conf > conf_thres]
    min_cons = np.minimum(min_cons, mid_cons)

    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    # removing baseload consumption before addition of residual cooling

    res_data = np.fmax(0, item_output_object.get("hybrid_input_data").get("true_disagg_res"))
    base_res = np.maximum(np.percentile(res_data, 15, axis=0),
                          np.percentile(input_data, 10, axis=0))
    base_res_2d = np.zeros_like(res_data)
    base_res_2d[:, :] = base_res[None, :]
    base_res_2d = np.maximum(base_res_2d, np.percentile(res_data, 3, axis=1)[:, None])

    diff = np.fmax(0, min_cons - disagg_cons)
    diff = np.minimum(diff, np.fmax(0, res_data - base_res_2d))
    min_cons = min_cons + diff

    diff = np.fmax(0, mid_cons - disagg_cons)
    diff = np.minimum(diff, np.fmax(0, res_data - base_res_2d))
    mid_cons = mid_cons + diff

    diff = np.fmax(0, max_cons - disagg_cons)
    diff = np.minimum(diff, np.fmax(0, res_data - base_res_2d))
    max_cons = max_cons + diff

    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Removing always on component from hvac, before calculating min,mid,max ts level consumption
    # This AO component will be added later

    min_cons = np.nan_to_num(min_cons)

    if item_input_object.get("item_input_params").get("ao_cool") is not None:
        min_cons = min_cons + item_input_object.get("item_input_params").get("ao_cool")
        mid_cons = mid_cons + item_input_object.get("item_input_params").get("ao_cool")
        max_cons = max_cons + item_input_object.get("item_input_params").get("ao_cool")

    min_cons = np.minimum(min_cons, mid_cons)
    max_cons = np.maximum(max_cons, mid_cons)

    max_cons = np.minimum(max_cons, input_data)
    min_cons = np.minimum(min_cons, input_data)

    mid_cons = copy.deepcopy(disagg_cons)

    ########## RULE 1 - Adding extra cooling signature detected from disagg residual into cooling output  ####################

    residual_cool = item_output_object.get("hvac_dict").get("cooling")

    # before adding the extra cooling signature, it is made sure that the added cooling in not inconsistent
    # For which, we remove cooling segments if detected for less days of chunks

    extra_cooling_seq = find_seq(residual_cool.sum(axis=1) > 0, np.zeros_like(residual_cool.sum(axis=1)),
                                 np.zeros_like(residual_cool.sum(axis=1)), overnight=0)

    cooling_days = np.sum(disagg_cons, axis=1) > 0

    cooling_days_seq = find_seq(cooling_days, np.zeros_like(cooling_days), np.zeros_like(cooling_days), overnight=0).astype(int)

    for i in range(len(cooling_days_seq)):

        if cooling_days_seq[i, seq_label] and (pilot in PilotConstants.INDIAN_PILOTS):
            cooling_days[cooling_days_seq[i, seq_end]:(cooling_days_seq[i, seq_end] + Cgbdisagg.DAYS_IN_MONTH)] = 1
            cooling_days[(cooling_days_seq[i, seq_start]-Cgbdisagg.DAYS_IN_MONTH):cooling_days_seq[i, seq_start]] = 1

    for i in range(len(extra_cooling_seq)):

        if extra_cooling_seq[i, seq_label] and (np.sum(cooling_days[extra_cooling_seq[i, seq_start]-2:extra_cooling_seq[i, seq_end]+3]) == 0):
            residual_cool[extra_cooling_seq[i, seq_start]:extra_cooling_seq[i, seq_end] + 1] = 0

    # Once the additional cooling signature is prepared, it is added to mid and max consumption values of cooling

    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    res_data = np.fmax(0, item_output_object.get("hybrid_input_data").get("true_disagg_res"))
    base_res = np.maximum(np.percentile(res_data, 10, axis=0),
                          np.percentile(input_data, 20, axis=0))
    base_res_2d = np.zeros_like(res_data)
    base_res_2d[:, :] = base_res[None, :]
    base_res_2d = np.maximum(base_res_2d, np.percentile(res_data, 3, axis=1)[:, None])

    mid_cons = mid_cons + np.fmax(0, np.minimum(input_data - base_res_2d, np.minimum(residual_cool, res_data)))
    max_cons = max_cons + np.fmax(0, np.minimum(input_data - base_res_2d, np.minimum(residual_cool, res_data)))

    mid_cons = np.maximum(mid_cons, disagg_cons * min_disagg_frac_required)
    max_cons = np.maximum(max_cons, disagg_cons * min_disagg_frac_required)
    min_cons = np.maximum(min_cons, disagg_cons * min_disagg_frac_required)

    ##### RULE 2 - Removing baseload type consumption inorder to handle stat app underestimation cases if required  #######

    vacation = np.logical_not(item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool))
    vacation[np.percentile(input_data, 25, axis=1) == 0] = 0

    min_baseload_perc = remove_baseload_cons(vacation, disagg_cons, config, item_output_object)

    logger.info("perc of input data to be removed from cooling output | %s ", min_baseload_perc)

    non_summer_days_present = (np.prod(season) == 0 or np.prod(season) < 0) and (len(input_data) > Cgbdisagg.DAYS_IN_MONTH*3)

    remove_baseload_condition = (non_summer_days_present and pilot in heavy_hvac_pilot and (np.sum(season < 0) > 0.1 * len(input_data))) or\
                                (non_summer_days_present and pilot not in heavy_hvac_pilot)

    if remove_baseload_condition and item_input_object.get('item_input_params').get('run_hybrid_v2_flag'):
        base_cons = np.percentile(input_data, min_baseload_perc, axis=0)
        min_data = copy.deepcopy(input_data)
        min_data = min_data - base_cons[None, :]
        max_cons = np.minimum(max_cons, min_data)
        min_cons = np.minimum(min_cons, min_data)
        mid_cons = np.minimum(mid_cons, min_data)

    max_cons = np.minimum(max_cons, input_data)
    min_cons = np.minimum(min_cons, input_data)
    mid_cons = np.minimum(mid_cons, input_data)

    ##################### RULE 3 - Not adding additional consumption in HVAC ao output  #############################

    appliance_list = item_input_object.get("item_input_params").get("app_list")
    ev_idx = np.where(np.array(appliance_list) == 'ev')[0][0]
    ev_cons = item_output_object.get("inference_engine_dict").get("output_data")[ev_idx, :, :]
    residual_cool[ev_cons > 0] = 0

    max_cons[np.logical_and(residual_cool == 0, disagg_cons == cool_ao)] = np.minimum(max_cons, disagg_cons)[
        np.logical_and(residual_cool == 0, disagg_cons == cool_ao)]
    mid_cons[np.logical_and(residual_cool == 0, disagg_cons == cool_ao)] = np.minimum(mid_cons, disagg_cons)[
        np.logical_and(residual_cool == 0, disagg_cons == cool_ao)]
    min_cons[np.logical_and(residual_cool == 0, disagg_cons == cool_ao)] = np.minimum(min_cons, disagg_cons)[
        np.logical_and(residual_cool == 0, disagg_cons == cool_ao)]

    # Updating the values in the original dictionary

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons)

    item_output_object["inference_engine_dict"]["output_data"][app_index, :, :] = disagg_cons

    t_end = datetime.now()

    logger.debug("Cooling inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def get_heat_inference(app_index, item_input_object, item_output_object, date_list, logger_pass):

    """
    Update heating consumption ranges using inference rules

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        date_list                   (np.ndarray): list of target dates for heatmap dumping
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    season = item_output_object.get("season")

    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    config = get_inf_config().get("heat")

    conf_thres = config.get('conf_thres')
    min_disagg_frac_required = config.get('min_disagg_frac_required')
    min_baseload_perc =  config.get('min_baseload_perc')
    cool_days_frac_thres = config.get('cool_days_frac_thres')
    min_baseload_perc_val = config.get('min_baseload_perc_val')

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_heating_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    max_cons = item_output_object.get("inference_engine_dict").get("appliance_max_values")[app_index, :, :]
    min_cons = item_output_object.get("inference_engine_dict").get("appliance_min_values")[app_index, :, :]
    app_conf = item_output_object.get("inference_engine_dict").get("appliance_conf")[app_index, :, :]
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]

    heat_ao = item_input_object.get("item_input_params").get("ao_heat")

    if heat_ao is None:
        heat_ao = np.zeros(disagg_cons.shape)

    max_cons = np.maximum(max_cons, disagg_cons)
    mid_cons[app_conf > conf_thres] = np.maximum(mid_cons, disagg_cons)[app_conf > conf_thres]
    min_cons[app_conf > conf_thres] = np.maximum(min_cons, disagg_cons)[app_conf > conf_thres]
    min_cons = np.minimum(min_cons, mid_cons)

    disagg_res = item_output_object.get("hybrid_input_data").get("true_disagg_res")

    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    ########## RULE 1 - Adding extra heating signature detected from disagg residual into cooling output  ####################

    residual_heat = np.fmax(0, np.minimum(item_output_object.get("hvac_dict").get("heating"), disagg_res))

    # before adding the extra heating signature, it is made sure that the added heating in not inconsistent
    # For which, we remove heating segments if detected for less days of chunks

    mid_cons = mid_cons + np.fmax(0, np.minimum(residual_heat, np.fmax(0, disagg_res)))
    max_cons = max_cons + np.fmax(0, np.minimum(residual_heat, np.fmax(0, disagg_res)))

    seq = find_seq(residual_heat.sum(axis=1) > 0, np.zeros_like(residual_heat.sum(axis=1)), np.zeros_like(residual_heat.sum(axis=1)), overnight=0)

    cooling_days = np.sum(disagg_cons, axis=1) > 0

    for i in range(len(seq)):
        if seq[i, seq_label] and (np.sum(cooling_days[seq[i, seq_start]-2:seq[i, seq_end] + 3]) == 0):
            residual_heat[seq[i, seq_start]:seq[i, seq_end] + 1] = 0

    # Removing always on component from hvac, before calculating min,mid,max ts level consumption
    # This AO component will be added later

    if item_input_object.get("item_input_params").get("ao_heat") is not None:
        min_cons = min_cons + item_input_object.get("item_input_params").get("ao_heat")
        mid_cons = mid_cons + item_input_object.get("item_input_params").get("ao_heat")
        max_cons = max_cons + item_input_object.get("item_input_params").get("ao_heat")

    min_cons = np.nan_to_num(min_cons)

    min_cons = np.minimum(min_cons, mid_cons)
    max_cons = np.maximum(max_cons, mid_cons)

    mid_cons = np.maximum(mid_cons, disagg_cons*min_disagg_frac_required)
    max_cons = np.maximum(max_cons, disagg_cons*min_disagg_frac_required)
    min_cons = np.maximum(min_cons, disagg_cons*min_disagg_frac_required)

    ##### RULE 2 - Removing baseload type consumption inorder to handle stat app underestimation cases if required  #######

    cool_idx = np.where(np.array(appliance_list) == 'cooling')[0][0]

    cooling_cons = item_output_object.get("inference_engine_dict").get("output_data")[cool_idx, :, :]

    # remove base consumption

    vacation = np.logical_not(item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool))
    vacation[np.percentile(input_data, 75, axis=1) == 0] = 0

    if np.any(vacation):

        cooling_days_fraction = (np.sum(cooling_cons[vacation] + disagg_cons[vacation], axis=1) > 0).sum() / len(disagg_cons[vacation])

        if cooling_days_fraction > cool_days_frac_thres[0]:
            min_baseload_perc = min_baseload_perc_val[0]
        if cooling_days_fraction > cool_days_frac_thres[1]:
            min_baseload_perc = min_baseload_perc_val[1]

    logger.info("perc of input data to be removed from heating output | %s ", min_baseload_perc)

    remove_baseload_cons_flag = np.prod(season) == 0 or np.prod(season) < 0 and (len(input_data) > Cgbdisagg.DAYS_IN_MONTH * 3)

    if remove_baseload_cons_flag and item_input_object.get('item_input_params').get('run_hybrid_v2_flag'):
        base_cons = np.percentile(input_data, min_baseload_perc, axis=0)

        min_data = copy.deepcopy(input_data)
        min_data = min_data - base_cons[None, :]
        max_cons = np.minimum(max_cons, min_data)
        min_cons = np.minimum(min_cons, min_data)
        mid_cons = np.minimum(mid_cons, min_data)

    max_cons = np.fmax(0, max_cons)
    max_cons = np.minimum(max_cons, input_data)
    min_cons = np.minimum(min_cons, input_data)
    mid_cons = np.minimum(mid_cons, input_data)

    ##################### RULE 3 - Not adding additional consumption in HVAC ao output  #############################

    appliance_list = item_input_object.get("item_input_params").get("app_list")
    ev_idx = np.where(np.array(appliance_list) == 'ev')[0][0]
    ev_cons = item_output_object.get("inference_engine_dict").get("output_data")[ev_idx, :, :]
    residual_heat[ev_cons > 0] = 0

    max_cons[np.logical_and(residual_heat == 0, disagg_cons == heat_ao)] = \
        np.minimum(max_cons, disagg_cons)[np.logical_and(residual_heat == 0, disagg_cons == heat_ao)]
    mid_cons[np.logical_and(residual_heat == 0, disagg_cons == heat_ao)] = \
        np.minimum(mid_cons, disagg_cons)[np.logical_and(residual_heat == 0, disagg_cons == heat_ao)]
    min_cons[np.logical_and(residual_heat == 0, disagg_cons == heat_ao)] = \
        np.minimum(min_cons, disagg_cons)[np.logical_and(residual_heat == 0, disagg_cons == heat_ao)]

    if np.sum(disagg_cons) > 0:
        mid_cons = np.fmin(mid_cons, np.max(disagg_cons))
        max_cons = np.fmin(max_cons, np.max(disagg_cons))
        min_cons = np.fmin(min_cons, np.max(disagg_cons))

    # Updating the values in the original dictionary

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons)

    item_output_object["inference_engine_dict"]["output_data"][app_index, :, :] = disagg_cons

    t_end = datetime.now()

    logger.debug("Heating inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def remove_baseload_cons(vacation, disagg_cons, config, item_output_object):

    """
     Update cooling consumption ranges using inference rules

     Parameters:
         vacation                   (np.ndarray) : vacation data
         disagg_cons                (np.ndarray) : cooling disagg consumption
         config                     (dict)       : WH config
         item_output_object         (dict)       : Dict containing all hybrid outputs

     Returns:
         min_baseload_perc          (int)      : percetile used to remove baseload consumption
     """

    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")

    heat_idx = np.where(np.array(appliance_list) == 'heating')[0][0]

    heating_cons = item_output_object.get("inference_engine_dict").get("output_data")[heat_idx, :, :]

    min_baseload_perc = config.get('min_baseload_perc')

    heating_days_frac = (np.sum(heating_cons[vacation] + disagg_cons[vacation], axis=1) > 0).sum() / len( disagg_cons[vacation])

    if np.any(vacation):
        if heating_days_frac > config.get('hvac_days_frac_thres')[0]:
            min_baseload_perc = config.get('min_baseload_perc_val')[0]
        if heating_days_frac > config.get('hvac_days_frac_thres')[1]:
            min_baseload_perc = config.get('min_baseload_perc_val')[1]
        if heating_days_frac > config.get('hvac_days_frac_thres')[2]:
            min_baseload_perc = config.get('min_baseload_perc_val')[2]
        if ((np.sum(disagg_cons[vacation], axis=1) > 0).sum() / len(disagg_cons[vacation]) > 0.95) and len(disagg_cons) > 300:
            min_baseload_perc = 5

    return min_baseload_perc
