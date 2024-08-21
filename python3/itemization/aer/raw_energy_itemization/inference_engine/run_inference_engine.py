
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Calculate inference rules for all appliances
"""

# Import python packages

import logging
import traceback
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.raw_energy_itemization.inference_engine.get_app_range import initialize_app_range

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.postprocess_app_ranges import adjust_disagg_app

from python3.itemization.aer.raw_energy_itemization.inference_engine.initialize_ranges import initialize_ts_level_ranges

from python3.itemization.aer.raw_energy_itemization.inference_engine.get_pp_inference import get_pp_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_ev_inference import get_ev_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_wh_inference import get_wh_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_ao_inference import get_ao_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_li_inference import get_li_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_ref_inference import get_ref_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_twh_inference import get_twh_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_hvac_inference import get_cool_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_hvac_inference import get_heat_inference

from python3.itemization.aer.raw_energy_itemization.inference_engine.get_ld_inference import get_ld_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_ent_inference import get_ent_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_cook_inference import get_cook_inference
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_residual_inference import get_res_inference


def get_app_cons_range(item_input_object, item_output_object, logger_pass):

    """
    Calculate ts level consumption range for all appliances

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    app_potential = item_output_object.get("app_potential")
    app_confidence = item_output_object.get("app_confidence")

    input_data = item_output_object.get("hybrid_input_data").get("input_data")
    output_data = item_output_object.get("hybrid_input_data").get("output_data")
    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")
    weather_analytics = item_output_object.get("hybrid_input_data").get("weather_analytics")
    original_residual = item_output_object.get("hybrid_input_data").get("updated_residual_data")
    residual_without_detected_sig = item_output_object.get("hybrid_input_data").get("updated_residual_without_detected_sig")

    output_data = output_data[1:, :, :]
    app_confidence = app_confidence[1:, :, :]
    app_potential = app_potential[1:, :, :]

    # Initialize appliance min/max/avg range

    min_range, mid_range, max_range, app_conf, app_pot, app_cons = initialize_app_range(output_data, app_confidence, app_potential)

    inference_engine_dict = dict()

    inference_engine_dict.update({
        "appliance_conf": app_conf,
        "appliance_pot": app_pot,
        "appliance_max_values": max_range,
        "appliance_min_values": min_range,
        "appliance_mid_values": mid_range,
        "residual_data": original_residual,
        "residual_without_detected_sig": residual_without_detected_sig,
        "input_data": input_data,
        "output_data": output_data,
        "appliance_list": appliance_list,
        "app_cons": app_cons
    })

    item_output_object["inference_engine_dict"] = inference_engine_dict

    regression, total_consumption, min_range, mid_range, max_range, app_cons, app_conf, app_pot = \
        initialize_ts_level_ranges(item_input_object, item_output_object, logger_pass)

    item_output_object["inference_engine_dict"].update({
        "appliance_conf": app_conf,
        "appliance_pot": app_pot,
        "appliance_max_values": max_range,
        "appliance_min_values": min_range,
        "appliance_mid_values": mid_range,
        "residual_data": original_residual,
        "input_data": input_data,
        "output_data": output_data,
        "appliance_list": appliance_list,
        "regression_dict": regression,
        "weather_analytics": weather_analytics
    })

    item_output_object = get_appliance_inference(item_input_object, item_output_object, appliance_list, logger_pass)

    return item_output_object


def get_appliance_inference(item_input_object, item_output_object, appliance_list, logger_pass):

    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    t_inference_start = datetime.now()

    date_list = item_output_object.get("date_list")

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_appliance_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t0 = datetime.now()

    try:
        item_output_object = \
            get_res_inference(item_output_object)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in residual inference module | %s', error_str)

    # Running modules to detect signatures in residual data

    t1 = datetime.now()

    for app in appliance_list:
        item_input_object, item_output_object = \
            call_app_inf_wrappers(app, item_input_object, item_output_object, appliance_list, date_list, logger, logger_pass)

    t_inference_end = datetime.now()
    # Adjust appliances ranges in scenarios of multi appliance users (overlapping output of true disagg appliances)

    t2 = datetime.now()

    item_input_object, item_output_object = adjust_disagg_app(item_input_object, item_output_object, logger_pass)

    t3 = datetime.now()

    logger.info("Inference part 1 took  %.3f s | ", get_time_diff(t0, t1))
    logger.info("Inference part 2 took %.3f s | ", get_time_diff(t1, t2))
    logger.info("Inference part 3 took %.3f s | ", get_time_diff(t2, t3))

    logger.info('Appliance inference module took | %.3f s ', get_time_diff(t_inference_start, t_inference_end))

    return item_output_object


def call_app_inf_wrappers(app, item_input_object, item_output_object, appliance_list, date_list, logger, logger_pass):

    """
    Calculate inference rules for target appliance

    Parameters:
        app                         (str)       : target appliance
        item_input_object           (dict)      : Dict containing all inputs
        item_output_object          (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        date_list                   (np.ndarray): list of dates of user input data
        logger                      (dict)      : logger object
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object          (dict)      : Dict containing all inputs
        item_output_object         (dict)      : Dict containing all outputs
    """

    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')

    # Calculating inference rules for heating appliance

    if app == 'heating':
        item_input_object, item_output_object = heat_inf_wrapper(item_input_object, item_output_object, appliance_list,
                                                                 logger, logger_pass)

    # Calculating inference rules for cooling appliance

    if app == 'cooling':
        item_input_object, item_output_object = cool_inf_wrapper(item_input_object, item_output_object, appliance_list,
                                                                 logger, logger_pass)

    # Calculating inference rules for pp appliance

    if app == 'pp':
        item_input_object, item_output_object = pp_inf_wrapper(item_input_object, item_output_object, appliance_list,
                                                               logger, logger_pass)

    # Calculating inference rules for ev appliance

    if app == 'ev':
        item_input_object, item_output_object = ev_inf_wrapper(item_input_object, item_output_object,
                                                               appliance_list, logger, logger_pass)

    # Calculating inference rules for wh appliance

    if app == 'wh':
        item_input_object, item_output_object = wh_inf_wrapper(item_input_object, item_output_object, appliance_list,
                                                               logger, logger_pass)

    if app == 'ao':
        item_input_object, item_output_object = ao_inf_wrapper(item_input_object, item_output_object, appliance_list,
                                                               logger, logger_pass)

    if app == 'ref':
        item_input_object, item_output_object = ref_inf_wrapper(item_input_object, item_output_object, appliance_list,
                                                                logger, logger_pass)

    if app == 'li':
        item_input_object, item_output_object = li_inf_wrapper(item_input_object, item_output_object, appliance_list,
                                                               logger, logger_pass)

    # Calculating inference rules for laundry appliance

    if app == 'ld' and run_hybrid_v2:
        item_input_object, item_output_object = ld_inf_wrapper(item_input_object, item_output_object, appliance_list,
                                                               date_list, logger, logger_pass)

    # Calculating inference rules for cooking appliance

    if app == 'cook' and run_hybrid_v2:
        item_input_object, item_output_object = cook_inf_wrapper(item_input_object, item_output_object, appliance_list,
                                                                 date_list, logger, logger_pass)

    # Calculating inference rules for entertainment appliance

    if app == 'ent'and run_hybrid_v2:
        item_input_object, item_output_object = ent_inf_wrapper(item_input_object, item_output_object, appliance_list,
                                                                logger, logger_pass)

    return item_input_object, item_output_object


def ent_inf_wrapper(item_input_object, item_output_object, appliance_list, logger, logger_pass):
    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object           (dict)      : Dict containing all inputs
        item_output_object          (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger                      (dict)      : logger object
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    ent_index = np.where(np.array(appliance_list) == 'ent')[0][0]

    try:
        item_output_object = get_ent_inference(ent_index, item_input_object, item_output_object, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ent app inference module | %s', error_str)

    return item_input_object, item_output_object


def ld_inf_wrapper(item_input_object, item_output_object, appliance_list, date_list, logger, logger_pass):
    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object           (dict)      : Dict containing all inputs
        item_output_object          (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger                      (dict)      : logger object
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    ld_index = np.where(np.array(appliance_list) == 'ld')[0][0]

    try:
        item_output_object = get_ld_inference(ld_index, item_input_object, item_output_object, date_list, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ld app inference module | %s', error_str)

    return item_input_object, item_output_object


def cook_inf_wrapper(item_input_object, item_output_object, appliance_list, date_list, logger, logger_pass):

    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object           (dict)      : Dict containing all inputs
        item_output_object          (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger                      (dict)      : logger object
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    cook_index = np.where(np.array(appliance_list) == 'cook')[0][0]

    try:
        item_output_object = get_cook_inference(cook_index, item_input_object, item_output_object, date_list, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in cook app inference module | %s', error_str)

    return item_input_object, item_output_object


def li_inf_wrapper(item_input_object, item_output_object, appliance_list, logger, logger_pass):

    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    li_index = np.where(np.array(appliance_list) == 'li')[0][0]

    try:
        item_output_object = get_li_inference(li_index, item_input_object, item_output_object, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in li app inference module | %s', error_str)

    return item_input_object, item_output_object


def ref_inf_wrapper(item_input_object, item_output_object, appliance_list, logger, logger_pass):

    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    date_list = item_output_object.get("date_list")

    ref_index = np.where(np.array(appliance_list) == 'ref')[0][0]

    try:
        item_output_object = get_ref_inference(ref_index, item_input_object, item_output_object, date_list,
                                               logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ref app inference module | %s', error_str)

    return item_input_object, item_output_object


def ao_inf_wrapper(item_input_object, item_output_object, appliance_list, logger, logger_pass):

    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    date_list = item_output_object.get("date_list")

    ao_index = np.where(np.array(appliance_list) == 'ao')[0][0]

    try:
        item_output_object = get_ao_inference(ao_index, item_input_object, item_output_object, date_list,
                                              logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ao app inference module | %s', error_str)

    return item_input_object, item_output_object


def wh_inf_wrapper(item_input_object, item_output_object, appliance_list, logger, logger_pass):

    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    date_list = item_output_object.get("date_list")

    wh_index = np.where(np.array(appliance_list) == 'wh')[0][0]

    try:
        if item_input_object.get("item_input_params").get("timed_wh_user"):
            item_output_object = get_twh_inference(wh_index, item_input_object, item_output_object, logger_pass)

        else:
            item_input_object, item_output_object = get_wh_inference(wh_index, item_input_object, item_output_object, date_list, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in wh app inference module | %s', error_str)

    return item_input_object, item_output_object


def ev_inf_wrapper(item_input_object, item_output_object, appliance_list, logger, logger_pass):

    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    ev_index = np.where(np.array(appliance_list) == 'ev')[0][0]

    try:
        item_output_object = get_ev_inference(ev_index, item_input_object, item_output_object, logger_pass)
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ev app inference module | %s', error_str)

    return item_input_object, item_output_object


def pp_inf_wrapper(item_input_object, item_output_object, appliance_list, logger, logger_pass):

    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    pp_index = np.where(np.array(appliance_list) == 'pp')[0][0]

    try:
        item_output_object = get_pp_inference(pp_index, item_input_object, item_output_object, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in pp app inference module | %s', error_str)

    return item_input_object, item_output_object


def cool_inf_wrapper(item_input_object, item_output_object, appliance_list, logger, logger_pass):

    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    cooling_index = np.where(np.array(appliance_list) == 'cooling')[0][0]

    try:
        item_output_object = \
            get_cool_inference(cooling_index, item_input_object, item_output_object, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in cool inference module | %s', error_str)

    return item_input_object, item_output_object


def heat_inf_wrapper(item_input_object, item_output_object, appliance_list, logger, logger_pass):

    """
    Calculate inference rules for all appliances

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        appliance_list              (list)      : list of appliance codes
        logger_pass                 (dict)      : logger dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    date_list = item_output_object.get("date_list")

    heating_index = np.where(np.array(appliance_list) == 'heating')[0][0]

    try:
        item_output_object = \
            get_heat_inference(heating_index, item_input_object, item_output_object, date_list, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in heat inference module | %s', error_str)

    return item_input_object, item_output_object
