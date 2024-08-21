
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Calculate potential and confidence for all appliances
"""

# Import python packages

import logging
import traceback
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_ao_potential import get_ao_potential
from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_pp_potential import get_pp_potential
from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_li_potential import get_li_potential
from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_ev_potential import get_ev_potential
from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_wh_potential import get_wh_potential
from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_twh_potential import get_twh_potential
from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_ref_potential import get_ref_potential
from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_ent_potential import get_ent_potential
from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_cook_potential import get_cook_potential
from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_ld_potential import get_ld_potential
from python3.itemization.aer.raw_energy_itemization.appliance_potential.get_hvac_potential import get_hvac_potential


def get_appliance_potential(item_input_object, item_output_object, logger_pass):

    """
    Calculate potential and confidence for all appliances

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    t_potential_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_appliance_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    sampling_rate = item_input_object.get("config").get("sampling_rate")
    output_data = item_output_object.get("hybrid_input_data").get("output_data")
    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")

    item_output_object["updated_output_data"] = output_data

    app_confidence = np.zeros(output_data.shape)
    app_potential = np.zeros(output_data.shape)

    item_output_object["app_confidence"] = app_confidence
    item_output_object["app_potential"] = app_potential

    for app in appliance_list:
        item_output_object = \
            calculate_appliance_potential_for_target_appliance(app, item_input_object, item_output_object, sampling_rate,
                                                               appliance_list, logger, logger_pass)

    t_potential_end = datetime.now()

    logger.info('Appliance potential module took | %.3f s ', get_time_diff(t_potential_start, t_potential_end))

    return item_output_object


def calculate_appliance_potential_for_target_appliance(app, item_input_object, item_output_object, sampling_rate,
                                                       appliance_list, logger, logger_pass):

    """
    Calculating potential and confidence for target appliance

    Parameters:
        app                       (str)           : target appliance
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    # Calculating potential and confidence for hvac appliance

    if app == 'cooling':
        item_output_object = hvac_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate,
                                                                appliance_list, logger, logger_pass)

    # Calculating potential and confidence for pp appliance

    if app == 'pp':
        item_output_object = \
            pp_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list,
                                             logger, logger_pass)

    # Calculating potential and confidence for li appliance

    if app == 'li':
        item_output_object = \
            li_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list,
                                             logger, logger_pass)

    # Calculating potential and confidence for ev appliance

    if app == 'ev':
        item_output_object = \
            ev_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list,
                                             logger, logger_pass)

    # Calculating potential and confidence for wh appliance

    if app == 'wh':
        item_output_object = \
            wh_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list,
                                             logger, logger_pass)

    # Calculating potential and confidence for AO appliance

    if app == 'ao':
        item_output_object = \
            ao_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list,
                                             logger, logger_pass)

    if app == 'ref' and run_hybrid_v2:
        item_output_object = \
            ref_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list,
                                              logger, logger_pass)

    if app == 'ld' and run_hybrid_v2:
        item_output_object = ld_potential_calculation_wrapper(vacation_days, item_input_object, item_output_object,
                                                              sampling_rate, appliance_list, logger, logger_pass)

    # Calculating potential and confidence for cooking appliance

    if app == 'cook' and run_hybrid_v2:
        item_output_object = \
            cook_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list,
                                               logger, logger_pass)

    # Calculating potential and confidence for entertainment appliance

    if app == 'ent' and run_hybrid_v2:
        item_output_object = \
            ent_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list,
                                              logger, logger_pass)

    return item_output_object


def cook_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list, logger, logger_pass):

    """
    Calculating potential and confidence for cooking appliance

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    cook_index = np.where(np.array(appliance_list) == 'cook')[0][0] + 1

    try:
        item_output_object = get_cook_potential(cook_index, item_input_object, item_output_object,
                                                sampling_rate, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in cook app potential module | %s', error_str)

    return item_output_object


def ent_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list, logger, logger_pass):

    """
    Calculating potential and confidence for ent appliance

    Parameters:
        vacation_days             (np.ndarray)    : vacation data
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    ent_index = np.where(np.array(appliance_list) == 'ent')[0][0] + 1
    try:
        item_output_object = get_ent_potential(ent_index, item_input_object, item_output_object,
                                               sampling_rate, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ent app potential module | %s', error_str)

    return item_output_object


def ld_potential_calculation_wrapper(vacation_days, item_input_object, item_output_object, sampling_rate, appliance_list, logger, logger_pass):

    """
    Calculating potential and confidence for ld appliance

    Parameters:
        vacation_days             (np.ndarray)    : vacation data
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    ld_index = np.where(np.array(appliance_list) == 'ld')[0][0] + 1

    try:
        item_output_object = get_ld_potential(ld_index, item_input_object, item_output_object,
                                              sampling_rate, vacation_days, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ld app potential module | %s', error_str)

    return item_output_object


def ev_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list, logger,
                                     logger_pass):
    """
    Calculating potential and confidence for ev appliance

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    weather_analytics = item_output_object.get("hybrid_input_data").get("weather_analytics")
    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    ev_index = np.where(np.array(appliance_list) == 'ev')[0][0] + 1

    try:
        item_output_object = get_ev_potential(ev_index, item_input_object, item_output_object,
                                              sampling_rate, weather_analytics, vacation_days, logger_pass)
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ev app potential module | %s', error_str)

    return item_output_object


def pp_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list, logger,
                                     logger_pass):
    """
    Calculating potential and confidence for pp appliance

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    pp_index = np.where(np.array(appliance_list) == 'pp')[0][0] + 1
    try:
        item_output_object = get_pp_potential(pp_index, item_input_object, item_output_object, sampling_rate,
                                              logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in pp app potential module | %s', error_str)

    return item_output_object


def li_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list, logger,
                                     logger_pass):
    """
    Calculating potential and confidence for li appliance

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    input_data = item_output_object.get("hybrid_input_data").get("input_data")
    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    li_index = np.where(np.array(appliance_list) == 'li')[0][0] + 1

    try:
        item_output_object = get_li_potential(li_index, item_input_object, item_output_object,
                                              input_data[Cgbdisagg.INPUT_SKYCOV_IDX, :, :], sampling_rate,
                                              vacation_days, logger_pass)
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in li app potential module | %s', error_str)

    return item_output_object


def ao_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list, logger, logger_pass):
    """
    Calculating potential and confidence for ao appliance

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    ao_index = np.where(np.array(appliance_list) == 'ao')[0][0] + 1

    try:
        item_output_object = get_ao_potential(ao_index, item_input_object, item_output_object, sampling_rate,
                                              logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ao app potential module | %s', error_str)

    return item_output_object


def hvac_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list, logger, logger_pass):

    """
    Calculating potential and confidence for heating/cooling appliance

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass               (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    cooling_index = np.where(np.array(appliance_list) == 'cooling')[0][0] + 1
    heating_index = np.where(np.array(appliance_list) == 'heating')[0][0] + 1
    try:
        item_output_object = \
            get_hvac_potential(cooling_index, heating_index, item_input_object, item_output_object,
                               sampling_rate, vacation_days,
                               item_input_object.get("item_input_params").get("ao_cool"),
                               item_input_object.get("item_input_params").get("ao_heat"), logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in hvac app potential module | %s', error_str)

    return item_output_object


def ref_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list, logger, logger_pass):

    """
    Calculating potential and confidence for ref appliance

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    ref_index = np.where(np.array(appliance_list) == 'ref')[0][0] + 1
    try:
        item_output_object = get_ref_potential(ref_index, item_input_object, item_output_object, sampling_rate,
                                               logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ref app potential module | %s', error_str)

    return item_output_object


def wh_potential_calculation_wrapper(item_input_object, item_output_object, sampling_rate, appliance_list, logger,
                                     logger_pass):
    """
   Calculating potential and confidence for wh appliance

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        sampling_rate             (int)           : sampling rate
        appliance_list            (list)          : lost of all appliances
        logger                    (logger)          : logger dictionary
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    wh_index = np.where(np.array(appliance_list) == 'wh')[0][0] + 1
    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    try:
        if item_input_object.get("item_input_params").get("timed_wh_user"):
            item_output_object = get_twh_potential(wh_index, item_input_object, item_output_object, sampling_rate,
                                                   logger_pass)

        else:
            item_output_object = get_wh_potential(wh_index, item_input_object, item_output_object,
                                                  sampling_rate, vacation_days,
                                                  item_input_object.get("item_input_params").get("final_thin_pulse"),
                                                  logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in wh app potential module | %s', error_str)

    return item_output_object
