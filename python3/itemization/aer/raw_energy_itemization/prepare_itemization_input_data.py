"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Master file for itemization pipeline
"""

# Import python packages

import copy
import pytz
import numpy as np
import pandas as pd

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.functions.get_day_data import get_hybrid_day_data


def prepare_itemization_data(item_input_object, item_output_object):

    """
    Perform 100% itemization

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    day_input_data = item_input_object.get("item_input_params").get("day_input_data")
    sampling_rate = item_input_object.get("config").get("sampling_rate")

    appliance_list = prepare_app_list(item_input_object)

    epoch_estimate = copy.deepcopy(item_output_object.get("epoch_estimate"))
    input_data = copy.deepcopy(item_input_object.get("input_data"))

    # Update input data if the user is a solar user

    if 'solar' in list(item_input_object.get("disagg_output_write_idx_map").keys()):
        solar_index = item_input_object.get("disagg_output_write_idx_map").get('solar')
        epoch_estimate = np.delete(epoch_estimate, np.s_[solar_index], axis=1)

    # Prepare input and output data
    # input data - (21, no of days, samples in a day)

    output_data = np.nan_to_num(epoch_estimate)
    input_data = np.nan_to_num(input_data)

    common_ts, input_ts, output_ts = np.intersect1d(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX], output_data[:, 0], return_indices=True)

    output_data = output_data[output_ts, :]
    input_data = input_data[input_ts, :]

    input_data, output_data, month_ts = get_hybrid_day_data(input_data, output_data, sampling_rate)

    # Reading input data, since it will be later user in plotting disagg cons

    pp_cons = get_disagg_2d_cons('pp', appliance_list, output_data, input_data)
    ev_cons = get_disagg_2d_cons('ev', appliance_list, output_data, input_data)
    cool_cons = get_disagg_2d_cons('cooling', appliance_list, output_data, input_data)
    heat_cons = get_disagg_2d_cons('heating', appliance_list, output_data, input_data)
    wh_cons = get_disagg_2d_cons('wh', appliance_list, output_data, input_data)
    ld_cons = get_disagg_2d_cons('ld', appliance_list, output_data, input_data)
    ent_cons = get_disagg_2d_cons('ent', appliance_list, output_data, input_data)
    cook_cons = get_disagg_2d_cons('cook', appliance_list, output_data, input_data)
    li_cons = get_disagg_2d_cons('li', appliance_list, output_data, input_data)
    ao_cons = get_disagg_2d_cons('ao', appliance_list, output_data, input_data)
    ref_cons = get_disagg_2d_cons('ref', appliance_list, output_data, input_data)

    # remove WH consumption during residual calculation in case of 0 wh app profile

    kill_wh = kill_wh_based_on_app_profile(item_input_object)

    # remove PP consumption during residual calculation in case of low confidence PP

    kill_pp = kill_pp_based_on_app_profile(item_input_object)

    # update stat app consumption in order to avoid overlap with disagg category

    input_data_without_disagg_app = input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX] - (heat_cons + cool_cons) * 0.6 - \
                                    ao_cons - ref_cons - wh_cons*0.7 - ev_cons - pp_cons

    if kill_wh:
        wh_cons[:, :] = 0

    if item_input_object.get('item_input_params').get('run_hybrid_v2_flag'):

        cook_idx = np.where(np.array(appliance_list) == 'cook')[0][0] + 1
        cook_cons = np.fmax(0, np.minimum(cook_cons, input_data_without_disagg_app))
        output_data[cook_idx] = cook_cons

        ent_idx = np.where(np.array(appliance_list) == 'ent')[0][0] + 1
        ent_cons = np.fmax(0, np.minimum(ent_cons, input_data_without_disagg_app))
        output_data[ent_idx] = ent_cons

        ld_idx = np.where(np.array(appliance_list) == 'ld')[0][0] + 1
        ld_cons = np.fmax(0, np.minimum(ld_cons, input_data_without_disagg_app))
        output_data[ld_idx] = ld_cons

        hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

        # Assign dishwasher consumption to cooking category, if dishwasher is to be included in cooking appliances

        if hybrid_config.get("dishwash_cat") == "cook":
            cook_cons, ld_cons = modify_ld_cook_based_on_dishwasher_config(appliance_list, input_data, output_data, item_output_object)

    # Calculate final residual

    residual = input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    residual = residual - (cool_cons + heat_cons + ev_cons + wh_cons + pp_cons + ao_cons + ref_cons + li_cons)

    hybrid_ref = (item_output_object.get("ref") is not None) and \
                 (item_output_object.get("ref").get("hybrid_ref") is not None) and \
                 (item_output_object.get("ref").get("hybrid_ref") > 0)

    if hybrid_ref:
        residual = residual + ref_cons

    if kill_pp:
        true_disagg_res = input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :] - \
                          (cool_cons + heat_cons + ev_cons + wh_cons + ao_cons + ref_cons)
        residual = residual + pp_cons
    else:
        true_disagg_res = input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :] - \
                          (cool_cons + heat_cons + ev_cons + wh_cons + pp_cons + ao_cons + ref_cons)

    if hybrid_ref:
        true_disagg_res = true_disagg_res + ref_cons

    for i in range(len(appliance_list)):
        output_data[i] = np.minimum(output_data[i], input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX])

    hybrid_input_data = dict()

    # Update input dictionary

    item_output_object["positive_residual"] = np.fmax(0, residual)
    item_output_object["positive_residual_copy"] = np.fmax(0, residual)
    item_output_object["negative_residual"] = np.fmax(0, -residual)
    item_output_object["original_input_data"] = input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]
    item_output_object["processed_input_data"] = day_input_data

    hybrid_input_data["positive_residual"] = np.fmax(0, residual)
    hybrid_input_data["positive_residual_copy"] = np.fmax(0, residual)
    hybrid_input_data["negative_residual"] = np.fmax(0, -residual)
    hybrid_input_data["original_input_data"] = input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]
    hybrid_input_data["processed_input_data"] = day_input_data
    hybrid_input_data["input_data"] = input_data
    hybrid_input_data["output_data"] = np.fmax(0, output_data)
    hybrid_input_data["original_res"] = residual
    hybrid_input_data["true_disagg_res"] = true_disagg_res
    hybrid_input_data["appliance_list"] = appliance_list
    hybrid_input_data["true_disagg_output"] = cool_cons + heat_cons + ev_cons + wh_cons + pp_cons

    hybrid_input_data["ao_cons"] = ao_cons

    item_output_object["hybrid_input_data"] = hybrid_input_data

    # Preparing intermediate disagg pool pump results, to be used for timed signature detection

    date = '2017-01-01'
    tz1 = pytz.timezone('UTC')
    tz2 = pytz.timezone(item_input_object.get("home_meta_data").get("timezone"))

    date = pd.to_datetime(date)
    diff = int((tz2.localize(date).astimezone(tz1) - tz1.localize(date)).seconds / sampling_rate)

    if item_input_object.get("disagg_special_outputs") is None or \
            item_input_object.get("disagg_special_outputs").get("pp_steps") is None:
        item_input_object["disagg_special_outputs"]["pp_steps"] = [np.zeros_like(input_data[0])] * 5

    else:
        for i in range(1, 5, 1):
            item_input_object.get("disagg_special_outputs").get("pp_steps")[i] = np.roll(
                item_input_object.get("disagg_special_outputs").get("pp_steps")[i], -diff, axis=1)

    return item_input_object, item_output_object


def get_disagg_2d_cons(app_name, appliance_list, output_data, input_data):

    """
    Fetch appliance disagg output

    Parameters:
        app_name                (str)       : app name
        appliance_list          (list)      : list of app
        input_data              (np.ndarray): raw input data
        output_data             (np.ndarray): disagg output data

    Returns:
        app_cons                (dict)      : Dict containing all inputs
    """

    if app_name in appliance_list:
        app_index = np.where(np.array(appliance_list) == app_name)[0][0] + 1
        output_data[app_index, :, :] = np.minimum(output_data[app_index, :, :], input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :])
        app_cons = output_data[app_index, :, :]
    else:
        app_cons = np.zeros_like(input_data[0])

    return app_cons


def kill_wh_based_on_app_profile(item_input_object):

    """
    calculate  bool to check if to kill wh

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs

    Returns:
        kill_wh                   (bool)      : bool if wh has to be killed
    """

    kill_wh = 0
    fuel_type = "ELECTRIC"
    app_profile = item_input_object.get("app_profile").get('wh')

    if app_profile is not None:
        fuel_type = app_profile.get("type", 'ELECTRIC')
        app_profile = app_profile.get("number", 0)
    else:
        app_profile = 1

    if (not app_profile) or fuel_type in ["SOLAR", "PROPANE", "GAS", "Gas", "SOLID_FUEL", "SOLID_FEUL", "OIL", "Oil", "WOOD", "Wood"]:
        kill_wh = 1

    return kill_wh


def kill_pp_based_on_app_profile(item_input_object):

    """
    calculate  bool to check if to kill wh

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs

    Returns:
        kill_pp                   (bool)      : bool if pp has to be killed
    """

    kill_pp = 0

    disagg_confidence = 1

    # kills PP if pp profile is none and pp detection confidence(calculated by hybrid) is less

    if (item_input_object.get('disagg_special_outputs') is not None) and \
            (item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') is not None):
        disagg_confidence = item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') / 100

    if (not item_input_object.get("item_input_params").get("pp_prof_present")) and disagg_confidence <= 0.65:
        kill_pp = 1

    return kill_pp


def prepare_app_list(item_input_object):

    """
    Prepares list of all app category for residential ami user

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs

    Returns:
        appliance_list            (list)      : list of appliances
    """

    appliance_list = []

    appliance_output_index = list(item_input_object.get("disagg_output_write_idx_map").keys())

    for key in appliance_output_index:

        if key != 'solar':
            if key == 'va':
                appliance_list.append("va1")
                appliance_list.append("va2")

            elif key == 'hvac':
                appliance_list.append("cooling")
                appliance_list.append("heating")
            else:
                appliance_list.append(key)

    return appliance_list

def modify_ld_cook_based_on_dishwasher_config(appliance_list, input_data, output_data, item_output_object):

    """
    Updated cooking/laundry output incase dishwasher should be included in cooking category

    Parameters:
        appliance_list            (list)      : list of all appliances
        input_data                (np.ndarray): raw input data
        output_data               (np.ndarray): disagg output data
        item_output_object        (dict)      : Dict containing all outputs

    Returns:
        cook_cons                 (np.ndarray): updated cooking output
        ld_cons                   (np.ndarray): updated laundry output
    """

    cook_idx = appliance_list.index("cook") + 1
    ld_idx = appliance_list.index("ld") + 1

    if item_output_object.get("debug").get("laundry_module_dict") is not None:
        output_data[cook_idx] = output_data[cook_idx] + item_output_object.get("debug").get("laundry_module_dict").get(
            "dish_washer_cons")
        output_data[ld_idx] = np.fmax(0, output_data[ld_idx] - item_output_object.get("debug").get(
            "laundry_module_dict").get("dish_washer_cons"))

    cook_cons = get_disagg_2d_cons('cook', appliance_list, output_data, input_data)
    ld_cons = get_disagg_2d_cons('ld', appliance_list, output_data, input_data)

    return cook_cons, ld_cons
