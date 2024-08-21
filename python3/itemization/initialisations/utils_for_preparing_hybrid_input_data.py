
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Prepare data required in itemization pipeline
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.get_day_data import get_day_data

from python3.itemization.aer.functions.hsm_utils import fetch_pp_hsm
from python3.itemization.aer.functions.hsm_utils import fetch_wh_hsm
from python3.itemization.aer.functions.hsm_utils import fetch_ev_hsm
from python3.itemization.aer.functions.hsm_utils import fetch_ref_hsm


def update_object_with_disagg_debug_params(item_input_object, sampling_rate):

    """
    update item input data with disagg debug parameters

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        sampling_rate             (int)       : sampling rate of the user

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
    """

    # Fetch wh thin pulse

    if item_input_object.get("disagg_special_outputs") is not None and \
            item_input_object.get("disagg_special_outputs").get("final_thin_pulse") is not None:
        item_input_object["item_input_params"]["final_thin_pulse"] = \
            get_day_data(item_input_object.get("disagg_special_outputs").get("final_thin_pulse"),
                         sampling_rate)[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    # Fetch wh fat pulse

    if item_input_object.get("disagg_special_outputs") is not None and \
            item_input_object.get("disagg_special_outputs").get("final_fat_pulse") is not None:
        item_input_object["item_input_params"]["final_fat_pulse"] = \
            get_day_data(item_input_object.get("disagg_special_outputs").get("final_fat_pulse"),
                         sampling_rate)[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]
        item_input_object['item_input_params']['timed_wh_user'] = 0

    # Fetch HVAC on demand and always on components

    if item_input_object.get("ao_seasonality") is not None and item_input_object.get("ao_seasonality").get("epoch_cooling") is not None:
        input_data = copy.deepcopy(item_input_object.get("input_data"))
        input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = item_input_object.get("ao_seasonality").get("epoch_cooling")

        item_input_object["item_input_params"]["ao_cool"] = \
            get_day_data(input_data, sampling_rate)[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]
    else:
        input_data = copy.deepcopy(item_input_object.get("input_data"))
        input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        item_input_object["item_input_params"]["ao_cool"] = \
            get_day_data(input_data, sampling_rate)[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    if item_input_object.get("ao_seasonality") is not None and item_input_object.get("ao_seasonality").get("epoch_heating") is not None:
        input_data = copy.deepcopy(item_input_object.get("input_data"))
        input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = item_input_object.get("ao_seasonality").get("epoch_heating")

        item_input_object["item_input_params"]["ao_heat"] = \
            get_day_data(input_data, sampling_rate)[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]
    else:
        input_data = copy.deepcopy(item_input_object.get("input_data"))
        input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        item_input_object["item_input_params"]["ao_heat"] = \
            get_day_data(input_data, sampling_rate)[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    return item_input_object


def get_app_list(item_input_object, output_data):

    """
    fetch app list

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        output_data               (np.ndarray): disagg output

    Returns:
        appliance_list            (list)      : list of all appliances
        appliance_output_index    (list)      : list of app output index
        output_data               (np.ndarray): disagg output
        solar_gen                 (bool)      : denotes whether solar was detected for this user
    """

    appliance_list = []

    solar_gen = np.zeros_like(output_data[:, 0])

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

    if 'solar' in list(item_input_object.get("disagg_output_write_idx_map").keys()):
        solar_index = item_input_object.get("disagg_output_write_idx_map").get('solar')

        solar_gen = copy.deepcopy(output_data[:, solar_index])
        output_data = np.delete(output_data, np.s_[solar_index], axis=1)

    return appliance_list, appliance_output_index, output_data, solar_gen


def fetch_wh_app_prof_info(item_input_object, logger):

    """
    fetch wh app prof attributes info
    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        logger                    (logger)    : logger info

    Returns:
        tankless_wh               (bool)      : true if wh is tankless in app prof info
    """

    tankless_wh = 0

    app_profile = item_input_object.get('app_profile').get('wh')

    if app_profile is not None:
        type = item_input_object.get("app_profile").get('wh').get("attributes", '')
        if type is not None and ("tankless" in type):
            tankless_wh = 1
            logger.info("WH is tankless based on app profile info | ")

    return tankless_wh


def prepare_hsm_data(item_input_object, day_input_data, samples_per_hour):

    """
    fetch HSM info for hybrid v2 pipeline

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        day_input_data            (np.ndarray): day input data
        samples_per_hour          (int)       : samples in an hour

    Returns:
        hsm_params                (list)      : list of required hsm info
    """

    valid_pp_hsm, pp_hsm = fetch_pp_hsm(item_input_object, day_input_data, samples_per_hour)
    valid_ev_hsm, ev_hsm = fetch_ev_hsm(item_input_object, day_input_data)
    valid_ref_hsm, ref_hsm = fetch_ref_hsm(item_input_object, day_input_data)
    valid_wh_hsm, wh_hsm = fetch_wh_hsm(item_input_object, day_input_data, samples_per_hour)

    life_hsm = None

    created_life_hsm = dict()

    valid_life_hsm = True

    less_days_in_inc_data = (item_input_object.get('config').get('disagg_mode') == 'incremental' and len(day_input_data) < 70)

    hsm_in, valid_hsm_present = check_validity_of_li_hsm(item_input_object)

    if valid_hsm_present:
        valid_life_hsm = False

    elif hsm_in == {} and ((item_input_object.get('config').get('disagg_mode') not in ['incremental', 'historical']) or less_days_in_inc_data):
        valid_life_hsm = True
    elif hsm_in != {}:
        life_hsm = hsm_in.get('attributes')

        if isinstance(life_hsm, list):
            tou = life_hsm[0].get('item_lunch')
            amplitude = life_hsm[0].get('item_occ_count')
            conf = life_hsm[0].get('item_occ_prof')
            hld = life_hsm[0].get('item_weekday_delta')
            hld2 = life_hsm[0].get('item_weekend_delta')
        else:
            tou = life_hsm.get('item_lunch')
            amplitude = life_hsm.get('item_occ_count')
            conf = life_hsm.get('item_occ_prof')
            hld = life_hsm.get('item_weekday_delta')
            hld2 = life_hsm.get('item_weekend_delta')

        valid_life_hsm = not (tou is None or amplitude is None or hld is None or hld2 is None or conf is None)

        if valid_life_hsm and int(len(hld) / Cgbdisagg.HRS_IN_DAY) != samples_per_hour:
            valid_life_hsm = 0

    disagg_mode = item_input_object.get('config').get('disagg_mode')

    if (disagg_mode in ['historical']) or less_days_in_inc_data:
        valid_pp_hsm = 0
        valid_ev_hsm = 0

    if disagg_mode in ['historical']:
        valid_wh_hsm = 0
        valid_ref_hsm = 0
        valid_life_hsm = 0

        valid_pp_hsm = 0
        valid_ev_hsm = 0

    hsm_params = [valid_pp_hsm, pp_hsm, valid_ev_hsm, ev_hsm, valid_wh_hsm, wh_hsm, valid_ref_hsm, ref_hsm, valid_life_hsm, created_life_hsm, life_hsm]

    return hsm_params


def check_validity_of_li_hsm(item_input_object):

    """
    fetch HSM info for hybrid v2 pipeline

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs

    Returns:
        hsm_params                (list)      : list of required hsm info
    """

    try:
        hsm_dic = item_input_object.get('appliances_hsm')
        hsm_in = hsm_dic.get('li')
    except KeyError:
        hsm_in = None

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (item_input_object.get('config').get('disagg_mode') == 'historical')


    valid_hsm_present = \
        (hsm_in != {}) and (hsm_fail or (hsm_in is None or len(hsm_in.get('attributes')) == 0 or
                                         hsm_in.get('attributes') is None or hsm_in.get('attributes').get('item_lunch') is None or
                                         hsm_in.get('attributes').get('item_occ_prof') is None or
                                         hsm_in.get('attributes').get('item_occ_count') is None or
                                         hsm_in.get('attributes').get('item_weekday_delta') is None or
                                         hsm_in.get('attributes').get('item_weekend_delta') is None))

    return hsm_in, valid_hsm_present
