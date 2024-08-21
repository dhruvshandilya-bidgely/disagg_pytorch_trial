
"""
Author - Nisha Agarwal
Date - 4th April 2021
limit the change in hvac consumption in hybrid v2 compared to disagg after 100% itemization
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants


from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def adjust_seasonal_app_max_limit(appliance_list, item_input_object, item_output_object, output_data, final_adj_arr):

    """
    Limit ts level consumption increase for HVAC appliances

    Parameters:
        appliance_list              (list)          : List of target appliances
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        output_data                 (np.ndarray)    : ts level true disagg output for all appliances
        final_adj_arr               (np.ndarray)    : array containing factor which will be multiplied to solve high overestimation cases

    Returns:
        final_adj_arr               (np.ndarray)    : updated final adjustment array (with HVAC being limited)
        detected_cool               (np.ndarray)    : detected cooling signature
        detected_heat               (np.ndarray)    : detected heating signature
    """

    wh_idx = np.where(appliance_list == 'wh')[0][0]
    cool_idx = np.where(appliance_list == 'cooling')[0][0]
    heat_idx = np.where(appliance_list == 'heating')[0][0]

    detected_cool = item_output_object.get("hvac_dict").get("cooling")
    detected_heat = item_output_object.get("hvac_dict").get("heating")

    disagg_cool = copy.deepcopy(output_data[cool_idx])
    disagg_heat = copy.deepcopy(output_data[heat_idx])

    # Increase HVAC consumption only at points with on demand hvac (and not ao HVAC)

    disagg_cool[disagg_cool == 0] = np.nan
    disagg_heat[disagg_heat == 0] = np.nan

    cool_ao = item_input_object.get("item_input_params").get("ao_cool")

    if cool_ao is None:
        increase_cool = np.nan_to_num(disagg_cool) > np.nan_to_num(np.nanpercentile(disagg_cool, 20, axis=1))[:, None]
    else:
        increase_cool = disagg_cool > 1.1*cool_ao

    heat_ao = item_input_object.get("item_input_params").get("ao_heat")

    if heat_ao is None:
        increase_heat = np.nan_to_num(disagg_heat) > np.nan_to_num(np.nanpercentile(disagg_heat, 20, axis=1))[:, None]
    else:
        increase_heat = disagg_heat > 1.1 * heat_ao

    increase_cool = np.logical_or(increase_cool, detected_cool)
    increase_heat = np.logical_or(increase_heat, detected_heat)

    # Increase consumption at either on demand HVAC points or seasonal signature detected points

    final_adj_arr[cool_idx][np.logical_not(increase_cool)] = \
        np.fmin(1, final_adj_arr[cool_idx][np.logical_not(increase_cool)])
    final_adj_arr[heat_idx][np.logical_not(increase_heat)] = \
        np.fmin(1, final_adj_arr[heat_idx][np.logical_not(increase_heat)])

    final_adj_arr[cool_idx][np.logical_not(detected_cool)] = \
        np.fmin(5, final_adj_arr[cool_idx][np.logical_not(detected_cool)])
    final_adj_arr[heat_idx][np.logical_not(detected_heat)] = \
        np.fmin(5, final_adj_arr[heat_idx][np.logical_not(detected_heat)])

    room_count = item_input_object.get('home_meta_data').get('totalRooms')

    if room_count is None:
        room_count = 0

    config = get_inf_config(samples_per_hour=1).get("wh")

    max_jump_allowed_based_on_room_count = config.get('max_jump_allowed_based_on_room_count')
    room_count_buckets = config.get('room_count_buckets')

    factor = max_jump_allowed_based_on_room_count[np.digitize(room_count, room_count_buckets)]

    final_adj_arr[wh_idx] = np.fmin(factor, final_adj_arr[wh_idx])

    return final_adj_arr, detected_cool, detected_heat


def limit_hvac_delta(item_input_object, item_output_object, final_tou_consumption, appliance_list,
                     output_data, processed_input_data, detected_cool, detected_heat):

    """
    This function is added to limit BC level delta from disagg of HVAC appliances
     (In order to control HVAC overestimation in hybrid module)

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        appliance_list              (list)          : list of appliances
        output_data                 (np.ndarray)    : ts level true disagg output for all appliances
        processed_input_data        (np.ndarray)    : ts level input data
        detected_cool               (np.ndarray)    : detected cooling signature in residual data
        detected_heat               (np.ndarray)    : detected heating signature in residual data

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    # This functions limits the increase of cooling (in hybrid module) to 30%, excluding the detected seasonal signature

    config = get_inf_config(samples_per_hour=1).get("wh")

    appliances = ['cooling', 'heating']
    added_seasonal_signature = [detected_cool, detected_heat]
    app_delta_thres = config.get('app_delta_thres')
    max_cons_thres = config.get('max_cons_thres')
    window = config.get('max_delta_calc_window_thres')

    for app in range(len(appliances)):

        app_idx = np.where(appliance_list == appliances[app])[0][0]
        seasonal_signature = added_seasonal_signature[app]
        thres = app_delta_thres[app]
        cons_thres = max_cons_thres[app]

        for i in range(0, len(processed_input_data)-1, window):

            idx = np.zeros(processed_input_data.shape)
            idx[i:i + window] = 1
            idx = idx.astype(bool)

            detected_cool_copy = seasonal_signature[i:i+window]

            delta_from_disagg = (final_tou_consumption[app_idx, idx].sum() - output_data[app_idx, idx].sum() - detected_cool_copy.sum())

            # not reducing consumption if absolute delta is less than a certain threshold

            if (output_data[app_idx, idx].sum() == 0) or (delta_from_disagg < cons_thres):
                continue

            delta_from_disagg = (output_data[app_idx, idx].sum() - final_tou_consumption[app_idx, idx].sum() - detected_cool_copy.sum()) / \
                   output_data[app_idx, idx].sum() * 100

            factor = (final_tou_consumption[app_idx, idx].sum()-detected_cool_copy.sum()) / ((1 - thres / 100) * output_data[app_idx, idx].sum())

            # not reducing consumption if percentage delta is less than a certain threshold

            if (output_data[app_idx, idx].sum() == 0) or ((delta_from_disagg > thres) or (factor < 1)):
                continue

            # reducing hybrid consumption to maintain max delta from disagg

            final_tou_consumption[app_idx, np.logical_and(idx, seasonal_signature == 0)] = \
                final_tou_consumption[app_idx, np.logical_and(idx, seasonal_signature == 0)] / factor

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    wh_idx = np.where(appliance_list == 'wh')[0][0]

    wh_cons = copy.deepcopy(final_tou_consumption[wh_idx])

    pilot = item_input_object.get("config").get("pilot_id")

    swh_pilot_user = pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS

    for i in range(len(unique_bc)):
        if swh_pilot_user and (np.sum(wh_cons[bc_list == unique_bc[i]]) < config.get('min_swh_monthly_cons')):
            wh_cons[bc_list == unique_bc[i]] = 0

    final_tou_consumption[wh_idx] = wh_cons

    return final_tou_consumption
