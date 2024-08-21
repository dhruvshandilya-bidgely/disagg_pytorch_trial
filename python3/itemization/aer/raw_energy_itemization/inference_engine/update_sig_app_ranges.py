
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Prepare data required in itemization pipeline
"""

# Import python packages

import copy
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# import functions from within the project

from python3.config.pilot_constants import PilotConstants

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val_for_valid_boxes
from python3.itemization.aer.functions.itemization_utils import get_index

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.functions.hsm_utils import check_validity_of_hsm

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config

from python3.itemization.aer.raw_energy_itemization.residual_analysis.box_activity_detection_wrapper import box_detection

from python3.itemization.aer.raw_energy_itemization.residual_analysis.config.get_residual_config import get_residual_config
from python3.itemization.aer.raw_energy_itemization.inference_engine.util_functions_to_maintain_min_max_cons_for_disagg_app import block_low_cons_pp


def add_extra_pp_cons_to_cooling(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals,
                                 cooling_idx, pp_idx, output_data, logger, pp_copy):

    """
    Modify appliance cooling mid/min/max ranges and add reduced cooling (durinng pp/cooling adjustment)

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        cooling_idx               (int)           : cooling index
        pp_idx                    (int)           : PP index
        output_data               (np.ndarray)    : TS level disagg output
        logger                    (logger)        : logger object

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    reduced_pp = np.fmax(0, pp_copy - mid_cons_vals[pp_idx])

    # add reduced pp consumption into HVAC (if reduced during true disagg adjustment)

    reduced_cooling = np.fmax(0, item_output_object.get("inference_engine_dict").get("appliance_mid_values")[cooling_idx] - mid_cons_vals[cooling_idx])

    reduced_pp = np.minimum(reduced_cooling, reduced_pp)

    mid_cons_vals[cooling_idx] = mid_cons_vals[cooling_idx] + reduced_pp
    max_cons_vals[cooling_idx] = max_cons_vals[cooling_idx] + reduced_pp
    min_cons_vals[cooling_idx] = min_cons_vals[cooling_idx] + reduced_pp

    # block low bc level pp consumption

    min_cons_vals, mid_cons_vals, max_cons_vals = block_low_cons_pp(min_cons_vals, mid_cons_vals, max_cons_vals, logger, item_input_object, output_data)

    return min_cons_vals, mid_cons_vals, max_cons_vals


def add_thin_pulse_to_wh_cons(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals, wh_idx, output_data, logger):


    """
    Modify appliance WH mid/min/max ranges by adding extra thin pulses

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        wh_idx                    (int)           : PP index
        output_data               (np.ndarray)    : TS level disagg output
        logger                    (logger)        : logger object

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    # handle wh estimate changes after true disagg adjustment and add leftover thin pulse from residual data to wh output

    thin_pulse = np.zeros_like(mid_cons_vals[0])

    if np.sum(output_data[wh_idx]) == 0 and np.sum(mid_cons_vals[wh_idx]) > 0:
        wh_tou = mid_cons_vals[wh_idx] > 0
        mid_cons_vals[wh_idx][wh_tou == 0] = 0
        max_cons_vals[wh_idx][wh_tou == 0] = 0
        min_cons_vals[wh_idx][wh_tou == 0] = 0

        thin_pulse = allot_thin_pulse_boxes(item_input_object, item_output_object, mid_cons_vals, wh_idx, output_data[wh_idx],
                                            item_output_object["hybrid_input_data"]["updated_residual_data"], logger)
        thin_pulse = np.fmax(0, thin_pulse)

        mid_cons_vals[wh_idx] = mid_cons_vals[wh_idx] + thin_pulse
        max_cons_vals[wh_idx] = max_cons_vals[wh_idx] + thin_pulse
        min_cons_vals[wh_idx] = min_cons_vals[wh_idx] + thin_pulse

    if np.sum(output_data[wh_idx]) > 0:

        thin_pulse = allot_thin_pulse_boxes(item_input_object, item_output_object, mid_cons_vals, wh_idx, output_data[wh_idx],
                                            item_output_object["hybrid_input_data"]["updated_residual_data"], logger)
        thin_pulse = np.fmax(0, thin_pulse)

        mid_cons_vals[wh_idx] = mid_cons_vals[wh_idx] + thin_pulse
        max_cons_vals[wh_idx] = max_cons_vals[wh_idx] + thin_pulse
        min_cons_vals[wh_idx] = min_cons_vals[wh_idx] + thin_pulse

    return min_cons_vals, mid_cons_vals, max_cons_vals


def add_seasonality_in_ref(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals):


    """
    Add seasonality in ref consumption

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    ref_season = hybrid_config.get('ref_season')

    date_list = item_output_object.get("date_list")
    month_list = pd.DatetimeIndex(date_list).month.values

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    unique_bc = unique_bc[counts > 5]

    for i in range(len(unique_bc)):

        season_val = ref_season[(month_list[bc_list == unique_bc[i]] - 1).astype(int)].mean() + 1
        mid_cons_vals[get_index('ref')][bc_list == unique_bc[i]] = mid_cons_vals[get_index('ref')][bc_list == unique_bc[i]] * season_val
        min_cons_vals[get_index('ref')][bc_list == unique_bc[i]] = min_cons_vals[get_index('ref')][bc_list == unique_bc[i]] * season_val
        max_cons_vals[get_index('ref')][bc_list == unique_bc[i]] = max_cons_vals[get_index('ref')][bc_list == unique_bc[i]] * season_val

    return min_cons_vals, mid_cons_vals, max_cons_vals


def handle_low_ev_cons(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals, ev_idx, output_data, samples):


    """
    Removing low duration EV cases

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        ev_idx                    (int)           : PP index
    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    # handle ev estimate change after true disagg adjustment

    mid_cons_vals[ev_idx] = remove_low_cons_ev_boxes(mid_cons_vals[ev_idx], output_data[ev_idx])
    max_cons_vals[ev_idx] = remove_low_cons_ev_boxes(max_cons_vals[ev_idx], output_data[ev_idx])
    min_cons_vals[ev_idx] = remove_low_cons_ev_boxes(min_cons_vals[ev_idx], output_data[ev_idx])

    mid_cons_vals, mid_cons_vals, max_cons_vals = check_ev_output(item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals, ev_idx, samples)

    return mid_cons_vals, mid_cons_vals, max_cons_vals


def adjust_stat_app_consistency(app, vacation, month, min_cons_vals, mid_cons_vals, max_cons_vals):

    """
    Modify appliance mid/min/max ranges od statistical app to avoid inconsistency in monthly level output
    Parameters:
        vacation                  (np.ndarray)    : day wise vacation tags
        month                     (np.ndarray)    : day wise month tags
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    max_dev_allowed = 1.3

    for app_cons in [mid_cons_vals, min_cons_vals, max_cons_vals]:

        monthly_cons = np.zeros(Cgbdisagg.MONTHS_IN_YEAR)

        for i in range(Cgbdisagg.MONTHS_IN_YEAR):

            target_days = month == i

            vacation_factor = (1 - (np.sum(vacation[target_days]) / np.sum(target_days)))

            if vacation_factor != 0:
                monthly_cons[i] = np.sum(mid_cons_vals[get_index(app)][target_days]) * Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days) / vacation_factor

        if np.sum(monthly_cons > 0) <= 0:
            continue

        median_cons = np.median(monthly_cons[monthly_cons > 0])

        high_cons_month = monthly_cons > median_cons * max_dev_allowed
        low_cons_month = monthly_cons < median_cons * (1 / max_dev_allowed)

        # reduce stat app in high cons billing cycles

        if np.any(high_cons_month):
            high_cons_month_list = np.where(high_cons_month)[0]

            for i in high_cons_month_list:
                target_days = month == i
                factor = (monthly_cons[i] / median_cons) / max_dev_allowed
                app_cons[get_index(app)][target_days] = app_cons[get_index(app)][target_days] / factor

        # increase stat app in high cons billing cycles

        if np.any(low_cons_month):
            high_cons_month_list = np.where(low_cons_month)[0]

            for i in high_cons_month_list:
                target_days = month == i
                factor = (monthly_cons[i] / median_cons) / max_dev_allowed
                app_cons[get_index(app)][target_days] = app_cons[get_index(app)][target_days] * factor

    return min_cons_vals, mid_cons_vals, max_cons_vals


def check_ev_output(item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals, ev_idx, samples):

    """
    Modify appliance mid/min/max ranges od statistical app to avoid inconsistency in monthly level output
    Parameters:
        vacation                  (np.ndarray)    : day wise vacation tags
        month                     (np.ndarray)    : day wise month tags
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    if np.sum(mid_cons_vals[ev_idx]) > 0 and samples > 1:

        ev_usage_arr = mid_cons_vals[ev_idx].flatten()

        thres = min(1.25*samples, 4)

        seq = find_seq(ev_usage_arr > 0, np.zeros_like(ev_usage_arr), np.zeros_like(ev_usage_arr))

        for i in range(len(seq)):

            if (seq[i, 3] < thres) and seq[i, 0]:
                ev_usage_arr[int(seq[i, 1]): int(seq[i, 2])+1] = 0

        ev_usage_arr = ev_usage_arr.reshape(mid_cons_vals[ev_idx].shape)

        ev_res = item_output_object.get("ev_residual")

        ev_usage_arr[ev_res > 0] = 1

        mid_cons_vals[ev_idx][ev_usage_arr == 0] = 0
        min_cons_vals[ev_idx][ev_usage_arr == 0] = 0
        max_cons_vals[ev_idx][ev_usage_arr == 0] = 0

    if np.sum(mid_cons_vals[ev_idx]) > 0 and samples > 1:
        ev_usage_arr = mid_cons_vals[ev_idx].flatten()

        thres = int(0.5 * samples) + 1

        seq = find_seq(ev_usage_arr > 0, np.zeros_like(ev_usage_arr), np.zeros_like(ev_usage_arr))

        for i in range(len(seq)):

            if (seq[i, 3] < thres) and seq[i, 0]:
                ev_usage_arr[int(seq[i, 1]): int(seq[i, 2]) + 1] = 0

        ev_usage_arr = ev_usage_arr.reshape(mid_cons_vals[ev_idx].shape)

        mid_cons_vals[ev_idx][ev_usage_arr == 0] = 0
        min_cons_vals[ev_idx][ev_usage_arr == 0] = 0
        max_cons_vals[ev_idx][ev_usage_arr == 0] = 0

    return min_cons_vals, mid_cons_vals, max_cons_vals


def add_daily_baseload_to_ent(vacation, min_cons_vals, mid_cons_vals, max_cons_vals, item_input_object,
                              item_output_object, negative_residual, input_data):
    """
    Modify appliance mid/min/max ENT ranges to add perenial baseload type consumption from disagg residual

    Parameters:
        vacation
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : TS level max cons vals of all app
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        negative_residual         (np.ndarray)    : tou where disagg residual is negative
        input_data                (np.ndarray)    : user input data

    Returns:
        mid_cons_vals             (np.ndarray)    : updated TS level mid cons vals of all app
        min_cons_vals             (np.ndarray)    : updated TS level min cons vals of all app
        max_cons_vals             (np.ndarray)    : updated TS level max cons vals of all app
    """

    pp_idx = get_index('pp')
    ev_idx = get_index('ev')
    wh_idx = get_index('wh')
    ent_idx = get_index('ent')
    cook_idx = get_index('cook')
    ld_idx = get_index('ld')

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    stat_idx = hybrid_config.get("app_seq").index('ent')
    ent_avg_cons_flag = hybrid_config.get("have_mid_cons")[stat_idx]
    ent_avg_cons = hybrid_config.get("mid_cons")[stat_idx]

    stat_idx = hybrid_config.get("app_seq").index('cook')
    cook_avg_cons_flag = hybrid_config.get("have_mid_cons")[stat_idx]
    cook_avg_cons = hybrid_config.get("mid_cons")[stat_idx]

    days_in_month = Cgbdisagg.DAYS_IN_MONTH

    ent_multiplier = 1

    if ent_avg_cons_flag and cook_avg_cons_flag:
        factor = min(1, (ent_avg_cons / cook_avg_cons))

        ent_multiplier = 1

        if factor < 0.9:
            ent_multiplier = 2
        if factor < 0.75:
            ent_multiplier = 4

    original_ent_tou = copy.deepcopy(mid_cons_vals[ent_idx].sum(axis=0) > 0)

    if np.any(np.logical_not(vacation)) and (item_input_object.get('config').get('disagg_mode') != 'mtd') and \
            (not np.all(negative_residual == 0)):

        # consumption without high confidence signature appliances

        input_data_copy = copy.deepcopy(np.fmax(0, input_data - mid_cons_vals[get_index('ao')] -
                                                mid_cons_vals[get_index('ref')] -
                                                mid_cons_vals[wh_idx] -
                                                mid_cons_vals[pp_idx] -
                                                mid_cons_vals[ev_idx]))

        input_data_copy = input_data_copy[np.logical_not(vacation)]

        baseload_typ_sig = np.zeros_like(input_data_copy)

        for i in range(0, len(input_data_copy), days_in_month):
            baseload_typ_sig[i:i + days_in_month] = np.percentile(input_data_copy[i:i + days_in_month], 20, axis=0)[None, :]

        baseload_typ_sig = np.percentile(baseload_typ_sig, 20, axis=0)

        samples_per_hour = int(input_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

        sleep_hours = copy.deepcopy(item_output_object.get("profile_attributes").get("sleep_hours"))

        # default sleep hours

        if np.all(sleep_hours == 0):
            sleep_hours = np.ones_like(sleep_hours)
            sleep_hours[np.arange(0, 6 * samples_per_hour)] = 0

        baseload_typ_sig = baseload_typ_sig - np.percentile(baseload_typ_sig[np.logical_not(sleep_hours)], 30)

        baseload_typ_sig[np.logical_not(sleep_hours)] = 0

        baseload_typ_sig = np.fmax(0, baseload_typ_sig)

        base_ent_cons_2d = np.zeros_like(input_data)

        config = get_residual_config(samples_per_hour).get('ent_cons')

        baseload_cons_max_cap = config.get('baseload_cons_max_cap')
        baseload_cons_lower_cap = config.get('baseload_cons_lower_cap')
        min_cap_for_baseload_cons = config.get('min_cap_for_baseload_cons')
        baseload_cons_scaling_factor = config.get('baseload_cons_scaling_factor')
        activity_curve_thres = config.get('activity_curve_thres')

        low_baseload_cons = baseload_typ_sig < baseload_cons_lower_cap[1]
        low_baseload_cons2 = baseload_typ_sig < baseload_cons_lower_cap[0]

        if np.any(low_baseload_cons) and np.max(baseload_typ_sig) < baseload_cons_max_cap[0]:
            base_ent_cons_2d[:, low_baseload_cons] = baseload_typ_sig[low_baseload_cons][ None, :]
        elif np.any(low_baseload_cons2) and np.max(baseload_typ_sig) < baseload_cons_max_cap[1]:
            base_ent_cons_2d[:, low_baseload_cons2] = baseload_typ_sig[low_baseload_cons2][None, :]

        base_ent_cons_2d[base_ent_cons_2d < min_cap_for_baseload_cons / samples_per_hour] = 0
        base_ent_cons_2d = base_ent_cons_2d * baseload_cons_scaling_factor

        val = 300

        base_ent_cons_2d = np.fmin(val / (ent_multiplier * samples_per_hour), base_ent_cons_2d)

        weekend_block = copy.deepcopy(base_ent_cons_2d)

        dow = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_DOW_IDX, :, 0]
        activity_curve = item_input_object.get("weekday_activity_curve")
        activity_curve = (activity_curve - np.percentile(activity_curve, 3)) / (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))
        weekend_days = np.logical_or(dow == 1, dow == 7)

        weekend_block[:, activity_curve < activity_curve_thres] = 0
        weekend_block[np.logical_not(weekend_days)] = 0

        weekday_block = copy.deepcopy(base_ent_cons_2d)

        activity_curve = item_input_object.get("weekend_activity_curve")
        activity_curve = (activity_curve - np.percentile(activity_curve, 3)) / (np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

        weekday_block[:, activity_curve < activity_curve_thres] = 0
        weekday_block[weekend_days] = 0

        base_ent_cons_2d = weekend_block + weekday_block

        diff = np.fmax(0, weekend_block + weekday_block)

        min_cons_vals[cook_idx][base_ent_cons_2d > 0] = \
            np.minimum(min_cons_vals[cook_idx], np.fmax(0, min_cons_vals[cook_idx] - diff))[base_ent_cons_2d > 0]
        mid_cons_vals[cook_idx][base_ent_cons_2d > 0] = \
            np.minimum(mid_cons_vals[cook_idx], np.fmax(0, mid_cons_vals[cook_idx] - diff))[base_ent_cons_2d > 0]
        max_cons_vals[cook_idx][base_ent_cons_2d > 0] = \
            np.minimum(max_cons_vals[cook_idx], np.fmax(0, max_cons_vals[cook_idx] - diff))[base_ent_cons_2d > 0]

        min_cons_vals[ld_idx][base_ent_cons_2d > 0] = \
            np.minimum(min_cons_vals[ld_idx], np.fmax(0, min_cons_vals[ld_idx] - diff))[base_ent_cons_2d > 0]
        max_cons_vals[ld_idx][base_ent_cons_2d > 0] = \
            np.minimum(max_cons_vals[ld_idx], np.fmax(0, max_cons_vals[ld_idx] - diff))[base_ent_cons_2d > 0]
        mid_cons_vals[ld_idx][base_ent_cons_2d > 0] = \
            np.minimum(mid_cons_vals[ld_idx], np.fmax(0, mid_cons_vals[ld_idx] - diff))[base_ent_cons_2d > 0]

        min_cons_vals[ent_idx][base_ent_cons_2d > 0] = \
            np.maximum(min_cons_vals[ent_idx], base_ent_cons_2d)[base_ent_cons_2d > 0]
        max_cons_vals[ent_idx][base_ent_cons_2d > 0] = \
            np.maximum(max_cons_vals[ent_idx], base_ent_cons_2d)[base_ent_cons_2d > 0]
        mid_cons_vals[ent_idx][base_ent_cons_2d > 0] = \
            np.maximum(mid_cons_vals[ent_idx], base_ent_cons_2d)[base_ent_cons_2d > 0]

        mid_cons_vals[ent_idx][:, np.logical_not(original_ent_tou)] = 0
        min_cons_vals[ent_idx][:, np.logical_not(original_ent_tou)] = 0
        max_cons_vals[ent_idx][:, np.logical_not(original_ent_tou)] = 0

    return min_cons_vals, mid_cons_vals, max_cons_vals


def check_wh_addition_bool(item_input_object, final_tou_consumption, samples_per_hour, wh_idx):

    """
    This function performs determined whether to add leftover thin pulses in hybrid wh output

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        samples_per_hour          (int)           : samples in an hour
        appliance_list            (list)          : list of appliances

    Returns:
        add_wh                    (bool)          : True if thin pulses shud be added to wh consumption
    """

    flow_wh = 0

    if np.sum(final_tou_consumption[wh_idx]):
        flow_wh = np.percentile(final_tou_consumption[wh_idx][final_tou_consumption[wh_idx] > 0], 75) > 6000 / samples_per_hour

    pilot = item_input_object.get("config").get("pilot_id")

    add_wh = final_tou_consumption[wh_idx].sum() > 0

    valid_wh_hsm = item_input_object.get("item_input_params").get('valid_wh_hsm')

    # check whether to add wh in the mtd mode, based on hsm info

    valid_hsm_flag = check_validity_of_hsm(valid_wh_hsm, item_input_object.get("item_input_params").get('wh_hsm'), 'item_hld')


    if (item_input_object.get('config').get('disagg_mode') == 'mtd') and valid_hsm_flag:
        wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_hld')
        hld = 0

        if wh_hsm is not None and isinstance(wh_hsm, list):
            hld = wh_hsm[0]
        elif wh_hsm is not None:
            hld = wh_hsm

        wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_type')
        wh_type = 2

        if wh_hsm is not None and isinstance(wh_hsm, list):
            wh_type = wh_hsm[0]
        elif wh_hsm is not None:
            wh_type = wh_hsm

        add_wh = hld

        add_wh = add_wh and (not (wh_type == 1))

    # thin pulses shouldnt be added for twh and swh

    add_wh = add_wh and item_input_object.get('item_input_params').get('swh_hld') == 0

    add_wh = add_wh and (not item_input_object.get("item_input_params").get("timed_wh_user"))
    add_wh = add_wh and (not flow_wh) and (pilot not in PilotConstants.SEASONAL_WH_ENABLED_PILOTS)

    if item_input_object.get('item_input_params').get('tankless_wh') > 0:
        add_wh = 0

    return add_wh


def adjust_pp_cons(disagg_cons, pp_cons, item_input_object, logger):

    """
    Modify appliance mid/min/max pp ranges to maintain TS level consistency after handling of PP overestimation

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        disagg_cons               (np.ndarray)    : TS level disagg output of all app
        pp_cons                   (np.ndarray)    : TS level PP cons

    Returns:
        pp_cons                   (np.ndarray)    : TS level PP cons
    """

    samples = int(len(pp_cons[0]) / Cgbdisagg.HRS_IN_DAY)

    min_pp_amp = 400/samples
    multi_mode_amp_thres = 600/samples

    if np.all(disagg_cons < min_pp_amp):
        return pp_cons

    if not np.any(pp_cons > 0):
        return pp_cons

    # checking if the PP has multiple amplitude or is variable speed PP

    ampp_quantile_distance = np.percentile(disagg_cons[disagg_cons > 200/samples], 90) - np.percentile(disagg_cons[disagg_cons > 200/samples], 25)

    multi_amp_pp_bool = ampp_quantile_distance > multi_mode_amp_thres

    variable_pp_flag = 0

    if (item_input_object.get('created_hsm') is not None) and (
            item_input_object.get('created_hsm').get('pp') is not None) and np.sum(disagg_cons):
        variable_pp_flag = item_input_object.get('created_hsm').get('pp').get('attributes').get('run_type_code')[0] == 3

    multi_amp_pp_bool = multi_amp_pp_bool or variable_pp_flag

    copy_data = copy.deepcopy(pp_cons > 0)

    disagg_confidence = 1

    # checking PP disagg detection confidence

    if item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') is not None:
        disagg_confidence = item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') / 100

    max_perc_val = [30, 40, 50][np.digitize(disagg_confidence, [0.6, 0.7])]

    # If the user is having multiple PP amplitude from disagg estimation, the similar distribution is maintained after adjustment
    # else single amplitude is alloted to all the PP points

    if multi_amp_pp_bool:

        pp_output_1d = disagg_cons[disagg_cons > np.percentile(disagg_cons[disagg_cons > 0], 7)].flatten()

        kmeans_model = KMeans(n_clusters=2, random_state=0).fit(pp_output_1d.reshape(-1, 1))
        consumption_level_boundry = np.mean(kmeans_model.cluster_centers_)

        pp_pred_ts = np.zeros(pp_cons.shape)
        pp_pred_ts[disagg_cons > consumption_level_boundry] = 1
        pp_pred_ts[disagg_cons == 0] = -2

        if np.any(np.logical_and(np.logical_and(disagg_cons > min_pp_amp, pp_cons > 0), pp_pred_ts == 0)):
            min_val = np.percentile(pp_cons[np.logical_and(np.logical_and(disagg_cons > min_pp_amp, pp_cons > 0), pp_pred_ts == 0)], max_perc_val)
            pp_cons[pp_pred_ts == 0] = np.fmin(pp_cons[pp_pred_ts == 0], min_val)

            pp_cons[pp_pred_ts == 0] = np.fmax(pp_cons[pp_pred_ts == 0], min_val)

        if np.any(np.logical_and(np.logical_and(disagg_cons > min_pp_amp, pp_cons > 0), pp_pred_ts == 1)):
            min_val = np.percentile(pp_cons[np.logical_and(np.logical_and(disagg_cons > min_pp_amp, pp_cons > 0), pp_pred_ts == 1)], max_perc_val)
            pp_cons[pp_pred_ts == 1] = np.fmin(pp_cons[pp_pred_ts == 1], min_val)

            pp_cons[pp_pred_ts == 1] = np.fmax(pp_cons[pp_pred_ts == 1], min_val)

    else:
        if np.sum(np.logical_and(disagg_cons > min_pp_amp, pp_cons > 0)):
            min_val = np.percentile(pp_cons[np.logical_and(disagg_cons > min_pp_amp, pp_cons > 0)], max_perc_val)
            pp_cons = np.fmin(pp_cons, min_val)
            pp_cons = np.fmax(pp_cons, min_val)

    pp_cons[copy_data == 0] = 0

    pp_cons[pp_cons > 0] = np.fmax(300 / samples, pp_cons[pp_cons > 0])

    if (not item_input_object.get("item_input_params").get("pp_prof_present")) and disagg_confidence <= 0.65:
        pp_cons[:, :] = 0
        item_input_object["item_input_params"]["pp_removed"] = 1

    return pp_cons


def remove_low_cons_ev_boxes(ev_output, disagg):
    """
    Remove EV boxes that has significant decrease in estimation after adjustment of overestimation cases.

    Parameters:
        ev_output                         (np.ndarray)    : EV Hybrid v2 consumption
        disagg                            (np.ndarray)    : EV disagg output

    Returns:
        ev_hybrid_output_copy             (np.ndarray)    : EV Hybrid v2 consumption
    """

    val = ev_output.flatten()
    val2 = disagg.flatten()

    output_seq = find_seq(disagg.flatten() > 0, np.zeros_like(ev_output.flatten()), np.zeros_like(ev_output.flatten())).astype(int)

    for i in np.arange(len(output_seq)):
        if output_seq[i, 0] > 0 and np.sum(val2[output_seq[i, 1]: output_seq[i, 2]+1]) > 0:
            factor = np.sum(val[output_seq[i, 1]: output_seq[i, 2]+1]) / np.sum(val2[output_seq[i, 1]: output_seq[i, 2]+1])

            if factor < 0.5:
                val[output_seq[i, 1]: output_seq[i, 2]+1] = 0
            else:
                val[output_seq[i, 1]: output_seq[i, 2] + 1] = \
                    np.fmin(np.median(val2[output_seq[i, 1]: output_seq[i, 2] + 1]),
                            val[output_seq[i, 1]: output_seq[i, 2] + 1])
    ev_output = val.reshape(disagg.shape)

    return ev_output


def allot_thin_pulse_boxes(item_input_object, item_output_object, mid_cons_vals, wh_idx, disagg_output, residual_data, logger):

    """
    This function performs postprocessing on itemized wh output to add leftover thin pulses

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
        wh_idx                    (int)           : WH index
        disagg_output             (np.ndarray)    : disagg output
        residual_data             (np.ndarray)    : disagg residual
        logger                    (logger)        : logger info

    Returns:
        mid_cons_vals             (np.ndarray)    : TS level mid cons vals of all app
    """

    samples_per_hour = int(mid_cons_vals.shape[2] / Cgbdisagg.HRS_IN_DAY)

    processed_input_data = item_output_object.get('original_input_data')

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(mid_cons_vals), axis=0)

    # check whether to add thin pulses based on type of wh

    add_wh = check_wh_addition_bool(item_input_object, mid_cons_vals, samples_per_hour, wh_idx)

    if not add_wh:
        logger.info('Not adding leftover thin pulse | ')
        return np.zeros_like(mid_cons_vals[0])

    box_label, box_cons, box_seq = \
        box_detection(0, residual_data, np.fmax(0, residual_data), np.zeros_like(residual_data),
                      min_amp=300 / samples_per_hour, max_amp=20000 / samples_per_hour, min_len=1,
                      max_len=4 * samples_per_hour, detect_wh=1)
    box_seq = box_seq.astype(int)

    config = get_inf_config().get('wh')

    thin_pulse_amp_max_thres = config.get('thin_pulse_amp_max_thres')
    thin_pulse_amp_min_thres = config.get('thin_pulse_amp_min_thres')
    thin_pulse_amp_max_ts_cons = config.get('thin_pulse_amp_max_ts_cons')
    thin_pulse_amp_buffer = config.get('thin_pulse_amp_buffer')
    thin_pulse_max_amp_factor = config.get('thin_pulse_max_amp_factor')
    max_thin_pulse_in_day = config.get('max_thin_pulse_in_day')

    max_amp = thin_pulse_amp_max_thres / samples_per_hour
    min_amp = thin_pulse_amp_min_thres / samples_per_hour
    val = thin_pulse_amp_max_ts_cons

    # determining amplitude based in disagg thin pulse box consumption

    if (item_input_object.get("item_input_params").get("final_thin_pulse") is not None) \
            and (np.sum(mid_cons_vals[wh_idx]) > 0)\
            and (np.sum(disagg_output) > 0)\
            and (item_input_object.get("item_input_params").get("final_thin_pulse").sum() > 0):

        thin_pulse = item_input_object.get("item_input_params").get("final_thin_pulse")
        thin_pulse_tou = thin_pulse > 0

        if np.sum(thin_pulse_tou) > 0:
            max_amp = np.max(thin_pulse[thin_pulse_tou]) * thin_pulse_max_amp_factor
            min_amp = np.median(thin_pulse[thin_pulse_tou]) - thin_pulse_amp_buffer/samples_per_hour

            val = max_amp

            if np.isnan(val):
                val = thin_pulse_amp_max_ts_cons
                max_amp = thin_pulse_amp_max_thres / samples_per_hour
                min_amp = thin_pulse_amp_min_thres / samples_per_hour
        else:
            val = thin_pulse_amp_max_ts_cons
            max_amp = thin_pulse_amp_max_thres / samples_per_hour
            min_amp = thin_pulse_amp_min_thres / samples_per_hour

    # determining valid thin pulse boxes

    valid_boxes = np.logical_and(box_seq[:, 0] == 1,
                                 np.logical_and(box_seq[:, 3] <= max(1, 0.5 * samples_per_hour),
                                                np.logical_and(box_seq[:, 4] >= min_amp, box_seq[:, 4] <= max_amp)))

    valid_idx = np.zeros(np.size(mid_cons_vals[0]))

    valid_idx = fill_arr_based_seq_val_for_valid_boxes(box_seq, valid_boxes, valid_idx, 1, 1)

    # picking boxes that are more likely be a thin pulse compoenent of WH

    valid_idx = np.reshape(valid_idx, mid_cons_vals[0].shape)
    valid_idx[valid_idx > 0] = valid_idx[valid_idx > 0] + other_cons_arr[valid_idx > 0]

    valid_idx = np.fmin(valid_idx, val)

    valid_idx[valid_idx < min_amp] = 0

    season = item_output_object.get("season")

    reduce_cons = np.mean(season) > 0 and item_input_object.get('config').get('disagg_mode') == 'mtd'

    valid_idx = valid_idx * (0.6 * (reduce_cons) + 1 * (not reduce_cons))

    valid_idx[mid_cons_vals[wh_idx] > 0] = 0

    if item_input_object.get('item_input_params').get('swh_hld') == 0:
        item_input_object["item_input_params"]["hybrid_thin_pulse"] = valid_idx

    # step to make sure that the thin pulse count is not more than certain limit for a given day

    valid_idx_tou = valid_idx > 0

    if np.sum(valid_idx) > 0:
        for i in range(len(valid_idx)):
            if np.sum(valid_idx_tou[i]) > max_thin_pulse_in_day:
                extra_thin_pulses = int(np.sum(valid_idx_tou[i])/(np.sum(valid_idx_tou[i]) - max_thin_pulse_in_day))
                thin_pulse_idx = (np.where(valid_idx_tou[i] > 0))[0]
                thin_pulse_idx = thin_pulse_idx[np.arange(0, len(thin_pulse_idx), extra_thin_pulses)]

                valid_idx_tou[i][thin_pulse_idx] = 0

    valid_idx[valid_idx_tou == 0] = 0

    return valid_idx


def block_sparse_wh_output_users_added_from_hybrid(mid_cons, disagg_cons, item_input_object):

    """
    Modify appliance  WH mid/min/max ranges by blocking sparse WH output cases

    Parameters:
        mid_cons                  (np.ndarray)    : WH hybrid v2 consumption
        disagg_cons               (np.ndarray)    : WH disagg  consumption
        item_input_object         (dict)          : Dict containing all hybrid inputs

    Returns:
        mid_cons                  (np.ndarray)    : udpated WH hybrid v2 consumption

    """

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    blocking_list = np.ones_like(unique_bc)

    samples = int(mid_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)

    min_cons = 5000
    min_points = 5 * samples

    # check whether feeble WH consumption is detected in a particular billing cycle

    for i in range(len(unique_bc)):

        arr = ((mid_cons + disagg_cons)[bc_list == unique_bc[i]]).sum()

        factor = 1

        if (arr < min_cons * factor) or (
                np.sum(((mid_cons + disagg_cons)[bc_list == unique_bc[i]]) > 0) < min_points * factor):
            blocking_list[i] = 2

    # all the billing cycles has less ev count
    if (np.sum(blocking_list == 2) == 0) or item_input_object.get("config").get('disagg_mode') == 'mtd':
        return mid_cons

    if np.all(blocking_list == 2):
        return np.zeros_like(mid_cons)

    wh_days_seq = find_seq(blocking_list, np.zeros_like(blocking_list), np.zeros_like(blocking_list), overnight=0)

    wh_days_seq = wh_days_seq.astype(int)

    # If a feeble wh consumption billing cycle is detected between 2 normal billing cycle, the feeble consumption billing cycle is ignored

    for i in range(len(wh_days_seq)):
        if wh_days_seq[i, seq_label] == 2 and wh_days_seq[i, seq_len] < 2:
            blocking_list[wh_days_seq[i, seq_start]: wh_days_seq[i, seq_end] + 1] = 1

    wh_days_seq = find_seq(blocking_list, np.zeros_like(blocking_list), np.zeros_like(blocking_list), overnight=0)
    wh_days_seq = wh_days_seq.astype(int)

    pilot = item_input_object.get("config").get("pilot_id")

    wh_box_count_thres = 4

    if item_input_object.get('item_input_params').get('tankless_wh') > 0:
        wh_box_count_thres = 6

    # If continuously high number of feeble consumption billing cycles are detected, WH is blocked for the user

    if (pilot not in PilotConstants.INDIAN_PILOTS) and np.sum(blocking_list == 2) > wh_box_count_thres:
        mid_cons[:] = 0

    if np.sum(wh_days_seq[:, 0] == 2) > 2:
        mid_cons[:] = 0

    return mid_cons
