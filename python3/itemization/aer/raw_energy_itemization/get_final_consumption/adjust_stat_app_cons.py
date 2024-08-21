

"""
Author - Nisha Agarwal
Date - 4th April 2021
this file contains function to maintain min/max bc level cons of stat app
"""

# Import python packages

import copy
import numpy as np
from numpy.random import RandomState

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.itemization.init_itemization_config import random_gen_config
from python3.itemization.aer.functions.get_config import get_hybrid_config
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.init_final_item_config import init_final_item_conf

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import get_app_idx
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import add_randomness
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import prepare_stat_tou
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import get_app_cons_in_target_bc
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import get_stat_app_min_cons_hard_limit
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import prepare_ts_level_additional_cons_points
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import add_prepare_box_type_additional_cons_points
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import add_step3_cons_into_stat_to_maintain_stability
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import prepare_additional_cons_points_to_satisfy_min_cons

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_apply_bc_level_limit_on_stat_app import apply_bc_level_min_limit_using_hvac_cons
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_apply_bc_level_limit_on_stat_app import bc_level_min_limit_using_step3_app


def apply_bc_level_min_limit_for_each_bc(final_tou_consumption, item_input_object, item_output_object, input_dict_for_min_cons):
    """
    maintain bc level level consistency of ld, cook, and ent appliances by picking cons from residual

    Parameters:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        input_dict_for_min_cons     (dict)          : additional inputs required for maintain bc level min cons

    Returns:
        final_tou_consumption       (np.ndarray)    : updated final ts level estimation of appliances
        app_cons                    (int)           : current bc level cons of the appliance
        extra_hvac_cons             (np.ndarray)    : amount of additional hvac app that can be partially alloted
                                                      to stat app to maintain consistency
    """
    inactive_hours = input_dict_for_min_cons.get('inactive_hours')
    app = input_dict_for_min_cons.get('app')
    max_cons = input_dict_for_min_cons.get('max_cons')
    target_days = input_dict_for_min_cons.get('target_days')
    stat_tou = input_dict_for_min_cons.get('stat_tou')
    app_cons = input_dict_for_min_cons.get('app_cons')
    hard_limit = input_dict_for_min_cons.get('hard_limit')
    extra_hvac_cons = input_dict_for_min_cons.get('extra_hvac_cons')

    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    app_idx = get_app_idx(appliance_list, app)

    if (app_cons < hard_limit) and (app_cons > 0):

        # picking required consumption from residual data
        # this consumption is picked in a box type signature

        additional_cons = np.zeros_like(processed_input_data[target_days])

        other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)
        other_cons_arr = np.fmax(0, other_cons_arr)
        additional_cons[:, stat_tou] = np.fmin(max_cons, other_cons_arr[target_days][:, stat_tou]) / (
            np.max(np.fmin(max_cons, other_cons_arr[target_days][:, stat_tou])))
        additional_cons = np.nan_to_num(additional_cons)

        additional_cons = add_randomness(additional_cons)
        additional_cons[vacation[target_days]] = 0

        other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

        cons_limit = 500 / samples_per_hour

        if np.any(final_tou_consumption[app_idx] > 0):
            cons_limit = np.percentile(final_tou_consumption[app_idx][final_tou_consumption[app_idx] > 0], 80)

        # Calculating epoch level consumption required to be added to maintain a minimum consumption
        # or maintaining stability in bc level consumption

        additional_cons[:, stat_tou] = other_cons_arr[target_days][:, stat_tou]

        additional_cons[other_cons_arr[target_days] < cons_limit] = 0

        if np.sum(np.nan_to_num(additional_cons)) > 0:
            seed = RandomState(random_gen_config.seed_value)
            diff = (hard_limit - app_cons) * Cgbdisagg.WH_IN_1_KWH * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)

            # checking consumption that can be picked up from hvac

            additional_cons = add_prepare_box_type_additional_cons_points(diff, final_tou_consumption, additional_cons,
                                                                          vacation, target_days, app_idx, seed)

            final_tou_consumption[app_idx, target_days, :] = final_tou_consumption[app_idx, target_days, :] + \
                                                             np.fmax(0, np.minimum(other_cons_arr[target_days, :] * 0.99,
                                                                                   additional_cons).astype(int))

            final_tou_consumption[app_idx, target_days, :] = np.minimum(final_tou_consumption[app_idx, target_days, :],
                                                                        processed_input_data[target_days])
            final_tou_consumption[app_idx, target_days, :] = np.fmin(final_tou_consumption[app_idx, target_days, :], max_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

    if (app_cons < hard_limit) and (app_cons >= 0):

        additional_cons = np.zeros_like(processed_input_data[target_days])

        active_hours = np.ones_like(stat_tou)
        active_hours[inactive_hours] = 0

        if app in ['cook', 'ld']:
            active_hours = stat_tou

        # picking required consumption from extra hvac cons
        # this consumption is picked in a box type signature
        # this extra cons is the additional cons that is added to hvac during 100% itemization,
        # but can be partially alloted to stat app to maintain consistency at billing cycle level

        additional_cons[:, active_hours] = np.fmin(max_cons, extra_hvac_cons[target_days][:, active_hours]) / \
                                           np.max(np.fmin(max_cons, extra_hvac_cons[target_days][:, active_hours]))
        additional_cons = np.nan_to_num(additional_cons)
        additional_cons = add_randomness(additional_cons)
        additional_cons[vacation[target_days]] = 0

        additional_cons[:, active_hours] = extra_hvac_cons[target_days][:, active_hours]

        cons_limit = 500 / samples_per_hour

        if np.any(final_tou_consumption[app_idx] > 0):
            cons_limit = np.percentile(final_tou_consumption[app_idx][final_tou_consumption[app_idx] > 0], 80)

        additional_cons[additional_cons < cons_limit] = 0

        # Calculating epoch level consumption required to be added to maintain a minimum consumption
        # or maintaining stability in bc level consumption

        if np.sum(np.nan_to_num(additional_cons)) > 0:
            seed = RandomState(random_gen_config.seed_value)
            diff = (hard_limit - app_cons) * Cgbdisagg.WH_IN_1_KWH * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)
            additional_cons = add_prepare_box_type_additional_cons_points(diff, final_tou_consumption, additional_cons,
                                                                          vacation, target_days, app_idx, seed)

            other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

            final_tou_consumption[app_idx, target_days, :] = \
                final_tou_consumption[app_idx, target_days, :] + \
                np.fmax(0, np.minimum(other_cons_arr[target_days], np.minimum(extra_hvac_cons[target_days], additional_cons)))

            other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)
            final_tou_consumption[app_idx, target_days, :] = np.minimum(final_tou_consumption[app_idx, target_days, :],
                                                                        processed_input_data[target_days])

            final_tou_consumption[app_idx, target_days, :] = np.fmin(final_tou_consumption[app_idx, target_days, :], max_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

            extra_hvac_cons[target_days] = \
                extra_hvac_cons[target_days] - np.minimum(other_cons_arr[target_days], np.minimum(extra_hvac_cons[target_days], additional_cons))

    if (app_cons < hard_limit) and (app_cons > 0):

        additional_cons = np.zeros_like(processed_input_data[target_days])

        # picking required consumption from residual data

        other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)
        other_cons_arr = np.fmax(0, other_cons_arr)
        additional_cons[:, stat_tou] = np.fmin(max_cons, other_cons_arr[target_days][:, stat_tou]) / (
            np.max(np.fmin(max_cons, other_cons_arr[target_days][:, stat_tou])))
        additional_cons = np.nan_to_num(additional_cons)

        additional_cons = add_randomness(additional_cons)
        additional_cons[vacation[target_days]] = 0

        # Calculating epoch level consumption required to be added to maintain a minimum consumption
        # or maintaining stability in bc level consumption

        if np.sum(np.nan_to_num(additional_cons)) > 0:
            additional_cons = prepare_additional_cons_points_to_satisfy_min_cons(additional_cons, hard_limit, target_days, app_cons)

            other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

            final_tou_consumption[app_idx, target_days, :] = \
                final_tou_consumption[app_idx, target_days, :] + \
                np.fmax(0, np.minimum(other_cons_arr[target_days, :] * 0.99, additional_cons).astype(int))

            final_tou_consumption[app_idx, target_days, :] = \
                np.minimum(final_tou_consumption[app_idx, target_days, :], processed_input_data[target_days])

            final_tou_consumption[app_idx, target_days, :] = \
                np.fmin(final_tou_consumption[app_idx, target_days, :], max_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

    return final_tou_consumption, app_cons, extra_hvac_cons


def apply_bc_level_min_limit(total_cons, item_input_object, item_output_object, final_tou_consumption, length,
                             monthly_app_cons, app_hsm_data, use_hsm=True):

    """
    Limit min bc level level consumption increase for ld, cook, and ent appliances

    Parameters:
        total_cons                  (float)         : total consumption
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        length                      (int)           : total non vacation days of the user
        monthly_app_cons            (list)          : monthly stat app cons
        app_hsm_data                (list)          : monthly stat app consumption based on HSM
        use_hsm                     (bool)          : This flag denote if stat app hsm can be used

    Returns:
        final_tou_consumption       (np.ndarray)    : updated final ts level estimation of appliances
        ld_monthly_cons             (float)         : monthly laundry consumption
        ent_monthly_cons            (float)         : monthly ent consumption
        cook_monthly_cons           (float)         : monthly cooking consumption
    """
    ld_monthly_cons = monthly_app_cons[0]
    ent_monthly_cons = monthly_app_cons[1]
    cook_monthly_cons = monthly_app_cons[2]
    hsm_cook = app_hsm_data[0]
    hsm_ent = app_hsm_data[1]
    hsm_ld = app_hsm_data[2]

    # fetching required inputs

    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')
    scaling_factor = Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH

    day_start = 6
    multipling_factor = (Cgbdisagg.HRS_IN_DAY / (Cgbdisagg.HRS_IN_DAY - day_start))

    total = (processed_input_data[np.logical_not(vacation)][day_start * samples_per_hour:]) * multipling_factor
    total = ((np.sum(total) / len(processed_input_data)) * (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH))

    cool_idx = np.where(appliance_list == "cooling")[0][0]
    heat_idx = np.where(appliance_list == "heating")[0][0]

    cooling_limit = item_input_object.get('pilot_level_config').get('cooling_config').get('bounds').get('min_cons')
    heating_limit = item_input_object.get('pilot_level_config').get('heating_config').get('bounds').get('min_cons')

    residual = processed_input_data - final_tou_consumption[np.where(appliance_list == "ao")[0][0]] - \
               final_tou_consumption[np.where(appliance_list == "ref")[0][0]] - \
               final_tou_consumption[np.where(appliance_list == "li")[0][0]] - \
               final_tou_consumption[heat_idx] - \
               final_tou_consumption[cool_idx] - \
               final_tou_consumption[np.where(appliance_list == "pp")[0][0]] - \
               final_tou_consumption[np.where(appliance_list == "wh")[0][0]] - \
               final_tou_consumption[np.where(appliance_list == "ev")[0][0]]

    for backup_app in item_input_object["item_input_params"]["backup_app"]:
        residual = residual + final_tou_consumption[np.where(appliance_list == backup_app)[0][0]]

    day_start = 6
    original_total = 0

    if np.any([np.logical_not(vacation)]):
        res = np.fmax(0, residual)
        day_total_cons = (res[np.logical_not(vacation)][:, day_start * samples_per_hour:]) * 24 / 18
        original_total = ((np.sum(day_total_cons) / len(res[np.logical_not(vacation)])) * (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH))

    # intializing minimum consumption required for stat app cons

    config = init_final_item_conf(total, total_cons).get('min_max_limit_conf')

    ent_limit =  config.get('ent_limit')
    cook_limit =  config.get('cook_limit')
    ld_limit = config.get('ld_limit')

    inactive_hours = get_index_array(item_output_object['profile_attributes']['sleep_time'] * samples_per_hour + samples_per_hour,
                                     item_output_object['profile_attributes']['wakeup_time'] * samples_per_hour - samples_per_hour,
                                     Cgbdisagg.HRS_IN_DAY * samples_per_hour).astype(int)

    tou_list = [np.zeros(final_tou_consumption.shape[2]), np.zeros(final_tou_consumption.shape[2]), np.zeros(final_tou_consumption.shape[2])]

    stat_app_array = [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons]
    dev_qa_idx_arr = [0, 2, 1]
    stat_app_list = ['ld', 'cook', 'ent']
    cons_limit_arr = [ld_limit, cook_limit, ent_limit]

    disagg_cons = final_tou_consumption[cool_idx] + final_tou_consumption[heat_idx]

    disagg_cons = disagg_cons - item_input_object.get("item_input_params").get("ao_cool")
    disagg_cons = disagg_cons - item_input_object.get("item_input_params").get("ao_heat")

    extra_hvac_cons = np.fmax(0, final_tou_consumption[cool_idx] + final_tou_consumption[heat_idx] -
                              item_output_object.get('inference_engine_dict').get('output_data')[cool_idx] -
                              item_output_object.get('inference_engine_dict').get('output_data')[heat_idx])

    disagg_cons = np.fmax(0, disagg_cons)

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    cons_limit_arr = get_stat_app_min_cons(item_input_object, cons_limit_arr)

    # preparing min cons based on hsm data

    if use_hsm:
        cons_limit_arr[0] = max(cons_limit_arr[0], hsm_ld)
        cons_limit_arr[1] = max(cons_limit_arr[1], hsm_cook)
        cons_limit_arr[2] = max(cons_limit_arr[2], hsm_ent)

    hvac_cons_val = np.zeros_like(disagg_cons)

    sleep_hours = copy.deepcopy(item_output_object.get("profile_attributes").get("sleep_hours"))

    if np.all(sleep_hours == 0):
        sleep_hours = np.ones_like(sleep_hours)
        sleep_hours[np.arange(1, 5 * samples_per_hour+1)] = 0

    hsm_cons = [hsm_ld, hsm_cook, hsm_ent]

    unique_bc, days_count = np.unique(bc_list, return_counts=1)
    unique_bc = unique_bc[days_count >= 5]
    unique_bc = unique_bc[unique_bc > 0]

    day_start = 6
    multipling_factor = (Cgbdisagg.HRS_IN_DAY / (Cgbdisagg.HRS_IN_DAY - day_start))

    # functionality to maintain min cons for each stat app for each billing cycle

    for j, app in enumerate(stat_app_list):

        app_idx = get_app_idx(appliance_list, app)

        stat_tou = prepare_stat_tou(final_tou_consumption, app_idx, use_hsm, hsm_cons[j], app,
                                    hybrid_config, sleep_hours)

        if not np.any(stat_tou):
            continue

        tou_list[j] = stat_tou

        max_cons = 0

        if np.sum(final_tou_consumption[app_idx] > 0) > 0:
            max_cons = np.percentile(final_tou_consumption[app_idx][final_tou_consumption[app_idx] > 0], 100)

        for current_bc in unique_bc:

            target_days = bc_list == current_bc

            if np.sum(final_tou_consumption[cool_idx]) < cooling_limit * 1.1:
                disagg_cons[target_days] = disagg_cons[target_days] - final_tou_consumption[cool_idx][target_days]

            if np.sum(final_tou_consumption[heat_idx]) < heating_limit * 1.1:
                disagg_cons[target_days] = disagg_cons[target_days] - final_tou_consumption[heat_idx][target_days]

            disagg_cons = np.fmax(0, disagg_cons)

            total = (processed_input_data[np.logical_not(vacation)][bc_list[np.logical_not(vacation)] == current_bc][:,day_start * samples_per_hour:]) * multipling_factor
            total = ((np.sum(total) / len(processed_input_data[target_days])) * (scaling_factor))
            ld_limit = min(cons_limit_arr[j], [8, 8, 15, 20, 25, 30, 40, 50, 70, 80][np.digitize(total, [300, 400, 700, 1000, 1500, 2000, 3000, 4000, 6000])])

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

            # calculating minimum amount of consumption to be present for target appliance for given billing cycle

            params = {
                'hsm_cook': hsm_cook,
                'use_hsm': use_hsm,
                'target_days': target_days,
                'cook_monthly_cons': cook_monthly_cons,
                'app_cons': app_cons,
                'total_cons': total_cons,
                'app': app,
                'initialized_min_cons': ld_limit
            }

            hard_limit, app_cons = \
                get_stat_app_min_cons_hard_limit(item_input_object, item_output_object,[hsm_ld, hsm_ent, hsm_cook],
                                                 original_total, [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons], current_bc, params)

            # picking consumption from residual

            params = {
                'hard_limit': hard_limit,
                'target_days': target_days,
                'stat_tou': stat_tou,
                'app_cons': app_cons,
                'max_cons': max_cons,
                'app': app,
                'inactive_hours': inactive_hours,
                'extra_hvac_cons': extra_hvac_cons
            }

            final_tou_consumption, app_cons, extra_hvac_cons = \
                apply_bc_level_min_limit_for_each_bc(final_tou_consumption, item_input_object, item_output_object, params)

            # picking consumption from hvac

            params = {
                'hard_limit': hard_limit,
                'target_days': target_days,
                'stat_tou': stat_tou,
                'app_cons': app_cons,
                'max_cons': max_cons,
                'app': app,
                'inactive_hours': inactive_hours,
                'disagg_cons': disagg_cons,
                'hvac_cons_val': hvac_cons_val
            }

            final_tou_consumption, app_cons, disagg_cons, hvac_cons_val = \
                apply_bc_level_min_limit_using_hvac_cons(final_tou_consumption, item_input_object, item_output_object,
                                                         params)
            # picking consumption from step 3

            params = {
                'hard_limit': hard_limit,
                'target_days': target_days,
                'stat_tou': stat_tou,
                'app_cons': app_cons,
                'max_cons': max_cons,
                'app': app,
                'inactive_hours': inactive_hours
            }

            final_tou_consumption = \
                bc_level_min_limit_using_step3_app(final_tou_consumption, item_input_object, item_output_object, params)

        final_tou_consumption[app_idx] = np.fmax(0, final_tou_consumption[app_idx])
        cons = final_tou_consumption[app_idx][np.logical_not(vacation)]
        stat_app_array[dev_qa_idx_arr[j]] = ((np.sum(cons) / length) * (scaling_factor))

    return final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons, tou_list


def get_stat_app_min_cons(item_input_object, cons_limit_arr):
    """
    Limit min bc level level consumption increase for ld, cook, and ent appliances

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        cons_limit_arr              (list)          : array containing list of min cons for stat app

    Returns:
        cons_limit_arr              (list)          : array containing list of updated min cons for stat app
    """
    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    ld_idx = hybrid_config.get("app_seq").index('ld')
    have_min_cons = hybrid_config.get("have_hard_min_lim")[ld_idx]
    min_cons = hybrid_config.get("hard_min_lim")[ld_idx]

    if have_min_cons:
        cons_limit_arr[0] = max(cons_limit_arr[0], min_cons)

    ld_idx = hybrid_config.get("app_seq").index('cook')
    have_min_cons = hybrid_config.get("have_hard_min_lim")[ld_idx]
    min_cons = hybrid_config.get("hard_min_lim")[ld_idx]

    if have_min_cons:
        cons_limit_arr[1] = max(cons_limit_arr[1], min_cons)

    ld_idx = hybrid_config.get("app_seq").index('ent')
    have_min_cons = hybrid_config.get("have_hard_min_lim")[ld_idx]
    min_cons = hybrid_config.get("hard_min_lim")[ld_idx]

    if have_min_cons:
        cons_limit_arr[2] = max(cons_limit_arr[2], min_cons)

    return cons_limit_arr


def apply_bc_level_min_limit_for_stat_app_for_each_billing_cycle(final_tou_consumption, item_input_object,
                                                                 item_output_object, input_dict_for_min_cons):
    """
    maintain bc level level consistency of ld, cook, and ent appliances by picking from residual data

    Parameters:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        input_dict_for_min_cons     (dict)          : additional inputs required for maintain bc level min cons

    Returns:
        final_tou_consumption       (np.ndarray)    : updated final ts level estimation of appliances
        disagg_cons                 (np.ndarray)    : possible hvac cons that can be alloted to stat app
        app_cons                    (int)           : current bc level cons of the appliance
    """
    app = input_dict_for_min_cons.get('app')
    max_cons = input_dict_for_min_cons.get('max_cons')
    target_days = input_dict_for_min_cons.get('target_days')
    stat_tou = input_dict_for_min_cons.get('stat_tou')
    app_cons = input_dict_for_min_cons.get('app_cons')
    hard_limit = input_dict_for_min_cons.get('hard_limit')
    disagg_cons = input_dict_for_min_cons.get('disagg_cons')
    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

    app_idx = get_app_idx(appliance_list, app)

    samples_per_hour = int(processed_input_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    backup_app_list = item_input_object["item_input_params"]["backup_app"]
    max_residual_picked = 0.99
    min_stat_app = 500

    if (app_cons < hard_limit) and (app_cons > 0):

        additional_cons = np.zeros_like(processed_input_data[target_days])

        # picking required consumption from residual data
        # this consumption is picked in a box type signature

        other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)
        other_cons_arr = np.fmax(0, other_cons_arr)
        additional_cons[:, stat_tou] = np.fmin(max_cons, other_cons_arr[target_days][:, stat_tou]) / (
            np.max(np.fmin(max_cons, other_cons_arr[target_days][:, stat_tou])))
        additional_cons = np.nan_to_num(additional_cons)

        additional_cons = add_randomness(additional_cons)
        additional_cons[vacation[target_days]] = 0

        other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

        cons_limit = min_stat_app / samples_per_hour

        if np.any(final_tou_consumption[app_idx] > 0):
            cons_limit = np.percentile(final_tou_consumption[app_idx][final_tou_consumption[app_idx] > 0], 80)

        additional_cons[other_cons_arr[target_days] < cons_limit] = 0

        if np.sum(np.nan_to_num(additional_cons)) > 0:
            seed = RandomState(random_gen_config.seed_value)
            diff = (hard_limit - app_cons) * Cgbdisagg.WH_IN_1_KWH * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)
            additional_cons = \
                add_prepare_box_type_additional_cons_points(diff, final_tou_consumption, additional_cons, vacation,
                                                            target_days, app_idx, seed)

            final_tou_consumption[app_idx, target_days, :] = final_tou_consumption[app_idx, target_days, :] + \
                                                             np.fmax(0, np.minimum(other_cons_arr[target_days, :] * max_residual_picked,
                                                                                   additional_cons).astype(int))
            final_tou_consumption[app_idx, target_days, :] = np.minimum(final_tou_consumption[app_idx, target_days, :], processed_input_data[target_days])
            final_tou_consumption[app_idx, target_days, :] = np.fmin(final_tou_consumption[app_idx, target_days, :], max_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

    if (app_cons < hard_limit) and (app_cons > 0):

        additional_cons = np.zeros_like(processed_input_data[target_days])

        # picking required monthly consumption from residual data
        other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)
        other_cons_arr = np.fmax(0, other_cons_arr)
        additional_cons[:, stat_tou] = np.fmin(max_cons, other_cons_arr[target_days][:, stat_tou]) / (
            np.max(np.fmin(max_cons, other_cons_arr[target_days][:, stat_tou])))
        additional_cons = np.nan_to_num(additional_cons)

        additional_cons = add_randomness(additional_cons)
        additional_cons[vacation[target_days]] = 0

        if np.sum(np.nan_to_num(additional_cons)) > 0:
            additional_cons = prepare_additional_cons_points_to_satisfy_min_cons(additional_cons, hard_limit,
                                                                                 target_days, app_cons)

            other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

            final_tou_consumption[app_idx, target_days, :] = final_tou_consumption[app_idx, target_days, :] + \
                                                             np.fmax(0, np.minimum(other_cons_arr[target_days, :] * max_residual_picked,
                                                                                   additional_cons).astype(int))
            final_tou_consumption[app_idx, target_days, :] = np.minimum(final_tou_consumption[app_idx, target_days, :], processed_input_data[target_days])
            final_tou_consumption[app_idx, target_days, :] = np.fmin(final_tou_consumption[app_idx, target_days, :], max_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

            app_cons = np.fmax(0, app_cons)

    input_dict_for_min_cons = {
        'disagg_cons': disagg_cons,
        'target_days': target_days,
        'app': app,
        'hard_limit': hard_limit,
        'stat_tou': stat_tou,
        'max_cons': max_cons,
        'app_cons': app_cons
    }

    final_tou_consumption, disagg_cons, app_cons = \
        apply_min_limit_for_stat_app_for_each_bc_by_picking_hvac_cons(final_tou_consumption, item_input_object,
                                                                      item_output_object, input_dict_for_min_cons)

    for backup_app in backup_app_list:
        if (app_cons < (hard_limit - 1)) and (app_cons >= 0):
            backup_id = np.where(appliance_list == backup_app)[0][0]

            # picking required consumption from step3 consumption(statistical output of PP/EV/WH)
            # A limited amount of monthly consumption is added from step3 to stat app
            # to maintain consistency at billing cycle level output

            final_tou_consumption, continue_flag = \
                add_step3_cons_into_stat_to_maintain_stability([final_tou_consumption, processed_input_data], stat_tou,
                                                               target_days, backup_id, max_cons,  hard_limit, app_idx,
                                                               vacation, app_cons, 0.1)

    return final_tou_consumption, disagg_cons, app_cons


def apply_min_limit_for_stat_app_for_each_bc_by_picking_hvac_cons(final_tou_consumption, item_input_object,
                                                                  item_output_object, input_dict_for_min_cons):
    """
    maintain bc level level consistency of ld, cook, and ent appliances by picking slight consumption from other appliances

    Parameters:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        input_dict_for_min_cons     (dict)          : additional inputs required for maintain bc level min cons

    Returns:
        final_tou_consumption       (np.ndarray)    : updated final ts level estimation of appliances
        disagg_cons                 (np.ndarray)    : possible hvac cons that can be alloted to stat app
        app_cons                    (int)           : current bc level cons of the appliance
    """

    target_days = input_dict_for_min_cons.get('target_days')
    disagg_cons = input_dict_for_min_cons.get('disagg_cons')
    stat_tou = input_dict_for_min_cons.get('stat_tou')
    hard_limit = input_dict_for_min_cons.get('hard_limit')
    app_cons = input_dict_for_min_cons.get('app_cons')
    app = input_dict_for_min_cons.get('app')
    max_cons = input_dict_for_min_cons.get('max_cons')

    max_disagg_change = 0.05
    min_stat_app = 500

    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

    app_idx = get_app_idx(appliance_list, app)

    cool_idx = np.where(appliance_list == "cooling")[0][0]
    heat_idx = np.where(appliance_list == "heating")[0][0]
    total_samples = processed_input_data.shape[1]

    samples_per_hour = int(processed_input_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    inactive_hours = get_index_array(
        (item_output_object['profile_attributes']['sleep_time'] +  1.5) * samples_per_hour,
        (item_output_object['profile_attributes']['wakeup_time'] - 1) * samples_per_hour, total_samples).astype(int)

    wh_in_1_kwh = Cgbdisagg.WH_IN_1_KWH

    if (app_cons < (hard_limit - 1)) and (app_cons >= 0):

        # picking required consumption from HVAC cons
        # A limited amount of monthly consumption is added from HVAC to stat app
        # to maintain consistency at billing cycle level output

        additional_cons = np.zeros_like(processed_input_data[target_days])

        pick_cons = min(max_disagg_change * (disagg_cons[target_days].sum() / wh_in_1_kwh),
                        ((hard_limit - app_cons) * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)) * 0.9) * wh_in_1_kwh

        cons_limit = min_stat_app / samples_per_hour
        if np.any(final_tou_consumption[app_idx] > 0):
            cons_limit = np.percentile(final_tou_consumption[app_idx][final_tou_consumption[app_idx] > 0], 80)

        additional_cons[:, stat_tou] = disagg_cons[target_days][:, stat_tou]

        additional_cons[disagg_cons[target_days] < cons_limit] = 0

        if (app_cons < hard_limit) and (app_cons > 0):

            additional_cons = np.zeros_like(processed_input_data[target_days])

            active_hours = np.ones_like(stat_tou)
            active_hours[inactive_hours] = 0

            other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)
            other_cons_arr = np.fmax(0, other_cons_arr)
            additional_cons[:, active_hours] = np.fmin(max_cons, other_cons_arr[target_days][:, active_hours]) / (
                np.max(np.fmin(max_cons, other_cons_arr[target_days][:, active_hours])))

            additional_cons = np.nan_to_num(additional_cons)

            additional_cons = add_randomness(additional_cons)

            additional_cons[vacation[target_days]] = 0

            if np.sum(np.nan_to_num(additional_cons)) > 0:

                # picking cons that can be added from others into stat app

                additional_cons = \
                    prepare_additional_cons_points_to_satisfy_min_cons(additional_cons, hard_limit, target_days,
                                                                       app_cons)

                other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

                final_tou_consumption[app_idx, target_days, :] = \
                    final_tou_consumption[app_idx, target_days, :] +\
                    np.fmax(0, np.minimum(other_cons_arr[target_days, :] * 0.99, additional_cons).astype(int))

                final_tou_consumption[app_idx, target_days, :] = np.minimum(final_tou_consumption[app_idx, target_days, :], processed_input_data[target_days])

                final_tou_consumption[app_idx, target_days, :] = np.fmin(final_tou_consumption[app_idx, target_days, :], max_cons)

                app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx, )

                app_cons = np.fmax(0, app_cons)

        if (not (np.max(disagg_cons[target_days][:, stat_tou]) == 0)) and (np.sum(np.nan_to_num(additional_cons)) > 0):
            seed = RandomState(random_gen_config.seed_value)

            # picking cons that can be added from others into stat app

            additional_cons = \
                add_prepare_box_type_additional_cons_points(pick_cons, final_tou_consumption, additional_cons, vacation,
                                                            target_days, app_idx, seed)
            additional_cons = np.nan_to_num(additional_cons)

            target = np.fmax(0, np.minimum(
                processed_input_data[target_days] - final_tou_consumption[app_idx, target_days, :],
                np.fmax(0, np.minimum(disagg_cons[target_days], additional_cons))))

            final_tou_consumption[app_idx, target_days, :] = final_tou_consumption[app_idx, target_days, :] + target
            final_tou_consumption[cool_idx, target_days, :] = np.fmax(0, final_tou_consumption[cool_idx, target_days, :] - target)
            final_tou_consumption[heat_idx, target_days, :] = np.fmax(0, final_tou_consumption[heat_idx, target_days, :] - target)
            disagg_cons[target_days] = disagg_cons[target_days] - np.minimum(disagg_cons[target_days], additional_cons)
            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

    if (app_cons < (hard_limit - 1)) and (app_cons >= 0):

        active_hours = np.ones_like(stat_tou)
        active_hours[inactive_hours] = 0

        additional_cons = np.zeros_like(processed_input_data[target_days])

        pick_cons = min(max_disagg_change * (disagg_cons[target_days].sum() / wh_in_1_kwh),
                        ((hard_limit - app_cons) * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)) * 0.9) * wh_in_1_kwh
        additional_cons[:, stat_tou] = np.fmin(max_cons, disagg_cons[target_days][:, stat_tou]) / np.max(
            np.fmin(max_cons, disagg_cons[target_days][:, stat_tou]))

        if not (np.max(disagg_cons[target_days][:, stat_tou]) == 0):

            # picking cons that can be added from hvac into stat app
            additional_cons = prepare_ts_level_additional_cons_points(additional_cons, vacation, pick_cons, target_days)

            target = np.fmax(0, np.minimum(processed_input_data[target_days] - final_tou_consumption[app_idx, target_days, :],
                                           np.fmax(0, np.minimum(disagg_cons[target_days], additional_cons))))

            final_tou_consumption[app_idx, target_days, :] = final_tou_consumption[app_idx, target_days, :] + target
            final_tou_consumption[cool_idx, target_days, :] = \
                np.fmax(0, final_tou_consumption[cool_idx, target_days, :] - target)
            final_tou_consumption[heat_idx, target_days, :] = \
                np.fmax(0, final_tou_consumption[heat_idx, target_days, :] - target)
            disagg_cons[target_days] = disagg_cons[target_days] - np.minimum(disagg_cons[target_days], additional_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

    return final_tou_consumption, disagg_cons, app_cons


def apply_bc_level_min_limit_for_stat_app(total_cons, item_input_object, item_output_object, final_tou_consumption, length,
                                          monthly_app_cons, app_hsm_data, tou_list, use_hsm=1):
    """
    Limit min bc level level consumption increase for ld, cook, and ent appliances

    Parameters:
        total_cons                  (float)         : total consumption
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        length                      (int)           : total non vacation days of the user
        monthly_app_cons            (float)         : monthly stat app consumption
        total_monthly_cons          (float)         : monthly total consumption
        app_hsm_data                (float)         : monthly stat app consumption based on HSM
        tou_list                    (list)          : list of stat app tou
        use_hsm                     (bool)          : This flag denote if stat app hsm can be used

    Returns:
        final_tou_consumption       (np.ndarray)    : updated final ts level estimation of appliances
        ld_monthly_cons             (float)         : monthly laundry consumption
        ent_monthly_cons            (float)         : monthly ent consumption
        cook_monthly_cons           (float)         : monthly cooking consumption
    """
    # fetching required inputs

    ld_monthly_cons = monthly_app_cons[0]
    ent_monthly_cons = monthly_app_cons[1]
    cook_monthly_cons = monthly_app_cons[2]
    hsm_cook = app_hsm_data[0]
    hsm_ent = app_hsm_data[1]
    hsm_ld = app_hsm_data[2]

    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))
    scaling_factor = Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH

    samples_per_hour = int(processed_input_data.shape[1] / 24)

    day_start = 6
    multipling_factor = (Cgbdisagg.HRS_IN_DAY / (Cgbdisagg.HRS_IN_DAY - day_start))

    total = (processed_input_data[np.logical_not(vacation)][day_start * samples_per_hour:]) * multipling_factor
    total = ((np.sum(total) / len(processed_input_data)) * (scaling_factor))

    limit_cons = (total - 5000) / 200 + 50
    limit_cons = min(90, limit_cons)
    limit_cons = max(limit_cons, 2)

    limit_cons = min(limit_cons, [8, 8, 15, 20, 25, 30, 40, 50, 70, 80][np.digitize(total_cons, [300, 400, 700, 1000, 1500, 2000, 3000, 4000, 6000])])

    # initializing min consumption required for stat appp based on cons level

    ent_limit = limit_cons
    cook_limit = limit_cons * 0.8
    ld_limit = limit_cons * 1.3

    cool_idx = np.where(appliance_list == "cooling")[0][0]
    heat_idx = np.where(appliance_list == "heating")[0][0]

    cooling_limit = item_input_object.get('pilot_level_config').get('cooling_config').get('bounds').get('min_cons')
    heating_limit = item_input_object.get('pilot_level_config').get('heating_config').get('bounds').get('min_cons')

    stat_app_array = [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons]
    dev_qa_idx_arr = [0, 2, 1]
    stat_app_list = ['ld', 'cook', 'ent']
    cons_limit_arr = [ld_limit, cook_limit, ent_limit]

    disagg_cons = final_tou_consumption[cool_idx] + final_tou_consumption[heat_idx]
    disagg_cons = disagg_cons - item_input_object.get("item_input_params").get("ao_cool")
    disagg_cons = disagg_cons - item_input_object.get("item_input_params").get("ao_heat")
    disagg_cons = np.fmax(0, disagg_cons)

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    cons_limit_arr = get_stat_app_min_cons(item_input_object, cons_limit_arr)

    unique_bc, days_count = np.unique(bc_list, return_counts=1)
    unique_bc = unique_bc[days_count >= 5]
    unique_bc = unique_bc[unique_bc > 0]

    sleep_hours = copy.deepcopy(item_output_object.get("profile_attributes").get("sleep_hours"))

    if np.all(sleep_hours == 0):
        sleep_hours = np.ones_like(sleep_hours)
        sleep_hours[np.arange(1, 5 * samples_per_hour+1)] = 0

    hsm_cons = [hsm_ld, hsm_cook, hsm_ent]
    monthly_cons = [ld_monthly_cons, cook_monthly_cons, ent_monthly_cons]

    day_start = 6
    multipling_factor = (Cgbdisagg.HRS_IN_DAY / (Cgbdisagg.HRS_IN_DAY - day_start))

    for j, app in enumerate(stat_app_list):

        app_idx = get_app_idx(appliance_list, app)

        stat_tou = tou_list[j]

        # preparing app tou

        if np.sum(stat_tou) == 0:
            stat_tou = prepare_stat_tou(final_tou_consumption, app_idx, use_hsm, hsm_cons[j], app, hybrid_config, sleep_hours)

            if not np.any(stat_tou):
                continue

        max_cons = 0
        if np.sum(final_tou_consumption[app_idx] > 0) > 0:
            max_cons = np.percentile(final_tou_consumption[app_idx][final_tou_consumption[app_idx] > 0], 100)

        for current_bc in unique_bc:

            target_days = bc_list == current_bc

            if np.sum(final_tou_consumption[cool_idx]) < cooling_limit * 1.1:
                disagg_cons[target_days] = disagg_cons[target_days] - final_tou_consumption[cool_idx][target_days]

            if np.sum(final_tou_consumption[heat_idx]) < heating_limit * 1.1:
                disagg_cons[target_days] = disagg_cons[target_days] - final_tou_consumption[heat_idx][target_days]

            disagg_cons = np.fmax(0, disagg_cons)

            app_cons = get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx)

            total = (processed_input_data[np.logical_not(vacation)][bc_list[np.logical_not(vacation)] == current_bc][:, day_start * samples_per_hour:]) * multipling_factor
            total = ((np.sum(total) / len(processed_input_data[target_days])) * (scaling_factor))

            ld_limit = min(cons_limit_arr[j], [8, 8, 15, 20, 25, 30, 40, 50, 70, 80][np.digitize(total, [300, 400, 700, 1000, 1500, 2000, 3000, 4000, 6000])])

            total = (processed_input_data[np.logical_not(vacation)][bc_list[np.logical_not(vacation)] == current_bc])
            total = ((np.sum(total) / len(processed_input_data[target_days])) * (scaling_factor))

            vacation_factor = 1 - (np.sum(vacation[target_days]) / 30)

            factor = total / total_cons
            factor = np.fmin(1, (np.fmax(factor, 0.9)))

            # initializing min consumption required for stat app based on cons level and hsm data

            input_dict_for_min_cons = {
                'use_hsm': use_hsm,
                'app_cons': app_cons,
                'vacation_factor': vacation_factor,
                'factor': factor,
                'monthly_cons': monthly_cons[j]
            }

            hard_limit, app_cons = \
                prepare_hard_limit_for_stat_app_cons(ld_limit, app, hsm_cons[j], input_dict_for_min_cons)

            input_dict_for_min_cons = {
                'app': app,
                'app_cons': app_cons,
                'max_cons': max_cons,
                'factor': factor,
                'stat_tou': stat_tou,
                'hard_limit': hard_limit,
                'target_days': target_days,
                'disagg_cons': disagg_cons
            }

            final_tou_consumption, disagg_cons, app_cons = \
                apply_bc_level_min_limit_for_stat_app_for_each_billing_cycle(final_tou_consumption, item_input_object,
                                                                             item_output_object, input_dict_for_min_cons)

        final_tou_consumption[app_idx] = np.fmax(0, final_tou_consumption[app_idx])
        cons = final_tou_consumption[app_idx][np.logical_not(vacation)]
        stat_app_array[dev_qa_idx_arr[j]] = ((np.sum(cons) / length) * (scaling_factor))

    return final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons


def prepare_hard_limit_for_stat_app_cons(initial_limit, app, hsm_cons, input_dict_for_min_cons):
    """
    prepare min bc level level consumption for consistency of ld, cook, and ent appliances

    Parameters:
        initial_limit               (int)           : initial min limit of the app
        app                         (np.ndarray)    : target app
        hsm_cons                    (np.ndarray)    : hsm params  of stat app
        input_dict_for_min_cons     (dict)          : additional inputs required for maintain bc level min cons

    Returns:
        hard_limit                  (int)           : min amount of cons to be maintained
        app_cons                    (int)           : current bc level cons of the appliance
    """
    monthly_cons = input_dict_for_min_cons.get('monthly_cons')
    factor = input_dict_for_min_cons.get('factor')
    vacation_factor = input_dict_for_min_cons.get('vacation_factor')
    app_cons = input_dict_for_min_cons.get('app_cons')
    use_hsm = input_dict_for_min_cons.get('use_hsm')

    if app in ['ent', 'ld']:
        # update min cons based on hsm

        if use_hsm:
            initial_limit = max(initial_limit, hsm_cons)

        hard_limit = max(initial_limit, monthly_cons * 1 * factor * vacation_factor)

    else:
        # update min cons based on hsm
        if use_hsm:
            initial_limit = max(initial_limit, hsm_cons)

        hard_limit = max(initial_limit, monthly_cons * 1 * factor * vacation_factor)

        if monthly_cons > 0 and app_cons == 0:
            app_cons = 1

    return hard_limit, app_cons
