

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

# import functions from within the project

from python3.config.pilot_constants import PilotConstants

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.raw_energy_itemization.inference_engine.update_sig_app_ranges import adjust_pp_cons
from python3.itemization.aer.raw_energy_itemization.inference_engine.update_sig_app_ranges import get_index
from python3.itemization.aer.raw_energy_itemization.inference_engine.update_sig_app_ranges import add_seasonality_in_ref
from python3.itemization.aer.raw_energy_itemization.inference_engine.get_backup_module_output import add_ev_pp_backup_cons
from python3.itemization.aer.raw_energy_itemization.inference_engine.update_sig_app_ranges import handle_low_ev_cons
from python3.itemization.aer.raw_energy_itemization.inference_engine.update_sig_app_ranges import add_thin_pulse_to_wh_cons
from python3.itemization.aer.raw_energy_itemization.inference_engine.update_sig_app_ranges import add_daily_baseload_to_ent
from python3.itemization.aer.raw_energy_itemization.inference_engine.update_sig_app_ranges import add_extra_pp_cons_to_cooling
from python3.itemization.aer.raw_energy_itemization.inference_engine.update_sig_app_ranges import adjust_stat_app_consistency
from python3.itemization.aer.raw_energy_itemization.inference_engine.update_sig_app_ranges import block_sparse_wh_output_users_added_from_hybrid

from python3.itemization.aer.raw_energy_itemization.inference_engine.util_functions_to_maintain_min_max_cons_for_disagg_app import maintain_wh_min_cons
from python3.itemization.aer.raw_energy_itemization.inference_engine.util_functions_to_maintain_min_max_cons_for_disagg_app import maintian_wh_max_cons
from python3.itemization.aer.raw_energy_itemization.inference_engine.util_functions_to_maintain_min_max_cons_for_disagg_app import apply_max_cap_on_ref
from python3.itemization.aer.raw_energy_itemization.inference_engine.util_functions_to_maintain_min_max_cons_for_disagg_app import maintain_min_cons_for_disagg_app
from python3.itemization.aer.raw_energy_itemization.inference_engine.util_functions_to_maintain_min_max_cons_for_disagg_app import maintain_min_cons_for_step3_app

from python3.itemization.aer.raw_energy_itemization.inference_engine.adjust_disagg_app_overlap import handle_neg_residual_cases
from python3.itemization.aer.raw_energy_itemization.inference_engine.util_functions_to_maintain_min_max_cons_for_disagg_app import handle_low_cons
from python3.itemization.aer.raw_energy_itemization.inference_engine.util_functions_to_maintain_min_max_cons_for_disagg_app import maintain_minimum_thin_pulse_cons


def adjust_disagg_app(item_input_object, item_output_object, logger_pass):

    """
    Modify appliance mid/min/max ranges in cases where true disagg appliances are overlapping

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs

    Returns:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    logger_base = logger_pass.get('logger_base').getChild('adjust_disagg_app')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    output_data = copy.deepcopy(item_output_object.get("inference_engine_dict").get("output_data"))
    mid_cons_vals = copy.deepcopy(item_output_object.get("inference_engine_dict").get("appliance_mid_values"))
    max_cons_vals = copy.deepcopy(item_output_object.get("inference_engine_dict").get("appliance_max_values"))
    min_cons_vals = copy.deepcopy(item_output_object.get("inference_engine_dict").get("appliance_min_values"))
    conf_vals = copy.deepcopy(item_output_object.get("inference_engine_dict").get("appliance_conf"))

    input_data = item_output_object.get('original_input_data')

    pp_idx = get_index('pp')
    ev_idx = get_index('ev')
    wh_idx = get_index('wh')
    ent_idx = get_index('ent')
    cook_idx = get_index('cook')
    ld_idx = get_index('ld')
    cooling_idx = get_index('cooling')

    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')

    swh_pilots = PilotConstants.SEASONAL_WH_ENABLED_PILOTS

    if np.all(conf_vals[pp_idx] == 0) and np.any(mid_cons_vals[pp_idx] > 0):
        conf_vals[pp_idx][:, :] = 1
        conf_vals[pp_idx][mid_cons_vals[pp_idx] == 0] = 0

    if np.all(conf_vals[wh_idx] == 0) and np.any(mid_cons_vals[wh_idx] > 0):
        conf_vals[wh_idx][:, :] = 1
        conf_vals[wh_idx][mid_cons_vals[wh_idx] == 0] = 0

    # removing stat app when ao and ref consumption are creating overestimation

    negative_residual = (input_data - mid_cons_vals[get_index('ao')] - mid_cons_vals[get_index('ref')]) < 0

    mid_cons_vals[ent_idx][negative_residual] = (mid_cons_vals[ent_idx] * 0)[negative_residual]
    mid_cons_vals[ld_idx][negative_residual] = (mid_cons_vals[ld_idx] * 0)[negative_residual]
    mid_cons_vals[cook_idx][negative_residual] = (mid_cons_vals[cook_idx] * 0)[negative_residual]

    min_cons_vals[ent_idx][negative_residual] = (min_cons_vals[ent_idx] * 0)[negative_residual]
    min_cons_vals[ld_idx][negative_residual] = (min_cons_vals[ld_idx] * 0)[negative_residual]
    min_cons_vals[cook_idx][negative_residual] = (min_cons_vals[cook_idx] * 0)[negative_residual]

    # remove consumption of the appliances that are to be killed based on pilot config and app profile config

    if run_hybrid_v2:
        app_killer = item_input_object['item_input_params']['app_killer']
        min_cons_vals[np.array(app_killer).astype(bool)] = 0
        max_cons_vals[np.array(app_killer).astype(bool)] = 0
        mid_cons_vals[np.array(app_killer).astype(bool)] = 0


    # adjusting true disagg appliance ranges in the ts where total disagg exceeds raw energy
    # based on appliances ts level detection confidences
    mid_cons_vals, min_cons_vals, max_cons_vals = \
        handle_neg_residual_cases(item_input_object, item_output_object, mid_cons_vals, min_cons_vals, max_cons_vals, conf_vals, input_data)



    # adjusting stat app disagg appliance ranges in the ts where total disagg exceeds raw energy, even after adjusting true diasgg appliances
    negative_residual = (input_data - np.sum(mid_cons_vals, axis=0)) < 0

    min_cons_vals[ent_idx][negative_residual] = (min_cons_vals[ent_idx] * 0)[negative_residual]
    min_cons_vals[ld_idx][negative_residual] = (min_cons_vals[ld_idx] * 0)[negative_residual]
    min_cons_vals[cook_idx][negative_residual] = (min_cons_vals[cook_idx] * 0.1)[negative_residual]

    month = pd.DatetimeIndex(item_output_object.get("date_list")).month.values - 1
    vacation = np.logical_or(mid_cons_vals[get_index('va2')], mid_cons_vals[get_index('va1')])[:, 0]

    # postprocessing to maintain consistency in stat app bc level consumption

    for app in ['cook', 'ent', 'ld']:
        min_cons_vals, mid_cons_vals, max_cons_vals = adjust_stat_app_consistency(app, vacation, month, min_cons_vals, mid_cons_vals, max_cons_vals)

    # checking if wh thin pulse is unchanged after adjusting ranges
    min_cons_vals, mid_cons_vals, max_cons_vals = \
        maintain_minimum_thin_pulse_cons(min_cons_vals, mid_cons_vals, max_cons_vals, item_input_object, input_data, output_data, swh_pilots, logger)


    if run_hybrid_v2:
        # adding leftover daily baseload activity as entertainment consumption
        min_cons_vals, mid_cons_vals, max_cons_vals = \
            add_daily_baseload_to_ent(vacation, min_cons_vals, mid_cons_vals, max_cons_vals,
                                      item_input_object, item_output_object, negative_residual, input_data)
    # checks to ensure ts level consistency in pp min/mid/max consumption values

    pp_copy = copy.deepcopy(mid_cons_vals[pp_idx])


    mid_cons_vals[pp_idx] = adjust_pp_cons(output_data[pp_idx], mid_cons_vals[pp_idx], item_input_object, logger)
    min_cons_vals[pp_idx] = adjust_pp_cons(output_data[pp_idx], min_cons_vals[pp_idx], item_input_object, logger)
    max_cons_vals[pp_idx] = adjust_pp_cons(output_data[pp_idx], max_cons_vals[pp_idx], item_input_object, logger)


    min_cons_vals, mid_cons_vals, max_cons_vals = \
        handle_low_cons(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals,
                        vacation, output_data, logger, input_data, swh_pilots)

    min_cons_vals, mid_cons_vals, max_cons_vals = \
        add_extra_pp_cons_to_cooling(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals,
                                     cooling_idx, pp_idx, output_data, logger, pp_copy)

    samples = int(mid_cons_vals.shape[2] /  Cgbdisagg.HRS_IN_DAY)

    # add additional thin pulse consumption from disagg residual into wh ts level output

    min_cons_vals, mid_cons_vals, max_cons_vals = \
        add_thin_pulse_to_wh_cons(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals,
                                  wh_idx, output_data, logger)

    # remove additional low duration ev boxes

    mid_cons_vals, mid_cons_vals, max_cons_vals = \
        handle_low_ev_cons(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals, ev_idx,
                           output_data, samples)


    min_cons_vals[cook_idx, :, :] = min_cons_vals[cook_idx, :, :] * 0.1
    min_cons_vals[ld_idx, :, :] = min_cons_vals[ld_idx, :, :] * 0.1
    min_cons_vals[ent_idx, :, :] = min_cons_vals[ent_idx, :, :] * 0.1

    residual = input_data - mid_cons_vals[get_index('ao')] - mid_cons_vals[pp_idx] - mid_cons_vals[wh_idx] - \
               mid_cons_vals[ev_idx] - mid_cons_vals[get_index('ref')] - mid_cons_vals[cooling_idx] - \
               mid_cons_vals[get_index('heating')] - mid_cons_vals[get_index('li')] - \
               mid_cons_vals[get_index('ent')] - mid_cons_vals[get_index('ld')] - mid_cons_vals[get_index('cook')]

    if np.sum(output_data[wh_idx]) == 0 and np.sum(mid_cons_vals[wh_idx]) > 0:
        mid_cons_vals[wh_idx] = block_sparse_wh_output_users_added_from_hybrid(mid_cons_vals[wh_idx], output_data[wh_idx], item_input_object)

        if np.sum(mid_cons_vals[wh_idx]) == 0:
            min_cons_vals[wh_idx][:, :] = 0
            max_cons_vals[wh_idx][:, :] = 0

    # add monthly level ev or pp consumption if not detected by algorithm and
    # ev/pp is to be given to all users with app profile as yes, based on pilot config

    if run_hybrid_v2:

        min_cons_vals, mid_cons_vals, max_cons_vals = \
            add_ev_pp_backup_cons(item_input_object, item_output_object, residual, min_cons_vals, mid_cons_vals, max_cons_vals, logger)



        appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

        # adding seasonality in ref consumption

        min_cons_vals, mid_cons_vals, max_cons_vals = add_seasonality_in_ref(item_input_object, item_output_object, min_cons_vals, mid_cons_vals, max_cons_vals)

        # function to ensure min/max consumption of true disagg appliances

        min_cons_vals = maintain_min_cons(appliance_list, input_data, min_cons_vals, item_input_object, item_output_object, logger)
        mid_cons_vals = maintain_min_cons(appliance_list, input_data, mid_cons_vals, item_input_object, item_output_object, logger)
        max_cons_vals = maintain_min_cons(appliance_list, input_data, max_cons_vals, item_input_object, item_output_object, logger)

        min_cons_vals, mid_cons_vals, max_cons_vals = \
            maintain_minimum_thin_pulse_cons(min_cons_vals, mid_cons_vals, max_cons_vals, item_input_object, input_data, output_data, swh_pilots, logger)

        app_killer = item_input_object['item_input_params']['app_killer']
        min_cons_vals[np.array(app_killer).astype(bool)] = 0
        max_cons_vals[np.array(app_killer).astype(bool)] = 0
        mid_cons_vals[np.array(app_killer).astype(bool)] = 0

        app_killer = item_input_object['item_input_params']['monthly_app_killer']

        date_list = item_output_object.get("date_list")
        month_list = pd.DatetimeIndex(date_list).month.values - 1

        for app in appliance_list:

            index = get_index(app)
            target_months = np.where(app_killer[index] > 0)[0]
            target_months = np.isin(month_list, target_months)
            min_cons_vals[index][target_months] = 0
            mid_cons_vals[index][target_months] = 0
            max_cons_vals[index][target_months] = 0

    item_output_object["inference_engine_dict"]["appliance_mid_values"] = np.fmax(0, mid_cons_vals)
    item_output_object["inference_engine_dict"]["appliance_max_values"] = np.fmax(0, max_cons_vals)
    item_output_object["inference_engine_dict"]["appliance_min_values"] = np.fmax(0, min_cons_vals)

    return item_input_object, item_output_object



def maintain_min_cons(appliance_list, input_data, final_tou_consumption, item_input_object, item_output_object, logger):

    """
    blocking of pp for low consuption billing cycles and blocking of hvac based on app profile

    Parameters:
        appliance_list            (np.ndarray)    : list of all appliances
        input_data                (np.ndarray)    :  user input data
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
        item_input_object         (dict)          : dict containing all input
        item_output_object        (dict)          : dict containing all output
        unique_bc                 (np.ndarray)    : list of unique billing cycles
        bc_list                   (np.ndarray)    : billing cycle data
        logger                    (np.ndarray)    : logger object

    Returns:
        final_tou_consumption     (np.ndarray)    : hybrid v2 output for all appliances
    """

    other_cons_arr = np.fmax(0, input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0))

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, counts = np.unique(bc_list, return_counts=1)
    unique_bc = unique_bc[counts >= 5]

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    # limiting amplitude change in ts level lighting output
    final_tou_consumption[np.where(appliance_list == 'li')[0][0]][vacation] = 0
    amp = np.max(final_tou_consumption[np.where(appliance_list == 'li')[0][0]]) * 2

    # maintaining a minimum bc level cons for true disagg appliances
    final_tou_consumption = maintain_min_cons_for_disagg_app(appliance_list, input_data, final_tou_consumption,
                                                             item_input_object, item_output_object, unique_bc, bc_list, other_cons_arr)

    final_tou_consumption[np.where(appliance_list == 'li')[0][0]] = np.fmin(final_tou_consumption[np.where(appliance_list == 'li')[0][0]], amp)


    # maintaining a minimum bc level cons for step3 appliances
    final_tou_consumption = maintain_min_cons_for_step3_app(appliance_list, input_data, final_tou_consumption,
                                                            item_input_object, item_output_object, unique_bc, bc_list, logger)


    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)
    unique_bc = unique_bc[counts > 5]

    # maintaining a max bc level cons for ref
    final_tou_consumption = apply_max_cap_on_ref(appliance_list, input_data, final_tou_consumption,
                                                 item_input_object, item_output_object, unique_bc, bc_list, logger)

    if item_input_object.get("item_input_params").get("timed_wh_user") > 0:
        return final_tou_consumption

    # maintaining a min bc level cons for wh


    final_tou_consumption = maintain_wh_min_cons(appliance_list, input_data, final_tou_consumption, item_input_object,
                                                 item_output_object, unique_bc, bc_list, logger)


    wh_idx = np.where(appliance_list == 'wh')[0][0]
    tou = final_tou_consumption[wh_idx] > 0

    # maintaining a max bc level cons for wh
    final_tou_consumption = maintian_wh_max_cons(appliance_list, input_data, final_tou_consumption, item_input_object,
                                                 item_output_object, unique_bc, bc_list, logger)

    thin_pulse = item_input_object.get("item_input_params").get("final_thin_pulse")

    thin_pulse_not_present = (thin_pulse is None) or (np.sum(final_tou_consumption[wh_idx]) == 0) or\
                             (item_input_object.get('item_input_params').get('swh_hld')) or \
                             (item_input_object.get("item_input_params").get("timed_wh_user"))

    if thin_pulse_not_present:
        thin_pulse = np.zeros_like(final_tou_consumption[0])
    final_tou_consumption[wh_idx][tou] = np.maximum(final_tou_consumption[wh_idx][tou], thin_pulse[tou])


    # maintaining the minimum thinpulse cons of storage wh users
    min_cons = item_input_object.get('pilot_level_config').get('wh_config').get('bounds').get('min_cons')

    other_cons_arr = np.fmax(0, input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0))
    wh_idx = np.where(appliance_list == 'wh')[0][0]

    wh_cons = final_tou_consumption[wh_idx]

    samples = int(wh_cons.shape[1]/24)

    if 'wh' in item_input_object["item_input_params"]["backup_app"]:
        return final_tou_consumption

    for bc in unique_bc:

        vac_factor = np.sum(vacation[bc_list == bc]) / np.sum(bc_list == bc)
        vac_factor = 1 - vac_factor

        diff = min_cons*vac_factor*(np.sum(bc_list == bc)/30) - final_tou_consumption[wh_idx][bc_list == bc].sum() / 1000

        if diff > 0 and (np.sum(wh_cons[bc_list == bc]) > 0):
            factor = (final_tou_consumption[wh_idx][bc_list == bc].sum() / (min_cons*vac_factor*1000)) * 30/np.sum(bc_list == bc)

            if factor < 1 and (final_tou_consumption[wh_idx][bc_list == bc].sum() > 4000) and \
                    ((final_tou_consumption[wh_idx][bc_list == bc] > 0).sum() > 5*samples):
                desired_val = final_tou_consumption[wh_idx][bc_list == bc] / factor  - final_tou_consumption[wh_idx][bc_list == bc]

                desired_val = np.minimum(desired_val, other_cons_arr[bc_list == bc])

                desired_val[vacation[bc_list == bc] > 0] = 0

                final_tou_consumption[wh_idx][bc_list == bc] = final_tou_consumption[wh_idx][bc_list == bc] + desired_val

    max_cons = item_input_object.get('pilot_level_config').get('wh_config').get('bounds').get('max_cons')

    thin_pulse = item_input_object.get("item_input_params").get("final_thin_pulse")

    if thin_pulse_not_present:
        thin_pulse = np.zeros_like(final_tou_consumption[0])

    tou = final_tou_consumption[wh_idx] > 0

    for bc in unique_bc:

        vac_factor = np.sum(vacation[bc_list == bc]) / np.sum(bc_list == bc)
        vac_factor = 1 - vac_factor

        diff = final_tou_consumption[wh_idx][bc_list == bc].sum() / 1000 - max_cons*vac_factor*(np.sum(bc_list == bc)/30)

        if diff > 0 and np.sum(wh_cons[bc_list == bc]) > 0:

            factor = (final_tou_consumption[wh_idx][bc_list == bc].sum() / (max_cons*vac_factor * 1000)) * 30 / np.sum(bc_list == bc)

            factor = max(factor, 1)

            final_tou_consumption[wh_idx][bc_list == bc] = final_tou_consumption[wh_idx][bc_list == bc] / factor

    final_tou_consumption[wh_idx][tou] = np.maximum(final_tou_consumption[wh_idx][tou], thin_pulse[tou])

    final_tou_consumption[np.where(appliance_list == 'li')[0][0]][vacation] = 0

    return final_tou_consumption
