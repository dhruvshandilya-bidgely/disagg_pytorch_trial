"""
Author - Mayank Sharan/Nisha Agarwal
Date - 4th April 2021
Stage 3 of itemization framework - calculate final tou level estimation
"""

# Import python packages

import copy
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.get_config import get_hybrid_config


from python3.itemization.aer.raw_energy_itemization.get_final_consumption.reduce_high_others import pick_potential_ent_cons_from_residual

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.consistency_check_utils import consistency_check_for_all_output_based_on_stat_estimation
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.consistency_check_utils import consistency_check_for_low_cons_app
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.consistency_check_utils import stat_app_consistency_check
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.consistency_check_utils import apply_consistency_in_neighbouring_bcs

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils import get_stat_app_hsm

from python3.itemization.aer.functions.itemization_utils import get_idx

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_apply_bc_level_limit_on_stat_app import apply_bc_level_max_limit_based_on_config
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_apply_bc_level_limit_on_stat_app import apply_bc_level_max_limit_based_on_config_and_hsm
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.adjust_stat_app_cons import apply_bc_level_min_limit
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_maintain_min_stat_cons import apply_soft_limit_on_bc_level_min_cons
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.adjust_stat_app_cons import apply_bc_level_min_limit_for_stat_app
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils_to_apply_bc_level_limit_on_stat_app import apply_bc_level_min_limit_for_step3_app

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils import block_low_cons_billing_cycle_output
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils import apply_max_ts_level_limit
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils import handle_leftover_neg_res_points

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hvac_postprocess_utils import adjust_seasonal_app_max_limit
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils import limit_wh_delta
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hvac_postprocess_utils import limit_hvac_delta

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.handle_overest_cases import handle_case_a
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.handle_overest_cases import handle_case_c
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.handle_underest_cases import handle_case_b
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.handle_underest_cases import handle_case_d
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.compute_saturation import compute_saturation

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.post_processing_utils import pp_post_process
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.post_processing_utils import ev_post_process
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.post_processing_utils import final_pp_post_process
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.post_processing_utils import post_process_based_on_pp_hsm

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.init_final_item_config import init_final_item_conf

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.post_processing_utils import twh_post_process
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.block_feeble_cons_utils import block_feeble_cons_in_wh
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.block_feeble_cons_utils import block_stat_app_feeble_cons
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.post_processing_utils import block_low_cons_wh
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.block_feeble_cons_utils import block_outlier_points
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.block_feeble_cons_utils import block_feeble_cons_thin_pulse
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.block_feeble_cons_utils import block_feeble_cons_in_pp
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.post_processing_utils import wh_seasonality_check

from python3.itemization.aer.raw_energy_itemization.residual_analysis.box_activity_detection_wrapper import box_detection
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.post_processing_utils import allot_thin_pulse_boxes

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.reduce_high_others import add_box_cons_to_stat_app
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.reduce_high_others import increase_stat_app_in_bc_with_high_others
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.reduce_high_others import add_leftover_potential_boxes_to_hvac

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import update_ev_hsm
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import update_pp_hsm
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import update_wh_hsm
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import update_li_hsm
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import update_ref_hsm

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import post_process_based_on_wh_hsm
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import ev_hsm_post_process
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.hsm_utils import update_stat_app_hsm

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils import update_output_object_for_hld_change_cases
from python3.itemization.aer.raw_energy_itemization.get_final_consumption.utils import modify_stat_app_based_on_app_prof

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.postprocess_app_ranges import maintain_min_cons


def get_final_consumption(item_input_object, item_output_object, logger_pass):

    """
    This function serve as a master function which calls all the submodules to perform final 100% itemization

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    # Initialize the logger

    logger_pass = logger_pass.copy()

    logger_base = logger_pass.get('logger_base').getChild('get_final_consumption')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    logger_pass['logger_base'] = logger_base

    config = init_final_item_conf()

    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')

    item_adjustment_dict = {}

    # Find saturation based on expected ranges

    inference_engine_dict = item_output_object.get('inference_engine_dict')
    appliance_min_values = inference_engine_dict.get('appliance_min_values')
    appliance_mid_values = inference_engine_dict.get('appliance_mid_values')
    appliance_max_values = inference_engine_dict.get('appliance_max_values')
    output_data = inference_engine_dict.get('output_data')

    app_list = np.array(inference_engine_dict.get('appliance_list'))

    pp_idx = get_idx('pp', app_list)
    cook_idx = get_idx('cook', app_list)
    ent_idx = get_idx('ent', app_list)
    ld_idx = get_idx('ld', app_list)
    wh_idx = get_idx('wh', app_list)
    ao_idx = get_idx('ao', app_list)
    ev_idx = get_idx('ev', app_list)
    ref_idx = get_idx('ref', app_list)
    li_idx = get_idx('li', app_list)
    heat_idx = get_idx('heating', app_list)

    scaling_factor = (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH)

    processed_input_data = copy.deepcopy(item_output_object.get('original_input_data'))
    num_days = processed_input_data.shape[0]

    # Calculate saturation at different times of the day using the activity curve and extracted info

    item_adjustment_dict = compute_saturation(item_input_object, item_output_object, item_adjustment_dict)

    logger.debug("Calculated ts level min/avg/max saturation")

    app_list = np.array(inference_engine_dict.get('appliance_list'))
    num_appliances = len(app_list)

    appliance_min_sum = np.sum(appliance_min_values, axis=0)
    appliance_mid_sum = np.sum(appliance_mid_values, axis=0)
    appliance_max_sum = np.sum(appliance_max_values, axis=0)

    denominator_input_data = copy.deepcopy(processed_input_data)
    invalid_input_data_bool = denominator_input_data == 0
    denominator_input_data[invalid_input_data_bool] = 1e-3

    appliance_min_saturation = np.divide(appliance_min_sum, denominator_input_data)
    appliance_mid_saturation = np.divide(appliance_mid_sum, denominator_input_data)
    appliance_max_saturation = np.divide(appliance_max_sum, denominator_input_data)

    # Prepare the guidance saturation values for comparison

    max_saturation_arr = item_adjustment_dict.get('max_saturation_arr')
    min_saturation_arr = item_adjustment_dict.get('min_saturation_arr')
    mid_saturation_arr = item_adjustment_dict.get('mid_saturation_arr')

    max_saturation_arr = np.tile(max_saturation_arr, reps=(num_days, 1))
    min_saturation_arr = np.tile(min_saturation_arr, reps=(num_days, 1))
    mid_saturation_arr = np.tile(mid_saturation_arr, reps=(num_days, 1))

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    # Calculate ts level minimum required consumption (in wh)

    min_others = np.ones_like(output_data[0])

    min_others[:, :] = config.get("res_conf").get("min_res") / samples_per_hour
    min_res_in_overest = config.get("res_conf").get("min_res_in_overest") / samples_per_hour

    # Reduce the minimum required for ts with high consumption appliances

    min_others[output_data[pp_idx, :, :] > 0] = min_res_in_overest
    min_others[output_data[ev_idx, :, :] > 0] = min_res_in_overest
    min_others[output_data[wh_idx, :, :] > 0] = min_res_in_overest
    min_others[output_data[cook_idx, :, :] > 0] = min_res_in_overest
    min_others[output_data[heat_idx, :, :] > 0] = min_res_in_overest

    total_consumption = np.reshape(processed_input_data, newshape=(1, processed_input_data.shape[0], processed_input_data.shape[1]))

    processed_input_data_copy = copy.deepcopy(processed_input_data)
    processed_input_data_copy[processed_input_data_copy == 0] = 1

    # Updating max saturation based on calculated min required others consumption
    array = 1 - np.divide(min_others, processed_input_data_copy)
    array[processed_input_data < 100] = 1
    max_saturation_arr = np.minimum(array, max_saturation_arr)

    sleep_hours = item_output_object.get("profile_attributes").get("sleep_hours")

    max_saturation_arr[:, np.logical_not(sleep_hours)] = 0.995

    max_saturation_arr[invalid_input_data_bool] = 0.0
    min_saturation_arr[invalid_input_data_bool] = 0.0
    mid_saturation_arr[invalid_input_data_bool] = 0.0

    logger.debug("Fetched required inputs")

    vacation = np.sum(item_input_object.get("item_input_params").get("vacation_data"), axis=1) > 0
    vacation = np.logical_or(vacation, processed_input_data.sum(axis=1) == 0)
    length = np.sum(np.logical_not(vacation))
    length = max(1, length)

    #################### TS level optimization for all appliances to maintain ts level 100% itemization ################
    #################### This is done based on ts level min/avg/ax consumption of all appliances  ################

    # Initialize 3d array which would contain the timestamp level itemization

    final_tou_consumption = copy.deepcopy(appliance_mid_values)

    # Start changing estimates case by case
    # Case A (Overestimation) : Mid Consumption is beyond max saturation but min consumption is not

    case_a_bool = np.logical_and(appliance_mid_saturation > max_saturation_arr,
                                 np.logical_not(appliance_min_saturation > max_saturation_arr))

    logger.info('Number of points in case A | %d', np.sum(case_a_bool))

    # Set consumption to minimum for all appliances and then bring them up to max saturation

    final_tou_consumption = \
        handle_case_a(item_input_object, item_output_object, item_adjustment_dict, final_tou_consumption,
                      case_a_bool, appliance_min_values, appliance_mid_values, appliance_max_values, logger)

    final_tou_consumption = np.nan_to_num(final_tou_consumption)


    # Case B (Underestimation) : Mid Consumption is below min saturation but max consumption is not

    case_b_bool = np.logical_and(appliance_mid_saturation < min_saturation_arr,
                                 np.logical_not(appliance_max_saturation < min_saturation_arr))

    logger.info('Number of points in case B | %d', np.sum(case_b_bool))

    # Post Processing for signature based appliances, to maintain appliance specific rules

    disagg_confidence = 0

    if item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') is not None:
        disagg_confidence = (item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') / 100)

    final_tou_consumption, predicted, pp_type = \
        pp_post_process(pp_idx, output_data, final_tou_consumption, item_input_object, processed_input_data,
                        disagg_confidence, logger)

    final_tou_consumption = \
        handle_case_b(item_input_object, item_output_object, item_adjustment_dict, final_tou_consumption,
                      case_b_bool, appliance_mid_values, appliance_max_values, logger)

    # Scale consumption for any points that do not reduce easily and are above 1

    new_sat = np.divide(np.sum(final_tou_consumption, axis=0), denominator_input_data)
    scale_bool = np.logical_and(new_sat > 1, case_b_bool)

    div_arr = copy.deepcopy(new_sat)
    div_arr[~scale_bool] = 1
    div_arr[scale_bool] = div_arr[scale_bool] + 1e-2

    final_tou_consumption = np.divide(final_tou_consumption, div_arr)

    # Case C (Massive Overestimation) : Min Consumption is above max saturation

    case_c_bool = appliance_min_saturation > max_saturation_arr

    inputs_for_case_c_resolution = {
        'case_c_bool': case_c_bool,
        'appliance_min_values': appliance_min_values,
        'ao_idx': ao_idx,
        'ref_idx': ref_idx
    }

    final_tou_consumption, case_c_bool_3d = \
        handle_case_c(item_input_object, item_output_object, item_adjustment_dict, final_tou_consumption,
                      inputs_for_case_c_resolution, logger)

    # Case D (Massive Underestimation) : Max Consumption is below min saturation

    case_d_bool = appliance_max_saturation < min_saturation_arr
    case_d_bool_3d = np.tile(case_d_bool, reps=(num_appliances, 1, 1))

    # For each point in case D process by allowing all appliances with a non zero max and scale if overshoots

    final_tou_consumption = \
        handle_case_d(item_output_object, item_adjustment_dict, final_tou_consumption, case_d_bool,
                      appliance_max_values, denominator_input_data, logger)

    # Bound the consumption by the min and max

    bound_bool = np.logical_not(np.logical_or(case_c_bool_3d, case_d_bool_3d))

    final_tou_consumption[bound_bool] = np.minimum(final_tou_consumption, appliance_max_values)[bound_bool]
    final_tou_consumption[bound_bool] = np.maximum(final_tou_consumption, appliance_min_values)[bound_bool]

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)
    final_tou_consumption = \
        twh_post_process(wh_idx, heat_idx, other_cons_arr, output_data, final_tou_consumption, item_input_object)


    final_tou_consumption = \
        ev_post_process(ev_idx, item_input_object, processed_input_data, output_data, final_tou_consumption, logger)

    if run_hybrid_v2:

        # Do a final adjustment

        final_adj_arr = np.ones_like(final_tou_consumption)
        final_scalable_sum = np.zeros_like(processed_input_data)

        scale_apps = ['ld', 'cook', 'wh', 'ent']

        if item_input_object.get("item_input_params").get("timed_wh_user"):
            scale_apps = ['ld', 'cook', 'ent']

        for app_name in scale_apps:
            app_idx = np.where(app_list == app_name)[0][0]
            final_scalable_sum = final_scalable_sum + final_tou_consumption[app_idx, :, :]

        non_scalable_sum = np.sum(final_tou_consumption, axis=0) - final_scalable_sum
        non_scalable_sat = np.divide(non_scalable_sum, denominator_input_data)
        scalable_sat = np.divide(final_scalable_sum, denominator_input_data)

        scale_sat_avail = np.zeros_like(non_scalable_sat)

        offset = 0

        mid_sat_diff = mid_saturation_arr - non_scalable_sat - offset
        type_1_pts_bool = mid_sat_diff > 0
        scale_sat_avail[type_1_pts_bool] = mid_sat_diff[type_1_pts_bool]

        max_sat_diff = max_saturation_arr - non_scalable_sat
        type_2_pts_bool = np.logical_and(max_sat_diff > 0, mid_sat_diff <= 0)
        scale_sat_avail[type_2_pts_bool] = max_sat_diff[type_2_pts_bool]

        scale_arr = np.divide(scale_sat_avail, scalable_sat + 1e-3)

        for app_name in scale_apps:
            final_adj_arr[np.where(app_list == app_name)[0][0], :, :] = scale_arr

        # Force scaling at points where non scalable appliances are causing issues

        force_scale_bool = non_scalable_sat > max_saturation_arr

        force_scale_arr = np.divide(max_saturation_arr, non_scalable_sat + 1e-3)
        force_scale_arr[~force_scale_bool] = 1

        non_scale_apps = ['pp', 'ev', 'li', 'cooling', 'heating']

        if item_input_object.get("item_input_params").get("timed_wh_user"):
            non_scale_apps = ['pp', 'wh', 'ev', 'li', 'cooling', 'heating']

        for app_name in non_scale_apps:
            final_adj_arr[np.where(app_list == app_name)[0][0], :, :] = force_scale_arr

        logger.info("Final adjustment done | ")

        output_data = inference_engine_dict.get('output_data')

        #################### Applying ts level postprocessing on true disagg appliances  ################
        ############ This is done in order to maintain ts level consistency and seasonality of these appliances  ################

        if final_tou_consumption[li_idx].sum() > 0:
            final_tou_consumption[li_idx] = np.fmin(final_tou_consumption[li_idx], np.percentile(final_tou_consumption[li_idx][final_tou_consumption[li_idx] > 0], 97))

        final_adj_arr, detected_cool, detected_heat = adjust_seasonal_app_max_limit(app_list, item_input_object,
                                                                                    item_output_object, output_data,
                                                                                    final_adj_arr)

        final_adj_arr = update_increament_based_on_app_survey(item_input_object, final_adj_arr, app_list)

        final_tou_consumption = np.multiply(final_adj_arr, final_tou_consumption)

        final_tou_consumption = maintain_min_cons(app_list, processed_input_data, final_tou_consumption, item_input_object, item_output_object, logger)

    # Few postprocessing to limit HVAC delta from disagg

    final_tou_consumption = ev_hsm_post_process(final_tou_consumption, item_input_object, output_data, ev_idx, length)

    # postprocessing of pp output in mtd mode based on pp hsm

    final_tou_consumption = \
        post_process_based_on_pp_hsm(final_tou_consumption, item_input_object, output_data, pp_idx, length)

    # postprocessing of pp output in mtd mode based on wh hsm

    final_tou_consumption, hsm_wh = post_process_based_on_wh_hsm(final_tou_consumption, item_input_object, wh_idx)

    # apply max limit on bc level change in hvac output compared to disagg
    detected_cool = item_output_object.get("hvac_dict").get("cooling")
    detected_heat = item_output_object.get("hvac_dict").get("heating")

    final_tou_consumption = \
        limit_hvac_delta(item_input_object, item_output_object, final_tou_consumption, app_list, output_data,
                         processed_input_data, detected_cool, detected_heat)

    output_data = inference_engine_dict.get('output_data')
    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]

    final_tou_consumption = np.fmax(0, final_tou_consumption)

    # Zero consumption for living load on vacation days

    final_tou_consumption = final_pp_post_process(pp_idx, predicted, pp_type, final_tou_consumption, item_input_object)

    # final check to maintain seasonality of wh

    final_tou_consumption = \
        wh_seasonality_check(processed_input_data, wh_idx, final_tou_consumption, item_input_object, item_output_object)

    if run_hybrid_v2:
        final_tou_consumption = stat_app_consistency_check(final_tou_consumption, app_list, bc_list, vacation, logger)

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    # post process twh output to maintain ts level consistency

    final_tou_consumption = \
        twh_post_process(wh_idx, heat_idx, other_cons_arr, output_data, final_tou_consumption, item_input_object)

    #################### BC level postprocessing for all stat appliances  ################
    #################### This is done to maintain min/max consumption of these appliances based on pilot config  ################

    final_tou_consumption[ld_idx, vacation, :] = 0
    final_tou_consumption[ent_idx, vacation, :] = 0
    final_tou_consumption[cook_idx, vacation, :] = 0
    final_tou_consumption[li_idx, vacation, :] = 0

    ld_monthly_cons = ((np.sum(final_tou_consumption[ld_idx]) / length) * scaling_factor)
    ent_monthly_cons = ((np.sum(final_tou_consumption[ent_idx]) / length) * scaling_factor)
    cook_monthly_cons = ((np.sum(final_tou_consumption[cook_idx]) / length) * scaling_factor)
    total_monthly_cons = ((np.sum(processed_input_data) / length) * scaling_factor)

    hsm_cook2 = cook_monthly_cons
    hsm_ld2 = ld_monthly_cons
    hsm_ent2 = ent_monthly_cons

    if run_hybrid_v2:

        # fetching stat app hsm, to maintain consistency in run level output

        use_hsm, hsm_cook2, hsm_ent2, hsm_ld2 = \
            get_stat_app_hsm(item_input_object, processed_input_data, hsm_cook2, hsm_ent2, hsm_ld2)

        # maintain stat app bc level max consumption
        final_tou_consumption = maintain_min_cons(app_list, processed_input_data, final_tou_consumption, item_input_object, item_output_object, logger)

        app_month_cons = [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons]
        stat_hsm_output = [hsm_cook2, hsm_ent2, hsm_ld2]

        final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons = \
            apply_bc_level_max_limit_based_on_config(item_input_object, item_output_object, final_tou_consumption, length,
                                                     app_month_cons, stat_hsm_output, total_monthly_cons)

        # maintain stat app bc level min consumption

        final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons = \
            apply_soft_limit_on_bc_level_min_cons(item_input_object, item_output_object, final_tou_consumption, length,
                                                  [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons], total_monthly_cons,
                                                  logger)

    # limiting wh change from disagg

    final_tou_consumption = \
        limit_wh_delta(item_input_object, item_output_object, final_tou_consumption, app_list, output_data,
                       processed_input_data, hsm_wh, logger)

    final_tou_consumption, item_output_object = apply_additional_hybridv2_checks(item_input_object, item_output_object, final_tou_consumption, total_monthly_cons, logger)

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    other_cons_arr = np.reshape(other_cons_arr, newshape=(1, other_cons_arr.shape[0], other_cons_arr.shape[1]))

    other_cons_arr = np.fmax(0, other_cons_arr)

    final_tou_consumption = np.r_[total_consumption, final_tou_consumption, other_cons_arr]

    final_tou_consumption = np.fmax(0, final_tou_consumption)

    item_output_object['final_itemization'] = {
        'appliance_list': np.r_[['total'], app_list, ['other']],
        'tou_itemization': final_tou_consumption,
    }

    return item_output_object


def apply_additional_hybridv2_checks(item_input_object, item_output_object, final_tou_consumption, total_monthly_cons, logger):

    """
    This function serve as a master function which calls all the submodules to perform final 100% itemization

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')

    # Find saturation based on expected ranges

    inference_engine_dict = item_output_object.get('inference_engine_dict')

    output_data = inference_engine_dict.get('output_data')

    cons_cap_for_vac_days = 500

    app_list = np.array(inference_engine_dict.get('appliance_list'))

    pp_idx = get_idx('pp', app_list)
    cook_idx = get_idx('cook', app_list)
    ent_idx = get_idx('ent', app_list)
    ld_idx = get_idx('ld', app_list)
    wh_idx = get_idx('wh', app_list)
    ao_idx = get_idx('ao', app_list)
    ev_idx = get_idx('ev', app_list)
    ref_idx = get_idx('ref', app_list)
    li_idx = get_idx('li', app_list)
    heat_idx = get_idx('heating', app_list)
    cooling_idx = get_idx('cooling', app_list)

    scaling_factor = (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH)

    processed_input_data = copy.deepcopy(item_output_object.get('original_input_data'))

    logger.debug("Calculated ts level min/avg/max saturation")

    app_list = np.array(inference_engine_dict.get('appliance_list'))

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    vacation = np.sum(item_input_object.get("item_input_params").get("vacation_data"), axis=1) > 0
    vacation = np.logical_or(vacation, processed_input_data.sum(axis=1) == 0)
    length = np.sum(np.logical_not(vacation))
    length = max(1, length)

    sleep_hours = item_output_object.get("profile_attributes").get("sleep_hours")

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]

    pilot = item_input_object.get("config").get("pilot_id")

    if run_hybrid_v2:

        ld_monthly_cons = ((np.sum(final_tou_consumption[ld_idx]) / length) * scaling_factor)
        ent_monthly_cons = ((np.sum(final_tou_consumption[ent_idx]) / length) * scaling_factor)
        cook_monthly_cons = ((np.sum(final_tou_consumption[cook_idx]) / length) * scaling_factor)

        hsm_cook2 = cook_monthly_cons
        hsm_ld2 = ld_monthly_cons
        hsm_ent2 = ent_monthly_cons

        vacation = np.sum(item_input_object.get("item_input_params").get("vacation_data"), axis=1) > 0
        vacation = np.logical_or(vacation, processed_input_data.sum(axis=1) == 0)
        total_cons = processed_input_data[np.logical_not(vacation)]
        length = np.sum(np.logical_not(vacation))
        length = max(1, length)
        total_cons = ((np.sum(total_cons) / length) * scaling_factor)

        # fetching stat app hsm, to maintain consistency in run level output

        use_hsm, hsm_cook2, hsm_ent2, hsm_ld2 = \
            get_stat_app_hsm(item_input_object, processed_input_data, hsm_cook2, hsm_ent2, hsm_ld2)

        monthly_app_cons = [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons]
        app_hsm_data = [hsm_cook2, hsm_ent2, hsm_ld2]

        # maintain stat app bc level min consumption

        final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons, tou_list = \
            apply_bc_level_min_limit(total_cons, item_input_object, item_output_object, final_tou_consumption, length,
                                     monthly_app_cons, app_hsm_data, use_hsm)

        final_tou_consumption = maintain_min_cons(app_list, processed_input_data, final_tou_consumption, item_input_object, item_output_object, logger)

        #################### Additional postprocessing steps to maintain max ts level output and stat app consistency ################

        # apply max limit on bc level change in wh output compared to disagg

        # blocking low cons wh output

        other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

        final_tou_consumption = \
            pick_potential_ent_cons_from_residual(item_input_object, final_tou_consumption, app_list, other_cons_arr, sleep_hours)

        ld_monthly_cons = ((np.sum(final_tou_consumption[ld_idx]) / length) * scaling_factor)
        ent_monthly_cons = ((np.sum(final_tou_consumption[ent_idx]) / length) * scaling_factor)
        cook_monthly_cons = ((np.sum(final_tou_consumption[cook_idx]) / length) * scaling_factor)

        hsm_cook2 = cook_monthly_cons
        hsm_ld2 = ld_monthly_cons
        hsm_ent2 = ent_monthly_cons

        final_tou_consumption = np.fmax(0, final_tou_consumption)

        use_hsm, hsm_cook2, hsm_ent2, hsm_ld2 = \
            get_stat_app_hsm(item_input_object, processed_input_data, hsm_cook2, hsm_ent2, hsm_ld2)

        # maintain stat app bc level max consumption

        app_month_cons = [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons]
        stat_hsm_output = [hsm_cook2, hsm_ent2, hsm_ld2]

        final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons = \
            apply_bc_level_max_limit_based_on_config(item_input_object, item_output_object, final_tou_consumption, length,
                                                     app_month_cons, stat_hsm_output, total_monthly_cons)

        # maintain stat app bc level consistency

        final_tou_consumption = \
            consistency_check_for_all_output_based_on_stat_estimation(processed_input_data, item_input_object, final_tou_consumption, app_list, bc_list,
                                                                      vacation)

        #################### apply ts level postprocessing for wh output ################
        #################### This includes blocking feeble consumption and adding thin pulses ################

        # blocking low cons wh output

    final_tou_consumption = \
        block_low_cons_wh(wh_idx, pilot, item_input_object, item_output_object, output_data, final_tou_consumption,
                          app_list, bc_list, vacation, logger)

    # blocking wh in bc with feeble consumption

    final_tou_consumption = \
        block_feeble_cons_in_wh(output_data, wh_idx, item_input_object, final_tou_consumption, app_list, bc_list, logger)

    final_tou_consumption = \
        block_outlier_points(final_tou_consumption, output_data, processed_input_data, app_list, logger)

    # adding wh if needed based on app prof

    # applying maximum ts level consumption limit on true disagg appliances

    wh_copy = copy.deepcopy(final_tou_consumption[wh_idx])

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    box_label, box_cons4, box_seq = \
        box_detection(pilot, other_cons_arr, np.fmax(0, other_cons_arr), np.zeros_like(other_cons_arr),
                      min_amp=200 / samples_per_hour, max_amp=10000 / samples_per_hour, min_len=1,
                      max_len=3 * samples_per_hour, detect_wh=1)

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    box_cons4 = np.fmax(0, np.minimum(other_cons_arr, box_cons4 * 10))

    # allot leftover thin pulses in residual to wh appliance

    final_tou_consumption = \
        allot_thin_pulse_boxes(item_input_object, item_output_object, final_tou_consumption, app_list, box_seq, logger)

    # blocking wh in bc with feeble consumption

    final_tou_consumption = \
        block_feeble_cons_in_wh(output_data, wh_idx, item_input_object, final_tou_consumption, app_list, bc_list, logger)

    final_tou_consumption = \
        block_feeble_cons_thin_pulse(wh_copy, item_input_object, final_tou_consumption, app_list, bc_list, logger)

    final_tou_consumption[ld_idx, vacation, :] = 0
    final_tou_consumption[ent_idx, vacation, :] = 0
    final_tou_consumption[cook_idx, vacation, :] = 0
    final_tou_consumption[li_idx, vacation, :] = 0
    final_tou_consumption[ev_idx, vacation, :] = 0

    final_tou_consumption = np.fmax(0, final_tou_consumption)

    item_input_object = \
        update_output_object_for_hld_change_cases(final_tou_consumption, item_input_object, output_data, wh_idx,
                                                  pp_idx, ev_idx, logger)

    #################### Update stat app bc level consumption inorder to decrease leftoover others ################


    final_tou_consumption = \
        apply_max_ts_level_limit(item_input_object, item_output_object, processed_input_data, app_list,
                                 output_data, final_tou_consumption)
    if run_hybrid_v2:

        final_tou_consumption, box_cons4 =  \
            add_box_cons_to_stat_app(pilot, final_tou_consumption, box_cons4, processed_input_data, app_list)

        app_month_cons = [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons]
        stat_hsm_output = [hsm_cook2, hsm_ent2, hsm_ld2]

        final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons = \
            apply_bc_level_max_limit_based_on_config_and_hsm(item_input_object, item_output_object, final_tou_consumption,
                                                             app_month_cons, stat_hsm_output, logger)

        final_tou_consumption[ld_idx, vacation, :] = 0
        final_tou_consumption[ent_idx, vacation, :] = 0
        final_tou_consumption[cook_idx, vacation, :] = 0
        final_tou_consumption[li_idx, vacation, :] = 0

        ld_monthly_cons = ((np.sum(final_tou_consumption[ld_idx]) / length) * scaling_factor)
        ent_monthly_cons = ((np.sum(final_tou_consumption[ent_idx]) / length) * scaling_factor)
        cook_monthly_cons = ((np.sum(final_tou_consumption[cook_idx]) / length) * scaling_factor)

        logger.info('cook step1 1 | %s', int(np.nan_to_num(cook_monthly_cons)))
        logger.info('ent step1 1 | %s', int(np.nan_to_num(ent_monthly_cons)))
        logger.info('ld step1 1 | %s', int(np.nan_to_num(ld_monthly_cons)))
        logger.info('cool step1 1 | %s', int( ((np.sum(final_tou_consumption[cooling_idx]) / length) * scaling_factor)))

        # increase stat app for bc with high others fraction

        final_tou_consumption = \
            consistency_check_for_low_cons_app(final_tou_consumption, app_list, bc_list, vacation, logger)

        ld_monthly_cons = ((np.sum(final_tou_consumption[ld_idx]) / length) * scaling_factor)
        ent_monthly_cons = ((np.sum(final_tou_consumption[ent_idx]) / length) * scaling_factor)
        cook_monthly_cons = ((np.sum(final_tou_consumption[cook_idx]) / length) * scaling_factor)

        monthly_app_cons = [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons]
        app_hsm_data = [hsm_cook2, hsm_ent2, hsm_ld2]

        final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons = \
            apply_bc_level_min_limit_for_stat_app(total_cons, item_input_object, item_output_object, final_tou_consumption, length,
                                                  monthly_app_cons, app_hsm_data, tou_list, use_hsm)

        final_tou_consumption = apply_bc_level_min_limit_for_step3_app(total_cons, item_input_object, item_output_object, final_tou_consumption, logger)

        final_tou_consumption = apply_consistency_in_neighbouring_bcs(item_input_object, final_tou_consumption, vacation, bc_list, app_list)

        final_tou_consumption = \
            increase_stat_app_in_bc_with_high_others(item_input_object, item_output_object, final_tou_consumption,
                                                     processed_input_data, bc_list, app_list, vacation, li_idx, logger)

        if final_tou_consumption[li_idx].sum() > 0:
            final_tou_consumption[li_idx] = np.fmin(final_tou_consumption[li_idx], np.percentile(final_tou_consumption[li_idx][final_tou_consumption[li_idx] > 0], 97))

        # maintain max consumption of stat app at bc level

        app_month_cons = [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons]
        stat_hsm_output = [hsm_cook2, hsm_ent2, hsm_ld2]

        final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons = \
            apply_bc_level_max_limit_based_on_config_and_hsm(item_input_object, item_output_object, final_tou_consumption,
                                                             app_month_cons, stat_hsm_output, logger)

        final_tou_consumption = \
            add_leftover_potential_boxes_to_hvac(final_tou_consumption, item_input_object, pilot, processed_input_data,
                                                 samples_per_hour, cooling_idx, heat_idx)

        if np.sum(final_tou_consumption[ref_idx] > 0) and ('ref' not in item_input_object["item_input_params"]["backup_app"]):
            final_tou_consumption[ref_idx] = \
                np.fmin(final_tou_consumption[ref_idx],
                        np.percentile(final_tou_consumption[ref_idx][final_tou_consumption[ref_idx] > 0], 90))

        final_tou_consumption[ld_idx, vacation, :] = 0
        final_tou_consumption[ent_idx, vacation, :] = 0
        final_tou_consumption[cook_idx, vacation, :] = 0
        final_tou_consumption[li_idx, vacation, :] = 0
        final_tou_consumption[ev_idx, vacation, :] = 0

    days = np.percentile(processed_input_data, 35, axis=1) < cons_cap_for_vac_days/samples_per_hour

    final_tou_consumption[cooling_idx, np.logical_and(vacation, days), :] = 0
    final_tou_consumption[heat_idx, np.logical_and(vacation, days), :] = 0

    final_tou_consumption[ev_idx][np.logical_and(vacation, np.sum(output_data[ev_idx], axis=1) == 0)] = 0

    if run_hybrid_v2:
        # blocking stat app in bc with feeble consumption

        final_tou_consumption = \
            block_stat_app_feeble_cons(item_input_object, final_tou_consumption, app_list, bc_list, vacation, logger)

        final_tou_consumption = maintain_min_cons(app_list, processed_input_data, final_tou_consumption, item_input_object, item_output_object, logger)

        # safety checks on stat app output based on app profile input

        final_tou_consumption = \
            modify_stat_app_based_on_app_prof(final_tou_consumption, ld_idx, ent_idx, cook_idx, vacation)

        #################### Update HSM based on final itemization output ################

        final_tou_consumption, item_output_object = \
            update_stat_app_hsm(final_tou_consumption, item_output_object, item_input_object, ld_idx, ent_idx, cook_idx,
                                length, logger)

        final_tou_consumption[ev_idx, vacation, :] = 0

        final_tou_consumption =  block_low_cons_billing_cycle_output(item_input_object, final_tou_consumption,
                                                                     cooling_idx, heat_idx, pp_idx, ev_idx,
                                                                     ref_idx, wh_idx, li_idx, logger)

        final_tou_consumption = block_feeble_cons_in_pp(pp_idx, final_tou_consumption)

        app_killer = item_input_object.get('item_input_params').get('app_killer')
        final_tou_consumption[np.array(app_killer).astype(bool)] = 0

        final_tou_consumption = \
            handle_leftover_neg_res_points(final_tou_consumption, ao_idx, samples_per_hour, processed_input_data)

        other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

        final_tou_consumption = \
            set_min_others(final_tou_consumption, other_cons_arr, processed_input_data, item_input_object, item_output_object,
                           app_list, processed_input_data)

    if np.sum(final_tou_consumption[ev_idx]) > 0 and (np.sum(output_data[ev_idx]) == 0):

        if ('ev' in item_input_object["item_input_params"]["backup_app"]):
            logger.info('EV detected by step 3 addition | ')
        else:
            logger.info('EV detected by step 2 addition | ')

    elif np.sum(final_tou_consumption[ev_idx]) > 0 and np.sum(output_data[ev_idx]) > 0:
        logger.info('EV detected by step 1 addition | ')

    logger.info('neg points present | %s', (np.any(other_cons_arr < -0.1) or np.any(final_tou_consumption < -0.1)))

    # updating ev/pp/wh mtd output based on hsm information

    item_output_object = update_pp_hsm(output_data, final_tou_consumption, pp_idx, length, item_input_object, item_output_object, logger)
    item_output_object = update_ev_hsm(final_tou_consumption, ev_idx, length, item_input_object, item_output_object, logger)
    item_output_object = update_wh_hsm(final_tou_consumption, wh_idx, length, item_input_object, item_output_object, logger)
    item_output_object = update_li_hsm(final_tou_consumption, li_idx, length, item_input_object, item_output_object, logger)
    item_output_object = update_ref_hsm(final_tou_consumption, ref_idx, length, item_input_object, item_output_object, logger)

    return final_tou_consumption, item_output_object


def set_min_others(final_tou_consumption, others, input_data, item_input_object, item_output_object, app_list, processed_input_data):

    """
    Maintain minimum others, if others value is less than threshold in a billing cycle,
     consumption is moved from cook/ent/ld to others

    Parameters:
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        others                      (np.ndarray)    : ts level leftover others
        input_data                  (np.ndarray)    : raw input data
        item_input_object           (dict)          : Dict containing all hybrid inputs
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    app_killer = item_input_object['item_input_params']['monthly_app_killer']
    date_list = item_output_object.get("date_list")
    month_list = pd.DatetimeIndex(date_list).month.values - 1
    pilot_config = item_input_object.get('pilot_level_config')

    cook_idx = get_idx('cook', app_list)
    ent_idx = get_idx('ent', app_list)
    ld_idx = get_idx('ld', app_list)
    ao_idx = get_idx('ao', app_list)

    for app in app_list:
        index = get_idx(app_list, app)
        target_months = np.where(app_killer[index] > 0)[0]
        target_months = np.isin(month_list, target_months)
        final_tou_consumption[index][target_months] = 0

        if (pilot_config.get(app + '_config') is not None) and (int(np.nan_to_num(pilot_config.get(app + '_config').get('coverage'))) == 0):
            final_tou_consumption[index] = 0

    others = np.fmax(others, 0)

    ent_cons = final_tou_consumption[ent_idx]
    cook_cons = final_tou_consumption[cook_idx]
    ld_cons = final_tou_consumption[ld_idx]

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    cook_bc_cons = np.zeros_like(unique_bc)
    ld_bc_cons = np.zeros_like(unique_bc)
    ent_bc_cons = np.zeros_like(unique_bc)

    for i in range(len(unique_bc)):

        required_monthly_residual = 0.01 * np.sum(input_data[bc_list == unique_bc[i]])

        cook_bc_cons[i] = np.sum(cook_cons[bc_list == unique_bc[i]])
        ent_bc_cons[i] = np.sum(ent_cons[bc_list == unique_bc[i]])
        ld_bc_cons[i] = np.sum(ld_cons[bc_list == unique_bc[i]])

        diff = np.fmax(0, required_monthly_residual - np.sum(others[bc_list == unique_bc[i]]))

        if diff > 0 and ld_bc_cons[i] > 0:
            factor = diff / ld_bc_cons[i]
            factor = min(0.5, max(0, factor))

            final_tou_consumption[ld_idx][bc_list == unique_bc[i]] = final_tou_consumption[ld_idx][bc_list == unique_bc[i]] * (1 - factor)

        others = input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)
        diff = np.fmax(0, required_monthly_residual - np.sum(others[bc_list == unique_bc[i]]))

        if diff > 0 and ent_bc_cons[i] > 0:
            factor = diff / ent_bc_cons[i]
            factor = min(0.5, max(0, factor))

            final_tou_consumption[ent_idx][bc_list == unique_bc[i]] = final_tou_consumption[ent_idx][bc_list == unique_bc[i]] * (1 - factor)

        others = input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)
        diff = np.fmax(0, required_monthly_residual - np.sum(others[bc_list == unique_bc[i]]))

        if diff > 0 and cook_bc_cons[i] > 0:

            factor = diff / cook_bc_cons[i]

            factor = min(0.5, max(0, factor))

            final_tou_consumption[cook_idx][bc_list == unique_bc[i]] = final_tou_consumption[cook_idx][bc_list == unique_bc[i]] * (1 - factor)

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    final_tou_consumption[ao_idx] = final_tou_consumption[ao_idx] + np.fmin(0, other_cons_arr)

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    final_tou_consumption[ao_idx] = final_tou_consumption[ao_idx] + np.fmin(0, other_cons_arr)

    return final_tou_consumption


def update_increament_based_on_app_survey(item_input_object, final_adj_arr, app_list):

    """
    This function serve as a master function which calls all the submodules to perform final 100% itemization

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    cook_idx = get_idx(app_list, 'cook')

    if not item_input_object.get("appliance_profile").get("default_cooking_flag"):

        cooking_app_count = copy.deepcopy(item_input_object.get("appliance_profile").get("cooking"))
        cooking_app_type = item_input_object.get("appliance_profile").get("cooking_type")

        if str(np.nan_to_num(item_input_object.get('pilot_level_config').get('cook_config').get('type'))) == 'GAS':
            cooking_app_count[cooking_app_type == 2] = cooking_app_count[cooking_app_type == 2] * 2
            cooking_app_count[cooking_app_type == 0] = item_input_object.get("appliance_profile").get("default_cooking_count")[cooking_app_type == 0]
        else:
            cooking_app_count[cooking_app_type == 0] = cooking_app_count[cooking_app_type == 0] * 0

        app_profile = item_input_object.get("app_profile").get(31)
        hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

        if hybrid_config.get("dishwash_cat") == "cook":
            if app_profile is not None:
                app_profile = app_profile.get("number", 0)
            else:
                app_profile = 0
        else:
            app_profile = 0

        cooking_app_count = cooking_app_count.sum() + app_profile * 2
        scaling_factor_based_on_app_survey = (cooking_app_count / item_input_object.get("appliance_profile").get("default_cooking_count").sum())

        final_adj_arr[cook_idx][final_adj_arr[cook_idx] > 1] = \
            final_adj_arr[cook_idx][final_adj_arr[cook_idx] > 1] * min(1, scaling_factor_based_on_app_survey)

    ent_idx = get_idx(app_list, 'ent')

    if not item_input_object.get("appliance_profile").get("default_ent_flag"):
        ent = item_input_object.get("appliance_profile").get("ent")
        cons = [200, 200, 700]
        scaling_factor_based_on_app_survey = max(0.5, min(1, (np.dot(ent, cons)/ 700)))

        final_adj_arr[ent_idx][final_adj_arr[ent_idx] > 1] = \
            final_adj_arr[ent_idx][final_adj_arr[ent_idx] > 1] * min(1, scaling_factor_based_on_app_survey)

    return final_adj_arr
