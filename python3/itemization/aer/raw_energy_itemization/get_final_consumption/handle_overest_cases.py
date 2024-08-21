
"""
Author - Mayank Sharan
Date - 4th April 2021
Stage 3 of itemization framework - calculate final tou level estimation
"""

# Import python packages

import copy
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.init_final_item_config import init_final_item_conf


def handle_case_a(item_input_object, item_output_object, item_adjustment_dict, final_tou_consumption, case_a_bool,
                  appliance_min_values, appliance_mid_values, appliance_max_values, logger):

    """
    This function is called for with raw energy less than total estimated consumption,
    but greater than sum of min consumption possible

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        item_adjustment_dict        (dict)          : DIct containing parameters required for adjustment
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        case_a_bool                 (np.ndarray)    : case a boolean array
        appliance_min_values        (np.ndarray)    : appliance ts level min consumption
        appliance_mid_values        (np.ndarray)    : appliance ts level mid consumption
        appliance_max_values        (np.ndarray)    : appliance ts level max consumption
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    case_a_t0 = datetime.now()

    config = init_final_item_conf().get("case_a_conf")

    inference_engine_dict = item_output_object.get('inference_engine_dict')
    processed_input_data = item_output_object.get('original_input_data')

    appliance_list = np.array(inference_engine_dict.get('appliance_list'))
    max_saturation_arr = item_adjustment_dict.get('max_saturation_arr')
    mid_saturation_arr = item_adjustment_dict.get('mid_saturation_arr')
    conf_arr = inference_engine_dict.get('appliance_conf')
    num_appliances = len(appliance_list)

    appliance_min_sum = np.sum(appliance_min_values, axis=0)
    denominator_input_data = copy.deepcopy(processed_input_data)
    invalid_input_data_bool = denominator_input_data == 0
    denominator_input_data[invalid_input_data_bool] = 1e-3

    appliance_min_saturation = np.divide(appliance_min_sum, denominator_input_data)

    case_a_bool_3d = np.tile(case_a_bool, reps=(num_appliances, 1, 1))
    ratio_arr = np.divide(mid_saturation_arr, appliance_min_saturation + 1e-3)

    final_tou_consumption[case_a_bool_3d] = np.multiply(appliance_min_values, ratio_arr)[case_a_bool_3d]
    final_tou_consumption[case_a_bool_3d] = np.minimum(final_tou_consumption[case_a_bool_3d],
                                                       appliance_max_values[case_a_bool_3d])
    final_tou_consumption[case_a_bool_3d] = np.maximum(final_tou_consumption[case_a_bool_3d],
                                                       appliance_min_values[case_a_bool_3d])

    # Initialize available consumption to start with distribution

    available_consumption_case_a = \
        np.multiply(max_saturation_arr - 1e-3, processed_input_data) - np.sum(final_tou_consumption, axis=0)
    available_consumption_case_a[~case_a_bool] = 0

    logger.info("Timed wh user bool value | %s", item_input_object.get("item_input_params").get("timed_wh_user"))

    if not item_input_object.get("item_input_params").get("timed_wh_user"):
        case_a_adj_order = config.get("non_twh_app_seq")
    else:
        case_a_adj_order = config.get("twh_app_seq")

    app_conf_arr_copy_all = copy.deepcopy(conf_arr)

    for app_name in case_a_adj_order:

        logger.debug("Adjusting appliance for case A | %s", app_name)

        # For a given appliance check the confidence value at the given points

        app_idx = np.where(appliance_list == app_name)[0][0]

        # Extract confidence array for user

        app_conf_arr_copy = app_conf_arr_copy_all[app_idx, :, :]
        app_cons_arr = final_tou_consumption[app_idx, :, :]

        app_mid_arr = appliance_mid_values[app_idx]

        # Prepare data for neighborhood check

        cons_left_shift = np.c_[app_mid_arr[:, 1:], app_mid_arr[:, 0]]
        cons_right_shift = np.c_[app_mid_arr[:, -1], app_mid_arr[:, :-1]]

        conf_left_shift = np.c_[app_conf_arr_copy[:, 1:], app_conf_arr_copy[:, 0]]
        conf_right_shift = np.c_[app_conf_arr_copy[:, -1], app_conf_arr_copy[:, :-1]]

        # Maximum adjustment we want to allow is to mid here

        app_max_adjustment = np.maximum(appliance_mid_values[app_idx, :, :] - app_cons_arr, 0)

        # check neighbouring points for adjustment guidance

        adj_arr = np.maximum(0.5 * ((cons_left_shift - app_cons_arr) + (cons_right_shift - app_cons_arr)), 0)
        adj_arr[~case_a_bool] = 0

        adj_factor = np.minimum(1 + app_conf_arr_copy - 0.5 * (conf_left_shift + conf_right_shift), 1)
        adj_factor[~case_a_bool] = 0

        adj_arr[:] = np.inf

        adj_arr = np.minimum(np.minimum(adj_arr, available_consumption_case_a), app_max_adjustment)

        # Update the consumption array and the available array

        final_tou_consumption[app_idx, :, :] = app_cons_arr + adj_arr
        available_consumption_case_a = available_consumption_case_a - adj_arr

    case_a_t1 = datetime.now()

    logger.info("Case A adjustment took | | %.3f", get_time_diff(case_a_t0, case_a_t1))

    return final_tou_consumption


def handle_case_c(item_input_object, item_output_object, item_adjustment_dict, final_tou_consumption,
                  inputs_for_case_c_resolution, logger):

    """
    This function is called for points with total raw energy less than total estimated consumption and also
    less than sum of min consumption possible

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        item_adjustment_dict        (dict)          : DIct containing parameters required for adjustment
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        inputs_for_case_c_resolution(dict)          : inputs required for handling case c epochs
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
        case_c_bool_3d              (np.ndarray)    : bool array denoting the points where consumption is getting overestimated for individual app

    """

    case_c_t0 = datetime.now()

    inference_engine_dict = item_output_object.get('inference_engine_dict')
    app_list = np.array(inference_engine_dict.get('appliance_list'))
    num_appliances = len(app_list)

    config = init_final_item_conf().get("case_c_conf")

    inference_engine_dict = item_output_object.get('inference_engine_dict')
    processed_input_data = item_output_object.get('original_input_data')

    appliance_list = np.array(inference_engine_dict.get('appliance_list'))
    max_saturation_arr = item_adjustment_dict.get('max_saturation_arr')
    conf_arr = inference_engine_dict.get('appliance_conf')

    ao_idx = inputs_for_case_c_resolution.get('ao_idx')
    ref_idx = inputs_for_case_c_resolution.get('ref_idx')
    case_c_bool = inputs_for_case_c_resolution.get('case_c_bool')
    appliance_min_values = inputs_for_case_c_resolution.get('appliance_min_values')

    eligible_bool = copy.deepcopy(case_c_bool)

    appliance_min_sum = np.sum(appliance_min_values, axis=0)
    denominator_input_data = copy.deepcopy(processed_input_data)
    invalid_input_data_bool = denominator_input_data == 0
    denominator_input_data[invalid_input_data_bool] = 1e-3

    appliance_min_saturation = np.divide(appliance_min_sum, denominator_input_data)

    overestimated_cons = np.multiply(appliance_min_saturation - max_saturation_arr, denominator_input_data)
    overestimated_cons = np.fmax(0, overestimated_cons)

    # For each point in case C process by reducing / removing appliances

    adj_arr = copy.deepcopy(appliance_min_values)
    conf_arr_case_c = copy.deepcopy(conf_arr)

    if not item_input_object.get("item_input_params").get("timed_wh_user"):
        case_c_adj_order = config.get("non_twh_app_seq")
    else:
        case_c_adj_order = config.get("twh_app_seq")

    for app_name in case_c_adj_order:

        logger.debug("Adjusting appliance for case C | %s", app_name)

        if not np.sum(eligible_bool):
            break

        index = np.where(appliance_list == app_name)[0][0]

        # Kill consumption for all appliances at less than 0.5 confidence

        cons_to_be_killed = np.minimum(adj_arr[index], overestimated_cons)

        cons_to_be_killed[np.logical_or(conf_arr_case_c[index] > 0.5, np.logical_not(eligible_bool))] = 0

        final_tou_consumption[index] = final_tou_consumption[index] - cons_to_be_killed

        overestimated_cons = overestimated_cons - cons_to_be_killed

        eligible_bool = overestimated_cons > 0

    case_c_t1 = datetime.now()

    logger.info("Case C adjustment took | | %.3f", get_time_diff(case_c_t0, case_c_t1))

    case_c_bool_3d = np.tile(eligible_bool, reps=(num_appliances, 1, 1))

    new_sat = np.divide(np.sum(adj_arr, axis=0), denominator_input_data)
    div_arr = copy.deepcopy(new_sat + 1e-3)
    div_arr[div_arr <= max_saturation_arr] = 1
    div_arr[~eligible_bool] = 1

    # Scale the consumption by the saturation in case the saturation is still over 1 after appliance kill

    adj_arr = np.divide(adj_arr, div_arr)

    ao_cons = copy.deepcopy(final_tou_consumption[ao_idx])
    ref_cons = copy.deepcopy(final_tou_consumption[ref_idx])

    final_tou_consumption[case_c_bool_3d] = adj_arr[case_c_bool_3d]

    final_tou_consumption[ao_idx] = ao_cons
    final_tou_consumption[ref_idx] = ref_cons

    return final_tou_consumption, case_c_bool_3d
