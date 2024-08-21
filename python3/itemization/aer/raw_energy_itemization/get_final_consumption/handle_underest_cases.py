

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


def handle_case_b(item_input_object, item_output_object, item_adjustment_dict, final_tou_consumption, case_b_bool,
                  appliance_mid_values, appliance_max_values, logger):

    """
    This function is called for with raw energy greater than total estimated consumption and
    less than sum of max consumption possible

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        item_adjustment_dict        (dict)          : DIct containing parameters required for adjustment
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        case_b_bool                 (np.ndarray)    : case b boolean array
        appliance_mid_values        (np.ndarray)    : appliance ts level mid consumption
        appliance_max_values        (np.ndarray)    : appliance ts level max consumption
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    case_b_t0 = datetime.now()

    config = init_final_item_conf().get("case_b_conf")

    inference_engine_dict = item_output_object.get('inference_engine_dict')
    processed_input_data = item_output_object.get('original_input_data')

    appliance_list = np.array(inference_engine_dict.get('appliance_list'))
    min_saturation_arr = item_adjustment_dict.get('min_saturation_arr')
    conf_arr = inference_engine_dict.get('appliance_conf')
    num_appliances = len(appliance_list)

    # Set consumption to maximum for all appliances and then bring them down to min saturation

    case_b_bool_3d = np.tile(case_b_bool, reps=(num_appliances, 1, 1))
    final_tou_consumption[case_b_bool_3d] = appliance_max_values[case_b_bool_3d]

    # Initialize available consumption to start with distribution

    reducible_consumption_case_b = \
        np.sum(final_tou_consumption, axis=0) - np.multiply(min_saturation_arr, processed_input_data)
    reducible_consumption_case_b[~case_b_bool] = 0

    if not item_input_object.get("item_input_params").get("timed_wh_user"):
        case_a_adj_order = config.get("non_twh_app_seq")
    else:
        case_a_adj_order = config.get("twh_app_seq")

    case_b_adj_order = np.flip(case_a_adj_order)

    for app_name in case_b_adj_order:

        logger.debug("Adjusting appliance for case B | %s", app_name)

        # For a given appliance check the confidence value at the given points

        app_idx = np.where(appliance_list == app_name)[0][0]

        app_mid_arr = appliance_mid_values[app_idx]

        # Extract confidence array for user

        app_conf_arr = conf_arr[app_idx, :, :]
        app_cons_arr = final_tou_consumption[app_idx, :, :]

        # Prepare data for neighborhood check

        cons_left_shift = np.c_[app_mid_arr[:, 1:], app_mid_arr[:, 0]]
        cons_right_shift = np.c_[app_mid_arr[:, -1], app_mid_arr[:, :-1]]

        conf_left_shift = np.c_[app_conf_arr[:, 1:], app_conf_arr[:, 0]]
        conf_right_shift = np.c_[app_conf_arr[:, -1], app_conf_arr[:, :-1]]

        # Maximum adjustment we want to allow is to mid here

        app_max_adjustment = np.maximum(app_cons_arr - appliance_mid_values[app_idx, :, :], 0)

        # check neighbouring points for adjustment guidance

        adj_arr = np.maximum(0.5 * ((app_cons_arr - cons_left_shift) + (app_cons_arr - cons_right_shift)), 0)
        adj_arr[~case_b_bool] = 0

        adj_factor = np.minimum(1 + 0.5 * (conf_left_shift + conf_right_shift) - app_conf_arr, 1)
        adj_factor[~case_b_bool] = 0

        adj_arr = np.multiply(adj_factor, adj_arr)

        if app_name in ['cook', 'ent', 'ld', 'li', 'wh', 'pp', 'ev']:
            adj_arr[:] = np.inf

        adj_arr = np.minimum(np.minimum(adj_arr, reducible_consumption_case_b), app_max_adjustment)

        # Update the consumption array and the available array

        final_tou_consumption[app_idx, :, :] = app_cons_arr - adj_arr
        reducible_consumption_case_b = reducible_consumption_case_b - adj_arr

    case_b_t1 = datetime.now()

    logger.info("Case B adjustment took | | %.3f", get_time_diff(case_b_t0, case_b_t1))

    return final_tou_consumption


def handle_case_d(item_output_object, item_adjustment_dict, final_tou_consumption, case_d_bool, appliance_max_values,
                  denominator_input_data, logger):

    """
    This function is called for with raw energy greater than total estimated consumption and
    and also greater than sum of max consumption possible

    Parameters:
        item_output_object        (dict)          : Dict containing all hybrid outputs
        item_adjustment_dict        (dict)          : DIct containing parameters required for adjustment
        final_tou_consumption       (np.ndarray)    : initialized appliance ts level estimates (before adjustment)
        case_d_bool                 (np.ndarray)    : case d boolean array
        appliance_max_values        (np.ndarray)    : appliance ts level max consumption
        denominator_input_data      (np.ndarray)    : input data which 0 values been replaced
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : appliance ts level estimates (after adjustment)
    """

    case_d_t0 = datetime.now()

    inference_engine_dict = item_output_object.get('inference_engine_dict')

    appliance_list = np.array(inference_engine_dict.get('appliance_list'))
    min_saturation_arr = item_adjustment_dict.get('min_saturation_arr')
    num_appliances = len(appliance_list)

    case_d_bool_3d = np.tile(case_d_bool, reps=(num_appliances, 1, 1))

    adj_arr = copy.deepcopy(appliance_max_values)

    new_sat = np.divide(np.sum(adj_arr, axis=0), denominator_input_data)

    div_arr = copy.deepcopy(new_sat)
    div_arr = np.divide(div_arr, min_saturation_arr + 1e-3)

    div_arr[new_sat <= min_saturation_arr] = 1
    div_arr[~case_d_bool] = 1

    # Scale the consumption by the saturation in case the saturation is still wrong after max

    adj_arr = np.divide(adj_arr, div_arr)
    final_tou_consumption[case_d_bool_3d] = adj_arr[case_d_bool_3d]

    case_d_t1 = datetime.now()

    logger.info("Case D adjustment took | | %.3f", get_time_diff(case_d_t0, case_d_t1))

    return final_tou_consumption
