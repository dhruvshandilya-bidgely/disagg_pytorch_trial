
"""
Author - Nisha Agarwal
Date - 4th April 2021
contain functions to performs feeble consumption blocking
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.functions.get_hybrid_v2_generic_config import get_hybrid_v2_generic_config

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def block_outlier_points(final_tou_consumption, output_data, processed_input_data, app_list, logger):

    """
    This function performs feeble consumption blocking by removing WH consumption if it is detected in very few days

    Parameters:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        output_data               (np.ndarray)    : disagg output data
        processed_input_data      (np.ndarray)    : input data
        app_list                  (list)          : list of appliances
        logger                    (logger)        : logger object

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    config = get_inf_config(samples_per_hour=1).get("wh")

    upper_perc_limit = config.get('upper_perc_limit')
    app = "wh"
    app_idx = np.where(app_list == app)[0][0]

    if np.sum(output_data[app_idx]) > 0:
        upper_perc_limit = config.get('disagg_upper_perc_limit')

    if np.sum(final_tou_consumption[app_idx]):
        limit = np.percentile(final_tou_consumption[app_idx][final_tou_consumption[app_idx] > 0], upper_perc_limit)
        final_tou_consumption[app_idx] = np.fmin(final_tou_consumption[app_idx], limit)
        final_tou_consumption[app_idx] = np.minimum(processed_input_data, final_tou_consumption[app_idx])

    # remove WH consumption if it is detected in very few days

    if (np.sum(final_tou_consumption[app_idx], axis=1) > 0).sum() < (0.05 * len(final_tou_consumption[0])):
        final_tou_consumption[app_idx, :, :] = 0
        logger.info('blocking low consumption wh after outlier blocking | ')

    return final_tou_consumption


def block_feeble_cons_in_wh(output_data, wh_idx, item_input_object, final_tou_consumption, app_list, bc_list, logger):

    """
    This function performs postprocessing on itemized wh output to block feeble consumption

    Parameters:
        output_data               (np.ndarray)    : array containing final ts level disagg output
        wh_idx                    (int)           : mapping for pp output
        item_input_object         (dict)          : Dict containing all hybrid inputs
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        app_list                  (list)          : list of appliances
        bc_list                   (list)          : list of bc start timestamp
        logger                    (logger)        : logger

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')
    config = get_inf_config(samples_per_hour).get("wh")

    min_points_required = config.get('min_box_req')
    min_cons_required = config.get('feeble_cons_thres') * Cgbdisagg.WH_IN_1_KWH

    disagg_cons_thres = config.get('disagg_cons_thres')
    max_cap_on_minimum_wh_days_required_in_month = config.get('max_cap_on_minimum_wh_days_required_in_month')
    min_cap_on_minimum_wh_days_required_in_month = config.get('min_cap_on_minimum_wh_days_required_in_month')
    minimum_wh_days_required_in_month = config.get('minimum_wh_days_required_in_month')

    if item_input_object.get('item_input_params').get('tankless_wh') > 0:
        min_points_required = config.get('tankless_wh_min_box_req')

    app = 'wh'

    app_idx = np.where(app_list == app)[0][0]

    if np.sum(final_tou_consumption[app_idx]) == 0:
        return final_tou_consumption

    # checking presence of feeble WH consumption in cases where WH is added from hybrid module rather than disagg

    run_feeble_cons_blocking_flag = ((np.sum(final_tou_consumption[app_idx]) == 0) or (item_input_object.get('item_input_params').get('swh_hld') > 0))

    unique_bc, reverse_idx, bc_days_count = np.unique(bc_list, return_inverse=True, return_counts=1)
    disagg_wh_output = np.round(np.bincount(reverse_idx, weights=output_data[wh_idx].sum(axis=1)), 2)
    hybrid_wh_output = np.round(np.bincount(reverse_idx, weights=final_tou_consumption[app_idx].sum(axis=1)), 2)
    hybrid_wh_points_count = np.round(np.bincount(reverse_idx, weights=(final_tou_consumption[app_idx]>0).sum(axis=1)), 2)

    check_feeble_cons_flag = np.logical_or(bc_days_count < 3, disagg_wh_output > disagg_cons_thres)

    feeble_wh_cons_present = np.logical_or(hybrid_wh_points_count < min_points_required, (np.divide(hybrid_wh_output, bc_days_count) * Cgbdisagg.DAYS_IN_MONTH) < min_cons_required)

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if run_feeble_cons_blocking_flag or check_feeble_cons_flag[i]:
            continue

        if feeble_wh_cons_present[i]:
            final_tou_consumption[app_idx][target_days] = 0
            logger.info("blocked wh due to feeble consumption | %s", unique_bc[i])

    min_points_required_swh = config.get('swh_min_box_req')

    min_points_required = config.get('wh_min_box_req')
    bc_level_cons_thres = config.get('wh_feeble_cons_thres')

    if item_input_object["item_input_params"]["timed_wh_user"] > 0:
        min_points_required = config.get('twh_min_box_req')
        bc_level_cons_thres = config.get('twh_feeble_cons_thres')

    hybrid_wh_output = np.round(np.bincount(reverse_idx, weights=final_tou_consumption[app_idx].sum(axis=1)), 2)
    hybrid_wh_points_count = np.round(np.bincount(reverse_idx, weights=(final_tou_consumption[app_idx]>0).sum(axis=1)), 2)

    # checking presence of feeble SWH consumption in cases where WH is added from hybrid module rather than disagg

    feeble_wh_cons_present_flag1 = np.logical_and( hybrid_wh_points_count < min_points_required_swh, bc_days_count < 3)
    feeble_wh_cons_present_flag1 = np.logical_and( disagg_wh_output == 0, feeble_wh_cons_present_flag1)

    # checking presence of feeble WH consumption in cases where WH is added from true disagg module
    feeble_wh_cons_present_flag2 = np.logical_and( hybrid_wh_points_count < min_points_required, hybrid_wh_output < bc_level_cons_thres * 1000)
    feeble_wh_cons_present_flag2 = np.logical_and( disagg_wh_output > disagg_cons_thres, feeble_wh_cons_present_flag2)

    for i in range(len(unique_bc)):

        # checking presence of feeble SWH consumption in cases where WH is added from hybrid module rather than disagg

        target_days = bc_list == unique_bc[i]

        if feeble_wh_cons_present_flag1[i] and item_input_object.get('item_input_params').get('swh_hld'):
            final_tou_consumption[app_idx][target_days] = 0
            logger.info("blocked wh due to feeble consumption | %s", unique_bc[i])

        # checking presence of feeble WH consumption in cases where WH is added from true disagg module

        if feeble_wh_cons_present_flag2[i]:
            final_tou_consumption[app_idx][target_days] = 0
            logger.info("blocked wh due to feeble consumption | %s", unique_bc[i])

        min_days_required = max(min_cap_on_minimum_wh_days_required_in_month,
                                min(max_cap_on_minimum_wh_days_required_in_month,
                                    minimum_wh_days_required_in_month * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)))

        # checking presence of feeble WH consumption in cases where WH is added for very few days

        feeble_cons_bool = (np.sum(final_tou_consumption[app_idx][target_days], axis=1) > 0).sum() < min_days_required

        if feeble_cons_bool:
            final_tou_consumption[app_idx][target_days] = 0
            logger.info("blocked wh due to less days | %s", unique_bc[i])

    return final_tou_consumption



def block_feeble_cons_in_pp( pp_idx,  final_tou_consumption):

    """
    This function performs postprocessing on PP output to block feeble consumption(less days schedule)

    Parameters:
        pp_idx                    (int)           : mapping for pp output
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    min_pp_days_required = 7
    seq_label = 0
    seq_start = 1
    seq_end = 2
    seq_len = 3

    if final_tou_consumption[pp_idx].sum() > 0:

        timed_app_days = final_tou_consumption[pp_idx].sum(axis=1) > 0

        timed_app_days_seq = find_seq(timed_app_days, np.zeros_like(timed_app_days), np.zeros_like(timed_app_days)).astype(int)

        for i in range(len(timed_app_days_seq)):
            if timed_app_days_seq[i, seq_label] == 0 and timed_app_days_seq[i, seq_len] < 3:
                timed_app_days[timed_app_days_seq[i, seq_start]:timed_app_days_seq[i, seq_end] + 1] = 1

        timed_app_days_seq = find_seq(timed_app_days, np.zeros_like(timed_app_days), np.zeros_like(timed_app_days)).astype(int)

        for i in range(len(timed_app_days_seq)):
            if timed_app_days_seq[i, seq_label] == 1 and timed_app_days_seq[i, seq_len] < min_pp_days_required:
                timed_app_days[timed_app_days_seq[i, seq_start]:timed_app_days_seq[i, seq_end] + 1] = 0

        final_tou_consumption[pp_idx][timed_app_days == 0] = 0

    return final_tou_consumption


def block_feeble_cons_thin_pulse(wh_copy, item_input_object, final_tou_consumption, app_list, bc_list, logger):

    """
    This function performs postprocessing on itemized wh output to block feeble consumption

    Parameters:
        wh_copy                   (np.darray)     : wh output
        item_input_object         (dict)          : Dict containing all hybrid inputs
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        app_list                  (list)          : list of appliances
        bc_list                   (list)          : list of bc start timestamp
        logger                    (logger)        : logger

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    config = get_hybrid_v2_generic_config()

    min_days_required = config.get('min_days_required')

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')
    config = get_inf_config(samples_per_hour).get("wh")

    min_points_required = config.get('wh_min_box_req')/2
    bc_level_cons_thres = config.get('wh_feeble_cons_thres')

    if item_input_object["item_input_params"]["timed_wh_user"] > 0:
        min_points_required = config.get('twh_min_box_req')
        bc_level_cons_thres = config.get('twh_feeble_cons_thres')

    app = 'wh'

    app_idx = np.where(app_list == app)[0][0]

    if np.sum(final_tou_consumption[app_idx]) == 0:
        return final_tou_consumption

    run_feeble_cons_blocking_flag = ((np.sum(final_tou_consumption[app_idx]) == 0) or (item_input_object.get('item_input_params').get('swh_hld') > 0))

    unique_bc, reverse_idx, bc_days_count = np.unique(bc_list, return_inverse=True, return_counts=1)
    disagg_wh_output = np.round(np.bincount(reverse_idx, weights=wh_copy.sum(axis=1)), 2)
    hybrid_wh_output = np.round(np.bincount(reverse_idx, weights=final_tou_consumption[app_idx].sum(axis=1)), 2)
    hybrid_wh_points_count = np.round(np.bincount(reverse_idx, weights=(final_tou_consumption[app_idx]>0).sum(axis=1)), 2)

    check_feeble_cons_flag = np.logical_or(bc_days_count < min_days_required, disagg_wh_output > 0)

    feeble_wh_cons_present = np.logical_and(hybrid_wh_points_count < min_points_required, hybrid_wh_output < bc_level_cons_thres * 1000)

    # checking presence of feeble consumption for each billing cycles

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if run_feeble_cons_blocking_flag or check_feeble_cons_flag[i]:
            continue

        if feeble_wh_cons_present[i]:
            final_tou_consumption[app_idx][target_days] = 0
            logger.info("blocked wh due to feeble consumption | %s", unique_bc[i])

    return final_tou_consumption


def block_stat_app_feeble_cons(item_input_object, final_tou_consumption, app_list, bc_list, vacation, logger):

    """
    This function performs postprocessing on itemized wh output to block feeble consumption

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        app_list                  (list)          : list of appliances
        bc_list                   (list)          : list of bc start timestamp
        vacation                  (np.ndarray)    : vacation list
        logger                    (logger)        : logger

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    unique_bc, bc_size = np.unique(bc_list, return_counts=True)

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    target_app_list = ['cook', 'ent', 'ld']

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    config = get_hybrid_v2_generic_config(samples_per_hour)

    min_points_required = config.get('min_points_required')
    cons_thres = config.get('cons_thres')
    min_days_required = config.get('min_days_required')
    min_li_points_required = config.get('min_li_points_required')
    cons_thres_for_li = config.get('cons_thres_for_li')

    # block low consumption cook/laundry/ent

    for app in target_app_list:

        ld_idx = hybrid_config.get("app_seq").index(app)
        min_cons = hybrid_config.get("min_cons")[ld_idx]
        cons_thres = min(cons_thres, min_cons*Cgbdisagg.WH_IN_1_KWH*0.5)

        logger.info("Blocking low consumption billing for appliance | %s", app)

        app_idx = np.where(app_list == app)[0][0]

        if np.sum(final_tou_consumption[app_idx]) == 0:
            continue

        for i in range(len(unique_bc)):

            target_days = bc_list == unique_bc[i]

            if np.sum(np.sum(target_days)) < min_days_required or np.sum(vacation[target_days] != 0) == 0:
                continue

            # scaling the consumption values based on vacation days in the month
            # for example if a 30 days billing cycle has 20 days of vacation, the output is scaled to 3 times
            # and then compared with min consumption threshold

            low_cons_bool = np.sum(final_tou_consumption[app_idx][target_days])
            low_cons_bool = (low_cons_bool / np.sum(vacation[target_days] != 0) * Cgbdisagg.DAYS_IN_MONTH) < cons_thres

            feeble_cons_bool = np.sum(final_tou_consumption[app_idx][target_days] > 0) < min_points_required

            if (low_cons_bool + feeble_cons_bool):
                final_tou_consumption[app_idx][target_days] = 0

                logger.info("blocked stat app in billing cycle | %s", unique_bc[i])

    # block low consumption lighting

    app = 'li'

    app_idx = np.where(app_list == app)[0][0]

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if np.sum(np.sum(target_days)) < min_days_required or np.sum(vacation[target_days] != 0) == 0:
            continue

        # scaling the consumption values based on vacation days in the month
        # for example if a 30 days billing cycle has 20 days of vacation, the output is scaled to 3 times
        # and then compared with min consumption threshold

        low_cons_bool = np.sum(final_tou_consumption[app_idx][target_days])
        low_cons_bool = (low_cons_bool / np.sum(vacation[target_days] != 0) * Cgbdisagg.DAYS_IN_MONTH) < cons_thres_for_li

        feeble_cons_bool = np.sum(final_tou_consumption[app_idx][target_days] > 0) < min_li_points_required

        if low_cons_bool + feeble_cons_bool:
            final_tou_consumption[app_idx][target_days] = 0

            logger.info("blocked stat app in billing cycle | %s", unique_bc[i])

    return final_tou_consumption

