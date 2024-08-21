
"""
Author - Nisha Agarwal
Date - 17th Feb 2021
Detect timed appliance in residual data
"""

# Import python packages

import copy
import logging
import numpy as np

# import functions from within the project

from python3.config.pilot_constants import PilotConstants

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import get_index_array
from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val

from python3.itemization.aer.functions.hsm_utils import check_validity_of_hsm

from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_appliance_allotment_utils import allot_timed_app
from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_app_detection_utils import fetch_required_inputs
from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_app_detection_utils import post_process_for_estimation
from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_app_detection_utils import eliminate_artifacts_fp_cases
from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_app_detection_utils import eliminate_nonconsistent_app
from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_app_detection_utils import get_valid_schedules
from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_app_detection_utils import extend_sigature_for_mid_duration_intervals
from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_app_detection_utils import extend_signature_for_low_duration_intervals

from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_app_detection_utils import check_consistency_to_extend_signature
from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_app_detection_utils import update_estimation_after_extension

from python3.itemization.aer.raw_energy_itemization.residual_analysis.extend_timed_appliance_cons import extend_pp_cons
from python3.itemization.aer.raw_energy_itemization.residual_analysis.extend_timed_appliance_cons import extend_cons_for_detected_twh_sig
from python3.itemization.aer.raw_energy_itemization.residual_analysis.extend_timed_appliance_cons import extend_twh_cons


def detect_timed_appliance(input_data, item_input_object, item_output_object, logger_pass):

    """
    Master function for detecting timed appliance in residual data

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        item_output_object        (dict)          : Dict containing all outputs
        logger_pass                 (dict)          : logger dictionary

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('detect_timed_signature')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetch required data

    season_label, samples_per_hour, twh_cons_disagg, pp_cons_disagg, positive_residual = \
        fetch_required_inputs(item_input_object, item_output_object)

    # check whether to calculate timed signature

    calculate_timed_sig = determine_timed_sig_calculation_bool(pp_cons_disagg, twh_cons_disagg, item_input_object, logger)

    logger.info("Calculate timed signature flag value | %s", bool(calculate_timed_sig))

    if (not calculate_timed_sig):
        logger.info("Timed signature detection not required | ")
        return positive_residual - item_output_object.get("negative_residual"), np.zeros_like(pp_cons_disagg),\
               np.zeros_like(pp_cons_disagg), np.zeros_like(pp_cons_disagg), np.zeros_like(pp_cons_disagg)

    valid_tou, box_seq, day_start, day_end, box_cons = get_schedule(input_data, item_output_object)

    # remove timed appliance, picked up due to false artifacts in the inputs data

    valid_tou, day_start, day_end = \
        eliminate_artifacts_fp_cases(input_data, day_start, day_end, samples_per_hour, valid_tou)

    pilot = item_input_object.get("config").get("pilot_id")

    if pilot in PilotConstants.TIMED_WH_JAPAN_PILOTS:
        var = valid_tou.sum(axis=0)[get_index_array(22*samples_per_hour, 2*samples_per_hour, samples_per_hour*24)].sum() / valid_tou.sum()
        if var > 0.85:
            valid_tou[:, :] = 0

    logger.info("Calculated schedule for timed appliance | %s", bool(calculate_timed_sig))

    # Extend using pp disagg intermediate results

    valid_tou, box_seq, day_start, day_end = \
        get_disagg_schedules(item_input_object, valid_tou, box_seq, day_start, day_end, input_data)

    logger.info("Updated the schedule using disagg intermediate results | %s", bool(calculate_timed_sig))

    # Post processing to extend timed appliance TOU,, based on consistency in input data

    updated_valid_tou, timed_estimation = \
        fill_days(input_data, copy.deepcopy(valid_tou), day_start.astype(int), day_end.astype(int))

    updated_valid_tou, non_pp = eliminate_nonconsistent_app(updated_valid_tou, samples_per_hour)

    logger.info("Post processing done to extend timed appliance TOU | %s", bool(calculate_timed_sig))

    timed_estimation[np.logical_not(updated_valid_tou)] = 0

    original_timed_estimation = copy.deepcopy(timed_estimation)

    timed_estimation_wh = copy.deepcopy(timed_estimation)

    # Post processing of timed appliance estimation

    if np.sum(timed_estimation):
        timed_estimation, original_timed_estimation, timed_estimation_wh = \
            post_process_for_estimation(timed_estimation, samples_per_hour, updated_valid_tou, input_data)

    max_cons = 0

    if np.any(pp_cons_disagg > 0):
        max_cons = np.median(pp_cons_disagg[pp_cons_disagg > 0])

    max_pp_cap = 3500

    # If high cons timed signature detected, timed signature is not assigned to PP
    high_cons_timed_sig_detected = \
        np.sum(original_timed_estimation > 0) and \
        (np.percentile(original_timed_estimation[original_timed_estimation > 0], 80) >
         max(max_pp_cap / samples_per_hour, max_cons))

    if high_cons_timed_sig_detected:
        non_pp = 1

    timed_estimation, timed_estimation_wh = fill_low_duration_gaps_in_timed_sig(timed_estimation, timed_estimation_wh, samples_per_hour)

    logger.info("Timed appliance consumption estimated at timestamp level | %s", bool(calculate_timed_sig))

    # Allots timed appliance to either PP/TWH category

    timed_estimation_copy = copy.deepcopy(timed_estimation)

    inputs_required_for_timed_sig_allotment = [timed_estimation, timed_estimation_wh, season_label, pp_cons_disagg,twh_cons_disagg]

    twh_cons, pp_cons, heating_cons = \
        allot_timed_app(item_input_object, inputs_required_for_timed_sig_allotment, valid_tou, non_pp, logger)

    # extends twh consumption to avoid wh underestimation in wh tou region

    twh_cons = extend_cons_for_detected_twh_sig(item_input_object, twh_cons, twh_cons_disagg, input_data, samples_per_hour)

    output_data = item_output_object.get("hybrid_input_data").get("output_data")
    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")

    ev_cons = output_data[np.where(np.array(appliance_list) == 'ev')[0][0] + 1, :, :]

    updated_twh_cons = extend_twh_cons(item_input_object, twh_cons_disagg, twh_cons_disagg, pp_cons_disagg, input_data, samples_per_hour)


    updated_pp_cons = extend_pp_cons(item_input_object, item_output_object, pp_cons_disagg, pp_cons_disagg,
                                     twh_cons_disagg + ev_cons, input_data, samples_per_hour)


    check_feeble_cons = 1

    if (np.sum(timed_estimation) == 0) and np.sum(updated_twh_cons - twh_cons_disagg) > 0:
        timed_estimation = updated_twh_cons - twh_cons_disagg
        twh_cons = twh_cons + timed_estimation
        check_feeble_cons = 0

    if np.sum(updated_twh_cons - twh_cons_disagg) > 0:
        timed_estimation = updated_twh_cons - twh_cons_disagg
        twh_cons = twh_cons + timed_estimation
        check_feeble_cons = 0

    if np.sum(updated_pp_cons - pp_cons_disagg) > 0:
        timed_estimation = updated_pp_cons - pp_cons_disagg
        pp_cons = pp_cons + timed_estimation
        check_feeble_cons = 0

    timed_app_days = np.sum(np.maximum(twh_cons, np.maximum(heating_cons, pp_cons)), axis=1) > 0

    timed_app_days_seq = find_seq(timed_app_days, np.zeros_like(timed_app_days), np.zeros_like(timed_app_days)).astype(int)

    feeble_type_cons_present =\
        check_feeble_cons and (np.sum(timed_app_days_seq[timed_app_days_seq[:, 0] == 0, 3] < 5) / np.sum(np.sum(timed_estimation, axis=1) > 0)) > 0.1

    if feeble_type_cons_present:
        pp_cons[:, :] = 0
        twh_cons[:, :] = 0
        heating_cons[:, :] = 0

    if np.sum(twh_cons):
        twh_cons = np.minimum(twh_cons_disagg+twh_cons, input_data)
        twh_cons[twh_cons_disagg > 0] = 0
        logger.info("Timed signature alloted to timed water heater | ")
    elif np.sum(pp_cons):
        pp_cons = np.minimum(pp_cons_disagg+pp_cons, input_data)
        pp_cons[pp_cons_disagg > 0] = 0
        logger.info("Timed signature allotted to PP | ")
    else:
        logger.info("Timed signature not found | ")

    logger.info("Timed appliance signature alloted to one of the appliances | %s", bool(calculate_timed_sig))

    final_residual = positive_residual

    if np.sum(twh_cons + pp_cons + heating_cons):
        final_residual = final_residual - timed_estimation

    return final_residual - item_output_object.get("negative_residual"), np.fmax(0, twh_cons), np.fmax(0, pp_cons),\
           np.fmax(0, heating_cons), timed_estimation_copy


def fill_low_duration_gaps_in_timed_sig(timed_estimation, timed_estimation_wh, samples_per_hour):

    """
    removing timed signature segments that are of less days

    Parameters:
        timed_estimation          (np.ndarray)    : timed estimation
        timed_estimation_wh       (np.ndarray)    : timed estimation for TWH
        samples_per_hour          (int)           : samples in an hour

    Returns:
        timed_estimation          (np.ndarray)    : timed estimation
        timed_estimation_wh       (np.ndarray)    : timed estimation for TWH

    """

    # removing timed signature segments that are less than 1.5 hours

    if np.sum(timed_estimation) > 0:

        posible_region = (timed_estimation > 0).flatten()

        posible_region = fill_arr_based_seq_val(posible_region, posible_region, max(samples_per_hour*1.5, 3), 1, 0, overnight_tag=0)

        posible_region = np.reshape(posible_region, timed_estimation.shape)

        timed_estimation[posible_region == 0] = 0

    if np.sum(timed_estimation_wh) > 0:

        posible_region = (timed_estimation_wh > 0).flatten()

        posible_region = fill_arr_based_seq_val(posible_region, posible_region, max(samples_per_hour*1.5, 3), 1, 0, overnight_tag=0)

        posible_region = np.reshape(posible_region, timed_estimation.shape)

        timed_estimation_wh[posible_region == 0] = 0

    # removing timed signature segments that are of less days

    if np.sum(timed_estimation_wh) > 0:

        posible_region = (timed_estimation_wh.sum(axis=1) > 0)

        posible_region = fill_arr_based_seq_val(posible_region, posible_region, 6, 1, 0, overnight_tag=0)

        timed_estimation_wh[posible_region == 0] = 0

    if np.sum(timed_estimation) > 0:

        posible_region = (timed_estimation.sum(axis=1) > 0)

        posible_region = fill_arr_based_seq_val(posible_region, posible_region, 6, 1, 0, overnight_tag=0)

        timed_estimation[posible_region == 0] = 0

    return timed_estimation, timed_estimation_wh


def determine_timed_sig_calculation_bool(pp_cons, twh_cons, item_input_object, logger):

    """
    check whether timed signature calculation is required

    Parameters:
        pp_cons                   (np.ndarray)    : pp disagg output
        item_input_object         (dict)          : Dict containing all inputs
        logger                    (dict)          : logger dictionary

    Returns:
        calculate_timed           (bool)          : True if timed signature calculation is required
    """

    # determine whether timed detection is needed
    # It can be true under following conditions
    # 1 - pp app prof is yes
    # 2 - pp hsm hld is 1
    # 3 - disagg pp is non zero
    # 4 - ev app prof is present
    # 5 - ev disagg is non-zero
    # 6 - twh disagg is non-zero
    # 7 - user blong to twh pilot
    # 8 - twh info present in wh hsm
    # 9 - wh app profile is present

    ev_app_prof = item_input_object.get("app_profile").get('ev')

    if ev_app_prof is not None:
        ev_app_prof = ev_app_prof.get("number", 0)
    else:
        ev_app_prof = 0

    # fetch app profile
    pp_app_prof = item_input_object.get("app_profile").get('pp')

    if pp_app_prof is not None:
        pp_app_prof = pp_app_prof.get("number", 0)
    else:
        pp_app_prof = 0

    logger.info('PP app profile | %s ', pp_app_prof)

    valid_pp_hsm = item_input_object.get("item_input_params").get('valid_pp_hsm')

    pp_hsm_hld = 1

    valid_hsm_flag = check_validity_of_hsm(valid_pp_hsm, item_input_object.get("item_input_params").get('pp_hsm'),
                                           'item_hld')

    # fetch hsm info

    if valid_hsm_flag:
        pp_hsm = item_input_object.get("item_input_params").get('pp_hsm').get('item_hld')

        if pp_hsm is None:
            pp_hsm_hld = 1
        elif isinstance(pp_hsm, list):
            pp_hsm_hld = pp_hsm[0]
        else:
            pp_hsm_hld = pp_hsm

        logger.info('PP HSM HLD | %s ', pp_hsm_hld)

    calculate_timed = np.sum(pp_cons > 0)

    calculate_timed = calculate_timed or (pp_app_prof)

    if (not pp_hsm_hld) and not np.sum(pp_cons):
        calculate_timed = 0

    calculate_timed = calculate_timed and len(pp_cons) > 15
    calculate_timed = calculate_timed or ev_app_prof

    twh_add_flag = get_wh_addition_bool(twh_cons, item_input_object)

    calculate_timed = calculate_timed or twh_add_flag

    return calculate_timed


def get_wh_addition_bool(twh_cons, item_input_object):

    """
    check whether timed signature calculation is required

    Parameters:
        twh_cons                   (np.ndarray)    : wh disagg output
        item_input_object          (dict)          : Dict containing all inputs
    Returns:
        app_profile                (bool)          : True if timed signature calculation is required
    """

    wh_app_prof = item_input_object.get("app_profile").get('wh')

    if wh_app_prof is not None:
        wh_app_prof = wh_app_prof.get("number", 0)
        wh_type = item_input_object.get("app_profile").get('wh').get("attributes", '')
        if wh_type is not None and (("storagetank" in wh_type) or ("heatpump" in wh_type)):
            wh_app_prof = 0
    else:
        wh_app_prof = 0

    valid_wh_hsm = item_input_object.get("item_input_params").get('valid_wh_hsm')
    wh_hsm_hld = 1
    wh_type = 1

    valid_hsm_flag = check_validity_of_hsm(valid_wh_hsm, item_input_object.get("item_input_params").get('wh_hsm'), 'item_hld')

    # add twh if hsm wh type is timed

    if valid_hsm_flag:
        wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_hld')

        if wh_hsm is None:
            wh_hsm_hld = 0
        elif isinstance(wh_hsm, list):
            wh_hsm_hld = wh_hsm[0]
        else:
            wh_hsm_hld = wh_hsm

        wh_type = item_input_object.get("item_input_params").get('wh_hsm').get('item_type')

        if wh_type is None:
            wh_type = 0
        elif isinstance(wh_type, list):
            wh_type = wh_type[0]

    pilot = item_input_object.get("config").get("pilot_id")

    # to add wh, wh shud be present in timed wh pilot list

    twh_pilots = PilotConstants.TIMED_WH_PILOTS

    twh_addition_flag = (wh_hsm_hld and wh_type == 1 and (wh_app_prof or (pilot in twh_pilots))) or np.sum(twh_cons > 0)

    return twh_addition_flag


def get_disagg_schedules(item_input_object, valid_tou_disagg, box_seq, day_start, day_end, input_data):

    """
    prepare intermediate pp detection results in disagg module, which can further be used for timed signature detection

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
        input_data                (np.ndarray)    : input data

    Returns:
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
    """

    # fetch edges data from disagg special outputs

    edges_data = item_input_object.get("disagg_special_outputs").get("pp_steps")[2] + item_input_object.get("disagg_special_outputs").get("pp_steps")[3]

    edges_data = edges_data[:len(input_data)]

    if np.sum(edges_data > 0) == 0:
        return valid_tou_disagg, box_seq, day_start, day_end

    disagg_edges = (np.sum(edges_data > 0, axis=1) > 0).astype(int)
    disagg_edges = find_seq(disagg_edges, np.zeros_like(disagg_edges), np.zeros_like(disagg_edges))

    for i in range(len(disagg_edges)):

        # For each disagg edge, check which index is the start ad end time in the nearby window,

        if disagg_edges[i, 0]:
            update_timed_sig_tou_for_disagg_edges(valid_tou_disagg, box_seq, day_start, day_end, input_data, i, disagg_edges, edges_data)

    return valid_tou_disagg, box_seq, day_start, day_end


def update_timed_sig_tou_for_disagg_edges(valid_tou_disagg, box_seq, day_start, day_end, input_data, index, disagg_edges, edges_data):

    """
    prepare intermediate pp detection results in disagg module, which can further be used for timed signature detection

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
        input_data                (np.ndarray)    : input data

    Returns:
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
    """

    derivative = input_data - np.roll(input_data, 1, axis=1)

    samples_per_hour = int(input_data.shape[1]/24)

    idx_strt = disagg_edges[index, 1]
    idx_end = disagg_edges[index, 2]

    idx_arr = get_index_array(idx_strt, idx_end, len(valid_tou_disagg))

    start = np.where(np.sum(edges_data[idx_arr], axis=0) > 0)[0][0]
    end = np.where(np.sum(edges_data[idx_arr], axis=0) < 0)[0][0]

    strt_idx_list = get_index_array(start - samples_per_hour, start + samples_per_hour, samples_per_hour * 24)
    end_idx_list = get_index_array(end - samples_per_hour, end + samples_per_hour, samples_per_hour * 24)

    start_flag = 0
    end_flag = 0

    for j in strt_idx_list:

        edge_derivative = derivative[idx_arr]
        pos_derivative = edge_derivative[:, j]

        tmp_flag = np.abs(
            np.percentile(pos_derivative, 75) - np.percentile(pos_derivative, 25)) < 600 / samples_per_hour

        tmp_flag = tmp_flag and np.percentile(pos_derivative, 50) > 300 / samples_per_hour

        start_flag = start_flag or tmp_flag

        if tmp_flag:
            start = j

    for j in end_idx_list:
        edge_derivative = derivative[idx_arr]
        pos_derivative = edge_derivative[:, j]

        tmp_flag = np.abs(
            np.percentile(pos_derivative, 75) - np.percentile(pos_derivative, 25)) < 600 / samples_per_hour

        tmp_flag = tmp_flag and np.percentile(pos_derivative, 50) < -300 / samples_per_hour

        end_flag = end_flag or tmp_flag

        if tmp_flag:
            end = j

    # Once start and end is determine check if the consumption is consistent throughout

    # IF the disagg edge is qualified, use this edge in further time signature detection

    day_start, day_end, valid_tou_disagg = \
        fill_days_based_on_disagg_results(idx_strt, input_data, day_start, day_end, valid_tou_disagg,
                                          start_flag, end_flag, [start, end], edges_data, idx_arr)

    return valid_tou_disagg, box_seq, day_start, day_end


def fill_days_based_on_disagg_results(idx_strt, input_data, day_start, day_end, valid_tou_disagg, start_flag,
                                      end_flag, schedule, edges_data, idx_arr):

    """
    prepare intermediate pp detection results in disagg module, which can further be used for timed signature detection

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
        input_data                (np.ndarray)    : input data

    Returns:
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
    """

    start = schedule[0]
    end = schedule[1]

    final_flag = start_flag and end_flag

    samples_per_hour = int(valid_tou_disagg.shape[1]/24)

    data = input_data[idx_arr]
    data = data[:, get_index_array(start, end, len(edges_data[0]))]

    low_data_perc = (np.sum(data < 400 / samples_per_hour) / np.size(data)) * 100

    final_flag = final_flag and (np.abs(np.percentile(data, 75) - np.percentile(data, 25)) < 1000 / samples_per_hour)
    final_flag = final_flag and low_data_perc < 10

    if final_flag and len(get_index_array(start, end, len(valid_tou_disagg[0]))) > samples_per_hour / 2:
        day_start[idx_arr] = start
        day_end[idx_arr] = end

        days_bool = np.zeros_like(edges_data)
        tou_bool = np.zeros_like(edges_data)

        days_bool[idx_arr] = 1
        tou_bool[:, get_index_array(day_start[idx_strt], day_end[idx_strt], len(edges_data[0]))] = 1

        valid_tou_disagg[np.logical_and(tou_bool, days_bool) > 0] = 1

    return day_start, day_end, valid_tou_disagg


def get_schedule(input_data, item_output_object):

    """
    Determine each timed schedule, based on whether there is presence of timed activity

    Parameters:
        input_data                (dict)          : input data
        item_output_object        (dict)          : Dict containing all outputs

    Returns:
        estimation                (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
    """

    box_cons = item_output_object.get("box_dict").get("box_cons")
    box_seq = item_output_object.get("box_dict").get("box_seq")

    samples_per_hour = int(box_cons.shape[1] / 24)

    valid_seq = np.zeros(len(box_seq))

    length = len(box_cons)

    day_start = np.zeros(length)
    day_end = np.zeros(length)

    valid_tou = np.zeros(box_cons.size)

    box_seq = box_seq.astype(int)

    box_cons_tmp = np.maximum(input_data.reshape(box_cons.size), box_cons.reshape(box_cons.size))

    end_day = length - 1
    interval = 20

    buffer = 1

    if len(input_data) < 160:
        interval = 30

    if len(input_data) < 32:
        interval = length

    # for each window of day, it determines whether there are sufficient activity with similar start and end time

    for k in range(0, end_day, interval):

        features = prepare_features(k, box_seq, interval, samples_per_hour)

        valid_tou_tmp = np.zeros(box_cons.size)

        thres = 6 * (samples_per_hour > 1) + 12 * (samples_per_hour == 1)

        # If number of activity are greater than a given threshold, determine whether they can be considered for
        # timed detection, based consistency in their consumption

        if np.max(features) >= thres:
            valid_tou = get_valid_schedules(features, buffer, box_seq, valid_tou, valid_tou_tmp, box_cons, box_cons_tmp, valid_seq)

    valid_tou = np.fmin(1, valid_tou)

    output = valid_tou.reshape(box_cons.shape)

    estimation = np.maximum(input_data, box_cons)
    estimation[output == 0] = 0

    val = 0.85

    if len(estimation) <= 120:
        val = 0.9

    # If multiple low duration schedules are found, timed signature is made 0

    if np.sum(estimation):
        seq = find_seq((np.sum(estimation, axis=1) > 0).astype(int), np.zeros(len(estimation)), np.zeros(len(estimation)))
        if np.sum(seq[seq[:, 0] == 1, 3] < 3)/np.sum(seq[:, 0] == 1) > val:
            estimation[:, :] = 0
            output[:, :] = 0

    # determine start and end tou of each schedule

    for i in range(length):

        if np.sum(output[i]):

            seq = find_seq(output[i], np.zeros(length), np.zeros(length))

            day_start[i] = seq[seq[:, 0] == 1, 1][0]
            day_end[i] = seq[seq[:, 0] == 1, 2][0]

    return estimation, box_seq, day_start, day_end, box_cons


def prepare_features(index, box_seq, interval, samples_per_hour):

    """
    Prepare timed features for a given activity in a window

    Parameters:
        index                (np.ndarray)    : tou of timed signature
        box_seq              (np.ndarray)    : list of schedules
        interval             (int)           : length of time window
        samples_per_hour     (int)           : samples in an hour

    Returns:
        features             (np.ndarray)    : prepared features
    """

    features = np.zeros((samples_per_hour * 24, samples_per_hour * 24))

    range_days = np.arange(index, index + interval)

    for i in range(len(box_seq)):
        seq_start = int(box_seq[i, 1] % (samples_per_hour*24))
        seq_len = int(box_seq[i, 3])

        if box_seq[i, 0] and int(box_seq[i, 1] / (samples_per_hour*24)) in range_days:

            buffer = min(1, int(samples_per_hour / 2))

            if samples_per_hour > 1:
                features[seq_start - buffer:seq_start + buffer, seq_len - buffer:seq_len + buffer] = \
                    features[seq_start - buffer:seq_start + buffer, seq_len - buffer:seq_len + buffer] + 1
            else:
                features[seq_start, seq_len] = features[seq_start, seq_len] + 1

    features[:, :int(samples_per_hour * 1.5) + 1] = 0

    return features


def extend_timed_signature(input_data, time_sig_days_seq, estimation, day_start, day_end, valid_tou):

    """
    # Extended timed signature edges to the regions where edges are not clear,
    # but there are chances of timed appliance being present
    # this helps in providing pp output at the regions with heavy hvac

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
        input_data                (np.ndarray)    : input data

    Returns:
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
    """

    time_sig_days_seq = time_sig_days_seq.astype(int)

    for i in range(len(time_sig_days_seq)):

        # If the non signature days are less than 10, consider the band for signature detection

        if (not time_sig_days_seq[i, 0]) and (time_sig_days_seq[i, 3] < 8):
            start = day_start[time_sig_days_seq[(i - 1)%len(time_sig_days_seq), 1]]
            end = day_end[time_sig_days_seq[(i - 1)%len(time_sig_days_seq), 1]]

            if start == end:
                continue

            idx_arr = get_index_array(start, end, len(input_data[0]))

            valid_ts = np.zeros_like(valid_tou)
            valid_ts[time_sig_days_seq[i, 1]: time_sig_days_seq[i, 2] + 1] = 1

            valid_tou[:, idx_arr] = valid_tou[:, idx_arr] + valid_ts[:, idx_arr]

        elif (not time_sig_days_seq[i, 0]):

            if (time_sig_days_seq[(i - 1)%len(time_sig_days_seq), 2] - time_sig_days_seq[(i - 1)%len(time_sig_days_seq), 1]) > 0:
                tmp_start = int(np.mean(day_start[time_sig_days_seq[(i - 1)%len(time_sig_days_seq), 1]:time_sig_days_seq[(i - 1)%len(time_sig_days_seq), 2]]))
                tmp_end = int(np.mean(day_end[time_sig_days_seq[(i - 1)%len(time_sig_days_seq), 1]:time_sig_days_seq[(i - 1)%len(time_sig_days_seq), 2]]))

                median_amp = estimation[time_sig_days_seq[(i - 1)%len(time_sig_days_seq), 1]:time_sig_days_seq[(i - 1)%len(time_sig_days_seq), 2]]
                median_amp = median_amp[:, get_index_array(tmp_start, tmp_end, len(input_data[0]))]
            else:
                continue

            estimation, day_start, day_end, valid_tou = \
                extend_timed_signature_based_on_neighbouring_bands(median_amp, input_data, time_sig_days_seq, i, tmp_start,
                                                                   tmp_end, estimation, day_start, day_end, valid_tou)

    return estimation, day_start, day_end, valid_tou


def fill_days(input_data, valid_tou, day_start, day_end):

    """
    # Extended timed signature edges to the regions where edges are not clear,
    # but there are chances of timed appliance being present
    # this helps in providing pp output at the regions with heavy hvac

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
        input_data                (np.ndarray)    : input data

    Returns:
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
    """

    valid_tou = (valid_tou > 0).astype(int)

    valid_days = (np.sum(valid_tou, axis=1) > 0).astype(int)

    input_data = input_data.astype(int)

    # determine the seq of days where edges were detected

    time_sig_days_seq = find_seq(valid_days, np.zeros(len(valid_days)), np.zeros(len(valid_days)), overnight=0)

    time_sig_days_seq = time_sig_days_seq.astype(int)

    estimation = np.zeros_like(input_data)

    for i in range(1, len(time_sig_days_seq)-1):

        # if the timed sig seq is less than 15 days, it is not considered for timed detected

        if (time_sig_days_seq[i, 0]) and time_sig_days_seq[i+1, 3] > 30 and time_sig_days_seq[i-1, 3] > 30 and (not time_sig_days_seq[i, 3] > 15) and len(input_data) > 90:
            valid_tou[time_sig_days_seq[i, 1]:time_sig_days_seq[i, 2]+1, :] = 0

        elif (not time_sig_days_seq[i, 0]) and time_sig_days_seq[i, 3] < 10:
            valid_tou, valid_ts, day_start, day_end = extend_signature_for_low_duration_intervals(i, valid_tou, day_start, day_end, time_sig_days_seq, input_data)

        elif not time_sig_days_seq[i, 0] and time_sig_days_seq[i, 3] < 40:
            valid_tou, valid_ts, day_start, day_end = extend_sigature_for_mid_duration_intervals(i, valid_tou, day_start, day_end, time_sig_days_seq, input_data)

    valid_tou = (valid_tou > 0).astype(int)

    valid_days = (np.sum(valid_tou, axis=1) > 0).astype(int)

    time_sig_days_seq = find_seq(valid_days, np.zeros(len(valid_days)), np.zeros(len(valid_days)), overnight=0)

    time_sig_days_seq = time_sig_days_seq.astype(int)

    # Update estimation consumption for new extended edges

    for i in range(0, len(time_sig_days_seq)):
        if (time_sig_days_seq[i, 0]) and (time_sig_days_seq[i, 3] <= 10) and len(input_data) > 90:
            valid_tou[time_sig_days_seq[i, 1]: time_sig_days_seq[i, 2] + 1] = 0

        elif (time_sig_days_seq[i, 0]) :
            estimation[get_index_array(time_sig_days_seq[i, 1], time_sig_days_seq[i, 2]-1, len(input_data))] = \
                input_data[get_index_array(time_sig_days_seq[i, 1], time_sig_days_seq[i, 2]-1, len(input_data))]

            estimation[valid_tou == 0] = 0

    valid_tou = (valid_tou > 0).astype(int)

    valid_days = (np.sum(valid_tou, axis=1) > 0).astype(int)

    input_data = input_data.astype(int)

    samples_per_hour = int(input_data.shape[1] / 24)

    time_sig_days_seq = find_seq(valid_days, np.zeros(len(valid_days)), np.zeros(len(valid_days)), overnight=0)

    time_sig_days_seq = time_sig_days_seq.astype(int)

    estimation, day_start, day_end, valid_tou = extend_timed_signature(input_data, time_sig_days_seq, estimation, day_start, day_end, valid_tou)

    valid_tou = (valid_tou > 0).astype(int)

    valid_days = (np.sum(valid_tou, axis=1) > 0).astype(int)

    time_sig_days_seq = find_seq(valid_days, np.zeros(len(valid_days)), np.zeros(len(valid_days)), overnight=0)

    time_sig_days_seq = time_sig_days_seq.astype(int)

    valid_tou = post_processing_to_fill_short_window_pp_days(valid_tou, day_start, time_sig_days_seq)

    estimation[valid_tou == 0] = 0
    valid_tou[np.sum(valid_tou, axis=1) > 20*samples_per_hour] = 0

    return valid_tou, estimation


def post_processing_to_fill_short_window_pp_days(valid_tou, day_start, time_sig_days_seq):

    """
    Extended timed signature edges to the regions where timed signature is not detected for only short window

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
        input_data                (np.ndarray)    : input data

    Returns:
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
    """

    for i in range(1, len(time_sig_days_seq) - 1):

        if (not time_sig_days_seq[i, 0]) and (time_sig_days_seq[i, 3] < 20):
            idx_arr = get_index_array(day_start[time_sig_days_seq[i - 1, 1]], day_start[time_sig_days_seq[i - 1, 2]], len(valid_tou[0]))

            valid_ts = np.zeros_like(valid_tou)
            valid_ts[time_sig_days_seq[i, 1]: time_sig_days_seq[i, 2] + 1] = 1

            valid_tou[:, idx_arr] = valid_tou[:, idx_arr] + valid_ts[:, idx_arr]

        if (time_sig_days_seq[i, 0]) and (time_sig_days_seq[i, 3] < 10) and len(valid_tou) > 90:
            valid_tou[time_sig_days_seq[i, 1]: time_sig_days_seq[i, 2] + 1] = 0

    return valid_tou


def extend_timed_signature_based_on_neighbouring_bands(median_amp, input_data, time_sig_days_seq, i, tmp_start,
                                                       tmp_end, estimation, day_start, day_end, valid_tou):

    """
    # Extended timed signature edges to the regions where edges are not clear,
    # but there are chances of timed appliance being present
    # this helps in providing pp output at the regions with heavy hvac

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
        input_data                (np.ndarray)    : input data

    Returns:
        valid_tou_disagg          (np.ndarray)    : tou of timed signature
        box_seq                   (np.ndarray)    : list of schedules
        day_start                 (np.ndarray)    : list of start tou
        day_end                   (np.ndarray)    : list of start tou
    """

    window = 15

    samples_per_hour = int(input_data.shape[1]/24)

    day_start_idx = time_sig_days_seq[i, 1]
    day_end_idx = time_sig_days_seq[i, 2]

    seq_idx_arr = get_index_array(tmp_start, tmp_end, len(input_data[0]))

    if len(median_amp):
        median_amp = np.percentile(median_amp, 50)

        idx_arr = get_index_array(tmp_start, tmp_end, len(input_data[0]))

        if time_sig_days_seq[(i - 1) % len(time_sig_days_seq), 0] and \
                time_sig_days_seq[(i - 1) % len(time_sig_days_seq), 3] > 5 and len(seq_idx_arr) > samples_per_hour / 2:

            for j in range(day_start_idx, max(day_start_idx + 1, day_end_idx - min(time_sig_days_seq[i, 3], window)), 5):

                extension_flag = \
                    check_consistency_to_extend_signature(input_data, j, window, idx_arr, median_amp, day_cons_thres=0.8, cons_thres=0.9)

                estimation, day_start, day_end, valid_tou =\
                    update_estimation_after_extension((i - 1) % len(time_sig_days_seq), estimation, [day_start, day_end],
                                                      valid_tou, j, time_sig_days_seq, [tmp_start, tmp_end], input_data, median_amp,
                                                      extension_flag)

    tmp_start = int(
        np.mean(day_start[time_sig_days_seq[(i + 1) % len(time_sig_days_seq), 1]:time_sig_days_seq[(i + 1) % len(
            time_sig_days_seq), 2] + 1]))
    tmp_end = int(
        np.mean(day_end[time_sig_days_seq[(i + 1) % len(time_sig_days_seq), 1]:time_sig_days_seq[(i + 1) % len(
            time_sig_days_seq), 2] + 1]))

    seq_idx_arr = get_index_array(tmp_start, tmp_end, len(input_data[0]))

    median_amp = estimation[time_sig_days_seq[(i + 1) % len(time_sig_days_seq), 1]:time_sig_days_seq[
        (i + 1) % len(time_sig_days_seq), 2]]
    median_amp = median_amp[:, get_index_array(tmp_start, tmp_end, len(input_data[0]))]

    if len(median_amp):
        median_amp = np.percentile(median_amp, 60)

        idx_arr = get_index_array(tmp_start, tmp_end, len(input_data[0]))

        if time_sig_days_seq[(i + 1) % len(time_sig_days_seq), 0] and \
                time_sig_days_seq[(i + 1) % len(time_sig_days_seq), 3] > 5 and len(seq_idx_arr) > samples_per_hour / 2:

            for j in range(day_start_idx, max(day_start_idx + 1, day_end_idx - min(time_sig_days_seq[i, 3], window)), 5):

                extension_flag = \
                    check_consistency_to_extend_signature(input_data, j, window, idx_arr, median_amp, day_cons_thres=0.9, cons_thres=0.6)

                estimation, day_start, day_end, valid_tou = \
                    update_estimation_after_extension((i + 1) % len(time_sig_days_seq), estimation, [day_start, day_end],
                                                      valid_tou, j, time_sig_days_seq, [tmp_start, tmp_end], input_data, median_amp,
                                                      extension_flag)

    return estimation, day_start, day_end, valid_tou
