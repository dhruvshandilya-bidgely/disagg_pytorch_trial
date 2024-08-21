"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update ev consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.hsm_utils import check_validity_of_hsm

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array
from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config

from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import fetch_ev_hsm_params
from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import filter_valid_ev_boxes
from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import get_required_ev_box_params
from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import get_hsm_of_incremental_mode
from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import calculate_ev_box_count_threshold

from python3.itemization.aer.raw_energy_itemization.inference_engine.detect_additional_l1_ev_boxes import ev_l1_postprocessing

from python3.itemization.aer.raw_energy_itemization.inference_engine.detect_additional_l2_ev_boxes import detect_leftover_ev_boxes
from python3.itemization.aer.raw_energy_itemization.inference_engine.detect_additional_l2_ev_boxes import add_ev_boxes_if_ev_detected_from_hybrid
from python3.itemization.aer.raw_energy_itemization.inference_engine.detect_additional_l2_ev_boxes import add_ev_boxes_to_maintain_cons_in_ev_tou
from python3.itemization.aer.raw_energy_itemization.inference_engine.detect_additional_l2_ev_boxes import add_ev_boxes_to_maintain_con

from python3.itemization.aer.raw_energy_itemization.inference_engine.blocking_false_ev_detection_cases import eliminate_seasonal_ev_cases
from python3.itemization.aer.raw_energy_itemization.inference_engine.blocking_false_ev_detection_cases import block_ev_cons_in_bc_with_less_boxes

from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import update_ev_hsm
from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import check_ev_output
from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import fetch_ev_l1_hsm_params
from python3.itemization.aer.raw_energy_itemization.inference_engine.ev_addition_utils import update_ev_hsm_with_l1_charger_params


def get_ev_inference(app_index, item_input_object, item_output_object, logger_pass):

    """
    Update ev consumption ranges using inference rules

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object          (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    ev_config = get_inf_config().get('ev')

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_ev_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    app_profile = item_input_object.get("app_profile").get('ev')
    additional_ev_boxes = item_output_object.get("ev_residual")
    app_conf = item_output_object.get("inference_engine_dict").get("appliance_conf")[app_index, :, :]
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    max_cons = item_output_object.get("inference_engine_dict").get("appliance_max_values")[app_index, :, :]
    min_cons = item_output_object.get("inference_engine_dict").get("appliance_min_values")[app_index, :, :]
    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    seq_label = seq_config.SEQ_LABEL

    if additional_ev_boxes is None:
        additional_ev_boxes = np.zeros_like(disagg_cons)

    max_cons = np.maximum(max_cons, disagg_cons)
    mid_cons = np.maximum(mid_cons, np.multiply(disagg_cons, app_conf))

    if np.all(app_conf == 0):
        app_conf = np.ones(app_conf.shape)

    if app_profile is not None:
        app_profile = app_profile.get("number", 0)
    else:
        app_profile = 0

    logger.info("EV app profile is %d | ", app_profile)

    # No inference calculation is consumption is zero, and app profile says no

    if np.all(disagg_cons == 0) and not app_profile:
        item_output_object = update_ev_hsm(item_input_object, item_output_object, disagg_cons)
        return item_output_object

    ##################   RULE 1 - Identify boxes similar to ev capacity in the disagg residual #########################

    max_cons_tou = max_cons > 0

    if np.any(mid_cons > 0):
        min_cons = np.minimum(min_cons, np.percentile(mid_cons[mid_cons > 0], 99)) + additional_ev_boxes
        max_cons = np.maximum(max_cons, np.percentile(mid_cons[mid_cons > 0], 2)) + additional_ev_boxes
        max_cons[np.logical_not(max_cons_tou)] = 0

    additional_ev_boxes, item_output_object = \
        prepare_additional_ev_cons_before_adding_into_mid_cons(app_index, item_input_object, item_output_object, mid_cons, logger)

    # final prepared ev boxes are added into min/max/mid values of ev category before further adjustment

    min_cons = np.multiply(min_cons, app_conf/np.max(app_conf))
    mid_cons = mid_cons + additional_ev_boxes
    min_cons = min_cons + additional_ev_boxes

    item_output_object["inference_engine_dict"]["appliance_conf"][app_index][additional_ev_boxes > 0] = 1

    ############ RULE 2 - Removing EV boxes where less boxes are present event after residual box addition  ###########

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    # Final sanity checks

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        ev_cons_1d = ((mid_cons)[target_days] > 0).flatten()

        if np.sum(ev_cons_1d) == 0:
            continue

        seq = find_seq(ev_cons_1d, np.zeros_like(ev_cons_1d), np.zeros_like(ev_cons_1d))

        if (np.sum(seq[:, seq_label] > 0) < 2):
            mid_cons[target_days] = 0

    min_cons = np.nan_to_num(min_cons)

    min_cons = np.minimum(min_cons, mid_cons)
    max_cons = np.maximum(max_cons, mid_cons)

    ######### RULE 3 - Extend EV l1 consumption to handle EV L1 estimation inconsistency or underestion cases  #######

    freq_of_ev_l1_usage, l1_tag = fetch_ev_l1_hsm_params(item_input_object, input_data)

    if (np.sum(mid_cons) > 0) or (l1_tag == 1 and freq_of_ev_l1_usage > 0):
        min_cons, mid_cons, max_cons, item_output_object = \
            ev_l1_postprocessing(l1_tag, freq_of_ev_l1_usage, item_input_object, app_index, min_cons, mid_cons,
                                 max_cons, item_output_object, logger)

    ###########  RULE 4 - mark ev hld as 0, if overall very few boxes are leftout in ev category consummption ##########

    box_count = np.reshape(disagg_cons, disagg_cons.size) > 0
    box_count = find_seq(box_count, np.zeros_like(box_count), np.zeros_like(box_count))

    if np.sum(disagg_cons) > 0 and np.sum(box_count[:, 0] > 0) < ev_config.get('user_min_box_count') and len(mid_cons) > 80:
        mid_cons[:, :] = 0
        min_cons[:, :] = 0
        max_cons[:, :] = 0

    max_cons = np.minimum(max_cons, input_data)
    min_cons = np.minimum(min_cons, input_data)
    mid_cons = np.minimum(mid_cons, input_data)

    # Updating the values in the original dictionary

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons)

    item_output_object = \
        update_ev_hsm_with_l1_charger_params(l1_tag, freq_of_ev_l1_usage, additional_ev_boxes, mid_cons,
                                             item_input_object, item_output_object, disagg_cons)

    item_output_object["inference_engine_dict"]["output_data"][app_index, :, :] = disagg_cons

    t_end = datetime.now()

    logger.info("EV inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def prepare_additional_ev_cons_before_adding_into_mid_cons(app_index, item_input_object, item_output_object, mid_cons, logger):

    """
    Update ev consumption picked form residual data

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        mid_cons                    (np.ndarray): EV TS level avg cons
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object          (dict)      : updated Dict containing all hybrid outputs
    """

    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    ev_config = get_inf_config().get('ev')

    additional_ev_boxes = item_output_object.get("ev_residual")
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
    unique_bc, bc_indices, counts = np.unique(bc_list, return_inverse=True, return_counts=True)

    # remove boxes if low duration boxes are present in detected ev signatures

    samples_per_hour = int(disagg_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)

    if np.sum(disagg_cons) == 0:
        additional_ev_boxes = check_ev_output(item_output_object, additional_ev_boxes, samples_per_hour)

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    # remove boxes if very few boxes are found in a particular billing cycle

    additional_ev_boxes = block_ev_cons_in_bc_with_less_boxes(item_input_object, vacation_days, additional_ev_boxes,
                                                              bc_list, unique_bc, ev_config, disagg_cons)

    # remove ev consumption in cases where ev boxes (detected in hybrid v2) aligns with a particular season
    # these checks are not done in cases where ev was already detected in previous disagg runs

    season = item_output_object.get("season")

    run_seasonal_check = 1

    if item_input_object.get("config").get('disagg_mode') in ['incremental', 'mtd'] and \
            item_input_object.get("item_input_params").get('ev_hsm') is not None \
            and item_input_object.get("item_input_params").get('ev_hsm').get('item_type') is not None:
        ev_type = item_input_object.get("item_input_params").get('ev_hsm').get('item_type')

        if isinstance(ev_type, list):
            ev_type = ev_type[0]

        run_seasonal_check = not (ev_type == 2 and len(input_data) > 90)

    logger.info('EV seasonal check flag %s | ', run_seasonal_check)

    vacation_days = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    if run_seasonal_check:
        additional_ev_boxes = eliminate_seasonal_ev_cases(item_input_object, vacation_days, additional_ev_boxes > 0,
                                                          samples_per_hour, copy.deepcopy(season), disagg_cons,
                                                          additional_ev_boxes, logger)

    # removing added ev boxes on vacation days

    additional_ev_boxes[vacation_days] = 0

    logger.info("final residual addition points | %s ", np.sum(additional_ev_boxes > 0))

    if np.sum(additional_ev_boxes) == 0 and np.sum(disagg_cons) == 0:
        item_output_object = update_ev_hsm(item_input_object, item_output_object, disagg_cons)

    # for each additional ev boxes, consumption is added on one data point each side of the box

    extended_ev_boxes = np.zeros_like(additional_ev_boxes).flatten()

    seq = find_seq((additional_ev_boxes > 0).flatten(), np.zeros_like((additional_ev_boxes > 0).flatten()),
                   np.zeros_like((additional_ev_boxes > 0).flatten()), overnight=0).astype(int)

    for i in range(1, len(seq)):
        if seq[i, 0] > 0:
            extended_ev_boxes[max(0, seq[i, seq_start] - 1)] = 1
            extended_ev_boxes[min(len(seq) - 1, seq[i, seq_end] + 1)] = 1

    extended_ev_boxes = extended_ev_boxes.reshape(additional_ev_boxes.shape)

    if np.sum(extended_ev_boxes) > 0 and np.sum(additional_ev_boxes + disagg_cons) > 0:
        amp = np.percentile((additional_ev_boxes + disagg_cons)[(additional_ev_boxes + disagg_cons) > 0], 90)
        extended_ev_boxes[input_data < 0.5 * amp] = 0
        extended_ev_boxes[mid_cons > 0] = 0

        extended_ev_boxes[extended_ev_boxes > 0] = input_data[extended_ev_boxes > 0]

        extended_ev_boxes = np.fmin(amp, extended_ev_boxes)

        additional_ev_boxes = additional_ev_boxes + extended_ev_boxes

    return additional_ev_boxes, item_output_object


def get_ev_residual_signature(excluded_cons_for_ev, item_input_object, item_output_object, ev_disagg, residual, input_data, samples_per_hour, logger, l1_bool=False):

    """
    Calculate EV consumption in the leftover residual data

    Parameters:
        item_input_object           (dict)        : Dict containing all hybrid outputs
        item_output_object          (dict)        : Dict containing all hybrid outputs
        residual                    (np.ndarray)  : disagg residual data
        ev_disagg                   (np.ndarray)  : EV TS level disagg output
        input_data                  (np.ndarray)  : raw data
        samples_per_hour            (int)         : number of samples in an hour
        logger                      (logger)      : Contains the logger and the logging dictionary to be passed on
        l1_bool                          (bool)        : true if ev l1_bool output has to be estimated, false in case of L2 charger type

    Returns:
        valid_idx                   (dict)        : updated Dict containing all hybrid outputs
        residual_cons               (np.ndarray)  : Calculates TS level consumption
    """

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    ev_config = get_inf_config().get('ev')

    timed = copy.deepcopy(item_output_object.get("timed_app_dict").get("timed_output") > 0)

    box_cons = item_output_object.get("box_dict").get("box_cons")
    box_seq = item_output_object.get("box_dict").get("box_seq")

    pilot = item_input_object.get("config").get("pilot_id")

    max_cap, min_cap, min_len, max_len, non_ev_hours = get_required_ev_box_params(item_input_object, l1_bool, samples_per_hour, ev_disagg, pilot)

    # update ev params incase of mtd mode

    max_cap, min_cap, min_len, max_len, amplitude, detect_ev, zero_type = \
        get_ev_box_params_based_on_hsm(item_input_object, samples_per_hour, max_cap, min_cap, min_len, max_len, l1_bool)

    if not detect_ev:
        return np.zeros_like(input_data), np.zeros_like(input_data)

    residual_cons = np.zeros_like(input_data)
    valid_idx = np.zeros_like(input_data)

    if zero_type == 0:
        if np.sum(ev_disagg) > 0 and (not l1_bool) and (np.percentile(ev_disagg[ev_disagg > 0], 90) * samples_per_hour < 2500):
            return np.zeros_like(input_data), np.zeros_like(input_data)

        logger.info("EV Boxes max amp | %d", max_cap[0])
        logger.info("EV Boxes min amp | %d", min_cap[0])

        valid_idx = np.zeros(residual.size)
        valid_seq = np.ones(len(box_seq))

        # Pick boxes that lies in the defined range of length and consumption

        valid_seq[box_seq[:, 3] < min_len[0]] = 0
        valid_seq[box_seq[:, 3] > max_len[0]] = 0
        valid_seq[box_seq[:, 4] < min_cap[0]] = 0
        valid_seq[box_seq[:, 4] > max_cap[0]] = 0

        valid_seq2 = np.ones(len(box_seq))
        valid_seq2[box_seq[:, 3] < min_len[2]] = 0
        valid_seq2[box_seq[:, 3] > max_len[2]] = 0
        valid_seq2[box_seq[:, 4] < min_cap[2]] = 0
        valid_seq2[box_seq[:, 4] > max_cap[2]] = 0

        valid_seq = np.logical_or(valid_seq, valid_seq2)

        box_seq = box_seq.astype(int)

        ev_box_count = 0

        timed = timed.flatten()

        # check number of valid ev boxes

        ev_box_count, valid_idx, box_seq = filter_valid_ev_boxes(ev_box_count, l1_bool, valid_idx, box_seq, timed, valid_seq, samples_per_hour, non_ev_hours)

        # calculates minimum number of EV boxes required

        ev_box_count_thres = calculate_ev_box_count_threshold(item_input_object, valid_idx, ev_disagg, amplitude, l1_bool, input_data, pilot, ev_box_count, logger)

        # if ev box count is less than threshold, no ev boxes will be added

        if ev_box_count < ev_box_count_thres:
            valid_idx = np.zeros_like(valid_idx)

        # add additional ev l2 boxes left out in residual data

        new_valid_seq, new_box_seq, new_boxes, cons = \
            update_ev_output_with_remaining_ev_boxes(item_input_object, residual, ev_disagg, box_cons, valid_idx, samples_per_hour, logger)

        box_params = [min_cap, max_cap, min_len, max_len]

        valid_idx, box_seq, ev_box_count, valid_seq = \
            update_ev_box_count(ev_box_count, valid_idx, box_seq, valid_seq, samples_per_hour, new_boxes, new_box_seq, new_valid_seq, box_params, timed)

        valid_idx = np.reshape(valid_idx, residual.shape)

        valid_idx, residual_cons, remove_ev = remove_false_ev_addition_cases(item_input_object, residual, box_cons, l1_bool, valid_idx, new_boxes, cons)

        residual_cons, valid_idx, sparse_ev_bool, ev_days = check_sparse_ev_scenario(l1_bool, item_input_object, valid_idx, vacation, residual_cons, ev_box_count, pilot, logger)

        if sparse_ev_bool and ((ev_days[-ev_config.get('recent_ev_days_thres'):].sum() / ev_days.sum()) < 0.7) and (ev_disagg.sum() == 0):
            logger.info('Removing ev output because of sparsity in ev output')
            residual_cons[:, :] = 0
            valid_idx[:, :] = 0

        if remove_ev or (not np.sum(valid_idx)) or \
                (np.percentile(input_data[input_data > 0], 50) > ev_config.get('max_ev_amp')/samples_per_hour) or \
                (np.sum(valid_idx) and ((not np.sum(ev_disagg)) and (np.sum(valid_idx[:, :2*samples_per_hour + 1]) == 0)) and ev_box_count < 40):
            residual_cons[:, :] = 0
            valid_idx[:, :] = 0

    valid_idx, residual_cons = \
        prepare_ev_signatures_by_picking_high_cons_box_from_data(valid_idx, residual_cons, item_input_object,
                                                                 item_output_object, l1_bool, ev_disagg, residual,
                                                                 input_data, excluded_cons_for_ev, logger)

    if np.sum(valid_idx) > 0:
        amp = np.median(residual_cons[residual_cons > 0])
        valid_idx[residual_cons < amp * 0.7] = 0
        residual_cons[residual_cons < amp * 0.7] = 0

    return valid_idx, residual_cons * 100


def prepare_ev_signatures_by_picking_high_cons_box_from_data(valid_idx, residual_cons, item_input_object,
                                                             item_output_object, l1_bool, ev_disagg, residual,
                                                             input_data, excluded_cons_for_ev, logger):
    """
    Calculate EV consumption in the leftover residual data by examining length, amplitude and consistency of high cons activity

    Parameters:
        valid_idx                   (dict)        : updated Dict containing all hybrid outputs
        residual_cons               (np.ndarray)  : Calculates TS level consumption
        item_input_object           (dict)        : Dict containing all hybrid outputs
        item_output_object          (dict)        : Dict containing all hybrid outputs
        l1_bool                     (bool)        : true if ev l1_bool output has to be estimated, false in case of L2 charger type
        ev_disagg                   (np.ndarray)  : EV TS level disagg output
        residual                    (np.ndarray)  : disagg residual data
        input_data                  (np.ndarray)  : raw data
        excluded_cons_for_ev        (np.ndarray)  : cons that wont be added into ev output
        logger                      (logger)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        valid_idx                   (dict)        : updated Dict containing all hybrid outputs
        residual_cons               (np.ndarray)  : Calculates TS level consumption
    """

    samples_per_hour = int(input_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    if l1_bool:
        return valid_idx, residual_cons

    run_checks = 1
    ev_type = 2
    ev_l2_cap = 2800

    # checking whether ev was detected in previous run

    if item_input_object.get("config").get('disagg_mode') in ['incremental', 'mtd'] and \
            item_input_object.get("item_input_params").get('ev_hsm') is not None \
            and item_input_object.get("item_input_params").get('ev_hsm').get('item_type') is not None:
        ev_type = item_input_object.get("item_input_params").get('ev_hsm').get('item_type')

        if isinstance(ev_type, list):
            ev_type = ev_type[0]

        if ev_type == 2:
            run_checks = 0

    # check if ev cons can be added based on length, amplitude and consistency of high cons activity

    valid_idx_step2, residual_cons_step2, amp = detect_leftover_ev_boxes(item_input_object, l1_bool, ev_disagg, residual, run_checks)

    if ev_type == 1 and amp > ev_l2_cap / samples_per_hour:
        valid_idx_step2[:, :] = 0
        amp = 0
        residual_cons_step2[:, :] = 0

    # if previous checks were successful, adding additional ev boxes

    if np.sum(valid_idx_step2) > 0 and np.sum(ev_disagg) == 0:
        valid_idx_step2, residual_cons_step2, amp = \
            add_ev_boxes_if_ev_detected_from_hybrid(run_checks, item_input_object, l1_bool, ev_disagg,
                                                    valid_idx_step2, input_data - excluded_cons_for_ev, item_input_object.get("config").get('disagg_mode'))

    # after preparing new ev boxes, checking the monthly consistency and adding extra ev boxes if required

    if ((np.sum(ev_disagg) > 0) or (np.sum(np.maximum(valid_idx, valid_idx_step2)) > 0)):
        season = item_output_object.get("season")

        valid_idx_step2, residual_cons_step2 = \
            add_ev_boxes_to_maintain_cons_in_ev_tou(item_input_object, l1_bool, ev_disagg, valid_idx_step2,
                                                    input_data - excluded_cons_for_ev, amp, season, logger)
        valid_idx_step2, residual_cons_step2 = \
            add_ev_boxes_to_maintain_con(item_input_object, l1_bool, ev_disagg, valid_idx_step2,
                                         input_data - excluded_cons_for_ev, amp, season, logger)

    valid_idx = np.maximum(valid_idx, valid_idx_step2)
    residual_cons = np.maximum(residual_cons, residual_cons_step2)

    return valid_idx, residual_cons


def get_ev_box_params_based_on_hsm(item_input_object, samples_per_hour, max_cap, min_cap, min_len, max_len, l1_bool):

    """
    update ev params based on hsm inputs

    Parameters:
        item_input_object           (dict)        : Dict containing all hybrid outputs
        samples_per_hour            (int)         : number of samples in an hour
        max_cap                     (int)         : original max ev capacity
        min_cap                     (int)         : original min ev capacity
        min_len                     (int)         : original min ev box len
        max_len                     (int)         : original max ev box len
        l1_bool                     (bool)        : true if ev l1_bool output has to be estimated, false in case of L2 charger type

    Returns:
        max_cap                     (int)         : updated max ev capacity
        min_cap                     (int)         : updated min ev capacity
        min_len                     (int)         : updated min ev box len
        max_len                     (int)         : updated max ev box len
        detect_ev                   (bool)        : 0 if ev shud not be added based on hsm
    """

    valid_ev_hsm = item_input_object.get("item_input_params").get('valid_ev_hsm')

    detect_ev = 1
    amplitude = 0

    zero_type = 0

    ev_config = get_inf_config(samples_per_hour).get('ev')

    ev_l1_amp_buffer = 2000
    ev_l2_amp_buffer = 500

    valid_hsm_flag = check_validity_of_hsm(valid_ev_hsm, item_input_object.get("item_input_params").get('ev_hsm'),
                                           'item_type')
    # update ev params incase of mtd mode

    if valid_hsm_flag:

        type, amplitude = fetch_ev_hsm_params(item_input_object)

        if type == 0:
            zero_type = 1

        if ((type == 1 and not l1_bool) or (type == 2 and l1_bool)):
            detect_ev = 0

        elif type == 1 and l1_bool:
            max_cap = [(amplitude+ev_l2_amp_buffer)/samples_per_hour, (amplitude+ev_l2_amp_buffer)/samples_per_hour,
                       (amplitude+ev_l2_amp_buffer)/samples_per_hour]
            min_cap = [(amplitude-ev_l2_amp_buffer)/samples_per_hour, (amplitude-ev_l2_amp_buffer)/samples_per_hour,
                       (amplitude-ev_l2_amp_buffer)/samples_per_hour]
            min_len = ev_config.get('mtd_l1_min_len')
            max_len = ev_config.get('mtd_l1_max_len')

        elif type == 2 and not l1_bool:
            max_cap = [(amplitude+ev_l1_amp_buffer)/samples_per_hour, (amplitude+ev_l1_amp_buffer)/samples_per_hour,
                       (amplitude+ev_l1_amp_buffer)/samples_per_hour]
            min_cap = [(amplitude-ev_l1_amp_buffer)/samples_per_hour, (amplitude-ev_l1_amp_buffer)/samples_per_hour,
                       (amplitude-ev_l1_amp_buffer)/samples_per_hour]
            min_len =  ev_config.get('mtd_l2_min_len')
            max_len =  ev_config.get('mtd_l2_max_len')

    return max_cap, min_cap, min_len, max_len, amplitude, detect_ev, zero_type


def get_new_ev_box_cons(residual, box_cons, valid_idx, new_boxes, ev_cons):

    """
    prepare ev estimation for new ev boxes

    Parameters:
        residual                    (np.ndarray)  : disagg residual data
        box_cons                    (np.ndarray)  : detected ev boxes cons
        valid_idx                   (np.ndarray)  : new ev time of use
        new_boxes                   (np.ndarray)  : new l2 boxes
        ev_cons                     (int)         : ev amplitude

    Returns:
        residual_cons               (np.ndarray)  : Calculates TS level consumption
    """

    residual_cons = np.zeros(residual.shape)
    residual_cons[valid_idx > 0] = box_cons[valid_idx > 0]

    if new_boxes:
        residual_cons[np.logical_and(valid_idx > 0, residual_cons == 0)] = ev_cons

    return residual_cons


def update_ev_box_count(ev_box_count, valid_idx, box_seq, valid_seq, samples_per_hour, new_boxes, new_box_seq, new_valid_seq, box_params, timed):

    """
    Calculate EV consumption in the leftover residual data

    Parameters:
        ev_box_count                (int)         : count of valid ev box
        valid_idx                   (np.ndarray)  : new ev time of use
        box_seq                     (np.ndarray)  : seq with params of all detected boxes
        valid_seq                   (np.ndarray)  : array containing information about which all boxes can be ev
        samples_per_hour            (int)         : number of samples in an hour
        new_boxes                   (bool)        : bool for whether to add additional ev boxx
        box_params                (np.ndarray)  : attributes of required ev boxes

    Returns:
        valid_idx                   (np.ndarray)  : new ev time of use
        box_seq                    (np.ndarray)  : seq with params of all detected boxes
        ev_box_count                (int)         : count of valid ev box
    """

    min_cap = box_params[0]
    max_cap = box_params[1]
    min_len = box_params[2]
    max_len = box_params[3]

    # update ev box count with new found boxes

    if np.sum(valid_idx):
        valid_seq[np.logical_and(np.logical_and(box_seq[:, 4] >= min_cap[1], box_seq[:, 4] < max_cap[1]),
                                 np.logical_and(box_seq[:, 3] >= min_len[1], box_seq[:, 3] <= max_len[1]))] = 1

        for i in range(len(valid_seq)):

            seq_strt = box_seq[i, 1]
            seq_end = box_seq[i, 2]
            seq_len = box_seq[i, 3]
            seq_val = box_seq[i, 0]

            total_samples = samples_per_hour * Cgbdisagg.HRS_IN_DAY

            invalid_ev_box = (seq_strt % total_samples not in np.arange(4 * samples_per_hour, 18 * samples_per_hour + 1)) or \
                             (seq_len > 8*samples_per_hour and (seq_strt % total_samples not in np.arange(4 * samples_per_hour, 15 * samples_per_hour + 1)))

            if seq_val and valid_seq[i] and invalid_ev_box:
                valid_idx[seq_strt: seq_end + 1] = 1
                box_seq[i, 0] = 0
                ev_box_count = ev_box_count + 1

    if np.sum(valid_idx) and new_boxes:
        for i in range(len(new_valid_seq)):

            new_seq_strt = new_box_seq[i, 1]
            new_seq_end = new_box_seq[i, 2]

            if new_valid_seq[i]:
                valid_idx[int(new_seq_strt): int(new_seq_end) + 1] = 1
                ev_box_count = ev_box_count + 1

    # if timed signature is found, remove boxes that are part of timed signature

    valid_idx, box_seq, ev_box_count, valid_seq = \
        remove_timed_boxes(ev_box_count, valid_idx, box_seq, valid_seq, samples_per_hour, timed)

    return valid_idx, box_seq, ev_box_count, valid_seq


def remove_timed_boxes(ev_box_count, valid_idx, box_seq, valid_seq, samples_per_hour, timed):

    """
    Calculate EV consumption in the leftover residual data

    Parameters:
        ev_box_count                (int)         : count of valid ev box
        valid_idx                   (np.ndarray)  : new ev time of use
        box_seq                     (np.ndarray)  : seq with params of all detected boxes
        valid_seq                   (np.ndarray)  : array containing information about which all boxes can be ev
        samples_per_hour            (int)         : number of samples in an hour
        new_boxes                   (bool)        : bool for whether to add additional ev boxx
        box_params                (np.ndarray)  : attributes of required ev boxes

    Returns:
        valid_idx                   (np.ndarray)  : new ev time of use
        box_seq                    (np.ndarray)  : seq with params of all detected boxes
        ev_box_count                (int)         : count of valid ev box
    """

    # if timed signature is found, remove boxes that are part of timed signature

    for i in range(len(valid_seq)):
        seq_strt = box_seq[i, 1]
        seq_end = box_seq[i, 2]

        if valid_seq[i] and timed[seq_strt: seq_end + 1].sum() > 2*samples_per_hour and len(timed) > 60:
            valid_idx[seq_strt: seq_end + 1] = 0

    return valid_idx, box_seq, ev_box_count, valid_seq


def remove_false_ev_addition_cases(item_input_object, residual, box_cons, l1_bool, valid_idx, new_boxes, cons):

    """
    Remove false ev addition cases

    Parameters:
        item_input_object           (dict)        : Dict containing all hybrid outputs
        residual                    (np.ndarray)  : disagg residual data
        l1_bool                     (bool)        : true if ev l1_bool output has to be estimated, false in case of L2 charger type
        valid_idx                   (np.ndarray)  : new ev time of use

    Returns:
        valid_idx                   (np.ndarray)  : new ev time of use
        residual_cons               (np.ndarray)  : Calculates TS level consumption
    """

    ev_config = get_inf_config().get('ev')

    verify_ev_tou_bool = l1_bool and np.sum(valid_idx) and (not (item_input_object.get("config").get('disagg_mode') == 'mtd'))

    # exclude days where ev is detected for continously for 20 days

    valid_idx_copy = copy.deepcopy(valid_idx)

    ev_days = (np.sum(valid_idx, axis=1) > 0).astype(int)
    ev_days_seq = find_seq(ev_days, np.zeros_like(ev_days), np.zeros_like(ev_days))

    ev_days_seq[np.logical_and(ev_days_seq[:, 0] == 1, ev_days_seq[:, 3] > ev_config.get('continous_ev_days_thres')), 0] = 0

    for i in range(len(ev_days_seq)):
        if not ev_days_seq[i, 0]:
            valid_idx[ev_days_seq[i, 1]:ev_days_seq[i, 2]+1] = 0

    if ((np.sum(valid_idx, axis=1) > 0).sum() < 20) and verify_ev_tou_bool:
        valid_idx[:, :] = 0
    elif verify_ev_tou_bool:
        valid_idx = valid_idx_copy

    residual_cons = get_new_ev_box_cons(residual, box_cons, valid_idx, new_boxes, cons)

    # exclude cases where timed signature kind of pattern is detected as ev usage

    usage_days = (np.sum(valid_idx, axis=1) > 0).astype(int)
    seq = find_seq(usage_days, np.zeros_like(usage_days), np.zeros_like(usage_days))

    for i in range(len(seq)):
        if (not seq[i, 0]) and seq[i, 3] <= 1:
            usage_days[seq[i, 1]: seq[i, 2]+1] = 1

    seq = find_seq(usage_days, np.zeros_like(usage_days), np.zeros_like(usage_days))

    remove_ev = 0

    for i in range(len(seq)):
        if (seq[i, 0]) and (seq[i, 3] >= 15):
            data = valid_idx[seq[i, 1]: seq[i, 2]+1][valid_idx[seq[i, 1]: seq[i, 2]+1].sum(axis=1) > 0]
            data = data.sum(axis=0)
            max_len = np.sum(valid_idx[seq[i, 1]: seq[i, 2]+1].sum(axis=1) > 0)

            remove_ev = remove_ev or (verify_ev_tou_bool and (np.any(data > 2) and
                                                              np.percentile(data[data > 2], 67) >= ev_config.get('timed_behaviour_thres')*max_len))

    return valid_idx, residual_cons, remove_ev


def check_sparse_ev_scenario(l1_bool, item_input_object, valid_idx, vacation, residual_cons, ev_box_count, pilot, logger):

    """
    Calculate EV consumption in the leftover residual data

    Parameters:
        l1_bool                     (bool)        : true if ev l1_bool output has to be estimated, false in case of L2 charger type
        item_input_object           (dict)        : Dict containing all hybrid outputs
        valid_idx                   (np.ndarray)  : new ev time of use
        vacation                    (np.ndarray)  : vacation data
        residual_cons               (np.ndarray)  : additional ev output
        ev_box_count                (int)         ; valid ev boxes
        pilot                       (int)         : pilot id
        ev_disagg                   (np.ndarray)  : EV TS level disagg output

    Returns:
        valid_idx                   (np.ndarray)  : new ev time of use
        residual_cons               (np.ndarray)  : additional ev output
    """

    ev_config = get_inf_config().get('ev')

    ev_days = (np.sum(valid_idx[np.logical_not(vacation)], axis=1) > 0).astype(int)

    if len(ev_days) == 0:
        return residual_cons, valid_idx, 0, ev_days

    if not (np.any(ev_days == 0) and (not (item_input_object.get("config").get('disagg_mode') == 'mtd'))):
        return residual_cons, valid_idx, 0, ev_days

    ev_seq = find_seq(ev_days, np.zeros_like(ev_days), np.zeros_like(ev_days), overnight=0)

    ev_seq = ev_seq[ev_seq[:, 0] == 0]
    ev_len = ev_seq[:, 3]

    max_days_count = 40
    min_days_count = 5
    min_days_count_l1 = 4

    if np.any(ev_len < max_days_count):
        ev_len = ev_len[ev_len < max_days_count]
    if np.any(ev_len > min_days_count):
        ev_len = ev_len[ev_len > min_days_count]
    if l1_bool and np.any(ev_len > min_days_count_l1):
        ev_len = ev_len[ev_len > min_days_count_l1]

    avg_ev_freq = np.mean(ev_len)

    freq_thres = ev_config.get('ev_freq_thres')

    ev_type = get_hsm_of_incremental_mode(item_input_object)

    if (ev_type == 1 and l1_bool) or (ev_type == 2 and not l1_bool) or (ev_box_count > 140):
        freq_thres = ev_config.get('mtd_ev_freq_thres')

    logger.info('ev frequency threshold | %s', freq_thres)

    sparse_ev_bool = ((avg_ev_freq > freq_thres) and (pilot not in ev_config.get('eu_pilots'))) or \
                     ((avg_ev_freq > ev_config.get('eu_ev_freq_thres')) and (pilot in ev_config.get('eu_pilots')))

    return residual_cons, valid_idx, sparse_ev_bool, ev_days


def update_ev_output_with_remaining_ev_boxes(item_input_object, residual, ev_disagg, box_cons, valid_idx, samples_per_hour, logger):

    """
    Calculate EV consumption in the leftover residual data

    Parameters:
        item_input_object           (dict)        : Dict containing all hybrid outputs
        residual                    (np.ndarray)  : disagg residual data
        valid_idx                   (np.ndarray)  : new ev time of use
        samples_per_hour            (int)         : number of samples in an hour

    Returns:
        new_valid_seq               (np.ndarray)  : array containing information about which all boxes can be ev
        new_box_seq                 (np.ndarray)  : seq with params of all detected boxes
        new_boxes                   (bool)        : bool for whether to add additional ev boxx
        ev_cons                     (int)         : ev box ampllitude
    """

    new_boxes = 0

    ev_config = get_inf_config().get('ev')

    new_valid_seq = np.zeros(2)
    new_box_seq = np.zeros((2, 5))
    ev_cons_from_res = 0

    if not (np.sum(ev_disagg) == 0 and np.sum(valid_idx) > 0):
        return new_valid_seq, new_box_seq, new_boxes, ev_cons_from_res

    ev_cons_from_res = np.percentile(residual[valid_idx.reshape(box_cons.shape).astype(bool)], 80)

    # add remaining l2 ev boxes leftout in residual data

    if ev_cons_from_res > ev_config.get('l2_box_thres') / samples_per_hour:

        logger.info('adding remaining l2 boxes | ')

        len_seq = find_seq(valid_idx, np.zeros_like(valid_idx), np.zeros_like(valid_idx))
        mean_ev_usage_len = np.median(len_seq[len_seq[:, 0] > 0, 3])

        new_boxes = 1

        # new ev boxes parameters

        new_boxes_ts = np.zeros_like(valid_idx.reshape(box_cons.shape))
        new_boxes_ts[np.logical_and(residual > ev_cons_from_res * ev_config.get('max_cons_factor'),
                                    residual < ev_config.get('min_cons_factor') * ev_cons_from_res)] = 1

        new_boxes_ts[valid_idx.reshape(box_cons.shape) > 0] = 0

        ev_possible_tou = valid_idx.reshape(box_cons.shape).sum(axis=0)

        start = 0
        val = 0

        for k in range(len(ev_possible_tou)):
            if np.sum(ev_possible_tou[get_index_array(k, k + mean_ev_usage_len, Cgbdisagg.HRS_IN_DAY * samples_per_hour)]) > val:
                val = np.sum(ev_possible_tou[get_index_array(k, k + mean_ev_usage_len, Cgbdisagg.HRS_IN_DAY * samples_per_hour)])
                start = k

        ev_possible_tou = get_index_array(start - ev_config.get('tou_buffer_hours') * samples_per_hour,
                                          start + ev_config.get('tou_buffer_hours') * samples_per_hour,
                                          Cgbdisagg.HRS_IN_DAY * samples_per_hour)

        logger.info('EV L2 tou %s | ', ev_possible_tou)

        new_box_seq = find_seq(new_boxes_ts.flatten(), np.zeros_like(new_boxes_ts.flatten()), np.zeros_like(new_boxes_ts.flatten()))

        min_l2_box_len = (1.5 * samples_per_hour) * (samples_per_hour > 1) + 1 * (samples_per_hour == 1)

        # checking whether new detected boxes are ev l2 boxes

        new_valid_seq = np.ones(len(new_box_seq))
        new_valid_seq[new_box_seq[:, 0] == 0] = 0

        max_l2_box_len = ev_config.get('max_l2_box_len') * samples_per_hour

        if item_input_object.get("config").get('disagg_mode') == 'mtd':
            new_valid_seq[new_box_seq[:, 3] < max(min_l2_box_len, mean_ev_usage_len * ev_config.get('mtd_max_box_len_factor'))] = 0
            new_valid_seq[new_box_seq[:, 3] > min(max_l2_box_len, mean_ev_usage_len * ev_config.get('mtd_min_box_len_factor'))] = 0
        else:
            new_valid_seq[new_box_seq[:, 3] < max(min_l2_box_len, mean_ev_usage_len * ev_config.get('max_box_len_factor'))] = 0
            new_valid_seq[new_box_seq[:, 3] > min(max_l2_box_len, mean_ev_usage_len * ev_config.get('min_box_len_factor'))] = 0

        new_valid_seq[np.logical_not(np.isin((new_box_seq[:, 1] % (samples_per_hour * Cgbdisagg.HRS_IN_DAY)), ev_possible_tou))] = 0

        if val == 0:
            new_valid_seq[:] = 0

    return new_valid_seq, new_box_seq, new_boxes, ev_cons_from_res
