"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update pp consumption ranges using inference rules
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array
from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val

from python3.itemization.aer.raw_energy_itemization.utils import update_hybrid_object

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def get_pp_inference(app_index, item_input_object, item_output_object, logger_pass):

    """
    Update pp consumption ranges using inference rules

    Parameters:
        app_index                   (int)       : Index of app in the appliance list
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        logger_pass                 (dict)      : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_pp_inference')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    season = item_output_object.get("season")

    config = get_inf_config().get("pp")

    min_pp_days_required = config.get('min_pp_days_required')
    max_gap_in_pp_days = config.get('max_gap_in_pp_days')
    max_allowed_cons_in_pp = config.get('max_allowed_cons_in_pp')

    app_pot = item_output_object.get("inference_engine_dict").get("appliance_pot")[app_index]
    app_conf = item_output_object.get("inference_engine_dict").get("appliance_conf")[app_index]
    disagg_cons = item_output_object.get("inference_engine_dict").get("output_data")[app_index, :, :]
    mid_cons = item_output_object.get("inference_engine_dict").get("appliance_mid_values")[app_index, :, :]
    max_cons = item_output_object.get("inference_engine_dict").get("appliance_max_values")[app_index, :, :]
    min_cons = item_output_object.get("inference_engine_dict").get("appliance_min_values")[app_index, :, :]

    residual = copy.deepcopy(item_output_object.get("inference_engine_dict").get("residual_data"))
    timed_output = item_output_object.get("timed_app_dict").get("pp")

    timed_output = block_low_duration_pp_schedules(item_input_object, timed_output, disagg_cons)

    max_cons = np.maximum(max_cons, disagg_cons)
    mid_cons = np.maximum(mid_cons, np.multiply(disagg_cons, app_conf + 0.1))
    max_cons = np.minimum(max_cons, disagg_cons)
    mid_cons = np.minimum(mid_cons, disagg_cons)
    min_cons = np.minimum(min_cons, disagg_cons)

    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    if np.all(disagg_cons == 0) and not np.sum(timed_output):
        item_output_object = update_pp_hsm(item_input_object, item_output_object, disagg_cons)

    ########################### RULE 1 - Checking seasonality in pp output ######################################

    min_cons, max_cons = \
        adjust_min_max_values_in_pp_overest_cases(app_pot, app_conf, residual, input_data, disagg_cons, [min_cons, max_cons], logger, season)

    item_output_object["inference_engine_dict"]["appliance_conf"][app_index][timed_output > 0] = 1

    ######################### RULE 2 - Adding timed signature detection in pp output ###########################


    timed_output = np.fmin(timed_output, max_allowed_cons_in_pp)

    mid_cons = mid_cons + timed_output
    max_cons = max_cons + timed_output
    min_cons = min_cons + timed_output

    mid_cons = np.fmax(0, mid_cons)
    max_cons = np.fmax(0, max_cons)
    min_cons = np.fmax(0, min_cons)

    pp_conf = 1

    pp_conf_present_flag =  \
        item_input_object.get('disagg_special_outputs') is not None and \
        item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') is not None

    if pp_conf_present_flag:
        pp_conf = item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') / 100

    ######################### RULE 3 - Handling TWH and PP overlap sceenario ###########################

    min_cons, mid_cons, max_cons, item_output_object = \
        handle_pp_twh_overlapping_case(item_input_object, item_output_object, residual, input_data, disagg_cons,
                                       pp_conf, [min_cons, mid_cons, max_cons], logger)

    if np.any(np.logical_and(disagg_cons > 0, timed_output == 0)):
        mid_cons[np.logical_and(disagg_cons > 0, timed_output == 0)] = \
            np.minimum(disagg_cons[np.logical_and(disagg_cons > 0, timed_output == 0)],
                       mid_cons[np.logical_and(disagg_cons > 0, timed_output == 0)])

    timed_app_days = np.sum(mid_cons, axis=1) > 0

    # post processing of derived twh region, based on the continuity

    thres = min_pp_days_required

    timed_app_days = fill_arr_based_seq_val(timed_app_days, timed_app_days, max_gap_in_pp_days, 0, 1)

    timed_app_days = fill_arr_based_seq_val(timed_app_days, timed_app_days, thres, 1, 0)

    mid_cons[timed_app_days == 0] = 0

    max_cons = np.minimum(max_cons, input_data)
    min_cons = np.minimum(min_cons, input_data)
    mid_cons = np.minimum(mid_cons, input_data)

    # Updating the values in the original dictionary

    item_output_object = update_hybrid_object(app_index, item_output_object, mid_cons, max_cons, min_cons)

    item_output_object["inference_engine_dict"]["output_data"][app_index, :, :] = disagg_cons

    t_end = datetime.now()

    logger.info("PP inference calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def adjust_min_max_values_in_pp_overest_cases(app_pot, app_conf, residual, input_data, disagg_cons, pp_ranges,
                                              logger, season):

    """
    adjust PP ts level min/max consumption incase of overestimation of PP
    which is causing overshoot of total disagg compared to input data

    Parameters:
        app_pot                     (np.ndarray)    : TS level PP usage potential
        app_conf                    (np.ndarray)    : TS level PP confidence
        residual                    (np.ndarray)    : disagg residual data
        input_data                  (np.ndarray)    : user input data
        disagg_cons                 (np.ndarray)    : pp disagg output
        min_cons                    (np.ndarray)    : ts level min PP consumption
        mid_cons                    (np.ndarray)    : ts level avg PP consumption
        max_cons                    (np.ndarray)    : ts level max PP consumption
        config                      (dict)          : pp config
        season                      (np.ndarray)    : season tags
        logger                      (logger)        : logger object

    Returns:
        min_cons                    (np.ndarray)    : ts level min PP consumption
        max_cons                    (np.ndarray)    : ts level max PP consumption

    """

    min_cons = pp_ranges[0]
    max_cons = pp_ranges[1]

    config = get_inf_config().get("pp")

    min_days_to_check_seasonality = config.get('min_days_to_check_seasonality')
    perc_cap_for_app_pot = config.get('perc_cap_for_app_pot')
    conf_score_offset_for_overestimated_points = config.get('conf_score_offset_for_overestimated_points')

    disagg_pp_absent = not np.all(disagg_cons == 0)

    if disagg_pp_absent:

        if np.any(max_cons > 0):
            max_cons = np.percentile(max_cons[max_cons > 0], perc_cap_for_app_pot) * app_pot

        if np.any(min_cons > 0):
            min_cons = np.percentile(min_cons[min_cons > 0], perc_cap_for_app_pot) * app_pot

        overest_ts = residual < 0.02*input_data

        modified_cons = copy.deepcopy(min_cons)

        modified_cons[overest_ts] = np.multiply(min_cons[overest_ts], app_conf[overest_ts] + conf_score_offset_for_overestimated_points)

        if np.any(min_cons > 0):
            min_cons = app_pot * np.median(modified_cons[modified_cons > 0])

        # Check if winter pp consumption is non zero and zero for summer months

        if len(input_data) > min_days_to_check_seasonality and np.sum(disagg_cons[season < 0]) > 0 and \
                (np.sum(season > 0)) and np.sum(disagg_cons[season > 0]) == 0:
            logger.debug("Opposite seasonality in pp consumption")
            min_cons = np.zeros(min_cons.shape)

    return min_cons, max_cons


def handle_pp_twh_overlapping_case(item_input_object, item_output_object, residual, input_data, disagg_cons, pp_conf,
                                   pp_ranges, logger):

    """
    Adjust PP consumption incase of overlap with twh disagg output

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        residual                    (np.ndarray)    : disagg residual data
        input_data                  (np.ndarray)    : user input data
        disagg_cons                 (np.ndarray)    : pp disagg output
        pp_conf                     (np.ndarray)    : PP detection conf
        min_cons                    (np.ndarray)    : ts level min PP consumption
        mid_cons                    (np.ndarray)    : ts level avg PP consumption
        max_cons                    (np.ndarray)    : ts level max PP consumption
        config                      (dict)          : pp config
        logger                      (logger)        : logger object

    Returns:
        min_cons                    (np.ndarray)    : ts level min PP consumption
        mid_cons                    (np.ndarray)    : ts level avg PP consumption
        max_cons                    (np.ndarray)    : ts level max PP consumption
        item_output_object          (dict)          : Dict containing all hybrid outputs

    """

    min_cons = pp_ranges[0]
    mid_cons = pp_ranges[1]
    max_cons = pp_ranges[2]

    config = get_inf_config().get("pp")

    min_pp_len_for_removing_twh = config.get('min_pp_len_for_removing_twh')
    min_pp_conf_for_removing_twh = config.get('min_pp_conf_for_removing_twh')
    min_pp_amp_for_removing_twh = config.get('min_pp_amp_for_removing_twh')
    wh_buffer_hours = config.get('wh_buffer_hours')
    min_pp_amp = config.get('min_pp_amp')
    min_pp_days_required = config.get('min_pp_days_required')

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START

    season = item_output_object.get("season")

    # check if TWH output is non zero

    if item_input_object.get("item_input_params").get("timed_wh_user"):

        appliance_list = item_output_object.get("inference_engine_dict").get("appliance_list")
        wh_index = np.where(np.array(appliance_list) == 'wh')[0][0]
        wh_disagg = item_output_object.get("inference_engine_dict").get("output_data")[wh_index, :, :]

        samples_per_hour = int(disagg_cons.shape[1]/Cgbdisagg.HRS_IN_DAY)

        overest_ts = residual < 0.02 * input_data

        # checks if there is overlap in pp and twh output

        wh_tou = (np.sum(np.logical_and(np.logical_and(overest_ts > 0, disagg_cons > 0), wh_disagg > 0),
                         axis=0) > 0).astype(int)

        wh_tou = find_seq(wh_tou, np.zeros_like(wh_tou), np.zeros_like(wh_tou), overnight=1)

        for wh_band in range(len(wh_tou)):

            if wh_tou[wh_band, seq_label]:

                tou_array = get_index_array(wh_tou[wh_band, seq_start] - wh_buffer_hours * samples_per_hour,
                                            wh_tou[wh_band, seq_start] + wh_buffer_hours * samples_per_hour,
                                            len(wh_disagg[0]))

                avg_length = np.sum(wh_disagg[:, tou_array] > 0, axis=1)
                avg_length = np.mean(avg_length[avg_length > 0])

                # in case of overlap, it is checked whether the PP if of high confidence,
                # if not, PP consumption is deleted if pp app profile is None
                # if not, PP consumption is reduced to 300 wh if pp app profile is yes

                if (avg_length <= min_pp_len_for_removing_twh * samples_per_hour) and \
                        pp_conf < min_pp_conf_for_removing_twh and \
                        np.percentile(mid_cons[:, tou_array][mid_cons[:, tou_array] > 0], 90) > (min_pp_amp_for_removing_twh / samples_per_hour):

                    logger.info("PP overlapping with TWH output, reducing PP consumption | ")

                    val = min_pp_amp * (item_input_object.get("item_input_params").get("pp_prof_present")) + \
                          0 * (not item_input_object.get("item_input_params").get("pp_prof_present"))

                    mid_cons[:, tou_array] = np.fmin(val / samples_per_hour, mid_cons[:, tou_array])
                    min_cons[:, tou_array] = np.fmin(val / samples_per_hour, mid_cons[:, tou_array])
                    max_cons[:, tou_array] = np.fmin(val / samples_per_hour, mid_cons[:, tou_array])

                    item_input_object["item_input_params"]["pp_removed"] = int(val > 0)

    min_cons = np.minimum(min_cons, mid_cons)
    max_cons = np.maximum(max_cons, mid_cons)

    pp_days = np.sum(mid_cons, axis=1) > 0

    kill_pp = 0

    # Check if winter pp consumption is non zero and zero for summer months

    opp_seasonality_detected_in_hybrid_pp = \
        np.all(disagg_cons == 0) and len(input_data) > config.get('min_days_to_check_seasonality') \
        and np.any(season > 0) and np.any(season <= 0) and \
        np.sum(mid_cons[season <= 0]) > 0 and np.sum(mid_cons[season > 0]) == 0

    if opp_seasonality_detected_in_hybrid_pp:
        logger.debug("Opposite seasonality in pp consumption")
        kill_pp = 1

    if np.sum(pp_days) < min_pp_days_required:
        kill_pp = 1

    if kill_pp:
        mid_cons[:, :] = 0
        min_cons[:, :] = 0
        max_cons[:, :] = 0

    return min_cons, mid_cons, max_cons, item_output_object


def block_low_duration_pp_schedules(item_input_object, timed_output, pp_disagg):

    """
    remove low days schedules from timed signature

    Parameters:
        pp_disagg                   (np.ndarray): pp disagg output
        timed_output                (np.ndarray): possible pp consumption - picked up from disagg residual
        item_input_object           (dict)      : Dict containing all hybrid inputs

    Returns:
        timed_output                (np.ndarray): updated time signature output
    """

    if item_input_object.get('config').get('disagg_mode') == 'mtd':
        return timed_output

    config = get_inf_config().get("pp")

    min_thres_on_pp_days_required = config.get('min_thres_on_pp_days_required')
    max_thres_on_pp_days_required = config.get('max_thres_on_pp_days_required')
    days_frac_for_pp_days_required = config.get('days_frac_for_pp_days_required')

    min_days_required = min(max_thres_on_pp_days_required, max(min_thres_on_pp_days_required, days_frac_for_pp_days_required*len(pp_disagg)))

    pp_days = np.sum(timed_output + pp_disagg, axis=1) > 0

    pp_days = fill_arr_based_seq_val(pp_days, pp_days, 10, 0, 1, overnight_tag=0)

    timed_output = fill_arr_based_seq_val(pp_days, timed_output, min_days_required, 1, 0, overnight_tag=0)

    return timed_output


def update_pp_hsm(item_input_object, item_output_object, disagg_cons):

    """
    Update pp hsm with hybrid attributes

    Parameters:
        disagg_cons                 (np.ndarray): pp disagg output
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    # updating HSM in cases where PP is not detected in disagg and hybrid v2

    created_hsm = dict({
        'item_tou': np.zeros(len(disagg_cons[0])),
        'item_hld': 0,
        'item_conf': 0,
        'item_amp': 0
    })

    created_hsm['item_tou'] = np.sum(disagg_cons, axis=0) > 0

    post_hsm_flag = item_input_object.get('item_input_params').get('post_hsm_flag')

    pp_hsm_present_flag = post_hsm_flag and (item_output_object.get('created_hsm').get('pp') is None)

    if pp_hsm_present_flag:
        item_output_object['created_hsm']['pp'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    pp_hsm_present_flag = post_hsm_flag and (item_output_object.get('created_hsm') is not None) and \
                          (item_output_object.get('created_hsm').get('pp') is not None) and \
                          (item_output_object.get('created_hsm').get('pp').get('attributes') is not None)

    if pp_hsm_present_flag:
        item_output_object['created_hsm']['pp']['attributes'].update(created_hsm)

    return item_output_object
