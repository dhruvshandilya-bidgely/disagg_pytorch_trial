

"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Master file for itemization pipeline
"""

# Import python packages

import traceback
import numpy as np

# import functions from within the project

from python3.itemization.aer.raw_energy_itemization.residual_analysis.assign_stat_app_boxes import allot_boxes
from python3.itemization.aer.raw_energy_itemization.residual_analysis.assign_wh_boxes_from_residual import allot_wh_boxes
from python3.itemization.aer.raw_energy_itemization.residual_analysis.detect_ev_sigature_wrapper import ev_signature_detection_wrapper
from python3.itemization.aer.raw_energy_itemization.residual_analysis.box_activity_detection_wrapper import box_activity_detection_wrapper
from python3.itemization.aer.raw_energy_itemization.residual_analysis.timed_activity_detection_wrapper import timed_activity_detection_wrapper
from python3.itemization.aer.raw_energy_itemization.residual_analysis.process_weather_analytics_output import process_weather_analytics_output
from python3.itemization.aer.raw_energy_itemization.residual_analysis.seasonal_signature_detection_wrapper import seasonal_sig_detection_wrapper


def run_residual_analyis_modules(item_input_object, item_output_object, logger, logger_pass):

    """
    Prepare hybrid input object

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object
        logger_pass               (dict)      : Contains base logger and logging dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    output_data = item_output_object.get("hybrid_input_data").get("output_data")
    residual_copy = item_output_object.get("hybrid_input_data").get("original_res")
    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")
    disagg_residual = item_output_object.get("hybrid_input_data").get("true_disagg_res")

    run_hybrid_v2 = item_input_object.get('item_input_params').get('run_hybrid_v2_flag')

    ################### Activity detection in residual data to pick PP ad EV signature  #########################

    try:

        box_label, box_cons, box_seq, box_label_wh, box_cons_wh, box_seq_wh = \
            box_activity_detection_wrapper(item_input_object, item_output_object, logger)

        box_dict = dict({
            "box_label": box_label,
            "box_cons": box_cons,
            "box_seq": box_seq,
            "box_label_wh": box_label_wh,
            "box_cons_wh": box_cons_wh,
            "box_seq_wh": box_seq_wh
        })

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in BOX detection module | %s', error_str)

        box_dict = dict({
            "box_label": np.zeros_like(disagg_residual),
            "box_cons": np.zeros_like(disagg_residual),
            "box_seq": np.zeros((1, 5)),
            "box_label_wh": np.zeros_like(disagg_residual),
            "box_cons_wh": np.zeros_like(disagg_residual),
            "box_seq_wh": np.zeros((1, 5)),
        })

        box_seq_wh = np.zeros((1, 5))
        box_label = np.zeros_like(disagg_residual)
        box_seq = np.zeros((1, 5))
        box_label_wh = np.zeros_like(disagg_residual)

    item_output_object.update({
        "box_dict": box_dict
    })

    logger.info("Calculated activity boxes in residual data | ")

    item_output_object["residual_detection"] = [0, 0, 0]

    weather_analytics, season, item_output_object = process_weather_analytics_output(item_input_object, item_output_object)

    ################### Timed Activity detection in residual data to pick PP ad TWH signature  #########################

    try:
        timed_app_dict, residual_copy, item_input_object, item_output_object = \
            timed_activity_detection_wrapper(item_input_object, item_output_object, logger, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in timed appliance detection module | %s', error_str)
        timed_app_dict = dict({
            "twh": np.zeros(residual_copy.shape),
            "pp": np.zeros(residual_copy.shape),
            "heating": np.zeros(residual_copy.shape),
            "timed_output": np.zeros(residual_copy.shape)
        })

    item_output_object.update({
        "timed_app_dict": timed_app_dict
    })

    item_output_object["hybrid_input_data"]["updated_residual_data"] = residual_copy
    item_output_object["hybrid_input_data"]["updated_residual_without_detected_sig"] = residual_copy

    ################################# EV residual detection to pick EV L1 or L2 signature  ###########################

    try:
        item_input_object, item_output_object, residual_copy = \
            ev_signature_detection_wrapper(item_input_object, item_output_object, season, logger)
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in ev box detection | %s', error_str)

    detected_heat = np.zeros(disagg_residual.shape)
    detected_cool = np.zeros(disagg_residual.shape)
    detected_wh = np.zeros(disagg_residual.shape)

    ################################# Seasonal signature detection to add leftover cooling/heating/SWH  ###########################

    try:
        detected_cool, detected_heat, detected_wh, residual_copy = \
            seasonal_sig_detection_wrapper(item_input_object, item_output_object, weather_analytics, logger, logger_pass)

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in hvac detection module | %s', error_str)

        hvac_dict = dict({
            "heating": detected_heat,
            "cooling": detected_cool,
            "wh": detected_wh
        })

        item_output_object.update({
            "hvac_dict": hvac_dict
        })

    item_output_object["hybrid_input_data"]["updated_residual_data"] = residual_copy
    item_output_object["hybrid_input_data"]["weather_analytics"] = weather_analytics

    ################################# Allocate residual boxes to WH/cook/ent/ld  ###########################

    try:
        box_score_wh = allot_wh_boxes(item_input_object, item_output_object, appliance_list, output_data, box_label_wh, logger)

        if run_hybrid_v2:
            box_score = allot_boxes(item_input_object, item_output_object, appliance_list, output_data, box_label, logger)
        else:
            box_score = np.zeros((len(box_seq_wh), 3))

        item_output_object["box_score_wh"] = box_score_wh
        item_output_object["box_score"] = box_score

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.error('Something went wrong in box allotmnent | %s', error_str)

        item_output_object["box_score"] = np.zeros((len(box_seq), 3))
        item_output_object["box_score_wh"] = np.zeros((len(box_seq), 3))

    return item_output_object
